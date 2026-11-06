# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# MXFP4 x MXFP4 MoE GEMM and MXFP4 quantization kernels. Vendored from ROCm/aiter
# (commit 652f2d4b768239aefd515c929e84817681ff598d), which adapted the GEMM from the triton_kernels package
# (matmul_ogs_details/_matmul_ogs.py, MIT):
#   aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a4w4.py   (xcd_swizzle, unswizzle_mx_scale_cdna4,
#                                                             _mxfp4_quant_kernel, _moe_gemm_a4w4)
#   aiter/ops/triton/_triton_kernels/quant/quant.py           (_mxfp4_quant_op)
#   aiter/ops/triton/_triton_kernels/moe/quant_moe.py         (_compute_static_fp8_quant)
#   aiter/ops/triton/utils/_triton/pid_preprocessing.py       (pid_grid)
# Changes: imports made local; the profiling-only launch_metadata hook of _moe_gemm_a4w4 removed;
# _mxfp4_quant_op computes AITER's production RoundUp e8m0 scale (see the comment in it).

import triton
import triton.language as tl

from .activations import _swiglu


@triton.jit
def _compute_static_fp8_quant(tensor, scale):
    tensor = tensor.to(tl.float32)
    tensor = tensor / scale
    tensor = tensor.to(tl.float8e4nv)
    return tensor


@triton.jit
def _mxfp4_quant_op(
    x,
    BLOCK_SIZE_N,
    BLOCK_SIZE_M,
    MXFP4_QUANT_BLOCK_SIZE,
):
    """
    Converts given x (in fp32) to mxfp4 format.
    x: [BLOCK_SIZE_M, BLOCK_SIZE_N], fp32

    """
    EXP_BIAS_FP32: tl.constexpr = 127
    EXP_BIAS_FP4: tl.constexpr = 1
    EBITS_F32: tl.constexpr = 8
    EBITS_FP4: tl.constexpr = 2
    MBITS_F32: tl.constexpr = 23
    MBITS_FP4: tl.constexpr = 1

    max_normal: tl.constexpr = 6
    min_normal: tl.constexpr = 1

    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE
    x = x.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE)
    # Calculate scale. Task change: AITER's production MXFP4 activation quantizers (the HIP
    # fused_dynamic_mx_quant_moe_sort and the FlyDSL MoE kernels) use the RoundUp scale mode
    # (kDefaultMxScaleRoundMode in csrc/include/mx_quant_utils.h): the dequantization scale is
    # ceil_pow2(max(amax, 1e-10) * (1 / 6)), so no value saturates. The upstream Triton op rounds amax to
    # the nearest power of two and divides by 4 instead.
    amax = tl.max(tl.abs(x), axis=-1, keep_dims=True)
    amax = tl.maximum(amax, 1e-10)
    dequant_scale = (amax * 0.16666666666666666).to(tl.uint32, bitcast=True)
    dequant_scale = (dequant_scale + 0x7FFFFF) & 0x7F800000
    biased_exponent = (dequant_scale >> 23).to(tl.int32)

    # blockscale_e8m0
    bs_e8m0 = biased_exponent.to(tl.uint8)

    # 2^(127 - biased_exponent), built from its bits (biased_exponent lies in [92, 253] here).
    quant_scale = ((254 - biased_exponent) << 23).to(tl.float32, bitcast=True)

    # Compute quantized x
    qx = x * quant_scale

    # Convert quantized fp32 tensor to uint32 before converting to mxfp4 format
    # Note: MXFP4  S:1-bit, E:2-bit, M:1-bit
    #   Zeros: S000 -> +/-0
    #   Denormal Numbers: S001 -> +/- 0.5
    #   Normal Numbers:
    #           S010 -> +/- 1.0
    #           S011 -> +/- 1.5
    #           S100 -> +/- 2.0
    #           S101 -> +/- 3.0
    #           S110 -> +/- 4.0
    #           S111 -> +/- 6.0
    qx = qx.to(tl.uint32, bitcast=True)

    # Extract sign
    s = qx & 0x80000000
    # Set everything to positive, will add sign back at the end
    qx = qx ^ s

    qx_fp32 = qx.to(tl.float32, bitcast=True)
    saturate_mask = qx_fp32 >= max_normal
    denormal_mask = (not saturate_mask) & (qx_fp32 < min_normal)
    normal_mask = not (saturate_mask | denormal_mask)

    # Denormal numbers
    denorm_exp: tl.constexpr = (
        (EXP_BIAS_FP32 - EXP_BIAS_FP4) + (MBITS_F32 - MBITS_FP4) + 1
    )
    denorm_mask_int: tl.constexpr = denorm_exp << MBITS_F32
    denorm_mask_float: tl.constexpr = tl.cast(denorm_mask_int, tl.float32, bitcast=True)

    denormal_x = qx_fp32 + denorm_mask_float
    denormal_x = denormal_x.to(tl.uint32, bitcast=True)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(tl.uint8)

    # Normal numbers
    normal_x = qx
    # resulting mantissa is odd
    mant_odd = (normal_x >> (MBITS_F32 - MBITS_FP4)) & 1
    # update exponent, rounding bias part 1
    val_to_add = ((EXP_BIAS_FP4 - EXP_BIAS_FP32) << MBITS_F32) + (1 << 21) - 1
    normal_x += val_to_add
    # rounding bias part 2
    normal_x += mant_odd
    # take the bits!
    normal_x = normal_x >> (MBITS_F32 - MBITS_FP4)
    normal_x = normal_x.to(tl.uint8)

    # Merge results
    e2m1_value = tl.full(qx.type.get_block_shapes(), 0x7, dtype=tl.uint8)
    e2m1_value = tl.where(normal_mask, normal_x, e2m1_value)
    e2m1_value = tl.where(denormal_mask, denormal_x, e2m1_value)
    # add sign back
    sign_lp = s >> (MBITS_F32 + EBITS_F32 - MBITS_FP4 - EBITS_FP4)
    sign_lp = sign_lp.to(tl.uint8)
    e2m1_value = e2m1_value | sign_lp
    e2m1_value = tl.reshape(
        e2m1_value, [BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE // 2, 2]
    )
    evens, odds = tl.split(e2m1_value)
    x_fp4 = evens | (odds << 4)
    x_fp4 = x_fp4.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2)

    return x_fp4, bs_e8m0.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS)


@triton.jit
def pid_grid(pid: int, num_pid_m: int, num_pid_n: int, GROUP_SIZE_M: tl.constexpr = 1):
    """
    Maps 1D pid to 2D grid coords (pid_m, pid_n).

    Args:
        - pid: 1D pid
        - num_pid_m: grid m size
        - num_pid_n: grid n size
        - GROUP_SIZE_M: tl.constexpr: default is 1
    """
    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        tl.assume(group_size_m >= 0)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    return pid_m, pid_n


@triton.jit
def xcd_swizzle(pid, domain_size, XCD_SWIZZLE: tl.constexpr):
    """
    Swizzle the program id based on integer XCD_SWIZZLE.
    This is useful for reording how blocks are ordered. A scheduler may, for example,
    assign sequential blocks 0, 1, 2, 3, ..., 8, 9, 10.. to its 8 hardware units 0, 1, 2, 3, ..., 0, 1, 2.
    This pattern may not be ideal for memory access, and it may be better to swizzle so the assignment
    becomes 0, 0, 0, 0, ..., 1, 1, 1, ... In the swizzled arrangement, sequential blocks are assigned to
    the same hardware unit.
    """
    # Number of pids per group in the new arrangement
    pids_per_group = domain_size // XCD_SWIZZLE
    extra_pid_groups = domain_size % XCD_SWIZZLE

    # Compute current current and local pid within the group
    group = pid % XCD_SWIZZLE
    local_pid = pid // XCD_SWIZZLE

    # Calculate new pid based on the new grouping
    new_pid = group * pids_per_group + min(group, extra_pid_groups) + local_pid
    return new_pid


@triton.jit
def unswizzle_mx_scale_cdna4(
    x,
    BLOCK_N: tl.constexpr,
    MX_SCALE_BLOCK_K: tl.constexpr,
    N_PRESHUFFLE_FACTOR: tl.constexpr = 32,
):
    x = x.reshape(BLOCK_N // N_PRESHUFFLE_FACTOR, MX_SCALE_BLOCK_K // 8, 4, 16, 2, 2, 1)
    x = x.permute(0, 5, 3, 1, 4, 2, 6)
    x = x.reshape(BLOCK_N, MX_SCALE_BLOCK_K)
    return x


@triton.jit
def _mxfp4_quant_kernel(
    x_ptr,
    x_fp4_ptr,
    bs_ptr,
    stride_x_m,
    stride_x_n,
    stride_fp4_m,
    stride_fp4_n,
    stride_bs_m,
    stride_bs_n,
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: tl.constexpr,
    EVEN_M_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    x_offs = offs_m[:, None] * stride_x_m + offs_n[None, :] * stride_x_n
    x = tl.load(x_ptr + x_offs, mask=mask, other=0).to(tl.float32)

    out_tensor, bs_e8m0 = _mxfp4_quant_op(
        x, BLOCK_SIZE_N, BLOCK_SIZE_M, MXFP4_QUANT_BLOCK_SIZE
    )

    # Store quantized x blocks
    out_offs_n = pid_n * BLOCK_SIZE_N // 2 + tl.arange(0, BLOCK_SIZE_N // 2)
    out_offs = offs_m[:, None] * stride_fp4_m + out_offs_n[None, :] * stride_fp4_n

    if EVEN_M_N:
        tl.store(x_fp4_ptr + out_offs, out_tensor)
    else:
        out_mask = (offs_m < M)[:, None] & (out_offs_n < (N // 2))[None, :]
        tl.store(x_fp4_ptr + out_offs, out_tensor, mask=out_mask)

    # Store scale blocks
    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE
    num_blocks_total = N // MXFP4_QUANT_BLOCK_SIZE
    bs_offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    bs_offs_n = pid_n * NUM_QUANT_BLOCKS + tl.arange(0, NUM_QUANT_BLOCKS)

    bs_offs = bs_offs_m[:, None] * stride_bs_m + bs_offs_n[None, :] * stride_bs_n
    bs_mask = (bs_offs_m < M)[:, None] & (bs_offs_n < num_blocks_total)[None, :]

    if EVEN_M_N:
        tl.store(bs_ptr + bs_offs, bs_e8m0)
    else:
        tl.store(
            bs_ptr + bs_offs,
            bs_e8m0,
            mask=bs_mask,
        )


@triton.jit
def _moe_gemm_a4w4(
    Y,
    stride_y_k,
    stride_y_m,
    stride_y_n,
    X,
    stride_x_m,
    stride_x_k,
    XMxScale,
    stride_x_mx_m,
    stride_x_mx_k,
    W,
    stride_w_e,
    stride_w_k,
    stride_w_n,
    WMxScale,
    stride_w_mx_e,
    stride_w_mx_k,
    stride_w_mx_n,
    X_static_scale,
    Quant_static_scale,
    B,
    stride_b_e,  # Bias
    Gammas,
    N,
    K,  # shapes
    # expt data
    GatherIndx,
    ExptHist,
    ExptOffs,
    ExptOffsSum,
    ExptData,
    # true grid size
    grid_m,
    grid_n,
    # fused activation function
    APPLY_SWIGLU: tl.constexpr,
    alpha,
    limit,
    ACTIVATION_REDUCTION_N: tl.constexpr,
    SWIGLU_ADD_RESIDUAL: tl.constexpr,
    # MoE config
    N_EXPTS_ACT: tl.constexpr,
    # optimization config
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    XCD_SWIZZLE: tl.constexpr,
    # One of ["CDNA4", None]
    SWIZZLE_MX_SCALE: tl.constexpr,
    EVEN_K: tl.constexpr,
    MASK_K_LIMIT: tl.constexpr,
    SPLIT_K: tl.constexpr,
    W_CACHE_MODIFIER: tl.constexpr,
    UPCAST_INDICES: tl.constexpr = False,
):
    tl.assume(stride_y_k >= 0)
    tl.assume(stride_y_m >= 0)
    tl.assume(stride_y_n >= 0)
    tl.assume(stride_x_m >= 0)
    tl.assume(stride_x_k >= 0)
    tl.assume(stride_w_e >= 0)
    tl.assume(stride_w_k >= 0)
    tl.assume(stride_w_n >= 0)
    if stride_x_mx_m is not None:
        tl.assume(stride_x_mx_m >= 0)
    if stride_x_mx_k is not None:
        tl.assume(stride_x_mx_k >= 0)
    if stride_w_mx_e is not None:
        tl.assume(stride_w_mx_e >= 0)
    if stride_w_mx_k is not None:
        tl.assume(stride_w_mx_k >= 0)
    if stride_w_mx_n is not None:
        tl.assume(stride_w_mx_n >= 0)
    if B is not None:
        tl.assume(stride_b_e >= 0)
    tl.assume(grid_m >= 0)
    tl.assume(grid_n >= 0)

    is_x_microscaled: tl.constexpr = XMxScale is not None
    MX_PACK_DIVISOR: tl.constexpr = 32
    w_type: tl.constexpr = W.dtype.element_ty
    tl.static_assert(w_type == tl.uint8, "mx_weight_ptr must be uint8 or fp8")
    tl.static_assert(
        WMxScale.dtype.element_ty == tl.uint8, "mx_scale_ptr must be uint8"
    )
    tl.static_assert(
        BLOCK_K % MX_PACK_DIVISOR == 0, "BLOCK_K must be a multiple of MX_PACK_DIVISOR"
    )
    x_type: tl.constexpr = X.dtype.element_ty
    if is_x_microscaled:
        tl.static_assert(x_type == tl.uint8, "mx_act_ptr must be uint8")
        tl.static_assert(
            XMxScale.dtype.element_ty == tl.uint8, "mx_scale_ptr must be uint8"
        )

    OUT_BLOCK_N: tl.constexpr = BLOCK_N // ACTIVATION_REDUCTION_N
    yN = N // ACTIVATION_REDUCTION_N

    pid = tl.program_id(0)
    if ExptOffsSum is not None and XCD_SWIZZLE > 1:
        # Determine how much padding there is on the expert data. This allows us to
        # know the true grid size and avoid processing padding tiles.
        padding_m = grid_m - tl.load(ExptOffsSum)
    else:
        padding_m: tl.constexpr = 0

    index_type: tl.constexpr = tl.int64 if UPCAST_INDICES else tl.int32

    unpadded_m = grid_m - padding_m
    tl.assume(unpadded_m >= 0)
    total_actual_tiles = unpadded_m * grid_n * SPLIT_K
    if padding_m > 0 and pid >= total_actual_tiles:
        return

    # swizzle program ids
    pid_emnk = pid
    if XCD_SWIZZLE != 1:
        pid_emnk = xcd_swizzle(pid_emnk, total_actual_tiles, XCD_SWIZZLE)
    # pid_e = pid_emnk // (unpadded_m * grid_n * SPLIT_K)
    pid_mnk = pid_emnk % (unpadded_m * grid_n * SPLIT_K)
    pid_k = pid_mnk % SPLIT_K
    pid_mn = pid_mnk // SPLIT_K
    pid_m, pid_n = pid_grid(pid_mn, unpadded_m, grid_n, GROUP_M)
    # For split-k, advance to the output k slice
    if SPLIT_K > 1:
        Y += pid_k.to(index_type) * stride_y_k
    # unpack expert data
    expt_data = tl.load(ExptData + pid_m)
    if expt_data == -1:
        return
    expt_id = expt_data & 0x0000FFFF
    block_id = expt_data >> 16
    M = tl.load(ExptHist + expt_id)
    start_m = tl.load(ExptOffs + expt_id)
    expt_id, block_id = expt_id.to(index_type), block_id.to(index_type)
    start_m = start_m.to(index_type)
    pid_n, pid_k = pid_n.to(index_type), pid_k.to(index_type)

    X_M_DIVISOR: tl.constexpr = 1
    X_K_DIVISOR: tl.constexpr = 2
    W_K_DIVISOR: tl.constexpr = 2
    W_N_DIVISOR: tl.constexpr = 1
    PACKED_BLOCK_M_X: tl.constexpr = BLOCK_M // X_M_DIVISOR
    PACKED_BLOCK_K_X: tl.constexpr = BLOCK_K // X_K_DIVISOR
    PACKED_BLOCK_K_W: tl.constexpr = BLOCK_K // W_K_DIVISOR
    PACKED_BLOCK_N_W: tl.constexpr = BLOCK_N // W_N_DIVISOR
    MX_SCALE_BLOCK_K: tl.constexpr = BLOCK_K // MX_PACK_DIVISOR

    # A pointers
    offs_x_m = PACKED_BLOCK_M_X * block_id + tl.arange(0, PACKED_BLOCK_M_X)
    offs_x_m = tl.max_contiguous(
        tl.multiple_of(offs_x_m % (M // X_M_DIVISOR), PACKED_BLOCK_M_X),
        PACKED_BLOCK_M_X,
    )
    if GatherIndx is None:
        X += start_m * stride_x_m
    else:
        GatherIndx += start_m
        # no needs to bounds-check here because `offs_x_m` wraps around M dim
        offs_x_m = tl.load(GatherIndx + offs_x_m) // N_EXPTS_ACT
    offs_x_k = PACKED_BLOCK_K_X * pid_k + tl.arange(0, PACKED_BLOCK_K_X)
    XPtrs = (
        X
        + offs_x_m.to(index_type)[:, None] * stride_x_m
        + offs_x_k.to(index_type)[None, :] * stride_x_k
    )

    WMxScale += expt_id * stride_w_mx_e
    if SWIZZLE_MX_SCALE == "CDNA4_SCALE":
        tl.static_assert(stride_w_mx_k is not None)
        tl.static_assert(stride_w_mx_n is not None)
        NON_K_PRESHUFFLE_BLOCK_SIZE: tl.constexpr = 32
        PACKED_MX_BLOCK: tl.constexpr = MX_SCALE_BLOCK_K * NON_K_PRESHUFFLE_BLOCK_SIZE
        SCALE_BLOCK_N: tl.constexpr = BLOCK_N // NON_K_PRESHUFFLE_BLOCK_SIZE
    else:
        PACKED_MX_BLOCK: tl.constexpr = MX_SCALE_BLOCK_K
        SCALE_BLOCK_N: tl.constexpr = BLOCK_N
    offs_w_n_scale = (pid_n * SCALE_BLOCK_N + tl.arange(0, SCALE_BLOCK_N)) % N
    offs_w_n_scale = tl.max_contiguous(
        tl.multiple_of(offs_w_n_scale, SCALE_BLOCK_N), SCALE_BLOCK_N
    )
    # K dimension must be the last dimension for the scales
    offs_w_k_scale = PACKED_MX_BLOCK * pid_k + tl.arange(0, PACKED_MX_BLOCK)
    WMxScalePtrs = (
        WMxScale
        + offs_w_k_scale.to(index_type)[None, :] * stride_w_mx_k
        + offs_w_n_scale.to(index_type)[:, None] * stride_w_mx_n
    )

    # B pointers
    offs_w_n = pid_n * PACKED_BLOCK_N_W + tl.arange(0, PACKED_BLOCK_N_W)
    offs_w_n = tl.max_contiguous(
        tl.multiple_of(offs_w_n % (N // W_N_DIVISOR), PACKED_BLOCK_N_W),
        PACKED_BLOCK_N_W,
    )
    offs_w_k = PACKED_BLOCK_K_W * pid_k + tl.arange(0, PACKED_BLOCK_K_W)
    W += expt_id * stride_w_e
    WPtrs = W + (
        offs_w_k.to(index_type)[:, None] * stride_w_k
        + offs_w_n.to(index_type)[None, :] * stride_w_n
    )

    if is_x_microscaled:
        if GatherIndx is None:
            XMxScale += start_m * stride_x_mx_m
        offs_x_k_scale = MX_SCALE_BLOCK_K * pid_k + tl.arange(0, MX_SCALE_BLOCK_K)
        XMxScalePtrs = (
            XMxScale
            + offs_x_m.to(index_type)[:, None] * stride_x_mx_m
            + offs_x_k_scale.to(index_type)[None, :] * stride_x_mx_k
        )

    num_k_iter = tl.cdiv(K, BLOCK_K * SPLIT_K)
    if not EVEN_K:
        num_k_iter -= 1

    # compute output
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(num_k_iter):
        x = tl.load(XPtrs)
        w = tl.load(WPtrs, cache_modifier=W_CACHE_MODIFIER)

        if is_x_microscaled:
            x_scales = tl.load(XMxScalePtrs)
        else:
            x_scales = tl.full((BLOCK_M, MX_SCALE_BLOCK_K), 127, dtype=tl.uint8)
        if SWIZZLE_MX_SCALE == "CDNA4_SCALE":
            w_scales = unswizzle_mx_scale_cdna4(
                tl.load(WMxScalePtrs, cache_modifier=W_CACHE_MODIFIER),
                BLOCK_N,
                MX_SCALE_BLOCK_K,
            )
        else:
            w_scales = tl.load(WMxScalePtrs)

        acc = tl.dot_scaled(
            x, x_scales, "e2m1", w, w_scales, "e2m1", acc=acc, fast_math=True
        )

        WMxScalePtrs += (PACKED_MX_BLOCK * SPLIT_K) * stride_w_mx_k
        if is_x_microscaled:
            XMxScalePtrs += (MX_SCALE_BLOCK_K * SPLIT_K) * stride_x_mx_k

        XPtrs += (PACKED_BLOCK_K_X * SPLIT_K) * stride_x_k
        WPtrs += (PACKED_BLOCK_K_W * SPLIT_K) * stride_w_k

    if not EVEN_K:
        mask_x_k = offs_x_k < (MASK_K_LIMIT // X_K_DIVISOR)
        mask_w_k = offs_w_k < (MASK_K_LIMIT // W_K_DIVISOR)
        if SWIZZLE_MX_SCALE is None:
            mask_w_k_scale = offs_w_k_scale * MX_PACK_DIVISOR < MASK_K_LIMIT
        if is_x_microscaled:
            mask_x_k_scale = offs_x_k_scale * MX_PACK_DIVISOR < MASK_K_LIMIT

        x = tl.load(XPtrs, mask=mask_x_k[None, :], other=0)
        w = tl.load(
            WPtrs, mask=mask_w_k[:, None], other=0, cache_modifier=W_CACHE_MODIFIER
        )

        if is_x_microscaled:
            x_scales = tl.load(XMxScalePtrs, mask=mask_x_k_scale[None, :])
        else:
            x_scales = tl.full((BLOCK_M, MX_SCALE_BLOCK_K), 127, dtype=tl.uint8)
        if SWIZZLE_MX_SCALE == "CDNA4_SCALE":
            w_scales = unswizzle_mx_scale_cdna4(
                tl.load(WMxScalePtrs, cache_modifier=W_CACHE_MODIFIER),
                BLOCK_N,
                MX_SCALE_BLOCK_K,
            )
        else:
            w_scales = tl.load(WMxScalePtrs, mask=mask_w_k_scale[None, :])

        acc = tl.dot_scaled(
            x, x_scales, "e2m1", w, w_scales, "e2m1", acc=acc, fast_math=True
        )

    # scalar fp8 scale
    if X_static_scale is not None:
        # should not go in here since static scale fp4 is disabled
        tl.static_assert(
            X_static_scale is None,
            f"Static scale is disabled for fp4 precision. got {X_static_scale}",
        )

    # bias
    offs_m = BLOCK_M * block_id + tl.arange(0, BLOCK_M)
    offs_y_n = BLOCK_N * pid_n + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_y_n < N
    if B is not None:
        BPtrs = B + expt_id * stride_b_e + offs_y_n
        if pid_k == 0:
            bias = tl.load(BPtrs, mask=mask_n, other=0, cache_modifier=W_CACHE_MODIFIER)
        else:
            bias = tl.full([BLOCK_N], 0, dtype=tl.float32)
        acc = acc + bias[None, :]
    if APPLY_SWIGLU and SPLIT_K == 1:
        out = _swiglu(acc, alpha, limit, ADD_RESIDUAL=SWIGLU_ADD_RESIDUAL)
        tl.static_assert(
            out.shape[1] == OUT_BLOCK_N,
            f"Activation fn out.shape[1] ({out.shape[1]}) doesn't match computed OUT_BLOCK_N ({OUT_BLOCK_N})",
        )
        offs_y_n = OUT_BLOCK_N * pid_n + tl.arange(0, OUT_BLOCK_N)
        mask_n = offs_y_n < yN
    else:
        tl.static_assert(
            ACTIVATION_REDUCTION_N == 1,
            "Activation reduction must be 1 if no activation fn is provided",
        )
        out = acc
    if Gammas is not None:
        gammas = tl.load(Gammas + start_m + offs_m, mask=mask_m, other=0.0)
        out *= gammas[:, None]
    # quant
    if Quant_static_scale is not None:
        out = _compute_static_fp8_quant(out, tl.load(Quant_static_scale))
    # write-back
    Y += start_m * stride_y_m
    offs_y_m = offs_m
    YPtrs = (
        Y
        + offs_y_m.to(index_type)[:, None] * stride_y_m
        + offs_y_n.to(index_type)[None, :] * stride_y_n
    )
    mask = mask_m[:, None] & mask_n[None, :]
    tl.store(YPtrs, out, mask=mask)
