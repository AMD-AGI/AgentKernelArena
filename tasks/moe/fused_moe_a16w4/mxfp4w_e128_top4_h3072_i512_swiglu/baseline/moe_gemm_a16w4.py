# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Vendored from ROCm/aiter (commit 652f2d4b768239aefd515c929e84817681ff598d):
#   aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a16w4.py  (xcd_swizzle, unswizzle_mx_scale_cdna4,
#                                                               _moe_gemm_a16w4)
#   aiter/ops/triton/_triton_kernels/moe/activations.py        (clip, _swiglu)
#   aiter/ops/triton/utils/_triton/pid_preprocessing.py         (pid_grid)
# AITER adapted these kernels from the triton_kernels package (MIT):
# adapted from triton_kernels package
# original code https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/matmul_ogs_details/_matmul_ogs.py
#
# Changes: imports made local; the profiling-only ``launch_metadata`` hook of _moe_gemm_a16w4 removed;
# the _swiglu parameter ``input`` renamed ``acc_in`` (the candidate rules reserve builtin names).

import triton
import triton.language as tl


@triton.jit
def clip(x, limit, clip_lower: tl.constexpr):
    res = tl.minimum(x, limit)
    if clip_lower:
        res = tl.maximum(-limit, res)
    return res


@triton.jit
def _swiglu(acc_in, alpha, limit, ADD_RESIDUAL: tl.constexpr):
    """
    SwiGLU activation

    s = silu(gelu), then returns s * (linear + 1) if ADD_RESIDUAL else s * linear.
    if alpha=1.0, then this is the same as the SiLU activation.
    """
    gelu, linear = tl.split(tl.reshape(acc_in, (acc_in.shape[0], acc_in.shape[1] // 2, 2)))
    gelu = gelu.to(tl.float32)
    if limit is not None:
        gelu = clip(gelu, limit, clip_lower=False)
    linear = linear.to(tl.float32)
    if limit is not None:
        linear = clip(linear, limit, clip_lower=True)
    s = gelu / (1 + tl.exp2(-1.44269504089 * alpha * gelu))
    if ADD_RESIDUAL:
        return tl.fma(s, linear, s)  # s * (linear + 1)
    else:
        return s * linear


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
def _moe_gemm_a16w4(
    Y,
    stride_y_k,
    stride_y_m,
    stride_y_n,
    X,
    stride_x_m,
    stride_x_k,
    W,
    stride_w_e,
    stride_w_k,
    stride_w_n,
    WMxScale,
    stride_w_mx_e,
    stride_w_mx_k,
    stride_w_mx_n,
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

    MX_PACK_DIVISOR: tl.constexpr = 32
    w_type: tl.constexpr = W.dtype.element_ty
    tl.static_assert(w_type == tl.uint8, "mx_weight_ptr must be uint8")
    tl.static_assert(
        WMxScale.dtype.element_ty == tl.uint8, "mx_scale_ptr must be uint8"
    )
    tl.static_assert(
        BLOCK_K % MX_PACK_DIVISOR == 0, "BLOCK_K must be a multiple of MX_PACK_DIVISOR"
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

    # X pointers
    offs_x_m = BLOCK_M * block_id + tl.arange(0, BLOCK_M)
    offs_x_m = tl.max_contiguous(tl.multiple_of(offs_x_m % M, BLOCK_M), BLOCK_M)
    if GatherIndx is None:
        X += start_m * stride_x_m
    else:
        GatherIndx += start_m
        # no needs to bounds-check here because `offs_x_m` wraps around M dim
        offs_x_m = tl.load(GatherIndx + offs_x_m) // N_EXPTS_ACT
    offs_x_k = BLOCK_K * pid_k + tl.arange(0, BLOCK_K)
    XPtrs = (
        X
        + offs_x_m.to(index_type)[:, None] * stride_x_m
        + offs_x_k.to(index_type)[None, :] * stride_x_k
    )

    W_K_DIVISOR: tl.constexpr = 2
    W_N_DIVISOR: tl.constexpr = 1
    PACKED_BLOCK_K_W: tl.constexpr = BLOCK_K // W_K_DIVISOR
    PACKED_BLOCK_N_W: tl.constexpr = BLOCK_N // W_N_DIVISOR
    MX_SCALE_BLOCK_K: tl.constexpr = BLOCK_K // MX_PACK_DIVISOR

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

    # W pointers
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

    num_k_iter = tl.cdiv(K, BLOCK_K * SPLIT_K)
    if not EVEN_K:
        num_k_iter -= 1

    # compute output
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(num_k_iter):
        x = tl.load(XPtrs)
        w = tl.load(WPtrs, cache_modifier=W_CACHE_MODIFIER)

        x_scales: tl.constexpr = None
        if SWIZZLE_MX_SCALE == "CDNA4_SCALE":
            w_scales = unswizzle_mx_scale_cdna4(
                tl.load(WMxScalePtrs, cache_modifier=W_CACHE_MODIFIER),
                BLOCK_N,
                MX_SCALE_BLOCK_K,
            )
        else:
            w_scales = tl.load(WMxScalePtrs)

        acc = tl.dot_scaled(
            x, x_scales, "bf16", w, w_scales, "e2m1", acc=acc, fast_math=True
        )

        WMxScalePtrs += (PACKED_MX_BLOCK * SPLIT_K) * stride_w_mx_k

        XPtrs += (BLOCK_K * SPLIT_K) * stride_x_k
        WPtrs += (PACKED_BLOCK_K_W * SPLIT_K) * stride_w_k

    if not EVEN_K:
        mask_x_k = offs_x_k < MASK_K_LIMIT
        mask_w_k = offs_w_k < (MASK_K_LIMIT // W_K_DIVISOR)
        if SWIZZLE_MX_SCALE is None:
            mask_w_k_scale = offs_w_k_scale * MX_PACK_DIVISOR < MASK_K_LIMIT

        x = tl.load(XPtrs, mask=mask_x_k[None, :], other=0.0)
        w = tl.load(
            WPtrs, mask=mask_w_k[:, None], other=0, cache_modifier=W_CACHE_MODIFIER
        )

        x_scales: tl.constexpr = None
        if SWIZZLE_MX_SCALE == "CDNA4_SCALE":
            w_scales = unswizzle_mx_scale_cdna4(
                tl.load(WMxScalePtrs, cache_modifier=W_CACHE_MODIFIER),
                BLOCK_N,
                MX_SCALE_BLOCK_K,
            )
        else:
            w_scales = tl.load(WMxScalePtrs, mask=mask_w_k_scale[None, :])

        acc = tl.dot_scaled(
            x, x_scales, "bf16", w, w_scales, "e2m1", acc=acc, fast_math=True
        )

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
