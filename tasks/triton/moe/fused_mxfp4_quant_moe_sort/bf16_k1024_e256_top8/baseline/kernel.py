# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Fused dynamic MXFP4 quantization and MoE scale sorting in Triton, adapted from AITER."""
import torch
import triton
import triton.language as tl


@triton.jit
def _fused_dynamic_mxfp4_quant_moe_sort_kernel(
    x_ptr,
    x_fp4_ptr,
    sorted_ids_ptr,
    num_valid_ids_ptr,
    blockscale_e8m0_sorted_ptr,
    Mx,
    Nx,
    scaleNx,
    stride_x_m,
    stride_x_n,
    stride_x_fp4_m,
    stride_x_fp4_n,
    stride_o3,  #: tl.constexpr,
    stride_o2,  #: tl.constexpr,
    stride_o1,  #: tl.constexpr,
    stride_o0,  #: tl.constexpr,
    stride_o4,  #: tl.constexpr,
    token_num,  #: tl.constexpr,
    N_i,  #: tl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_Mx: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    TOPK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_x = tl.cdiv(Mx, BLOCK_SIZE_Mx) * scaleNx

    stride_x_m = tl.cast(stride_x_m, tl.int64)
    stride_x_n = tl.cast(stride_x_n, tl.int64)
    stride_x_fp4_m = tl.cast(stride_x_fp4_m, tl.int64)
    stride_x_fp4_n = tl.cast(stride_x_fp4_n, tl.int64)

    if pid < num_pid_x:
        pid_m = pid // scaleNx
        pid_n = pid % scaleNx

        x_offs_m = pid_m * BLOCK_SIZE_Mx + tl.arange(0, BLOCK_SIZE_Mx)
        x_offs_n = pid_n * MXFP4_QUANT_BLOCK_SIZE + tl.arange(0, MXFP4_QUANT_BLOCK_SIZE)
        x_offs = x_offs_m[:, None] * stride_x_m + x_offs_n[None, :] * stride_x_n
        x_mask = (x_offs_m < Mx)[:, None] & (x_offs_n < Nx)[None, :]
        x = tl.load(x_ptr + x_offs, mask=x_mask).to(tl.float32)

        # Calculate scale
        amax = tl.max(tl.abs(x), axis=1, keep_dims=True)
        amax = amax.to(tl.int32, bitcast=True)
        amax = (amax + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
        amax = amax.to(tl.float32, bitcast=True)
        scale_e8m0_unbiased = tl.log2(amax).floor() - 2
        scale_e8m0_unbiased = tl.clamp(scale_e8m0_unbiased, min=-127, max=127)
        quant_scale = tl.exp2(-scale_e8m0_unbiased)

        # Compute quantized x
        qx = x * quant_scale

        # blockscale_e8m0
        # bs_e8m0 = scale_e8m0_unbiased.to(tl.uint8) + 127

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

        # Extract sign, exponents and mantissa fields from FP32
        s = qx & 0x80000000
        e = (qx >> 23) & 0xFF
        m = qx & 0x7FFFFF

        E8_BIAS: tl.constexpr = 127
        E2_BIAS: tl.constexpr = 1

        # Denormal numbers
        # If exponent is less than 127, then it's a denormal number
        # See above, for denormal number mantissa is always 1 and we set bit 1 of mantissa
        adjusted_exponents = tl.core.sub(E8_BIAS, e + 1, sanitize_overflow=False)
        m = tl.where(e < E8_BIAS, (0x400000 | (m >> 1)) >> adjusted_exponents, m)

        # For normal numbers, bias is changed from 127 to 1, and for subnormals, we keep exponent as 0.
        # Note: E8_BIAS - E2_BIAS = 126, so for normals we subtract that.
        e = tl.maximum(e, E8_BIAS - E2_BIAS) - (E8_BIAS - E2_BIAS)

        # Combine sign, exponent, and mantissa, while saturating
        # rounding nearest with tie breaking up by adding +1 to one bit right of the LSB, then shift right
        e2m1_tmp = tl.minimum((((e << 2) | (m >> 21)) + 1) >> 1, 0x7)
        e2m1_value = ((s >> 28) | e2m1_tmp).to(tl.uint8)

        e2m1_value = tl.reshape(
            e2m1_value, [BLOCK_SIZE_Mx, MXFP4_QUANT_BLOCK_SIZE // 2, 2]
        )
        evens, odds = tl.split(e2m1_value)
        out_tensor = evens | (odds << 4)

        out_offs_m = pid_m * BLOCK_SIZE_Mx + tl.arange(0, BLOCK_SIZE_Mx)
        out_offs_n = pid_n * MXFP4_QUANT_BLOCK_SIZE // 2 + tl.arange(
            0, MXFP4_QUANT_BLOCK_SIZE // 2
        )
        out_offs = (
            out_offs_m[:, None] * stride_x_fp4_m + out_offs_n[None, :] * stride_x_fp4_n
        )
        out_mask = (out_offs_m < Mx)[:, None] & (out_offs_n < (Nx // 2))[None, :]
        tl.store(x_fp4_ptr + out_offs, out_tensor, mask=out_mask)

        return

    pid -= num_pid_x
    num_pid_n = tl.cdiv(N_i, BLOCK_SIZE_N * 2)
    pid_m = pid // num_pid_n  # * 2
    pid_n = pid % num_pid_n  # * 2
    # pid_m = tl.program_id(0) * 2
    # pid_n = tl.program_id(1) * 2
    num_valid_ids = tl.load(num_valid_ids_ptr)
    if pid_m * BLOCK_SIZE_M * 2 >= num_valid_ids:
        return
    stride_o0 = tl.cast(stride_o0, tl.int64)
    stride_o1 = tl.cast(stride_o1, tl.int64)
    stride_o2 = tl.cast(stride_o2, tl.int64)
    stride_o3 = tl.cast(stride_o3, tl.int64)
    stride_o4 = tl.cast(stride_o4, tl.int64)

    BLOCK_SIZE_Nb: tl.constexpr = BLOCK_SIZE_N * 2 * MXFP4_QUANT_BLOCK_SIZE
    sorted_ids_offs_m = pid_m * BLOCK_SIZE_M * 2 + tl.arange(0, BLOCK_SIZE_M * 2)
    sorted_ids_offs = sorted_ids_offs_m
    sorted_ids_mask = sorted_ids_offs_m < num_valid_ids
    sorted_ids = tl.load(
        sorted_ids_ptr + sorted_ids_offs,
        mask=sorted_ids_mask,
        other=token_num,
        # sorted_ids_ptr + sorted_ids_offs, mask=sorted_ids_mask, other=Mx
    )
    topk_ids = sorted_ids >> 24
    sorted_ids = sorted_ids & 0xFFFFFF
    if TOPK == 1:
        x_offs_m = sorted_ids
    else:
        x_offs_m = sorted_ids * TOPK + topk_ids
    # if pid == 0:
    #     tl.device_print("x_offs_m", x_offs_m)
    x_offs_n = pid_n * BLOCK_SIZE_Nb + tl.arange(0, BLOCK_SIZE_Nb)
    x_offs = x_offs_m[:, None] * stride_x_m + x_offs_n[None, :] * stride_x_n
    x_mask = (sorted_ids < token_num)[:, None] & (x_offs_n < Nx)[None, :]
    # x_mask = (x_offs_m < Mx)[:, None] & (x_offs_n < Nx)[None, :]
    x = tl.load(x_ptr + x_offs, mask=x_mask).to(tl.float32)
    x = x.reshape(BLOCK_SIZE_M * 2, BLOCK_SIZE_N * 2, MXFP4_QUANT_BLOCK_SIZE)

    # Calculate scale
    amax = tl.max(tl.abs(x), axis=-1, keep_dims=True)
    amax = amax.to(tl.int32, bitcast=True)
    amax = (amax + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    amax = amax.to(tl.float32, bitcast=True)
    scale_e8m0_unbiased = tl.log2(amax).floor() - 2
    scale_e8m0_unbiased = tl.clamp(scale_e8m0_unbiased, min=-127, max=127)
    # blockscale_e8m0
    bs_e8m0 = scale_e8m0_unbiased.to(tl.uint8) + 127
    bs_e8m0 = (
        bs_e8m0.reshape(2, BLOCK_SIZE_M, 2, BLOCK_SIZE_N)
        .permute(1, 3, 2, 0)
        .reshape(BLOCK_SIZE_M, BLOCK_SIZE_N, 4)
    )
    out = bs_e8m0

    # Store the result
    # 16x4 uint32 -> 32x2 uint8
    offs_0 = tl.arange(0, BLOCK_SIZE_M)
    offs_1 = tl.arange(0, BLOCK_SIZE_N)
    offs_2 = pid_n  # // 2
    offs_3 = pid_m  # // 2
    offs_4 = tl.arange(0, 4)
    offs = (
        offs_0[:, None, None] * stride_o0
        + offs_1[None, :, None] * stride_o1  # * BLOCK_SIZE_M
        + offs_2 * stride_o2  # * BLOCK_SIZE_M * BLOCK_SIZE_N
        + offs_3 * stride_o3  # * BLOCK_SIZE_M * BLOCK_SIZE_N * N_i // BLOCK_SIZE_N
        + offs_4[None, None, :] * stride_o4
    )
    # blockscale_e8m0_sorted_mask = (blockscale_e8m0_sorted_offs_m < M_o)[:, None] & (
    #     blockscale_e8m0_sorted_offs_n < N_o
    # )[None, :]
    tl.store(
        blockscale_e8m0_sorted_ptr + offs,
        out,
        # mask=blockscale_e8m0_sorted_mask,
    )


# ============================================================================
# LAUNCH
# ============================================================================


def run(*, x, sorted_ids, num_valid_ids, token_num, x_fp4, x_scales_sorted):
    M, N = x.shape
    topk = M // token_num
    assert (N // 2) % 2 == 0

    # This is fixed by the MXFP4 specification.
    MXFP4_QUANT_BLOCK_SIZE = 32
    scaleN = triton.cdiv(N, MXFP4_QUANT_BLOCK_SIZE)

    # Smaller quant block for small token counts reduces wasted masked work.
    BLOCK_SIZE_Mx = 32 if M <= 32 else 128

    BLOCK_SIZE_M, BLOCK_SIZE_N = 32, 8
    BLOCK_SIZE_M_u32, BLOCK_SIZE_N_u32 = 16, 4

    N_i = scaleN
    M_o = sorted_ids.shape[0]
    assert (N_i // 2) % 2 == 0

    # The sorted scales are stored as [M_o / 32, N_i / 8, 4, 16, 4] byte tiles.
    scales = x_scales_sorted.view(
        triton.cdiv(M_o, BLOCK_SIZE_M), triton.cdiv(N_i, BLOCK_SIZE_N), BLOCK_SIZE_N_u32, BLOCK_SIZE_M_u32, 4
    )

    num_pid = triton.cdiv(M, BLOCK_SIZE_Mx) * scaleN + triton.cdiv(M_o, BLOCK_SIZE_M) * triton.cdiv(
        N_i, BLOCK_SIZE_N
    )
    _fused_dynamic_mxfp4_quant_moe_sort_kernel[(num_pid,)](
        x,
        x_fp4,
        sorted_ids,
        num_valid_ids,
        scales,
        M,
        N,
        scaleN,
        *x.stride(),
        *x_fp4.stride(),
        *scales.stride(),
        token_num=token_num,
        N_i=N_i,
        MXFP4_QUANT_BLOCK_SIZE=MXFP4_QUANT_BLOCK_SIZE,
        BLOCK_SIZE_Mx=BLOCK_SIZE_Mx,
        BLOCK_SIZE_M=BLOCK_SIZE_M // 2,
        BLOCK_SIZE_N=BLOCK_SIZE_N // 2,
        TOPK=topk,
    )
