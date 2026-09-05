#!/usr/bin/env python3
"""
Test harness for fused_dynamic_mxfp4_quant_moe_sort kernel.
Modes: --correctness, --profile, --benchmark, --full-benchmark

Self-contained: zero aiter dependency. All reference helpers and supporting
triton kernels are inlined from aiter (op_tests + aiter.utility.fp4_utils).
"""

import argparse
import itertools
import math
import os
import sys
from _aka_benchmark import benchmark_cuda_graph_or_events_samples


def benchmark_cuda_graph_or_events(*args, **kwargs):
    samples, metadata = benchmark_cuda_graph_or_events_samples(*args, **kwargs)
    values = sorted(samples)
    midpoint = len(values) // 2
    median_ms = (
        values[midpoint]
        if len(values) % 2
        else (values[midpoint - 1] + values[midpoint]) / 2.0
    )
    return median_ms, metadata

# Ensure line-buffered stdout
sys.stdout.reconfigure(line_buffering=True)

import torch
import triton
import triton.language as tl

torch.manual_seed(42)

# Kernel under test — kernel.py sits next to this harness. Python adds the
# script's directory to sys.path[0] automatically.
from kernel import fused_dynamic_mxfp4_quant_moe_sort  # noqa: E402


def _is_fp4_avail() -> bool:
    """MXFP4 support is gated on gfx950 / gfx1250."""
    try:
        arch = triton.runtime.driver.active.get_current_target().arch
    except Exception:
        return False
    return arch in ("gfx950", "gfx1250")


# ============================================================================
# INLINED DTYPE ALIASES (from aiter.utility.dtypes)
# ============================================================================

_8bit_fallback = torch.uint8
_fp4x2 = getattr(torch, "float4_e2m1fn_x2", _8bit_fallback)
_fp8_e8m0 = getattr(torch, "float8_e8m0fnu", _8bit_fallback)


# ============================================================================
# INLINED REFERENCE HELPERS
# from op_tests/triton_tests/gemm/basic/test_gemm_afp4wfp4.py
# ============================================================================


SCALE_GROUP_SIZE = 32


def mxfp4_to_f32(x):
    # 2 because we pack fp4 in uint8.
    x = x.repeat_interleave(2, dim=1)
    x[:, ::2] = x[:, ::2] & 0xF
    x[:, 1::2] = x[:, 1::2] >> 4
    mxfp4_list = [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
    mxfp4_in_f32 = torch.tensor(mxfp4_list, dtype=torch.float32, device="cuda")
    return mxfp4_in_f32[x.long()]


def e8m0_to_f32(x):
    x_f32 = 2 ** ((x - 127).to(torch.float32))
    x_f32[x_f32 == 128] = float("nan")
    return x_f32


# from op_tests/triton_tests/quant/test_fused_mxfp4_quant.py
def convert_mxfp4_to_fp32(x, x_scales):
    x_f32 = mxfp4_to_f32(x)
    x_scales = x_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
    x_scales_f32 = e8m0_to_f32(x_scales)[:, : x_f32.shape[1]]
    x_f32 = x_f32 * x_scales_f32
    return x_f32


# ============================================================================
# INLINED REFERENCE QUANT (replaces aiter.ops.quant.per_1x32_f4_quant_hip)
# Pure-torch MXFP4 quantization, mirrors the HIP op semantics exactly.
# Adapted from the gold-standard L3/fused_moe_mxfp4 harness.
# ============================================================================


def _torch_dynamic_mxfp4_quant(x: torch.Tensor):
    """Quantize a bf16/fp16 tensor to MXFP4 (packed uint8) + E8M0 scale."""
    MXFP4_QUANT_BLOCK_SIZE = 32
    x_shape = x.shape
    if x.shape[-1] % MXFP4_QUANT_BLOCK_SIZE != 0:
        shape = list(x_shape)
        shape[-1] = (
            (shape[-1] - 1 + MXFP4_QUANT_BLOCK_SIZE) // MXFP4_QUANT_BLOCK_SIZE
        ) * MXFP4_QUANT_BLOCK_SIZE
        x_padded = torch.zeros(tuple(shape), device=x.device, dtype=x.dtype)
        x_padded[..., : x.shape[-1]] = x
    else:
        x_padded = x

    x_padded = x_padded.reshape(
        -1, x_padded.shape[-1] // MXFP4_QUANT_BLOCK_SIZE, MXFP4_QUANT_BLOCK_SIZE
    ).to(torch.float32)
    amax, _ = torch.max(torch.abs(x_padded), dim=-1)
    amax = amax.view(torch.int32)
    amax = (amax + 0x200000) & 0xFF800000
    amax = amax.view(torch.float32)
    scale_e8m0_unbiased = torch.log2(amax).floor() - 2
    scale_e8m0_unbiased = torch.clamp(scale_e8m0_unbiased, min=-127, max=127)
    quant_scale = torch.exp2(-scale_e8m0_unbiased)
    qx = x_padded * quant_scale.unsqueeze(-1)
    bs_e8m0 = scale_e8m0_unbiased.to(torch.uint8) + 127

    qx = qx.view(torch.int32)
    s = qx & 0x80000000
    e = (qx >> 23) & 0xFF
    m = qx & 0x7FFFFF
    E8_BIAS = 127
    E2_BIAS = 1
    adjusted_exponents = E8_BIAS - e - 1
    m = torch.where(e < E8_BIAS, (0x400000 | (m >> 1)) >> adjusted_exponents, m)
    e = torch.where(e > E8_BIAS - E2_BIAS, e, E8_BIAS - E2_BIAS) - (E8_BIAS - E2_BIAS)
    combined_val = (((e << 2) | (m >> 21)) + 1) >> 1
    e2m1_tmp = torch.where(combined_val < 0x7, combined_val, 0x7)
    e2m1_value = (((s >> 28) & 0xF) | e2m1_tmp).to(torch.uint8)
    x_mxfp4 = e2m1_value[..., ::2] | (e2m1_value[..., 1::2] << 4)
    x_mxfp4 = torch.flatten(x_mxfp4, -2, -1)
    if x.shape[-1] % MXFP4_QUANT_BLOCK_SIZE != 0:
        x_mxfp4 = x_mxfp4[..., : x.shape[-1] // 2]

    mxfp4_shape = tuple(list(x_shape)[:-1] + [x_shape[-1] // 2])
    x_mxfp4 = x_mxfp4.reshape(mxfp4_shape)
    bs_e8m0_shape = tuple(
        list(x_shape)[:-1] + [x_shape[-1] // MXFP4_QUANT_BLOCK_SIZE]
    )
    bs_e8m0 = bs_e8m0.reshape(bs_e8m0_shape)
    return x_mxfp4, bs_e8m0


# ============================================================================
# INLINED TRITON KERNEL: dynamic_mxfp4_quant
# from aiter.utility.fp4_utils._dynamic_mxfp4_quant_kernel_asm_layout
# Used only to produce the *triton-native* scales that the dequant comparison
# step requires (matches the reference test exactly).
# ============================================================================


@triton.jit
def _dynamic_mxfp4_quant_kernel_asm_layout(
    x_ptr,
    x_fp4_ptr,
    bs_ptr,
    stride_x_m,
    stride_x_n,
    stride_x_fp4_m,
    stride_x_fp4_n,
    stride_bs_m,
    stride_bs_n,
    M: tl.constexpr,
    N: tl.constexpr,
    scaleN: tl.constexpr,
    scaleM_pad: tl.constexpr,
    scaleN_pad: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: tl.constexpr,
    SCALING_MODE: tl.constexpr,
    SHUFFLE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    stride_x_m = tl.cast(stride_x_m, tl.int64)
    stride_x_n = tl.cast(stride_x_n, tl.int64)
    stride_x_fp4_m = tl.cast(stride_x_fp4_m, tl.int64)
    stride_x_fp4_n = tl.cast(stride_x_fp4_n, tl.int64)

    x_offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x_offs_n = pid_n * MXFP4_QUANT_BLOCK_SIZE + tl.arange(0, MXFP4_QUANT_BLOCK_SIZE)
    x_offs = x_offs_m[:, None] * stride_x_m + x_offs_n[None, :] * stride_x_n
    x_mask = (x_offs_m < M)[:, None] & (x_offs_n < N)[None, :]
    x = tl.load(x_ptr + x_offs, mask=x_mask).to(tl.float32)

    # Calculate scale
    amax = tl.max(tl.abs(x), axis=1, keep_dims=True)
    amax = amax.to(tl.int32, bitcast=True)
    amax = (amax + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    amax = amax.to(tl.float32, bitcast=True)
    scale_e8m0_unbiased = tl.log2(amax).floor() - 2
    scale_e8m0_unbiased = tl.clamp(scale_e8m0_unbiased, min=-127, max=127)
    quant_scale = tl.exp2(-scale_e8m0_unbiased)

    qx = x * quant_scale

    bs_e8m0 = scale_e8m0_unbiased.to(tl.uint8) + 127

    EXP_BIAS_FP32: tl.constexpr = 127
    EXP_BIAS_FP4: tl.constexpr = 1
    EBITS_F32: tl.constexpr = 8
    EBITS_FP4: tl.constexpr = 2
    MBITS_F32: tl.constexpr = 23
    MBITS_FP4: tl.constexpr = 1

    max_normal: tl.constexpr = 6
    min_normal: tl.constexpr = 1

    qx = qx.to(tl.uint32, bitcast=True)

    s = qx & 0x80000000
    qx = qx ^ s

    qx_fp32 = qx.to(tl.float32, bitcast=True)
    saturate_mask = qx_fp32 >= max_normal
    denormal_mask = (not saturate_mask) & (qx_fp32 < min_normal)
    normal_mask = not (saturate_mask | denormal_mask)

    denorm_exp: tl.constexpr = (
        (EXP_BIAS_FP32 - EXP_BIAS_FP4) + (MBITS_F32 - MBITS_FP4) + 1
    )
    denorm_mask_int: tl.constexpr = denorm_exp << MBITS_F32
    denorm_mask_float: tl.constexpr = tl.cast(denorm_mask_int, tl.float32, bitcast=True)

    denormal_x = qx_fp32 + denorm_mask_float
    denormal_x = denormal_x.to(tl.uint32, bitcast=True)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(tl.uint8)

    normal_x = qx
    mant_odd = (normal_x >> (MBITS_F32 - MBITS_FP4)) & 1
    val_to_add = ((EXP_BIAS_FP4 - EXP_BIAS_FP32) << MBITS_F32) + (1 << 21) - 1
    normal_x += val_to_add
    normal_x += mant_odd
    normal_x = normal_x >> (MBITS_F32 - MBITS_FP4)
    normal_x = normal_x.to(tl.uint8)

    e2m1_value = tl.full(qx.type.get_block_shapes(), 0x7, dtype=tl.uint8)
    e2m1_value = tl.where(normal_mask, normal_x, e2m1_value)
    e2m1_value = tl.where(denormal_mask, denormal_x, e2m1_value)

    sign_lp = s >> (MBITS_F32 + EBITS_F32 - MBITS_FP4 - EBITS_FP4)
    sign_lp = sign_lp.to(tl.uint8)
    e2m1_value = e2m1_value | sign_lp

    e2m1_value = tl.reshape(e2m1_value, [BLOCK_SIZE, MXFP4_QUANT_BLOCK_SIZE // 2, 2])
    evens, odds = tl.split(e2m1_value)
    out_tensor = evens | (odds << 4)

    out_offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_offs_n = pid_n * MXFP4_QUANT_BLOCK_SIZE // 2 + tl.arange(
        0, MXFP4_QUANT_BLOCK_SIZE // 2
    )
    out_offs = (
        out_offs_m[:, None] * stride_x_fp4_m + out_offs_n[None, :] * stride_x_fp4_n
    )
    out_mask = (out_offs_m < M)[:, None] & (out_offs_n < (N // 2))[None, :]
    tl.store(x_fp4_ptr + out_offs, out_tensor, mask=out_mask)

    bs_offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    bs_offs_n = pid_n

    if SHUFFLE:
        bs_offs_0 = bs_offs_m[:, None] // 32
        bs_offs_1 = bs_offs_m[:, None] % 32
        bs_offs_2 = bs_offs_1 % 16
        bs_offs_1 = bs_offs_1 // 16
        bs_offs_3 = bs_offs_n[None, :] // 8
        bs_offs_4 = bs_offs_n[None, :] % 8
        bs_offs_5 = bs_offs_4 % 4
        bs_offs_4 = bs_offs_4 // 4
        bs_offs = (
            bs_offs_1
            + bs_offs_4 * 2
            + bs_offs_2 * 2 * 2
            + bs_offs_5 * 2 * 2 * 16
            + bs_offs_3 * 2 * 2 * 16 * 4
            + bs_offs_0 * 2 * 16 * scaleN_pad
        )
        bs_mask1 = (bs_offs_m < M)[:, None] & (bs_offs_n < scaleN)[None, :]
        bs_mask2 = (bs_offs_m < scaleM_pad)[:, None] & (bs_offs_n < scaleN_pad)[None, :]
        bs_e8m0 = tl.where(bs_mask1, bs_e8m0, 0)
        tl.store(bs_ptr + bs_offs, bs_e8m0, mask=bs_mask2)
    else:
        bs_offs = bs_offs_m[:, None] * stride_bs_m + bs_offs_n[None, :] * stride_bs_n
        bs_mask = (bs_offs_m < M)[:, None] & (bs_offs_n < N)[None, :]
        tl.store(bs_ptr + bs_offs, bs_e8m0, mask=bs_mask)


def dynamic_mxfp4_quant(
    x: torch.Tensor, scaling_mode: str = "even", shuffle: bool = False
):
    """Quantize a tensor to MX FP4 format (triton)."""
    M, N = x.shape
    assert (N // 2) % 2 == 0

    MXFP4_QUANT_BLOCK_SIZE = 32

    x_fp4 = torch.empty((M, N // 2), dtype=torch.uint8, device=x.device)
    scaleM = triton.cdiv(M, 32) * 32
    scaleN_valid = triton.cdiv(N, MXFP4_QUANT_BLOCK_SIZE)
    scaleN = triton.cdiv(scaleN_valid, 8) * 8
    blockscale_e8m0 = torch.empty(
        (triton.cdiv(M, 256) * 256, scaleN),
        dtype=torch.uint8,
        device=x.device,
    )

    BLOCK_SIZE = 128
    grid = (triton.cdiv(M, BLOCK_SIZE), scaleN)
    _dynamic_mxfp4_quant_kernel_asm_layout[grid](
        x,
        x_fp4,
        blockscale_e8m0,
        *x.stride(),
        *x_fp4.stride(),
        *blockscale_e8m0.stride(),
        M=M,
        N=N,
        scaleN=scaleN_valid,
        scaleM_pad=scaleM,
        scaleN_pad=scaleN,
        BLOCK_SIZE=BLOCK_SIZE,
        MXFP4_QUANT_BLOCK_SIZE=MXFP4_QUANT_BLOCK_SIZE,
        SCALING_MODE=0,
        SHUFFLE=shuffle,
    )

    if not shuffle:
        blockscale_e8m0 = blockscale_e8m0[:M, :scaleN_valid].contiguous()

    return (x_fp4.view(_fp4x2), blockscale_e8m0.view(_fp8_e8m0))


# ============================================================================
# INLINED TRITON KERNELS: moe_mxfp4_sort (used for reference sort)
# from aiter.utility.fp4_utils
# ============================================================================


@triton.jit
def _moe_mxfp4_sort_kernel(
    blockscale_e8m0_ptr,
    sorted_ids_ptr,
    num_valid_ids_ptr,
    blockscale_e8m0_sorted_ptr,
    stride_blockscale_e8m0_m: tl.int64,
    stride_blockscale_e8m0_n: tl.int64,
    stride_o3: tl.int64,
    stride_o2: tl.int64,
    stride_o1: tl.int64,
    stride_o0: tl.int64,
    token_num,
    N_i,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    TOPK: tl.constexpr,
):
    pid_m = tl.program_id(0) * 2
    pid_n = tl.program_id(1) * 2
    num_valid_ids = tl.load(num_valid_ids_ptr)
    if pid_m * BLOCK_SIZE_M >= num_valid_ids:
        return

    out = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.uint32)
    for m_idx in range(2):
        m = m_idx * BLOCK_SIZE_M
        sorted_ids_offs_m = pid_m * BLOCK_SIZE_M + m + tl.arange(0, BLOCK_SIZE_M)
        sorted_ids_mask = sorted_ids_offs_m < num_valid_ids
        raw_ids = tl.load(
            sorted_ids_ptr + sorted_ids_offs_m, mask=sorted_ids_mask, other=token_num
        )
        token_ids = raw_ids & 0xFFFFFF
        if TOPK == 1:
            blockscale_e8m0_offs_m = token_ids
        else:
            blockscale_e8m0_offs_m = token_ids * TOPK + (raw_ids >> 24)
        row_addrs = blockscale_e8m0_offs_m[:, None] * stride_blockscale_e8m0_m
        row_mask = (token_ids < token_num)[:, None]

        for n_idx in range(2):
            i = m_idx + n_idx * 2
            col_offs = (
                pid_n * BLOCK_SIZE_N + n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            )
            gather_offs = row_addrs + col_offs[None, :] * stride_blockscale_e8m0_n
            col_mask = (col_offs < N_i)[None, :]
            sub = tl.load(
                blockscale_e8m0_ptr + gather_offs,
                mask=row_mask & col_mask,
            ).to(tl.uint8, bitcast=True)
            out = out | (sub.to(tl.uint32) << (i * 8))

    offs_0 = tl.arange(0, BLOCK_SIZE_M)
    offs_1 = tl.arange(0, BLOCK_SIZE_N)
    offs = (
        offs_0[:, None] * stride_o0
        + offs_1[None, :] * stride_o1
        + pid_n // 2 * stride_o2
        + pid_m // 2 * stride_o3
    )
    tl.store(blockscale_e8m0_sorted_ptr + offs, out)


@triton.jit
def _moe_mxfp4_sort_kernel_fused_n(
    blockscale_e8m0_ptr,
    sorted_ids_ptr,
    num_valid_ids_ptr,
    blockscale_e8m0_sorted_ptr,
    stride_blockscale_e8m0_m: tl.int64,
    stride_blockscale_e8m0_n: tl.int64,
    stride_o3: tl.int64,
    stride_o2: tl.int64,
    stride_o1: tl.int64,
    stride_o0: tl.int64,
    token_num,
    N_i,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    TOPK: tl.constexpr,
    N_TILES: tl.constexpr,
):
    pid_m = tl.program_id(0) * 2
    num_valid_ids = tl.load(num_valid_ids_ptr)
    if pid_m * BLOCK_SIZE_M >= num_valid_ids:
        return

    offs_m0 = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    raw_0 = tl.load(
        sorted_ids_ptr + offs_m0, mask=offs_m0 < num_valid_ids, other=token_num
    )
    tid_0 = raw_0 & 0xFFFFFF
    if TOPK == 1:
        ridx_0 = tid_0
    else:
        ridx_0 = tid_0 * TOPK + (raw_0 >> 24)
    raddr_0 = ridx_0[:, None] * stride_blockscale_e8m0_m
    rmask_0 = (tid_0 < token_num)[:, None]

    offs_m1 = offs_m0 + BLOCK_SIZE_M
    raw_1 = tl.load(
        sorted_ids_ptr + offs_m1, mask=offs_m1 < num_valid_ids, other=token_num
    )
    tid_1 = raw_1 & 0xFFFFFF
    if TOPK == 1:
        ridx_1 = tid_1
    else:
        ridx_1 = tid_1 * TOPK + (raw_1 >> 24)
    raddr_1 = ridx_1[:, None] * stride_blockscale_e8m0_m
    rmask_1 = (tid_1 < token_num)[:, None]

    offs_row = tl.arange(0, BLOCK_SIZE_M)
    offs_col = tl.arange(0, BLOCK_SIZE_N)
    store_base = pid_m // 2 * stride_o3

    for n_tile in range(N_TILES):
        pid_n = n_tile * 2
        out = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.uint32)

        for m_idx in range(2):
            if m_idx == 0:
                cur_raddr = raddr_0
                cur_rmask = rmask_0
            else:
                cur_raddr = raddr_1
                cur_rmask = rmask_1

            for n_idx in range(2):
                i = m_idx + n_idx * 2
                col_offs = (
                    pid_n * BLOCK_SIZE_N
                    + n_idx * BLOCK_SIZE_N
                    + tl.arange(0, BLOCK_SIZE_N)
                )
                gather_offs = cur_raddr + col_offs[None, :] * stride_blockscale_e8m0_n
                col_mask = (col_offs < N_i)[None, :]
                sub = tl.load(
                    blockscale_e8m0_ptr + gather_offs,
                    mask=cur_rmask & col_mask,
                ).to(tl.uint8, bitcast=True)
                out = out | (sub.to(tl.uint32) << (i * 8))

        store_offs = (
            offs_row[:, None] * stride_o0
            + offs_col[None, :] * stride_o1
            + n_tile * stride_o2
            + store_base
        )
        tl.store(blockscale_e8m0_sorted_ptr + store_offs, out)


def moe_mxfp4_sort(
    blockscale_e8m0: torch.Tensor,
    sorted_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    token_num: int,
    block_size: int = 32,
) -> torch.Tensor:
    """Sort the blockscale_e8m0 tensor based on the sorted_ids tensor."""
    BLOCK_SIZE_M, BLOCK_SIZE_N = 32, 8
    BLOCK_SIZE_M_u32, BLOCK_SIZE_N_u32 = 16, 4

    topk = 1
    if len(blockscale_e8m0.shape) == 3:
        topk = blockscale_e8m0.shape[1]
        blockscale_e8m0 = blockscale_e8m0.view(-1, blockscale_e8m0.shape[-1])
    M_i, N_i = blockscale_e8m0.shape
    M_o, N_o = sorted_ids.shape[0], N_i
    assert (N_i // 2) % 2 == 0
    assert block_size % BLOCK_SIZE_M == 0

    blockscale_e8m0_sorted = torch.empty(
        (
            triton.cdiv(M_o, BLOCK_SIZE_M),
            triton.cdiv(N_o, BLOCK_SIZE_N),
            BLOCK_SIZE_N_u32,
            BLOCK_SIZE_M_u32,
        ),
        dtype=torch.uint32,
        device=blockscale_e8m0.device,
    )

    _FUSED_N_THRESHOLD = 2048

    common_args = (
        blockscale_e8m0.view(torch.uint8),
        sorted_ids,
        num_valid_ids,
        blockscale_e8m0_sorted,
        *blockscale_e8m0.stride(),
        *blockscale_e8m0_sorted.stride(),
    )
    common_kwargs = dict(
        token_num=token_num,
        N_i=N_i,
        BLOCK_SIZE_M=BLOCK_SIZE_M // 2,
        BLOCK_SIZE_N=BLOCK_SIZE_N // 2,
        TOPK=topk,
    )

    if token_num > _FUSED_N_THRESHOLD:
        N_TILES = triton.cdiv(N_i, BLOCK_SIZE_N)
        grid = (triton.cdiv(M_o, BLOCK_SIZE_M),)
        _moe_mxfp4_sort_kernel_fused_n[grid](
            *common_args, **common_kwargs, N_TILES=N_TILES
        )
    else:
        grid = (triton.cdiv(M_o, BLOCK_SIZE_M), triton.cdiv(N_i, BLOCK_SIZE_N))
        _moe_mxfp4_sort_kernel[grid](*common_args, **common_kwargs)

    return blockscale_e8m0_sorted.view(_fp8_e8m0).view(-1, N_o)


# ============================================================================
# Reference / triton wrappers — match op_tests semantics
# ============================================================================


def run_fused_dynamic_mxfp4_quant_moe_sort_ref(
    x,
    sorted_ids,
    token_num,
    topk,
    q_dtype_a,
    num_local_tokens,
    num_valid_ids,
    block_size_M,
):
    # Reference quant: pure-torch MXFP4 (mirrors per_1x32_f4_quant_hip semantics)
    x_fp4, x_scales_not_sorted = _torch_dynamic_mxfp4_quant(x)
    x_scales = moe_mxfp4_sort(
        x_scales_not_sorted[: token_num * topk, :].view(token_num, topk, -1),
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=token_num,
        block_size=block_size_M,
    )
    return x_fp4, x_scales, x_scales_not_sorted


def run_fused_dynamic_mxfp4_quant_moe_sort_triton(
    x,
    sorted_ids,
    token_num,
    topk,
    q_dtype_a,
    num_local_tokens,
    num_valid_ids,
    block_size_M,
):
    x_fp4, x_scales = fused_dynamic_mxfp4_quant_moe_sort(
        x,
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=token_num,
        topk=topk,
        block_size=block_size_M,
    )
    return x_fp4, x_scales


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WARMUP = 50
ITERATIONS = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200"))

# ---------------------------------------------------------------------------
# Build the ordered full case stream (matches pytest parametrize order)
# pytest decorators (top-to-bottom): hidden_dim, token_num, (tns,nvi), topk, dtype
# pytest iterates outermost = last decorator (dtype), innermost = first (hidden_dim)
# So: dtype (outer) x topk x (token_num_sort,num_valid_ids_0) x token_num x hidden_dim (inner)
# ---------------------------------------------------------------------------
_dtypes = [torch.bfloat16]
_topks = [1, 8]
_token_num_sort_valid = [(1, 1), (32, 32), (1024, 1024), (1024, 512)]
_token_nums = [1, 32, 1024]
_hidden_dims = [256]

ALL_CONFIGS_RAW = list(
    itertools.product(
        _dtypes,
        _topks,
        _token_num_sort_valid,
        _token_nums,
        _hidden_dims,
    )
)
ALL_CONFIGS = [
    (hd, tn, tns_nvi, topk, dtype)
    for dtype, topk, tns_nvi, tn, hd in ALL_CONFIGS_RAW
]


def _pick(configs, count):
    if len(configs) <= count:
        return list(range(len(configs)))
    n = len(configs)
    return [round(i * (n - 1) / (count - 1)) for i in range(count)]


def _make_inputs(cfg):
    """Build inputs for a single config, returns dict of tensors + metadata."""
    hidden_dim, token_num, (token_num_sort, num_valid_ids_0), topk, dtype = cfg
    block_size_M = 128
    q_dtype_a = _fp4x2

    torch.manual_seed(42)

    num_valid_ids = torch.zeros(2, dtype=torch.int64, device="cuda")
    num_valid_ids[0] = num_valid_ids_0
    num_valid_ids[1] = token_num

    topk_ids = torch.randint(0, max(topk, 1), (token_num_sort,), device="cuda")
    topk_ids, _ = torch.sort(topk_ids)
    sorted_ids = torch.randint(0, token_num, (token_num_sort,), device="cuda")
    sorted_ids = (topk_ids << 24) | sorted_ids

    x = torch.randn((token_num, topk, hidden_dim), dtype=dtype, device="cuda") / 20
    x = x.view(-1, hidden_dim)

    return dict(
        x=x,
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=token_num,
        topk=topk,
        block_size_M=block_size_M,
        q_dtype_a=q_dtype_a,
        hidden_dim=hidden_dim,
        token_num_sort=token_num_sort,
        num_valid_ids_0=num_valid_ids_0,
        dtype=dtype,
    )


def _cfg_label(cfg):
    hidden_dim, token_num, (token_num_sort, num_valid_ids_0), topk, dtype = cfg
    return (
        f"hidden_dim={hidden_dim} token_num={token_num} "
        f"token_num_sort={token_num_sort} num_valid_ids_0={num_valid_ids_0} "
        f"topk={topk}"
    )


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------
def run_correctness(indices):
    print(f"Running correctness on {len(indices)} configs...")
    all_pass = True
    for idx in indices:
        cfg = ALL_CONFIGS[idx]
        label = _cfg_label(cfg)
        inp = _make_inputs(cfg)

        try:
            x_fp4_ref, x_scales_ref, x_scales_ref_not_sorted = (
                run_fused_dynamic_mxfp4_quant_moe_sort_ref(
                    inp["x"],
                    inp["sorted_ids"],
                    inp["token_num"],
                    inp["topk"],
                    inp["q_dtype_a"],
                    None,
                    inp["num_valid_ids"],
                    inp["block_size_M"],
                )
            )

            x_fp4_triton, x_scales_triton = run_fused_dynamic_mxfp4_quant_moe_sort_triton(
                inp["x"],
                inp["sorted_ids"],
                inp["token_num"],
                inp["topk"],
                inp["q_dtype_a"],
                None,
                inp["num_valid_ids"],
                inp["block_size_M"],
            )

            tol = 0.1
            nvi = inp["num_valid_ids"][0].item()
            x_scales_ref_c = x_scales_ref[:nvi]
            x_scales_triton_c = x_scales_triton[:nvi]
            torch.testing.assert_close(
                x_scales_ref_c.view(torch.uint8),
                x_scales_triton_c.view(torch.uint8),
                atol=tol,
                rtol=tol,
            )

            # dequant round-trip — use triton scales for apples-to-apples
            _, x_scales_ref_triton_ns = dynamic_mxfp4_quant(inp["x"])
            x_scales_ref_triton_ns = x_scales_ref_triton_ns[
                : x_scales_ref_not_sorted.shape[0],
                : x_scales_ref_not_sorted.shape[1],
            ]
            x_ref = convert_mxfp4_to_fp32(
                x_fp4_ref.view(torch.uint8),
                x_scales_ref_not_sorted.view(torch.uint8),
            )
            x_triton = convert_mxfp4_to_fp32(
                x_fp4_triton.view(torch.uint8),
                x_scales_ref_triton_ns.view(torch.uint8),
            )
            torch.testing.assert_close(x_ref, x_triton, atol=tol, rtol=tol)

            print(f"  [{idx}] PASS  {label}")
        except Exception as e:
            print(f"  [{idx}] FAIL  {label}: {e}")
            all_pass = False

    return all_pass


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def run_benchmark(indices):
    print(f"Running benchmark on {len(indices)} configs...")
    latencies = []
    methods = []
    for idx in indices:
        cfg = ALL_CONFIGS[idx]
        label = _cfg_label(cfg)
        inp = _make_inputs(cfg)

        def fn():
            fused_dynamic_mxfp4_quant_moe_sort(
                inp["x"],
                sorted_ids=inp["sorted_ids"],
                num_valid_ids=inp["num_valid_ids"],
                token_num=inp["token_num"],
                topk=inp["topk"],
                block_size=inp["block_size_M"],
            )

        ms, metadata = benchmark_cuda_graph_or_events(
            fn, warmup=WARMUP, repetition=ITERATIONS
        )
        latencies.append(ms)
        methods.append(metadata["benchmark_method"])
        print(f"  [{idx}] {label}  {ms:.4f}ms")

    log_sum = sum(math.log(max(lat, 1e-12)) for lat in latencies)
    geo_mean = math.exp(log_sum / len(latencies))

    print(f"GEAK_SHAPES_USED={indices}")
    print(f"GEAK_RESULT_LATENCY_MS={geo_mean:.6f}")
    method = methods[0] if len(set(methods)) == 1 else "mixed:" + ",".join(sorted(set(methods)))
    print(f"GEAK_BENCHMARK_METHOD={method}")
    return geo_mean


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------
def run_profile(indices):
    print(f"Running profile on {len(indices)} configs...")
    for idx in indices:
        cfg = ALL_CONFIGS[idx]
        label = _cfg_label(cfg)
        inp = _make_inputs(cfg)

        for _ in range(3):
            fused_dynamic_mxfp4_quant_moe_sort(
                inp["x"],
                sorted_ids=inp["sorted_ids"],
                num_valid_ids=inp["num_valid_ids"],
                token_num=inp["token_num"],
                topk=inp["topk"],
                block_size=inp["block_size_M"],
            )
        torch.cuda.synchronize()

        fused_dynamic_mxfp4_quant_moe_sort(
            inp["x"],
            sorted_ids=inp["sorted_ids"],
            num_valid_ids=inp["num_valid_ids"],
            token_num=inp["token_num"],
            topk=inp["topk"],
            block_size=inp["block_size_M"],
        )
        torch.cuda.synchronize()
        print(f"  [{idx}] profiled  {label}")

    print(f"GEAK_SHAPES_USED={indices}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--correctness", action="store_true")
    group.add_argument("--benchmark", action="store_true")
    group.add_argument("--full-benchmark", action="store_true")
    group.add_argument("--profile", action="store_true")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Number of benchmark iterations (overrides GEAK_BENCHMARK_ITERATIONS env var)")
    args = parser.parse_args()
    if args.iterations is not None:
        global ITERATIONS
        ITERATIONS = args.iterations

    if not _is_fp4_avail():
        print("MXFP4 not supported on this architecture")
        sys.exit(1)

    all_indices = list(range(len(ALL_CONFIGS)))

    if args.correctness:
        indices = list(range(len(ALL_CONFIGS)))
        ok = run_correctness(indices)
        print(f"GEAK_SHAPES_USED={indices}")
        if not ok:
            sys.exit(1)

    elif args.benchmark:
        run_benchmark(all_indices)

    elif args.full_benchmark:
        run_benchmark(all_indices)

    elif args.profile:
        indices = _pick(ALL_CONFIGS, 5)
        run_profile(indices)


if __name__ == "__main__":
    main()
