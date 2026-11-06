# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused SiLU-and-mul Triton kernel and its launch dispatch from AITER.

Source: aiter/ops/triton/activation.py (`fused_silu_mul`) and
aiter/ops/triton/_triton_kernels/activation.py (`fused_silu_mul_kernel`, `_silu_exp2`).

Op: the first d lanes of each row go through SiLU (exp2 form) in fp32, are truncated to the
input dtype, and are multiplied by the second d lanes; the row of 2*d lanes yields d outputs.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _silu_exp2(x):
    return x / (1.0 + tl.exp2(-(x * 1.44269504089)))


@triton.jit
def fused_silu_mul_kernel(
    inp_ptr,
    out_ptr,
    n_rows,
    n_cols,
    row_stride_in,
    col_stride_in,
    row_stride_out,
    col_stride_out,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    SiLU on the first half of the last dimension, multiply by the second half.
    Each row has 2 * n_cols input elements; writes n_cols outputs.
    2D grid: axis 0 tiles rows (BLOCK_M), axis 1 tiles columns (BLOCK_N).
    """
    m_pid = tl.program_id(0)
    n_pid = tl.program_id(1)
    m_offs = tl.arange(0, BLOCK_M)
    n_offs = tl.arange(0, BLOCK_N)
    row_idx = m_pid * BLOCK_M + m_offs
    col_idx = n_pid * BLOCK_N + n_offs

    row_in = row_idx * row_stride_in
    row_out = row_idx * row_stride_out

    first_half_ptrs = inp_ptr + row_in[:, None] + col_idx[None, :] * col_stride_in
    second_half_ptrs = (
        inp_ptr + row_in[:, None] + (n_cols + col_idx)[None, :] * col_stride_in
    )
    out_ptrs = out_ptr + row_out[:, None] + col_idx[None, :] * col_stride_out

    mask = (row_idx < n_rows)[:, None] & (col_idx < n_cols)[None, :]
    a = tl.load(first_half_ptrs, mask=mask, other=0.0).to(tl.float32)
    silu_a = _silu_exp2(a).to(inp_ptr.dtype.element_ty)
    b = tl.load(second_half_ptrs, mask=mask, other=0.0)
    o = (silu_a * b).to(out_ptr.dtype.element_ty)
    tl.store(out_ptrs, o, mask=mask)


def _pick_block_n(d: int, n_rows: int) -> int:
    """Tile size along the reduced last dim (cap 1024); at least 32 for vectorization."""
    n = max(d, 1)
    # Kimi-K2.5 TP4 (d=512): prefill favors one 512-wide N tile; decode keeps 256x2.
    if n == 512:
        return 512 if n_rows > 4096 else 256
    # GLM-4.7 TP4 (d=384): wider decode rows use 256x2; larger batches favor 128x3 N tiles.
    if n == 384:
        return 256 if n_rows <= 128 else 128
    upper = min(n, 1024)
    p = 1
    while p * 2 <= upper:
        p *= 2
    return max(32, p)


def _pick_block_m(n_rows: int, block_n: int, d: int) -> int:
    """Row tile size: latency shapes use wide M tiles; prefill uses tuned (d, n_rows) pairs."""
    if n_rows <= 64:
        return min(32, max(4, triton.next_power_of_2(n_rows)))
    if d == 384 and n_rows > 128:
        return 32 if n_rows > 8192 else 8
    if d == 512 and n_rows > 4096:
        return 8
    if d == 512 and 128 < n_rows <= 4096:
        return 8
    if block_n >= 1024:
        return 8
    if block_n >= 512:
        return 8
    return 16


def _pick_num_warps(n_rows: int, block_m: int, block_n: int) -> int:
    """ROCm: 8 warps for tiny full-wavefront decode tiles; 2 warps for larger tiles."""
    if n_rows <= 128 and block_m >= 16 and block_n >= 128:
        return 8
    return 2


def run(*, x, out):
    n_rows, last = x.shape
    d = last // 2

    block_n = _pick_block_n(d, n_rows)
    block_m = _pick_block_m(n_rows, block_n, d)
    grid = (triton.cdiv(n_rows, block_m), triton.cdiv(d, block_n))
    num_warps = _pick_num_warps(n_rows, block_m, block_n)

    fused_silu_mul_kernel[grid](
        x,
        out,
        n_rows,
        d,
        2 * d,
        1,
        d,
        1,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
        waves_per_eu=0,
    )
