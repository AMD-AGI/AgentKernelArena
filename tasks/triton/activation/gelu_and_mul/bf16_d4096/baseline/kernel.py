# Copyright (C) Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Derived from ROCm/aiter aiter/ops/triton/_triton_kernels/activation.py
# (_gelu, _gelu_tanh, fused_silu_mul_kernel) and aiter/ops/triton/activation.py
# (fused_silu_mul launch heuristics) at commit 652f2d4b768239aefd515c929e84817681ff598d.
"""Fused gated GELU (act(x[:, :d]) * x[:, d:]) in Triton, adapted from AITER."""
import triton
import triton.language as tl

ACTIVATION = "gelu"


@triton.jit
def _tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _gelu(x):
    M_SQRT1_2 = 0.70710678118654752440
    ALPHA = M_SQRT1_2
    return 0.5 * x * (1.0 + tl.erf(x * ALPHA))


@triton.jit
def _gelu_tanh(x):
    M_SQRT2 = 1.41421356237309504880
    M_2_SQRTPI = 1.12837916709551257390
    BETA = M_SQRT2 * M_2_SQRTPI * 0.5
    KAPPA = 0.044715
    x_cube = x * x * x
    inner = BETA * (x + KAPPA * x_cube)
    return 0.5 * x * (1.0 + _tanh(inner))


@triton.jit
def _act_mul_kernel(
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
    ACT: tl.constexpr,
):
    # 2D grid: axis 0 tiles rows (BLOCK_M), axis 1 tiles columns (BLOCK_N).
    m_pid = tl.program_id(0)
    n_pid = tl.program_id(1)
    row_idx = m_pid * BLOCK_M + tl.arange(0, BLOCK_M)
    col_idx = n_pid * BLOCK_N + tl.arange(0, BLOCK_N)

    row_in = row_idx.to(tl.int64) * row_stride_in
    row_out = row_idx.to(tl.int64) * row_stride_out

    first_half_ptrs = inp_ptr + row_in[:, None] + col_idx[None, :] * col_stride_in
    second_half_ptrs = inp_ptr + row_in[:, None] + (n_cols + col_idx)[None, :] * col_stride_in
    out_ptrs = out_ptr + row_out[:, None] + col_idx[None, :] * col_stride_out

    mask = (row_idx < n_rows)[:, None] & (col_idx < n_cols)[None, :]
    a = tl.load(first_half_ptrs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(second_half_ptrs, mask=mask, other=0.0).to(tl.float32)
    if ACT == "gelu":
        act = _gelu(a)
    else:
        act = _gelu_tanh(a)
    tl.store(out_ptrs, (act * b).to(out_ptr.dtype.element_ty), mask=mask)


def _pick_block_n(d):
    upper = min(max(d, 1), 1024)
    p = 1
    while p * 2 <= upper:
        p *= 2
    return max(32, p)


def _pick_block_m(n_rows, block_n):
    if n_rows <= 64:
        return min(32, max(4, triton.next_power_of_2(n_rows)))
    if block_n >= 512:
        return 8
    return 16


def _pick_num_warps(n_rows, block_m, block_n):
    if n_rows <= 128 and block_m >= 16 and block_n >= 128:
        return 8
    return 2


def run(*, x, out):
    n_rows, last = x.shape
    d = last // 2
    block_n = _pick_block_n(d)
    block_m = _pick_block_m(n_rows, block_n)
    num_warps = _pick_num_warps(n_rows, block_m, block_n)
    grid = (triton.cdiv(n_rows, block_m), triton.cdiv(d, block_n))
    _act_mul_kernel[grid](
        x, out, n_rows, d,
        x.stride(0), x.stride(1), out.stride(0), out.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, ACT=ACTIVATION,
        num_warps=num_warps, waves_per_eu=0,
    )
