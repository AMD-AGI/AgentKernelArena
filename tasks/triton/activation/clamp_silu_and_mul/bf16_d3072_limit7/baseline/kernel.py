# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Clamped SwiGLU activation in Triton, adapted from AITER ``fused_clamp_act_mul``.

``_fused_clamp_silu_mul_kernel`` is AITER's device kernel with the weight-multiply
and activation-selection variants removed (this scenario uses SiLU and no weights).
Input rows hold the gate in the first N columns and the up projection in the last N:

    gate = min(gate, limit); up = clamp(up, -limit, limit); out = silu(gate) * up

Computation is float32; the result is rounded to the input dtype.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _silu(x):
    return x / (1.0 + tl.exp2(-(x * 1.44269504089)))


@triton.jit
def _fused_clamp_silu_mul_kernel(
    inp_ptr,
    out_ptr,
    n_half,
    inp_stride_m,
    inp_stride_n,
    out_stride_m,
    out_stride_n,
    swiglu_limit,
    BLOCK_SIZE_N: tl.constexpr,
    HAVE_SWIGLU_CLAMP: tl.constexpr,
):
    m_pid = tl.program_id(0).to(tl.int64)
    n_offs = tl.arange(0, BLOCK_SIZE_N)
    mask = n_offs < n_half

    gate = tl.load(
        inp_ptr + m_pid * inp_stride_m + n_offs * inp_stride_n,
        mask=mask,
        other=0.0,
        cache_modifier=".cg",
    ).to(tl.float32)
    up = tl.load(
        inp_ptr + m_pid * inp_stride_m + (n_half + n_offs) * inp_stride_n,
        mask=mask,
        other=0.0,
        cache_modifier=".cg",
    ).to(tl.float32)

    if HAVE_SWIGLU_CLAMP:
        up = tl.clamp(up, -swiglu_limit, swiglu_limit)
        gate = tl.minimum(gate, swiglu_limit)

    out = _silu(gate) * up

    tl.store(
        out_ptr + m_pid * out_stride_m + n_offs * out_stride_n,
        out.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def run(*, x, swiglu_limit, out):
    M, D = x.shape
    n_half = D // 2
    assert n_half % 128 == 0

    BLOCK_SIZE_N = triton.next_power_of_2(n_half)
    if BLOCK_SIZE_N <= 512:
        num_warps = 1
    elif BLOCK_SIZE_N <= 2048:
        num_warps = 4
    else:
        num_warps = 8

    _fused_clamp_silu_mul_kernel[(M,)](
        x,
        out,
        n_half,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        swiglu_limit,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        HAVE_SWIGLU_CLAMP=swiglu_limit > 0,
        num_warps=num_warps,
    )
