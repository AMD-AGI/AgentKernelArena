# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""GPT-OSS clamped SwiGLU-and-mul in Triton.

A Triton rewrite of AITER's ``swiglu_and_mul`` (csrc/kernels/activation_kernels.cu,
``swiglu_act_and_mul_kernel``): one program per (row, column block); gate and up are read in
bfloat16, the activation is evaluated in float32 and the result is rounded to bfloat16.
"""
import triton
import triton.language as tl

ALPHA = 1.702
LIMIT = 7.0


@triton.jit
def _swigluoai_and_mul_kernel(
    out_ptr,
    out_stride,
    x_ptr,
    x_stride,
    d,
    alpha: tl.constexpr,
    limit: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0).to(tl.int64)
    offsets = tl.program_id(axis=1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < d
    x_row = x_ptr + x_stride * row

    gate = tl.load(x_row + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_row + offsets + d, mask=mask, other=0.0).to(tl.float32)

    gate = tl.minimum(gate, limit)
    up = tl.minimum(tl.maximum(up, -limit), limit)
    sigmoid = 1.0 / (1.0 + tl.exp(-alpha * gate))
    result = gate * sigmoid * (up + 1.0)
    tl.store(out_ptr + out_stride * row + offsets, result.to(out_ptr.dtype.element_ty), mask=mask)


def run(*, x, out):
    rows, two_d = x.shape
    d = two_d // 2
    block = 1024
    _swigluoai_and_mul_kernel[(rows, triton.cdiv(d, block))](
        out, out.stride(0), x, x.stride(0), d, alpha=ALPHA, limit=LIMIT, BLOCK_SIZE=block)
