# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Activation and top-k reduction of the fused-MoE layer in Triton.

vLLM runs ``gelu_tanh_and_mul`` and ``moe_sum`` as CUDA ops; these are Triton equivalents written for this task:
gelu_tanh(gate) * up with the gate computed in float32 and rounded to bfloat16 before the multiply, and the sum over
the top-k expert outputs accumulated in float32.
"""
import triton
import triton.language as tl


@triton.jit
def _gelu_tanh_and_mul_kernel(x_ptr, out_ptr, d, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < d
    gate = tl.load(x_ptr + row * 2 * d + offs, mask=mask, other=0.0)
    up = tl.load(x_ptr + row * 2 * d + d + offs, mask=mask, other=0.0)
    g = gate.to(tl.float32)
    inner = 0.7978845608028654 * (g + 0.044715 * g * g * g)
    # tanh(u) = 1 - 2 / (exp(2u) + 1), saturating correctly for large |u|
    tanh = 1.0 - 2.0 / (tl.exp(2.0 * inner) + 1.0)
    act = (0.5 * g * (1.0 + tanh)).to(gate.dtype)
    tl.store(out_ptr + row * d + offs, (act * up).to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _moe_sum_kernel(x_ptr, out_ptr, hidden, TOP_K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < hidden
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for k in range(TOP_K):
        acc += tl.load(x_ptr + (row * TOP_K + k) * hidden + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * hidden + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


def gelu_tanh_and_mul(x, out):
    rows, two_d = x.shape
    d = two_d // 2
    block = 1024
    _gelu_tanh_and_mul_kernel[(rows, triton.cdiv(d, block))](x, out, d, BLOCK=block)


def moe_sum(x, out):
    tokens, top_k, hidden = x.shape
    block = 1024
    _moe_sum_kernel[(tokens, triton.cdiv(hidden, block))](x, out, hidden, TOP_K=top_k, BLOCK=block)
