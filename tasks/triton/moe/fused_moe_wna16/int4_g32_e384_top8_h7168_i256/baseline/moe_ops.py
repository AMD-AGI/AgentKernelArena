# SPDX-License-Identifier: Apache-2.0
"""Triton versions of two CUDA ops vLLM's fused_experts_impl calls between and after the GEMMs.

silu_and_mul follows vLLM's act_and_mul kernel (csrc/libtorch_stable/activation_kernels.cu): silu(x) in
float32 rounded to the activation dtype, times y in float32, rounded again. moe_sum follows vLLM's moe_sum
kernels (csrc/libtorch_stable/moe/moe_align_sum_kernels.cu): the top-k expert outputs of each token are added
in order in float32 and rounded once.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _silu_and_mul_kernel(x_ptr, out_ptr, d, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < d
    gate = tl.load(x_ptr + row * 2 * d + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_ptr + row * 2 * d + d + cols, mask=mask, other=0.0)
    act = (gate / (1.0 + tl.exp(-gate))).to(up.dtype)
    tl.store(out_ptr + row * d + cols, (act.to(tl.float32) * up.to(tl.float32)).to(up.dtype), mask=mask)


@triton.jit
def _moe_sum_kernel(x_ptr, out_ptr, d, TOPK: tl.constexpr, BLOCK: tl.constexpr):
    token = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < d
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(TOPK):
        acc += tl.load(x_ptr + (token * TOPK + k) * d + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + token * d + cols, acc.to(out_ptr.dtype.element_ty), mask=mask)


def silu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    """out[r] = silu(x[r, :d]) * x[r, d:] for x of shape [rows, 2 * d]."""
    rows, d = out.shape
    block = min(triton.next_power_of_2(d), 1024)
    _silu_and_mul_kernel[(rows, triton.cdiv(d, block))](x, out, d, BLOCK=block)


def moe_sum(x: torch.Tensor, out: torch.Tensor) -> None:
    """out[t] = sum over k of x[t, k] for x of shape [tokens, top_k, d]."""
    tokens, top_k, d = x.shape
    block = 1024
    _moe_sum_kernel[(tokens, triton.cdiv(d, block))](x, out, d, TOPK=top_k, BLOCK=block)
