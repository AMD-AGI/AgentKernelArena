# SPDX-License-Identifier: BSD-3-Clause
"""Elementwise tanh-approximate GELU in Triton.

Standalone rewrite of the pointwise kernel PyTorch Inductor generates for
torch.nn.functional.gelu(x, approximate="tanh") on bfloat16 input: flat 1-D
grid, float32 arithmetic, one rounding to bfloat16 on store.
"""
import triton
import triton.language as tl


@triton.jit
def _gelu_tanh_kernel(x_ptr, out_ptr, numel, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    tanh = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
    y = 0.5 * x * (1.0 + tanh)
    tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def run(*, x, out):
    numel = x.numel()
    block = 2048
    _gelu_tanh_kernel[(triton.cdiv(numel, block),)](x, out, numel, BLOCK=block, num_warps=4)
