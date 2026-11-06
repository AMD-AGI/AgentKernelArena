# SPDX-License-Identifier: Apache-2.0
# Derived from SGLang (sgl-project/sglang), licensed under the Apache License, Version 2.0.
"""Fused dual-residual RMSNorm in Triton, from SGLang.

Kernel and launch configuration: python/sglang/kernels/ops/elementwise/elementwise.py
(fused_dual_residual_rmsnorm_kernel, fused_dual_residual_rmsnorm with autotune=False, _is_hip=True).
"""
import triton
import triton.language as tl

_is_hip = True


@triton.jit
def fused_dual_residual_rmsnorm_kernel(
    output_ptr,
    mid_ptr,
    activ_ptr,
    residual_ptr,
    weight1_ptr,
    weight2_ptr,
    eps: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    input_start = pid * hidden_dim

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim

    a_ = tl.load(activ_ptr + input_start + offsets, mask=mask, other=0.0)
    a = a_.to(tl.float32)
    rms = tl.sqrt(tl.sum(a * a, axis=0) / hidden_dim + eps)

    r = tl.load(residual_ptr + input_start + offsets, mask=mask, other=0.0)
    w1_ = tl.load(weight1_ptr + offsets, mask=mask, other=0.0)
    w1 = w1_.to(tl.float32)

    a2r = r + (a / rms * w1).to(r.dtype)
    tl.store(
        mid_ptr + input_start + offsets,
        a2r,
        mask=mask,
    )

    a2r = a2r.to(tl.float32)
    rms2 = tl.sqrt(tl.sum(a2r * a2r, axis=0) / hidden_dim + eps)

    w2_ = tl.load(weight2_ptr + offsets, mask=mask, other=0.0)
    w2 = w2_.to(tl.float32)

    tl.store(
        output_ptr + input_start + offsets,
        a2r / rms2 * w2,  # implicitly casts to output dtype here
        mask=mask,
    )


def run(*, x, residual, weight1, weight2, epsilon, out, mid):
    bs, hidden_dim = x.shape
    max_warps = 16 if _is_hip else 32
    fused_dual_residual_rmsnorm_kernel[(bs,)](
        out,
        mid,
        x,
        residual,
        weight1,
        weight2,
        eps=epsilon,
        hidden_dim=hidden_dim,
        BLOCK_SIZE=triton.next_power_of_2(hidden_dim),
        num_warps=max(min(triton.next_power_of_2(triton.cdiv(hidden_dim, 256)), max_warps), 4),
    )
