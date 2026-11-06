# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SwiGLU-step-and-mul in Triton, vendored from vLLM's layers/activation.py."""
import triton
import triton.language as tl


@triton.jit
def _swiglustep_and_mul_kernel(
    o_ptr,
    o_stride,
    x_ptr,
    x_stride,
    limit: tl.constexpr,
    d: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    i = tl.program_id(axis=0).to(tl.int64)
    j = tl.program_id(axis=1)
    o_row_ptr = o_ptr + o_stride * i
    x_row_ptr = x_ptr + x_stride * i
    offsets = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < d

    gate = tl.load(x_row_ptr + offsets, mask=mask).to(tl.float32)
    up = tl.load(x_row_ptr + offsets + d, mask=mask).to(tl.float32)

    gate_silu = tl.sigmoid(gate) * gate
    gate_clamped = tl.minimum(gate_silu, limit)
    up_clamped = tl.minimum(tl.maximum(up, -limit), limit)

    result = gate_clamped * up_clamped
    result = result.to(x_ptr.dtype.element_ty)
    tl.store(o_row_ptr + offsets, result, mask=mask)


def run(*, x, limit, out):
    b, n = x.shape
    d = n // 2

    def grid(meta):
        return (b, triton.cdiv(d, meta["BLOCK_SIZE"]))

    _swiglustep_and_mul_kernel[grid](
        out,
        out.stride(0),
        x,
        x.stride(0),
        limit=limit,
        d=d,
        BLOCK_SIZE=1024,
    )
