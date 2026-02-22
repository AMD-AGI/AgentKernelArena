"""SwiGLU-step-and-mul activation kernel using Triton, adapted from vLLM activation.py."""
import torch
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


def swiglustep_and_mul(
    input: torch.Tensor, limit: float = 7.0
) -> torch.Tensor:
    """Wrapper for swiglustep_and_mul kernel.

    Computes silu(x[:,:d]).clamp(max=limit) * x[:,d:].clamp(-limit, limit)
    where d = input.shape[-1] // 2.
    """
    assert input.ndim == 2, "Input must be 2D"
    b, n = input.shape
    assert n % 2 == 0, "Last dimension must be even"
    d = n // 2

    output = torch.empty((b, d), dtype=input.dtype, device=input.device)

    def grid(meta):
        return (b, triton.cdiv(d, meta["BLOCK_SIZE"]))

    _swiglustep_and_mul_kernel[grid](
        output,
        output.stride(0),
        input,
        input.stride(0),
        limit=limit,
        d=d,
        BLOCK_SIZE=1024,
    )

    return output
