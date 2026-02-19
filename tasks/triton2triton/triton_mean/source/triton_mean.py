"""Mean reduction kernel using Triton, adapted from vLLM batch_invariant.py."""
import torch
import triton
import triton.language as tl


@triton.jit
def mean_kernel(
    input_ptr,
    output_ptr,
    input_stride0,
    input_stride1,
    input_stride2,
    output_stride0,
    output_stride1,
    M,
    N,
    K,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel for computing mean along a single dimension.
    Input is viewed as (M, N, K) where N is the dimension being reduced.
    """
    pid = tl.program_id(0)

    m_idx = pid // K
    k_idx = pid % K

    if m_idx >= M or k_idx >= K:
        return

    acc = 0.0
    for n_start in range(0, N, BLOCK_SIZE):
        n_offsets = n_start + tl.arange(0, BLOCK_SIZE)
        mask = n_offsets < N
        input_idx = (
            m_idx * input_stride0 + n_offsets * input_stride1 + k_idx * input_stride2
        )
        vals = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
        acc += tl.sum(vals)

    mean_val = acc / N
    output_idx = m_idx * output_stride0 + k_idx * output_stride1
    tl.store(output_ptr + output_idx, mean_val)


def mean_dim(
    input: torch.Tensor,
    dim: int,
    keepdim: bool = False,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Triton implementation of torch.mean with single dimension reduction."""
    assert -input.ndim <= dim < input.ndim, (
        f"Invalid dimension {dim} for tensor with {input.ndim} dimensions"
    )

    if dim < 0:
        dim = dim + input.ndim

    if dtype is None:
        if input.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            dtype = torch.float32
        else:
            dtype = input.dtype

    if input.dtype != dtype:
        input = input.to(dtype)

    shape = list(input.shape)

    M = 1
    for i in range(dim):
        M *= shape[i]

    N = shape[dim]

    K = 1
    for i in range(dim + 1, len(shape)):
        K *= shape[i]

    input_3d = input.reshape(M, N, K)

    if keepdim:
        output_shape = shape.copy()
        output_shape[dim] = 1
    else:
        output_shape = shape[:dim] + shape[dim + 1:]

    output = torch.empty(output_shape, dtype=dtype, device=input.device)

    output_2d = output.reshape(M, 1, K).squeeze(1) if keepdim else output.reshape(M, K)

    grid = (M * K,)
    BLOCK_SIZE = 1024

    mean_kernel[grid](
        input_3d,
        output_2d,
        input_3d.stride(0),
        input_3d.stride(1),
        input_3d.stride(2),
        output_2d.stride(0),
        output_2d.stride(1) if output_2d.ndim > 1 else 0,
        M,
        N,
        K,
        BLOCK_SIZE,
    )

    return output
