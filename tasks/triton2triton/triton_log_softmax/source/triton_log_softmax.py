"""Log softmax kernel using Triton, adapted from vLLM batch_invariant.py."""
import torch
import triton
import triton.language as tl


@triton.jit
def _log_softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute log_softmax along the last dimension of a 2D tensor.
    Each block handles one row of the input tensor.
    """
    row_idx = tl.program_id(0).to(tl.int64)

    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride

    # Step 1: Find maximum value in the row for numerical stability
    max_val = -float("inf")
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=-float("inf"))
        max_val = tl.max(tl.maximum(vals, max_val))

    # Step 2: Compute sum of exp(x - max_val)
    sum_exp = 0.0
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(tl.where(mask, exp_vals, 0.0))

    log_sum_exp = tl.log(sum_exp)

    # Step 3: Compute final log_softmax values: x - max_val - log_sum_exp
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals = tl.load(row_start_ptr + col_idx, mask=mask)
        output = vals - max_val - log_sum_exp
        tl.store(output_row_start_ptr + col_idx, output, mask=mask)


def log_softmax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute log_softmax using Triton kernel along the last dimension."""
    if dim != -1 and dim != input.ndim - 1:
        raise ValueError(
            "This implementation only supports log_softmax along the last dimension"
        )

    original_shape = input.shape
    input_2d = input.reshape(-1, input.shape[-1])
    input_2d = input_2d.contiguous()

    n_rows, n_cols = input_2d.shape
    output = torch.empty_like(input_2d)

    BLOCK_SIZE = 1024
    grid = (n_rows,)
    _log_softmax_kernel[grid](
        input_2d,
        output,
        input_2d.stride(0),
        output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output.reshape(original_shape)
