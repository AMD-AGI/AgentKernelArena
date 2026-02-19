"""Apply staged write kernel from vLLM v1 worker/gpu/buffer_utils.py."""
import torch
import triton
import triton.language as tl


@triton.jit
def _apply_write_kernel(
    output_ptr,
    output_stride,
    write_indices_ptr,
    write_starts_ptr,
    write_contents_ptr,
    write_cu_lens_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_idx = tl.load(write_indices_ptr + pid)
    start_idx = tl.load(write_starts_ptr + pid)

    cu_start = tl.load(write_cu_lens_ptr + pid - 1) if pid > 0 else 0
    cu_end = tl.load(write_cu_lens_ptr + pid)
    content_len = cu_end - cu_start

    for i in range(0, content_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < content_len
        content = tl.load(write_contents_ptr + cu_start + block, mask=mask)
        tl.store(
            output_ptr + row_idx * output_stride + start_idx + block, content, mask=mask
        )


def apply_write(
    output: torch.Tensor,
    write_indices: torch.Tensor,
    write_starts: torch.Tensor,
    write_contents: torch.Tensor,
    write_cu_lens: torch.Tensor,
) -> None:
    """Apply staged writes to a 2D output tensor.

    Each write operation copies a segment of write_contents into a row of output.
    write_indices[i] specifies the row, write_starts[i] specifies the column offset,
    and write_cu_lens defines cumulative lengths to index into write_contents.

    Args:
        output: [num_rows, row_size] output tensor
        write_indices: [num_writes] row indices
        write_starts: [num_writes] start column offsets
        write_contents: [total_content_len] flat content to write
        write_cu_lens: [num_writes] cumulative lengths into write_contents
    """
    n = write_indices.shape[0]
    if n == 0:
        return
    _apply_write_kernel[(n,)](
        output,
        output.stride(0),
        write_indices,
        write_starts,
        write_contents,
        write_cu_lens,
        BLOCK_SIZE=1024,
    )
