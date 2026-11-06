# SPDX-License-Identifier: Apache-2.0
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


def run(*, output, write_indices, write_starts, write_contents, write_cu_lens):
    """Apply staged writes to a 2D output tensor.

    write_indices[i] is the row, write_starts[i] the column offset and write_cu_lens the
    cumulative segment lengths into write_contents.
    """
    _apply_write_kernel[(write_indices.shape[0],)](
        output,
        output.stride(0),
        write_indices,
        write_starts,
        write_contents,
        write_cu_lens,
        BLOCK_SIZE=1024,
    )
