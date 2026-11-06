# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
"""Row-maximum kernel in Triton: one program reduces a block of rows, each loaded as a whole row block."""
import triton
import triton.language as tl

BLOCK_M = 128


@triton.jit
def load_reduce_kernel(
    x_ptr,
    y_ptr,
    n_rows,
    stride_xm,
    stride_xn,
    stride_y,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row0 = tl.program_id(0).to(tl.int64) * BLOCK_M
    x_block = tl.make_block_ptr(base=x_ptr + row0 * stride_xm, shape=(n_rows - row0, BLOCK_N),
                                strides=(stride_xm, stride_xn), offsets=(0, 0),
                                block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
    x = tl.load(x_block, boundary_check=(0,))
    y = tl.max(x, axis=1)
    rows = row0 + tl.arange(0, BLOCK_M)
    tl.store(y_ptr + rows * stride_y, y, mask=rows < n_rows)


def run(*, x, out):
    n_rows, n_cols = x.shape
    grid = ((n_rows + BLOCK_M - 1) // BLOCK_M,)
    load_reduce_kernel[grid](x, out, n_rows, x.stride(0), x.stride(1), out.stride(0),
                             BLOCK_M=BLOCK_M, BLOCK_N=n_cols)
