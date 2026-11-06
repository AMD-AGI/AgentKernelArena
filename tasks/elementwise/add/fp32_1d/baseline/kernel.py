# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
"""Element-wise addition of two 1D float32 tensors (ROCm Triton benchmark add_kernel)."""
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x_block_ptr = tl.make_block_ptr(base=x_ptr, shape=(n_elements, ), strides=(1, ), offsets=(pid * BLOCK_SIZE, ),
                                    block_shape=(BLOCK_SIZE, ), order=(0, ))
    x = tl.load(x_block_ptr, boundary_check=(0, ), padding_option='zero')

    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def run(*, x, y, out):
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, 1024), )
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)
