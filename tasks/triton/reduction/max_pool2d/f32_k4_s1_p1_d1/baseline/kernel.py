# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileCopyrightText: Copyright contributors to the PyTorch project
"""Windowed max-pool kernel in Triton, modeled on the PyTorch Inductor pointwise kernel for max_pool2d."""
import triton
import triton.language as tl

KERNEL_SIZE = 4
STRIDE = 1
PADDING = 1
DILATION = 1


@triton.jit
def _max_pool2d_kernel(in_ptr, out_ptr, n_out, height, width, out_height, out_width,
                       KERNEL: tl.constexpr, STRIDE: tl.constexpr, PADDING: tl.constexpr,
                       DILATION: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0).to(tl.int64) * BLOCK
    index = offset + tl.arange(0, BLOCK)
    valid = index < n_out
    ow = index % out_width
    oh = (index // out_width) % out_height
    plane = index // (out_width * out_height)
    base = plane * height * width
    acc = tl.full([BLOCK], float("-inf"), tl.float32)
    for p in tl.static_range(KERNEL):
        ih = oh * STRIDE - PADDING + p * DILATION
        row_ok = valid & (ih >= 0) & (ih < height)
        for q in tl.static_range(KERNEL):
            iw = ow * STRIDE - PADDING + q * DILATION
            ok = row_ok & (iw >= 0) & (iw < width)
            v = tl.load(in_ptr + base + ih * width + iw, mask=ok, other=float("-inf"))
            acc = tl.maximum(acc, v)
    tl.store(out_ptr + index, acc, mask=valid)


def run(*, x, out):
    batch, channels, height, width = x.shape
    out_height, out_width = out.shape[2], out.shape[3]
    n_out = batch * channels * out_height * out_width
    block = 1024
    grid = ((n_out + block - 1) // block,)
    _max_pool2d_kernel[grid](x, out, n_out, height, width, out_height, out_width,
                             KERNEL_SIZE, STRIDE, PADDING, DILATION, BLOCK=block)
