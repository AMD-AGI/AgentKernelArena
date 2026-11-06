# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
"""Reverse every row of a [N, 64] float32 tensor (ROCm Triton benchmark flip_kernel, grid over row tiles)."""
import torch
import triton
import triton.language as tl


@triton.jit
def flip_kernel(X, Z, n_rows, M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    cols = tl.arange(0, M)
    destination = rows[:, None] * M + cols[None, :]
    valid = (rows[:, None] < n_rows) & (cols[None, :] < M)
    values = tl.load(X + destination, mask=valid, other=0)
    values = tl.flip(values, 1)
    tl.store(Z + destination, values, mask=valid)


def run(*, x, out):
    n_rows, m = x.shape
    block_n = 64
    grid = (triton.cdiv(n_rows, block_n), )
    flip_kernel[grid](x, out, n_rows, M=m, BLOCK_N=block_n)
