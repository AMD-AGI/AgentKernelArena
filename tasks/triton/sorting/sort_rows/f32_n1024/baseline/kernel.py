# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
"""Row-wise ascending sort with tl.sort, adapted from the ROCm triton test_sort kernel."""
import torch
import triton
import triton.language as tl


@triton.jit
def sort_kernel(X, Z, N: tl.constexpr, M: tl.constexpr, descending: tl.constexpr):
    # One program sorts a tile of N rows of M elements along the last axis.
    pid = tl.program_id(0)
    base = pid * (N * M)
    offx = tl.arange(0, M)
    offy = tl.arange(0, N) * M
    off2d = offx[None, :] + offy[:, None]
    x = tl.load(X + base + off2d)
    x = tl.sort(x, descending=descending)
    tl.store(Z + base + off2d, x)


def run(*, x, out):
    n_rows, n_cols = x.shape
    sort_kernel[(n_rows,)](x, out, 1, n_cols, False)
