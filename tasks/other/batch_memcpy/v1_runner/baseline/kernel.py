# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batched byte copy in Triton, adapted from vLLM v1 mamba_utils.batch_memcpy_kernel.

One program instance copies one (source, destination, size) triple, 1024 bytes at a time.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def batch_memcpy_kernel(src_pool, src_offsets, dst_pool, dst_offsets, sizes, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    src_ptr = src_pool + tl.load(src_offsets + pid)
    dst_ptr = dst_pool + tl.load(dst_offsets + pid)
    size = tl.load(sizes + pid)

    offsets = tl.arange(0, BLOCK_SIZE)
    for i in range(0, size, BLOCK_SIZE):
        mask = (i + offsets) < size

        data = tl.load(src_ptr + i + offsets, mask=mask)
        tl.store(dst_ptr + i + offsets, data, mask=mask)


def run(*, src_pool, src_offsets, sizes, dst_offsets, dst_pool):
    batch = src_offsets.shape[0]
    batch_memcpy_kernel[(batch,)](src_pool, src_offsets, dst_pool, dst_offsets, sizes, BLOCK_SIZE=1024)
