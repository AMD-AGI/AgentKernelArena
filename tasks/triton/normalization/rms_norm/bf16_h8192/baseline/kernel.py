# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RMS normalization in Triton, adapted from vLLM batch_invariant.py."""
import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(input_ptr, weight_ptr, output_ptr, input_row_stride, output_row_stride, n_cols, eps,
                     BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0).to(tl.int64)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    sum_sq = tl.zeros([1], dtype=tl.float32)
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(tl.where(mask, vals * vals, 0.0))
    inv_rms = 1.0 / tl.sqrt(sum_sq / n_cols + eps)
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
        weight = tl.load(weight_ptr + col_idx, mask=mask, other=1.0)
        output = vals.to(tl.float32) * inv_rms * weight.to(tl.float32)
        tl.store(output_row_start_ptr + col_idx, output.to(vals.dtype), mask=mask)


def run(*, x, weight, epsilon, out):
    n_rows, n_cols = x.shape
    _rms_norm_kernel[(n_rows,)](x, weight, out, x.stride(0), out.stride(0), n_cols, epsilon, BLOCK_SIZE=1024)
