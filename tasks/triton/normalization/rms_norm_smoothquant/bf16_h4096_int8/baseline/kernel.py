# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""RMSNorm with SmoothQuant and per-token INT8 quantization in Triton (persistent program per row)."""
import functools

import torch
import triton
import triton.language as tl


@triton.jit
def _per_token_quant(
    x,
    y_scale_ptr,
    row_max,
    row_idx,
    DTYPE_MAX: tl.constexpr,
):
    scale_out = row_max / DTYPE_MAX
    scale_out = tl.where(scale_out == 0, 1.0, scale_out)

    scale_recip = 1 / scale_out

    qx = x * scale_recip

    tl.store(y_scale_ptr + row_idx, scale_out.to(y_scale_ptr.dtype.element_ty))

    return qx


@triton.jit
def _quant_rms_norm_kernel(
    # Pointers to matrices
    input_ptr,
    output_ptr,
    x_scale_ptr,
    y_scale_ptr,
    g_ptr,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `input_row_stride` is
    # how much to increase `input_ptr` by to get the element one row down.
    input_row_stride,
    output_row_stride,
    # Matrix dimensions
    n_rows,
    n_cols,
    # Epsilon to avoid division by zero
    epsilon,
    # Dtype max for quantization
    DTYPE_MAX: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
    NUM_PRGMS: tl.constexpr,
):
    # Map the program id to the first row of input and output it should compute.
    row_start = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    mask = col_offsets < n_cols
    for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=2):
        input_ptrs = input_ptr + row_idx * input_row_stride + col_offsets
        input_ptrs = tl.multiple_of(input_ptrs, (16,))
        row = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(
            tl.float32
        )
        g = tl.load(g_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        row_norm = row * row
        row_norm = tl.sum(row_norm, axis=-1)
        norm_factor = tl.math.rsqrt((row_norm / n_cols) + epsilon)

        rms_norm = row * norm_factor * g

        x_scale_ptrs = x_scale_ptr + col_offsets
        x_scale_ptrs = tl.multiple_of(x_scale_ptrs, (16,))
        x_scale = tl.load(x_scale_ptrs, mask=mask, other=0.0, cache_modifier=".cg")
        rms_norm *= x_scale

        row_max = tl.max(tl.abs(rms_norm), axis=-1)
        rms_norm = _per_token_quant(
            rms_norm,
            y_scale_ptr,
            row_max,
            row_idx,
            DTYPE_MAX,
        )

        # Round to nearest (halves away from zero): the conversion below truncates.
        rms_norm = rms_norm + tl.where(rms_norm >= 0, 0.5, -0.5)

        output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
        output_ptrs = tl.multiple_of(output_ptrs, (16,))
        tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty), mask=mask)


@functools.lru_cache(maxsize=None)
def _num_sms(device_index):
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def run(*, x, weight, x_scale, epsilon, out, y_scale):
    n_rows, n_cols = x.shape

    blk_size = triton.next_power_of_2(n_cols)
    num_prgms = min(n_rows, _num_sms(x.device.index))

    grid = lambda meta: (num_prgms,)  # noqa: E731
    _quant_rms_norm_kernel[grid](
        x,
        out,
        x_scale,
        y_scale,
        weight,
        x.stride(0),
        out.stride(0),
        n_rows,
        n_cols,
        epsilon,
        127,  # INT8 maximum
        blk_size,
        num_prgms,
    )
