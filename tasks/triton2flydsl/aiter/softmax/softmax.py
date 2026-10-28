# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Standalone row-wise online-softmax Triton kernel.

Provenance: ported verbatim from aiter.ops.triton.softmax (`softmax` /
`_softmax_kernel_online`); the constexpr-aware kernel-naming helper is inlined so
the module depends only on `triton` + `torch`.

Op:
    y[i, :] = softmax(x[i, :])  over the last dimension of a 2D tensor.
The kernel uses a one-pass online (running-max + rescaled-sum) reduction so a
single program handles a full row regardless of n_cols (block-strided loop).
"""

import torch
import triton
import triton.language as tl


# --- inlined constexpr-aware kernel naming (utils._triton.kernel_repr) ---
def _sanitize_constexpr_value(value):
    if value is None:
        return "NONE"
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else str(value)
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")
    return cleaned.upper() if cleaned else "NONE"


def make_kernel_repr(base_name, config_keys):
    def _repr(specialization):
        constants = specialization.constants
        parts = [
            f"{key}_{_sanitize_constexpr_value(constants.get(key, None))}"
            for key in config_keys
        ]
        return f"{base_name}_{'_'.join(parts)}" if parts else base_name

    return _repr


_softmax_kernel_online_repr = make_kernel_repr(
    "_softmax_kernel_online",
    [
        "BLOCK_SIZE",
    ],
)


@triton.jit(repr=_softmax_kernel_online_repr)
def _softmax_kernel_online(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):

    row_start = tl.program_id(0)
    row_idx = row_start

    # loop 1, find max and sum
    m = -float("inf")  # Initial value of max
    row_sum = 0.0
    row_start_ptr = input_ptr + row_idx * input_row_stride
    for b in tl.range(0, n_cols, BLOCK_SIZE):
        col_offsets = b + tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row_block = tl.load(
            input_ptrs, mask=mask, other=-float("inf"), cache_modifier=".cg"
        )  # load block
        m_p = tl.max(row_block, axis=0)  # find block max
        m_p = tl.maximum(m, m_p)  # Find new max across all blocks so far
        row_sum = row_sum * tl.exp(m - m_p)  # Adjust previous sum
        row_sum += tl.sum(
            tl.exp(row_block - m_p)
        )  # Add to exponentiated sum of this block
        m = m_p  # save max

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    # Loop 2
    for b in tl.range(0, n_cols, BLOCK_SIZE):
        col_offsets = b + tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row_block = tl.load(
            input_ptrs, mask=mask, other=-float("inf"), cache_modifier=".cg"
        )  # load block
        # subtract, exponentiate and divide by sum
        softmax_output = tl.exp(row_block - m) / row_sum
        # store
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


def softmax(x):
    """
    Computes row-wise softmax of a 2D input tensor.

    Args:
        x (torch.Tensor): Input tensor with shape (n_rows, n_cols). Must be on GPU.

    Returns:
        torch.Tensor: Output with same shape as x, softmax applied along last dimension.
    """
    # The kernel accepts a row stride; pack non-unit column strides explicitly.
    # Existing contiguous scored inputs take the unchanged no-copy path.
    if x.stride(1) != 1:
        x = x.contiguous()
    n_rows, n_cols = x.shape

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols))
    y = torch.empty_like(x)

    waves_per_eu = 2
    num_warps = 8
    num_stages = 2

    num_programs = n_rows

    grid = lambda meta: (num_programs,)  # noqa: E731
    _softmax_kernel_online[grid](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_cols,
        BLOCK_SIZE,
        waves_per_eu=waves_per_eu,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return y
