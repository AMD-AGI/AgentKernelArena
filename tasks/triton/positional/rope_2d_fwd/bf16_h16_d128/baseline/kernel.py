# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""2-D (height, width) NeoX RoPE forward (Triton), from ROCm/aiter rope_fwd_2d.

One program rotates a whole image (all height * width tokens) of one head: the first half of the
head dimension is rotated with the per-row (height) cos/sin table and the second half with the
per-column (width) table. run() fixes the scenario's configuration (NeoX style, tables holding
head_dim // 2 entries per position, no NoPE part) and writes into the preallocated output.
"""

import triton
import triton.language as tl


@triton.jit
def _rope_fwd_2d_kernel_neox(
    x_ptr,
    cos_h_ptr,
    sin_h_ptr,
    cos_w_ptr,
    sin_w_ptr,
    out_ptr,
    stride_x_b,
    stride_x_wh,
    stride_x_h,
    stride_x_d,
    stride_cos_h_b,
    stride_cos_h_ht,
    stride_cos_h_h,
    stride_cos_h_d,
    stride_cos_w_b,
    stride_cos_w_w,
    stride_cos_w_h,
    stride_cos_w_d,
    WH: tl.constexpr,
    HEIGHT: tl.constexpr,
    WEIGHT: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)

    # load cos_h [HT, BLOCK_D]
    offs_wh = tl.arange(0, WH)
    offs_cos_h_h = offs_wh // WEIGHT
    offs_d = tl.arange(0, BLOCK_D)
    offs_cos_h = (
        stride_cos_h_h * offs_cos_h_h[:, None] + stride_cos_h_d * offs_d[None, :]
    )
    mask_cos_h = offs_d < BLOCK_D // 2
    cos_h = tl.load(cos_h_ptr + offs_cos_h, mask=mask_cos_h[None, :], other=0.0)

    # load sin_h
    sin_h = tl.load(sin_h_ptr + offs_cos_h, mask=mask_cos_h[None, :], other=0.0)

    # load cos_w
    offs_cos_w_w = offs_wh % WEIGHT
    offs_cos_w_d = offs_d - BLOCK_D // 2
    offs_cos_w = (
        stride_cos_w_w * offs_cos_w_w[:, None] + stride_cos_w_d * offs_cos_w_d[None, :]
    )
    mask_cos_w = (offs_cos_w_d >= 0) & (offs_cos_w_d < BLOCK_D // 2)
    cos_w = tl.load(cos_w_ptr + offs_cos_w, mask=mask_cos_w[None, :], other=0.0)

    # load sin_w
    sin_w = tl.load(sin_w_ptr + offs_cos_w, mask=mask_cos_w[None, :], other=0.0)

    # load x
    offs_wh = tl.arange(0, WH)
    offs_x = (
        stride_x_b * b
        + stride_x_wh * offs_wh[:, None]
        + stride_x_h * h
        + stride_x_d * offs_d[None, :]
    )
    x = tl.load(x_ptr + offs_x)

    # load x_rotated
    offs_wh = tl.arange(0, WH)
    offs_d_rotated = tl.where(offs_d < BLOCK_D // 4, offs_d + BLOCK_D // 4, offs_d)
    offs_d_rotated = tl.where(
        (offs_d >= BLOCK_D // 4) & (offs_d < BLOCK_D // 2),
        offs_d_rotated - BLOCK_D // 4,
        offs_d_rotated,
    )
    offs_d_rotated = tl.where(
        (offs_d >= BLOCK_D // 2) & (offs_d < 3 * BLOCK_D // 4),
        offs_d_rotated + BLOCK_D // 4,
        offs_d_rotated,
    )
    offs_d_rotated = tl.where(
        (offs_d >= 3 * BLOCK_D // 4) & (offs_d < BLOCK_D),
        offs_d_rotated - BLOCK_D // 4,
        offs_d_rotated,
    )
    offs_x_rotated = (
        stride_x_b * b
        + stride_x_wh * offs_wh[:, None]
        + stride_x_h * h
        + stride_x_d * offs_d_rotated[None, :]
    )
    x_rotated = tl.load(x_ptr + offs_x_rotated)
    neg_x_rotated = tl.where((offs_d >= BLOCK_D // 4) & (offs_d < BLOCK_D // 2), 1, 0)
    neg_x_rotated = tl.where(
        (offs_d >= 3 * BLOCK_D // 4) & (offs_d < BLOCK_D), 1, neg_x_rotated
    )
    x_rotated = tl.where(neg_x_rotated, x_rotated, -x_rotated)

    # compute x1
    x1 = x * cos_h + x_rotated * sin_h

    # compute x2
    x2 = x * cos_w + x_rotated * sin_w

    # compute output
    out = x1 + x2

    # store output
    tl.store(out_ptr + offs_x, out)


def run(*, x, cos_h, sin_h, cos_w, sin_w, out):
    b, wh, h, d = x.shape
    grid = (b, h, 1)
    _rope_fwd_2d_kernel_neox[grid](
        x,
        cos_h,
        sin_h,
        cos_w,
        sin_w,
        out,
        *x.stride(),
        *cos_h.stride(),
        *cos_w.stride(),
        wh,
        cos_h.shape[1],
        cos_w.shape[1],
        BLOCK_D=d,
    )
