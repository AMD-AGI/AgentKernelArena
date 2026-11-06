# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Variable-length (thd) RoPE forward (Triton), from ROCm/aiter rope_thd_fwd.

The kernel keeps the upstream GPT-J, NoPE and in-place branches (compile-time flags); run() fixes the
scenario's configuration: NeoX rotation, no NoPE, freqs holding one angle per rotated pair.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _get_neox_rotated_x(
    x,
    x_rotated_mask,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
    IS_BWD: tl.constexpr = False,
):
    if IS_BWD:
        x_rotated = tl.where(x_rotated_mask, -x, x)
    else:
        x_rotated = tl.where(x_rotated_mask, x, -x)

    x_rotated = tl.reshape(x_rotated, (BLOCK_T, 2, BLOCK_D_HALF))
    x_rotated = tl.flip(x_rotated, 2)
    x_rotated = tl.reshape(
        x_rotated,
        (
            BLOCK_T,
            BLOCK_D,
        ),
    )
    x_rotated = tl.flip(x_rotated, 1)
    return x_rotated


@triton.jit
def _get_gptj_rotated_x(
    x,
    x_rotated_mask,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
    IS_BWD: tl.constexpr = False,
):
    if IS_BWD:
        x_rotated = tl.where(x_rotated_mask, -x, x)
    else:
        x_rotated = tl.where(x_rotated_mask, x, -x)

    x_rotated = tl.reshape(x_rotated, (BLOCK_T, BLOCK_D_HALF, 2))
    x_rotated = tl.flip(x_rotated, 2)
    x_rotated = tl.reshape(
        x_rotated,
        (
            BLOCK_T,
            BLOCK_D,
        ),
    )
    return x_rotated



@triton.jit
def _rope_kernel_thd_fwd(
    x_ptr,
    cu_seqlens_ptr,
    freqs_ptr,
    out_ptr,
    stride_x_t,
    stride_x_h,
    stride_x_d,
    stride_freqs_t,
    stride_freqs_b,
    stride_freqs_h,
    stride_freqs_d,
    stride_out_t,
    stride_out_h,
    stride_out_d,
    HAVE_NOPE: tl.constexpr,
    NOPE_FIRST: tl.constexpr,
    INPLACE: tl.constexpr,
    REUSE_FREQS_FRONT_PART: tl.constexpr,
    IS_NEOX: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    pid_t = tl.program_id(2)

    t_start = tl.load(cu_seqlens_ptr + b)
    t_end = tl.load(cu_seqlens_ptr + b + 1)
    T = t_end - t_start
    if pid_t * BLOCK_T >= T:
        return

    t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    d_offs = tl.arange(0, BLOCK_D)
    t_mask = t_offs < T

    if REUSE_FREQS_FRONT_PART:
        if IS_NEOX:
            d_freqs_offs = tl.where(
                (d_offs >= BLOCK_D_HALF) & (d_offs < BLOCK_D),
                d_offs - BLOCK_D_HALF,
                d_offs,
            ).to(d_offs.dtype)
            d_freqs_mask = d_freqs_offs < BLOCK_D
        else:
            d_freqs_offs = d_offs // 2
            d_freqs_mask = d_freqs_offs < BLOCK_D_HALF
    else:
        d_freqs_offs = d_offs
        d_freqs_mask = d_freqs_offs < BLOCK_D

    freqs_mask = t_mask[:, None] & d_freqs_mask[None, :]
    freqs_offs = (
        t_offs[:, None] * stride_freqs_t + d_freqs_offs[None, :] * stride_freqs_d
    )
    freqs = tl.load(freqs_ptr + freqs_offs, mask=freqs_mask)
    cos = tl.cos(freqs.to(tl.float32))
    sin = tl.sin(freqs.to(tl.float32))

    nope_offs = 0
    if HAVE_NOPE and NOPE_FIRST:
        nope_offs = BLOCK_D

    x_mask = t_mask[:, None] & (d_offs < BLOCK_D)[None, :]

    if IS_NEOX:
        x_rotated_mask = (d_offs < BLOCK_D_HALF)[None, :]
    else:
        x_rotated_mask = (d_offs % 2 == 0)[None, :]

    d_offs += nope_offs
    x_offs = (
        (t_start + t_offs)[:, None] * stride_x_t
        + h * stride_x_h
        + d_offs[None, :] * stride_x_d
    )
    x = tl.load(x_ptr + x_offs, mask=x_mask)

    if IS_NEOX:
        x_rotated = _get_neox_rotated_x(
            x, x_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF
        )
    else:
        x_rotated = _get_gptj_rotated_x(
            x, x_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF
        )

    out_x = x * cos + x_rotated * sin
    out_x = out_x.to(x_ptr.dtype.element_ty)
    x_out_offs = (
        (t_start + t_offs)[:, None] * stride_out_t
        + h * stride_out_h
        + d_offs[None, :] * stride_out_d
    )

    tl.store(out_ptr + x_out_offs, out_x, mask=x_mask)

    if HAVE_NOPE and not INPLACE:
        if NOPE_FIRST:
            x = tl.load(x_ptr + x_offs - BLOCK_D * stride_x_d, mask=x_mask)
            tl.store(out_ptr + x_out_offs - BLOCK_D * stride_out_d, x, mask=x_mask)
        else:
            x = tl.load(x_ptr + x_offs + BLOCK_D * stride_x_d, mask=x_mask)
            tl.store(out_ptr + x_out_offs + BLOCK_D * stride_out_d, x, mask=x_mask)



def run(*, x, cu_seqlens, freqs, out):
    b = cu_seqlens.shape[0] - 1
    t, h, d = x.shape
    BLOCK_T = 32
    grid = (b, h, triton.cdiv(t, BLOCK_T))
    _rope_kernel_thd_fwd[grid](
        x,
        cu_seqlens,
        freqs,
        out,
        *x.stride(),
        *freqs.stride(),
        *out.stride(),
        HAVE_NOPE=False,
        NOPE_FIRST=False,
        INPLACE=False,
        REUSE_FREQS_FRONT_PART=True,
        IS_NEOX=True,
        BLOCK_T=BLOCK_T,
        BLOCK_D=d,
        BLOCK_D_HALF=d // 2,
        num_warps=4,
        waves_per_eu=0,
    )
