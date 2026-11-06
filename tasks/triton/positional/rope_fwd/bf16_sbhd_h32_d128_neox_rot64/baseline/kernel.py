# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""RoPE forward (sbhd) Triton kernel from ROCm/aiter.

Provenance: aiter.ops.triton.rope.rope (`_rope_fwd`) and its device kernel
`_rope_kernel_sbhd_fwd` plus the rotate helpers `_get_neox_rotated_x` /
`_get_gptj_rotated_x` (aiter.ops.triton._triton_kernels.rope.rope), inlined into one file.
The kernel keeps its GPT-J and NoPE branches (compile-time flags); run() fixes the scenario's
configuration: NeoX rotation of the first half of the head (rotary_pct 0.5), NoPE slice after it.

Op (rotary position embedding, sbhd layout):
    x is [S, B, H, D]; freqs is [S, 1, 1, freqs_D]. For each (s,b,h) the rotary
    half of the head dim is rotated by cos/sin(freqs) (NEOX or GPTJ rotate_style),
    with `reuse_freqs_front_part` (cos/sin reused across the two halves) and an
    optional NOPE (non-rotary) split kept verbatim (`nope_first` controls which
    side carries the pass-through). cos/sin are computed in fp32; output is the
    input dtype.
"""

from enum import IntEnum

import torch
import triton
import triton.language as tl


class RotateStyle(IntEnum):
    NEOX = (0,)
    GPTJ = 1


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
def _rope_kernel_sbhd_fwd(
    x_ptr,
    freqs_ptr,
    out_ptr,
    stride_x_s,
    stride_x_b,
    stride_x_h,
    stride_x_d,
    stride_freqs_s,
    stride_freqs_b,
    stride_freqs_h,
    stride_freqs_d,
    stride_out_s,
    stride_out_b,
    stride_out_h,
    stride_out_d,
    S,
    HAVE_NOPE: tl.constexpr,
    NOPE_FIRST: tl.constexpr,
    INPLACE: tl.constexpr,
    REUSE_FREQS_FRONT_PART: tl.constexpr,
    IS_NEOX: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
):
    # Parallelize over batch and head. Handle 1 sequence per program
    b = tl.program_id(0)
    h = tl.program_id(1)
    pid_s = tl.program_id(2)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    d_offs = tl.arange(0, BLOCK_D)
    s_mask = s_offs < S

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

    freqs_mask = s_mask[:, None] & d_freqs_mask[None, :]
    freqs_offs = (
        s_offs[:, None] * stride_freqs_s + d_freqs_offs[None, :] * stride_freqs_d
    )

    freqs = tl.load(freqs_ptr + freqs_offs, mask=freqs_mask)
    cos = tl.cos(freqs.to(tl.float32))
    sin = tl.sin(freqs.to(tl.float32))

    nope_offs = 0
    if HAVE_NOPE and NOPE_FIRST:
        nope_offs = BLOCK_D

    x_offs = (
        b * stride_x_b
        + s_offs[:, None] * stride_x_s
        + h * stride_x_h
        + (d_offs + nope_offs)[None, :] * stride_x_d
    )
    x_mask = s_mask[:, None] & (d_offs < BLOCK_D)[None, :]
    x = tl.load(x_ptr + x_offs, mask=x_mask)

    if IS_NEOX:
        x_rotated_mask = (d_offs < BLOCK_D_HALF)[None, :]
        x_rotated = _get_neox_rotated_x(
            x, x_rotated_mask, BLOCK_S, BLOCK_D, BLOCK_D_HALF
        )
    else:
        x_rotated_mask = (d_offs % 2 == 0)[None, :]
        x_rotated = _get_gptj_rotated_x(
            x, x_rotated_mask, BLOCK_S, BLOCK_D, BLOCK_D_HALF
        )

    out_x = x * cos + x_rotated * sin
    out_x = out_x.to(x_ptr.dtype.element_ty)
    x_out_offs = (
        b * stride_out_b
        + s_offs[:, None] * stride_out_s
        + h * stride_out_h
        + (d_offs + nope_offs)[None, :] * stride_out_d
    )

    tl.store(out_ptr + x_out_offs, out_x, mask=x_mask)

    if HAVE_NOPE and not INPLACE:
        if NOPE_FIRST:
            x = tl.load(x_ptr + x_offs - BLOCK_D * stride_x_d, mask=x_mask)
            tl.store(out_ptr + x_out_offs - BLOCK_D * stride_out_d, x, mask=x_mask)
        else:
            x = tl.load(x_ptr + x_offs + BLOCK_D * stride_x_d, mask=x_mask)
            tl.store(out_ptr + x_out_offs + BLOCK_D * stride_out_d, x, mask=x_mask)


# TODO: For now BLOCK_D is assumed to be power of 2. Expand to handle other value of D.
def _rope_fwd(
    x: torch.Tensor,
    out: torch.Tensor,
    freqs: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part: bool,
    nope_first: bool,
    inplace: bool,
    transpose_output: bool = False,
) -> torch.Tensor:
    s, b, h, d = x.shape

    if freqs.shape[-1] == d // 2:
        if reuse_freqs_front_part:
            have_nope = False
        else:
            have_nope = True
    elif freqs.shape[-1] == d // 4:
        have_nope = True
    else:
        have_nope = False

    if have_nope:
        BLOCK_D = d // 2
        BLOCK_D_HALF = d // 4
    else:
        BLOCK_D = d
        BLOCK_D_HALF = d // 2

    # TODO: performance optimization
    BLOCK_S = 32
    num_warps = 4
    waves_per_eu = 0
    grid = (b, h, triton.cdiv(s, BLOCK_S))

    _rope_kernel_sbhd_fwd[grid](
        x,
        freqs,
        out,
        *x.stride(),
        *freqs.stride(),
        *out.stride(),
        s,
        HAVE_NOPE=have_nope,
        NOPE_FIRST=nope_first,
        INPLACE=inplace,
        REUSE_FREQS_FRONT_PART=reuse_freqs_front_part,
        IS_NEOX=(rotate_style == RotateStyle.NEOX),
        BLOCK_S=BLOCK_S,
        BLOCK_D=BLOCK_D,
        BLOCK_D_HALF=BLOCK_D_HALF,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )

    return out


def run(*, x, freqs, out):
    """NeoX rotation of the first D/2 channels with freqs[s, 0, 0, :D/4] shared by both halves of that slice;
    the remaining D/2 channels (NoPE, after the rotary slice) are copied unchanged."""
    _rope_fwd(x, out, freqs, RotateStyle.NEOX, True, False, False)
