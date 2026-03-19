# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Fused QKV Split + QK RoPE Kernel Implementation

Based on aiter's fused_qkv_split_qk_rope implementation (ROCm/aiter):
- Fuses QKV tensor splitting with rotary position embedding application
- Supports both NeoX and GPT-J rotation styles
- Supports nope (no-position-embedding) dimensions
- Reduces memory bandwidth by avoiding intermediate tensors

All Triton kernel code and reference implementations are inlined
for self-contained execution without aiter dependency.
"""

from __future__ import annotations
import math
from enum import IntEnum
from typing import Tuple

import torch
import triton
import triton.language as tl


# ============================================================================
# INLINED: aiter/ops/triton/_triton_kernels/rope/rope.py (subset)
# ============================================================================


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


# ============================================================================
# INLINED: aiter/ops/triton/_triton_kernels/rope/fused_qkv_split_qk_rope.py
# ============================================================================


@triton.jit
def _fused_qkv_split_qk_rope_kernel(
    qkv_ptr,
    cos_ptr,
    sin_ptr,
    pos_ptr,
    off_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    T,
    stride_qkv_t,
    stride_qkv_d,
    stride_cos_t,
    stride_cos_d,
    stride_pos_t,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_kv_t,
    stride_kv_h,
    stride_kv_d,
    HAVE_NOPE: tl.constexpr,
    NOPE_FIRST: tl.constexpr,
    REUSE_FREQS_FRONT_PART: tl.constexpr,
    IS_NEOX: tl.constexpr,
    HAVE_POS: tl.constexpr,
    HAVE_OFFS: tl.constexpr,
    QH: tl.constexpr,
    KVH: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
):
    tl.assume(stride_qkv_t > 0)
    tl.assume(stride_qkv_d > 0)
    tl.assume(stride_cos_t > 0)
    tl.assume(stride_cos_d > 0)
    tl.assume(stride_pos_t > 0)
    tl.assume(stride_q_t > 0)
    tl.assume(stride_q_h > 0)
    tl.assume(stride_q_d > 0)
    tl.assume(stride_kv_t > 0)
    tl.assume(stride_kv_h > 0)
    tl.assume(stride_kv_d > 0)

    pid_t = tl.program_id(0)
    hq = tl.program_id(1)

    tl.assume(pid_t >= 0)
    tl.assume(hq >= 0)

    t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    d_offs = tl.arange(0, BLOCK_D)
    t_mask = t_offs < T

    if HAVE_POS:
        pos_offs = t_offs * stride_pos_t
        pos = tl.load(pos_ptr + pos_offs, mask=t_mask)
        if HAVE_OFFS:
            offset = tl.load(off_ptr + pos_offs, mask=t_mask)
            t_cos_offs = pos + offset
        else:
            t_cos_offs = pos
    else:
        t_cos_offs = t_offs

    if REUSE_FREQS_FRONT_PART:
        if IS_NEOX:
            d_cos_offs = d_offs
            d_cos_offs = tl.where(
                (d_cos_offs < BLOCK_D_HALF),
                d_cos_offs,
                d_cos_offs - BLOCK_D_HALF,
            ).to(d_cos_offs.dtype)
            d_cos_mask = d_cos_offs < BLOCK_D_HALF
        else:
            d_cos_offs = tl.arange(0, BLOCK_D) // 2
            d_cos_mask = d_cos_offs < BLOCK_D_HALF
    else:
        d_cos_offs = d_offs
        d_cos_mask = d_cos_offs < BLOCK_D

    cos_mask = t_mask[:, None] & d_cos_mask[None, :]
    cos_offs = t_cos_offs[:, None] * stride_cos_t + d_cos_offs[None, :] * stride_cos_d
    cos = tl.load(cos_ptr + cos_offs, mask=cos_mask)
    sin = tl.load(sin_ptr + cos_offs, mask=cos_mask)

    nope_offs = 0
    if HAVE_NOPE and NOPE_FIRST:
        nope_offs = BLOCK_D

    offs_nope_ratio = 1
    if HAVE_NOPE:
        offs_nope_ratio = 2

    x_mask = t_mask[:, None] & (d_offs < BLOCK_D)[None, :]

    if IS_NEOX:
        qk_rotated_mask = (d_offs < BLOCK_D_HALF)[None, :]
    else:
        qk_rotated_mask = (d_offs % 2 == 0)[None, :]

    H_OFFS_SIZE = hq * BLOCK_D
    d_offs += nope_offs
    q_in_offs = (
        t_offs[:, None] * stride_qkv_t
        + (H_OFFS_SIZE * offs_nope_ratio + d_offs)[None, :] * stride_qkv_d
    )
    q = tl.load(qkv_ptr + q_in_offs, mask=x_mask)

    if IS_NEOX:
        q_rotated = _get_neox_rotated_x(
            q, qk_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF
        )
    else:
        q_rotated = _get_gptj_rotated_x(
            q, qk_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF
        )

    q_out_offs = (
        t_offs[:, None] * stride_q_t + d_offs[None, :] * stride_q_d + hq * stride_q_h
    )
    q = q * cos + q_rotated * sin
    q = q.to(q_ptr.dtype.element_ty)
    tl.store(q_ptr + q_out_offs, q, mask=x_mask)

    if HAVE_NOPE:
        if NOPE_FIRST:
            q = tl.load(qkv_ptr + q_in_offs - BLOCK_D * stride_qkv_d, mask=x_mask)
            tl.store(q_ptr + q_out_offs - BLOCK_D * stride_q_d, q, mask=x_mask)
        else:
            q = tl.load(qkv_ptr + q_in_offs + BLOCK_D * stride_qkv_d, mask=x_mask)
            tl.store(q_ptr + q_out_offs + BLOCK_D * stride_q_d, q, mask=x_mask)

    if hq < KVH:
        Q_SIZE = QH * BLOCK_D
        KV_SIZE = KVH * BLOCK_D
        k_in_offs = (
            t_offs[:, None] * stride_qkv_t
            + ((Q_SIZE + H_OFFS_SIZE) * offs_nope_ratio + d_offs)[None, :]
            * stride_qkv_d
        )
        v_in_offs = (
            t_offs[:, None] * stride_qkv_t
            + ((Q_SIZE + KV_SIZE + H_OFFS_SIZE) * offs_nope_ratio + d_offs)[None, :]
            * stride_qkv_d
        )
        k = tl.load(qkv_ptr + k_in_offs, mask=x_mask)
        v = tl.load(qkv_ptr + v_in_offs, mask=x_mask)

        if IS_NEOX:
            k_rotated = _get_neox_rotated_x(
                k, qk_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF
            )
        else:
            k_rotated = _get_gptj_rotated_x(
                k, qk_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF
            )

        kv_out_offs = (
            t_offs[:, None] * stride_kv_t
            + d_offs[None, :] * stride_kv_d
            + hq * stride_kv_h
        )
        k = k * cos + k_rotated * sin
        k = k.to(k_ptr.dtype.element_ty)
        tl.store(k_ptr + kv_out_offs, k, mask=x_mask)
        v = v.to(v_ptr.dtype.element_ty)
        tl.store(v_ptr + kv_out_offs, v, mask=x_mask)

        if HAVE_NOPE:
            if NOPE_FIRST:
                k = tl.load(qkv_ptr + k_in_offs - BLOCK_D * stride_qkv_d, mask=x_mask)
                tl.store(k_ptr + kv_out_offs - BLOCK_D * stride_kv_d, k, mask=x_mask)
                v = tl.load(qkv_ptr + v_in_offs - BLOCK_D * stride_qkv_d, mask=x_mask)
                tl.store(v_ptr + kv_out_offs - BLOCK_D * stride_kv_d, v, mask=x_mask)
            else:
                k = tl.load(qkv_ptr + k_in_offs + BLOCK_D * stride_qkv_d, mask=x_mask)
                tl.store(k_ptr + kv_out_offs + BLOCK_D * stride_kv_d, k, mask=x_mask)
                v = tl.load(qkv_ptr + v_in_offs + BLOCK_D * stride_qkv_d, mask=x_mask)
                tl.store(v_ptr + kv_out_offs + BLOCK_D * stride_kv_d, v, mask=x_mask)


# ============================================================================
# PYTHON WRAPPER
# ============================================================================


def fused_qkv_split_qk_rope(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    qh: int,
    kvh: int,
    head_dim: int,
    is_neox: bool = True,
    offsets: torch.Tensor = None,
    reuse_freqs_front_part: bool = True,
    nope_first: bool = False,
):
    T = qkv.shape[0]
    q_size = qh * head_dim
    kv_size = kvh * head_dim

    assert qh >= kvh and qh % kvh == 0, "qh must be mutiple of kvh"

    q = torch.empty((qkv.shape[0], qh, head_dim), dtype=qkv.dtype, device=qkv.device)
    k = torch.empty((qkv.shape[0], kvh, head_dim), dtype=qkv.dtype, device=qkv.device)
    v = torch.empty((qkv.shape[0], kvh, head_dim), dtype=qkv.dtype, device=qkv.device)

    if cos.shape[-1] == head_dim // 2:
        if reuse_freqs_front_part:
            have_nope = False
        else:
            have_nope = True
    elif cos.shape[-1] == head_dim // 4:
        have_nope = True
    else:
        have_nope = False

    assert qkv.shape[-1] == q_size + 2 * kv_size, "Shape error"
    assert head_dim // ((2 if have_nope else 1)) == triton.next_power_of_2(
        head_dim // ((2 if have_nope else 1))
    ), "head_dim should be power of 2"

    if have_nope:
        BLOCK_D = head_dim // 2
        BLOCK_D_HALF = head_dim // 4
    else:
        BLOCK_D = head_dim
        BLOCK_D_HALF = head_dim // 2

    BLOCK_T = 32
    num_warps = 4
    waves_per_eu = 0
    grid = (triton.cdiv(T, BLOCK_T), qh, 1)

    _fused_qkv_split_qk_rope_kernel[grid](
        qkv,
        cos,
        sin,
        positions,
        offsets,
        q,
        k,
        v,
        T,
        *qkv.stride(),
        cos.stride(0),
        cos.stride(-1),
        *positions.stride(),
        *q.stride(),
        *k.stride(),
        HAVE_NOPE=have_nope,
        NOPE_FIRST=nope_first,
        REUSE_FREQS_FRONT_PART=reuse_freqs_front_part,
        IS_NEOX=is_neox,
        HAVE_POS=(positions is not None),
        HAVE_OFFS=(offsets is not None),
        QH=qh,
        KVH=kvh,
        BLOCK_T=BLOCK_T,
        BLOCK_D=BLOCK_D,
        BLOCK_D_HALF=BLOCK_D_HALF,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )

    return q, k, v


def triton_op(qkv, cos, sin, positions, qh, kvh, head_dim, is_neox,
              reuse_freqs_front_part, nope_first):
    return fused_qkv_split_qk_rope(
        qkv, cos, sin, positions, qh, kvh, head_dim,
        is_neox=is_neox, offsets=None,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=nope_first,
    )


##################################################################################################################################################

# ============================================================================
# REFERENCE IMPLEMENTATIONS
# ============================================================================


class RotateStyle(IntEnum):
    NEOX = 0
    GPTJ = 1


def rotate_half_neox(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rotate_half_gptj(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def ref_rope_sbhd_fwd(
    x_,
    freqs_,
    rotate_style,
    reuse_freqs_front_part,
    nope_first,
    simulate_cached=False,
    comp_with_fp32=False,
):
    x = x_.to(dtype=torch.float32) if comp_with_fp32 else x_
    freqs = freqs_.to(dtype=torch.float32) if comp_with_fp32 else freqs_
    rotate_half = (
        rotate_half_neox if rotate_style == RotateStyle.NEOX else rotate_half_gptj
    )
    rotate_dim = freqs.shape[-1] * (2 if reuse_freqs_front_part else 1)
    if nope_first:
        d = x.shape[-1]
        x, x_forward = x[..., d - rotate_dim :], x[..., : d - rotate_dim]
    else:
        x, x_forward = x[..., :rotate_dim], x[..., rotate_dim:]
    if reuse_freqs_front_part:
        if rotate_style == RotateStyle.NEOX:
            freqs = freqs.repeat([1] * (freqs.dim() - 1) + [2])
        elif rotate_style == RotateStyle.GPTJ:
            freqs = freqs.repeat_interleave(2, dim=-1)
    cos = (
        torch.cos(freqs).to(dtype=freqs_.dtype).to(dtype=torch.float32)
        if simulate_cached and comp_with_fp32
        else torch.cos(freqs)
    )
    sin = (
        torch.sin(freqs).to(dtype=freqs_.dtype).to(dtype=torch.float32)
        if simulate_cached and comp_with_fp32
        else torch.sin(freqs)
    )
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return (
        torch.cat((x_forward, x_embed.to(dtype=x.dtype)), dim=-1).to(dtype=x_.dtype)
        if nope_first
        else torch.cat((x_embed.to(dtype=x.dtype), x_forward), dim=-1).to(
            dtype=x_.dtype
        )
    )


def generate_rope_cached_freqs(B, max_embed_positions, freqs_D, dtype):
    pos = torch.randint(0, max_embed_positions, (B,), device="cuda")
    freqs = torch.randn(
        (max_embed_positions, 1, 1, freqs_D), dtype=dtype, device="cuda"
    )
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    cos_sin = torch.cat((cos, sin), dim=-1)
    cos, sin = torch.chunk(cos_sin, 2, dim=-1)
    return pos, freqs, cos, sin


def generate_qkv_inputs(
    B, QH_PER_KH, KH, D, nope, nope_first, dtype
):
    qkv = torch.randn(
        (B, (QH_PER_KH * KH + 2 * KH) * (D * (2 if nope else 1))),
        dtype=dtype,
        device="cuda",
    )
    return qkv


def torch_op(
    qkv,
    QH_PER_KH,
    KH,
    D,
    ref_freqs,
    reuse_freqs_front_part,
    nope,
    nope_first,
    rotate_style,
):
    q_size = QH_PER_KH * KH * D
    kv_size = KH * D
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    q = q.view(-1, QH_PER_KH * KH, D).contiguous()
    k = k.view(-1, KH, D).contiguous()
    v = v.view(-1, KH, D).contiguous()

    q = ref_rope_sbhd_fwd(
        q,
        ref_freqs,
        rotate_style=rotate_style,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=nope_first,
    )
    k = ref_rope_sbhd_fwd(
        k,
        ref_freqs,
        rotate_style=rotate_style,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=nope_first,
    )

    return q, k, v


# ============================================================================
# TEST CONFIGURATIONS
# ============================================================================

# Full parameter space from test_fused_qkv_split_qk_rope_harness.py
_B_VALUES = [1, 4, 8, 16, 32]
_QH_PER_KH_VALUES = [1, 2, 4, 8, 16]
_KH_VALUES = [1, 4]
_D_VALUES = [64, 128]
_ROTATE_STYLES = [RotateStyle.GPTJ, RotateStyle.NEOX]
_MAX_EMBED_POSITIONS = 131072
_NOPE_CONFIGS = [(False, False), (True, False), (True, True)]
_REUSE_FREQS = [False, True]
_DTYPE = torch.bfloat16

ALL_CONFIGS = []
for B in _B_VALUES:
    for QH_PER_KH in _QH_PER_KH_VALUES:
        for KH in _KH_VALUES:
            for D in _D_VALUES:
                for rotate_style in _ROTATE_STYLES:
                    for nope, nope_first in _NOPE_CONFIGS:
                        for reuse in _REUSE_FREQS:
                            ALL_CONFIGS.append(
                                (B, QH_PER_KH, KH, D, rotate_style, nope, nope_first, reuse)
                            )

_n_all = len(ALL_CONFIGS)
if _n_all <= 25:
    HARNESS_CONFIGS = ALL_CONFIGS
else:
    _harness_indices = [int(round(i * (_n_all - 1) / 24)) for i in range(25)]
    HARNESS_CONFIGS = [ALL_CONFIGS[i] for i in _harness_indices]

_profile_indices = [int(round(i * (_n_all - 1) / 4)) for i in range(5)]
PROFILE_CONFIGS = [ALL_CONFIGS[i] for i in _profile_indices]

# For backward compatibility
EVAL_CONFIGS = HARNESS_CONFIGS
PROFILE_SHAPES = PROFILE_CONFIGS

RTOL, ATOL = 1e-2, 1e-2


# ============================================================================
# TEST HARNESS
# ============================================================================


def _run_single_correctness(B, QH_PER_KH, KH, D, rotate_style, nope, nope_first,
                            reuse_freqs_front_part, dtype=_DTYPE):
    """Run a single correctness check. Returns (passed, error_msg)."""
    head_dim = D * (2 if nope else 1)
    qkv = generate_qkv_inputs(B, QH_PER_KH, KH, D, nope, nope_first, dtype)

    pos, freqs, cos, sin = generate_rope_cached_freqs(
        B, _MAX_EMBED_POSITIONS,
        (D // 2) if reuse_freqs_front_part else D,
        dtype,
    )
    ref_freqs = freqs[pos].squeeze(-2)

    q_triton, k_triton, v_triton = fused_qkv_split_qk_rope(
        qkv, cos, sin, pos,
        QH_PER_KH * KH, KH, head_dim,
        is_neox=(rotate_style == RotateStyle.NEOX),
        offsets=None,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=nope_first,
    )
    q_torch, k_torch, v_torch = torch_op(
        qkv, QH_PER_KH, KH, head_dim,
        ref_freqs, reuse_freqs_front_part, nope, nope_first, rotate_style,
    )

    torch.testing.assert_close(q_torch, q_triton, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(k_torch, k_triton, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(v_torch, v_triton, atol=ATOL, rtol=RTOL)


def run_correctness(configs=None, verbose=True):
    if configs is None:
        configs = HARNESS_CONFIGS
    print(f"Running correctness on {len(configs)} configs...")
    results, failures = [], []
    for idx, (B, QH_PER_KH, KH, D, rs, nope, nope_first, reuse) in enumerate(configs):
        tag = f"B={B} QH_PER_KH={QH_PER_KH} KH={KH} D={D} rs={rs.name} nope={nope} nope_first={nope_first} reuse={reuse}"
        try:
            _run_single_correctness(B, QH_PER_KH, KH, D, rs, nope, nope_first, reuse)
            results.append(tag)
            if verbose:
                print(f"  PASS: {tag}")
        except Exception as e:
            failures.append({"config": tag, "error": str(e)})
            if verbose:
                print(f"  FAIL: {tag} - {str(e)[:60]}")
        torch.cuda.empty_cache()

    if verbose:
        print("-" * 62)
        status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{len(configs)})"
        print(f"{'Status:':<22} {status}")

    return {
        "correct": len(failures) == 0,
        "num_correct": len(results),
        "num_failed": len(failures),
        "failures": failures,
    }


def run_profile(configs=None, warmup=50, iters=200, verbose=True):
    if configs is None:
        configs = PROFILE_CONFIGS
    if verbose:
        print(f"Profile: {len(configs)} config(s), {warmup} warmup, {iters} iter(s)")

    dtype = _DTYPE
    for B, QH_PER_KH, KH, D, rs, nope, nope_first, reuse in configs:
        head_dim = D * (2 if nope else 1)
        qkv = generate_qkv_inputs(B, QH_PER_KH, KH, D, nope, nope_first, dtype)
        pos, freqs, cos, sin = generate_rope_cached_freqs(
            B, _MAX_EMBED_POSITIONS, (D // 2) if reuse else D, dtype,
        )
        for _ in range(warmup):
            fused_qkv_split_qk_rope(
                qkv, cos, sin, pos, QH_PER_KH * KH, KH, head_dim,
                is_neox=(rs == RotateStyle.NEOX), reuse_freqs_front_part=reuse,
                nope_first=nope_first,
            )
        torch.cuda.synchronize()
        for _ in range(iters):
            fused_qkv_split_qk_rope(
                qkv, cos, sin, pos, QH_PER_KH * KH, KH, head_dim,
                is_neox=(rs == RotateStyle.NEOX), reuse_freqs_front_part=reuse,
                nope_first=nope_first,
            )
        torch.cuda.synchronize()
        if verbose:
            print(f"  B={B} QH_PER_KH={QH_PER_KH} KH={KH} D={D} rs={rs.name} done")
        del qkv
        torch.cuda.empty_cache()


def run_benchmark(configs=None, warmup=50, iters=200, verbose=True):
    if configs is None:
        configs = HARNESS_CONFIGS
    dtype = _DTYPE
    latencies = []
    speedups = []
    results = []

    print(f"Running benchmark on {len(configs)} configs, {warmup} warmup, {iters} iterations each...")
    if verbose:
        print(f"{'Config':<50} {'PyTorch':>10} {'Triton':>10} {'Speedup':>10}")
        print("-" * 90)

    for B, QH_PER_KH, KH, D, rs, nope, nope_first, reuse in configs:
        head_dim = D * (2 if nope else 1)
        qkv = generate_qkv_inputs(B, QH_PER_KH, KH, D, nope, nope_first, dtype)
        pos, freqs, cos, sin = generate_rope_cached_freqs(
            B, _MAX_EMBED_POSITIONS, (D // 2) if reuse else D, dtype,
        )
        ref_freqs = freqs[pos].squeeze(-2)

        for _ in range(warmup):
            fused_qkv_split_qk_rope(
                qkv, cos, sin, pos, QH_PER_KH * KH, KH, head_dim,
                is_neox=(rs == RotateStyle.NEOX), reuse_freqs_front_part=reuse,
                nope_first=nope_first,
            )
        torch.cuda.synchronize()

        triton_times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fused_qkv_split_qk_rope(
                qkv, cos, sin, pos, QH_PER_KH * KH, KH, head_dim,
                is_neox=(rs == RotateStyle.NEOX), reuse_freqs_front_part=reuse,
                nope_first=nope_first,
            )
            end.record()
            torch.cuda.synchronize()
            triton_times.append(start.elapsed_time(end))

        triton_ms = sorted(triton_times)[len(triton_times) // 2]

        torch_times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            torch_op(qkv, QH_PER_KH, KH, head_dim, ref_freqs, reuse, nope, nope_first, rs)
            end.record()
            torch.cuda.synchronize()
            torch_times.append(start.elapsed_time(end))

        torch_ms = sorted(torch_times)[len(torch_times) // 2]
        speedup = torch_ms / triton_ms if triton_ms > 0 else 1.0
        latencies.append(triton_ms)
        speedups.append(speedup)

        tag = f"B={B} QH={QH_PER_KH} KH={KH} D={D} {rs.name} nope={nope}"
        results.append({"config": tag, "torch_ms": torch_ms, "triton_ms": triton_ms, "speedup": speedup})

        if verbose:
            marker = " *" if speedup > 1.0 else ""
            print(f"{tag:<50} {torch_ms:>8.4f}ms {triton_ms:>8.4f}ms {speedup:>8.2f}x{marker}")

        del qkv
        torch.cuda.empty_cache()

    log_sum = sum(math.log(t) for t in latencies)
    geomean_latency = math.exp(log_sum / len(latencies))

    log_sum_speedup = sum(math.log(s) for s in speedups)
    geomean_speedup = math.exp(log_sum_speedup / len(speedups))

    if verbose:
        print("-" * 90)
        print(f"{'Geometric mean latency:':<50} {geomean_latency:.4f} ms")
        print(f"{'Geometric mean speedup:':<50} {geomean_speedup:.2f}x")
        print(f"GEAK_RESULT_LATENCY_MS={geomean_latency:.4f}")
        print(f"GEAK_RESULT_SPEEDUP={geomean_speedup:.2f}")

    return {
        "geomean_latency_ms": geomean_latency,
        "geomean_speedup": geomean_speedup,
        "results": results,
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fused QKV Split + QK RoPE Kernel Test Harness")
    parser.add_argument(
        "--correctness",
        action="store_true",
        help="Run correctness tests on benchmark configs",
    )
    parser.add_argument(
        "--profile", action="store_true", help="Run minimal profiling workload"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark on HARNESS_CONFIGS (25 uniformly sampled)",
    )
    parser.add_argument(
        "--full-benchmark",
        action="store_true",
        help="Run benchmark on ALL_CONFIGS (complete set)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Number of warmup iterations (default: 50)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Number of benchmark iterations (default: 200)",
    )
    args = parser.parse_args()

    print("=" * 62)
    print("Fused QKV Split + QK RoPE Kernel Test Harness")
    print("=" * 62)

    if args.correctness:
        print("\n[Correctness Mode]")
        run_correctness(HARNESS_CONFIGS)
    elif args.profile:
        print("\n[Profile Mode]")
        run_profile(PROFILE_CONFIGS, warmup=args.warmup, iters=args.iterations)
    elif args.full_benchmark:
        print("\n[Full Benchmark Mode]")
        run_benchmark(ALL_CONFIGS, warmup=args.warmup, iters=args.iterations)
    else:
        print("\n[Benchmark Mode]")
        run_benchmark(HARNESS_CONFIGS, warmup=args.warmup, iters=args.iterations)

    print("=" * 62)
