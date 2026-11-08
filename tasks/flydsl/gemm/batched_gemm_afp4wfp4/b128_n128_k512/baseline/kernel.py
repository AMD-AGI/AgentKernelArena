# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Strided-batched MXFP4 x MXFP4 preshuffle GEMM launcher (gfx950): out[b] = dequant(x[b]) @ dequant(w[b])^T with per-1x32
e8m0 scales folded into a scaled 16x16x128 MFMA.

Provenance: the gfx950 path of `flydsl_batched_gemm_mxfp4` in ROCm/aiter aiter/ops/flydsl/batched_gemm_mxfp4.py (the
launcher, its argument checks and its default tile 128x128x256, layout 'bmn', fp4 activations) driving the FlyDSL
kernel `launch_gemm` of ROCm/FlyDSL kernels/gemm/mxfp4_preshuffle.py, vendored beside this file as
`mxfp4_preshuffle.py`. Operands are expected in the preshuffled layout (see the task README): `w` shuffled with
`shuffle_weight_w4(., 16)` and the scales with `shuffle_scale_w4`, `x_scales` padded to a multiple of 32 rows first.
"""
from __future__ import annotations

import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

from .mxfp4_preshuffle import launch_gemm

_A_DTYPE = "fp4"  # MXFP4 activations; a8w4 and a6w4 variants of the kernel are not used
_B_DTYPE = "fp4"  # MXFP4 weights
_OUT_DTYPE = "bf16"
_TILE_M, _TILE_N, _TILE_K = 128, 128, 256  # the launcher's defaults


def _ptr(t: torch.Tensor):
    """Raw device pointer as an fx.Pointer kernel argument (no per-launch DLPack conversion)."""
    return flyc.from_c_void_p(fx.Uint8, t.data_ptr())


@functools.lru_cache(maxsize=None)
def _launch_constants(batch: int, n: int, k: int):
    """Constexpr launch arguments of `launch_gemm` that follow `(c, a, b, scale_a, scale_b, M, N, stream)`."""
    # tile_m % 16 / tile_n % 64 must be exact or the kernel's chunk counts silently drop work.
    if _TILE_M % 16 != 0:
        raise RuntimeError(f"[FlyDSL] tile_m ({_TILE_M}) must be a multiple of 16")
    if _TILE_N % 64 != 0:
        raise RuntimeError(f"[FlyDSL] tile_n ({_TILE_N}) must be a multiple of 64")
    if n % _TILE_N != 0:
        raise RuntimeError(f"[FlyDSL] N ({n}) must be a multiple of tile_n ({_TILE_N})")
    if k % 256 != 0:
        raise RuntimeError(f"[FlyDSL] K ({k}) must be a multiple of 256")
    if _TILE_K not in (128, 256) or k % _TILE_K != 0:
        raise RuntimeError(f"[FlyDSL] tile_k must be 128/256 dividing K; got {_TILE_K}, K={k}")
    # bmn layout: A, scale_a and C use the contiguous [B, M, *] defaults (all strides < 0); waves_per_eu unset (0).
    strides = (-1, -1, -1, -1, -1, -1)
    return (n, k, _TILE_M, _TILE_N, _TILE_K, _A_DTYPE, _OUT_DTYPE, _B_DTYPE, batch, *strides, 0)


def run(*, x, x_scales, w, w_scales, out) -> None:
    """out[B, M, N] = dequant(x) @ dequant(w)^T; x [B, M, K // 2] codes, x_scales [B, ceil(M / 32) * 32, K // 32]
    shuffled e8m0, w [B, N, K // 2] and w_scales [B, N, K // 32] preshuffled."""
    batch, m, half_k = x.shape
    n = out.shape[2]
    constants = _launch_constants(batch, n, half_k * 2)
    launch_gemm(
        _ptr(out),
        _ptr(x),
        _ptr(w),
        _ptr(x_scales),
        _ptr(w_scales),
        m,
        n,
        torch.cuda.current_stream(),
        *constants,
    )
