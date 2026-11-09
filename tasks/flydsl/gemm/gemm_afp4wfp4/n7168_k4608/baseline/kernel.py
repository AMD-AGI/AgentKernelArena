# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MXFP4 x MXFP4 preshuffle GEMM launcher (gfx950): out = dequant(a) @ dequant(w)^T with per-1x32 e8m0 scales folded
into a scaled 16x16x128 MFMA.

The launch follows `_mxfp4_launcher` of tests/kernels/test_preshuffle_gemm.py in ROCm/FlyDSL: `launch_gemm` of
`mxfp4_preshuffle.py` (vendored beside this file) with one batch element, tile 64x256x256 and fp4 activations.
Operands are expected in the preshuffled layout: `w` shuffled as `shuffle_weight_w4(., 16)`, both scale tensors
as `shuffle_scale_w4(., 1, False)`, and `a_scales` padded to a multiple of 32 rows.
"""
from __future__ import annotations

import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

from .mxfp4_preshuffle import launch_gemm

_TILE_M, _TILE_N, _TILE_K = 64, 256, 256  # the tile of the large-M cases of the FlyDSL test


def _ptr(t: torch.Tensor):
    """Raw device pointer as an fx.Pointer kernel argument (no per-launch DLPack conversion)."""
    return flyc.from_c_void_p(fx.Uint8, t.data_ptr())


@functools.lru_cache(maxsize=None)
def _launch_constants(n: int, k: int):
    """Constexpr arguments of `launch_gemm` that follow `(c, a, b, scale_a, scale_b, M, N, stream)`."""
    if n % _TILE_N != 0:
        raise RuntimeError(f"[FlyDSL] N ({n}) must be a multiple of tile_n ({_TILE_N})")
    if k % _TILE_K != 0:
        raise RuntimeError(f"[FlyDSL] K ({k}) must be a multiple of tile_k ({_TILE_K})")
    # One batch element; contiguous A, scale_a and C (all strides < 0); waves_per_eu unset (0).
    strides = (-1, -1, -1, -1, -1, -1)
    return (n, k, _TILE_M, _TILE_N, _TILE_K, "fp4", "bf16", "fp4", 1, *strides, 0)


def run(*, a, a_scales, w, w_scales, out) -> None:
    """out [m, n] = dequant(a) @ dequant(w)^T; a [m, k // 2] codes, a_scales [ceil(m / 32) * 32, k // 32] shuffled
    e8m0, w [n, k // 2] and w_scales [n, k // 32] preshuffled."""
    m, half_k = a.shape
    n = out.shape[1]
    constants = _launch_constants(n, half_k * 2)
    launch_gemm(
        _ptr(out),
        _ptr(a),
        _ptr(w),
        _ptr(a_scales),
        _ptr(w_scales),
        m,
        n,
        torch.cuda.current_stream(),
        *constants,
    )
