# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Implicit-GEMM conv3d launcher (BF16, NCDHW in and out): the single-pass path of `_conv3d_impl` in
kernels/conv/conv3d_implicit.py of ROCm/FlyDSL.

The activation is first transposed NCDHW -> NDHWC by the tiled transpose kernel (as `_ncdhw_to_ndhwc` does on every
call upstream), then the double-buffered implicit-GEMM kernel with the default 128x128 tile (2x4 waves) and
workgroup-swizzle 1 writes the NCDHW result directly. The weight arrives already packed as `[K, T*R*S*C]`, the
result of `_prep_weight` upstream, which a model produces once at load time.
"""
import functools

import torch

from .conv3d_implicit import DEFAULT_TILE, compile_conv3d_implicit, compile_transpose_ncdhw_ndhwc

KT, KH, KW = 3, 3, 3
STRIDE = (1, 1, 1)
PADDING = (0, 1, 1)  # the two causal frames are part of the input; no padding along depth


@functools.lru_cache(maxsize=None)
def _conv_launcher(n, c, d, h, w, k):
    return compile_conv3d_implicit(
        n, c, d, h, w, k, KT, KH, KW, *STRIDE, *PADDING, True, 1, DEFAULT_TILE, 1
    )


@functools.lru_cache(maxsize=None)
def _transpose_launcher(n, c, s):
    return compile_transpose_ncdhw_ndhwc(n, c, s)


def run(*, x, weight, bias, out) -> None:
    """out [N, K, D - 2, H, W] = conv3d(x [N, C, D, H, W], weight packed as [K, T*R*S*C], bias fp32 [K])."""
    n, c, d, h, w = x.shape
    k = weight.shape[0]
    stream = torch.cuda.current_stream()
    x_ndhwc = torch.empty((n, d, h, w, c), device=x.device, dtype=x.dtype)
    _transpose_launcher(n, c, d * h * w)(x_ndhwc, x, stream)
    _conv_launcher(n, c, d, h, w, k)(out, x_ndhwc, weight, bias, stream)
