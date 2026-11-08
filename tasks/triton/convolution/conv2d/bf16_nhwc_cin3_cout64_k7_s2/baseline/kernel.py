# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""General convolution (NHWC, bf16, 7x7 stride 2) from ROCm/aiter's Triton conv2d.

Vendored from ROCm/aiter @ 652f2d4b768239aefd515c929e84817681ff598d: `_make_mn_grid` and `_launch_general` of aiter/ops/triton/conv/_launch.py, unchanged, with the device kernel
`_conv2d_general_kernel` of aiter/ops/triton/_triton_kernels/conv/conv_general.py (conv_general.py in this package).
aiter's `_resolve_route` returns Route.GENERAL for every kernel that is neither 1x1 nor 3x3.

`run` is the task entry point. It replaces the dispatcher of aiter/ops/triton/conv/conv2d.py
(`conv2d(..., layout="nhwc")`) by the single route that dispatcher selects for every shape of this scenario.
Allocation of the output, the weight prepack and the bias cast are not done in the call: the weight (and the fp32
bias) arrive in the kernel-facing layout produced by this task's `layout` transform, and the output is the
preallocated NHWC tensor.
"""
import triton

from .conv_config_utils import format_shape_key
from .conv_general import (
    _conv2d_general_kernel,
)
from .conv_general import (
    _get_config as _get_config_general,
)


def _make_mn_grid(M_total, K_out):
    """Grid for the GEMM-style conv kernels (1x1, 3x3 nhwc/cblocked, general):
    one program per (BLOCK_M tile of M_total) x (BLOCK_N tile of K_out)."""

    def grid(meta):
        return (
            triton.cdiv(M_total, meta["BLOCK_M"]) * triton.cdiv(K_out, meta["BLOCK_N"]),
        )

    return grid


def _launch_general(
    x,
    w_k,
    bias_fp32,
    y,
    N,
    C,
    H,
    W_in,
    K_out,
    R,
    S,
    P,
    Q,
    K_pad,
    stride,
    padding,
    dilation,
    block_k,
    activation,
    layout="nchw",
):
    """Launch general conv kernel.
    layout: "nchw" or "nhwc" (case-insensitive).
    """
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation

    M_total = N * P * Q

    shape_key = format_shape_key(
        N=N,
        C=C,
        H=H,
        W=W_in,
        K=K_out,
        R=R,
        S=S,
        sh=sh,
        sw=sw,
        ph=ph,
        pw=pw,
        dh=dh,
        dw=dw,
    )
    config = _get_config_general(shape_key=shape_key, M=M_total)

    _conv2d_general_kernel[_make_mn_grid(M_total, K_out)](
        x,
        w_k,
        bias_fp32,
        y,
        N,
        C,
        H,
        W_in,
        K_out,
        R,
        S,
        P,
        Q,
        K_pad,
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
        M_total,
        HAS_BIAS=bias_fp32 is not None,
        ACTIVATION=activation,
        LAYOUT=layout,
        **config,
    )


BLOCK_K = 64  # BLOCK_K of aiter/ops/triton/conv/_utils.py: the padding granularity of the packed weight


def run(*, x, weight, out):
    N, H, W_in, C = x.shape
    K_out = weight.shape[0]
    P, Q = out.shape[1], out.shape[2]
    _launch_general(
        x,
        weight,
        None,
        out,
        N,
        C,
        H,
        W_in,
        K_out,
        7,
        7,
        P,
        Q,
        weight.shape[1],
        (2, 2),
        (3, 3),
        (1, 1),
        BLOCK_K,
        "none",
        layout="nhwc",
    )
