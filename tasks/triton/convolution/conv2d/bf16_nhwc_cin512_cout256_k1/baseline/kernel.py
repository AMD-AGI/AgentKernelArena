# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""1x1 convolution (NHWC, bf16) from ROCm/aiter's Triton conv2d.

Vendored from ROCm/aiter @ 652f2d4b768239aefd515c929e84817681ff598d: `_make_mn_grid` and `_launch_1x1` of aiter/ops/triton/conv/_launch.py, unchanged, with the device kernel
`_conv2d_1x1_kernel` of aiter/ops/triton/_triton_kernels/conv/conv_1x1.py (conv_1x1.py in this package).
aiter's `_resolve_route` returns Route.ONE_X_ONE for every 1x1, dilation-1 convolution.

`run` is the task entry point. It replaces the dispatcher of aiter/ops/triton/conv/conv2d.py
(`conv2d(..., layout="nhwc")`) by the single route that dispatcher selects for every shape of this scenario.
Allocation of the output, the weight prepack and the bias cast are not done in the call: the weight (and the fp32
bias) arrive in the kernel-facing layout produced by this task's `layout` transform, and the output is the
preallocated NHWC tensor.
"""
import triton

from .conv_1x1 import (
    _conv2d_1x1_kernel,
)
from .conv_1x1 import (
    _get_config as _get_config_1x1,
)
from .conv_config_utils import format_shape_key


def _make_mn_grid(M_total, K_out):
    """Grid for the GEMM-style conv kernels (1x1, 3x3 nhwc/cblocked, general):
    one program per (BLOCK_M tile of M_total) x (BLOCK_N tile of K_out)."""

    def grid(meta):
        return (
            triton.cdiv(M_total, meta["BLOCK_M"]) * triton.cdiv(K_out, meta["BLOCK_N"]),
        )

    return grid


def _launch_1x1(
    x,
    w_oihw,
    bias_fp32,
    y,
    N,
    C,
    H,
    W_in,
    K_out,
    P,
    Q,
    stride,
    padding,
    activation,
    layout="nchw",
):
    """Launch specialized 1x1 kernel.
    layout: "nchw" or "nhwc" (case-insensitive).
    """
    sh, sw = stride
    ph, pw = padding

    w = w_oihw.squeeze(-1).squeeze(-1).contiguous()  # [K_out, C]

    M_total = N * P * Q

    shape_key = format_shape_key(
        N=N,
        C=C,
        H=H,
        W=W_in,
        K=K_out,
        R=1,
        S=1,
        sh=sh,
        sw=sw,
        ph=ph,
        pw=pw,
        dh=1,
        dw=1,
    )
    config = _get_config_1x1(shape_key=shape_key, M=M_total)

    _conv2d_1x1_kernel[_make_mn_grid(M_total, K_out)](
        x,
        w,
        bias_fp32,
        y,
        N,
        C,
        H,
        W_in,
        K_out,
        P,
        Q,
        sh,
        sw,
        ph,
        pw,
        M_total,
        HAS_BIAS=bias_fp32 is not None,
        ACTIVATION=activation,
        LAYOUT=layout,
        **config,
    )


def run(*, x, weight, bias, out):
    N, H, W_in, C = x.shape
    K_out = weight.shape[0]
    _launch_1x1(
        x,
        weight.view(K_out, C, 1, 1),
        bias,
        out,
        N,
        C,
        H,
        W_in,
        K_out,
        H,
        W_in,
        (1, 1),
        (0, 0),
        "none",
        layout="nhwc",
    )
