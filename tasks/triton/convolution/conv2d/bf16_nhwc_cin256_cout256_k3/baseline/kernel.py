# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""3x3 convolution (NHWC, bf16, direct kernel) from ROCm/aiter's Triton conv2d.

Vendored from ROCm/aiter @ 652f2d4b768239aefd515c929e84817681ff598d: `_make_mn_grid` and `_launch_3x3_nhwc` of aiter/ops/triton/conv/_launch.py, unchanged, with the device kernel
`_conv2d_3x3_nhwc_kernel` of aiter/ops/triton/_triton_kernels/conv/conv_3x3.py (conv_3x3.py in this package; the
channel-blocked kernel of that file is not used).
aiter's `_resolve_route` with layout="nhwc" returns Route.NHWC_3X3 unless `_select_3x3_method` picks Winograd, which
needs C >= 512 and K >= 512; C = 256 here.

`run` is the task entry point. It replaces the dispatcher of aiter/ops/triton/conv/conv2d.py
(`conv2d(..., layout="nhwc")`) by the single route that dispatcher selects for every shape of this scenario.
Allocation of the output, the weight prepack and the bias cast are not done in the call: the weight (and the fp32
bias) arrive in the kernel-facing layout produced by this task's `layout` transform, and the output is the
preallocated NHWC tensor.
"""
import triton

from .conv_3x3 import (
    _conv2d_3x3_nhwc_kernel,
    _get_config_nhwc,
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


def _launch_3x3_nhwc(
    x,
    w_3x3,
    bias_fp32,
    y,
    N,
    C,
    H,
    W_in,
    K_out,
    P,
    Q,
    C_pad,
    stride,
    padding,
    dilation,
    activation,
):
    """Launch specialized 3x3 NHWC kernel (hardcoded stride_c=1, stride_k=1)."""
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
        R=3,
        S=3,
        sh=sh,
        sw=sw,
        ph=ph,
        pw=pw,
        dh=dh,
        dw=dw,
    )
    config = _get_config_nhwc(shape_key=shape_key, M=M_total)

    _conv2d_3x3_nhwc_kernel[_make_mn_grid(M_total, K_out)](
        x,
        w_3x3,
        bias_fp32,
        y,
        N,
        C,
        H,
        W_in,
        K_out,
        P,
        Q,
        C_pad,
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
        M_total,
        HAS_BIAS=bias_fp32 is not None,
        ACTIVATION=activation,
        **config,
    )


def run(*, x, weight, bias, out):
    N, H, W_in, C = x.shape
    K_out = weight.shape[0]
    _launch_3x3_nhwc(
        x,
        weight,
        bias,
        out,
        N,
        C,
        H,
        W_in,
        K_out,
        H,
        W_in,
        C,
        (1, 1),
        (1, 1),
        (1, 1),
        "none",
    )
