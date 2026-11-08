# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""3x3 convolution (NHWC, bf16, Winograd F(4x4,3x3)) from ROCm/aiter's Triton conv2d.

Vendored from ROCm/aiter @ 652f2d4b768239aefd515c929e84817681ff598d: the three grid helpers and `_launch_winograd_f4x3` of aiter/ops/triton/conv/_launch.py, unchanged, with the input
transform, batched GEMM and output transform kernels of
aiter/ops/triton/_triton_kernels/conv/conv_3x3_winograd_f4x3.py (conv_3x3_winograd_f4x3.py in this package; the
channel-blocked input transform of that file is not used).
aiter's `_resolve_route` with layout="nhwc" returns Route.WF4X3 when `_select_3x3_method` picks Winograd (C >= 512,
K >= 512 and T >= 98 output tiles); this scenario fixes C = K = 512 and its axes keep T >= 100.

`run` is the task entry point. It replaces the dispatcher of aiter/ops/triton/conv/conv2d.py
(`conv2d(..., layout="nhwc")`) by the single route that dispatcher selects for every shape of this scenario.
Allocation of the output, the weight prepack and the bias cast are not done in the call: the weight (and the fp32
bias) arrive in the kernel-facing layout produced by this task's `layout` transform, and the output is the
preallocated NHWC tensor.
"""
import torch
import triton

from .conv_3x3_winograd_f4x3 import (
    _get_config_gemm as _get_config_wino_gemm,
)
from .conv_3x3_winograd_f4x3 import (
    _get_config_input as _get_config_wino_input,
)
from .conv_3x3_winograd_f4x3 import (
    _get_config_output as _get_config_wino_output,
)
from .conv_3x3_winograd_f4x3 import (
    _winograd_f4x3_batched_gemm_kernel,
    _winograd_f4x3_input_transform_kernel,
    _winograd_f4x3_output_transform_kernel,
)
from .conv_config_utils import format_shape_key


def _make_wino_input_grid(T, C_pad):
    """Grid for the Winograd F(4,3) input-transform kernels: one program per
    tile T x (BLOCK_C tile of C_pad)."""

    def grid(meta):
        return (T, triton.cdiv(C_pad, meta["BLOCK_C"]))

    return grid


def _make_wino_gemm_grid(T, K_out):
    """Grid for the Winograd F(4,3) batched GEMM: (BLOCK_M tile of T) x
    (BLOCK_N tile of K_out) program blocks, batched over the 36 tile elements."""

    def grid(meta):
        return (
            triton.cdiv(T, meta["BLOCK_M"]) * triton.cdiv(K_out, meta["BLOCK_N"]),
            36,
        )

    return grid


def _make_wino_output_grid(T, K_out):
    """Grid for the Winograd F(4,3) output-transform kernels: one program per
    tile T x (BLOCK_K tile of K_out)."""

    def grid(meta):
        return (T, triton.cdiv(K_out, meta["BLOCK_K"]))

    return grid


def _launch_winograd_f4x3(
    x,
    U,
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
    padding,
    activation,
    layout="nchw",
):
    """Launch Winograd F(4x4,3x3) pipeline: input transform -> batched GEMM -> output transform."""
    ph, pw = padding
    tile_H = (P + 3) // 4
    tile_W = (Q + 3) // 4
    T = N * tile_H * tile_W

    input_dtype = x.dtype
    V = torch.empty((36, T, C_pad), device=x.device, dtype=input_dtype)
    M = torch.empty((36, T, K_out), device=x.device, dtype=torch.float32)

    shape_key = format_shape_key(
        N=N,
        C=C,
        H=H,
        W=W_in,
        K=K_out,
        R=3,
        S=3,
        sh=1,
        sw=1,
        ph=ph,
        pw=pw,
        dh=1,
        dw=1,
    )
    input_config = _get_config_wino_input(shape_key=shape_key, M=T)
    gemm_config = _get_config_wino_gemm(shape_key=shape_key, M=T)
    output_config = _get_config_wino_output(shape_key=shape_key, M=T)

    # 1. Input transform
    _winograd_f4x3_input_transform_kernel[_make_wino_input_grid(T, C_pad)](
        x,
        V,
        N,
        C,
        C_pad,
        H,
        W_in,
        tile_H,
        tile_W,
        T,
        ph,
        pw,
        LAYOUT=layout,
        **input_config,
    )

    # 2. Batched GEMM
    _winograd_f4x3_batched_gemm_kernel[_make_wino_gemm_grid(T, K_out)](
        V,
        U,
        M,
        T,
        K_out,
        C_pad,
        **gemm_config,
    )

    # 3. Output transform
    _winograd_f4x3_output_transform_kernel[_make_wino_output_grid(T, K_out)](
        M,
        bias_fp32,
        y,
        N,
        K_out,
        P,
        Q,
        tile_H,
        tile_W,
        T,
        HAS_BIAS=bias_fp32 is not None,
        ACTIVATION=activation,
        LAYOUT=layout,
        **output_config,
    )


def run(*, x, weight, bias, out):
    N, H, W_in, C = x.shape
    K_out = weight.shape[1]
    _launch_winograd_f4x3(
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
        "none",
        layout="nhwc",
    )
