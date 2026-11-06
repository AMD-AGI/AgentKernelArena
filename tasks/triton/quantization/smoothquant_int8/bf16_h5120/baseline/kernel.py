# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""SmoothQuant dynamic per-token INT8 quantization in Triton.

A Triton rewrite of the AITER `smoothquant_fwd` operator (Composable Kernel) built on the
quantization step of AITER's Triton `_quant_rms_norm_kernel` with the normalization removed: one
program per row multiplies the activation by the per-channel smoothing scale, takes the row maximum,
and stores truncated INT8 codes and the float32 row scale.

Adaptation: the scale and the quotient use IEEE correctly-rounded division (`tl.div_rn`) instead of `/`,
which the ROCm Triton build lowers to an approximate division; see README.md.
"""
import triton
import triton.language as tl


@triton.jit
def _smoothquant_kernel(
    x_ptr,
    x_scale_ptr,
    out_ptr,
    y_scale_ptr,
    stride_x,
    stride_out,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(x_scale_ptr + cols, mask=mask, other=0.0)
    y = x * s

    row_max = tl.max(tl.abs(y))
    scale = tl.div_rn(row_max, 127.0)
    scale = tl.where(scale == 0, 1.0, scale)

    q = tl.div_rn(y, scale)
    q = tl.minimum(tl.maximum(q, -128.0), 127.0)
    # float -> int8 conversion truncates toward zero
    tl.store(out_ptr + row * stride_out + cols, q.to(tl.int8), mask=mask)
    tl.store(y_scale_ptr + row, scale)


def run(*, x, x_scale, out, y_scale):
    M, N = x.shape
    BLOCK = triton.next_power_of_2(N)
    num_warps = min(max(BLOCK // 256, 1), 8)
    _smoothquant_kernel[(M,)](
        x,
        x_scale,
        out,
        y_scale,
        x.stride(0),
        out.stride(0),
        N,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
