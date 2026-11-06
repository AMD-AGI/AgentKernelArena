# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Dynamic per-tensor FP8 quantization (Triton).

Vendored from ROCm/aiter, aiter/ops/triton/_triton_kernels/quant/quant.py and
aiter/ops/triton/quant/quant.py (dynamic_per_tensor_quant_fp8_i8). Three launches:
zero the scale, reduce the tensor amax into it with atomics, then quantize.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _zero_scale_kernel(scale_ptr):
    tl.store(scale_ptr, 0.0)


@triton.jit
def _static_per_tensor_quant_fp8_i8_kernel(
    qx_ptr,
    x_in_ptr,
    scale_in_ptr,
    cols: int,
    x_in_stride_r: int,
    NUM_COL_POW2: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    tl.assume(x_in_stride_r > 0)

    offs = pid * x_in_stride_r + tl.arange(0, NUM_COL_POW2)
    mask = tl.arange(0, NUM_COL_POW2) < cols
    x = tl.load(x_in_ptr + offs, mask=mask, cache_modifier=".cg")

    scale = tl.load(scale_in_ptr)
    scale_recip = 1 / scale

    qx = (x * scale_recip).to(qx_ptr.dtype.element_ty)

    tl.store(qx_ptr + offs, qx, mask=mask)


@triton.jit
def _dynamic_per_tensor_quant_fp8_i8_kernel(
    x_in_ptr,
    scale_out_ptr,
    cols: int,
    x_in_stride_r: int,
    NUM_COL_POW2: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    tl.assume(x_in_stride_r > 0)

    offs = pid * x_in_stride_r + tl.arange(0, NUM_COL_POW2)
    mask = tl.arange(0, NUM_COL_POW2) < cols
    x = tl.load(x_in_ptr + offs, mask=mask, cache_modifier=".cg")

    m = tl.max(tl.abs(x)).to(tl.float32)
    tl.atomic_max(scale_out_ptr, m / DTYPE_MAX, sem="relaxed")


def run(*, x, qx, scale) -> None:
    rows, cols = x.shape
    NUM_COL_POW2 = triton.next_power_of_2(cols)
    grid = (rows,)
    _zero_scale_kernel[(1,)](scale)
    _dynamic_per_tensor_quant_fp8_i8_kernel[grid](
        x,
        scale,
        cols,
        x.stride(0),
        NUM_COL_POW2=NUM_COL_POW2,
        DTYPE_MAX=torch.finfo(qx.dtype).max,
    )
    _static_per_tensor_quant_fp8_i8_kernel[grid](
        qx, x, scale, cols, x.stride(0), NUM_COL_POW2=NUM_COL_POW2
    )
