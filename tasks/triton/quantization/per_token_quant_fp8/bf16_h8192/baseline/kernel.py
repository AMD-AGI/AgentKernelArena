# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Dynamic per-token FP8 quantization (Triton).

Vendored from ROCm/aiter, aiter/ops/triton/_triton_kernels/quant/quant.py
(`_dynamic_per_token_quant_fp8_i8_kernel`) and aiter/ops/triton/quant/quant.py
(`dynamic_per_token_quant_fp8_i8`): one program per row computes the row amax, the scale
amax / FP8_MAX and the codes.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _dynamic_per_token_quant_fp8_i8_kernel(
    qx_ptr,
    scale_out_ptr,
    x_in_ptr,
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

    m = tl.max(tl.abs(x), axis=-1)
    scale_out = m.to(tl.float32) / DTYPE_MAX
    scale_recip = 1 / scale_out

    qx = x * scale_recip
    qx = qx.to(qx_ptr.dtype.element_ty)

    scale_offs = pid
    tl.store(scale_out_ptr + scale_offs, scale_out)

    tl.store(qx_ptr + offs, qx, mask=mask, cache_modifier=".cs")


def run(*, x, qx, scale) -> None:
    rows = x.shape[0]
    cols = x.shape[1]
    NUM_COL_POW2 = triton.next_power_of_2(cols)
    grid = lambda meta: (rows,)  # noqa: E731
    _dynamic_per_token_quant_fp8_i8_kernel[grid](
        qx,
        scale,
        x,
        cols,
        x.stride(0),
        NUM_COL_POW2=NUM_COL_POW2,
        DTYPE_MAX=torch.finfo(qx.dtype).max,
    )
