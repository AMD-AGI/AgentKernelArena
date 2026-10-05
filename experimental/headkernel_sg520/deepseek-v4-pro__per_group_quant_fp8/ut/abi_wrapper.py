# SPDX-License-Identifier: MIT
# Frozen exact AITER ABI body; native dependency is injected by native.py.
import torch
from torch import Tensor
from aiter.utility import dtypes

def per_group_quant_hip(
    x: Tensor,
    scale: Tensor | None = None,
    quant_dtype: torch.dtype = dtypes.i8,
    group_size: int = 128,
    transpose_scale: bool = False,
    num_rows: "torch.Tensor | None" = None,
    num_rows_factor: int = 1,
    scale_type: torch.dtype = dtypes.fp32,
) -> "tuple[Tensor, Tensor]":
    shape = x.shape
    device = x.device
    if scale is None:
        scale = torch.empty(
            (*shape[:-1], shape[-1] // group_size), dtype=scale_type, device=device
        )
    else:
        raise ValueError("unsupported: static per token quant")
    assert group_size in [
        32,
        64,
        128,
    ], f"unsupported group size {group_size=}, only support [32, 64, 128]"
    y = torch.empty(shape, dtype=quant_dtype, device=device)
    if scale_type == dtypes.fp8_e8m0:
        dynamic_per_group_scaled_quant(
            y,
            x,
            scale,
            group_size,
            shuffle_scale=transpose_scale,
            num_rows=num_rows,
            num_rows_factor=num_rows_factor,
        )
    else:
        dynamic_per_token_scaled_quant(
            y,
            x.view(-1, group_size),
            scale,
            shuffle_scale=transpose_scale,
            num_rows=num_rows,
            num_rows_factor=num_rows_factor,
        )
    return y, scale
