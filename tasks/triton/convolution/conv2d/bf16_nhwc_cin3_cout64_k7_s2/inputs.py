"""Filters of the 7x7 stem convolution and the kernel-facing weight layout.

Logical `weight` is the OIHW tensor [out_channels, in_channels, 7, 7] drawn with torchvision's ResNet initialization
(`kaiming_normal_(mode="fan_out", nonlinearity="relu")`: normal with std sqrt(2 / (out_channels * 7 * 7))) and stored
in bfloat16.
"""
import math

import torch

_BLOCK_K = 64  # padding granularity of aiter's prepack (BLOCK_K in aiter/ops/triton/conv/_utils.py)


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    out_channels, in_channels = axes["out_channels"], axes["in_channels"]
    std = math.sqrt(2.0 / (out_channels * 7 * 7))
    weight = torch.randn(out_channels, in_channels, 7, 7, generator=generator, device=device, dtype=torch.float32)
    return {"weight": (weight * std).to(torch.bfloat16)}


def to_kernel_layout(case, inputs):
    """aiter prepack_oihw_to_kmajor: [out_channels, in_channels * 7 * 7] zero-padded to a multiple of 64 columns."""
    inputs = dict(inputs)
    weight = inputs["weight"]
    out_channels = weight.shape[0]
    reduction = weight.shape[1] * weight.shape[2] * weight.shape[3]
    padded = ((reduction + _BLOCK_K - 1) // _BLOCK_K) * _BLOCK_K
    flat = weight.reshape(out_channels, reduction)
    if padded != reduction:
        pad = torch.zeros((out_channels, padded - reduction), device=weight.device, dtype=weight.dtype)
        flat = torch.cat([flat, pad], dim=1)
    inputs["weight"] = flat.contiguous()
    return inputs
