"""Filters of the 1x1 convolution and the kernel-facing weight layout.

Logical `weight` is the OIHW tensor [out_channels, in_channels, 1, 1] drawn like nn.Conv2d's default
initialization (uniform in +-1/sqrt(fan_in), fan_in = in_channels) and stored in bfloat16.
"""
import math

import torch


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    out_channels, in_channels = axes["out_channels"], axes["in_channels"]
    bound = 1.0 / math.sqrt(in_channels)
    weight = torch.rand(out_channels, in_channels, 1, 1, generator=generator, device=device, dtype=torch.float32)
    weight = (weight * (2 * bound) - bound).to(torch.bfloat16)
    return {"weight": weight}


def to_kernel_layout(case, inputs):
    """aiter's `_launch_1x1` consumes the filters as the [out_channels, in_channels] matrix."""
    inputs = dict(inputs)
    weight = inputs["weight"]
    inputs["weight"] = weight.reshape(weight.shape[0], weight.shape[1]).contiguous()
    return inputs
