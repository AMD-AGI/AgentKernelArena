# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

def fused_leaky_relu_fn(
    x: torch.Tensor,
    bias: torch.Tensor,
    negative_slope: float,
    scale: float,
) -> torch.Tensor:
    # Add bias (reshaped), slice to match input channels, apply leaky_relu, then scale
    x = x + bias.reshape(1, -1, 1, 1)[:, :x.shape[1]]
    x = F.leaky_relu(x, negative_slope=negative_slope, inplace=True)
    x = x * scale
    return x

def module_fn(
    x: torch.Tensor,
    bias: torch.Tensor,
    negative_slope: float,
    scale: float,
) -> torch.Tensor:
    return fused_leaky_relu_fn(x, bias, negative_slope, scale)

class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, x, fn=module_fn):
        return fn(x, self.bias, self.negative_slope, self.scale)

def get_inputs():
    """
    Generate multiple test cases for FusedLeakyReLU
    HIP kernel requires 4D input [N, C, H, W]
    """
    configs = [
        # (N, C, H, W) - 4D tensors only
        ([8, 4, 32, 32],),
        ([16, 4, 32, 32],),
        ([8, 64, 32, 32],),
        ([16, 128, 64, 64],),
        ([32, 256, 128, 128],),
    ]
    
    for shape in configs:
        # Unpack tuple if shape is a tuple containing a list (e.g., ([1024],) -> [1024])
        shape_list = shape[0] if isinstance(shape, tuple) and len(shape) == 1 else shape
        x = torch.randn(shape_list, dtype=torch.float32)
        yield [x]


def get_init_inputs():
    # Return multiple channel options to match test case requirements
    # The test framework will use the appropriate one based on test case
    return [[], {'channel': 256}]  # Use max channel count from test cases
