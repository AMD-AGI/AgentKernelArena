# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import torch
from torch import nn
from torch.nn.functional import leaky_relu


class FusedLeakyReLU(nn.Module):

    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, x):
        return self.scale * leaky_relu(x + self.bias.reshape((1, -1, 1, 1))
            [:, :x.shape[1]], self.negative_slope, inplace=True)


def get_inputs():
    """
    Generate multiple test cases for FusedLeakyReLU
    HIP kernel requires 4D input [N, C, H, W]
    All test cases must use C<=256 to match max channel count
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
    # Use max channel count to match functional version
    return [[], {'channel': 256}]
