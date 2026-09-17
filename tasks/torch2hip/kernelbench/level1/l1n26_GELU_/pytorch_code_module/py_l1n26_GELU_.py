# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import torch
import torch.nn as nn

class GELU_(nn.Module):
    """
    Simple model that performs a GELU activation.
    """
    def __init__(self):
        super(GELU_, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies GELU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with GELU applied, same shape as input.
        """
        return torch.nn.functional.gelu(x)

batch_size = 4096
dim = 393216

def get_inputs():
    # Element-wise GELU across multiple ranks (gpumode activation template, perf scale).
    configs = [
        [16777216],            # 1D
        [8192, 16384],         # 2D
        [256, 1024, 1024],     # 3D
        [32, 64, 512, 512],    # 4D (feature-map-like)
        [4096, 32768],         # large 2D
    ]
    for shape in configs:
        yield [torch.rand(shape).mul_(12).sub_(6)]


def get_init_inputs():
    return []  # No special initialization inputs needed
