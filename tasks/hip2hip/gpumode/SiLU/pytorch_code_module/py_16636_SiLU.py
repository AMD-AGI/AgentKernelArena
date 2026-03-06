# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import torch
import torch.nn as nn


class SiLU(nn.Module):

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_inputs():
    """
    Generate multiple test cases for activation functions covering:
    - Different tensor shapes and sizes
    - Different value ranges
    """
    configs = [
        # 1D
        ([1024],),
        ([4096],),
        # 2D
        ([64, 128],),
        ([128, 256],),
        # 3D
        ([32, 64, 128],),
        ([64, 128, 256],),
        # 4D (CNN feature maps)
        ([8, 64, 32, 32],),
        ([16, 128, 64, 64],),
        ([32, 256, 128, 128],),
        # Large batches
        ([128, 512],),
        ([256, 1024],),
    ]
    
    for shape in configs:
        # Unpack tuple if shape is a tuple containing a list (e.g., ([1024],) -> [1024])
        shape_list = shape[0] if isinstance(shape, tuple) and len(shape) == 1 else shape
        x = torch.randn(shape_list, dtype=torch.float32)
        yield [x]


def get_init_inputs():
    return [[], {}]
