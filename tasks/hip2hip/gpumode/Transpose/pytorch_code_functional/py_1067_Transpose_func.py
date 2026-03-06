# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

def module_fn(
    input: torch.Tensor,
    dim1: int,
    dim2: int,
) -> torch.Tensor:
    """
    Applies transpose between two dimensions and returns a contiguous tensor.

    Args:
        input (torch.Tensor): Input tensor.
        dim1 (int): First dimension to be transposed.
        dim2 (int): Second dimension to be transposed.

    Returns:
        torch.Tensor: Transposed and contiguous tensor.
    """
    return input.transpose(dim1, dim2).contiguous()


class Transpose(nn.Module):
    def __init__(self, dim1=0, dim2=1):
        super(Transpose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, input, fn=module_fn):
        return fn(input, self.dim1, self.dim2)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + 'between=' + str(self.dim1
            ) + ',' + str(self.dim2) + ')'


def get_inputs():
    """
    Generate multiple test cases for transpose operations
    """
    configs = [
        ([4, 8, 16, 32],),
        ([8, 16, 32, 64],),
        ([16, 32, 64, 128],),
        ([32, 64, 128, 256],),
        ([64, 128, 256, 512],),
    ]
    
    for shape in configs:
        # Unpack tuple if shape is a tuple containing a list (e.g., ([1024],) -> [1024])
        shape_list = shape[0] if isinstance(shape, tuple) and len(shape) == 1 else shape
        x = torch.randn(shape_list, dtype=torch.float32)
        yield [x]


def get_init_inputs():
    return [[], {}]
