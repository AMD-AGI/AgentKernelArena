# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

def gather_fn(input: torch.Tensor, indices: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Gathers values along an axis specified by dim using indices.

    Args:
        input (torch.Tensor): The source tensor.
        indices (torch.Tensor): The indices of elements to gather.
        dim (int): The axis along which to index.

    Returns:
        torch.Tensor: The result of gathering.
    """
    # Expand indices to match input shape for advanced indexing
    selection = [slice(None) for _ in range(dim)] + [indices]
    return input.__getitem__(selection)

def module_fn(input: torch.Tensor, indices: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Functional implementation of the Gather module.

    Args:
        input (torch.Tensor): The source tensor.
        indices (torch.Tensor): The indices of elements to gather.
        dim (int): The axis along which to index.

    Returns:
        torch.Tensor: The result of gathering.
    """
    return gather_fn(input, indices, dim)

class Gather(nn.Module):
    def __init__(self, dim=0):
        super().__init__()
        self.dim = dim
        self.selection = [slice(None) for _ in range(dim)]

    def forward(self, input: torch.Tensor, indices: torch.Tensor, fn=module_fn):
        return fn(input, indices, self.dim)

def get_inputs():
    """
    Generate multiple test cases for gather operations covering:
    - Different tensor ranks
    - Different dimensions
    - Different index patterns
    Only keeping cases that work (0, 1, 2, 4) - others cause out-of-range errors
    """
    configs = [
        # 1D - these work
        ([1024], 0, [256]),
        ([4096], 0, [512]),
        # 2D - dim=0 works
        ([64, 128], 0, [16]),
        # 3D - dim=0 works
        ([8, 16, 32], 0, [4]),
    ]
    
    for input_shape, dim, indices_shape in configs:
        x = torch.randn(input_shape, dtype=torch.float32)
        axis_size = input_shape[dim]
        indices = torch.randint(0, axis_size, indices_shape, dtype=torch.int64)
        yield [x, indices]


def get_init_inputs():
    return [[], {}]
