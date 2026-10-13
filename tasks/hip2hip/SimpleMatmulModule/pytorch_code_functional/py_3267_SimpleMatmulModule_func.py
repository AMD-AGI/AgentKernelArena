# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

def module_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Performs batched matrix multiplication of a with (b + b).

    Args:
        a (torch.Tensor): Input tensor of shape (..., M, K)
        b (torch.Tensor): Input tensor of shape (..., K, N)

    Returns:
        torch.Tensor: Output tensor of shape (..., M, N)
    """
    return torch.matmul(a, b + b)

class SimpleMatmulModule(nn.Module):
    def __init__(self):
        super(SimpleMatmulModule, self).__init__()

    def forward(self, a, b, fn=module_fn):
        return fn(a, b)

def get_inputs():
    """
    Generate multiple test cases for matmul/linear kernels covering:
    - Different matrix dimensions
    - Different batch sizes
    - Square and rectangular matrices
    """
    # Test case configs: (batch_dims, m, n, k) or (a_shape, b_shape)
    configs = [
        # Small
        ([4, 4], [4, 4]),
        ([8, 8], [8, 8]),
        # Medium
        ([16, 16], [16, 16]),
        ([32, 32], [32, 32]),
        # Large
        ([64, 64], [64, 64]),
        ([128, 128], [128, 128]),
        # Rectangular
        ([16, 32], [32, 16]),
        ([32, 64], [64, 32]),
        # Batched
        ([4, 16, 16], [4, 16, 16]),
        ([8, 32, 32], [8, 32, 32]),
        # 4D (CNN-like)
        ([4, 64, 32, 32], [4, 64, 32, 32]),
    ]
    
    for a_shape, b_shape in configs:
        a = torch.rand(a_shape, dtype=torch.float32)
        b = torch.rand(b_shape, dtype=torch.float32)
        yield [a, b]


def get_init_inputs():
    return [[], {}]
