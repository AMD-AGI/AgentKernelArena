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
        ([1024, 1024], [1024, 1024]),
        ([2048, 2048], [2048, 2048]),
        ([4096, 4096], [4096, 4096]),
        ([2048, 8192], [8192, 2048]),
        ([8192, 1024], [1024, 4096]),
        ([8, 1024, 1024], [8, 1024, 1024]),
        ([32, 512, 512], [32, 512, 512]),
        ([4, 8, 256, 512], [4, 8, 512, 256]),
    ]
    
    for a_shape, b_shape in configs:
        a = torch.rand(a_shape, dtype=torch.float32)
        b = torch.rand(b_shape, dtype=torch.float32)
        yield [a, b]


def get_init_inputs():
    return [[], {}]
