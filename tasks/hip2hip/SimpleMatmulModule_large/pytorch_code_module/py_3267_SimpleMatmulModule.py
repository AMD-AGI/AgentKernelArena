# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import torch
import torch.jit
import torch.onnx
import torch.nn


class SimpleMatmulModule(torch.nn.Module):

    def __init__(self):
        super(SimpleMatmulModule, self).__init__()

    def forward(self, a, b):
        return a.matmul(b + b)


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
