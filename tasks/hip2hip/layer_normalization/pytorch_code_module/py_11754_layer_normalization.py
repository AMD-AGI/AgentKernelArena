# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import torch
import torch.nn as nn


class layer_normalization(nn.Module):

    def __init__(self, features, epsilon=1e-08):
        """Applies layer normalization.

        Args:
          epsilon: A floating number. A very small number for preventing ZeroDivision Error.
        """
        super(layer_normalization, self).__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))

    def forward(self, x, fn=None):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.epsilon) + self.beta


def get_inputs():
    """
    Generate multiple test cases for normalization covering:
    - Different feature dimensions
    - Different batch and sequence sizes
    """
    configs = [
        # Small - ensure last dimension matches features=4
        ([4, 4],),
        ([8, 4],),
        ([4, 8, 4],),
        ([8, 16, 4],),
        # Medium
        ([16, 32, 4],),
        ([32, 64, 4],),
    ]
    
    for shape in configs:
        # Unpack tuple if shape is a tuple containing a list (e.g., ([1024],) -> [1024])
        shape_list = shape[0] if isinstance(shape, tuple) and len(shape) == 1 else shape
        x = torch.randn(shape_list, dtype=torch.float32)
        yield [x]


def get_init_inputs():
    return [[], {'features': 4}]
