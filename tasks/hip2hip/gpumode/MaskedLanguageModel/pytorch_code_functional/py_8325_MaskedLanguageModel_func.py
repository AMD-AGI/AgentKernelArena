# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

def module_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Applies a linear transformation followed by log softmax along the last dimension.

    Args:
        x (torch.Tensor): Input tensor of shape (..., hidden)
        weight (torch.Tensor): Linear layer weight of shape (vocab_size, hidden)
        bias (torch.Tensor): Linear layer bias of shape (vocab_size)

    Returns:
        torch.Tensor: Output tensor of shape (..., vocab_size)
    """
    x = F.linear(x, weight, bias)
    x = F.log_softmax(x, dim=-1)
    return x

class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        linear = nn.Linear(hidden, vocab_size)
        self.weight = nn.Parameter(linear.weight)
        self.bias = nn.Parameter(linear.bias)

    def forward(self, x, fn=module_fn):
        return fn(x, self.weight, self.bias)

def get_inputs():
    """
    Generate multiple test cases with varying sizes
    HIP kernel requires 4D input [B, S1, S2, H] where H=hidden=4
    """
    configs = [
        # (B, S1, S2, H) - 4D tensors where H must be 4 to match hidden=4
        ([4, 4, 4, 4],),  # B=4, S1=4, S2=4, H=4
        ([8, 4, 4, 4],),  # B=8, S1=4, S2=4, H=4
        ([16, 4, 4, 4],),  # B=16, S1=4, S2=4, H=4
        ([32, 4, 4, 4],),  # B=32, S1=4, S2=4, H=4
        ([64, 4, 4, 4],),  # B=64, S1=4, S2=4, H=4
    ]
    
    for shape in configs:
        shape_list = shape[0] if isinstance(shape, tuple) and len(shape) == 1 else shape
        # Only yield x - weight and bias are model parameters
        x = torch.randn(shape_list, dtype=torch.float32)
        yield [x]


def get_init_inputs():
    return [[], {'hidden': 4, 'vocab_size': 4}]
