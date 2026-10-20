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
    A 512-to-4096 vocabulary projection and log-softmax with 1/17/63/246/256
    token rows.
    """
    shapes = [
        [(1, 1, 1, 512)],
        [(1, 1, 17, 512)],
        [(3, 1, 21, 512)],
        [(2, 3, 41, 512)],
        [(4, 4, 16, 512)],
    ]
    for case_shapes in shapes:
        yield [torch.randn(shape, dtype=torch.float32) for shape in case_shapes]


def get_init_inputs():
    return [[], {'hidden': 512, 'vocab_size': 4096}]
