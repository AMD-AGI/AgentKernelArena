# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import torch
import torch.optim.lr_scheduler
import torch.nn as nn
import torch.optim
import torch.onnx.operators


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
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, fn=None):
        return self.softmax(self.linear(x))


def get_inputs():
    """
    Generate multiple test cases with varying sizes
    HIP kernel requires 4D input [B, S1, S2, H] where H=hidden (must match hidden).
    Sizes scaled up so Linear(hidden, vocab)+LogSoftmax is a real GEMM rather than
    launch-overhead-bound.
    """
    configs = [
        # (B, S1, S2, H) - H must match hidden=512; rows = B*S1*S2
        ([64, 4, 4, 512],),   # 1024 rows
        ([128, 4, 4, 512],),  # 2048 rows
        ([256, 4, 4, 512],),  # 4096 rows
        ([512, 4, 4, 512],),  # 8192 rows
    ]
    
    for shape in configs:
        # Unpack tuple if shape is a tuple containing a list (e.g., ([1024],) -> [1024])
        shape_list = shape[0] if isinstance(shape, tuple) and len(shape) == 1 else shape
        # Only yield x - weight and bias are model parameters
        x = torch.randn(shape_list, dtype=torch.float32)
        yield [x]


def get_init_inputs():
    return [[], {'hidden': 512, 'vocab_size': 4096}]
