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
        # Unpack tuple if shape is a tuple containing a list (e.g., ([1024],) -> [1024])
        shape_list = shape[0] if isinstance(shape, tuple) and len(shape) == 1 else shape
        # Only yield x - weight and bias are model parameters
        x = torch.randn(shape_list, dtype=torch.float32)
        yield [x]


def get_init_inputs():
    return [[], {'hidden': 4, 'vocab_size': 4}]
