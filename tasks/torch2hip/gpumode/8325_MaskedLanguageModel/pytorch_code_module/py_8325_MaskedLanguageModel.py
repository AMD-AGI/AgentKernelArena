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
