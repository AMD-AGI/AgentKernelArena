# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import torch
from torch import nn


class PositionWiseFeedForward(nn.Module):
    """
    Position-Wise Feed-Forward Network

    Parameters
    ----------
    d_model : int
        Size of word embeddings

    hidden_size : int
        Size of position-wise feed forward network

    dropout : float
        Dropout
    """

    def __init__(self, d_model: 'int', hidden_size: 'int', dropout: 'float'=0.5
        ) ->None:
        super(PositionWiseFeedForward, self).__init__()
        self.W_1 = nn.Linear(d_model, hidden_size)
        self.W_2 = nn.Linear(hidden_size, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: 'torch.Tensor', fn=None) ->torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor (batch_size, word_pad_len, d_model)
            Output of multi-head self-attention network

        Returns
        -------
        out : torch.Tensor (batch_size, word_pad_len, d_model)
            Output of position-wise feed-forward network
        """
        out = self.W_2(self.relu(self.W_1(x)))
        out = self.dropout(out)
        out += x
        out = self.layer_norm(out)
        return out


def get_inputs():
    """
    Generate multiple test cases for PositionWiseFeedForward covering:
    - Different batch sizes and sequence lengths
    - Input shape: (batch_size, word_pad_len, d_model)
    """
    # d_model scaled up so W_1/W_2 are real GEMMs and the LayerNorm reduction is
    # non-trivial; M = batch * word_pad_len.
    configs = [
        ([32, 32, 512],),    # 1024 rows
        ([64, 64, 512],),    # 4096 rows
        ([128, 64, 512],),   # 8192 rows
        ([128, 128, 512],),  # 16384 rows
    ]
    
    for shape in configs:
        shape_list = shape[0] if isinstance(shape, tuple) and len(shape) == 1 else shape
        x = torch.randn(shape_list, dtype=torch.float32)
        yield [x]


def get_init_inputs():
    return [[], {'d_model': 512, 'hidden_size': 2048}]
