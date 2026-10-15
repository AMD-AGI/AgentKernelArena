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
    Residual ReLU FFN and LayerNorm at 512/2048 widths, from one token through
    1024 tokens with row tails.
    """
    shapes = [
        [(1, 1, 512)],
        [(1, 17, 512)],
        [(2, 31, 512)],
        [(3, 43, 512)],
        [(4, 64, 512)],
        [(1, 257, 512)],
        [(5, 63, 512)],
        [(8, 64, 512)],
        [(4, 129, 512)],
        [(8, 128, 512)],
    ]
    for case_shapes in shapes:
        yield [torch.randn(shape, dtype=torch.float32) for shape in case_shapes]


def get_init_inputs():
    return [[], {'d_model': 512, 'hidden_size': 2048}]
