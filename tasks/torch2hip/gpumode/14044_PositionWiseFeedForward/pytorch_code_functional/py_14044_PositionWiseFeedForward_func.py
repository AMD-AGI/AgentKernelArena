# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

def module_fn(
    x: torch.Tensor,
    W_1_weight: torch.Tensor,
    W_1_bias: torch.Tensor,
    W_2_weight: torch.Tensor,
    W_2_bias: torch.Tensor,
    layer_norm_weight: torch.Tensor,
    layer_norm_bias: torch.Tensor,
    dropout_p: float,
    training: bool = True,
) -> torch.Tensor:
    """
    Functional implementation of Position-Wise Feed-Forward Network.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, word_pad_len, d_model)
        W_1_weight (torch.Tensor): Weight for first linear layer (hidden_size, d_model)
        W_1_bias (torch.Tensor): Bias for first linear layer (hidden_size)
        W_2_weight (torch.Tensor): Weight for second linear layer (d_model, hidden_size)
        W_2_bias (torch.Tensor): Bias for second linear layer (d_model)
        layer_norm_weight (torch.Tensor): LayerNorm weight (d_model)
        layer_norm_bias (torch.Tensor): LayerNorm bias (d_model)
        dropout_p (float): Dropout probability

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, word_pad_len, d_model)
    """
    # First linear layer + ReLU
    out = F.linear(x, W_1_weight, W_1_bias)
    out = F.relu(out)
    # Second linear layer
    out = F.linear(out, W_2_weight, W_2_bias)
    # Dropout
    out = F.dropout(out, p=dropout_p, training=training)
    # Residual connection
    out = out + x
    # LayerNorm
    out = F.layer_norm(out, out.shape[-1:], layer_norm_weight, layer_norm_bias)
    return out

class PositionWiseFeedForward(nn.Module):
    """
    Position-Wise Feed-Forward Network
    """
    def __init__(self, d_model: int, hidden_size: int, dropout: float = 0.5):
        super(PositionWiseFeedForward, self).__init__()
        self.W_1 = nn.Linear(d_model, hidden_size)
        self.W_2 = nn.Linear(hidden_size, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_p = dropout

    def forward(self, x: torch.Tensor, fn=module_fn):
        if fn is module_fn:
            return fn(
                x,
                self.W_1.weight,
                self.W_1.bias,
                self.W_2.weight,
                self.W_2.bias,
                self.layer_norm.weight,
                self.layer_norm.bias,
                self.dropout_p,
                self.training,
            )
        effective_dropout = self.dropout_p if self.training else 0.0
        return fn(
            x,
            self.W_1.weight,
            self.W_1.bias,
            self.W_2.weight,
            self.W_2.bias,
            self.layer_norm.weight,
            self.layer_norm.bias,
            effective_dropout,
        )

def get_inputs():
    return [torch.rand([4, 4, 4, 4])]

def get_init_inputs():
    return [[], {'d_model': 4, 'hidden_size': 4}]
