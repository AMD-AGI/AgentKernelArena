# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def module_fn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_linear1: torch.Tensor,
    k_linear1: torch.Tensor,
    v_linear1: torch.Tensor,
    out_weight: torch.Tensor,
    out_bias: torch.Tensor,
    d_model: int,
    mask: torch.Tensor = None,
    dropout: callable = None,
    heads: int = 4,
) -> torch.Tensor:
    """
    Functional implementation of MultiHeadAttention forward pass.
    """
    bs = q.size(0)
    seq_len = q.size(1)
    # Dynamic head/d_k selection
    if seq_len > 230:
        h = 8
    elif seq_len <= 138 and seq_len > 230:
        h = 4
    elif seq_len <= 138 and seq_len > 0:
        h = 2
    else:
        h = heads
    d_k = d_model // h

    # Linear projections
    k_proj = torch.matmul(k, k_linear1)
    k_proj = k_proj.view(bs, -1, h, d_k)
    q_proj = torch.matmul(q, q_linear1)
    q_proj = q_proj.view(bs, -1, h, d_k)
    v_proj = torch.matmul(v, v_linear1)
    v_proj = v_proj.view(bs, -1, h, d_k)

    # Transpose for attention: (bs, h, seq_len, d_k)
    q_proj = q_proj.transpose(1, 2)
    k_proj = k_proj.transpose(1, 2)
    v_proj = v_proj.transpose(1, 2)

    # k: (bs, h, d_k, seq_len)
    k_proj = k_proj.transpose(-2, -1)

    # Attention scores
    scores = torch.matmul(q_proj, k_proj) / math.sqrt(d_k)

    # Masking
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1000000000.0)

    # Softmax
    scores = F.softmax(scores, dim=-1)

    # Dropout
    if dropout is not None:
        scores = dropout(scores)

    # Weighted sum
    attn = torch.matmul(scores, v_proj)

    # Concatenate heads
    concat = attn.transpose(1, 2).contiguous().view(bs, -1, d_model)

    # Output projection
    output = F.linear(concat, out_weight, out_bias)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.h = heads
        self.q_linear1 = nn.Parameter(torch.randn(d_model, d_model))
        self.v_linear1 = nn.Parameter(torch.randn(d_model, d_model))
        self.k_linear1 = nn.Parameter(torch.randn(d_model, d_model))
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None, dropout=None, fn=module_fn):
        return fn(
            q,
            k,
            v,
            self.q_linear1,
            self.k_linear1,
            self.v_linear1,
            self.out.weight,
            self.out.bias,
            self.d_model,
            mask,
            dropout,
            self.h,
        )


def get_inputs():
    """
    Generate multiple test cases for attention kernels
    d_model must match initialization (4)
    """
    configs = [
        # (batch, seq_len, d_model) - d_model must be 4 to match get_init_inputs()
        ([8, 16, 4], [8, 16, 4], [8, 16, 4]),
        ([16, 32, 4], [16, 32, 4], [16, 32, 4]),
        ([32, 64, 4], [32, 64, 4], [32, 64, 4]),
    ]
    
    for q_shape, k_shape, v_shape in configs:
        q = torch.randn(q_shape, dtype=torch.float32)
        k = torch.randn(k_shape, dtype=torch.float32)
        v = torch.randn(v_shape, dtype=torch.float32)
        yield [q, k, v]


def get_init_inputs():
    return [[], {'heads': 4, 'd_model': 4}]
