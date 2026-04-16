# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import math
import torch
import torch.nn.functional as F
import torch.nn as nn


class MultiHeadAttention(nn.Module):

    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear1 = nn.Parameter(torch.randn(d_model, d_model))
        self.v_linear1 = nn.Parameter(torch.randn(d_model, d_model))
        self.k_linear1 = nn.Parameter(torch.randn(d_model, d_model))
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None, dropout=None):
        bs = q.size(0)
        if q.size(1) > 230:
            self.h = 8
            self.d_k = self.d_model // self.h
        elif q.size(1) <= 138 and q.size(1) > 230:
            self.h = 4
            self.d_k = self.d_model // self.h
        elif q.size(1) <= 138 and q.size(1) > 0:
            self.h = 2
            self.d_k = self.d_model // self.h
        k = torch.matmul(k, self.k_linear1)
        k = k.view(bs, -1, self.h, self.d_k)
        q = torch.matmul(q, self.q_linear1)
        q = q.view(bs, -1, self.h, self.d_k)
        v = torch.matmul(v, self.v_linear1)
        v = v.view(bs, -1, self.h, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        k = k.transpose(-2, -1)
        scores = torch.matmul(q, k)
        scores = scores / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1000000000.0)
        scores = F.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        scores = torch.matmul(scores, v)
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output


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
