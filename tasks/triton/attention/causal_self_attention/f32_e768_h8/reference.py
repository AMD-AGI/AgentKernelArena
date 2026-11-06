"""High-precision minGPT CausalSelfAttention forward (eval mode), for correctness only."""
import math

import torch

N_HEAD = 8
# Upper bound on the float64 score block (batch items x heads x queries x keys) held at once.
_SCORE_ELEMENTS = 1 << 27


def reference(*, x, c_attn_weight, c_attn_bias, c_proj_weight, c_proj_bias):
    batch, seq_len, n_embd = x.shape
    head_dim = n_embd // N_HEAD
    qkv = torch.matmul(x.double(), c_attn_weight.double().t()) + c_attn_bias.double()
    q, k, v = qkv.split(n_embd, dim=-1)

    def heads(t):  # [batch, seq_len, n_embd] -> [batch, heads, seq_len, head_dim]
        return t.reshape(batch, seq_len, N_HEAD, head_dim).transpose(1, 2)

    q, k, v = heads(q), heads(k), heads(v)
    visible = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).tril()
    y = torch.empty(batch, N_HEAD, seq_len, head_dim, dtype=torch.float64, device=x.device)
    step = max(1, _SCORE_ELEMENTS // (N_HEAD * seq_len * seq_len))
    for b0 in range(0, batch, step):
        b1 = min(batch, b0 + step)
        att = torch.matmul(q[b0:b1], k[b0:b1].transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        att = torch.softmax(att.masked_fill(~visible, float("-inf")), dim=-1)
        y[b0:b1] = torch.matmul(att, v[b0:b1])
    y = y.transpose(1, 2).reshape(batch, seq_len, n_embd)
    return torch.matmul(y, c_proj_weight.double().t()) + c_proj_bias.double()
