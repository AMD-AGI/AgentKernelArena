"""Float64 reference of the minGPT transformer block (embedding 768, 8 heads), for correctness only."""
import math

import torch

N_HEAD = 8


def _layer_norm(x, weight, bias, eps):
    mean = x.mean(dim=-1, keepdim=True)
    var = (x - mean).square().mean(dim=-1, keepdim=True)
    return (x - mean) * torch.rsqrt(var + eps) * weight.double() + bias.double()


def _new_gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def reference(*, x, ln1_weight, ln1_bias, c_attn_weight, c_attn_bias, attn_proj_weight, attn_proj_bias,
              ln2_weight, ln2_bias, c_fc_weight, c_fc_bias, mlp_proj_weight, mlp_proj_bias, ln_eps):
    """out = h + MLP(LN2(h)) with h = x + Attn(LN1(x)): minGPT's Block in eval mode (no dropout).

    Attn: q, k, v = split(a c_attn_weight^T + c_attn_bias) into 8 heads of 96; per head
    softmax(q k^T / sqrt(96)) over the keys j <= i; the heads side by side through attn_proj.
    MLP: c_fc, minGPT's tanh GELU (NewGELU), mlp_proj. LayerNorm uses the biased variance.
    """
    batch, seq_len, n_embd = x.shape
    head_dim = n_embd // N_HEAD
    x64 = x.double()
    causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).tril()

    a = _layer_norm(x64, ln1_weight, ln1_bias, ln_eps)
    qkv = a @ c_attn_weight.double().t() + c_attn_bias.double()
    del a
    y = torch.empty(batch, seq_len, n_embd, dtype=torch.float64, device=x.device)
    for b in range(batch):  # one sequence at a time keeps the score matrix small
        q, k, v = (t.reshape(seq_len, N_HEAD, head_dim).transpose(0, 1)
                   for t in qkv[b].split(n_embd, dim=-1))
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        att = torch.softmax(att.masked_fill(~causal, float("-inf")), dim=-1)
        y[b] = (att @ v).transpose(0, 1).reshape(seq_len, n_embd)
    del qkv
    h = x64 + (y @ attn_proj_weight.double().t() + attn_proj_bias.double())
    del y
    m = _new_gelu(_layer_norm(h, ln2_weight, ln2_bias, ln_eps) @ c_fc_weight.double().t() + c_fc_bias.double())
    return h + (m @ mlp_proj_weight.double().t() + mlp_proj_bias.double())
