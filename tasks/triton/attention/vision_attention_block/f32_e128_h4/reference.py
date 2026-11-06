"""Float64 vision attention block (multi-head self-attention over image positions, residual
LayerNorm), used only for correctness."""
import math

import torch

NUM_HEADS = 4
SCORE_ELEMENTS = 1 << 26  # elements of one float64 score chunk (512 MiB)


def reference(*, x, in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias, norm_weight, norm_bias,
              epsilon):
    x64 = x.double()
    batch, embed_dim, hw = x64.shape
    head_dim = embed_dim // NUM_HEADS
    tokens = x64.transpose(1, 2)  # [batch, hw, embed_dim]: one row per image position
    qkv = tokens @ in_proj_weight.double().t() + in_proj_bias.double()
    q, k, v = (t.reshape(batch, hw, NUM_HEADS, head_dim).transpose(1, 2)
               for t in qkv.split(embed_dim, dim=-1))  # [batch, heads, hw, head_dim]
    scale = 1.0 / math.sqrt(head_dim)
    attention = torch.empty_like(q)
    rows = max(1, SCORE_ELEMENTS // (batch * NUM_HEADS * hw))
    for start in range(0, hw, rows):  # chunk the queries to bound the hw x hw score matrix
        scores = (q[:, :, start:start + rows] @ k.transpose(-1, -2)) * scale
        attention[:, :, start:start + rows] = torch.softmax(scores, dim=-1) @ v
    attention = attention.transpose(1, 2).reshape(batch, hw, embed_dim)
    z = attention @ out_proj_weight.double().t() + out_proj_bias.double() + tokens
    mean = z.mean(dim=-1, keepdim=True)
    centered = z - mean
    variance = centered.square().mean(dim=-1, keepdim=True)
    y = centered * torch.rsqrt(variance + epsilon) * norm_weight.double() + norm_bias.double()
    return y.transpose(1, 2).contiguous()  # [batch, embed_dim, hw]
