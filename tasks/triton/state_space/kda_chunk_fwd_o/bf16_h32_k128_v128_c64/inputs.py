"""Gates and intra-chunk attention matrix of the KDA chunk output task.

g: the chunk-local cumulative log2-decay that the fused KDA gate produces. Per token and key
channel the natural-log decay is lower_bound * sigmoid(exp(A_log) * (x + dt_bias)) with the
safe-gate bound -5, x ~ N(0, 1) and per-case A_log ~ U(0, 1.5) per head, dt_bias ~ U(-8, 0) per
channel, which spreads the channels from no decay to fast decay. It is divided by ln 2 and summed
within each 64-token chunk, so g is non-positive and non-increasing inside every chunk.

A: the intra-chunk attention matrix Aqk. Entries on and below the diagonal (j <= i inside the
chunk) are N(0, 0.01^2), about the size of scale * q_i . k_j * 2^(g_i - g_j) for L2-normalized q
and k. The producer never writes most of the part above the diagonal, so those entries are
arbitrary: here U(-1, 1), and the operator must ignore them.
"""
import math

import torch

LOWER_BOUND = -5.0


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    tokens, heads = axes["n_tokens"], axes["num_heads"]
    dim, chunk = axes["head_dim_k"], axes["chunk_size"]
    a = torch.exp(torch.rand(heads, 1, generator=generator, device=device) * 1.5)
    bias = torch.rand(heads, dim, generator=generator, device=device) * -8.0
    x = torch.randn(tokens, heads, dim, generator=generator, device=device)
    step = (LOWER_BOUND / math.log(2.0)) * torch.sigmoid(a * (x + bias))
    del x
    pad = (-tokens) % chunk
    if pad:
        step = torch.cat([step, step.new_zeros(pad, heads, dim)])
    g = step.view(-1, chunk, heads, dim).cumsum(dim=1).view(-1, heads, dim)[:tokens].contiguous()
    del step
    lower = torch.randn(tokens, heads, chunk, generator=generator, device=device) * 0.01
    garbage = torch.rand(tokens, heads, chunk, generator=generator, device=device) * 2.0 - 1.0
    row = torch.arange(tokens, device=device) % chunk
    above = torch.arange(chunk, device=device)[None, :] > row[:, None]
    A = torch.where(above[:, None, :], garbage, lower).to(torch.bfloat16)
    return {"g": g, "A": A}
