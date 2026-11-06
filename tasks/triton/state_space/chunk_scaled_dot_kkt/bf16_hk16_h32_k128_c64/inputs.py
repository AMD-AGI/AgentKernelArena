"""Chunk-local cumulative decay gates of the Gated DeltaNet prefill.

In production `g` is the output of the chunk-local cumulative sum of the per-token log-decay
`-exp(A_log) * softplus(a + dt_bias)`, which is non-positive. The generator draws such per-token
log-decays (non-positive, between 0 and 0.2 in magnitude) and cumulates them within consecutive
chunks of 64 tokens in float32, so `g` is non-increasing inside every chunk, as in production.
"""
import torch

CHUNK_SIZE = 64


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    tokens, heads = axes["n_tokens"], axes["num_heads"]
    step = -torch.rand(tokens, heads, generator=generator, device=device, dtype=torch.float32) * 0.2
    pad = (-tokens) % CHUNK_SIZE
    if pad:
        step = torch.cat([step, step.new_zeros(pad, heads)])
    g = step.view(-1, CHUNK_SIZE, heads).cumsum(dim=1).view(-1, heads)[:tokens].contiguous()
    return {"g": g}
