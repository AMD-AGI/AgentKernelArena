"""Chunk-local cumulative per-channel decay gates of Kimi Delta Attention.

In production the gate ``g`` handed to the chunked kernels is the chunk-local cumulative sum of the
per-token, per-channel log-decay (non-positive), already multiplied by 1/ln(2) so the kernels can
use exp2. The generator draws non-positive per-token log2-decays in [-0.25, 0] per channel and
cumulates them inside consecutive 64-token chunks in float32, so ``g`` is non-increasing in every
chunk and ``2^(g[i] - g[j]) <= 1`` below the diagonal, as in production.
"""
import torch

CHUNK_SIZE = 64


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    tokens, heads, dim = axes["n_tokens"], axes["num_heads"], axes["head_dim"]
    g = torch.empty(tokens, heads, dim, dtype=torch.float32, device=device)
    rows = 8192 * CHUNK_SIZE  # whole chunks per step, to bound temporaries
    for start in range(0, tokens, rows):
        stop = min(start + rows, tokens)
        step = -0.25 * torch.rand(stop - start, heads, dim, generator=generator, device=device,
                                  dtype=torch.float32)
        pad = (-(stop - start)) % CHUNK_SIZE
        if pad:
            step = torch.cat([step, step.new_zeros(pad, heads, dim)])
        acc = step.view(-1, CHUNK_SIZE, heads, dim).cumsum(dim=1).view(-1, heads, dim)
        g[start:stop] = acc[: stop - start]
    return {"g": g}
