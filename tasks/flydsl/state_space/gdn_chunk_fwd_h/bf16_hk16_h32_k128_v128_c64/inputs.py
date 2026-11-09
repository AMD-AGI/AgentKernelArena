"""Chunk-local base-2 log-gates of the GDN chunk state task."""
import math

import torch
import torch.nn.functional as F


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    n_tokens, heads, chunk = axes["n_tokens"], axes["num_heads"], axes["chunk_size"]
    # Per-token log-decay 0.03 * logsigmoid(x) (small, like -exp(A_log) * softplus(.) in the model),
    # summed within each chunk so the gate decays monotonically from the start of every chunk. With
    # key standard deviation 128^-1/2 this keeps the delta-rule recurrence contractive over very long
    # sequences. The kernel takes the gate in base 2 (the production prefill scales it by log2(e)).
    raw = F.logsigmoid(torch.randn(n_tokens, heads, generator=generator, device=device)) * 0.03
    pad = (-n_tokens) % chunk
    if pad:
        raw = torch.cat([raw, raw.new_zeros(pad, heads)])
    g = raw.view(-1, chunk, heads).cumsum(dim=1).reshape(-1, heads)[:n_tokens] * math.log2(math.e)
    return {"g": g.t().reshape(1, heads, n_tokens).contiguous()}
