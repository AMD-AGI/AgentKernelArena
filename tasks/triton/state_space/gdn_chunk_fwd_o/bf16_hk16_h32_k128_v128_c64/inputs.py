"""Chunk-local log-gates of the GDN chunk output task."""
import torch
import torch.nn.functional as F


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    n_tokens, heads, chunk = axes["n_tokens"], axes["num_heads"], axes["chunk_size"]
    # Per-token log-decay 0.03 * logsigmoid(x) (small, like -exp(A_log) * softplus(.) in the model),
    # summed within each chunk so the gate decays monotonically from the start of every chunk.
    raw = F.logsigmoid(torch.randn(n_tokens, heads, generator=generator, device=device)) * 0.03
    pad = (-n_tokens) % chunk
    if pad:
        raw = torch.cat([raw, raw.new_zeros(pad, heads)])
    g = raw.view(-1, chunk, heads).cumsum(dim=1).reshape(-1, heads)[:n_tokens]
    return {"g": g.reshape(1, n_tokens, heads).contiguous()}
