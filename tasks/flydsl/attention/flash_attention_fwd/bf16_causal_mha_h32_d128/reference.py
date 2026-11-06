"""High-precision causal attention reference (float64), used only for correctness."""
import math

import torch

QUERY_CHUNK = 256


def reference(*, q, k, v):
    batch, seq_len, n_heads, head_dim = q.shape
    scale = 1.0 / math.sqrt(head_dim)
    out = torch.empty(q.shape, dtype=torch.float64, device=q.device)
    for b in range(batch):
        keys = k[b].double()
        values = v[b].double()
        for start in range(0, seq_len, QUERY_CHUNK):
            stop = min(start + QUERY_CHUNK, seq_len)
            queries = q[b, start:stop].double()
            scores = torch.einsum("qhd,khd->hqk", queries, keys[:stop]) * scale
            rows = torch.arange(start, stop, device=q.device)[:, None]
            cols = torch.arange(stop, device=q.device)[None, :]
            scores = scores.masked_fill(cols > rows, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            out[b, start:stop] = torch.einsum("hqk,khd->qhd", probs, values[:stop])
    return out
