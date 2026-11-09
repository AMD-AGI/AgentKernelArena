"""High-precision causal sliding-window attention reference (float64), used only for correctness."""
import math

import torch

QUERY_CHUNK = 256


def reference(*, q, k, v, window_left):
    batch, seq_len, n_heads, head_dim = q.shape
    n_kv_heads = k.shape[2]
    group = n_heads // n_kv_heads
    scale = 1.0 / math.sqrt(head_dim)
    out = torch.empty(q.shape, dtype=torch.float64, device=q.device)
    for b in range(batch):
        keys = k[b].double()
        values = v[b].double()
        for start in range(0, seq_len, QUERY_CHUNK):
            stop = min(start + QUERY_CHUNK, seq_len)
            first = max(0, start - window_left)
            queries = q[b, start:stop].double().reshape(stop - start, n_kv_heads, group, head_dim)
            scores = torch.einsum("qkgd,ckd->kgqc", queries, keys[first:stop]) * scale
            rows = torch.arange(start, stop, device=q.device)[:, None]
            cols = torch.arange(first, stop, device=q.device)[None, :]
            keep = (cols <= rows) & (cols >= rows - window_left)
            scores = scores.masked_fill(~keep, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            res = torch.einsum("kgqc,ckd->qkgd", probs, values[first:stop])
            out[b, start:stop] = res.reshape(stop - start, n_heads, head_dim)
    return out
