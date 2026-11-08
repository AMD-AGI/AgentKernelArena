"""High-precision causal GQA attention over dequantized FP8 inputs (float64), used only for correctness."""
import math

import torch

QUERY_CHUNK = 256


def reference(*, q, k, v, q_descale, k_descale, v_descale):
    batch, seq_len, n_heads, head_dim = q.shape
    n_kv_heads = k.shape[2]
    group = n_heads // n_kv_heads
    scale = 1.0 / math.sqrt(head_dim)
    q_scale = q_descale.double().reshape(())
    k_scale = k_descale.double().reshape(())
    v_scale = v_descale.double().reshape(())
    out = torch.empty(q.shape, dtype=torch.float64, device=q.device)
    for b in range(batch):
        keys = k[b].double() * k_scale
        values = v[b].double() * v_scale
        for start in range(0, seq_len, QUERY_CHUNK):
            stop = min(start + QUERY_CHUNK, seq_len)
            queries = (q[b, start:stop].double() * q_scale).reshape(stop - start, n_kv_heads, group, head_dim)
            scores = torch.einsum("qkgd,ckd->kgqc", queries, keys[:stop]) * scale
            rows = torch.arange(start, stop, device=q.device)[:, None]
            cols = torch.arange(stop, device=q.device)[None, :]
            scores = scores.masked_fill(cols > rows, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            out[b, start:stop] = torch.einsum("kgqc,ckd->qkgd", probs, values[:stop]).reshape(
                stop - start, n_heads, head_dim)
    return out
