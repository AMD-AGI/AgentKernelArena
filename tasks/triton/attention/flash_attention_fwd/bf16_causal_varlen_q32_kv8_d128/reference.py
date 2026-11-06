"""High-precision causal varlen GQA prefill attention (float64), used only for correctness."""
import torch

QUERY_CHUNK = 256


def reference(*, q, k, v, b_start_loc, b_seq_len, max_input_len, sm_scale):
    n_tokens, n_heads, head_dim = q.shape
    n_kv_heads = k.shape[1]
    group = n_heads // n_kv_heads
    out = torch.zeros(q.shape, dtype=torch.float64, device=q.device)
    for start, length in zip(b_start_loc.tolist(), b_seq_len.tolist()):
        keys = k[start:start + length].double()          # [L, kv_heads, d]
        values = v[start:start + length].double()
        for q0 in range(0, length, QUERY_CHUNK):
            q1 = min(q0 + QUERY_CHUNK, length)
            # Query head h = kv_head * group + g reads KV head h // group.
            queries = q[start + q0:start + q1].double().view(q1 - q0, n_kv_heads, group, head_dim)
            scores = torch.einsum("ikgd,jkd->kgij", queries, keys[:q1]) * sm_scale
            rows = torch.arange(q0, q1, device=q.device)[:, None]
            cols = torch.arange(q1, device=q.device)[None, :]
            scores = scores.masked_fill(cols > rows, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            chunk = torch.einsum("kgij,jkd->ikgd", probs, values[:q1])
            out[start + q0:start + q1] = chunk.reshape(q1 - q0, n_heads, head_dim)
    return out
