"""High-precision GQA attention backward (float64), used only for correctness.

The gradients are those of out = softmax(sm_scale * q k^T, causal) v recomputed in float64 from q, k and v;
the saved o and lse inputs are not used, so a baseline that consumes them in bfloat16/float32 is judged
against the exact gradient.
"""
import torch

QUERY_CHUNK = 256
CAUSAL = False


def reference(*, q, k, v, o, do, lse, sm_scale):
    batch, seqlen, n_heads, head_dim = q.shape
    n_kv_heads = k.shape[2]
    group = n_heads // n_kv_heads
    dq = torch.empty(q.shape, dtype=torch.float64, device=q.device)
    dk = torch.zeros(k.shape, dtype=torch.float64, device=q.device)
    dv = torch.zeros(v.shape, dtype=torch.float64, device=q.device)
    for b in range(batch):
        keys = k[b].double()                      # [S, kv, d]
        values = v[b].double()
        for q0 in range(0, seqlen, QUERY_CHUNK):
            q1 = min(q0 + QUERY_CHUNK, seqlen)
            k1 = q1 if CAUSAL else seqlen
            queries = q[b, q0:q1].double().view(q1 - q0, n_kv_heads, group, head_dim)
            d_out = do[b, q0:q1].double().view(q1 - q0, n_kv_heads, group, head_dim)
            scores = torch.einsum("ikgd,jkd->kgij", queries, keys[:k1]) * sm_scale
            if CAUSAL:
                rows = torch.arange(q0, q1, device=q.device)[:, None]
                cols = torch.arange(k1, device=q.device)[None, :]
                scores = scores.masked_fill(cols > rows, float("-inf"))
            probs = torch.softmax(scores, dim=-1)                       # [kv, g, i, j]
            out = torch.einsum("kgij,jkd->ikgd", probs, values[:k1])
            d_probs = torch.einsum("ikgd,jkd->kgij", d_out, values[:k1])
            delta = (d_out * out).sum(-1).permute(1, 2, 0)              # [kv, g, i]
            d_scores = probs * (d_probs - delta[..., None])
            dq[b, q0:q1] = (torch.einsum("kgij,jkd->ikgd", d_scores, keys[:k1]) * sm_scale).reshape(
                q1 - q0, n_heads, head_dim)
            dk[b, :k1] += torch.einsum("kgij,ikgd->jkd", d_scores, queries) * sm_scale
            dv[b, :k1] += torch.einsum("kgij,ikgd->jkd", probs, d_out)
    return dq, dk, dv
