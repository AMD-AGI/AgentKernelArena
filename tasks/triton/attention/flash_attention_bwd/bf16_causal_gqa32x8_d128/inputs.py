"""Forward results saved for the backward: o and lse of attention over the drawn q, k, v.

o is computed in float32 and rounded to bfloat16, lse is the float32 natural-log log-sum-exp of the
scaled (and, when causal, masked) scores, as the forward pass of a bfloat16 attention stores them.
"""
import torch

QUERY_CHUNK = 256
CAUSAL = True


def make_inputs(case, inputs, generator, device):
    q, k, v = inputs["q"], inputs["k"], inputs["v"]
    sm_scale = case["scalars"].get("sm_scale", 1.0 / q.shape[-1] ** 0.5)
    batch, seqlen, n_heads, head_dim = q.shape
    n_kv_heads = k.shape[2]
    group = n_heads // n_kv_heads
    o = torch.empty_like(q)
    lse = torch.empty((batch, n_heads, seqlen), dtype=torch.float32, device=q.device)
    for b in range(batch):
        keys = k[b].float()
        values = v[b].float()
        for q0 in range(0, seqlen, QUERY_CHUNK):
            q1 = min(q0 + QUERY_CHUNK, seqlen)
            k1 = q1 if CAUSAL else seqlen
            queries = q[b, q0:q1].float().view(q1 - q0, n_kv_heads, group, head_dim)
            scores = torch.einsum("ikgd,jkd->kgij", queries, keys[:k1]) * sm_scale
            if CAUSAL:
                rows = torch.arange(q0, q1, device=q.device)[:, None]
                cols = torch.arange(k1, device=q.device)[None, :]
                scores = scores.masked_fill(cols > rows, float("-inf"))
            row_lse = torch.logsumexp(scores, dim=-1)                   # [kv, g, i]
            probs = torch.exp(scores - row_lse[..., None])
            out = torch.einsum("kgij,jkd->ikgd", probs, values[:k1])
            o[b, q0:q1] = out.reshape(q1 - q0, n_heads, head_dim).to(q.dtype)
            lse[b, :, q0:q1] = row_lse.reshape(n_heads, q1 - q0)
    return {"o": o, "lse": lse}
