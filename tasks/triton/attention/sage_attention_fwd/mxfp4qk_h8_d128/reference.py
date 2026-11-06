"""float64 causal attention, used only for correctness (SageAttention approximates this)."""
import torch

_QUERY_CHUNK = 1024


def reference(*, q, k, v):
    batch, seqlen, heads, head_dim = q.shape
    scale = head_dim ** -0.5
    out = torch.empty(batch, seqlen, heads, head_dim, dtype=torch.float32, device=q.device)
    for b in range(batch):
        qb = q[b].permute(1, 0, 2).double()     # [h, s, d]
        kb = k[b].permute(1, 0, 2).double()
        vb = v[b].permute(1, 0, 2).double()
        for start in range(0, seqlen, _QUERY_CHUNK):
            stop = min(start + _QUERY_CHUNK, seqlen)
            keys = kb[:, :stop]
            scores = torch.einsum("hqd,hkd->hqk", qb[:, start:stop], keys) * scale
            rows = torch.arange(start, stop, device=q.device)[:, None]
            cols = torch.arange(stop, device=q.device)[None, :]
            scores = scores.masked_fill(cols > rows, float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            out[b, start:stop] = torch.einsum("hqk,hkd->hqd", weights, vb[:, :stop]).permute(1, 0, 2).float()
    return out
