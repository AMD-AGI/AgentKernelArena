"""float64 ragged sparse attention with a per-head attention sink, used only for correctness."""
import torch

_CHUNK = 64


def reference(*, q, kv, indices, indptr, attn_sink, scale):
    n_queries, num_heads, head_dim = q.shape
    out = torch.zeros(n_queries, num_heads, head_dim, dtype=torch.float32, device=q.device)
    lengths = indptr[1:] - indptr[:-1]
    longest = int(lengths.max())
    if longest == 0:
        return out
    kv64 = kv.double()
    sink = attn_sink.double()[None, :, None]
    offsets = torch.arange(longest, device=q.device)
    for start in range(0, n_queries, _CHUNK):
        stop = min(start + _CHUNK, n_queries)
        begin = indptr[start:stop].long()
        position = (begin[:, None] + offsets[None, :]).clamp(max=indices.numel() - 1)
        slot = indices[position].long()
        valid = (offsets[None, :] < lengths[start:stop, None]) & (slot >= 0) & (slot < kv.shape[0])
        rows = kv64[slot.clamp(0, kv.shape[0] - 1)]                                 # [c, L, D]
        scores = torch.einsum("chd,cld->chl", q[start:stop].double(), rows) * float(scale)
        scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
        peak = torch.maximum(scores.amax(dim=-1, keepdim=True), sink)               # finite: the sink is
        weights = torch.exp(scores - peak)
        denominator = weights.sum(dim=-1, keepdim=True) + torch.exp(sink - peak)
        out[start:stop] = (torch.einsum("chl,cld->chd", weights, rows) / denominator).float()
    return out
