"""float64 ragged sparse attention over MLA-latent rows, used only for correctness."""
import torch

_CHUNK = 128


def reference(*, q, kv, indices, indptr, scale):
    n_queries, num_heads, head_dim = q.shape
    out = torch.zeros(n_queries, num_heads, head_dim, dtype=torch.float32, device=q.device)
    lengths = indptr[1:] - indptr[:-1]
    longest = int(lengths.max())
    if longest == 0:
        return out
    kv64 = kv.double()
    offsets = torch.arange(longest, device=q.device)
    for start in range(0, n_queries, _CHUNK):
        stop = min(start + _CHUNK, n_queries)
        begin = indptr[start:stop].long()
        valid = offsets[None, :] < lengths[start:stop, None]
        position = (begin[:, None] + offsets[None, :]).clamp(max=indices.numel() - 1)
        slot = indices[position].long().clamp(0, kv.shape[0] - 1)
        rows = kv64[slot]                                              # [c, L, D]
        scores = torch.einsum("chd,cld->chl", q[start:stop].double(), rows) * float(scale)
        scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
        peak = scores.amax(dim=-1, keepdim=True)
        peak = torch.where(torch.isfinite(peak), peak, torch.zeros_like(peak))
        weights = torch.exp(scores - peak)
        # A non-empty row's sum includes exp(0) = 1, so it is >= 1; an empty row's weights (and
        # numerator) are all zero. Dividing that zero by 1 yields the empty row's zeros in any
        # floating dtype (a tiny clamp such as 1e-300 underflows to 0 in float32 and gives 0/0).
        denominator = weights.sum(dim=-1, keepdim=True)
        denominator = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
        out[start:stop] = (torch.einsum("chl,cld->chd", weights, rows) / denominator).float()
    return out
