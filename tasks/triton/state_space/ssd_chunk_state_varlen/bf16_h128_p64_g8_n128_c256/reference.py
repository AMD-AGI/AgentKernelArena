"""High-precision Mamba2 varlen chunk state reference used only for correctness."""
import torch


def reference(*, x, B, dt, dA_cumsum, cu_seqlens, chunk_states, initial_states):
    heads = x.shape[1]
    groups = B.shape[1]
    ratio = heads // groups
    size = dt.shape[2]
    bounds = cu_seqlens.tolist()
    out = torch.empty(initial_states.shape, dtype=torch.float64, device=x.device)
    for b in range(len(bounds) - 1):
        start, end = bounds[b], bounds[b + 1]
        c = (end - 1) // size
        chunk_start = c * size
        limit = end - chunk_start
        first = max(start - chunk_start, 0)
        dA = dA_cumsum[:, c, :limit].double()                       # [H, limit]
        last = dA[:, limit - 1]                                     # [H]
        scale = torch.exp(last[:, None] - dA[:, first:]) * dt[:, c, first:limit].double()   # [H, K]
        xs = x[chunk_start + first:end].double()                    # [K, H, P]
        bs = B[chunk_start + first:end].double().repeat_interleave(ratio, dim=1)   # [K, H, N]
        acc = torch.einsum("khp,khn,hk->hpn", xs, bs, scale)
        if start < chunk_start:
            past = chunk_states[c].double()
            boundary = torch.zeros_like(last)
        else:
            past = initial_states[b].double()
            boundary = dA[:, start - chunk_start - 1] if start > chunk_start else torch.zeros_like(last)
        out[b] = acc + past * torch.exp(last - boundary)[:, None, None]
    return out
