"""High-precision Mamba2 chunk scan reference used only for correctness."""
import torch


def reference(*, cb, x, dt, dA_cumsum, C, states, D, cu_chunk_seqlens, seq_idx):
    tokens, heads, head_dim = x.shape
    _, chunks, size = dt.shape
    groups = C.shape[1]
    ratio = heads // groups
    bounds = cu_chunk_seqlens.tolist()
    sequences = seq_idx.tolist()
    out = torch.empty(tokens, heads, head_dim, dtype=torch.float64, device=x.device)
    for c in range(chunks):
        start, end = bounds[c], bounds[c + 1]
        n = end - start
        xc = x[start:end].double()                                   # [n, H, P]
        dA = dA_cumsum[:, c, :n].double()                            # [H, n]
        step = dt[:, c, :n].double()                                 # [H, n]
        Cc = C[start:end].double().repeat_interleave(ratio, dim=1)   # [n, H, N]
        if c > 0 and sequences[c] == sequences[c - 1]:
            inter = torch.einsum("mhn,hpn->mhp", Cc, states[c - 1].double())
            inter = inter * torch.exp(dA).t()[:, :, None]
        else:
            inter = torch.zeros_like(xc)
        cbc = cb[c, :, :n, :n].double().repeat_interleave(ratio, dim=0)   # [H, n, n]
        causal = torch.ones(n, n, dtype=torch.bool, device=x.device).tril()
        decay = torch.where(causal, torch.exp(dA[:, :, None] - dA[:, None, :]), torch.zeros((), dtype=torch.float64,
                                                                                         device=x.device))
        weights = cbc * decay * step[:, None, :]
        intra = torch.einsum("hmk,khp->mhp", weights, xc)
        out[start:end] = inter + intra + D.double()[None, :, None] * xc
    return out
