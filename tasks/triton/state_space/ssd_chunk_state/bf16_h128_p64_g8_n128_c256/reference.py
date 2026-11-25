"""High-precision Mamba2 chunk state reference used only for correctness."""
import torch


def reference(*, x, B, dt, dA_cumsum, cu_chunk_seqlens):
    heads, head_dim = x.shape[1], x.shape[2]
    _, chunks, size = dt.shape
    groups, state_dim = B.shape[1], B.shape[2]
    ratio = heads // groups
    bounds = cu_chunk_seqlens.tolist()
    states = torch.empty(chunks, heads, head_dim, state_dim, dtype=torch.float64, device=x.device)
    for c in range(chunks):
        start, end = bounds[c], bounds[c + 1]
        n = end - start
        dA = dA_cumsum[:, c].double()                                 # [H, size]
        scale = torch.exp(dA[:, size - 1:size] - dA[:, :n]) * dt[:, c, :n].double()   # [H, n]
        Bc = B[start:end].double().repeat_interleave(ratio, dim=1)    # [n, H, N]
        states[c] = torch.einsum("khp,khn->hpn", x[start:end].double(), Bc * scale.t()[:, :, None])
    return states
