"""High-precision reference of the Gated DeltaNet WY recompute (w and u), used only for correctness."""
import torch

CHUNK_SIZE = 64
CHUNKS_PER_STEP = 256  # whole chunks per step, to bound float64 temporaries


def reference(*, k, v, beta, g, A):
    tokens, key_heads, key_dim = k.shape
    heads, value_dim = v.shape[1], v.shape[2]
    group = heads // key_heads
    w = torch.empty(tokens, heads, key_dim, dtype=torch.float64, device=k.device)
    u = torch.empty(tokens, heads, value_dim, dtype=torch.float64, device=k.device)
    rows = CHUNKS_PER_STEP * CHUNK_SIZE
    for start in range(0, tokens, rows):
        stop = min(start + rows, tokens)
        n = stop - start
        pad = (-n) % CHUNK_SIZE
        chunks = (n + pad) // CHUNK_SIZE

        def chunked(x):
            x = x[start:stop].double()
            if pad:
                x = torch.cat([x, x.new_zeros(pad, *x.shape[1:])])
            return x.view(chunks, CHUNK_SIZE, *x.shape[1:])

        a = chunked(A)                                    # [c, 64, H, 64]
        b = chunked(beta)                                 # [c, 64, H]
        gate = torch.exp(chunked(g))                      # [c, 64, H]
        kb = chunked(k).repeat_interleave(group, dim=2) * (b * gate)[..., None]
        vb = chunked(v) * b[..., None]
        w[start:stop] = torch.einsum("cihj,cjhk->cihk", a, kb).reshape(-1, heads, key_dim)[:n]
        u[start:stop] = torch.einsum("cihj,cjhv->cihv", a, vb).reshape(-1, heads, value_dim)[:n]
    return w, u
