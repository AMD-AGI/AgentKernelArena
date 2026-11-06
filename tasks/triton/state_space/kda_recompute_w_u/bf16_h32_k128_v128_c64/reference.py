"""High-precision reference of the KDA recompute of w, u and the decayed keys, for correctness only."""
import torch

CHUNK_SIZE = 64


def reference(*, k, v, beta, A, gk):
    tokens, heads, key_dim = k.shape
    value_dim = v.shape[-1]
    w = torch.empty(tokens, heads, key_dim, dtype=torch.float64, device=k.device)
    u = torch.empty(tokens, heads, value_dim, dtype=torch.float64, device=k.device)
    kg = torch.empty_like(w)
    rows = 1024 * CHUNK_SIZE  # whole chunks per step, to bound float64 temporaries
    for start in range(0, tokens, rows):
        stop = min(start + rows, tokens)
        n = stop - start
        pad = (-n) % CHUNK_SIZE
        last = CHUNK_SIZE - 1 - pad  # index of the last valid token in the final chunk
        for h in range(heads):
            kh, vh, gh = (x[start:stop, h].double() for x in (k, v, gk))
            bh = beta[start:stop, h].double()
            ah = A[start:stop, h].double()
            if pad:
                kh, vh, gh, ah = (torch.cat([x, x.new_zeros(pad, x.shape[-1])]) for x in (kh, vh, gh, ah))
                bh = torch.cat([bh, bh.new_zeros(pad)])
            kc, vc, gc = (x.view(-1, CHUNK_SIZE, x.shape[-1]) for x in (kh, vh, gh))
            ac = ah.view(-1, CHUNK_SIZE, CHUNK_SIZE)
            bc = bh.view(-1, CHUNK_SIZE)[:, :, None]
            u[start:stop, h] = (ac @ (bc * vc)).reshape(-1, value_dim)[:n]
            w[start:stop, h] = (ac @ (bc * kc * torch.exp2(gc))).reshape(-1, key_dim)[:n]
            ends = torch.full((kc.shape[0],), CHUNK_SIZE - 1, dtype=torch.long, device=k.device)
            if pad:
                ends[-1] = last
            g_end = gc[torch.arange(kc.shape[0], device=k.device), ends][:, None, :]
            kg[start:stop, h] = (kc * torch.exp2(g_end - gc)).reshape(-1, key_dim)[:n]
    return {"w": w, "u": u, "kg": kg}
