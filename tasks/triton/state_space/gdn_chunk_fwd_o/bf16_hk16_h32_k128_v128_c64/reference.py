"""High-precision GDN chunk output reference used only for correctness."""
import torch

CHUNK_SIZE = 64
CHUNKS_PER_STEP = 64  # chunks processed together, to bound float64 temporaries


def reference(*, q, k, v, h, g, scale):
    B, T, Hg, K = q.shape
    H, V = v.shape[2], v.shape[3]
    NT = h.shape[1]
    rep = H // Hg
    out = torch.empty(B, T, H, V, dtype=torch.float64, device=q.device)
    causal = torch.ones(CHUNK_SIZE, CHUNK_SIZE, dtype=torch.bool, device=q.device).tril()
    for c0 in range(0, NT, CHUNKS_PER_STEP):
        nc = min(CHUNKS_PER_STEP, NT - c0)
        t0, t1 = c0 * CHUNK_SIZE, min((c0 + nc) * CHUNK_SIZE, T)
        pad = nc * CHUNK_SIZE - (t1 - t0)

        def chunks(x, heads_repeat=1):
            x = x[:, t0:t1].double()
            if pad:
                x = torch.cat([x, x.new_zeros(B, pad, *x.shape[2:])], dim=1)
            x = x.reshape(B, nc, CHUNK_SIZE, *x.shape[2:])
            if heads_repeat > 1:
                x = x.repeat_interleave(heads_repeat, dim=3)
            return x

        qc = chunks(q, rep).transpose(2, 3)              # [B, nc, H, L, K]
        kc = chunks(k, rep).transpose(2, 3)              # [B, nc, H, L, K]
        vc = chunks(v).transpose(2, 3)                   # [B, nc, H, L, V]
        gc = chunks(g).transpose(2, 3)                   # [B, nc, H, L]
        hc = h[:, c0:c0 + nc].double()                   # [B, nc, H, V, K]
        inter = (qc @ hc.transpose(-1, -2)) * torch.exp(gc)[..., None]
        diff = gc[..., :, None] - gc[..., None, :]
        decay = torch.where(diff <= 0, torch.exp(diff), torch.zeros_like(diff))
        attn = torch.where(causal, (qc @ kc.transpose(-1, -2)) * decay, torch.zeros_like(decay))
        o = (inter + attn @ vc) * float(scale)           # [B, nc, H, L, V]
        o = o.transpose(2, 3).reshape(B, nc * CHUNK_SIZE, H, V)
        out[:, t0:t1] = o[:, :t1 - t0]
    return out
