"""Float64 KDA chunk output, used only for correctness.

For every chunk c of 64 tokens (the last may be partial), head and token i of the chunk:

    o_i = scale * sum_k q_ik * 2^g_ik * h_c[:, k] + sum_{j <= i in the chunk} A_ij * v_j
"""
import torch

CHUNKS_PER_STEP = 64  # chunks processed together, to bound float64 temporaries


def reference(*, q, v, g, h, A, scale):
    tokens, heads, _ = q.shape
    dim_v = v.shape[-1]
    chunk = A.shape[-1]
    n_chunks = h.shape[0]
    out = torch.empty(tokens, heads, dim_v, dtype=torch.float64, device=q.device)
    causal = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).tril()
    for c0 in range(0, n_chunks, CHUNKS_PER_STEP):
        nc = min(CHUNKS_PER_STEP, n_chunks - c0)
        t0, t1 = c0 * chunk, min((c0 + nc) * chunk, tokens)
        pad = nc * chunk - (t1 - t0)

        def chunks(x):
            x = x[t0:t1].double()
            if pad:
                x = torch.cat([x, x.new_zeros(pad, *x.shape[1:])])
            return x.view(nc, chunk, *x.shape[1:])

        qc, vc, gc, Ac = chunks(q), chunks(v), chunks(g), chunks(A)   # [nc, L, H, *]
        hc = h[c0:c0 + nc].double()                                   # [nc, H, V, K]
        inter = torch.einsum("cihk,chvk->cihv", qc * torch.exp2(gc), hc) * float(scale)
        Am = torch.where(causal[None, :, None, :], Ac, torch.zeros_like(Ac))
        intra = torch.einsum("cihj,cjhv->cihv", Am, vc)
        out[t0:t1] = (inter + intra).reshape(nc * chunk, heads, dim_v)[:t1 - t0]
    return out
