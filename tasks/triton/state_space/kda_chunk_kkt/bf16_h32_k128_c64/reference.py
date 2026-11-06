"""High-precision reference of the KDA gated intra-chunk K K^T and Q K^T, used only for correctness."""
import torch

CHUNK_SIZE = 64


def reference(*, q, k, g, beta, scale):
    tokens, heads, _ = k.shape
    A = torch.zeros(tokens, heads, CHUNK_SIZE, dtype=torch.float64, device=k.device)
    Aqk = torch.zeros_like(A)
    ones = torch.ones(CHUNK_SIZE, CHUNK_SIZE, dtype=torch.bool, device=k.device)
    strict = torch.tril(ones, diagonal=-1)
    lower = torch.tril(ones)
    rows = 1024 * CHUNK_SIZE  # whole chunks per step, to bound float64 temporaries
    for start in range(0, tokens, rows):
        stop = min(start + rows, tokens)
        n = stop - start
        pad = (-n) % CHUNK_SIZE
        for h in range(heads):
            qh, kh, gh = (x[start:stop, h].double() for x in (q, k, g))
            bh = beta[start:stop, h].double()
            if pad:
                qh, kh, gh = (torch.cat([x, x.new_zeros(pad, x.shape[-1])]) for x in (qh, kh, gh))
                bh = torch.cat([bh, bh.new_zeros(pad)])
            qc, kc, gc = (x.view(-1, CHUNK_SIZE, x.shape[-1]) for x in (qh, kh, gh))
            # sum_k x[i, k] * k[j, k] * 2^(g[i, k] - g[j, k]) = (x * 2^g) @ (k * 2^-g)^T
            right = (kc * torch.exp2(-gc)).transpose(1, 2)
            gram_k = (kc * torch.exp2(gc)) @ right
            gram_q = (qc * torch.exp2(gc)) @ right
            a = bh.view(-1, CHUNK_SIZE)[:, :, None] * gram_k
            aqk = float(scale) * gram_q
            a = torch.where(strict, a, torch.zeros_like(a))
            aqk = torch.where(lower, aqk, torch.zeros_like(aqk))
            A[start:stop, h] = a.reshape(-1, CHUNK_SIZE)[:n]
            Aqk[start:stop, h] = aqk.reshape(-1, CHUNK_SIZE)[:n]
    return {"A": A, "Aqk": Aqk}
