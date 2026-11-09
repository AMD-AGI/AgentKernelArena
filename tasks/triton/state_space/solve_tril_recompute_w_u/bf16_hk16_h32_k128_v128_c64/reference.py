"""High-precision reference of w = Ai @ (beta * 2^g * k) and u = Ai @ (beta * v), Ai = (I + L)^-1.

L is the strict lower triangle of every 64 x 64 chunk block of A_raw (everything on and above the
diagonal is ignored, including the random values the generator puts in the blocks above the diagonal blocks); a partial last chunk inverts
only its valid rows and columns (the padding rows and columns are zero in the inputs, so they never
contribute)."""
import torch

CHUNK_SIZE = 64


def reference(*, k, v, beta, A_raw, g_cumsum):
    tokens, key_heads, key_dim = k.shape
    heads, value_dim = v.shape[1], v.shape[2]
    group = heads // key_heads
    w = torch.zeros(heads, tokens, key_dim, dtype=torch.float64, device=k.device)
    u = torch.zeros(heads, tokens, value_dim, dtype=torch.float64, device=k.device)
    step = 1024 * CHUNK_SIZE  # whole chunks per step, to bound float64 temporaries
    for start in range(0, tokens, step):
        stop = min(start + step, tokens)
        n = stop - start
        pad = (-n) % CHUNK_SIZE
        chunks = (n + pad) // CHUNK_SIZE
        for h in range(heads):
            kh = k[start:stop, h // group].double()
            vh = v[start:stop, h].double()
            bh = beta[start:stop, h].double()
            gh = g_cumsum[h, start:stop].double()
            blocks = A_raw[start:stop, h].double()  # [n, 64], row = token, column = chunk-local token
            if pad:
                kh = torch.cat([kh, kh.new_zeros(pad, key_dim)])
                vh = torch.cat([vh, vh.new_zeros(pad, value_dim)])
                bh = torch.cat([bh, bh.new_zeros(pad)])
                gh = torch.cat([gh, gh.new_zeros(pad)])
                blocks = torch.cat([blocks, blocks.new_zeros(pad, CHUNK_SIZE)])
            lower = torch.tril(blocks.view(chunks, CHUNK_SIZE, CHUNK_SIZE), diagonal=-1)
            eye = torch.eye(CHUNK_SIZE, dtype=torch.float64, device=k.device)
            inverse = torch.linalg.solve_triangular(eye + lower, eye.expand(chunks, -1, -1), upper=False)
            left_w = (kh * (bh * torch.exp2(gh))[:, None]).view(chunks, CHUNK_SIZE, key_dim)
            left_u = (vh * bh[:, None]).view(chunks, CHUNK_SIZE, value_dim)
            w[h, start:stop] = (inverse @ left_w).reshape(-1, key_dim)[:n]
            u[h, start:stop] = (inverse @ left_u).reshape(-1, value_dim)[:n]
    return w, u
