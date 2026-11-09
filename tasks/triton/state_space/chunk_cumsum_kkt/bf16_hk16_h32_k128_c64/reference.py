"""High-precision reference of the fused chunk-local gate cumsum and gated, beta-scaled K K^T."""
import math

import torch

CHUNK_SIZE = 64
LOG2_E = math.log2(math.e)


def reference(*, k, beta, g):
    tokens, key_heads, _ = k.shape
    heads = beta.shape[-1]
    group = heads // key_heads
    out_g = torch.zeros(heads, tokens, dtype=torch.float64, device=k.device)
    out_a = torch.zeros(tokens, heads, CHUNK_SIZE, dtype=torch.float64, device=k.device)
    lower = torch.tril(torch.ones(CHUNK_SIZE, CHUNK_SIZE, dtype=torch.bool, device=k.device), diagonal=-1)
    rows = 1024 * CHUNK_SIZE  # whole chunks per step, to bound float64 temporaries
    for start in range(0, tokens, rows):
        stop = min(start + rows, tokens)
        n = stop - start
        pad = (-n) % CHUNK_SIZE
        for h in range(heads):
            kh = k[start:stop, h // group].double()
            bh = beta[start:stop, h].double()
            gh = g[start:stop, h].double()
            if pad:
                kh = torch.cat([kh, kh.new_zeros(pad, kh.shape[-1])])
                bh = torch.cat([bh, bh.new_zeros(pad)])
                gh = torch.cat([gh, gh.new_zeros(pad)])
            kc = kh.view(-1, CHUNK_SIZE, kh.shape[-1])
            gc = gh.view(-1, CHUNK_SIZE).cumsum(dim=1)  # chunk-local cumulative log-decay
            out_g[h, start:stop] = (gc * LOG2_E).reshape(-1)[:n]
            gram = kc @ kc.transpose(1, 2)
            decay = torch.exp(gc[:, :, None] - gc[:, None, :])
            block = bh.view(-1, CHUNK_SIZE)[:, :, None] * gram * decay
            block = torch.where(lower, block, torch.zeros_like(block))
            out_a[start:stop, h] = block.reshape(-1, CHUNK_SIZE)[:n]
    return out_g, out_a
