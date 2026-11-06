"""High-precision reference of the gated, beta-scaled intra-chunk K K^T, used only for correctness."""
import torch

CHUNK_SIZE = 64


def reference(*, k, beta, g):
    tokens, key_heads, _ = k.shape
    heads = beta.shape[-1]
    group = heads // key_heads
    out = torch.zeros(tokens, heads, CHUNK_SIZE, dtype=torch.float64, device=k.device)
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
            gram = kc @ kc.transpose(1, 2)
            gc = gh.view(-1, CHUNK_SIZE)
            decay = torch.exp(gc[:, :, None] - gc[:, None, :])
            block = bh.view(-1, CHUNK_SIZE)[:, :, None] * gram * decay
            block = torch.where(lower, block, torch.zeros_like(block))
            out[start:stop, h] = block.reshape(-1, CHUNK_SIZE)[:n]
    return out
