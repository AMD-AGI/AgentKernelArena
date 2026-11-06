"""High-precision chunk-local cumulative sum reference used only for correctness."""
import torch

CHUNK_SIZE = 64


def reference(*, g):
    T, H, S = g.shape
    out = torch.empty(T, H, S, dtype=torch.float64, device=g.device)
    rows = 1024 * CHUNK_SIZE  # whole chunks per step, to bound float64 temporaries
    for start in range(0, T, rows):
        part = g[start:start + rows].double()
        n = part.shape[0]
        pad = (-n) % CHUNK_SIZE
        if pad:
            part = torch.cat([part, part.new_zeros(pad, H, S)])
        summed = part.view(-1, CHUNK_SIZE, H, S).cumsum(dim=1).view(-1, H, S)
        out[start:start + n] = summed[:n]
    return out
