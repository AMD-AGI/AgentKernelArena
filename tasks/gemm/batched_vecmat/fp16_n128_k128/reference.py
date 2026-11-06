"""Float64 batched vector-matrix product used only for correctness."""
import torch

_ROW_CHUNK = 8192


def reference(*, a, b):
    out = torch.empty(a.shape[0], b.shape[1], dtype=torch.float64, device=a.device)
    for start in range(0, a.shape[0], _ROW_CHUNK):
        stop = start + _ROW_CHUNK
        out[start:stop] = (a[start:stop].double()[:, None, :] * b[start:stop].double()).sum(dim=2)
    return out
