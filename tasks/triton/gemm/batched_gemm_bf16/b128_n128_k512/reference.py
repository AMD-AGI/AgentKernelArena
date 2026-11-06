"""Float64 batched GEMM reference used only for correctness."""
import torch

_BATCH_CHUNK = 16


def reference(*, x, w):
    out = torch.empty(x.shape[0], x.shape[1], w.shape[1], dtype=torch.float64, device=x.device)
    for start in range(0, x.shape[0], _BATCH_CHUNK):
        stop = start + _BATCH_CHUNK
        out[start:stop] = torch.bmm(x[start:stop].double(), w[start:stop].double().transpose(1, 2))
    return out
