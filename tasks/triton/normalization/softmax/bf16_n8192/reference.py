"""Float64 row softmax reference."""
import torch

CHUNK_ROWS = 1 << 15


def reference(*, x):
    out = torch.empty(x.shape, dtype=torch.float64, device=x.device)
    for start in range(0, x.shape[0], CHUNK_ROWS):
        out[start:start + CHUNK_ROWS] = torch.softmax(x[start:start + CHUNK_ROWS].double(), dim=-1)
    return out
