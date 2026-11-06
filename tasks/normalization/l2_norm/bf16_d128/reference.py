"""Float64 reference of L2 normalization over the last dimension (no mean subtraction)."""
import torch

CHUNK_ROWS = 1 << 20


def reference(*, x, epsilon):
    out = torch.empty(x.shape, dtype=torch.float64, device=x.device)
    for start in range(0, x.shape[0], CHUNK_ROWS):
        block = x[start:start + CHUNK_ROWS].double()
        out[start:start + CHUNK_ROWS] = block * torch.rsqrt(block.square().sum(dim=-1, keepdim=True) + epsilon)
    return out
