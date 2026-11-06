"""Float64 reference of the gated RMSNorm: rmsnorm(x) * weight, then multiplied by silu(z)."""
import torch

CHUNK_ROWS = 1 << 18


def reference(*, x, z, weight, epsilon):
    out = torch.empty(x.shape, dtype=torch.float64, device=x.device)
    w = weight.double()
    for start in range(0, x.shape[0], CHUNK_ROWS):
        x64 = x[start:start + CHUNK_ROWS].double()
        z64 = z[start:start + CHUNK_ROWS].double()
        normed = x64 * torch.rsqrt(x64.square().mean(dim=-1, keepdim=True) + epsilon) * w
        out[start:start + CHUNK_ROWS] = normed * (z64 * torch.sigmoid(z64))
    return out
