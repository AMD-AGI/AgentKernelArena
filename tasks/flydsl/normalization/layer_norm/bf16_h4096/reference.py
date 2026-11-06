"""High-precision LayerNorm reference used only for correctness."""
import torch

_CHUNK_ROWS = 8192
_EPSILON = 1.0e-5


def reference(*, x, weight, bias):
    n_rows = x.shape[0]
    out = torch.empty(x.shape, dtype=torch.float64, device=x.device)
    w64 = weight.double()
    b64 = bias.double()
    for start in range(0, n_rows, _CHUNK_ROWS):
        rows = slice(start, start + _CHUNK_ROWS)
        xr = x[rows].double()
        centered = xr - xr.mean(dim=-1, keepdim=True)
        var = centered.square().mean(dim=-1, keepdim=True)
        out[rows] = centered * torch.rsqrt(var + _EPSILON) * w64 + b64
    return out
