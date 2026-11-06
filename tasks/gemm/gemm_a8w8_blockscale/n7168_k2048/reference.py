"""Float64 reference: out = dequant(a) @ dequant(b)^T for 1x128-scaled activations and 128x128-scaled weights.

Receives the logical operands: a_scale [m, k/128] indexed by (row, group), b in natural [n, k] order.
"""
import torch

_BLOCK = 128
_ROW_CHUNK = 8192


def reference(*, a, a_scale, b, b_scale):
    m, k = a.shape
    n = b.shape[0]
    # float8 values and float32 scales are exact in float64.
    b_dequant = b.float().double() * b_scale.double().repeat_interleave(_BLOCK, dim=0).repeat_interleave(_BLOCK, dim=1)
    out = torch.empty(m, n, dtype=torch.float64, device=a.device)
    for start in range(0, m, _ROW_CHUNK):
        rows = slice(start, min(start + _ROW_CHUNK, m))
        a_dequant = a[rows].float().double() * a_scale[rows].double().repeat_interleave(_BLOCK, dim=1)
        out[rows] = a_dequant @ b_dequant.T
    return out
