"""High-precision reference for gelu_and_mul, computed in float64 in row chunks."""
import math

import torch

_CHUNK_ROWS = 16384


def _act(g):
    return 0.5 * g * (1.0 + torch.erf(g / math.sqrt(2.0)))


def reference(*, x):
    d = x.shape[1] // 2
    out = torch.empty(x.shape[0], d, dtype=torch.float64, device=x.device)
    for start in range(0, x.shape[0], _CHUNK_ROWS):
        rows = x[start:start + _CHUNK_ROWS].double()
        out[start:start + _CHUNK_ROWS] = _act(rows[:, :d]) * rows[:, d:]
    return out
