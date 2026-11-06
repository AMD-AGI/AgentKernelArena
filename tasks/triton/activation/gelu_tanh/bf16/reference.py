"""High-precision tanh-approximate GELU reference, float64 in chunks."""
import math

import torch

_CHUNK = 1 << 26


def reference(*, x):
    out = torch.empty(x.shape, dtype=torch.float64, device=x.device)
    for start in range(0, x.numel(), _CHUNK):
        v = x[start:start + _CHUNK].double()
        out[start:start + _CHUNK] = 0.5 * v * (
            1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (v + 0.044715 * v * v * v)))
    return out
