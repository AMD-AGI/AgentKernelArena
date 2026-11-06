"""High-precision exact GELU reference used only for correctness."""
import math

import torch

_CHUNK = 1 << 26


def reference(*, x):
    flat = x.reshape(-1)
    out = torch.empty(flat.shape, dtype=torch.float64, device=x.device)
    for start in range(0, flat.numel(), _CHUNK):
        v = flat[start:start + _CHUNK].double()
        out[start:start + _CHUNK] = 0.5 * v * (1.0 + torch.erf(v / math.sqrt(2.0)))
    return out
