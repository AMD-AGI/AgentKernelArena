"""High-precision swiglustep reference used only for correctness."""
import torch

_CHUNK_ROWS = 16384


def reference(*, x, limit):
    d = x.shape[-1] // 2
    out = torch.empty((x.shape[0], d), dtype=torch.float64, device=x.device)
    for start in range(0, x.shape[0], _CHUNK_ROWS):
        rows = x[start:start + _CHUNK_ROWS].double()
        gate, up = rows[:, :d], rows[:, d:]
        silu = gate * torch.sigmoid(gate)
        out[start:start + _CHUNK_ROWS] = torch.clamp(silu, max=float(limit)) * torch.clamp(
            up, min=-float(limit), max=float(limit))
    return out
