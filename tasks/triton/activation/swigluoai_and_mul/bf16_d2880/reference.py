"""High-precision GPT-OSS clamped SwiGLU reference used only for correctness."""
import torch

_ALPHA = 1.702
_LIMIT = 7.0
_CHUNK_ROWS = 16384


def reference(*, x):
    d = x.shape[-1] // 2
    out = torch.empty((x.shape[0], d), dtype=torch.float64, device=x.device)
    for start in range(0, x.shape[0], _CHUNK_ROWS):
        rows = x[start:start + _CHUNK_ROWS].double()
        gate = torch.clamp(rows[:, :d], max=_LIMIT)
        up = torch.clamp(rows[:, d:], min=-_LIMIT, max=_LIMIT)
        out[start:start + _CHUNK_ROWS] = gate * torch.sigmoid(_ALPHA * gate) * (up + 1.0)
    return out
