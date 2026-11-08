"""High-precision GPT-OSS clamped SwiGLU reference (block-interleaved input) used only for correctness."""
import torch

_ALPHA = 1.702
_LIMIT = 7.0
_LANES = 16
_CHUNK_ROWS = 16384


def reference(*, x):
    n_rows, two_d = x.shape
    d = two_d // 2
    out = torch.empty((n_rows, d), dtype=torch.float64, device=x.device)
    for start in range(0, n_rows, _CHUNK_ROWS):
        rows = x[start:start + _CHUNK_ROWS].double().view(-1, d // _LANES, 2, _LANES)
        gate = torch.clamp(rows[:, :, 0].reshape(-1, d), max=_LIMIT)
        up = torch.clamp(rows[:, :, 1].reshape(-1, d), min=-_LIMIT, max=_LIMIT)
        out[start:start + _CHUNK_ROWS] = gate * torch.sigmoid(_ALPHA * gate) * (up + 1.0)
    return out
