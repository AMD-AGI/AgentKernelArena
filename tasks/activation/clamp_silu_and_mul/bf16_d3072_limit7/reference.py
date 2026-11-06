"""High-precision clamped-SwiGLU reference used only for correctness."""
import torch

_CHUNK_ROWS = 32768


def reference(*, x, swiglu_limit):
    n_half = x.shape[1] // 2
    out = torch.empty((x.shape[0], n_half), dtype=torch.float64, device=x.device)
    limit = float(swiglu_limit)
    for start in range(0, x.shape[0], _CHUNK_ROWS):
        rows = x[start:start + _CHUNK_ROWS].double()
        gate, up = rows[:, :n_half], rows[:, n_half:]
        if limit > 0:
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
        out[start:start + _CHUNK_ROWS] = gate * torch.sigmoid(gate) * up
    return out
