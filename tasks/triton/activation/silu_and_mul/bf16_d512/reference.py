import torch

_CHUNK_ROWS = 8192


def reference(*, x):
    d = x.shape[1] // 2
    out = torch.empty((x.shape[0], d), dtype=torch.float64, device=x.device)
    for start in range(0, x.shape[0], _CHUNK_ROWS):
        rows = x[start:start + _CHUNK_ROWS].double()
        gate, up = rows[:, :d], rows[:, d:]
        out[start:start + _CHUNK_ROWS] = gate * torch.sigmoid(gate) * up
    return out
