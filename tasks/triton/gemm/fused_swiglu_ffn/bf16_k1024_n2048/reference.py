"""High-precision float64 fused SwiGLU reference used only for correctness."""
import torch

_CHUNK_ROWS = 8192


def reference(*, x, w_gate, w_up):
    wg, wu = w_gate.double(), w_up.double()
    out = torch.empty((x.shape[0], wg.shape[1]), dtype=torch.float64, device=x.device)
    for start in range(0, x.shape[0], _CHUNK_ROWS):
        xc = x[start:start + _CHUNK_ROWS].double()
        gate = xc @ wg
        out[start:start + _CHUNK_ROWS] = gate * torch.sigmoid(gate) * (xc @ wu)
    return out
