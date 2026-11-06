"""High-precision LLaMA feed-forward reference used only for correctness."""
import torch

_CHUNK = 1024


def reference(*, x, w1, w3, rms_w, epsilon):
    w1 = w1.double()
    w3 = w3.double()
    scale = rms_w.double()
    out = torch.empty(x.shape[0], w1.shape[0], dtype=torch.float64, device=x.device)
    for start in range(0, x.shape[0], _CHUNK):
        rows = x[start:start + _CHUNK].double()
        normed = rows * torch.rsqrt(rows.square().mean(dim=-1, keepdim=True) + float(epsilon)) * scale
        gate = normed @ w1.T
        out[start:start + _CHUNK] = gate * torch.sigmoid(gate) * (normed @ w3.T)
    return out
