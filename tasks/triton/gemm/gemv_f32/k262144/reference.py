"""float64 reference for the fp32 matrix-vector product, used only for correctness."""
import torch

_ROW_CHUNK = 256


def reference(*, a, x):
    x64 = x.to(torch.float64)
    rows = [a[start:start + _ROW_CHUNK].to(torch.float64) @ x64 for start in range(0, a.shape[0], _ROW_CHUNK)]
    return torch.cat(rows, dim=0)
