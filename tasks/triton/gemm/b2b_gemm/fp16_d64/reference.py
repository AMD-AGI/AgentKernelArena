"""High-precision back-to-back GEMM reference used only for correctness."""
import torch

_ROW_CHUNK = 2048


def reference(*, a, b, c):
    b64 = b.double()
    c64 = c.double()
    out = torch.empty(a.shape[0], c.shape[1], dtype=torch.float64, device=a.device)
    for start in range(0, a.shape[0], _ROW_CHUNK):
        rows = a[start:start + _ROW_CHUNK].double()
        out[start:start + _ROW_CHUNK] = (rows @ b64.T) @ c64
    return out
