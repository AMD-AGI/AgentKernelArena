"""float64 reference for the cast-before-matmul GEMM, used only for correctness."""
import torch

_ROW_CHUNK = 8192


def reference(*, a, b):
    # The operator rounds A to float16 before the product; B is already float16.
    b64 = b.to(torch.float64)
    rows = []
    for start in range(0, a.shape[0], _ROW_CHUNK):
        a64 = a[start:start + _ROW_CHUNK].to(torch.float16).to(torch.float64)
        rows.append(a64 @ b64)
    return torch.cat(rows, dim=0)
