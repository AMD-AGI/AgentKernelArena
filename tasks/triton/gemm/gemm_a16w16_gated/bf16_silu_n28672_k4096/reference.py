"""Float64 gated GEMM reference used only for correctness: silu(x @ w_gate^T) * (x @ w_up^T)."""
import torch

_ROW_CHUNK = 1024


def reference(*, x, w):
    n_out = w.shape[0] // 2
    w64 = w.double()
    out = torch.empty(x.shape[0], n_out, dtype=torch.float64, device=x.device)
    for start in range(0, x.shape[0], _ROW_CHUNK):
        pre = x[start:start + _ROW_CHUNK].double() @ w64.T
        gate, up = pre[:, :n_out], pre[:, n_out:]
        out[start:start + _ROW_CHUNK] = gate * torch.sigmoid(gate) * up
    return out
