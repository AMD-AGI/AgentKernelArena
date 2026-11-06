"""High-precision dual-residual RMSNorm reference used only for correctness."""
import torch

_CHUNK_ROWS = 16384


def _rms_norm(v, weight, epsilon):
    return v * torch.rsqrt(v.square().mean(dim=-1, keepdim=True) + epsilon) * weight


def reference(*, x, residual, weight1, weight2, epsilon):
    n_rows, n_cols = x.shape
    out = torch.empty((n_rows, n_cols), dtype=torch.float64, device=x.device)
    mid = torch.empty((n_rows, n_cols), dtype=torch.float64, device=x.device)
    w1 = weight1.double()
    w2 = weight2.double()
    for start in range(0, n_rows, _CHUNK_ROWS):
        rows = slice(start, start + _CHUNK_ROWS)
        # The first norm result and the residual stream are stored in the input dtype, as in the operator.
        inner = _rms_norm(x[rows].double(), w1, epsilon).to(x.dtype).double()
        summed = (residual[rows].double() + inner).to(x.dtype).double()
        out[rows] = _rms_norm(summed, w2, epsilon)
        mid[rows] = summed
    return out, mid
