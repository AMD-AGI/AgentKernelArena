"""High-precision fused residual-add + RMSNorm reference used only for correctness."""
import torch

_CHUNK_ROWS = 16384


def reference(*, x, residual, weight, epsilon):
    n_rows, n_cols = x.shape
    out = torch.empty((n_rows, n_cols), dtype=torch.float64, device=x.device)
    residual_out = torch.empty((n_rows, n_cols), dtype=torch.float64, device=x.device)
    w64 = weight.double()
    for start in range(0, n_rows, _CHUNK_ROWS):
        rows = slice(start, start + _CHUNK_ROWS)
        # The residual stream is stored in the input dtype and the normalization reads the stored value.
        summed = (x[rows].double() + residual[rows].double()).to(x.dtype).double()
        mean_square = summed.square().mean(dim=-1, keepdim=True)
        out[rows] = summed * torch.rsqrt(mean_square + epsilon) * w64
        residual_out[rows] = summed
    return out, residual_out
