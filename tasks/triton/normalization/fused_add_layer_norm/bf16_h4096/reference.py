"""High-precision fused residual-add + LayerNorm reference used only for correctness."""
import torch

_CHUNK_ROWS = 8192


def reference(*, x, residual, weight, bias, epsilon):
    n_rows, n_cols = x.shape
    out = torch.empty((n_rows, n_cols), dtype=torch.float64, device=x.device)
    residual_out = torch.empty((n_rows, n_cols), dtype=torch.float64, device=x.device)
    w64 = weight.double()
    b64 = bias.double()
    for start in range(0, n_rows, _CHUNK_ROWS):
        rows = slice(start, start + _CHUNK_ROWS)
        # The residual stream is stored in the input dtype and the normalization reads the stored value.
        summed = (x[rows].double() + residual[rows].double()).to(x.dtype).double()
        mean = summed.mean(dim=-1, keepdim=True)
        centered = summed - mean
        var = centered.square().mean(dim=-1, keepdim=True)
        out[rows] = centered * torch.rsqrt(var + epsilon) * w64 + b64
        residual_out[rows] = summed
    return out, residual_out
