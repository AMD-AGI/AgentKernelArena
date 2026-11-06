"""Float64 reference of affine LayerNorm over the last axis, with the row mean and reciprocal std."""
import torch

CHUNK_ROWS = 1 << 19


def reference(*, x, weight, bias, epsilon):
    out = torch.empty(x.shape, dtype=torch.float64, device=x.device)
    mean = torch.empty(x.shape[0], dtype=torch.float64, device=x.device)
    rstd = torch.empty(x.shape[0], dtype=torch.float64, device=x.device)
    w, b = weight.double(), bias.double()
    for start in range(0, x.shape[0], CHUNK_ROWS):
        block = x[start:start + CHUNK_ROWS].double()
        m = block.mean(dim=-1, keepdim=True)
        centered = block - m
        r = torch.rsqrt(centered.square().mean(dim=-1, keepdim=True) + epsilon)
        out[start:start + CHUNK_ROWS] = centered * r * w + b
        mean[start:start + CHUNK_ROWS] = m.squeeze(-1)
        rstd[start:start + CHUNK_ROWS] = r.squeeze(-1)
    return {"out": out, "mean": mean, "rstd": rstd}
