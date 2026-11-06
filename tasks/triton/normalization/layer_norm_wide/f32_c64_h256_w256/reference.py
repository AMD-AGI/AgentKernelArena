"""High-precision LayerNorm reference over the whole sample, used only for correctness."""
import torch


def reference(*, x, weight, bias, epsilon):
    out = torch.empty(x.shape, dtype=torch.float64, device=x.device)
    w64 = weight.double()
    b64 = bias.double()
    for i in range(x.shape[0]):
        xi = x[i].double()
        centered = xi - xi.mean()
        var = centered.square().mean()
        out[i] = centered * torch.rsqrt(var + epsilon) * w64 + b64
    return out
