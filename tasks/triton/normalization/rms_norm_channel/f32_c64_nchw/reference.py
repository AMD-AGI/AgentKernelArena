"""High-precision channel-axis RMS normalization reference used only for correctness."""
import torch


def reference(*, x, epsilon):
    out = torch.empty(x.shape, dtype=torch.float64, device=x.device)
    for i in range(x.shape[0]):
        xi = x[i].double()
        out[i] = xi / torch.sqrt(xi.square().mean(dim=0, keepdim=True) + epsilon)
    return out
