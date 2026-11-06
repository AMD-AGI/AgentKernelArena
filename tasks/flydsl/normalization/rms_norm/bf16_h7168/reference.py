"""High-precision RMSNorm reference used only for correctness (epsilon fixed at 1e-5)."""
import torch

EPSILON = 1e-5


def reference(*, x, weight):
    x64 = x.double()
    mean_square = x64.square().mean(dim=-1, keepdim=True)
    return x64 * torch.rsqrt(mean_square + EPSILON) * weight.double()
