"""High-precision RMSNorm reference used only for correctness."""
import torch


def reference(*, x, weight, epsilon):
    x64 = x.double()
    mean_square = x64.square().mean(dim=-1, keepdim=True)
    return x64 * torch.rsqrt(mean_square + epsilon) * weight.double()
