"""High-precision scaled tanh reference used only for correctness."""
import torch


def reference(*, x, a, scale):
    scaled = x.double().mul_(float(a))
    return torch.tanh(scaled).mul_(float(scale))
