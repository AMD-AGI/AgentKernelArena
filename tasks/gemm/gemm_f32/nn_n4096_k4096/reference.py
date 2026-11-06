"""Float64 matrix product used only for correctness."""
import torch


def reference(*, a, b):
    return torch.matmul(a.double(), b.double())
