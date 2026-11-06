"""Float64 batched matrix product used only for correctness."""
import torch


def reference(*, a, b):
    return torch.bmm(a.double(), b.double())
