"""High-precision row softmax reference used only for correctness."""
import torch


def reference(*, x):
    x64 = x.double()
    e = torch.exp(x64 - x64.amax(dim=-1, keepdim=True))
    return e / e.sum(dim=-1, keepdim=True)
