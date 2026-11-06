"""High-precision log-softmax reference used only for correctness."""
import torch


def reference(*, x):
    return torch.log_softmax(x.double(), dim=-1)
