import torch


def reference(*, x):
    return torch.sort(x.double(), dim=-1).values
