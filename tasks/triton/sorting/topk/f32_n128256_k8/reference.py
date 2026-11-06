import torch


def reference(*, x, k):
    values, indices = torch.topk(x.double(), int(k), dim=-1, largest=True, sorted=True)
    return values, indices
