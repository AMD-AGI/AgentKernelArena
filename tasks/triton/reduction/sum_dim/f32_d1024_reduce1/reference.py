import torch


def reference(*, x):
    # Accumulate in float64, one batch element at a time to bound temporary memory.
    out = torch.empty((x.shape[0], 1, x.shape[2]), dtype=torch.float64, device=x.device)
    for b in range(x.shape[0]):
        out[b, 0] = x[b].double().sum(dim=0)
    return out
