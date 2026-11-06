import torch

_CHUNK = 1 << 24


def reference(*, x):
    out = torch.empty(x.shape, dtype=torch.float64, device=x.device)
    flat = x.reshape(-1)
    result = out.reshape(-1)
    for start in range(0, flat.numel(), _CHUNK):
        chunk = flat[start:start + _CHUNK].double()
        result[start:start + _CHUNK] = chunk * torch.sigmoid(chunk)
    return out
