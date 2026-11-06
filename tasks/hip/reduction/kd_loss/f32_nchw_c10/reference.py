"""High-precision knowledge-distillation loss reference used only for correctness."""
import torch

_CHUNK = 64


def reference(*, input, target, temp_factor):
    n = input.shape[0]
    temperature = float(temp_factor)
    pixel_kl = torch.empty((n,) + tuple(input.shape[2:]), dtype=torch.float64, device=input.device)
    for start in range(0, n, _CHUNK):
        x = input[start:start + _CHUNK].double() / temperature
        t = target[start:start + _CHUNK].double()
        log_p = torch.log_softmax(x, dim=1)
        pixel_kl[start:start + _CHUNK] = (torch.xlogy(t, t) - t * log_p).sum(dim=1)
    return {"pixel_kl": pixel_kl, "out": pixel_kl.sum() * temperature ** 2 / n}
