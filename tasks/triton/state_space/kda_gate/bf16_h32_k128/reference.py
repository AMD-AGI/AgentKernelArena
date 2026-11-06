"""High-precision reference of the KDA decay gate, used only for correctness."""
import torch


def reference(*, g, A_log, dt_bias):
    tokens = g.shape[0]
    rate = torch.exp(A_log.double())[None, :, None]
    bias = dt_bias.double()[None]
    out = torch.empty(g.shape, dtype=torch.float64, device=g.device)
    step = 16384  # tokens per step, to bound float64 temporaries
    for start in range(0, tokens, step):
        x = g[start:start + step].double() + bias
        softplus = torch.where(x > 20.0, x, torch.log1p(torch.exp(torch.clamp(x, max=20.0))))
        out[start:start + step] = -rate * softplus
    return out
