"""High-precision RMSNorm backward reference used only for correctness."""
import torch

CHUNK_ROWS = 1 << 15


def reference(*, dy, x, gamma, rsigma):
    n_tokens, hidden = x.shape
    g = gamma.double()
    dx = torch.empty(x.shape, dtype=torch.float64, device=x.device)
    dgamma = torch.zeros(hidden, dtype=torch.float64, device=x.device)
    for start in range(0, n_tokens, CHUNK_ROWS):
        stop = start + CHUNK_ROWS
        dy64 = dy[start:stop].double()
        x64 = x[start:stop].double()
        r = rsigma[start:stop].double()[:, None]
        mean_dot = (dy64 * x64 * g).mean(dim=-1, keepdim=True)
        dx[start:stop] = dy64 * r * g - r * r * r * x64 * mean_dot
        dgamma += (dy64 * x64 * r).sum(dim=0)
    return {"dx": dx, "dgamma": dgamma}
