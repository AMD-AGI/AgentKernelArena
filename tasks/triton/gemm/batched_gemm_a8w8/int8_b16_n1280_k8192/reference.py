"""Float64 batched int8 GEMM reference used only for correctness."""
import torch


def reference(*, x, x_scale, w, w_scale):
    batch, m, _ = x.shape
    n = w.shape[1]
    out = torch.empty(batch, m, n, dtype=torch.float64, device=x.device)
    for b in range(batch):
        # int8 products summed in float64 are exact, matching an integer accumulation.
        product = x[b].double() @ w[b].double().T
        out[b] = product * (x_scale[b].double() * w_scale[b].double())
    return out
