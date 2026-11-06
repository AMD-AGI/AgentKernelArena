"""Float64 scale_factor * sum(max_pool1d(x @ weight.T + bias, 2), dim=1) used only for correctness."""
import torch
import torch.nn.functional as F

POOL_KERNEL = 2
# Output features per step, a multiple of the pool window, to bound the float64 temporaries.
CHUNK = 4096


def reference(*, x, weight, bias, scale_factor):
    x64 = x.double()
    total = torch.zeros(x.shape[0], dtype=torch.float64, device=x.device)
    for start in range(0, weight.shape[0], CHUNK):
        y = torch.addmm(bias[start:start + CHUNK].double(), x64, weight[start:start + CHUNK].double().t())
        total += F.max_pool1d(y.unsqueeze(1), POOL_KERNEL).squeeze(1).sum(dim=1)
    return total * scale_factor
