"""Float64 reference for max(gelu(avg_pool1d(x @ weight.T + bias, 16)) * scale_factor, dim=1).values,
used only for correctness."""
import math

import torch

POOL = 16  # AvgPool1d kernel size and stride (task.yaml pool_size)


def reference(*, x, weight, bias, scale_factor):
    # nn.Linear: x @ weight.T + bias.
    y = torch.addmm(bias.double(), x.double(), weight.double().t())
    batch, features = y.shape
    # AvgPool1d(16) over the feature axis: stride 16, no padding, features % 16 == 0.
    pooled = y.view(batch, features // POOL, POOL).mean(dim=-1)
    # Exact (erf) GELU, F.gelu(approximate="none").
    g = 0.5 * pooled * (1.0 + torch.erf(pooled * (1.0 / math.sqrt(2.0))))
    return (g * scale_factor).amax(dim=1)
