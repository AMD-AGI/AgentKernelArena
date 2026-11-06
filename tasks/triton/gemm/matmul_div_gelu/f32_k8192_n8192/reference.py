"""Float64 reference for gelu((x @ weight.T + bias) / divisor), used only for correctness."""
import math

import torch


def reference(*, x, weight, bias, divisor):
    # nn.Linear: x @ weight.T + bias.
    y = torch.addmm(bias.double(), x.double(), weight.double().t())
    y = y / divisor
    # Exact (erf) GELU, F.gelu(approximate="none").
    return 0.5 * y * (1.0 + torch.erf(y * (1.0 / math.sqrt(2.0))))
