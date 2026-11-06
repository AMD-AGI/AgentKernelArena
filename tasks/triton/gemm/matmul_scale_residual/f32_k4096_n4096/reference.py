"""Float64 linear layer with a scaled residual, used only for correctness."""
import torch


def reference(*, x, weight, bias, scaling_factor):
    y = torch.addmm(bias.double(), x.double(), weight.double().t())
    # out = y * scaling_factor + y, the residual being the linear output itself.
    return y.mul(scaling_factor).add_(y)
