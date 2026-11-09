"""Task-owned generator of the forward's saved statistic."""
import torch

EPSILON = 1.0e-5


def make_inputs(case, inputs, generator, device):
    """rsigma = rsqrt(mean(x^2) + epsilon) per token, as the forward pass of the normalization saved it."""
    x = inputs["x"].float()
    return {"rsigma": torch.rsqrt(x.square().mean(dim=-1) + EPSILON)}
