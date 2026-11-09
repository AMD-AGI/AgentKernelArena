"""Float64 reference of the 3x3x3 convolution with bias, zero padding of one along height and width only."""
import torch
import torch.nn.functional as F


def reference(*, x, weight, bias):
    return F.conv3d(x.double(), weight.double(), bias.double(), stride=1, padding=(0, 1, 1))
