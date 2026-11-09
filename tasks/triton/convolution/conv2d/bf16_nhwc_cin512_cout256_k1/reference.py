"""Float64 reference for a 1x1, stride-1 NHWC convolution with bias.

`x` is [batch, height, width, in_channels]; `weight` is the logical OIHW tensor [out_channels, in_channels, 1, 1].
"""
import torch
import torch.nn.functional as F


def reference(*, x, weight, bias):
    out = F.conv2d(x.double().permute(0, 3, 1, 2), weight.double(), bias.double(), stride=1, padding=0)
    return out.permute(0, 2, 3, 1)
