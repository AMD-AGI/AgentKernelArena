"""Float64 reference for a 3x3, stride-1, padding-1 NHWC convolution with bias.

`x` is [batch, height, width, in_channels]; `weight` is the logical OIHW filter tensor
[out_channels, in_channels, 3, 3] (not its Winograd-domain transform, which only the kernel-facing input carries).
"""
import torch
import torch.nn.functional as F


def reference(*, x, weight, bias):
    out = F.conv2d(x.double().permute(0, 3, 1, 2), weight.double(), bias.double(), stride=1, padding=1)
    return out.permute(0, 2, 3, 1)
