"""Float64 reference for the 7x7, stride-2, padding-3 NHWC convolution without bias.

`x` is [batch, height, width, in_channels]; `weight` is the logical OIHW tensor [out_channels, in_channels, 7, 7].
"""
import torch
import torch.nn.functional as F


def reference(*, x, weight):
    out = F.conv2d(x.double().permute(0, 3, 1, 2), weight.double(), None, stride=2, padding=3)
    return out.permute(0, 2, 3, 1)
