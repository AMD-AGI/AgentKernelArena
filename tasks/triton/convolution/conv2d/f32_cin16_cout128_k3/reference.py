"""Float64 reference for a 3x3, stride-1, unpadded, bias-free NCHW convolution."""
import torch
import torch.nn.functional as F


def reference(*, x, weight):
    return F.conv2d(x.double(), weight.double(), None, stride=1, padding=0, dilation=1, groups=1)
