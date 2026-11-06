"""float64 reference for conv2d -> InstanceNorm2d -> / divide_by."""
import torch
import torch.nn.functional as F


def reference(*, x, conv_weight, conv_bias, epsilon, divide_by):
    # Valid (unpadded) stride-1 convolution, as nn.Conv2d(64, 128, 3).
    y = F.conv2d(x.double(), conv_weight.double(), conv_bias.double())
    batch, channels, height, width = y.shape
    # InstanceNorm2d(affine=False, track_running_stats=False): every (sample, channel) plane
    # is normalized with its own mean and biased variance.
    planes = y.view(batch, channels, -1)
    variance, mean = torch.var_mean(planes, dim=-1, correction=0, keepdim=True)
    y = (planes - mean).mul_(torch.rsqrt(variance + epsilon)).view(batch, channels, height, width)
    return y.div_(divide_by)
