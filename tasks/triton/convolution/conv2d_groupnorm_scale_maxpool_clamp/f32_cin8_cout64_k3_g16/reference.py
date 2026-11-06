"""float64 reference for conv2d -> GroupNorm -> * scale -> max_pool2d -> clamp."""
import torch
import torch.nn.functional as F

NUM_GROUPS = 16
POOL = 4


def reference(*, x, conv_weight, conv_bias, gn_weight, gn_bias, scale, epsilon, clamp_min, clamp_max):
    # Valid (unpadded) stride-1 convolution, as nn.Conv2d(8, 64, 3).
    y = F.conv2d(x.double(), conv_weight.double(), conv_bias.double())
    batch, channels, height, width = y.shape
    # GroupNorm: per sample and group of 4 channels, mean and biased variance over
    # channels x the full (unpooled) conv output.
    grouped = y.view(batch, NUM_GROUPS, -1)
    variance, mean = torch.var_mean(grouped, dim=-1, correction=0, keepdim=True)
    y = (grouped - mean).mul_(torch.rsqrt(variance + epsilon)).view(batch, channels, height, width)
    y.mul_(gn_weight.double().view(1, -1, 1, 1)).add_(gn_bias.double().view(1, -1, 1, 1))
    # Per-channel scale of shape (out_channels, 1, 1), applied before pooling.
    y.mul_(scale.double())
    # MaxPool2d(4): stride 4, no padding, floor mode (trailing rows and columns are dropped).
    y = F.max_pool2d(y, kernel_size=POOL, stride=POOL)
    return torch.clamp(y, clamp_min, clamp_max)
