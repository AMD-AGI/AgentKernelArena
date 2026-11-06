"""Float64 reference: 3x3 convolution with bias, Mish, then eval-mode batch normalization."""
import torch
import torch.nn.functional as F


def reference(*, x, conv_weight, conv_bias, bn_weight, bn_bias, running_mean, running_var, eps):
    y = F.conv2d(x.double(), conv_weight.double(), conv_bias.double(), stride=1, padding=0, dilation=1, groups=1)
    y = torch.tanh(F.softplus(y)) * y
    return F.batch_norm(y, running_mean.double(), running_var.double(), bn_weight.double(), bn_bias.double(),
                        training=False, eps=float(eps))
