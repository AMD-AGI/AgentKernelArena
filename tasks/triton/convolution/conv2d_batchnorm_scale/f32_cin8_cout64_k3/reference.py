"""Float64 Conv2d -> BatchNorm2d (inference) -> scale reference used only for correctness."""
import torch


def reference(*, x, conv_weight, conv_bias, bn_weight, bn_bias, bn_running_mean, bn_running_var,
              bn_eps, scaling_factor):
    def per_channel(value):
        return value.double().view(1, -1, 1, 1)

    out = torch.nn.functional.conv2d(x.double(), conv_weight.double(), conv_bias.double())
    invstd = torch.rsqrt(per_channel(bn_running_var) + bn_eps)
    # In place to keep one float64 output-sized buffer for the largest cases.
    out.sub_(per_channel(bn_running_mean)).mul_(invstd * per_channel(bn_weight))
    out.add_(per_channel(bn_bias)).mul_(scaling_factor)
    return out
