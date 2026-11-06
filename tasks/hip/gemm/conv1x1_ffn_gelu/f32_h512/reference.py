"""High-precision reference for the transformer feed-forward block, used only for correctness."""
import torch


def reference(*, x, conv_weight, conv_bias, linear_weight, linear_bias):
    # The k=1 convolution is a per-token linear map; its output is scaled by kernel_size ** -0.5 == 1.
    hidden = torch.nn.functional.gelu(x.double() @ conv_weight.double().T + conv_bias.double())
    return hidden @ linear_weight.double().T + linear_bias.double()
