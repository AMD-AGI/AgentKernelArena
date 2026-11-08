"""Activations and filters of the 3x3 convolution, and the Winograd-domain filter layout.

Logical `x` is SiLU of a standard normal (the input of a convolution that follows GroupNorm and SiLU, as in a VAE
resnet block) in bfloat16. Logical `weight` is the OIHW tensor [out_channels, in_channels, 3, 3] drawn like nn.Conv2d's
default initialization (uniform in +-1/sqrt(fan_in), fan_in = in_channels * 9) and stored in bfloat16.
"""
import math

import torch

_BLOCK_C = 64  # channel padding granularity of aiter's prepack (BLOCK_K in aiter/ops/triton/conv/_utils.py)

# G of Winograd F(4x4, 3x3), as in aiter prepack_winograd_filter_f4x3
_G = [
    [1.0 / 4, 0.0, 0.0],
    [-1.0 / 6, -1.0 / 6, -1.0 / 6],
    [-1.0 / 6, 1.0 / 6, -1.0 / 6],
    [1.0 / 24, 1.0 / 12, 1.0 / 6],
    [1.0 / 24, -1.0 / 12, 1.0 / 6],
    [0.0, 0.0, 1.0],
]


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    batch, height, width = axes["batch"], axes["height"], axes["width"]
    in_channels, out_channels = axes["in_channels"], axes["out_channels"]
    pre = torch.randn(batch, height, width, in_channels, generator=generator, device=device, dtype=torch.float32)
    x = (pre * torch.sigmoid(pre)).to(torch.bfloat16)
    del pre
    bound = 1.0 / math.sqrt(in_channels * 9)
    weight = torch.rand(out_channels, in_channels, 3, 3, generator=generator, device=device, dtype=torch.float32)
    weight = (weight * (2 * bound) - bound).to(torch.bfloat16)
    return {"x": x, "weight": weight}


def to_kernel_layout(case, inputs):
    """aiter prepack_winograd_filter_f4x3: U = G g G^T in float32 per (o, c), stored as bfloat16 [36, o, c_padded]."""
    inputs = dict(inputs)
    weight = inputs["weight"]
    out_channels, in_channels = weight.shape[0], weight.shape[1]
    padded = ((in_channels + _BLOCK_C - 1) // _BLOCK_C) * _BLOCK_C
    g_matrix = torch.tensor(_G, dtype=torch.float32, device=weight.device)
    transformed = torch.einsum("ij,kcjl,lm->kcim", g_matrix, weight.float(), g_matrix.t())
    transformed = transformed.reshape(out_channels, in_channels, 36).permute(2, 0, 1).contiguous()
    if padded != in_channels:
        pad = torch.zeros((36, out_channels, padded - in_channels), device=weight.device, dtype=torch.float32)
        transformed = torch.cat([transformed, pad], dim=2)
    inputs["weight"] = transformed.to(weight.dtype).contiguous()
    return inputs
