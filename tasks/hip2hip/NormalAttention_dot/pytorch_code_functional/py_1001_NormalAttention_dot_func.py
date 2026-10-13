# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

def normal_attention_dot_fn(
    x: torch.Tensor,
    query_weight: torch.Tensor,
    query_bias: torch.Tensor,
    key_weight: torch.Tensor,
    key_bias: torch.Tensor,
    value_weight: torch.Tensor,
    value_bias: torch.Tensor,
    gamma_weight: torch.Tensor,
    gamma_bias: torch.Tensor,
) -> torch.Tensor:
    """
    Functional implementation of NormalAttention_dot.
    """
    B, C, H, W = x.size()
    # Query projection: (B, C, H, W) -> (B, C//k, H, W)
    proj_query = F.conv2d(x, query_weight, query_bias)
    # (B, C//k, H, W) -> (B, C//k, H*W) -> (B, H*W, C//k)
    proj_query = proj_query.view(B, -1, H * W).permute(0, 2, 1)
    # Key projection: (B, C, H, W) -> (B, C//k, H, W)
    proj_key = F.conv2d(x, key_weight, key_bias)
    # (B, C//k, H, W) -> (B, C//k, H*W)
    proj_key = proj_key.view(B, -1, H * W)
    # Energy: (B, H*W, C//k) x (B, C//k, H*W) -> (B, H*W, H*W)
    energy = torch.bmm(proj_query, proj_key)
    # Nonlinearity
    energy = F.elu(energy)
    # Normalize
    energy = energy / (H * W)
    # Value projection: (B, C, H, W) -> (B, C, H, W)
    proj_value = F.conv2d(x, value_weight, value_bias)
    # (B, C, H, W) -> (B, C, H*W)
    proj_value = proj_value.view(B, C, H * W)
    # Output: (B, C, H*W) x (B, H*W, H*W) -> (B, C, H*W)
    out = torch.bmm(proj_value, energy)
    # (B, C, H*W) -> (B, C, H, W)
    out = out.view(B, C, H, W)
    # Final 1x1 conv
    out = F.conv2d(out, gamma_weight, gamma_bias)
    return out

def module_fn(
    x: torch.Tensor,
    query_weight: torch.Tensor,
    query_bias: torch.Tensor,
    key_weight: torch.Tensor,
    key_bias: torch.Tensor,
    value_weight: torch.Tensor,
    value_bias: torch.Tensor,
    gamma_weight: torch.Tensor,
    gamma_bias: torch.Tensor,
) -> torch.Tensor:
    return normal_attention_dot_fn(
        x,
        query_weight, query_bias,
        key_weight, key_bias,
        value_weight, value_bias,
        gamma_weight, gamma_bias,
    )

class NormalAttention_dot(nn.Module):
    def __init__(self, input_channel_num, k=4):
        super(NormalAttention_dot, self).__init__()
        self.c_in = input_channel_num
        self.query_conv = nn.Conv2d(in_channels=self.c_in, out_channels=self.c_in // k, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=self.c_in, out_channels=self.c_in // k, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=self.c_in, out_channels=self.c_in, kernel_size=1)
        self.nonlin = nn.ELU()
        self.gamma = nn.Conv2d(in_channels=self.c_in, out_channels=self.c_in, kernel_size=1)

    def forward(self, x, fn=module_fn):
        return fn(
            x,
            self.query_conv.weight, self.query_conv.bias,
            self.key_conv.weight, self.key_conv.bias,
            self.value_conv.weight, self.value_conv.bias,
            self.gamma.weight, self.gamma.bias,
        )

def get_inputs():
    """
    Generate multiple test cases for attention kernels
    All test cases must use C=4 to match input_channel_num=4 in get_init_inputs()
    """
    configs = [
        ([4, 4, 4, 4],),  # (B=4, C=4, H=4, W=4)
        ([8, 4, 8, 8],),  # (B=8, C=4, H=8, W=8) - keep C=4
        ([16, 4, 16, 16],),  # (B=16, C=4, H=16, W=16) - keep C=4
        ([32, 4, 32, 32],),  # (B=32, C=4, H=32, W=32) - keep C=4
        ([64, 4, 64, 64],),  # (B=64, C=4, H=64, W=64) - keep C=4
    ]
    
    for shape in configs:
        shape_list = shape[0] if isinstance(shape, tuple) and len(shape) == 1 else shape
        x = torch.randn(shape_list, dtype=torch.float32)
        yield [x]


def get_init_inputs():
    return [[], {'input_channel_num': 4}]
