# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import torch
import torch.nn as nn


class NormalAttention_dot(nn.Module):

    def __init__(self, input_channel_num, k=4):
        super(NormalAttention_dot, self).__init__()
        self.c_in = input_channel_num
        self.query_conv = nn.Conv2d(in_channels=self.c_in, out_channels=
            self.c_in // k, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=self.c_in, out_channels=self.
            c_in // k, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=self.c_in, out_channels=
            self.c_in, kernel_size=1)
        self.nonlin = nn.ELU()
        self.gamma = nn.Conv2d(in_channels=self.c_in, out_channels=self.
            c_in, kernel_size=1)

    def forward(self, x, fn=None):
        B, C, H, W = x.size()
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(B, -1, H * W)
        energy = torch.bmm(proj_query, proj_key)
        energy = self.nonlin(energy)
        energy = energy / (H * W)
        proj_value = self.value_conv(x).view(B, -1, H * W)
        out = torch.bmm(proj_value, energy).view(B, C, H, W)
        out = self.gamma(out)
        return out


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
