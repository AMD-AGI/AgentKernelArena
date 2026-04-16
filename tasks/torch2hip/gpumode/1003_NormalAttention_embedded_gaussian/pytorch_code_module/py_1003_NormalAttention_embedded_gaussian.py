# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import torch
import torch.nn as nn


class NormalAttention_embedded_gaussian(nn.Module):

    def __init__(self, input_channel_num, k=4):
        super(NormalAttention_embedded_gaussian, self).__init__()
        self.c_in = input_channel_num
        self.query_conv = nn.Conv2d(in_channels=self.c_in, out_channels=
            self.c_in // k, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=self.c_in, out_channels=self.
            c_in // k, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=self.c_in, out_channels=
            self.c_in, kernel_size=1)
        self.gamma = nn.Conv2d(in_channels=self.c_in, out_channels=self.
            c_in, kernel_size=1)

    def forward(self, x, fn=None):
        B, C, H, W = x.size()
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(B, -1, H * W)
        energy = torch.bmm(proj_query, proj_key)
        energy = torch.exp(energy)
        energy_sum = torch.sum(energy, dim=2, keepdim=True)
        energy = energy / energy_sum
        proj_value = self.value_conv(x).view(B, -1, H * W)
        out = torch.bmm(proj_value, energy).view(B, C, H, W)
        out = self.gamma(out)
        return out


def get_inputs():
    """
    Generate multiple test cases for attention kernels
    """
    configs = [
        ([4, 4, 4, 4],),  # (B, C, H, W)
        ([8, 8, 8, 8],),
        ([16, 16, 16, 16],),
        ([32, 32, 32, 32],),
        ([64, 64, 64, 64],),
    ]
    
    for shape in configs:
        shape_list = shape[0] if isinstance(shape, tuple) and len(shape) == 1 else shape
        x = torch.randn(shape_list, dtype=torch.float32)
        yield [x]


def get_init_inputs():
    return [[], {'input_channel_num': 4}]
