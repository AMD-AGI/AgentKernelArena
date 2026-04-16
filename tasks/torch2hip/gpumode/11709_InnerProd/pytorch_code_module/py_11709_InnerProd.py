# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import torch
import torch.nn as nn


class InnerProd(nn.Module):

    def __init__(self, fc_dim):
        super(InnerProd, self).__init__()
        self.scale = nn.Parameter(torch.ones(fc_dim))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, feat_img, feat_sound, fn=None):
        sound_size = feat_sound.size()
        B, C = sound_size[0], sound_size[1]
        feat_img = feat_img.view(B, 1, C)
        z = torch.bmm(feat_img * self.scale, feat_sound.view(B, C, -1)).view(B,
            1, *sound_size[2:])
        z = z + self.bias
        return z

    def forward_nosum(self, feat_img, feat_sound):
        B, C, _H, _W = feat_sound.size()
        feat_img = feat_img.view(B, C)
        z = (feat_img * self.scale).view(B, C, 1, 1) * feat_sound
        z = z + self.bias
        return z

    def forward_pixelwise(self, feats_img, feat_sound):
        B, C, HI, WI = feats_img.size()
        B, C, HS, WS = feat_sound.size()
        feats_img = feats_img.view(B, C, HI * WI)
        feats_img = feats_img.transpose(1, 2)
        feat_sound = feat_sound.view(B, C, HS * WS)
        z = torch.bmm(feats_img * self.scale, feat_sound).view(B, HI, WI,
            HS, WS)
        z = z + self.bias
        return z


def get_inputs():
    """
    Generate multiple test cases with varying sizes
    HIP kernel expects feat_img: [B, 1, C] (3D with middle dim=1) and feat_sound: [B, C, H, W]
    """
    configs = [
        # (B, C, H, W) for feat_sound, feat_img will be [B, 1, C]
        (4, 4, 4, 4),  # B=4, C=4, H=4, W=4
        (8, 4, 8, 8),  # Keep C=4 to match fc_dim=4
        (16, 4, 16, 16),  # Keep C=4
        (32, 4, 32, 32),  # Keep C=4
        (64, 4, 64, 64),  # Keep C=4
    ]
    
    for B, C, H, W in configs:
        # HIP kernel requires feat_img to be [B, 1, C], not [B, C]
        feat_img = torch.randn([B, 1, C], dtype=torch.float32)
        feat_sound = torch.randn([B, C, H, W], dtype=torch.float32)
        yield [feat_img, feat_sound]


def get_init_inputs():
    return [[], {'fc_dim': 4}]
