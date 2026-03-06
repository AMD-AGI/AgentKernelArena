# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

# Functional kernel for InnerProd.forward
def innerprod_forward_fn(
    feat_img: torch.Tensor,
    feat_sound: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    # feat_img: (B, 1, C)
    # feat_sound: (B, C, H, W)
    # scale: (C,)
    # bias: (1,)
    sound_size = feat_sound.size()
    B, C = sound_size[0], sound_size[1]
    feat_img_ = feat_img.view(B, 1, C)
    scale_ = scale.view(1, 1, C)
    feat_sound_ = feat_sound.view(B, C, -1)
    z = torch.bmm(feat_img_ * scale_, feat_sound_).view(B, 1, *sound_size[2:])
    z = z + bias
    return z

# Functional kernel for InnerProd.forward_nosum
def innerprod_forward_nosum_fn(
    feat_img: torch.Tensor,
    feat_sound: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    # feat_img: (B, 1, C)
    # feat_sound: (B, C, H, W)
    # scale: (C,)
    # bias: (1,)
    B, C, H, W = feat_sound.size()
    feat_img_ = feat_img.view(B, C)
    scale_ = scale.view(1, C)
    z = (feat_img_ * scale_).view(B, C, 1, 1) * feat_sound
    z = z + bias
    return z

# Functional kernel for InnerProd.forward_pixelwise
def innerprod_forward_pixelwise_fn(
    feats_img: torch.Tensor,
    feat_sound: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    # feats_img: (B, C, HI, WI)
    # feat_sound: (B, C, HS, WS)
    # scale: (C,)
    # bias: (1,)
    B, C, HI, WI = feats_img.size()
    B, C, HS, WS = feat_sound.size()
    feats_img_ = feats_img.view(B, C, HI * WI).transpose(1, 2)  # (B, HI*WI, C)
    scale_ = scale.view(1, 1, C)
    feat_sound_ = feat_sound.view(B, C, HS * WS)
    z = torch.bmm(feats_img_ * scale_, feat_sound_).view(B, HI, WI, HS, WS)
    z = z + bias
    return z

# Main functional interface
def module_fn(
    feat_img: torch.Tensor,
    feat_sound: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    mode: str = "forward",
) -> torch.Tensor:
    """
    Functional implementation of InnerProd module.
    mode: "forward", "forward_nosum", or "forward_pixelwise"
    """
    if mode == "forward":
        return innerprod_forward_fn(feat_img, feat_sound, scale, bias)
    elif mode == "forward_nosum":
        return innerprod_forward_nosum_fn(feat_img, feat_sound, scale, bias)
    elif mode == "forward_pixelwise":
        return innerprod_forward_pixelwise_fn(feat_img, feat_sound, scale, bias)
    else:
        raise ValueError(f"Unknown mode: {mode}")

class InnerProd(nn.Module):
    def __init__(self, fc_dim):
        super(InnerProd, self).__init__()
        self.scale = nn.Parameter(torch.ones(fc_dim))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, feat_img, feat_sound, fn=module_fn):
        return fn(feat_img, feat_sound, self.scale, self.bias, mode="forward")

    def forward_nosum(self, feat_img, feat_sound, fn=module_fn):
        return fn(feat_img, feat_sound, self.scale, self.bias, mode="forward_nosum")

    def forward_pixelwise(self, feats_img, feat_sound, fn=module_fn):
        return fn(feats_img, feat_sound, self.scale, self.bias, mode="forward_pixelwise")

def get_inputs():
    """
    Generate multiple test cases with varying sizes
    HIP kernel expects feat_img: [B, 1, C] (3D with middle dim=1) and feat_sound: [B, C, H, W]
    All test cases must use C=4 to match fc_dim=4 in get_init_inputs()
    """
    configs = [
        # (B, C, H, W) for feat_sound, feat_img will be [B, 1, C]
        # Keep C=4 to match fc_dim=4
        (4, 4, 4, 4),  # B=4, C=4, H=4, W=4
        (8, 4, 8, 8),  # B=8, C=4, H=8, W=8 - keep C=4
        (16, 4, 16, 16),  # B=16, C=4, H=16, W=16 - keep C=4
        (32, 4, 32, 32),  # B=32, C=4, H=32, W=32 - keep C=4
        (64, 4, 64, 64),  # B=64, C=4, H=64, W=64 - keep C=4
    ]
    
    for B, C, H, W in configs:
        # HIP kernel requires feat_img to be [B, 1, C], not [B, C]
        feat_img = torch.randn([B, 1, C], dtype=torch.float32)
        feat_sound = torch.randn([B, C, H, W], dtype=torch.float32)
        yield [feat_img, feat_sound]


def get_init_inputs():
    return [[], {'fc_dim': 4}]
