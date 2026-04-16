# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import torch
import torch.nn as nn
import torch.utils.data


class KDLoss(nn.Module):

    def __init__(self, temp_factor):
        super(KDLoss, self).__init__()
        self.temp_factor = temp_factor
        self.kl_div = nn.KLDivLoss(reduction='sum')

    def forward(self, input, target, fn=None):
        log_p = torch.log_softmax(input / self.temp_factor, dim=1)
        loss = self.kl_div(log_p, target) * self.temp_factor ** 2 / input.size(
            0)
        return loss


def get_inputs():
    """
    Generate multiple test cases for KL divergence loss
    HIP kernel requires 4D input [N,C,H,W] for image-based knowledge distillation
    """
    configs = [
        # (N, C, H, W) - 4D tensors for image-based KD loss
        ([8, 10, 32, 32], [8, 10, 32, 32]),
        ([16, 10, 32, 32], [16, 10, 32, 32]),
        ([8, 10, 64, 64], [8, 10, 64, 64]),
        ([16, 10, 64, 64], [16, 10, 64, 64]),
    ]
    
    for pred_shape, target_shape in configs:
        pred = torch.randn(pred_shape, dtype=torch.float32)
        # Generate probability distribution (softmax to ensure sum=1.0 along channel dim)
        target = torch.randn(target_shape, dtype=torch.float32)
        target = torch.softmax(target, dim=1)  # Convert to probability distribution along channel dim
        yield [pred, target]


def get_init_inputs():
    return [[], {'temp_factor': 4}]
