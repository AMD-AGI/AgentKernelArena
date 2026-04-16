# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import torch
import torch.nn as nn


class GateGRUSelectionLayer(nn.Module):

    def __init__(self, dim_model, dim_ff, prob_dropout):
        super(GateGRUSelectionLayer, self).__init__()
        self.reset = nn.Linear(dim_model * 2, dim_model)
        self.update = nn.Linear(dim_model * 2, dim_model)
        self.proposal = nn.Linear(dim_model * 2, dim_model)

    def forward(self, x_1, x_2, *args, fn=None):
        reset = torch.sigmoid(self.reset(torch.cat([x_1, x_2], -1)))
        update = torch.sigmoid(self.update(torch.cat([x_1, x_2], -1)))
        proposal = torch.tanh(self.proposal(torch.cat([reset * x_1, x_2], -1)))
        out = (1 - update) * x_1 + update * proposal
        return out


def get_inputs():
    """
    Generate multiple test cases with varying sizes
    GateGRUSelectionLayer expects 4D tensors [B0, B1, B2, D]
    All test cases must use D=4 to match dim_model=4 in get_init_inputs()
    """
    configs = [
        # 4D tensors: (B0, B1, B2, D) where D must be 4
        ([4, 4, 4, 4],),  # (4, 4, 4, 4)
        ([8, 4, 4, 4],),  # (8, 4, 4, 4) - keep D=4
        ([16, 4, 4, 4],),  # (16, 4, 4, 4) - keep D=4
        ([32, 4, 4, 4],),  # (32, 4, 4, 4) - keep D=4
        ([64, 4, 4, 4],),  # (64, 4, 4, 4) - keep D=4
    ]
    
    for shape in configs:
        shape_list = shape[0] if isinstance(shape, tuple) and len(shape) == 1 else shape
        x_1 = torch.randn(shape_list, dtype=torch.float32)
        x_2 = torch.randn(shape_list, dtype=torch.float32)
        yield [x_1, x_2]


def get_init_inputs():
    return [[], {'dim_model': 4, 'dim_ff': 4, 'prob_dropout': 0.5}]
