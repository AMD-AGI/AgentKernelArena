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
    GateGRUSelectionLayer expects 4D tensors [B0, B1, B2, D].
    D scaled to 512 so the three Linear(2D, D) gates are real GEMMs against
    fused rocBLAS; rows = B0*B1*B2.
    """
    configs = [
        # 4D tensors: (B0, B1, B2, D) where D must match dim_model=512
        ([64, 4, 4, 512],),    # 1024 rows
        ([128, 4, 4, 512],),   # 2048 rows
        ([256, 4, 4, 512],),   # 4096 rows
        ([512, 4, 4, 512],),   # 8192 rows
        ([1024, 4, 4, 512],),  # 16384 rows
    ]
    
    for shape in configs:
        shape_list = shape[0] if isinstance(shape, tuple) and len(shape) == 1 else shape
        x_1 = torch.randn(shape_list, dtype=torch.float32)
        x_2 = torch.randn(shape_list, dtype=torch.float32)
        yield [x_1, x_2]


def get_init_inputs():
    return [[], {'dim_model': 512, 'dim_ff': 2048, 'prob_dropout': 0.5}]
