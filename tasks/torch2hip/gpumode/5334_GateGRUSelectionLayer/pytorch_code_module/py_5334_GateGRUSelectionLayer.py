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
    Three 1024-to-512 gated projections, with 1/17/63/258/1024 flattened rows
    and distinct input tensors.
    """
    shapes = [
        [(1, 1, 1, 512), (1, 1, 1, 512)],
        [(1, 1, 17, 512), (1, 1, 17, 512)],
        [(3, 1, 21, 512), (3, 1, 21, 512)],
        [(2, 3, 43, 512), (2, 3, 43, 512)],
        [(8, 4, 32, 512), (8, 4, 32, 512)],
    ]
    for case_shapes in shapes:
        yield [torch.randn(shape, dtype=torch.float32) for shape in case_shapes]


def get_init_inputs():
    return [[], {'dim_model': 512, 'dim_ff': 2048, 'prob_dropout': 0.5}]
