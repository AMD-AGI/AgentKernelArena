# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

def gate_gru_selection_layer_fn(
    x_1: torch.Tensor,
    x_2: torch.Tensor,
    reset_weight: torch.Tensor,
    reset_bias: torch.Tensor,
    update_weight: torch.Tensor,
    update_bias: torch.Tensor,
    proposal_weight: torch.Tensor,
    proposal_bias: torch.Tensor,
) -> torch.Tensor:
    # Concatenate x_1 and x_2 along the last dimension for reset and update gates
    cat_x1_x2 = torch.cat([x_1, x_2], dim=-1)
    # Reset gate: sigmoid(linear(cat(x_1, x_2)))
    reset = torch.sigmoid(F.linear(cat_x1_x2, reset_weight, reset_bias))
    # Update gate: sigmoid(linear(cat(x_1, x_2)))
    update = torch.sigmoid(F.linear(cat_x1_x2, update_weight, update_bias))
    # Proposal: tanh(linear(cat(reset * x_1, x_2)))
    cat_resetx1_x2 = torch.cat([reset * x_1, x_2], dim=-1)
    proposal = torch.tanh(F.linear(cat_resetx1_x2, proposal_weight, proposal_bias))
    # Output: (1 - update) * x_1 + update * proposal
    out = (1 - update) * x_1 + update * proposal
    return out

def module_fn(
    x_1: torch.Tensor,
    x_2: torch.Tensor,
    reset_weight: torch.Tensor,
    reset_bias: torch.Tensor,
    update_weight: torch.Tensor,
    update_bias: torch.Tensor,
    proposal_weight: torch.Tensor,
    proposal_bias: torch.Tensor,
) -> torch.Tensor:
    return gate_gru_selection_layer_fn(
        x_1, x_2,
        reset_weight, reset_bias,
        update_weight, update_bias,
        proposal_weight, proposal_bias
    )

class GateGRUSelectionLayer(nn.Module):
    def __init__(self, dim_model, dim_ff, prob_dropout):
        super(GateGRUSelectionLayer, self).__init__()
        self.reset = nn.Linear(dim_model * 2, dim_model)
        self.update = nn.Linear(dim_model * 2, dim_model)
        self.proposal = nn.Linear(dim_model * 2, dim_model)

    def forward(
        self,
        x_1,
        x_2,
        fn=module_fn
    ):
        return fn(
            x_1,
            x_2,
            self.reset.weight,
            self.reset.bias,
            self.update.weight,
            self.update.bias,
            self.proposal.weight,
            self.proposal.bias,
        )

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
