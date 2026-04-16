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
