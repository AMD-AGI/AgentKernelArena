# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

def module_fn(
    x: torch.Tensor,
    y: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
) -> torch.Tensor:
    """
    Functional implementation of the Feedforward model.

    Args:
        x (torch.Tensor): Input tensor of shape (..., input_size)
        y (torch.Tensor): Input tensor of shape (..., input_size)
        fc1_weight (torch.Tensor): Weight for first linear layer (hidden_size, input_size)
        fc1_bias (torch.Tensor): Bias for first linear layer (hidden_size)
        fc2_weight (torch.Tensor): Weight for second linear layer (1, hidden_size)
        fc2_bias (torch.Tensor): Bias for second linear layer (1)

    Returns:
        torch.Tensor: Output tensor after feedforward pass
    """
    # Stack x and y along the first dimension
    inp = torch.vstack([x, y])
    # First linear layer
    hidden = F.linear(inp, fc1_weight, fc1_bias)
    # ReLU activation
    relu = F.relu(hidden)
    # Second linear layer
    output = F.linear(relu, fc2_weight, fc2_bias)
    # Sigmoid activation
    output = torch.sigmoid(output)
    return output

class Feedforward(nn.Module):
    def __init__(self, input_size, hidden_size=100):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        fc1 = nn.Linear(self.input_size, self.hidden_size)
        fc2 = nn.Linear(self.hidden_size, 1)
        self.fc1_weight = nn.Parameter(fc1.weight)
        self.fc1_bias = nn.Parameter(fc1.bias)
        self.fc2_weight = nn.Parameter(fc2.weight)
        self.fc2_bias = nn.Parameter(fc2.bias)

    def forward(self, x, y, fn=module_fn):
        return fn(
            x, y,
            self.fc1_weight, self.fc1_bias,
            self.fc2_weight, self.fc2_bias
        )

def get_inputs():
    """
    Decode and prefill row counts, including unequal x/y batches and tile
    tails.
    """
    shapes = [
        [(1, 128), (3, 128)],
        [(7, 128), (9, 128)],
        [(31, 128), (33, 128)],
        [(128, 128), (129, 128)],
        [(257, 128), (255, 128)],
    ]
    for case_shapes in shapes:
        yield [torch.randn(shape, dtype=torch.float32) for shape in case_shapes]


def get_init_inputs():
    return [[], {'input_size': 128, 'hidden_size': 2048}]
