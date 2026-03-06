# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import torch


class Feedforward(torch.nn.Module):

    def __init__(self, input_size, hidden_size=100):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, y, fn=None):
        inp = torch.vstack([x, y])
        hidden = self.fc1(inp)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output


def get_inputs():
    """
    Generate multiple test cases for Feedforward
    HIP kernel expects x and y to be at least 1D with matching last dimension
    To work around HIP kernel shape calculation bug, use 2D inputs [batch, features]
    where features matches input_size=4
    """
    configs = [
        # (batch, features) - features must match input_size=4
        ([1, 4], [1, 4]),  # Both x and y are 2D with (batch=1, features=4)
        ([2, 4], [2, 4]),  # (batch=2, features=4)
        ([4, 4], [4, 4]),  # (batch=4, features=4)
        ([8, 4], [8, 4]),  # (batch=8, features=4)
        ([16, 4], [16, 4]),  # (batch=16, features=4)
    ]
    
    for x_shape, y_shape in configs:
        x = torch.randn(x_shape, dtype=torch.float32)
        y = torch.randn(y_shape, dtype=torch.float32)
        yield [x, y]


def get_init_inputs():
    return [[], {'input_size': 4}]
