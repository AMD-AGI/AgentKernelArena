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
    Use 2D inputs [batch, features] where features matches input_size.
    input_size kept at 128 (the reference kernel caps cached features at 256);
    hidden_size scaled to 2048 so fc1 is a real GEMM. vstack(x,y) -> 2*batch rows.
    """
    configs = [
        # (batch, features) - features must match input_size=128
        ([512, 128], [512, 128]),    # 1024 rows after vstack
        ([1024, 128], [1024, 128]),  # 2048 rows
        ([2048, 128], [2048, 128]),  # 4096 rows
        ([4096, 128], [4096, 128]),  # 8192 rows
    ]
    
    for x_shape, y_shape in configs:
        x = torch.randn(x_shape, dtype=torch.float32)
        y = torch.randn(y_shape, dtype=torch.float32)
        yield [x, y]


def get_init_inputs():
    return [[], {'input_size': 128, 'hidden_size': 2048}]
