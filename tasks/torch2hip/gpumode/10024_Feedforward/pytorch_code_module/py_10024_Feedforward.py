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
