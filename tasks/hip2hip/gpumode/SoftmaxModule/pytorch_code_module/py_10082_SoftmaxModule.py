# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import torch
import torch.nn as nn


class SoftmaxModule(nn.Module):

    def __init__(self, axis):
        super().__init__()
        self.axis = axis

    def forward(self, v, fn=None):
        return v.softmax(self.axis)


def get_inputs():
    """
    Generate multiple test cases with varying sizes
    """
    configs = [
        ([4, 4, 4],),
        ([8, 8, 8],),
        ([16, 16, 16],),
        ([32, 32, 32],),
        ([64, 64, 64],),
    ]
    
    for shape in configs:
        # Unpack tuple if shape is a tuple containing a list (e.g., ([1024],) -> [1024])
        shape_list = shape[0] if isinstance(shape, tuple) and len(shape) == 1 else shape
        # Only yield one input tensor - axis is a model parameter
        v = torch.randn(shape_list, dtype=torch.float32)
        yield [v]


def get_init_inputs():
    # Use axis=2 (last dimension) for 3D tensors, or axis=-1 (same thing)
    return [[], {'axis': 2}]
