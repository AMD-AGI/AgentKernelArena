"""Per-head decay slopes of MiniMax-Text-01 lightning attention.

The slopes are the ALiBi-style geometric sequence of `MiniMaxText01LinearAttention._build_slope_tensor`
scaled by the layer factor `1 - layer_idx / (num_layers - 1) + 1e-5`; layer 20 of 80 is used.
"""
import math

import torch

LAYER_IDX = 20
NUM_LAYERS = 80


def _slopes(n):
    def power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * start ** i for i in range(n)]

    if math.log2(n).is_integer():
        return power_of_2(n)
    closest = 2 ** math.floor(math.log2(n))
    return power_of_2(closest) + _slopes(2 * closest)[0::2][: n - closest]


def make_inputs(case, inputs, generator, device):
    heads = case["axes"]["num_heads"]
    slopes = torch.tensor(_slopes(heads), dtype=torch.float32, device=device)
    return {"slope_rate": slopes * (1 - LAYER_IDX / (NUM_LAYERS - 1) + 1e-5)}
