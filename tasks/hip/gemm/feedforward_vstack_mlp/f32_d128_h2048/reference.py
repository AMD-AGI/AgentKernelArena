"""High-precision reference for the stacked two-layer MLP, used only for correctness."""
import torch


def reference(*, x, y, fc1_weight, fc1_bias, fc2_weight, fc2_bias):
    inp = torch.vstack([x, y]).double()
    hidden = torch.relu(inp @ fc1_weight.double().T + fc1_bias.double())
    return torch.sigmoid(hidden @ fc2_weight.double().T + fc2_bias.double())
