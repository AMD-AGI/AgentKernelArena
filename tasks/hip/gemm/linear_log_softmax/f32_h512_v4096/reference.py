"""High-precision reference for the fused linear layer and log_softmax, used only for correctness."""
import torch


def reference(*, x, weight, bias):
    logits = x.double() @ weight.double().T + bias.double()
    return torch.log_softmax(logits, dim=-1)
