"""Float64 linear layer followed by a softmax over the output features, used only for correctness."""
import torch


def reference(*, x, weight, bias):
    logits = torch.addmm(bias.double(), x.double(), weight.double().t())
    return torch.softmax(logits, dim=-1)
