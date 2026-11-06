"""Float64 softmax(gelu(x @ weight.T + bias), dim=1) used only for correctness."""
import torch
import torch.nn.functional as F


def reference(*, x, weight, bias):
    y = torch.addmm(bias.double(), x.double(), weight.double().t())
    return torch.softmax(F.gelu(y), dim=1)
