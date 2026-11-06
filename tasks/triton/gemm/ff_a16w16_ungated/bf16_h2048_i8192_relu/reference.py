"""High-precision reference for the ungated feed-forward block, used only for correctness."""
import torch


def reference(*, x, w_up, w_down):
    x64 = x.double()
    pre = x64 @ w_up.double().T
    hidden = torch.relu(pre)
    return hidden @ w_down.double().T
