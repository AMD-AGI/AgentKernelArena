"""High-precision reference for the ungated feed-forward block, used only for correctness."""
import torch


def reference(*, x, w_up, w_down):
    x64 = x.double()
    pre = x64 @ w_up.double().T
    hidden = 0.5 * pre * (1.0 + torch.tanh(0.7978845608028654 * (pre + 0.044715 * pre ** 3)))  # GELU (tanh form)
    return hidden @ w_down.double().T
