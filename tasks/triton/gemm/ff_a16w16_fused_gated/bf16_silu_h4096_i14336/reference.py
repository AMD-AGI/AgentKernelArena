"""Float64 reference of the gated feed-forward block, used only for correctness."""
import torch


def reference(*, x, w_up, w_down):
    n_out = w_up.shape[0] // 2
    pre = x.double() @ w_up.double().T
    gate, up = pre[:, :n_out], pre[:, n_out:]
    hidden = gate * torch.sigmoid(gate) * up  # silu(gate) * up, not rounded to bfloat16
    return hidden @ w_down.double()
