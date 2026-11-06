"""float64 identity-expert combine reference, used only for correctness."""
import torch


def reference(*, hidden_states, expert_scales):
    return hidden_states.double() * expert_scales.double().sum(dim=-1, keepdim=True)
