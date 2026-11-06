"""High-precision experts-combine reference used only for correctness."""
import math

import torch

CHUNK = 16384


def reference(*, moe_hidden_states, mlp_hidden_states):
    out = torch.empty(mlp_hidden_states.shape, dtype=torch.float64, device=mlp_hidden_states.device)
    scale = 1.0 / math.sqrt(2.0)
    for start in range(0, out.shape[0], CHUNK):
        stop = start + CHUNK
        routed = moe_hidden_states[start:stop].double().sum(dim=1)
        out[start:stop] = (routed + mlp_hidden_states[start:stop].double()) * scale
    return out
