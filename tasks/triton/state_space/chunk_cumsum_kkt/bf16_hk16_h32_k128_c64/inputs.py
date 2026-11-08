"""Raw per-token gate log-decays of the Gated DeltaNet prefill.

In production `g` is `-exp(A_log) * softplus(a + dt_bias)`, which is non-positive. The generator draws
such per-token log-decays: `a + dt_bias ~ N(0, 1)` and a per-head rate `exp(A_log)` that spans
0.02 to 0.4 (log-uniform), so that a head keeps anywhere from about exp(-1) to exp(-20) of its state
over a 64-token chunk. These rates are a design choice of this task, not values of a trained model.
"""
import math

import torch


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    tokens, heads = axes["n_tokens"], axes["num_heads"]
    low, high = math.log(0.02), math.log(0.4)
    rate = torch.exp(torch.rand(heads, generator=generator, device=device, dtype=torch.float32) * (high - low) + low)
    x = torch.randn(tokens, heads, generator=generator, device=device, dtype=torch.float32)
    g = -rate * torch.nn.functional.softplus(x)
    return {"g": g.contiguous()}
