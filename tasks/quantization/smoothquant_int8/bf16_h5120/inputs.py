"""Activation generator of the SmoothQuant INT8 scenario.

Standard-normal rows, with every 97th row (starting at row 5) all zero so the exact comparison also
covers the all-zero-row convention (row scale 1, zero codes) that a random draw never produces.
"""
import torch


def make_inputs(case, inputs, generator, device):
    rows, width = case["axes"]["n_tokens"], case["axes"]["hidden_size"]
    x = torch.randn((rows, width), generator=generator, device=device, dtype=torch.float32)
    x[torch.arange(5, max(rows, 5), 97, device=device)] = 0.0
    return {"x": x.to(torch.bfloat16)}
