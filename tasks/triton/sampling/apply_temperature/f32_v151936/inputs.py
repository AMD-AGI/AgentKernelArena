"""Inputs for apply_temperature: heterogeneous per-request temperatures."""
import torch


def make_inputs(case, inputs, generator, device):
    n_tokens = case["axes"]["n_tokens"]
    temperature = 0.3 + 1.2 * torch.rand(n_tokens, generator=generator, device=device, dtype=torch.float32)
    ids = torch.arange(n_tokens, device=device)
    temperature = torch.where(ids % 16 == 2, torch.zeros_like(temperature), temperature)   # greedy
    temperature = torch.where(ids % 16 == 9, torch.ones_like(temperature), temperature)    # no-op
    return {"temperature": temperature}
