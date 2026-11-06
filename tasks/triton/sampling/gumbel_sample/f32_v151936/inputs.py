"""Inputs for gumbel_sample: logits, heterogeneous per-request temperatures and positions."""
import torch


def make_inputs(case, inputs, generator, device):
    n_tokens = case["axes"]["n_tokens"]
    logits = 3.0 * torch.randn(n_tokens, case["axes"]["vocab_size"], generator=generator,
                               device=device, dtype=torch.float32)
    temperature = 0.3 + 1.2 * torch.rand(n_tokens, generator=generator, device=device, dtype=torch.float32)
    ids = torch.arange(n_tokens, device=device)
    temperature = torch.where(ids % 8 == 3, torch.zeros_like(temperature), temperature)   # greedy requests
    return {"logits": logits, "temperature": temperature}
