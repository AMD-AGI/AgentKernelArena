"""Interleaved-RoPE cos/sin tables (base 10000, bfloat16), as the model's rotary cache holds them."""
import torch


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    max_position, rope_dim = axes["max_position"], axes["rope_dim"]
    inv_freq = 10000.0 ** (-torch.arange(0, rope_dim, 2, dtype=torch.float64, device=device) / rope_dim)
    angles = torch.arange(max_position, dtype=torch.float64, device=device)[:, None] * inv_freq[None, :]
    return {"cos": torch.cos(angles).to(torch.bfloat16), "sin": torch.sin(angles).to(torch.bfloat16)}
