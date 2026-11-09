"""NeoX RoPE tables (base 10^7, bfloat16) over the rotated 64 values, and unique cache slots."""
import torch

_ROPE_THETA = 10000000.0


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    n_tokens, max_position, rotary_dim = axes["n_tokens"], axes["max_position"], axes["rotary_dim"]
    inv_freq = _ROPE_THETA ** (-torch.arange(0, rotary_dim, 2, dtype=torch.float64, device=device) / rotary_dim)
    angles = torch.arange(max_position, dtype=torch.float64, device=device)[:, None] * inv_freq[None, :]
    slots = torch.randperm(axes["num_slots"], generator=generator, device=device)[:n_tokens]
    return {"cos": torch.cos(angles).to(torch.bfloat16), "sin": torch.sin(angles).to(torch.bfloat16),
            "slot_mapping": slots.to(torch.int64)}
