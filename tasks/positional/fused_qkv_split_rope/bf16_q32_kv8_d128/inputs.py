import torch


def make_inputs(case, inputs, generator, device):
    """NeoX cos/sin tables (base 1e6, first-half angles) rounded to bfloat16."""
    axes = case["axes"]
    max_position, head_dim = axes["max_position"], axes["head_dim"]
    inv_freq = 1000000.0 ** (-torch.arange(0, head_dim, 2, dtype=torch.float64, device=device) / head_dim)
    angles = torch.arange(max_position, dtype=torch.float64, device=device)[:, None] * inv_freq[None, :]
    return {"cos": torch.cos(angles).to(torch.bfloat16), "sin": torch.sin(angles).to(torch.bfloat16)}
