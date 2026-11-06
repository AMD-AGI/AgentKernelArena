import torch


def make_inputs(case, inputs, generator, device):
    """Row-major [cos | sin] cache (base 1e6, float32) as the production rotary embedding stores it."""
    axes = case["axes"]
    max_position, head_dim = axes["max_position"], axes["head_dim"]
    inv_freq = 1000000.0 ** (-torch.arange(0, head_dim, 2, dtype=torch.float64, device=device) / head_dim)
    angles = torch.arange(max_position, dtype=torch.float64, device=device)[:, None] * inv_freq[None, :]
    cache = torch.cat((torch.cos(angles), torch.sin(angles)), dim=1).to(torch.float32)
    return {"cos_sin_cache": cache}
