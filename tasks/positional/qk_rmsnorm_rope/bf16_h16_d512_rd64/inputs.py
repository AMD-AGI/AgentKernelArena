import torch


def make_inputs(case, inputs, generator, device):
    """GPT-J RoPE tables over the rotary tail: angle[p, k] = p * 10000^(-2k / rope_dim), rounded to bfloat16."""
    axes = case["axes"]
    rope_dim, max_position = axes["rope_dim"], axes["max_position"]
    inv_freq = 10000.0 ** (-torch.arange(0, rope_dim, 2, dtype=torch.float64, device=device) / rope_dim)
    angles = torch.arange(max_position, dtype=torch.float64, device=device)[:, None] * inv_freq[None, :]
    return {"cos_cache": torch.cos(angles).to(torch.bfloat16),
            "sin_cache": torch.sin(angles).to(torch.bfloat16)}
