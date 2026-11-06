import torch


def _table(count, half_dim, device):
    """cos/sin of position * 10000^(-k / half_dim), shape [1, count, 1, half_dim], rounded to bfloat16."""
    inv_freq = 10000.0 ** (-torch.arange(half_dim, dtype=torch.float64, device=device) / half_dim)
    angles = torch.arange(count, dtype=torch.float64, device=device)[:, None] * inv_freq[None, :]
    shape = (1, count, 1, half_dim)
    return torch.cos(angles).to(torch.bfloat16).reshape(shape), torch.sin(angles).to(torch.bfloat16).reshape(shape)


def make_inputs(case, inputs, generator, device):
    """Height and width cos/sin tables with the standard RoPE frequency schedule over head_dim // 2 pairs."""
    axes = case["axes"]
    cos_h, sin_h = _table(axes["height"], axes["half_dim"], device)
    cos_w, sin_w = _table(axes["width"], axes["half_dim"], device)
    return {"cos_h": cos_h, "sin_h": sin_h, "cos_w": cos_w, "sin_w": sin_w}
