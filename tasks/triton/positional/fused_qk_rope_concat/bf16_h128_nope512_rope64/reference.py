import torch


def _rope_interleaved(x, cos, sin):
    """x: [T, H, D]; cos, sin: [T, D // 2] (float64). Pairs (x[2i], x[2i+1]) rotate together."""
    cos = cos.repeat_interleave(2, dim=-1)[:, None, :]
    sin = sin.repeat_interleave(2, dim=-1)[:, None, :]
    rotated = torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2)
    return x * cos + rotated * sin


def reference(*, q_nope, q_pe, k_nope, k_pe, positions, cos, sin):
    cos_t = cos.double()[positions]
    sin_t = sin.double()[positions]
    q_out = torch.cat((q_nope.double(), _rope_interleaved(q_pe.double(), cos_t, sin_t)), dim=-1)
    k_out = torch.cat((k_nope.double(), _rope_interleaved(k_pe.double(), cos_t, sin_t)), dim=-1)
    return {"q_out": q_out, "k_out": k_out}
