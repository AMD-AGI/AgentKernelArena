import torch


def _rope_interleaved(x, cos, sin):
    """x: [T, H, D]; cos, sin: [T, D // 2] (float64). Pairs (x[2i], x[2i+1]) rotate together."""
    cos = cos.repeat_interleave(2, dim=-1)[:, None, :]
    sin = sin.repeat_interleave(2, dim=-1)[:, None, :]
    rotated = torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2)
    return x * cos + rotated * sin


def reference(*, q_nope, q_pe, k_nope, k_pe, positions, slot_mapping, cos, sin, kv_cache):
    n_tokens, heads, lora = q_nope.shape
    pe = q_pe.shape[-1]
    cos_t = cos.double()[positions]
    sin_t = sin.double()[positions]
    q_out = torch.empty((n_tokens, heads, lora + pe), dtype=torch.float64, device=q_nope.device)
    q_out[..., :lora] = q_nope
    q_out[..., lora:] = _rope_interleaved(q_pe.double(), cos_t, sin_t)
    k_pe_out = _rope_interleaved(k_pe.double(), cos_t, sin_t)
    cache = kv_cache.double()
    cache[slot_mapping] = torch.cat((k_nope.double(), k_pe_out), dim=-1)
    return {"q_out": q_out, "k_pe_out": k_pe_out, "kv_cache": cache}
