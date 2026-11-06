import torch

_EPS = 1.0e-6
_CHUNK = 8192  # tokens per float64 chunk, bounding temporaries


def _rope_tail(x, cos, sin, rope_dim):
    """GPT-J pair-interleaved RoPE of the last rope_dim channels; x is [t, ..., D], cos/sin [t, rope_dim//2]."""
    lead = x.shape[1:-1]
    shape = (x.shape[0],) + (1,) * len(lead) + (rope_dim // 2,)
    cos = cos.reshape(shape)
    sin = sin.reshape(shape)
    nope = x.shape[-1] - rope_dim
    pairs = x[..., nope:].reshape(*x.shape[:-1], rope_dim // 2, 2)
    even, odd = pairs[..., 0], pairs[..., 1]
    rotated = torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1).reshape(*x.shape[:-1], rope_dim)
    return torch.cat([x[..., :nope], rotated], dim=-1)


def reference(*, q, kv, kv_weight, cos_cache, sin_cache, positions):
    n_tokens, _, head_dim = q.shape
    rope_dim = 2 * cos_cache.shape[-1]
    q_out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    kv_out = torch.empty(kv.shape, dtype=torch.float32, device=q.device)
    weight = kv_weight.double()
    for start in range(0, n_tokens, _CHUNK):
        end = min(start + _CHUNK, n_tokens)
        pos = positions[start:end].long()
        cos = cos_cache[pos].double()
        sin = sin_cache[pos].double()
        q64 = q[start:end].double()
        q64 = q64 * torch.rsqrt(q64.square().mean(-1, keepdim=True) + _EPS)
        kv64 = kv[start:end].double()
        kv64 = kv64 * torch.rsqrt(kv64.square().mean(-1, keepdim=True) + _EPS) * weight
        q_out[start:end] = _rope_tail(q64, cos, sin, rope_dim).float()
        kv_out[start:end] = _rope_tail(kv64, cos, sin, rope_dim).float()
    return {"q_out": q_out, "kv_out": kv_out}
