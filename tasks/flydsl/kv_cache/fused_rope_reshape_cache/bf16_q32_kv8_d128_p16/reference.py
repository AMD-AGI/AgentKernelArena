import torch


def _rope_neox(x, cos, sin):
    """x: [T, H, D]; cos, sin: [T, D // 2] (float64). Rotate-half pairing (x[i], x[i + D // 2])."""
    half = x.shape[-1] // 2
    cos = cos[:, None, :]
    sin = sin[:, None, :]
    first, second = x[..., :half], x[..., half:]
    return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)


def reference(*, q, k, v, positions, cos_cache, sin_cache, slot_mapping, key_cache, value_cache):
    pos = positions.long()
    cos = cos_cache.double()[pos]
    sin = sin_cache.double()[pos]
    q_out = _rope_neox(q.double(), cos, sin)
    k_out = _rope_neox(k.double(), cos, sin)
    block_size = key_cache.shape[1]
    slots = slot_mapping.long()
    valid = slots >= 0
    blocks, offsets = slots[valid] // block_size, slots[valid] % block_size
    key_new = key_cache.double()
    value_new = value_cache.double()
    key_new[blocks, offsets] = k_out[valid]
    value_new[blocks, offsets] = v.double()[valid]
    return {"q_out": q_out, "k_out": k_out, "key_cache": key_new, "value_cache": value_new}
