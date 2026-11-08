"""Float64 reference of the fused QKV split, QK RMSNorm, partial NeoX RoPE and paged KV write (gated, NHD)."""
import torch


def _rms_norm(x, weight, eps):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * (1.0 + weight.double())


def _rope_neox_partial(x, cos, sin):
    """x [T, H, D]; cos, sin [T, R // 2]. Rotate-half over the first R values, the rest passes through."""
    half = cos.shape[-1]
    cos, sin = cos[:, None, :], sin[:, None, :]
    x1, x2 = x[..., :half], x[..., half:2 * half]
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin, x[..., 2 * half:]), dim=-1)


def reference(*, qkv, q_weight, k_weight, cos, sin, positions, slot_mapping, key_cache, value_cache, eps):
    n_tokens = qkv.shape[0]
    head_dim = q_weight.shape[0]
    kv_heads = key_cache.shape[2]
    q_heads = (qkv.shape[1] // head_dim - 2 * kv_heads) // 2
    block_size = key_cache.shape[1]
    x = qkv.double()
    qg = x[:, :2 * q_heads * head_dim].reshape(n_tokens, q_heads, 2, head_dim)
    q_in, gate = qg[:, :, 0], qg[:, :, 1]
    k_in = x[:, 2 * q_heads * head_dim:(2 * q_heads + kv_heads) * head_dim].reshape(n_tokens, kv_heads, head_dim)
    v = x[:, (2 * q_heads + kv_heads) * head_dim:].reshape(n_tokens, kv_heads, head_dim)
    cos_t = cos.double()[positions]
    sin_t = sin.double()[positions]
    eps = float(eps)
    q = _rope_neox_partial(_rms_norm(q_in, q_weight, eps), cos_t, sin_t)
    k = _rope_neox_partial(_rms_norm(k_in, k_weight, eps), cos_t, sin_t)
    key = key_cache.double()
    value = value_cache.double()
    key[slot_mapping // block_size, slot_mapping % block_size] = k
    value[slot_mapping // block_size, slot_mapping % block_size] = v
    return {"q": q, "gate": gate, "k": k, "v": v, "key_cache": key, "value_cache": value}
