import torch


def reference(*, qkv, cos, sin, positions):
    half = cos.shape[1]
    head_dim = 2 * half
    q_heads, kv_heads = 32, 8  # scenario constants (num_q_heads, num_kv_heads)
    total_heads = q_heads + 2 * kv_heads
    x = qkv.double().reshape(qkv.shape[0], total_heads, head_dim)
    q, k, v = x[:, :q_heads], x[:, q_heads:q_heads + kv_heads], x[:, q_heads + kv_heads:]
    pos = positions.long()
    c = cos.double()[pos][:, None, :]
    s = sin.double()[pos][:, None, :]

    def rope(t):
        first, second = t[..., :half], t[..., half:]
        return torch.cat((first * c - second * s, second * c + first * s), dim=-1)

    return {"q": rope(q), "k": rope(k), "v": v.clone()}
