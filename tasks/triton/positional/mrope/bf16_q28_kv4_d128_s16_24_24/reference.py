import torch


def reference(*, q, k, cos_sin_cache, positions):
    head_dim = 128
    section = (16, 24, 24)  # scenario constants: temporal / height / width sections of the half head
    half = head_dim // 2
    cache = cos_sin_cache.double()
    cos_table, sin_table = cache[:, :half], cache[:, half:]
    owner = torch.repeat_interleave(torch.arange(3, device=q.device), torch.tensor(section, device=q.device))
    cols = torch.arange(half, device=q.device)
    pos = positions.long()[owner]  # [half, T]: the position that governs each rotary index
    cos = cos_table[pos, cols[:, None]].T[:, None, :]  # [T, 1, half]
    sin = sin_table[pos, cols[:, None]].T[:, None, :]

    def rope(x):
        t = x.double().reshape(x.shape[0], -1, head_dim)
        first, second = t[..., :half], t[..., half:]
        out = torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)
        return out.reshape(x.shape)

    return {"q_out": rope(q), "k_out": rope(k)}
