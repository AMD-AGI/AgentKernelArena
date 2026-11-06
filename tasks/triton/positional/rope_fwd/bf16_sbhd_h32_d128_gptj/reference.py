import torch

_CHUNK = 16384  # positions per float64 chunk, bounding temporaries


def reference(*, x, freqs):
    seq_len, _, _, head_dim = x.shape
    half = head_dim // 2
    out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    for start in range(0, seq_len, _CHUNK):
        end = min(start + _CHUNK, seq_len)
        angles = freqs[start:end].double()
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        pairs = x[start:end].double().reshape(*x[start:end].shape[:-1], half, 2)
        even, odd = pairs[..., 0], pairs[..., 1]
        rotated = torch.stack([even * cos - odd * sin, odd * cos + even * sin], dim=-1)
        out[start:end] = rotated.reshape(x[start:end].shape).float()
    return out
