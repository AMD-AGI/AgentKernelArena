import torch

_CHUNK = 16384  # positions per float64 chunk, bounding temporaries


def reference(*, x, freqs):
    seq_len, _, _, head_dim = x.shape
    rot = head_dim // 2  # rotary slice width
    half = rot // 2
    out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    for start in range(0, seq_len, _CHUNK):
        end = min(start + _CHUNK, seq_len)
        angles = freqs[start:end].double()
        cos = torch.cos(angles).repeat(1, 1, 1, 2)
        sin = torch.sin(angles).repeat(1, 1, 1, 2)
        x64 = x[start:end].double()
        part, rest = x64[..., :rot], x64[..., rot:]
        rotated = torch.cat((-part[..., half:], part[..., :half]), dim=-1)
        out[start:end] = torch.cat([part * cos + rotated * sin, rest], dim=-1).float()
    return out
