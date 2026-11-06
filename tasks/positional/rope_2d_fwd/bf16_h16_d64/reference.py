import torch

_CHUNK = 64  # images per float64 chunk, bounding temporaries


def _rotate_half(v):
    half = v.shape[-1] // 2
    return torch.cat((-v[..., half:], v[..., :half]), dim=-1)


def reference(*, x, cos_h, sin_h, cos_w, sin_w):
    batch, seq_len, heads, head_dim = x.shape
    height, width = cos_h.shape[1], cos_w.shape[1]
    out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    ch, sh = cos_h.double()[:, :, None], sin_h.double()[:, :, None]  # [1, H, 1, 1, d/2], broadcast over width
    cw, sw = cos_w.double()[:, None], sin_w.double()[:, None]  # [1, 1, W, 1, d/2], broadcast over height
    for start in range(0, batch, _CHUNK):
        end = min(start + _CHUNK, batch)
        grid = x[start:end].double().reshape(end - start, height, width, heads, head_dim)
        first, second = grid.chunk(2, dim=-1)
        first = first * ch + _rotate_half(first) * sh
        second = second * cw + _rotate_half(second) * sw
        out[start:end] = torch.cat([first, second], dim=-1).reshape(end - start, seq_len, heads, head_dim).float()
    return out
