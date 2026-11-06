import torch


def reference(*, x, cu_seqlens, freqs):
    """NeoX RoPE per sequence in float64; the position within a sequence indexes the angle table."""
    n_tokens, _, head_dim = x.shape
    half = head_dim // 2
    bounds = cu_seqlens.long().tolist()
    out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    angles = freqs.double().reshape(freqs.shape[0], half)
    chunk = 8192
    for begin, end in zip(bounds[:-1], bounds[1:]):
        for lo in range(begin, end, chunk):
            hi = min(lo + chunk, end)
            ang = angles[lo - begin:hi - begin][:, None, :]
            cos, sin = torch.cos(ang), torch.sin(ang)
            xs = x[lo:hi].double()
            first, second = xs[..., :half], xs[..., half:]
            out[lo:hi, :, :half] = (first * cos - second * sin).float()
            out[lo:hi, :, half:] = (second * cos + first * sin).float()
    return out
