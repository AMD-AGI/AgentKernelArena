"""Float64 reference of RMSNorm followed by per-token dynamic FP8 quantization."""
import torch

FP8_MAX = 448.0  # float8_e4m3fn
CHUNK_ROWS = 1 << 16


def reference(*, x, weight, epsilon):
    n_tokens, hidden = x.shape
    codes = torch.empty(x.shape, dtype=torch.float64, device=x.device)
    scales = torch.empty((n_tokens, 1), dtype=torch.float64, device=x.device)
    w64 = weight.double()
    for start in range(0, n_tokens, CHUNK_ROWS):
        stop = start + CHUNK_ROWS
        x64 = x[start:stop].double()
        y = x64 * torch.rsqrt(x64.square().mean(dim=-1, keepdim=True) + epsilon) * w64
        scale = y.abs().amax(dim=-1, keepdim=True) / FP8_MAX
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        codes[start:stop] = y / scale
        scales[start:stop] = scale
    return {"out": codes, "scale": scales}
