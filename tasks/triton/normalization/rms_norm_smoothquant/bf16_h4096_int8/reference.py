"""Float64 reference of RMSNorm, SmoothQuant and per-token INT8 quantization."""
import torch

CHUNK_ROWS = 1 << 14
INT8_MAX = 127.0


def reference(*, x, weight, x_scale, epsilon):
    n_tokens = x.shape[0]
    codes = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    scales = torch.empty((n_tokens,), dtype=torch.float64, device=x.device)
    gain = weight.double() * x_scale.double()
    for start in range(0, n_tokens, CHUNK_ROWS):
        x64 = x[start:start + CHUNK_ROWS].double()
        rstd = torch.rsqrt(x64.square().mean(dim=-1, keepdim=True) + epsilon)
        hidden = x64 * rstd * gain
        amax = hidden.abs().amax(dim=-1, keepdim=True)
        scale = amax / INT8_MAX
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        q = hidden / scale
        rounded = torch.where(q >= 0, torch.floor(q + 0.5), torch.ceil(q - 0.5))
        codes[start:start + CHUNK_ROWS] = rounded.clamp(-128, 127).to(torch.int8)
        scales[start:start + CHUNK_ROWS] = scale.squeeze(-1)
    return {"out": codes, "y_scale": scales}
