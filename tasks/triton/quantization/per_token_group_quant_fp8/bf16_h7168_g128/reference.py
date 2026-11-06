"""Float64 reference for per-token-group FP8 (e4m3fn) quantization, used only for correctness."""
import torch

FP8_MAX = 448.0  # largest finite float8_e4m3fn
GROUP_SIZE = 128
EPS = 1e-10
_CHUNK_ROWS = 2048


def reference(*, x):
    rows, width = x.shape
    groups = width // GROUP_SIZE
    codes = torch.empty(x.shape, dtype=torch.float64, device=x.device)
    scales = torch.empty((rows, groups), dtype=torch.float64, device=x.device)
    for start in range(0, rows, _CHUNK_ROWS):
        g = x[start:start + _CHUNK_ROWS].double().reshape(-1, groups, GROUP_SIZE)
        scale = g.abs().amax(dim=-1, keepdim=True).clamp(min=EPS) / FP8_MAX
        codes[start:start + _CHUNK_ROWS] = (g / scale).clamp(-FP8_MAX, FP8_MAX).reshape(-1, width)
        scales[start:start + _CHUNK_ROWS] = scale.squeeze(-1)
    return codes, scales
