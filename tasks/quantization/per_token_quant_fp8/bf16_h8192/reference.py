"""Float64 reference for dynamic per-token FP8 (e4m3fn) quantization, used only for correctness."""
import torch

FP8_MAX = 448.0  # largest finite float8_e4m3fn
_CHUNK_ROWS = 4096


def reference(*, x):
    codes = torch.empty(x.shape, dtype=torch.float64, device=x.device)
    scales = torch.empty((x.shape[0], 1), dtype=torch.float64, device=x.device)
    for start in range(0, x.shape[0], _CHUNK_ROWS):
        rows = x[start:start + _CHUNK_ROWS].double()
        scale = rows.abs().amax(dim=-1, keepdim=True) / FP8_MAX
        codes[start:start + _CHUNK_ROWS] = rows / scale
        scales[start:start + _CHUNK_ROWS] = scale
    return codes, scales
