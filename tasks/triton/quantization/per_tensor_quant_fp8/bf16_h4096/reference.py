"""Float64 reference for dynamic per-tensor FP8 (e4m3fn) quantization, used only for correctness."""
import torch

FP8_MAX = 448.0  # largest finite float8_e4m3fn
_CHUNK_ROWS = 4096


def reference(*, x):
    amax = torch.zeros((), dtype=torch.float64, device=x.device)
    for start in range(0, x.shape[0], _CHUNK_ROWS):
        amax = torch.maximum(amax, x[start:start + _CHUNK_ROWS].double().abs().max())
    scale = amax / FP8_MAX
    codes = torch.empty(x.shape, dtype=torch.float64, device=x.device)
    for start in range(0, x.shape[0], _CHUNK_ROWS):
        chunk = x[start:start + _CHUNK_ROWS].double() / scale
        codes[start:start + _CHUNK_ROWS] = chunk.clamp(-FP8_MAX, FP8_MAX)
    return codes, scale.reshape(1)
