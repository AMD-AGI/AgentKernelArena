"""Float64 reference of silu-and-mul with per-token-group FP8 quantization (group size 128)."""
import torch

GROUP_SIZE = 128
FP8_MAX = 448.0  # float8_e4m3fn
EPS = 1e-10
CHUNK_ROWS = 8192


def reference(*, x):
    n_tokens, width = x.shape
    d = width // 2
    n_groups = d // GROUP_SIZE
    codes = torch.empty((n_tokens, d), dtype=torch.float64, device=x.device)
    scales = torch.empty((n_groups, n_tokens), dtype=torch.float64, device=x.device)
    for start in range(0, n_tokens, CHUNK_ROWS):
        block = x[start:start + CHUNK_ROWS].double()
        gate, up = block[:, :d], block[:, d:]
        y = (gate * torch.sigmoid(gate) * up).reshape(-1, n_groups, GROUP_SIZE)
        scale = (y.abs().amax(dim=-1) .clamp(min=EPS)) / FP8_MAX
        codes[start:start + CHUNK_ROWS] = (y / scale[..., None]).clamp(-FP8_MAX, FP8_MAX).reshape(-1, d)
        scales[:, start:start + CHUNK_ROWS] = scale.t()
    return codes, scales
