"""Float64 reference of two RMSNorms with FP8 per-group quantization of the first (group size 128)."""
import torch

GROUP_SIZE = 128
FP8_MAX = 448.0  # float8_e4m3fn
EPS = 1e-10
CHUNK_ROWS = 1 << 16


def _rms_norm(x, weight, epsilon):
    x64 = x.double()
    return x64 * torch.rsqrt(x64.square().mean(dim=-1, keepdim=True) + epsilon) * weight.double()


def reference(*, x1, weight1, x2, weight2, epsilon1, epsilon2):
    n_tokens, n1 = x1.shape
    n_groups = n1 // GROUP_SIZE
    codes = torch.empty((n_tokens, n1), dtype=torch.float64, device=x1.device)
    scales = torch.empty((n_tokens, n_groups), dtype=torch.float64, device=x1.device)
    out2 = torch.empty(x2.shape, dtype=torch.float64, device=x1.device)
    for start in range(0, n_tokens, CHUNK_ROWS):
        stop = start + CHUNK_ROWS
        y = _rms_norm(x1[start:stop], weight1, epsilon1).reshape(-1, n_groups, GROUP_SIZE)
        scale = y.abs().amax(dim=-1).clamp(min=EPS) / FP8_MAX
        codes[start:stop] = (y / scale[..., None]).clamp(-FP8_MAX, FP8_MAX).reshape(-1, n1)
        scales[start:stop] = scale
        out2[start:stop] = _rms_norm(x2[start:stop], weight2, epsilon2)
    return {"out1_q": codes, "out1_scale": scales, "out2": out2}
