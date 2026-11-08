"""Block-scaled fp8 operands as the quantizing producers emit them: activations with one scale per 1x128 block, weights
with one scale per 128x128 block (scale = block amax / 448).

The activations are the intermediate of a shared expert (standard normal, so the block scales are realistic), the weight
is a normal weight with std 0.06 so that the product (std about 1) is comparable in size to a * b (std 1.0).
"""
import torch

_FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
_BLOCK = 128
_ROW_CHUNK = 16384


def _quantize_rows(values):
    """values [rows, k] -> float8_e4m3fn codes [rows, k] and float32 scales [rows, k // 128] (block amax / 448)."""
    rows, k = values.shape
    codes = torch.empty(rows, k, dtype=torch.float8_e4m3fn, device=values.device)
    scales = torch.empty(rows, k // _BLOCK, dtype=torch.float32, device=values.device)
    for start in range(0, rows, _ROW_CHUNK):
        stop = min(start + _ROW_CHUNK, rows)
        blocks = values[start:stop].float().view(stop - start, k // _BLOCK, _BLOCK)
        scale = (blocks.abs().amax(dim=-1) / _FP8_MAX).clamp_min(1e-12)
        codes[start:stop] = (blocks / scale.unsqueeze(-1)).clamp_(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn).view(stop - start, k)
        scales[start:stop] = scale
    return codes, scales


def _quantize_blocks(values):
    """values [n, k] -> float8_e4m3fn codes [n, k] and float32 scales [n // 128, k // 128], one per 128x128 block."""
    n, k = values.shape
    blocks = values.view(n // _BLOCK, _BLOCK, k // _BLOCK, _BLOCK)
    scale = (blocks.abs().amax(dim=(1, 3)) / _FP8_MAX).clamp_min(1e-12)
    full = scale.repeat_interleave(_BLOCK, dim=0).repeat_interleave(_BLOCK, dim=1)
    codes = (values / full).clamp_(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return codes, scale


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    m, n, k = axes["m"], axes["n"], axes["k"]
    activations = torch.randn(m, k, generator=generator, device=device, dtype=torch.float32)
    x, x_scales = _quantize_rows(activations)
    weights = 0.06 * torch.randn(n, k, generator=generator, device=device, dtype=torch.float32)
    w, w_scales = _quantize_blocks(weights)
    return {"x": x, "x_scales": x_scales, "w": w, "w_scales": w_scales}
