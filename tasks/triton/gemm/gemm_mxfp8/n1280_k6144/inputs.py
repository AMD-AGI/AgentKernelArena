"""MXFP8 operands for the dense GEMM: per-token activations and a persisted weight.

Both operands are quantized the way the serving stack produces them: values in blocks of 32 along K share an E8M0
scale (byte b means 2^(b - 127)) chosen as ceil(log2(amax / 448)) + 127, and the scaled values are rounded to
float8_e4m3fn. The activation is drawn as N(0, 0.5^2) and the weight as N(0, 0.1^2) in bfloat16 before quantization.
"""
import torch

_BLOCK = 32
_FP8_MAX = 448.0
_ROW_CHUNK = 16384


def _quantize(x):
    """Quantize [rows, K] (K % 32 == 0) to (float8_e4m3fn values, uint8 E8M0 scales [rows, K // 32])."""
    rows, k = x.shape
    blocks = x.to(torch.float32).view(rows, k // _BLOCK, _BLOCK)
    amax = blocks.abs().amax(dim=-1).clamp(min=torch.finfo(torch.float32).tiny)
    biased = (torch.ceil(torch.log2(amax / _FP8_MAX)) + 127.0).clamp(0, 254)
    descale = torch.exp2(biased - 127.0)
    scaled = (blocks / descale.unsqueeze(-1)).clamp(-_FP8_MAX, _FP8_MAX)
    return scaled.view(rows, k).to(torch.float8_e4m3fn), biased.to(torch.uint8)


def _draw(rows, k, std, generator, device):
    values = torch.empty(rows, k, dtype=torch.float8_e4m3fn, device=device)
    scales = torch.empty(rows, k // _BLOCK, dtype=torch.uint8, device=device)
    for start in range(0, rows, _ROW_CHUNK):
        stop = min(start + _ROW_CHUNK, rows)
        bf16 = (torch.randn(stop - start, k, generator=generator, device=device, dtype=torch.float32) * std).to(
            torch.bfloat16)
        values[start:stop], scales[start:stop] = _quantize(bf16)
    return values, scales


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    m, n, k = axes["m"], axes["n"], axes["k"]
    x, x_scale = _draw(m, k, 0.5, generator, device)
    w, w_scale = _draw(n, k, 0.1, generator, device)
    return {"x": x, "x_scale": x_scale, "w": w, "w_scale": w_scale}
