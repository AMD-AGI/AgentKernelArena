"""Block-quantized fp8 operands and their kernel-facing layout.

Logical operands (what the reference sees): a [m, k] fp8, a_scale [m, k // 128] (one scale per 1x128 token group),
b [n, k] fp8, b_scale [n // 128, k // 128] (one scale per 128x128 block). Scales are amax / 448 of their group
(the e4m3fn maximum) and codes are value / scale cast to float8_e4m3fn.

Kernel-facing layout (what the baseline and a candidate receive), as the AITER preshuffle GEMM consumes it:
  * b is the (16, 16) preshuffled weight: [n, k] viewed as [n // 16, 16, k // 32, 2, 16], permuted to
    [n // 16, k // 32, 2, 16, 16] and flattened to [n // 16, k * 16] (weight preparation done once at load time).
  * a_scale is stored transposed, [k // 128, m]: the activation quantization kernel writes its scales
    column-major, so element (row r, group g) lives at g * m + r.
"""
import torch

_BLOCK = 128
_FP8_MAX = 448.0  # e4m3fn saturation magnitude


def _quantize_groups(values, group_shape):
    """Quantize `values` [rows, cols] in groups of group_shape = (rows_per_group, cols_per_group)."""
    rows, cols = values.shape
    gr, gc = group_shape
    blocks = values.view(rows // gr, gr, cols // gc, gc)
    scale = (blocks.abs().amax(dim=(1, 3)) / _FP8_MAX).clamp_min(1e-12)
    codes = (blocks / scale[:, None, :, None]).clamp_(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return codes.view(rows, cols), scale


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    m, n, k = axes["m"], axes["n"], axes["k"]
    activations = torch.randn(m, k, generator=generator, device=device, dtype=torch.float32)
    a, a_scale = _quantize_groups(activations, (1, _BLOCK))
    del activations
    weights = 0.02 * torch.randn(n, k, generator=generator, device=device, dtype=torch.float32)
    b, b_scale = _quantize_groups(weights, (_BLOCK, _BLOCK))
    return {"a": a, "a_scale": a_scale, "b": b, "b_scale": b_scale}


def to_kernel_layout(case, inputs):
    inputs = dict(inputs)
    b = inputs["b"]
    n, k = b.shape
    raw = b.view(torch.uint8).view(n // 16, 16, k // 32, 2, 16)
    inputs["b"] = raw.permute(0, 2, 3, 1, 4).contiguous().view(n // 16, k * 16).view(torch.float8_e4m3fn)
    inputs["a_scale"] = inputs["a_scale"].t().contiguous()
    return inputs
