"""Block-quantized int8 operands: activations with one scale per 1x128 token group, weights per 128x128 block.

Both use the symmetric int8 rule of vLLM's per-token-group quantization: scale = amax / 127 (floored at 1e-10),
code = round(value / scale) clamped to [-128, 127].
"""
import torch

_BLOCK = 128
_INT8_MAX = 127.0
_EPS = 1e-10


def _quantize_groups(values, group_shape):
    """Quantize `values` [rows, cols] in groups of group_shape = (rows_per_group, cols_per_group)."""
    rows, cols = values.shape
    gr, gc = group_shape
    blocks = values.view(rows // gr, gr, cols // gc, gc)
    scale = (blocks.abs().amax(dim=(1, 3)) / _INT8_MAX).clamp_min(_EPS)
    codes = (blocks / scale[:, None, :, None]).round_().clamp_(-128, 127).to(torch.int8)
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
