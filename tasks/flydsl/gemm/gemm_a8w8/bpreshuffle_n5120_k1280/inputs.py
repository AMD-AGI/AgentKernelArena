"""Per-token and per-channel fp8 quantization of normal data, and the (16, 16) weight preshuffle."""
import torch

_FP8_MAX = 448.0  # e4m3fn saturation magnitude


def _quantize_rows(values):
    """fp8 codes and float32 row scales (row amax / 448) of a [rows, k] float32 matrix."""
    scale = values.abs().amax(dim=-1, keepdim=True) / _FP8_MAX
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return (values / scale).to(torch.float8_e4m3fn), scale


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    m, n, k = axes["m"], axes["n"], axes["k"]
    xq, x_scale = _quantize_rows(torch.randn(m, k, generator=generator, device=device))
    w, w_scale = _quantize_rows(torch.randn(n, k, generator=generator, device=device))
    return {"xq": xq, "x_scale": x_scale, "w": w, "w_scale": w_scale}


def preshuffle_weight(w, layout=(16, 16)):
    """AITER shuffle_weight(w, layout=(16, 16)) for one-byte elements: a permutation of w [n, k] in 16 x 32 tiles."""
    in_n, in_k = layout
    block_k = in_k * 2
    vec = 16 // w.element_size()
    n, k = w.shape
    if n % in_n or k % block_k:
        raise ValueError(f"w [{n}, {k}] must be divisible by ({in_n}, {block_k})")
    shuffled = w.view(n // in_n, in_n, k // block_k, block_k // vec, vec)
    return shuffled.permute(0, 2, 3, 1, 4).contiguous().view(n, k)


def to_kernel_layout(case, inputs):
    inputs = dict(inputs)
    inputs["w"] = preshuffle_weight(inputs["w"])
    return inputs
