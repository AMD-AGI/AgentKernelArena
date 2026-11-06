"""Row-wise int8 quantization of normal activations and weights, as a quantizing producer would emit."""
import torch

_INT8_MAX = 127.0


def _quantize_rows(values):
    """int8 codes and float32 row scales (amax / 127) of a [rows, k] float32 matrix."""
    scale = (values.abs().amax(dim=-1, keepdim=True) / _INT8_MAX).clamp_min(1e-12)
    codes = (values / scale).round().clamp_(-_INT8_MAX, _INT8_MAX).to(torch.int8)
    return codes, scale


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    m, n, k = axes["m"], axes["n"], axes["k"]
    a, scale_a = _quantize_rows(torch.randn(m, k, generator=generator, device=device))
    w, scale_b = _quantize_rows(torch.randn(n, k, generator=generator, device=device))
    return {"a": a, "w": w, "scale_a": scale_a, "scale_b": scale_b}
