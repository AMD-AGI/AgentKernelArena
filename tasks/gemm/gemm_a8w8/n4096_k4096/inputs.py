"""Row-wise float8 (e4m3fn) quantization of normal activations and weights, as a quantizing producer would emit."""
import torch

_FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)


def _quantize_rows(values):
    """float8_e4m3fn codes and float32 row scales (amax / 448) of a [rows, k] float32 matrix."""
    scale = (values.abs().amax(dim=-1, keepdim=True) / _FP8_MAX).clamp_min(1e-12)
    codes = (values / scale).clamp_(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return codes, scale


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    m, n, k = axes["m"], axes["n"], axes["k"]
    x, x_scale = _quantize_rows(torch.randn(m, k, generator=generator, device=device))
    w, w_scale = _quantize_rows(torch.randn(n, k, generator=generator, device=device))
    return {"x": x, "w": w, "x_scale": x_scale, "w_scale": w_scale.T.contiguous()}
