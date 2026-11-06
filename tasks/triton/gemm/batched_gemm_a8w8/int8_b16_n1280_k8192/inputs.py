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
    batch, m, n, k = axes["batch"], axes["m"], axes["n"], axes["k"]
    x = torch.empty(batch, m, k, dtype=torch.int8, device=device)
    x_scale = torch.empty(batch, m, 1, dtype=torch.float32, device=device)
    w = torch.empty(batch, n, k, dtype=torch.int8, device=device)
    w_scale = torch.empty(batch, 1, n, dtype=torch.float32, device=device)
    for b in range(batch):
        x[b], x_scale[b] = _quantize_rows(torch.randn(m, k, generator=generator, device=device))
        codes, scale = _quantize_rows(torch.randn(n, k, generator=generator, device=device))
        w[b] = codes
        w_scale[b] = scale.T
    return {"x": x, "x_scale": x_scale, "w": w, "w_scale": w_scale}
