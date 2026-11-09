"""fp8 (e4m3fn) weights with one per-tensor scale, as an MLA absorbed-projection weight is stored."""
import torch

_FP8_MAX = 448.0  # e4m3fn saturation magnitude


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    batch, n, k = axes["batch"], axes["n"], axes["k"]
    w = 0.05 * torch.randn(batch, n, k, generator=generator, device=device, dtype=torch.float32)
    # One scale for the whole batched tensor: amax over every head, clamped like the dynamic per-batched-tensor
    # quantization of the serving frameworks; the operator multiplies the accumulator by this scale (amax / 448).
    amax = w.abs().amax().clamp_min(1e-10)
    wq = (w * (_FP8_MAX / amax)).clamp_(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return {"wq": wq, "w_scale": (amax / _FP8_MAX).reshape(1)}
