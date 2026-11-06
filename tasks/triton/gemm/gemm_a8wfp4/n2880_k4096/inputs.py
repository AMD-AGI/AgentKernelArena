"""fp8 activations with a per-token scale and MXFP4 weights: consistent codes and scales.

Activations are quantized per row (scale = amax / 448, e4m3fn codes). Weights are quantized to MXFP4: one e8m0 scale
per 32 values of K (the group maximum rounded to a power of two, minus the exponent of the largest e2m1 value),
e2m1 codes rounded to nearest with ties to the even code, packed two per byte with the even K index in the low nibble.
"""
import torch

_FP8_MAX = 448.0  # e4m3fn saturation magnitude
_GROUP = 32
# Decision boundaries between neighbouring e2m1 magnitudes (0, .5, 1, 1.5, 2, 3, 4, 6); a value exactly on a
# boundary goes to the even code.
_BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_TIE_UP = (False, True, False, True, False, True, False)


def _encode_e2m1(values):
    magnitude = values.abs()
    code = torch.zeros(values.shape, dtype=torch.uint8, device=values.device)
    for bound, up in zip(_BOUNDS, _TIE_UP):
        code += ((magnitude >= bound) if up else (magnitude > bound)).to(torch.uint8)
    return code | ((values < 0).to(torch.uint8) << 3)


def _quantize_mxfp4(values):
    """values [rows, k] float32 -> (packed codes [rows, k // 2] uint8, e8m0 scales [rows, k // 32] uint8)."""
    rows, k = values.shape
    groups = values.view(rows, k // _GROUP, _GROUP)
    amax = groups.abs().amax(dim=-1, keepdim=True)
    rounded = (amax.contiguous().view(torch.int32) + 0x200000) & -0x800000
    exponent = (((rounded >> 23) & 0xFF) - 127 - 2).clamp(-126, 127)
    scaled = groups * torch.exp2(-exponent.float())
    codes = _encode_e2m1(scaled).view(rows, k)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed.contiguous(), (exponent + 127).squeeze(-1).to(torch.uint8)


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    m, n, k = axes["m"], axes["n"], axes["k"]
    a_float = torch.randn(m, k, generator=generator, device=device, dtype=torch.float32)
    a_scales = (a_float.abs().amax(dim=1, keepdim=True) / _FP8_MAX).clamp_min(1e-12)
    a = (a_float / a_scales).to(torch.float8_e4m3fn)
    del a_float
    w_float = 0.05 * torch.randn(n, k, generator=generator, device=device, dtype=torch.float32)
    w, w_scales = _quantize_mxfp4(w_float)
    return {"a": a, "a_scales": a_scales, "w": w, "w_scales": w_scales}
