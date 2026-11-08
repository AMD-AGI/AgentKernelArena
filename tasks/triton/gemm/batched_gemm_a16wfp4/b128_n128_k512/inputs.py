"""MXFP4 weights: e2m1 codes packed two per byte (low nibble first) with one e8m0 scale per 32 values of K."""
import torch

_GROUP = 32
_FP4_MAX = 6.0
# Ascending magnitudes of the eight e2m1 codes.
_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# Decision boundaries between neighbouring grid points; a value exactly on a boundary goes to the even code.
_BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_TIE_UP = (False, True, False, True, False, True, False)


def _encode_e2m1(values):
    """Round float32 values to e2m1 codes (round to nearest, ties to even code, saturating at 6)."""
    magnitude = values.abs()
    code = torch.zeros(values.shape, dtype=torch.uint8, device=values.device)
    for bound, up in zip(_BOUNDS, _TIE_UP):
        code += ((magnitude >= bound) if up else (magnitude > bound)).to(torch.uint8)
    return code | ((values < 0).to(torch.uint8) << 3)


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    batch, n, k = axes["batch"], axes["n"], axes["k"]
    w_float = 0.05 * torch.randn(batch, n, k, generator=generator, device=device, dtype=torch.float32)
    groups = w_float.view(batch, n, k // _GROUP, _GROUP)
    amax = groups.abs().amax(dim=-1)
    exponent = (torch.log2((amax / _FP4_MAX).clamp_min(1e-30)).round() + 127).clamp(1, 254)
    scale = torch.exp2(exponent - 127).unsqueeze(-1)
    codes = _encode_e2m1(groups / scale).view(batch, n, k)
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    return {"w": packed.contiguous(), "w_scales": exponent.to(torch.uint8)}
