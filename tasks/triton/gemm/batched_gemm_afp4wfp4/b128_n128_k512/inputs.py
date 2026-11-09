"""MXFP4 activations and weights: e2m1 codes packed two per byte with e8m0 scales per 32 values of K.

Both operands are quantized from normal data the way AITER's dynamic MXFP4 quantization does: the scale of a 1x32
group is the group maximum rounded to a power of two (mantissa >= 0.75 rounds the exponent up) minus the exponent
of the largest e2m1 value, and codes round to nearest with ties to the even code, saturating at 6. The even K index is
the low nibble of a byte. Every batch element (attention head) draws its own activations and weights.
"""
import torch

_GROUP = 32
# Decision boundaries between neighbouring e2m1 magnitudes (0, .5, 1, 1.5, 2, 3, 4, 6); a value exactly on a
# boundary goes to the even code.
_BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_TIE_UP = (False, True, False, True, False, True, False)
_ROW_CHUNK = 8192


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


def _quantized_normal(batch, rows, k, std, generator, device):
    packed = torch.empty(batch, rows, k // 2, dtype=torch.uint8, device=device)
    scales = torch.empty(batch, rows, k // _GROUP, dtype=torch.uint8, device=device)
    for b in range(batch):
        for start in range(0, rows, _ROW_CHUNK):
            stop = min(start + _ROW_CHUNK, rows)
            values = std * torch.randn(stop - start, k, generator=generator, device=device, dtype=torch.float32)
            packed[b, start:stop], scales[b, start:stop] = _quantize_mxfp4(values)
    return packed, scales


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    batch, m, n, k = axes["batch"], axes["m"], axes["n"], axes["k"]
    x, x_scales = _quantized_normal(batch, m, k, 1.0, generator, device)
    w, w_scales = _quantized_normal(batch, n, k, 0.05, generator, device)
    return {"x": x, "x_scales": x_scales, "w": w, "w_scales": w_scales}
