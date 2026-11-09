"""MXFP4 activations and weights: e2m1 codes packed two per byte with e8m0 scales per 32 values of K.

Both operands are quantized from normal data the way AITER's dynamic MXFP4 quantization does: the scale of a 1x32
group is the group maximum rounded to a power of two (mantissa >= 0.75 rounds the exponent up) minus the exponent
of the largest e2m1 value, and codes round to nearest with ties to the even code, saturating at 6. The even K index is
the low nibble of a byte.

`to_kernel_layout` produces what the FlyDSL preshuffle GEMM reads: the weight codes and the weight scales in the
layout of `shuffle_weight_w4(., NLane=16)` and `shuffle_scale_w4(., 1, False)` (FlyDSL tests/kernels/utils/
gemm_common_utils.py), and the activation scales padded to a multiple of 32 rows (with scale 1) and shuffled like the
weight scales. The activation codes keep their plain row-major layout.
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


def _quantized_normal(rows, k, std, generator, device):
    packed = torch.empty(rows, k // 2, dtype=torch.uint8, device=device)
    scales = torch.empty(rows, k // _GROUP, dtype=torch.uint8, device=device)
    for start in range(0, rows, _ROW_CHUNK):
        stop = min(start + _ROW_CHUNK, rows)
        values = std * torch.randn(stop - start, k, generator=generator, device=device, dtype=torch.float32)
        packed[start:stop], scales[start:stop] = _quantize_mxfp4(values)
    return packed, scales


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    m, n, k = axes["m"], axes["n"], axes["k"]
    a, a_scales = _quantized_normal(m, k, 1.0, generator, device)
    w, w_scales = _quantized_normal(n, k, 0.05, generator, device)
    return {"a": a, "a_scales": a_scales, "w": w, "w_scales": w_scales}


def _shuffle_weight(w):
    """shuffle_weight_w4(w, NLane=16): a permutation of 16 x 64 byte tiles, w [n, k // 2]."""
    n, k_packed = w.shape
    if n % 16 or k_packed % 64:
        raise ValueError(f"w [{n}, {k_packed}] must be divisible by (16, 64)")
    shuffled = w.view(n // 16, 16, k_packed // 64, 4, 16)  # (N0, NLane, K0, KLane, KPack)
    return shuffled.permute(0, 2, 3, 1, 4).contiguous().view(n, k_packed)


def _shuffle_scales(scales):
    """shuffle_scale_w4(scales, 1, False): scales [rows, k // 32], rows a multiple of 32."""
    rows, k_scales = scales.shape
    if rows % 32 or k_scales % 8:
        raise ValueError(f"scales [{rows}, {k_scales}] must be divisible by (32, 8)")
    # (N1, N_Pack, N_Lane, K1, K_Pack, K_Lane) -> (N1, K1, K_Lane, N_Lane, K_Pack, N_Pack)
    shuffled = scales.view(rows // 32, 2, 16, k_scales // 8, 2, 4)
    return shuffled.permute(0, 3, 5, 2, 4, 1).contiguous().view(rows, k_scales)


def to_kernel_layout(case, inputs):
    inputs = dict(inputs)
    a_scales = inputs["a_scales"]
    m, k_scales = a_scales.shape
    padded = torch.full(((m + 31) // 32 * 32, k_scales), 127, dtype=torch.uint8, device=a_scales.device)
    padded[:m] = a_scales
    inputs["a_scales"] = _shuffle_scales(padded)
    inputs["w"] = _shuffle_weight(inputs["w"])
    inputs["w_scales"] = _shuffle_scales(inputs["w_scales"])
    return inputs
