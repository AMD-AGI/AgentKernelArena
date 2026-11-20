"""MXFP4 activations and weights as the production producers emit them, and the preshuffled weight layout.

Both operands are quantized from normal data the way AITER's dynamic MXFP4 quantization does (`dynamic_mxfp4_quant`,
scaling mode "even"): the scale of a 1x32 group is the group maximum rounded to a power of two (mantissa >= 0.75 rounds
the exponent up) minus the exponent of the largest e2m1 value, and codes round to nearest with ties to the even code,
saturating at 6. The even K index is the low nibble of a byte. The quantizer is the one of the Triton and FlyDSL tasks
of this operator.

Activation scales are generated directly in the layout the asm kernel reads, which is what AITER's activation quantizer
writes with `shuffle=True` (aiter/utility/fp4_utils.py::dynamic_mxfp4_quant, the quantizer AITER's own a4w4 test and
callers of `gemm_a4w4` use): rows padded to a multiple of 256, the padding rows zero, and the [rows, k / 32] matrix
permuted as `e8m0_shuffle`. Generating that layout directly keeps the activation path free of any untimed transform.
The weight codes and weight scales are generated in their logical layout; `to_kernel_layout` applies the load-time
weight preparation (`shuffle_weight`, `e8m0_shuffle`).
"""
import torch

_GROUP = 32
# Decision boundaries between neighbouring e2m1 magnitudes (0, .5, 1, 1.5, 2, 3, 4, 6); a value exactly on a
# boundary goes to the even code.
_BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_TIE_UP = (False, True, False, True, False, True, False)
_ROW_CHUNK = 8192
_SCALE_ROW_PAD = 256


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


def e8m0_shuffle(scales):
    """AITER `e8m0_shuffle` (aiter/ops/shuffle.py::shuffle_scale) of scales [rows, k // 32], rows a multiple of 32.

    View as (rows / 32, 2, 16, k / 256, 2, 4) and permute to (rows / 32, k / 256, 4, 16, 2, 2).
    """
    rows, k_scales = scales.shape
    shuffled = scales.view(rows // 32, 2, 16, k_scales // 8, 2, 4).permute(0, 3, 5, 2, 4, 1)
    return shuffled.contiguous().view(rows, k_scales)


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    m, n, k = axes["m"], axes["n"], axes["k"]
    a, a_scales = _quantized_normal(m, k, 1.0, generator, device)
    w, w_scales = _quantized_normal(n, k, 0.05, generator, device)
    padded = torch.zeros((m + _SCALE_ROW_PAD - 1) // _SCALE_ROW_PAD * _SCALE_ROW_PAD, k // _GROUP,
                         dtype=torch.uint8, device=device)
    padded[:m] = a_scales
    return {"a": a, "a_scales": e8m0_shuffle(padded), "w": w, "w_scales": w_scales}


def to_kernel_layout(case, inputs):
    """Load-time weight preparation: AITER `shuffle_weight(w, layout=(16, 16))` and `e8m0_shuffle(w_scales)`.

    For 1-byte elements `shuffle_weight` views w [n, k / 2] as [n / 16, 16, k / 64, 2, 16] and permutes to
    [n / 16, k / 64, 2, 16, 16]: 16 (n) x 32 (bytes of k) tiles, each stored as [k half][n row][16 consecutive bytes].
    n = 7168 is a multiple of 256, so `e8m0_shuffle` adds no padding rows to w_scales.
    """
    inputs = dict(inputs)
    w = inputs["w"]
    n, k_bytes = w.shape
    inputs["w"] = w.view(n // 16, 16, k_bytes // 32, 2, 16).permute(0, 2, 3, 1, 4).contiguous().view(n, k_bytes)
    inputs["w_scales"] = e8m0_shuffle(inputs["w_scales"])
    return inputs
