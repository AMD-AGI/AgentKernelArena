"""MXFP4 (OCP E2M1 codes, one E8M0 scale per 32 values) numerics shared by the inputs and the reference.

The block scale follows AITER's dynamic MXFP4 quantizer: the block's absolute maximum is rounded to a
power of two (``(bits(amax) + 0x200000) & 0xFF800000``, i.e. up when its mantissa is at least 1.75) and
the scale is that power of two divided by 4, clamped to [2^-127, 2^127]. Each value divided by the scale
is rounded to the nearest E2M1 value {0, 0.5, 1, 1.5, 2, 3, 4, 6}, ties to the even code, saturating at
6. Codes are packed two per byte along the last axis, the even element in the low nibble; the scale byte
is the biased exponent (bias 127).
"""
import torch

BLOCK = 32
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _scale_exponent(blocks: torch.Tensor) -> torch.Tensor:
    """Unbiased E8M0 exponent of each float32 block [..., n, 32] -> [..., n, 1] (int64)."""
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    bits = amax.view(torch.int32).to(torch.int64)
    field = (((bits + 0x200000) & 0xFF800000) >> 23) & 0xFF
    # A zero (or tiny subnormal) maximum gives log2(0) = -inf upstream, clamped to -127.
    exponent = torch.where(field == 0, torch.full_like(field, -127), field - 127 - 2)
    return exponent.clamp(-127, 127)


def _grid_round(magnitude: torch.Tensor) -> torch.Tensor:
    """Nearest E2M1 magnitude, ties to the even code (float64 in, float64 out)."""
    rounded = torch.where(magnitude < 2.0, torch.round(magnitude * 2.0) / 2.0,
                          torch.where(magnitude < 4.0, torch.round(magnitude),
                                      torch.round(magnitude / 2.0) * 2.0))
    return rounded.clamp(max=6.0)


def mxfp4_round(x: torch.Tensor) -> torch.Tensor:
    """Quantize x along its last axis to MXFP4 and return the represented values in float64."""
    blocks = x.to(torch.float32).unflatten(-1, (-1, BLOCK))
    scale = torch.exp2(_scale_exponent(blocks).to(torch.float64))
    scaled = blocks.to(torch.float64) / scale
    return (torch.sign(scaled) * _grid_round(scaled.abs()) * scale).flatten(-2)


def quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP4 codes packed two per byte (uint8 [..., K // 2]) and E8M0 scales (uint8 [..., K // 32])."""
    blocks = x.to(torch.float32).unflatten(-1, (-1, BLOCK))
    exponent = _scale_exponent(blocks)
    scaled = blocks.to(torch.float64) / torch.exp2(exponent.to(torch.float64))
    magnitude = _grid_round(scaled.abs())
    code = torch.where(magnitude < 2.0, magnitude * 2.0,
                       torch.where(magnitude < 4.0, magnitude + 2.0, magnitude / 2.0 + 4.0)).to(torch.int64)
    code = code | torch.where(torch.signbit(blocks), 8, 0)
    code = code.flatten(-2)
    packed = (code[..., 0::2] | (code[..., 1::2] << 4)).to(torch.uint8)
    return packed, (exponent + 127).squeeze(-1).to(torch.uint8)


def dequantize(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Float64 values of packed MXFP4 codes [..., K // 2] with E8M0 scales [..., K // 32]."""
    table = torch.tensor(_E2M1, dtype=torch.float64, device=packed.device)
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(-2).long()
    scale = torch.exp2(scales.to(torch.float64) - 127.0).repeat_interleave(BLOCK, dim=-1)
    return table[codes] * scale
