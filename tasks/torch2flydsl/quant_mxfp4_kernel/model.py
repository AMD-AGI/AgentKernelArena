# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Pure-PyTorch reference for MXFP4 (E2M1) per-1x32 dynamic quantization.

The op quantizes a ``[m, n]`` tensor to MXFP4: each contiguous block of 32
elements along the last dim shares one E8M0 (power-of-two) block scale, and the
32 values are encoded as 4-bit E2M1 codes packed two-per-byte. The op returns
``(packed, scale)`` where ``packed`` is ``[m, n // 2]`` (``float4_e2m1fn_x2``
bytes) and ``scale`` is ``[m, n // 32]`` E8M0 bytes (``float8_e8m0fnu``).

AMD-runtime semantics (option-b): this mirrors the AMD runtime
``quant_mxfp4_hip`` / ``per_1x32_f4_quant`` op at the project default round
mode ``RoundUp`` (NV
ROUND_UP / torchao RCEIL): the block scale is ``ceil_pow2(amax / 6)`` and the
E2M1 codes use the gfx950 hardware round-to-nearest-even conversion
(``v_cvt_pk_f4_*``). The arithmetic is bit-exact against the device kernel on
gfx950 (validated byte-for-byte by ``op_tests/test_quant_mxfp4.py``), so the
packed codes and the E8M0 scale match the hardware output exactly.
"""
import torch
import torch.nn as nn

_BLOCK = 32
_F32_MIN_NORMAL = 2.0 ** (-126)
# 0xFF800000 as a signed int32: keeps sign + 8-bit exponent (strips mantissa).
_E8M0_STRIP_MANT = -8388608

# Typed views matching the AMD op's return dtypes; fall back to uint8 if absent.
_FP4X2 = getattr(torch, "float4_e2m1fn_x2", torch.uint8)
_FP8_E8M0 = getattr(torch, "float8_e8m0fnu", torch.uint8)


def _rceil_pow2_div6(max_abs):
    """RoundUp / RCEIL block scale: ``scale = ceil_pow2(amax / 6)``.

    NaN/Inf pass through (their exponent is 0xFF and is never bumped).
    """
    m = max_abs.to(torch.float32)
    zero_mask = m == 0
    scaled = m / 6.0
    as_int = scaled.view(torch.int32)
    mant_nonzero = (as_int & 0x7FFFFF) != 0
    exp_bits = (as_int >> 23) & 0xFF
    bump = mant_nonzero & (exp_bits < 0xFF)
    bumped = torch.where(bump, as_int + 0x800000, as_int)  # exp += 1
    rounded = bumped & _E8M0_STRIP_MANT  # strip mantissa
    out = rounded.view(torch.float32)
    out = torch.where(zero_mask, torch.full_like(out, _F32_MIN_NORMAL), out)
    out = out.log2().floor().clamp(min=-127, max=127).exp2()
    return out


def _fp32_to_e2m1_rne(val):
    """E2M1 (4-bit) quantization with round-to-nearest-even (gfx950 HW)."""
    qx = val.float().contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    s = qx & 0x80000000
    qx = qx ^ s

    abs_f = qx.to(torch.int32).view(torch.float32)
    sat = abs_f >= 6.0
    denorm = (~sat) & (abs_f < 1.0)
    normal = ~(sat | denorm)

    denorm_const = 149 << 23
    # Bit pattern ``149 << 23`` is the exactly representable float 2**22.
    # Using the Python scalar lets PyTorch scalar-lift it into the kernel and
    # avoids allocating/copying a device tensor inside a captured callback.
    d = abs_f + 4194304.0
    d = (d.view(torch.int32).to(torch.int64) & 0xFFFFFFFF) - denorm_const

    mant_odd = (qx >> 22) & 1
    val_to_add = ((1 - 127) << 23) + (1 << 21) - 1
    n = (qx + (val_to_add & 0xFFFFFFFF) + mant_odd) >> 22

    e2m1 = torch.full_like(qx, 7)
    e2m1 = torch.where(normal, n, e2m1)
    e2m1 = torch.where(denorm, d, e2m1)
    e2m1 = e2m1 | (s >> 28)
    return e2m1.to(torch.uint8)


class Model(nn.Module):
    """MXFP4 per-1x32 dynamic quantizer. ``Model(group_size=32)``."""

    def __init__(self, group_size=32):
        super().__init__()
        self.group_size = int(group_size)

    def forward(self, input):
        g = self.group_size
        x = input.float()
        rows, cols = x.shape
        n_groups = cols // g

        grouped = x.reshape(rows, n_groups, g)
        group_max = grouped.abs().amax(dim=-1)
        dq_scale = _rceil_pow2_div6(group_max)

        q_scale = torch.where(
            dq_scale == 0, torch.zeros_like(dq_scale), 1.0 / dq_scale
        )
        scaled = grouped * q_scale.unsqueeze(-1)

        nibbles = _fp32_to_e2m1_rne(scaled).reshape(rows, cols)
        packed = (nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)).contiguous()

        scale_e8m0 = ((dq_scale.view(torch.int32) >> 23) & 0xFF).to(torch.uint8)

        return packed.view(_FP4X2), scale_e8m0.view(_FP8_E8M0)


def get_inputs():
    return [torch.randn(4096, 256, dtype=torch.bfloat16)]


def get_init_inputs():
    return [32]
