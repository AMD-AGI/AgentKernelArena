# SPDX-License-Identifier: Apache-2.0
"""MXFP4 activation quantization on the device, as aiter's Triton ``_mxfp4_quant_op``
rounds it (the rule of ``mxfp4_quant_rows`` and of the compact-A4 experts' input and
SwiGLU re-quantization), plus the ballot lane count the expert plan needs.

A 32-value block's E8M0 scale comes from the block's max |x|: the max is rounded
at mantissa bit 21, the unbiased exponent is ``floor(log2) - 2`` clamped at -127,
and each value times ``2^-unbiased`` is rounded to E2M1 half to even, saturating at
6, with the 0 / 0.5 / 1 denormal step taken by the 2^22 addition. This is the
"even" scaling of aiter's ``_dynamic_mxfp4_quant_kernel``, not the ceil rule of
``common.mx.code_ceil`` (vLLM's MXFP8), so it lives here until ``common.mx``
takes it (M2 plan B.2).

``lanes_before`` counts by 32-bit halves: ``common.ops.lanes_below`` shifts a
64-bit one by the lane and miscounts past lane 31, which a 64-pick ballot round
reaches.
"""

import flydsl.expr as fx

from .ops import popcount

E2M1_MAX = 6.0


def e8m0_f32(byte):
    """An E8M0 byte (as loaded, any width) -> the f32 power of two it encodes."""
    return ((fx.Int32(byte) & 0xFF) << 23).bitcast(fx.Float32)


def mxfp4_scale(amax):
    """The MXFP4 block scale of a block whose max |x| is ``amax`` (fp32, >= 0):
    (the E8M0 code as an Int32, the quantizing factor ``2^-unbiased`` as fp32)."""
    bits = (amax.bitcast(fx.Int32) + fx.Int32(0x200000)) & fx.Int32(-0x800000)
    unb = fx.max(((bits >> 23) & 0xFF) - 129, fx.Int32(-127))
    return unb + 127, ((fx.Int32(127) - unb) << 23).bitcast(fx.Float32)


def e2m1_code(x, qscale):
    """The E2M1 code (an Int32 holding 4 bits, the sign in bit 3) of ``x * qscale``:
    half to even, saturating at 6, the denormal step through the 2^22 addition."""
    qb = (x * qscale).bitcast(fx.Int32)
    neg = qb < 0
    ab = qb & fx.Int32(0x7FFFFFFF)
    af = ab.bitcast(fx.Float32)
    den = ((af + fx.Float32(4194304.0)).bitcast(fx.Int32) - fx.Int32(149 << 23)) & 0xFF
    nrm = ((ab + fx.Int32((-126 << 23) + 0x1FFFFF) + ((ab >> 22) & 1)) >> 22) & 0xFF
    code = (af >= fx.Float32(E2M1_MAX)).select(
        fx.Int32(7), (af < fx.Float32(1.0)).select(den, nrm)
    )
    return code | neg.select(fx.Int32(8), fx.Int32(0))


def lanes_before(bal, lane):
    """How many lanes of ballot ``bal`` (Int64) lie below ``lane``, by 32-bit halves."""
    lo = fx.Int32(bal & fx.Int64(0xFFFFFFFF))
    hi = fx.Int32(bal >> fx.Int64(32))
    low_part = popcount(lo & ((fx.Int32(1) << fx.min(lane, 31)) - 1))
    high_part = popcount(lo) + popcount(
        hi & ((fx.Int32(1) << fx.max(lane - 32, 0)) - 1)
    )
    return (lane < 32).select(low_part, high_part)
