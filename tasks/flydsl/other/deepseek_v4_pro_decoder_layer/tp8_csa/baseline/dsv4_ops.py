# SPDX-License-Identifier: Apache-2.0
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 4cd75a2
# (ROCm/ATOM#2559), MIT License, Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/models/deepseek_v4/mono/kernels/ops.py
# Vendored for AgentKernelArena from sgl-amd-mono 52486703: imports made relative;
# ``lds_atomic_add_i32`` added for FlyDSL 0.3.2.
"""Device primitives of the layer kernel beyond the shared ``common.ops``: traced
helpers, buffer / LDS element access, natural exp, wave-wide sums and unsigned max and
FP8 / MX conversions.
"""

from __future__ import annotations

import flydsl.expr as fx
from . import buffer_ops as bo
from flydsl._mlir.dialects import llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value

from .mono_ops import (
    butterfly,
    hw_exp2,
    hw_rcp,
    uniform,
    xred,
)
from .dsv4_config import (
    FP8_MAX,
)

NEG = -1.0e30
LOG2E = 1.4426950408889634


def traced(fn):
    """Apply FlyDSL's AST rewriting to a helper, so its ``while`` / ``if`` over traced
    values lower to scf ops. Kernels rewrite only their own body."""
    return ASTRewriter.transform(fn)


def wave_sum(v):
    return butterfly(v, (32, 16, 8, 4, 2, 1))


def uniform_f32(value):
    return uniform(fx.Float32(value).bitcast(fx.Int32)).bitcast(fx.Float32)


def write_lane_i32(value, lane, vector):
    """Write one i32 into ``lane`` of a wave-distributed value."""

    return fx.Int32(
        llvm.call_intrinsic(
            T.i32,
            "llvm.amdgcn.writelane.i32",
            [as_ir_value(fx.Int32(item)) for item in (value, lane, vector)],
            [],
            [],
        )
    )


def lds_atomic_add_i32(ptr, value):
    """Workgroup-scope monotonic atomic add of an i32 to the LDS word at ``ptr`` (an llvm
    ``atomicrmw``). AgentKernelArena addition: FlyDSL 0.3.2 has no ``fx.atomic_add``."""
    llvm.atomicrmw(
        llvm.AtomicBinOp.add,
        fx.to_llvm_ptr(ptr),
        as_ir_value(fx.Int32(value)),
        llvm.AtomicOrdering.monotonic,
        syncscope="workgroup",
        alignment=4,
    )


def exp(value):
    return hw_exp2(fx.Float32(value) * LOG2E)


def wave_umax(value):
    return fx.Int32(fx.coop.warp_reduce(fx.Uint32(value), fx.ReductionOp.MAX, width=64))


def wave_umin(value):
    return fx.Int32(fx.coop.warp_reduce(fx.Uint32(value), fx.ReductionOp.MIN, width=64))


def ld_f32(r, i):
    """One f32 from buffer resource ``r`` at element ``i``."""

    return fx.Float32(bo.buffer_load(r, i, vec_width=1, dtype=T.f32))


def ld_bf16(r, i):
    """One bf16 from buffer resource ``r`` at element ``i``, widened to f32."""

    return fx.Float32(fx.BFloat16(bo.buffer_load(r, i, vec_width=1, dtype=T.bf16)))


def lds_ld(ptr, i):
    return fx.ptr_load(ptr + i)


def lds_st(ptr, i, v):
    fx.ptr_store(v, ptr + i)


def fp8_roundtrip(lhs, rhs):
    """Round an f32 pair through E4M3FN and return the f32 pair."""

    word = rocdl.cvt_pk_fp8_f32(T.i32, lhs, rhs, fx.Int32(0), False)
    pair_type = fx.Vector.make_type(2, fx.Float32)
    pair = fx.Vector(rocdl.cvt_pk_f32_fp8(res=pair_type, src=word, word_sel=False))
    return pair[0], pair[1]


def fp8_to_bf16x8(word0, word1):
    """Convert two dwords of eight FP8 values to a BF16 vector."""

    one = as_ir_value(fx.Float32(1.0))
    parts = []
    for word in (word0, word1):
        for half in range_constexpr(2):
            pair = fx.Vector(
                rocdl.cvt_scalef32_pk_bf16_fp8(
                    T.vec(2, T.bf16), as_ir_value(word), one, bool(half)
                )
            )
            parts += [pair[0], pair[1]]
    return fx.Vector.from_elements(parts, fx.BFloat16)


def mxfp8_to_bf16x8(word0, word1, scale):
    """Convert two dwords of eight FP8 values with one E8M0 block scale to BF16."""

    scale = as_ir_value(scale)
    parts = []
    for word in (word0, word1):
        for half in range_constexpr(2):
            pair = fx.Vector(
                rocdl.cvt_scalef32_pk_bf16_fp8(
                    T.vec(2, T.bf16), as_ir_value(word), scale, bool(half)
                )
            )
            parts += [pair[0], pair[1]]
    return fx.Vector.from_elements(parts, fx.BFloat16)


def mxfp4_to_bf16x8(word, scale):
    """Convert one packed dword of eight scaled E2M1 values to BF16."""

    parts = []
    for select in range_constexpr(4):
        pair = fx.Vector(
            rocdl.cvt_scalef32_pk_bf16_fp4(
                T.vec(2, T.bf16),
                as_ir_value(word),
                as_ir_value(scale),
                select,
            )
        )
        parts += [pair[0], pair[1]]
    return fx.Vector.from_elements(parts, fx.BFloat16)


def pow2_ceil(x):
    """Smallest power of two >= x: an e8m0 block scale rounded up (V4's FP4 and FP8 blocks)."""
    bits = fx.Float32(x).bitcast(fx.Int32)
    man = bits & ((1 << 23) - 1)
    e = ((bits >> 23) & 0xFF) - 127 + (man != 0).select(fx.Int32(1), fx.Int32(0))
    return ((e + 127) << 23).bitcast(fx.Float32)


def fp8_block_dequant(vs, lanes):
    """``vs`` (an even count of this lane's values of one block, the block spread over
    ``lanes`` consecutive lanes) quantized as ATOM's activations to its FP8 GEMMs are:
    E4M3 under one e8m0 scale ceil_pow2(amax / 448), then dequantized -- exact in bf16,
    so a bf16 MFMA on the result multiplies what ATOM's FP8 MFMA does."""
    amax = fmath.absf(vs[0])
    for v in vs[1:]:
        amax = fx.max(amax, fmath.absf(v))
    off = lanes // 2
    while off:
        amax = xred(amax, off, fx.max)
        off //= 2
    nz = amax > 0.0
    scale = nz.select(pow2_ceil(amax * (1.0 / FP8_MAX)), fx.Float32(1.0))
    inv = nz.select(hw_rcp(scale), fx.Float32(1.0))  # exact: a power of two
    out = []
    for i in range_constexpr(len(vs) // 2):
        q0, q1 = (
            fx.min(fx.max(v * inv, -FP8_MAX), FP8_MAX) for v in vs[2 * i : 2 * i + 2]
        )
        d0, d1 = fp8_roundtrip(q0, q1)
        out += [d0 * scale, d1 * scale]
    return out
