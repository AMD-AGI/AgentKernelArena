# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 97359d6, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/device/ops.py
# atom/mono/device/ranks.py
# atom/mono/device/stamps.py
# Shared GLM helpers adapted from ROCm/aiter#6173 at 56d6390def and
# vllm-project/vllm at ca328e297b (Apache-2.0), originally
# aiter/ops/flydsl/kernels/glm5_mono/ops.py:
# Copyright (c) 2025 FlyDSL Project Contributors
# (``global_load`` and ``ld_i32`` from ATOM at 454d0e3023, by way of
# sgl-project/sglang#43984)
# Vendored from sgl-project/sgl-amd-mono at 52486703fbf27e77f05b573e4d931d4eec87118d
# (sgl_amd_mono/common/ops.py). AKA: package-relative imports of the vendored buffer_ops,
# dpp_utils and plan; KernArgs, global_ptr and global_load (unused by this kernel; they need
# FlyDSL >= 0.3.4 or a standard-library import) are left out.
"""Device primitives of the mono kernels: wave ops, hardware math, packing and buffer access."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value

from . import buffer_ops as bo
from .dpp_utils import update_dpp_i32
from .plan import THREADS, WAVES

# gfx95x cache policy bits: SC1 = device scope, SC0 | SC1 = system scope (peers over XGMI)
CM_DEV = 16


CM_SYS = 17


# NT = 2: nontemporal, for weights a layer reads once
CM_NT = 2


def rsrc(addr, nbytes=None):
    return bo.create_buffer_resource_from_addr(addr, num_records_bytes=nbytes)


def permlane_swap(off, x, y):
    """v_permlane{32,16}_swap on registers (x, y); returns (x', y')."""
    swap = rocdl.permlane32_swap if off == 32 else rocdl.permlane16_swap
    pr = swap(
        llvm.StructType.get_literal([T.i32, T.i32]),
        as_ir_value(fx.Int32(x)),
        as_ir_value(fx.Int32(y)),
        False,
        False,
    )
    return tuple(fx.Int32(llvm.extractvalue(T.i32, pr, [j])) for j in range(2))


def lane_gather(v, src_lane):
    """Lane ``src_lane``'s value of v, a source lane a lane (ds_bpermute)."""
    return fx.Int32(rocdl.ds_bpermute(T.i32, fx.Int32(src_lane) * 4, fx.Int32(v)))


def fresh(v):
    """Pass ``v`` through an opaque move, so LLVM does not hoist lane math into long-lived registers."""
    return fx.Int32(
        llvm.inline_asm(
            T.i32,
            [fx.Int32(v).ir_value()],
            "v_mov_b32 $0, $1",
            "=v,v",
            has_side_effects=True,
        )
    )


def sfresh64(v):
    """Pass a wave-uniform i64 through an opaque scalar move: LLVM cannot merge it with another use or hoist it."""
    return fx.Int64(
        llvm.inline_asm(
            T.i64,
            [fx.Int64(v).ir_value()],
            "s_mov_b64 $0, $1",
            "=s,s",
            has_side_effects=True,
        )
    )


def uniform64(v):
    v = fx.Int64(v)
    return (fx.Int64(uniform(fx.Int32(v >> 32))) << 32) | fx.Int64(
        fx.Uint32(uniform(fx.Int32(v)))
    )


def sload64(addr):
    """The wave-uniform i64 at device address ``addr``, which no thread writes during the launch (scalar load).

    ``addr`` goes through readfirstlane: inside a divergent branch FlyDSL's bundled LLVM otherwise takes it
    as divergent and emits a vector load and a waterfall loop around the use.
    """
    from flydsl._mlir import ir

    p = llvm.inttoptr(ir.Type.parse("!llvm.ptr<4>"), uniform64(addr).ir_value())
    return fx.Int64(llvm.load(T.i64, p, alignment=8, invariant=True))


def uniform(v):
    return fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(v)))


def hw_rsq(x):
    """v_rsq_f32 (~1 ulp, no libm range fixups)."""
    return fx.Float32(rocdl.rsq(T.f32, fx.Float32(x).ir_value()))


def hw_rcp(x):
    return fx.Float32(rocdl.rcp(T.f32, fx.Float32(x)))


def hw_exp2(x):
    return fx.Float32(rocdl.exp2(T.f32, fx.Float32(x).ir_value()))


def div_rn(x, d, r):
    """Return ``x / d`` rounded to nearest, bit-identical to IEEE division, given ``r = 1.0 / d``."""
    q = x * r
    return fmath.fma(fmath.fma(-q, d, x), r, q)


def xshfl(v, off):
    """Return the value of lane ``lane ^ off`` (power-of-two offsets only)."""
    assert off in (1, 2, 4, 8, 16, 32), off
    if off >= 16:
        return v.shuffle_xor(off, 64)
    is_f = isinstance(v, fx.Float32)
    x = v.bitcast(fx.Int32) if is_f else fx.Int32(v)
    if off == 8:
        y = fx.Int32(update_dpp_i32(x, x, 0x118, 0xF, 0xC, False))
        y = fx.Int32(update_dpp_i32(y, x, 0x108, 0xF, 0x3, False))
    elif off == 4:
        y = fx.Int32(update_dpp_i32(x, x, 0x114, 0xF, 0xA, False))
        y = fx.Int32(update_dpp_i32(y, x, 0x104, 0xF, 0x5, False))
    elif off == 2:
        y = fx.Int32(update_dpp_i32(x, x, 0x4E, 0xF, 0xF, False))
    else:
        y = fx.Int32(update_dpp_i32(x, x, 0xB1, 0xF, 0xF, False))
    return y.bitcast(fx.Float32) if is_f else y


def xred(v, off, op):
    """op(v, lane ^ off) for a symmetric op; 32 / 16 take both halves of one swap."""
    if off < 16:
        return op(v, xshfl(v, off))
    is_f = isinstance(v, fx.Float32)
    x = v.bitcast(fx.Int32) if is_f else fx.Int32(v)
    a, b = permlane_swap(off, x, x)
    if is_f:
        return op(a.bitcast(fx.Float32), b.bitcast(fx.Float32))
    return op(type(v)(a), type(v)(b))


def _add(a, b):
    return a + b


def butterfly(v, offsets, op=_add):
    """Butterfly reduction over the xor offsets in the given order (the order sets the result bits)."""
    for off in offsets:
        v = xred(v, off, op)
    return v


def wave_max(v):
    return fx.coop.warp_reduce(v, fx.ReductionOp.MAX, width=64)


def wave_umax(v):
    """Unsigned wave maximum, returned as i32 bits for packed routing keys."""
    return fx.Int32(fx.coop.warp_reduce(fx.Uint32(v), fx.ReductionOp.MAX, width=64))


def wave_sum(v):
    return butterfly(v, (32, 16, 8, 4, 2, 1))


def _xpartner(v, off, lane):
    """Lane ``lane ^ off``'s v: DPP within a row, v_permlane*_swap across rows
    (``xshfl`` takes ``shuffle_xor`` there)."""
    if off < 16:
        return xshfl(v, off)
    a, b = permlane_swap(off, v, v)
    return ((lane & off) != 0).select(a, b)


def wave_incl_scan(v, lane):
    """Inclusive prefix sum of an Int32 over the wave, in lane order."""
    blk = v
    for off in (1, 2, 4, 8, 16, 32):
        o = _xpartner(blk, off, lane)
        v = v + ((lane & off) != 0).select(o, 0)
        blk = blk + o
    return v


@flyc.jit
def block_excl_scan(v, lane, wave, buf):
    """(exclusive prefix of v over the CTA in thread order, CTA total); ``buf``
    needs WAVES words."""
    incl = wave_incl_scan(v, lane)
    gpu.barrier()
    if lane == 63:
        fx.ptr_store(incl, buf + wave)
    gpu.barrier()
    off = fx.Int32(0)
    tot = fx.Int32(0)
    for w in range_constexpr(WAVES):
        cnt = fx.ptr_load(buf + w)
        off = off + (w < wave).select(cnt, 0)
        tot = tot + cnt
    return incl - v + off, tot


def ld_i32(ptr, i, cm=0):
    """Word ``i`` of a buffer, an Int32 load (cache policy ``cm``)."""
    return fx.Int32(
        bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.i32, cache_modifier=cm)
    )


LOAD_BATCH = 4  # rounds of a strided loop whose loads go out together


def first_rounds(tid, n, load, batch=LOAD_BATCH):
    """``batched_rounds``' first batch of loads, issued ahead of a wait they do not
    depend on (pass them as its ``first``)."""
    rounds = -(-n // THREADS)
    return [load(fx.min(tid + THREADS * i, n - 1)) for i in range(min(batch, rounds))]


@flyc.jit
def batched_rounds(tid, n, load, use, batch=LOAD_BATCH, first=None):
    """Strided CTA loop over v < n: ``batch`` rounds of ``load(v)`` issued before each
    ``use``. ``first``: the first batch's loads, already issued (``first_rounds``)."""
    rounds = -(-n // THREADS)
    assert first is None or len(first) == min(batch, rounds), (len(first), n)
    for b0 in range_constexpr(0, rounds, batch):
        idx = range(b0, min(b0 + batch, rounds))
        if const_expr(b0 == 0 and first is not None):
            got = first
        else:
            got = [load(fx.min(tid + THREADS * i, n - 1)) for i in idx]
        for i, g in zip(idx, got):
            v = tid + THREADS * i
            if v < n:
                use(v, g)


def mfma_bf16(a, b, c):
    return fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))


def row_sum(red, r, t, waves=None):
    """Return row r, token t of the waves' MFMA accumulators, summed in wave order."""
    tot = fx.Float32(0.0)
    for w in waves if waves is not None else range(WAVES):
        tot = tot + fx.ptr_load(red + ((w * 64 + 16 * (r // 4) + t) * 4 + r % 4))
    return tot


def ballot(pred):
    """The wave's lanes where ``pred`` holds, a bit a lane."""
    return fx.Int64(rocdl.ballot(T.i64, pred))


def popcount(x):
    return fx.Int32(fx.ctpop(x))


def lanes_below(bal, lane):
    """How many of the lanes in ballot ``bal`` are below ``lane``."""
    return popcount(bal & ((fx.Int64(1) << fx.Int64(lane)) - 1))


def load_ptr64(table, i):
    """Load entry ``i`` of an i64 table, wave-uniform."""
    pw = fx.Vector(bo.buffer_load(rsrc(table), i * 2, vec_width=2, dtype=T.i32))
    return (fx.Int64(uniform(pw[1])) << 32) | fx.Int64(fx.Uint32(uniform(pw[0])))


def bf16_round(a):
    return fx.Float32(fx.Float32(a).to(fx.BFloat16))


def bf16_pair(a, b):
    """Two f32 -> one f32-typed word holding (bf16(a), bf16(b))."""
    return (
        fx.Vector.from_elements([a, b], fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Float32)[0]
    )


def bf_lo(w):
    """The f32 of a packed bf16 pair word's low half."""
    return (w << 16).bitcast(fx.Float32)


def bf_hi(w):
    """The f32 of a packed bf16 pair word's high half."""
    return (w & fx.Int32(-65536)).bitcast(fx.Float32)


def bf2_f32(w):
    """Packed bf16 pair word -> (f32 low, f32 high)."""
    return bf_lo(w), bf_hi(w)


def fp8_pack4(a, b, c, d):
    """Four f32 (already in range) -> one dword of four E4M3 bytes, RNE."""
    lo = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, a, b, fx.Int32(0), False))
    return fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, c, d, lo, True))


# ---------------------------------------------------------------- TP ranks
def peer_bases(peers, tp):
    """Load every rank's peer buffer base from the ``peers`` table once."""
    return [load_ptr64(peers, p) for p in range(tp)]


def sum_partials(poll, own, pair_of, tp):
    """Poll the ``tp`` ranks' bf16 pair partials and sum them in fp32 in rank order."""
    ws = poll([(own, pair_of(src), 1) for src in range(tp)])
    acc0, acc1 = bf2_f32(ws[0][0])
    for src in range_constexpr(1, tp):
        lo, hi = bf2_f32(ws[src][0])
        acc0 = acc0 + lo
        acc1 = acc1 + hi
    return acc0, acc1


def uniform_f32(value):
    return uniform(fx.Float32(value).bitcast(fx.Int32)).bitcast(fx.Float32)


def read_lane_i32(value, lane):
    """Lane ``lane``'s i32 ``value`` (v_readlane)."""

    return fx.Int32(rocdl.readlane(T.i32, fx.Int32(value), fx.Int32(lane)))


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


def bpermute_i32(byte_offset, value):
    """Read an i32 VGPR value from the lane selected by a byte offset."""

    return fx.Int32(rocdl.ds_bpermute(T.i32, fx.Int32(byte_offset), fx.Int32(value)))


def clock():
    """Read the device-wide 64-bit realtime counter."""

    return fx.Int64(llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], []))


def fp8_roundtrip(lhs, rhs):
    """Round an f32 pair through E4M3FN and return the f32 pair."""

    word = rocdl.cvt_pk_fp8_f32(T.i32, lhs, rhs, fx.Int32(0), False)
    pair_type = fx.Vector.make_type(2, fx.Float32)
    pair = fx.Vector(rocdl.cvt_pk_f32_fp8(res=pair_type, src=word, word_sel=False))
    return pair[0], pair[1]


def f8_word(k):
    """Map an FP8 activation index to the packed LDS word order."""

    return (k // 64) * 16 + ((k % 32) // 8) * 4 + ((k % 64) // 32) * 2 + (k % 8) // 4


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


def mxfp4_to_bf16x8(word, scale):
    """Convert one packed dword of eight scaled E2M1 values to BF16."""

    parts = []
    for select in range_constexpr(4):
        pair = fx.Vector(
            rocdl.cvt_scalef32_pk_bf16_fp4(
                T.vec(2, T.bf16),
                word,
                scale,
                select,
            )
        )
        parts += [pair[0], pair[1]]
    return fx.Vector.from_elements(parts, fx.BFloat16)


def hw_exp(value):
    """e^x as the hardware exp2 of x * log2(e)."""
    return hw_exp2(fx.Float32(value) * 1.4426950408889634)
