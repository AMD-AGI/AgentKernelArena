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
# (``ld_i32`` from ATOM at 454d0e3023, by way of sgl-project/sglang#43984)
"""Device primitives of the mono kernels: packing, uniform values and buffer access.

AKA: only the primitives this launch uses are kept, unchanged (cache policies, ``rsrc``, ``sfresh64`` /
``uniform64`` / ``sload64``, ``uniform``, ``ld_i32``, ``mfma_bf16``, ``load_ptr64``, the bf16 packing and
``clock``); the wave reductions, hardware math, fp8 / mxfp4 helpers, ``KernArgs`` and the global
pointer / load helpers are not vendored.
"""

import flydsl.expr as fx
from . import buffer_ops as bo
from flydsl._mlir.dialects import llvm
from flydsl.expr import rocdl
from flydsl.expr.typing import T

# gfx95x cache policy bits: SC1 = device scope, SC0 | SC1 = system scope (peers over XGMI)
CM_DEV = 16


CM_SYS = 17


# NT = 2: nontemporal, for weights a layer reads once
CM_NT = 2


def rsrc(addr, nbytes=None):
    return bo.create_buffer_resource_from_addr(addr, num_records_bytes=nbytes)


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


def ld_i32(ptr, i, cm=0):
    """Word ``i`` of a buffer, an Int32 load (cache policy ``cm``)."""
    return fx.Int32(
        bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.i32, cache_modifier=cm)
    )


def mfma_bf16(a, b, c):
    return fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))


def load_ptr64(table, i):
    """Load entry ``i`` of an i64 table, wave-uniform."""
    pw = fx.Vector(bo.buffer_load(rsrc(table), i * 2, vec_width=2, dtype=T.i32))
    return (fx.Int64(uniform(pw[1])) << 32) | fx.Int64(fx.Uint32(uniform(pw[0])))


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


def clock():
    """Read the device-wide 64-bit realtime counter."""

    return fx.Int64(llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], []))
