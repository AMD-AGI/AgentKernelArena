# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3's device helpers that ``sgl_amd_mono.common`` does not hold: the
step's mailbox tag, wave and CTA reductions, and a mailbox whose plain addresses
are device-scope scratch.

The wave reductions, ``readlane`` and ``unlikely`` are adapted
from ROCm/ATOM (https://github.com/ROCm/ATOM) at 454d0e3023 (MIT License,
Copyright (C) 2026, Advanced Micro Devices, Inc.)."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from . import buffer_ops as bo
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value

from .compat import AtomicOrdering, memory_fence

from .ops import butterfly, rsrc, uniform, wave_max
from .plan import WAVES
from .sync import POLL_MAX, Addr, Mailbox, Space
from .config import LAYER_SLOTS


def step_tag(epoch_ptr, layer):
    """The layer's mailbox tag this step, ``step * LAYER_SLOTS + layer + 1``.

    The step is word 0 of the epoch buffer, which the host moves on after a step's last
    launch, so the tag is graph-safe and no mailbox is cleared between steps."""
    step = uniform(bo.buffer_load(rsrc(epoch_ptr), 0, vec_width=1, dtype=T.i32))
    return step * LAYER_SLOTS + layer + 1


def _scratch(a):
    return a if isinstance(a, Addr) else Addr("scratch", Space.SCRATCH, a)


class StepMailbox(Mailbox):
    """The common mailbox, taking a plain address as device-scope scratch (the
    kernels address their scratch by offset)."""

    def put(self, a, i, v):
        super().put(_scratch(a), i, v)

    def put_bf(self, a, i, vs):
        super().put_bf(_scratch(a), i, vs)

    def put_words(self, a, i, words):
        super().put_words(_scratch(a), i, words)

    def poll(self, specs, batch=POLL_MAX):
        return super().poll([(_scratch(a), i, n) for a, i, n in specs], batch)


def release_stores():
    """Order this wave's earlier device-scope (SC1) stores before a flag other CTAs poll.

    A workgroup-scope release fence does not wait for global stores, so the flag could
    land first and a consumer read stale data; the stores write through to L2, so their
    completion is all it must wait for."""
    rocdl.s_waitcnt(vmcnt=0)
    memory_fence(
        syncscope=rocdl.SyncScope.Workgroup, ordering=AtomicOrdering.Release
    )


def ar_pack_sumsq(vals):
    """One logical thread of the custom all-reduce epilogue: its 8 elements'
    squares accumulated in order (hipcc contracts ``acc += v * v`` into fma)."""
    acc = fx.Float32(0.0)
    for v in vals:
        acc = fmath.fma(v, v, acc)
    return acc


def readlane(v, src_lane):
    """Lane ``src_lane``'s value of v (wave-uniform)."""
    bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
    r = fx.Int32(rocdl.readlane(T.i32, bits, fx.Int32(src_lane)))
    return r.bitcast(fx.Float32) if isinstance(v, fx.Float32) else r


def unlikely(cond):
    """``cond`` marked cold (llvm.expect), so LLVM lays the branch out of the hot path:
    a rare path's code inline among the hot stages costs them even when it never runs."""
    return fx.Boolean(
        llvm.call_intrinsic(
            fx.Boolean.ir_type,
            "llvm.expect.i1",
            [as_ir_value(cond), as_ir_value(fx.Boolean(False))],
            [],
            [],
        )
    )


def wave_sum(v):
    return butterfly(v, (32, 16, 8, 4, 2, 1))


@flyc.jit
def block_max(v, lane, wave, red):
    """Max of v over the CTA's WAVES waves; ``red`` needs WAVES words."""
    wv = wave_max(v)
    gpu.barrier()
    if lane == 0:
        fx.ptr_store(wv, red + wave)
    gpu.barrier()
    t = fx.ptr_load(red + 0)
    for w in range_constexpr(1, WAVES):
        t = fx.max(t, fx.ptr_load(red + w))
    gpu.barrier()
    return t
