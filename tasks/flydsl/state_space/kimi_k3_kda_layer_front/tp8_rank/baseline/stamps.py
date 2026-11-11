# SPDX-License-Identifier: Apache-2.0
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at a526f0d, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.: atom/mono/device/stamps.py
# Vendored from sgl-amd-mono @ 52486703 (sgl_amd_mono/kimi_k3/gfx950/stamps.py). Modified for
# AgentKernelArena: ``clock`` is imported from ops (upstream re-exports it through common/guard.py).
# The task always builds with the timeline off, so none of this traces to device code (the
# ``fx.generic_store`` in ``stamp_flush``, FlyDSL >= 0.3.4, is never reached on the 0.3.2 runtime).
"""A timeline build's per-CTA phase stamps, for debugging only.

A stamp records the 100 MHz clock at which the CTA passes point k. The record
stays in LDS until the kernel ends. A global store a stamp would sit on every
later ``s_waitcnt vmcnt(0)``.

``on`` is the build's compile-time switch (``K2Build.timeline``): off, every
call here traces to nothing, so a serving build pays for none of it.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr

from .ops import clock


@flyc.jit
def stamp_begin(on, tls, tid, points):
    """Zero the LDS record, then point 0."""
    if const_expr(on):  # noqa: SIM102
        if tid < points:  # the stamping thread's wave: ordered
            fx.ptr_store(fx.Int64(0), tls + tid)
    stamp(on, tls, tid, 0)


@flyc.jit
def stamp(on, tls, tid, k):
    # compile-time gate outside, traced condition inside: they cannot be one `and`
    if const_expr(on):  # noqa: SIM102
        if tid == 0:
            fx.ptr_store(clock(), tls + k)


@flyc.jit
def stamp_flush(on, tls, tl, tid, bid, points):
    """The CTA's record -> ``tl`` [CTA][points] int64."""
    if const_expr(on):  # noqa: SIM102
        if tid < points:
            fx.generic_store(
                fx.inttoptr(
                    fx.PointerType.get(fx.Int64.ir_type, fx.AddressSpace.Global, 8),
                    tl + fx.Int64((bid * points + tid) * 8),
                ),
                fx.ptr_load(tls + tid),
            )
