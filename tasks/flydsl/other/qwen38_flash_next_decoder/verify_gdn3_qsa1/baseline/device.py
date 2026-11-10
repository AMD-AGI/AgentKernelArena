# SPDX-License-Identifier: Apache-2.0
# Vendored from sgl-amd-mono @ 52486703 (sgl_amd_mono/qwen38_flash_next/gfx950/device.py).
# Modified for AgentKernelArena: imports are package-relative and the unguarded / fault-injection
# wait (``common.sync._spin``) is not vendored: every wait here runs under the launch's guard.
"""Qwen3.8-Flash-Next's device helpers that ``sgl_amd_mono.common`` does not hold:
the guarded wait of a one-GPU launch."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T

from . import buffer_ops as bo
from . import guard
from .ops import ballot, rsrc

CHECK_POLLS = 64  # a pending wait checks its deadline this often (common/sync.py)


def _deadline(epoch):
    """The guard's deadline in clock ticks (words 1-2 of the epoch buffer), loaded
    into scalar registers: a wait's check takes no vector register."""
    w = fx.Vector(
        bo.buffer_load(rsrc(epoch), 0, vec_width=4, dtype=T.i32, is_scalar=True)
    )
    lo, hi = w[guard.GUARD_DEADLINE], w[guard.GUARD_DEADLINE + 1]
    return (fx.Int64(hi) << 32) | (fx.Int64(lo) & fx.Int64(0xFFFFFFFF))


@flyc.jit
def spin(load_all, pending):
    """``load_all()`` again while ``pending(value)``; the value.

    The wait of a launch that runs on one GPU (TP 1, no peers): a pending wait
    re-polls with a single ``s_nop`` between loads and checks its deadline every
    ``CHECK_POLLS`` polls. Past it, it sets the error word and stops the launch
    (``guard.Guard.fail``): with no peers there is no failure elsewhere to watch,
    and a failed wait on this GPU has already stopped it. The fault injection's
    stall runs ``common.sync._spin``."""
    g = guard.current()
    assert g is not None and g.peers is None and g.fault_site is None, (
        "the vendored launch waits under its one-GPU guard only"
    )
    site = g.next_site()
    epoch, fail = g.epoch, g.fail
    v = load_all()
    t0 = fx.Int64(0)
    polls = fx.Int32(0)
    late = fx.Int32(0)
    # a wave-uniform loop (until no lane is pending), so its poll count and clock
    # stay in scalar registers; a satisfied lane reloads words that no longer change
    while (ballot(pending(v)) != 0) & (late == 0):
        rocdl.s_nop(0)
        v = load_all()
        polls = polls + 1
        if (polls & (CHECK_POLLS - 1)) == 0:
            now = guard.clock()
            t0 = (polls == CHECK_POLLS).select(now, t0)
            late = ((now - t0) > _deadline(epoch)).select(fx.Int32(1), fx.Int32(0))
    if late != 0:
        fail(site)
    return v
