# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-Flash-Next's device helpers that ``sgl_amd_mono.common`` does not hold:
the guarded wait of a one-GPU launch."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl

from . import guard
from .ops import ballot

CHECK_POLLS = 64  # a pending wait checks its deadline every CHECK_POLLS polls


@flyc.jit
def spin(load_all, pending):
    """``load_all()`` again while ``pending(value)``; the value.

    The wait of a launch that runs on one GPU (TP 1, no peers): a pending wait
    re-polls with a single ``s_nop`` between loads and checks its deadline every
    ``CHECK_POLLS`` polls. Past it, it sets the error word and stops the launch
    (``guard.Guard.fail``): with no peers there is no failure elsewhere to watch,
    and a failed wait on this GPU has already stopped it."""
    g = guard.current()
    assert g is not None and g.peers is None, "a one-GPU wait runs under a guard without peers"
    site = g.next_site()
    fail = g.fail
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
            late = ((now - t0) > fx.Int64(guard.DEADLINE_TICKS)).select(fx.Int32(1), fx.Int32(0))
    if late != 0:
        fail(site)
    return v
