# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 97359d6, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/device/sync.py
#
# Vendored for AgentKernelArena from sgl-amd-mono 52486703 (sgl_amd_mono/common/sync.py):
# only the guarded spin wait the DeepSeek-V4 layer uses (``_spin``, ``POLL_MAX``,
# ``CHECK_POLLS``); the msgspec mailbox classes and sentinel regions are left out.
"""The tagged mailbox spin wait between CTAs (device scope) and GPUs (system scope)."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, rocdl

from . import mono_guard as guard
from .mono_ops import uniform

POLL_MAX = 12  # mailbox specs polled per batch (bounds live registers)
CHECK_POLLS = 64  # a pending wait checks its deadline and the failure flags this often


def _first_load(load_all, first):
    """``first`` when the first load is in flight already, else ``load_all()`` (plain
    Python: the choice is made at trace time)."""
    return load_all() if first is None else first


@flyc.jit
def _spin(load_all, pending, guarded=True, first=None):
    """Re-issue ``load_all`` until ``pending`` clears (``first``: the first load's
    words, issued already).

    Under a guard (``guard.scope``) the wait also stops once its deadline passes
    or once this GPU or a peer failed: a timeout sets the error word and the
    peers' flags, a failure seen elsewhere sets the ``peer`` word, and then the
    launch stops (``guard.stop_launch``): it never runs on with words that did not
    come. ``guarded=False``: a wait the guard need not bound"""
    g = guard.current()
    if const_expr(g is None or not guarded):
        v = _first_load(load_all, first)
        while pending(v):
            rocdl.s_sleep(1)
            v = load_all()
        return v
    site = g.next_site()
    # bound before the traced loops: a local whose method a loop calls becomes loop state
    deadline_of, stopped, fail, note_peer = g.deadline, g.stopped, g.fail, g.note_peer
    v = _first_load(load_all, first)
    if const_expr(g.stop and g.fault_site != site):
        # a failed check stops the launch inside the loop, so the loop carries no stop
        # value: one copies the polled words every poll and spills at large batches
        scalar_deadline, has_peers = g.scalar_deadline, g.has_peers
        t0 = fx.Int64(0)
        polls = fx.Int32(0)
        while pending(v):
            rocdl.s_sleep(1)
            v = load_all()
            polls = uniform(polls + 1)
            if (polls & (CHECK_POLLS - 1)) == 0:
                now = guard.clock()
                t0 = (polls == CHECK_POLLS).select(now, t0)
                if fx.Int32((now - t0) >> guard.TICK_SHIFT) > scalar_deadline():
                    fail(site)
                # a failed wait on this GPU has stopped the launch already
                if const_expr(has_peers):
                    if stopped():
                        note_peer()
        return v
    stop = fx.Int32(0)  # 1: this wait timed out; 2: a failure elsewhere
    if const_expr(g.fault_site == site):
        # SGLANG_AMD_MONO_FAULT_INJECT: once armed, this rank stalls here until the deadline
        armed = g.armed()
        t0 = guard.clock()
        deadline = deadline_of()
        while armed & (stop == 0):
            rocdl.s_sleep(1)
            v = load_all()
            late = (guard.clock() - t0) > deadline
            stop = late.select(fx.Int32(1), stopped().select(fx.Int32(2), fx.Int32(0)))
    # a wait the data reaches within CHECK_POLLS polls checks nothing; the deadline
    # runs from the first check
    t0 = fx.Int64(0)
    polls = fx.Int32(0)
    while pending(v) & (stop == 0):
        rocdl.s_sleep(1)
        v = load_all()
        polls = polls + 1
        if (polls & (CHECK_POLLS - 1)) == 0:
            now = guard.clock()
            t0 = (polls == CHECK_POLLS).select(now, t0)
            late = (now - t0) > deadline_of()
            stop = late.select(fx.Int32(1), stopped().select(fx.Int32(2), fx.Int32(0)))
    if stop == 1:
        fail(site)
    if stop == 2:
        note_peer()
    return v
