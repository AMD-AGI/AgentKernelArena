# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 97359d6, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/device/sync.py
# Vendored from sgl-amd-mono @ 52486703 (sgl_amd_mono/common/sync.py). Modified for
# AgentKernelArena: imports are package-relative and Addr is a frozen dataclass (no msgspec).
"""The tagged mailbox between CTAs (device scope) and GPUs (system scope)."""

from dataclasses import dataclass
from enum import Enum

import flydsl.compiler as flyc
import flydsl.expr as fx
from . import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from . import guard
from .ops import (
    CM_DEV,
    CM_SYS,
    bf2_f32,
    bf16_pair,
    rsrc,
    uniform,
)

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


class Space(Enum):
    SCRATCH = "scratch"  # this GPU's memory: device-scope hand-off between CTAs
    PEER = "peer"  # a peer-symmetric region: system-scope hand-off between GPUs


@dataclass(frozen=True)
class Addr:
    """A mailbox region's name, space and traced address; the space sets the scope."""

    region: str
    space: Space
    value: object


def sreg(base, offset, region):
    """Scratch region ``region`` at byte ``offset`` of ``base``."""
    return Addr(region, Space.SCRATCH, base + fx.Int64(offset))


def preg(base, offset, region):
    """Return the address of peer-symmetric region ``region`` at byte ``offset`` of buffer ``base``."""
    return Addr(region, Space.PEER, base + fx.Int64(offset))


def shift(a, delta):
    """``a`` moved by ``delta`` bytes (a traced or Python integer), same region."""
    return Addr(a.region, a.space, a.value + delta)


# the store / load cache policy of a hand-off in each space
_SCOPE_CM = {Space.SCRATCH: CM_DEV, Space.PEER: CM_SYS}


class Mailbox:
    """Tagged-pair hand-off between CTAs or GPUs: each value is stored next to the launch tag."""

    def __init__(self, tag):
        self.tag = tag

    def put(self, a, i, v):
        bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
        bo.buffer_store(
            fx.Vector.from_elements([bits, self.tag], fx.Int32),
            rsrc(a.value),
            i * 2,
            cache_modifier=_SCOPE_CM[a.space],
        )

    def put_bf(self, a, i, vs):
        """Store elements i .. i + len(vs) as packed bf16 pairs in one store."""
        words = []
        for j in range_constexpr(len(vs) // 2):
            words += [bf16_pair(vs[2 * j], vs[2 * j + 1]).bitcast(fx.Int32), self.tag]
        bo.buffer_store(
            fx.Vector.from_elements(words, fx.Int32),
            rsrc(a.value),
            i,
            cache_modifier=_SCOPE_CM[a.space],
        )

    def put_words(self, a, i, words):
        """Store raw 32-bit words at pairs i..; integer payloads must use this path."""
        vals = []
        for w in words:
            vals += [fx.Int32(w), self.tag]
        bo.buffer_store(
            fx.Vector.from_elements(vals, fx.Int32),
            rsrc(a.value),
            i * 2,
            cache_modifier=_SCOPE_CM[a.space],
        )

    def poll(self, specs, batch=POLL_MAX):
        """Poll the pairs in ``specs`` (one space) until every tag is current; return the value words."""
        if const_expr(len(specs) == 0):
            return []
        if const_expr(len(specs) > batch):
            return self.poll(specs[:batch], batch) + self.poll(specs[batch:], batch)
        load_all, pending, unpack = self._batch(specs)
        return unpack(_spin(load_all, pending))

    def poll_issue(self, specs):
        """Issue the first loads of ``poll(specs)`` (at most ``POLL_MAX``) now, ahead of
        loads that would otherwise hold their results back (loads return in issue
        order); ``poll_wait`` checks them."""
        assert 0 < len(specs) <= POLL_MAX, len(specs)
        return self._batch(specs)[0]()

    def poll_wait(self, specs, first):
        """``poll(specs)`` from the words ``poll_issue(specs)`` returned."""
        load_all, pending, unpack = self._batch(specs)
        return unpack(_spin(load_all, pending, first=first))

    def _batch(self, specs):
        spaces = {a.space for a, _, _ in specs}
        assert len(spaces) == 1, f"one poll batch spans {spaces}"
        cm = _SCOPE_CM[spaces.pop()]
        bases = [a.value for a, _, _ in specs]
        tag = self.tag

        def load_all():
            words = []
            for b, (_, i, n) in zip(bases, specs):
                w = fx.Vector(
                    bo.buffer_load(
                        rsrc(b),
                        fx.Int32(i) * 2,
                        vec_width=2 * n,
                        dtype=T.i32,
                        cache_modifier=cm,
                    )
                )
                words += [w[e] for e in range(2 * n)]
            return fx.Vector.from_elements(words, fx.Int32)

        nw = sum(2 * n for _, _, n in specs)

        def pending(v):
            bad = v[1] != tag
            for e in range_constexpr(3, nw, 2):
                bad = bad | (v[e] != tag)
            return bad

        def unpack(v):
            outs, e = [], 0
            for _, _, n in specs:
                outs.append([v[e + 2 * q] for q in range(n)])
                e += 2 * n
            return outs

        return load_all, pending, unpack


@flyc.jit
def publish(put, a, i, v, who):
    """Drain this CTA's plain stores, then store mailbox pair (``a``, i) := v from thread ``who``."""
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if who:
        put(a, i, v)


# ---------------------------------------------------------------- sentinel words
# A sentinel region holds plain words, not tagged pairs: an unwritten word is
# SENT (all ones, a bf16 -NaN pair) and a pushed word never is (``sent_word``).
# Its reader polls a word off SENT and re-arms it, so a region needs no tag,
# half a tagged pair's bytes, and nothing to zero a step. A word must not be
# pushed again before its reader re-armed it: the caller alternates two copies.
SENT = -1
SENT_NAN = -4128832  # 0xFFC0FFC0: what a pushed SENT word becomes


def sent_word(w):
    """A bf16-pair word to push to a sentinel region (never SENT)."""
    w = w.bitcast(fx.Int32) if isinstance(w, fx.Float32) else fx.Int32(w)
    return (w == fx.Int32(SENT)).select(fx.Int32(SENT_NAN), w)


def put_sent(a, i, word):
    """Word ``i`` of sentinel region ``a`` := ``word`` (from ``sent_word``)."""
    bo.buffer_store(fx.Int32(word), rsrc(a.value), i, cache_modifier=_SCOPE_CM[a.space])


def poll_sent(a, idxs, lives=None, rearm=True):
    """Words ``idxs`` of this rank's sentinel region ``a`` once none is SENT,
    re-armed unless ``rearm`` is off (the caller's ``rearm_sent`` then). A dead
    index (``lives``) is neither waited for nor re-armed. A read consumes the
    word: one thread a word."""
    cm = _SCOPE_CM[a.space]
    base = a.value

    def load_all():
        return fx.Vector.from_elements(
            [
                fx.Int32(
                    bo.buffer_load(
                        rsrc(base),
                        fx.Int32(i),
                        vec_width=1,
                        dtype=T.i32,
                        cache_modifier=cm,
                    )
                )
                for i in idxs
            ],
            fx.Int32,
        )

    def pending(v):
        bad = None
        for e in range_constexpr(len(idxs)):
            b = v[e] == fx.Int32(SENT)
            if const_expr(lives is not None):
                b = b & lives[e]
            bad = b if bad is None else bad | b
        return bad

    v = _spin(load_all, pending)
    if const_expr(rearm):
        rearm_sent(a, idxs, lives)
    return [v[e] for e in range(len(idxs))]


def rearm_sent(a, idxs, lives=None):
    """Words ``idxs`` of this rank's sentinel region ``a`` := SENT (a dead index
    skipped), after this thread read them."""
    cm = _SCOPE_CM[a.space]
    for e in range_constexpr(len(idxs)):
        if const_expr(lives is None):
            bo.buffer_store(
                fx.Int32(SENT), rsrc(a.value), fx.Int32(idxs[e]), cache_modifier=cm
            )
        else:
            _rearm(a.value, idxs[e], lives[e], cm)


@flyc.jit
def _rearm(base, i, live, cm):
    if live:
        bo.buffer_store(fx.Int32(SENT), rsrc(base), fx.Int32(i), cache_modifier=cm)


def sent_ix(src, col, t, s, width):
    """The word of (source rank, bf16 pair at ``col``, token t) in a [rank][token]
    [width] sentinel all-reduce region, blocked [src][16 columns][token][8 pairs]
    so ``sent_item``'s consecutive items read consecutive words."""
    return ((src * (width // 16) + col // 16) * s + t) * 8 + (col % 16) // 2


def sent_item(idx, s):
    """Reduce item ``idx``'s (token, column pair) within a 128-column task."""
    g = idx // (8 * s)
    r = idx % (8 * s)
    return r // 8, g * 8 + r % 8


def sent_sum(own, col, t, s, width, tp, rearm=True):
    """The ``tp`` ranks' words of (col pair, t) in sentinel region ``own``,
    summed in fp32 in rank order (as ``ops.sum_partials``)."""
    ws = poll_sent(
        own, [sent_ix(src, col, t, s, width) for src in range(tp)], rearm=rearm
    )
    acc0, acc1 = bf2_f32(ws[0])
    for src in range_constexpr(1, tp):
        lo, hi = bf2_f32(ws[src])
        acc0 = acc0 + lo
        acc1 = acc1 + hi
    return acc0, acc1
