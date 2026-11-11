# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 97359d6, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/device/sync.py
# Vendored from sgl-amd-mono @ 52486703 (sgl_amd_mono/common/sync.py). Modified for
# AgentKernelArena: imports are package-relative; ``Addr`` is a frozen dataclass instead of a
# msgspec Struct; only the scratch-space tagged mailbox the Kimi-K3 launches use is kept (the
# peer space, the sentinel words and their reductions are not); ``_spin`` keeps only its unguarded
# loop: the Kimi-K3 launches open no ``guard.scope``, so upstream's ``guard.current()`` is None
# for every wait they trace and the guarded branches are never taken.
"""The tagged mailbox between CTAs (device scope)."""

from dataclasses import dataclass
from enum import Enum

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from . import buffer_ops as bo
from .ops import CM_DEV, bf16_pair, rsrc

POLL_MAX = 12  # mailbox specs polled per batch (bounds live registers)


def _first_load(load_all, first):
    """``first`` when the first load is in flight already, else ``load_all()`` (plain
    Python: the choice is made at trace time)."""
    return load_all() if first is None else first


@flyc.jit
def _spin(load_all, pending, first=None):
    """Re-issue ``load_all`` until ``pending`` clears (``first``: the first load's
    words, issued already)."""
    v = _first_load(load_all, first)
    while pending(v):
        rocdl.s_sleep(1)
        v = load_all()
    return v


class Space(Enum):
    SCRATCH = "scratch"  # this GPU's memory: device-scope hand-off between CTAs


@dataclass(frozen=True)
class Addr:
    """A mailbox region's name, space and traced address; the space sets the scope."""

    region: str
    space: Space
    value: object


def sreg(base, offset, region):
    """Scratch region ``region`` at byte ``offset`` of ``base``."""
    return Addr(region, Space.SCRATCH, base + fx.Int64(offset))


def shift(a, delta):
    """``a`` moved by ``delta`` bytes (a traced or Python integer), same region."""
    return Addr(a.region, a.space, a.value + delta)


# the store / load cache policy of a hand-off in each space
_SCOPE_CM = {Space.SCRATCH: CM_DEV}


class Mailbox:
    """Tagged-pair hand-off between CTAs: each value is stored next to the launch tag."""

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
