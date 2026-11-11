# SPDX-License-Identifier: Apache-2.0
"""Column-slice collectives of the wide-row launch.

The residual stream is column-sharded: rank r owns columns [r sl, (r + 1) sl)
of an [m, H] activation, sl = H / TP. Three collectives move slices:

* RS: every rank holds a partial [m, H]; the owner of slice j sums the TP
  partials of its columns -> [m, sl].
* AG: every owner holds its [m, sl]; every rank gets the [m, H].
* AR (two-shot): RS, then AG of the reduced slice.

Two formats:

* "tag": each bf16 pair travels next to the launch tag (``Mailbox``), so a
  consumer polls exactly the pairs it reads and needs no flag. Ties or beats
  "flag" at every measured size (8 .. 384 rows x 7168) despite twice the bytes.
* "flag": plain 16 B stores into the peer's uncached memory, drained with
  ``s_waitcnt vmcnt(0)`` before a tagged flag a (chunk, source); CTA b handles
  chunk b of every slice on every rank, and a consumer waits on its chunk's
  flags. Half the link bytes, for when the link is shared with overlapped work.
  The vmcnt drain orders the stores before the flag on gfx950 (acked uncached
  writes), which is measured, not documented: ``coll_bench.py --stress``.

Both reduce in f32 in rank order, so the result does not depend on timing.
Each region is double-buffered by the launch parity: a fast rank may push the
next launch's slices while a slow peer still polls this launch's, and the next
tag would never match that poll.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from . import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from . import guard
from .ops import (
    CM_SYS,
    bf2_f32,
    bf16_pair,
    clock,
    ld_i32,
    load_ptr64,
    rsrc,
    sfresh64,
    sload64,
    uniform,
)
from .plan import BLOCKS, THREADS, WAVES, key_tuple
from .sync import Mailbox, preg, shift
from .tag import step_tag

SOURCES = "aka-flat"  # AKA: no source digest (FlyDSL hashes the flat package's helpers)

TP = 8
KINDS = ("rs", "ag", "ar")
FORMATS = ("tag", "flag")


def layout(m, h, fmt, blocks=BLOCKS):
    """Byte offsets of one parity half: {"rs", "ag", "fl"} and the half's size.
    "tag" stores a 4 B tag next to every bf16 pair (2x the payload)."""
    eb = 4 if fmt == "tag" else 2  # bytes an element in a region
    rs = TP * m * (h // TP) * eb
    ag = m * h * eb
    fl = 2 * TP * blocks * 8 if fmt == "flag" else 0
    half = (rs + ag + fl + 255) // 256 * 256
    return {"rs": 0, "ag": rs, "fl": rs + ag}, half


def peer_bytes(m, h, fmt, blocks=BLOCKS):
    """The peer buffer a standalone collective needs: two halves + the guard flags."""
    return 2 * layout(m, h, fmt, blocks)[1] + guard.FLAG_BYTES


def flags_offset(m, h, fmt, blocks=BLOCKS):
    return 2 * layout(m, h, fmt, blocks)[1]


def _sum8(words_of_src):
    """TP sources' bf16 pair words -> 2 f32 a word, summed in rank order."""
    acc = None
    for src in range_constexpr(TP):
        cur = [v for w in words_of_src[src] for v in bf2_f32(w)]
        acc = cur if acc is None else [a + b for a, b in zip(acc, cur)]
    return acc


def _pack(vals):
    return [
        bf16_pair(vals[2 * i], vals[2 * i + 1]).bitcast(fx.Int32)
        for i in range(len(vals) // 2)
    ]


# ------------------------------------------------------------------ tagged
# Unit: 2 bf16 pairs (4 elements) a thread, one 16 B tagged store.


@flyc.jit
def tag_push(c, src, lds, m, sl, dst, d0, ldd, w0, nw):
    """Workers w0 .. of nw copy the [m, sl] block at ``src`` (row stride ``lds``
    elements) into tagged pairs at ``dst`` (row stride ``ldd``, + ``d0`` elements)."""
    r = rsrc(src)
    q4 = sl // 4
    for it in range(w0, m * q4, nw):
        row, q = it // q4, it % q4
        v = fx.Vector(
            bo.buffer_load(r, (row * lds + 4 * q) // 2, vec_width=2, dtype=T.i32)
        )
        c["put_words"](dst, (d0 + row * ldd + 4 * q) // 2, [v[0], v[1]])


@flyc.jit
def tag_reduce(c, m, sl, w0, nw, emit):
    """The owner sums its slice's TP tagged partials (rs region [src][m][sl]) and
    hands ``emit(row, col, words)`` 2 packed bf16 pair words a unit."""
    q4 = sl // 4
    for it in range(w0, m * q4, nw):
        row, q = it // q4, it % q4
        got = c["poll"](
            [
                (c["rs_own"], (src * m * sl + row * sl + 4 * q) // 2, 2)
                for src in range(TP)
            ]
        )
        emit(row, 4 * q, _pack(_sum8(got)))


@flyc.jit
def tag_take(c, m, h, y, w0, nw):
    """Poll the gathered [m, h] (ag region) and store it to ``y``."""
    r = rsrc(y)
    for it in range(w0, m * h // 4, nw):
        w = c["poll"]([(c["ag_own"], 2 * it, 2)])[0]
        bo.buffer_store(fx.Vector.from_elements(w, fx.Int32), r, 2 * it)


def tag_collective(c, kind, x, y, m, h):
    """One collective in the tagged format over the whole grid. CTA group j
    (BLOCKS / TP CTAs) pushes to rank j, so a wave's destination is uniform."""
    sl = h // TP
    gid = c["bid"] * THREADS + c["tid"]
    nall = BLOCKS * THREADS
    grp = BLOCKS // TP
    j = c["bid"] // grp
    gw = (c["bid"] % grp) * THREADS + c["tid"]
    rank = c["rank"]
    dst_rs = c["rs_of"](j)
    dst_ag = c["ag_of"](j)
    if const_expr(kind in ("rs", "ar")):
        tag_push(
            c,
            x + fx.Int64(j * sl) * 2,
            h,
            m,
            sl,
            dst_rs,
            rank * m * sl,
            sl,
            gw,
            grp * THREADS,
        )
    if const_expr(kind == "ag"):
        tag_push(c, x, sl, m, sl, dst_ag, rank * sl, h, gw, grp * THREADS)
    if const_expr(kind == "rs"):
        r = rsrc(y)

        def store(row, col, words):
            bo.buffer_store(
                fx.Vector.from_elements(words, fx.Int32), r, (row * sl + col) // 2
            )

        tag_reduce(c, m, sl, gid, nall, store)
    else:
        if const_expr(kind == "ar"):
            ags = [c["ag_of"](p) for p in range(TP)]

            def bcast(row, col, words):
                for p in range_constexpr(TP):
                    c["put_words"](ags[p], (row * h + rank * sl + col) // 2, words)

            tag_reduce(c, m, sl, gid, nall, bcast)
        if const_expr("late" in c):
            c["late"]()
        tag_take(c, m, h, y, gid, nall)


# -------------------------------------------------------------------- flag
# Unit: 16 B (8 bf16) a thread; CTA b owns units [b C, (b + 1) C) of a slice.


@flyc.jit
def put_each_peer(c, puts):
    """``puts``: [(region, i, v)]; lane 0 of wave p puts each to rank p's region (p < TP).

    One wave-uniform peer address per wave: with ``if tid == p`` LLVM folds the p branches into a switch on tid
    and merges the puts behind a per-lane address, a waterfall loop the bundled backend miscompiles.
    """
    assert WAVES >= TP
    tid = c["tid"]
    p = uniform(tid // 64)
    if (p < TP) & (tid % 64 == 0):
        for region, i, v in puts:
            c["put"](c[f"{region}_of"](p), i, v)


def _flag_signal(c, phase):
    """Drain this CTA's stores, then flag (phase, this rank, this CTA) on every rank."""
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    put_each_peer(
        c, [("fl", (phase * TP + c["rank"]) * BLOCKS + c["bid"], fx.Int32(1))]
    )


@flyc.jit
def _flag_wait(c, phase):
    """Wait for every rank's (phase, CTA) flag."""
    if c["tid"] < TP:
        c["poll"]([(c["fl_own"], (phase * TP + c["tid"]) * BLOCKS + c["bid"], 1)])
    gpu.barrier()


def _ld4(addr, word, cm=0):
    return fx.Vector(
        bo.buffer_load(rsrc(addr), word, vec_width=4, dtype=T.i32, cache_modifier=cm)
    )


def _st4(vec, addr, word, cm=0):
    bo.buffer_store(vec, rsrc(addr), word, cache_modifier=cm)


@flyc.jit
def flag_collective(c, kind, x, y, m, h):
    sl = h // TP
    q8 = sl // 8
    units = m * q8
    per = (units + BLOCKS - 1) // BLOCKS
    lo = c["bid"] * per
    hi = fx.min(lo + per, units)
    tid, rank = c["tid"], c["rank"]
    if const_expr(kind in ("rs", "ar")):
        for j in range_constexpr(TP):
            dst = c["rs_of"](j)
            for u in range(lo + tid, hi, THREADS):
                row, q = u // q8, u % q8
                v = _ld4(x, (row * h + j * sl + 8 * q) // 2)
                _st4(v, dst, (rank * m * sl + row * sl + 8 * q) // 2, CM_SYS)
        _flag_signal(c, 0)
        _flag_wait(c, 0)
        for u in range(lo + tid, hi, THREADS):
            row, q = u // q8, u % q8
            got = [
                _ld4(c["rs_own"], (src * m * sl + row * sl + 8 * q) // 2, CM_SYS)
                for src in range(TP)
            ]
            words = _pack(_sum8([[g[e] for e in range(4)] for g in got]))
            vec = fx.Vector.from_elements(words, fx.Int32)
            if const_expr(kind == "rs"):
                _st4(vec, y, (row * sl + 8 * q) // 2)
            else:
                for p in range_constexpr(TP):
                    _st4(vec, c["ag_of"](p), (row * h + rank * sl + 8 * q) // 2, CM_SYS)
    else:
        for p in range_constexpr(TP):
            dst = c["ag_of"](p)
            for u in range(lo + tid, hi, THREADS):
                row, q = u // q8, u % q8
                _st4(
                    _ld4(x, (row * sl + 8 * q) // 2),
                    dst,
                    (row * h + rank * sl + 8 * q) // 2,
                    CM_SYS,
                )
    if const_expr(kind != "rs"):
        _flag_signal(c, 1)
        _flag_wait(c, 1)
        if const_expr("late" in c):
            c["late"]()
        for src in range_constexpr(TP):
            for u in range(lo + tid, hi, THREADS):
                row, q = u // q8, u % q8
                word = (row * h + src * sl + 8 * q) // 2
                _st4(_ld4(c["ag_own"], word, CM_SYS), y, word)


# -------------------------------------------------------------- standalone


def context(peers, rank, epoch, slot, m, h, fmt, tid, bid, launches=1, parity="launch"):
    """The collective's mailbox, regions (this launch's parity half) and ids.

    A step bumps the epoch once and runs ``launches`` launches, slot 0 .. on one
    peer buffer. "launch" parity alternates the half every launch (global launch
    index e * launches + slot), so a rank that races into the next launch writes
    the other half; launch n + 2 may reuse n's half because every launch has an
    exchange in which each rank waits for every other, so no rank can reach n + 2
    before all have left n. "epoch" (the K3 tag scheme's) shares one half within a
    step: only for the test that shows the hazard."""
    offs, half = layout(m, h, fmt)
    return region_context(
        peers, rank, epoch, slot, offs, half, tid, bid, launches, parity
    )


def region_context(
    peers, rank, epoch, slot, offs, half, tid, bid, launches=1, parity="launch"
):
    """``context`` for any named peer regions (``offs``: name -> offset in a half):
    ``<name>_of(p)`` is rank p's region, ``<name>_own`` this rank's."""
    assert parity in ("launch", "epoch")
    mb = Mailbox(step_tag(epoch, slot))
    e = uniform(ld_i32(epoch, 0))
    bit = (e * launches + slot) % 2 if parity == "launch" else e % 2
    par = fx.Int64(bit) * half

    def base_of(p):
        # read from the peers table where used (scalar load through an opaque copy of the table address):
        # bases formed once at the entry stay live, or spilled, across the whole launch
        return sload64(sfresh64(peers) + fx.Int64(p) * 8)

    def region(name):
        def of(p):
            base = base_of(p) if isinstance(p, int) else load_ptr64(peers, p)
            return shift(preg(base, offs[name], name), par)

        return of

    c = _Context(
        {
            "tid": tid,
            "bid": bid,
            "rank": rank,
            "put": mb.put,
            "put_words": mb.put_words,
            "poll": mb.poll,
        }
    )
    for name in offs:
        c[f"{name}_of"] = region(name)
        c.lazy[f"{name}_own"] = lambda name=name: shift(
            preg(base_of(rank), offs[name], name), par
        )
    return c


class _Context(dict):
    """A region context whose ``<name>_own`` addresses are formed at each use (``lazy``), not at the entry."""

    def __init__(self, *a):
        super().__init__(*a)
        self.lazy = {}

    def __getitem__(self, k):
        if not dict.__contains__(self, k) and k in self.lazy:
            return self.lazy[k]()
        return super().__getitem__(k)

    def __contains__(self, k):
        return dict.__contains__(self, k) or k in self.lazy

    def get(self, k, default=None):
        return self[k] if k in self else default


# AKA: ``_delay`` and the standalone collective builder (test/kernels/kimi_k31/coll_bench.py only) are not
# vendored.
