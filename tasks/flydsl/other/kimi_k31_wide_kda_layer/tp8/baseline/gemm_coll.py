# SPDX-License-Identifier: Apache-2.0
"""Tile GEMMs fused with the column-slice collectives (one launch).

GEMM -> RS (o_proj, shared down): every rank computes its partial
out[m, n] = x @ W^T in [bm, bn] tiles split over ``ks`` K slices. A K slice
merges into a local scratch tile (device scope: the reader may sit on another
XCD), then counts itself in the tile's arrival counter and waits for the
tile's other slices; each slice then reads 1 / ks of the finished tile's rows,
pushes them to the owner of its columns (``n0 // sl``, so bn must divide sl)
and, with linbf, zeroes them (the consumer zeroes what it reads). After their
GEMM tasks the CTAs reduce this rank's slice as the pieces land, so the RS
overlaps the GEMM tail.

Remote stores are the cost: one CTA pushes only ~2-4 GB/s, so the push is
spread over every slice's CTA, and at 128+ rows the tagged format's doubled
bytes reach the links' limit, so "flag" (plain stores + a flag a piece) wins.
With ks = 1 there is no wait, no atomic and nothing to zero (deterministic).

Merge modes: "linbf" packed-bf16 atomics into a zeroed bf16 tile (K-slice
order follows timing); "part" f32 partials [ks, m, n] the last slice sums in
order (deterministic). The counters only grow: a launch adds ks a tile, so
the last arrival is the one that makes the count a multiple of ks.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from . import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from . import guard
from .compat import global_atomic_add
from .ops import (
    CM_DEV,
    CM_SYS,
    bf2_f32,
    bf16_pair,
    ld_i32,
    load_ptr64,
    rsrc,
)
from .plan import BLOCKS, THREADS, key_tuple
from .sync import _spin
from . import coll
from .tile import atomic_lin
from .tile_mma import DEPTH, RW, mma_tile

SOURCES = "aka-flat"  # AKA: no source digest (FlyDSL hashes the flat package's helpers)

TP = coll.TP
MODES = ("linbf", "part")


def scratch_bytes(m, n, ks, mode):
    """The local merge scratch: a bf16 [m, n] tile buffer (zeroed once) or f32 partials."""
    return ks * m * n * 4 if mode == "part" and ks > 1 else m * n * 2


def counters(m, n, bm, bn):
    return ((m + bm - 1) // bm) * (n // bn)


@flyc.jit
def _arrive(tid, cnt, tile, ks, slot):
    """Drain this CTA's merge, count it in ``tile``'s counter; True on the last slice."""
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if tid == 0:
        fx.ptr_store(global_atomic_add(cnt + fx.Int64(tile * 4), fx.Int32(1), syncscope="agent"), slot)
    gpu.barrier()
    return (fx.ptr_load(slot) + 1) % ks == 0


@flyc.jit
def _arrive_all(tid, cnt, tile, ks, slot, stride=1):
    """Count this K slice in ``tile``'s counter, then wait until every slice of
    the tile has: they are consecutive tasks of one round, so the wait is short."""
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if tid == 0:
        old = global_atomic_add(cnt + fx.Int64(tile * (4 * stride)), fx.Int32(1), syncscope="agent")
        target = (old // ks + 1) * ks
        _spin(lambda: ld_i32(cnt, tile * stride, CM_DEV), lambda v: (v - target) < 0)
    gpu.barrier()


def dev_emit(out, ldo, n0, m, n, f32):
    """``emit`` storing bf16 (or f32 partials) at device scope: another XCD's CTA reads them."""
    r_o = rsrc(out, m * ldo * (4 if f32 else 2))

    def emit(row, col, v, live):
        ok = (row < m) & (n0 + col < n)
        ok = ok if live is None else ok & live
        if const_expr(f32):
            bo.buffer_store(
                v, r_o, row * ldo + n0 + col, mask=ok, cache_modifier=CM_DEV
            )
        else:
            words = fx.Vector.from_elements(
                [
                    bf16_pair(v[0], v[1]).bitcast(fx.Int32),
                    bf16_pair(v[2], v[3]).bitcast(fx.Int32),
                ],
                fx.Int32,
            )
            bo.buffer_store(
                words, r_o, (row * ldo + n0 + col) // 2, mask=ok, cache_modifier=CM_DEV
            )

    return emit


@flyc.jit
def _push_tile(c, acc, m, n, m0, rb, n0, bn, ks, mode, sl):
    """Read rows m0 .. m0 + rb of the finished tile at column n0 (bn wide) and push
    them to their owner's rs region: every load is issued before the first push."""
    owner = n0 // sl
    dst = c["rs_of"](owner)
    r = rsrc(acc)
    q4 = bn // 4
    units = rb * q4
    base = c["rank"] * m * sl - owner * sl
    part = mode == "part" and ks > 1
    got = []
    for i in range_constexpr((units + THREADS - 1) // THREADS):
        it = fx.min(c["tid"] + i * THREADS, units - 1)
        row = m0 + it // q4
        rowc = fx.min(row, m - 1)
        col = n0 + 4 * (it % q4)
        if const_expr(part):
            vs = [
                fx.Vector(
                    bo.buffer_load(
                        r,
                        k * m * n + rowc * n + col,
                        vec_width=4,
                        dtype=T.f32,
                        cache_modifier=CM_DEV,
                    )
                )
                for k in range(ks)
            ]
        else:
            vs = [
                fx.Vector(
                    bo.buffer_load(
                        r,
                        (rowc * n + col) // 2,
                        vec_width=2,
                        dtype=T.i32,
                        cache_modifier=CM_DEV,
                    )
                )
            ]
        got.append((row, col, vs))
    for row, col, vs in got:
        if const_expr(part):
            s = [vs[0][e] for e in range(4)]
            for k in range_constexpr(1, ks):
                s = [s[e] + vs[k][e] for e in range(4)]
            words = [
                bf16_pair(s[0], s[1]).bitcast(fx.Int32),
                bf16_pair(s[2], s[3]).bitcast(fx.Int32),
            ]
        else:
            words = [vs[0][0], vs[0][1]]
        if row < m:
            c["put_words"](dst, (base + row * sl + col) // 2, words)
            if const_expr(not part and ks > 1):
                bo.buffer_store(
                    fx.Vector.filled(2, 0, fx.Int32),
                    r,
                    (row * n + col) // 2,
                    cache_modifier=CM_DEV,
                )


@flyc.jit
def _push_band_flag(c, acc, m, n, m0, rb, n0, bn, ks, mode, sl, piece):
    """Flag format: rows m0 .. m0 + rb of the finished tile as plain 16 B stores into
    the owner's rs region [src][m][sl], drained, then flag (this rank, piece)."""
    owner = n0 // sl
    dst = c["rs_of"](owner)
    r = rsrc(acc)
    q8 = bn // 8
    units = rb * q8
    part = mode == "part" and ks > 1
    got = []
    for i in range_constexpr((units + THREADS - 1) // THREADS):
        it = fx.min(c["tid"] + i * THREADS, units - 1)
        row = m0 + it // q8
        rowc = fx.min(row, m - 1)
        col = n0 + 8 * (it % q8)
        if const_expr(part):
            vs = [
                fx.Vector(
                    bo.buffer_load(
                        r,
                        k * m * n + rowc * n + col + 4 * h,
                        vec_width=4,
                        dtype=T.f32,
                        cache_modifier=CM_DEV,
                    )
                )
                for k in range(ks)
                for h in range(2)
            ]
        else:
            vs = [
                fx.Vector(
                    bo.buffer_load(
                        r,
                        (rowc * n + col) // 2,
                        vec_width=4,
                        dtype=T.i32,
                        cache_modifier=CM_DEV,
                    )
                )
            ]
        got.append((row, col, vs))
    for row, col, vs in got:
        if const_expr(part):
            s = [vs[h][e] for h in range(2) for e in range(4)]
            for k in range_constexpr(1, ks):
                s = [
                    s[4 * h + e] + vs[2 * k + h][e] for h in range(2) for e in range(4)
                ]
            words = [
                bf16_pair(s[2 * j], s[2 * j + 1]).bitcast(fx.Int32) for j in range(4)
            ]
        else:
            words = [vs[0][e] for e in range(4)]
        if row < m:
            bo.buffer_store(
                fx.Vector.from_elements(words, fx.Int32),
                rsrc(dst.value),
                (c["rank"] * m * sl + row * sl + col - owner * sl) // 2,
                cache_modifier=CM_SYS,
            )
            if const_expr(not part and ks > 1):
                bo.buffer_store(
                    fx.Vector.filled(4, 0, fx.Int32),
                    r,
                    (row * n + col) // 2,
                    cache_modifier=CM_DEV,
                )
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    # a partial tile's pieces past row m: no piece the owner reduces
    if (c["tid"] == 0) & (m0 < m):
        c["put"](c["fl_of"](owner), c["rank"] * BLOCKS + piece, fx.Int32(1))


# AKA: ``_reduce_flag`` and the standalone ``gemm_rs`` / ``ag_gemm`` builders (gemm_coll_bench.py and
# ag_gemm_bench.py only) are not vendored.
