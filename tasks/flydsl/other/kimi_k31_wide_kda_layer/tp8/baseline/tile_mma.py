# SPDX-License-Identifier: Apache-2.0
"""bf16 MFMA tile GEMM stage through LDS (the wide-row launch's dense GEMMs).

A task is a [bm token rows x bn weight rows] output tile over one K slice:
out[m0 .., n0 ..] = x[m0 .., k0 ..] @ W[n0 .., k0 ..]^T. The 8 waves form a
wm x wn grid of (bm / wm) x (bn / wn) sub-tiles, so no wave shares another's
output and nothing is reduced across waves. K walks in BK = 64 steps: the CTA
loads the step's x and W tiles (16 B a thread a piece) DEPTH steps ahead into
registers, writes them into one of two LDS buffers, and the waves read their
MFMA operands from LDS. LDS rows are padded to 72 bf16 (36 words): 16 rows of
one ds_read_b128 then start on 16 distinct 4-bank groups.

K is split across CTAs (``ks``) because a full-K 128 x 128 tile leaves most
CUs idle at these widths; with ks > 1 a task leaves f32 partials for the
consumer to sum (``tile.f32_emit``).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from . import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int64, T

from .ops import CM_NT, mfma_bf16, rsrc
from .plan import THREADS, WAVES, key_tuple
from .tile import (
    atomic_emit,
    atomic_lin,
    bf16_emit,
    f32_emit,
)

SOURCES = "aka-flat"  # AKA: no source digest (FlyDSL hashes the flat package's helpers)

BK = 64
RW = (BK + 8) // 2  # i32 words an LDS row
PIECES = BK // 8  # 16 B pieces a row a step
DEPTH = 3  # K steps of loads in flight


def lds_words(bm, bn):
    """i32 words of LDS a (bm, bn) task uses: two buffers of x and W rows (the
    shuffled epilogue's f32 tile reuses them)."""
    return max(2 * (bm + bn) * RW, bm * (bn + 4))


def _shuffle(tid, lane, wave, acc, cl, bm, bn, tm, tn, wy, wx, m0, lin):
    """Accumulators -> the f32 [bm, bn + 4] tile in LDS ``cl`` -> ``lin`` = (e,
    add): consecutive lanes add consecutive e-element runs of a row."""
    e, add = lin
    gpu.barrier()
    for t in range_constexpr(len(acc)):
        for b in range_constexpr(len(acc[0])):
            row = wy * tm + t * 16 + lane % 16
            col = wx * tn + b * 16 + 4 * (lane // 16)
            fx.ptr_store(acc[t][b], cl + row * (bn + 4) + col)
    gpu.barrier()
    per = 64 * e
    assert bn % per == 0 or per % bn == 0
    for i in range_constexpr(bm * bn // (per * WAVES)):
        idx = (i * WAVES + wave) * per + lane * e
        row, col = idx // bn, idx % bn
        p = cl + row * (bn + 4) + col
        if const_expr(e > 1):
            v = fx.Vector(
                fx.ptr_load(p, result_type=fx.Vector.make_type(e, fx.Float32))
            )
            add(m0 + row, col, [v[d] for d in range(e)])
        else:
            add(m0 + row, col, [fx.ptr_load(p)])


def _loads(tid, xr, ldx, m, m0, wr, ldw, n, n0, k0, bm, bn, xcm=0, wrow=None):
    """One K step's pieces: x rows m0 .. + bm, W rows n0 .. + bn (clamped; ``wrow`` maps a
    W row to the row it is stored at)."""
    out = []
    for rows, base, r, ld, lim, cm, rmap in (
        (bm, m0, xr, ldx, m, xcm, None),
        (bn, n0, wr, ldw, n, CM_NT, wrow),
    ):
        tot = rows * PIECES
        for i in range_constexpr((tot + THREADS - 1) // THREADS):
            e = (
                fx.min(tid + THREADS * i, tot - 1)
                if tot % THREADS
                else tid + THREADS * i
            )
            row = fx.min(base + e // PIECES, lim - 1)
            if const_expr(rmap is not None):
                row = rmap(row)
            word = (row * ld + k0 + (e % PIECES) * 8) // 2
            v = fx.Vector(
                bo.buffer_load(r, word, vec_width=4, dtype=T.i32, cache_modifier=cm)
            )
            out.append((e, v))
    return out


def _store(pieces, xs, ws, buf, bm, bn):
    """Write a step's pieces into LDS buffer ``buf`` (a clamped duplicate rewrites
    the same words with the same data)."""
    nx = len(pieces) - (bn * PIECES + THREADS - 1) // THREADS
    for j, (e, v) in enumerate(pieces):
        lds, rows = (xs, bm) if j < nx else (ws, bn)
        fx.ptr_store(v, lds + buf * rows * RW + (e // PIECES) * RW + (e % PIECES) * 4)


def _frag(lds, buf, rows, row, kk, lane):
    return fx.Vector(
        fx.ptr_load(
            lds + buf * rows * RW + row * RW + kk * 16 + (lane // 16) * 4,
            result_type=fx.Vector.make_type(4, fx.Int32),
        )
    )


def mma_tile(
    tid,
    x,
    ldx,
    m,
    m0,
    w,
    ldw,
    n,
    n0,
    k0,
    kt,
    bm,
    bn,
    wm,
    wn,
    xs,
    ws,
    emit,
    depth=DEPTH,
    lin=None,
    cl=None,
    xcm=0,
    wrow=None,
):
    """One task: K columns k0 .. k0 + kt of the [bm, bn] tile at (m0, n0);
    ``emit(row, col, v, live)`` gets 4 output columns n0 + col .. of token
    row ``row`` (absolute; the emit masks row >= m and columns >= n). With ``lin``
    (``tile.atomic_lin``) the tile is shuffled through the f32 LDS ``cl`` (aliasing
    xs / ws) into lane-contiguous adds instead. ``xcm``: x's load cache policy
    (CM_DEV when another XCD's CTA wrote x in this launch). ``wrow``: W row -> the row it
    is stored at (W rows gathered from slices of one buffer)."""
    assert wm * wn == WAVES and kt % BK == 0
    lane, wave = tid % 64, tid // 64
    tm, tn = bm // wm, bn // wn
    mt, nt = tm // 16, tn // 16
    assert mt * 16 == tm and nt * 16 == tn
    wy, wx = wave // wn, wave % wn
    xr, wr = rsrc(x), rsrc(w)
    steps = kt // BK
    acc = [[fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(nt)] for _ in range(mt)]

    def loads(j):
        return _loads(
            tid, xr, ldx, m, m0, wr, ldw, n, n0, k0 + j * BK, bm, bn, xcm, wrow
        )

    pend = [loads(j) for j in range(min(depth, steps))]
    rocdl.sched_barrier(0)
    _store(pend.pop(0), xs, ws, 0, bm, bn)
    if depth < steps:
        pend.append(loads(depth))
    gpu.barrier()
    for j in range_constexpr(steps):
        buf = j % 2
        for kk in range_constexpr(BK // 32):
            a = [
                _frag(ws, buf, bn, wx * tn + b * 16 + lane % 16, kk, lane)
                for b in range(nt)
            ]
            for t in range_constexpr(mt):
                xb = _frag(xs, buf, bm, wy * tm + t * 16 + lane % 16, kk, lane).bitcast(
                    fx.BFloat16
                )
                for b in range_constexpr(nt):
                    acc[t][b] = mfma_bf16(a[b].bitcast(fx.BFloat16), xb, acc[t][b])
        rocdl.sched_barrier(0)
        if j + 1 < steps:
            _store(pend.pop(0), xs, ws, 1 - buf, bm, bn)
            if j + 1 + depth < steps:
                pend.append(loads(j + 1 + depth))
            rocdl.sched_barrier(0)
            gpu.barrier()
    if const_expr(lin is not None):
        _shuffle(tid, lane, wave, acc, cl, bm, bn, tm, tn, wy, wx, m0, lin)
        return
    for t in range_constexpr(mt):
        for b in range_constexpr(nt):
            row = m0 + wy * tm + t * 16 + lane % 16
            col = wx * tn + b * 16 + 4 * (lane // 16)
            emit(row, col, acc[t][b], None)


# AKA: the standalone ``mma_tile`` launch builder (test/kernels/kimi_k31/mma_bench.py only) is not vendored.
