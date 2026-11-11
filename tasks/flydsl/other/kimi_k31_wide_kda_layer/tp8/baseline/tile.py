# SPDX-License-Identifier: Apache-2.0
"""bf16 row-tile GEMM stage of the wide-row layer launch (16 .. 384 rows).

A task is ``16 nb`` weight rows (output columns n0 ..) against all ``m`` token
rows: out[0 .. m, n0 ..] = x @ W[n0 ..]^T. The 8 waves split K inside the CTA
(no split-K across CTAs: its finish, not the weight stream, bounds the dense
GEMMs at these widths) and reduce their partials through LDS at the end.

Both operands go straight to registers, in the K order of ``gemv``: K is
walked in 128-wide chunks, wave w taking chunks w, w + 8, ..., and a chunk is
four 32-K units (one ``mfma_f32_16x16x32_bf16`` each); lane group g = lane / 16
reads 8 elements at 32 q + 8 g. x does not fit LDS at these widths (128 x 7168
bf16 = 1.8 MB) and no wave shares another's K, so LDS only holds the K
reduction. Weights stream nontemporal; x is reread by every task from L2, so a
task's x traffic is m / (16 nb) times its weight traffic: wide m wants wide nb.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from . import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int64, T

from .ops import CM_NT, bf16_pair, mfma_bf16, rsrc
from .plan import THREADS, WAVES, key_tuple

SOURCES = "aka-flat"  # AKA: no source digest (FlyDSL hashes the flat package's helpers)

ROWS = 16
KCH = 128
UNIT = 32
DEPTH = 4  # 32-K units a wave keeps in flight
GROUP = 8  # 16x16 tiles a reduction round: WAVES x GROUP x 1 KB of LDS


def red_words(m, nb):
    """f32 words of LDS the reduction of an (m, nb) task uses."""
    return WAVES * min(GROUP, m_tiles(m) * nb) * 64 * 4


def m_tiles(m):
    return (m + ROWS - 1) // ROWS


def _units(k):
    n = k // KCH
    per = (n + WAVES - 1) // WAVES
    return per * (KCH // UNIT), per, n


def _load(r, word, cm):
    return fx.Vector(
        bo.buffer_load(r, word, vec_width=4, dtype=T.i32, cache_modifier=cm)
    )


def _unit_loads(lane, wave, j, xr, ldx, m, wr, ldw, nrows, nb, n, per):
    """Unit ``j`` of this wave: ``nb`` weight operands, ``m_tiles(m)`` x operands."""
    i, q = j // (KCH // UNIT), j % (KCH // UNIT)
    ch = fx.min(wave + WAVES * i, n - 1)
    kk = ch * KCH + UNIT * q + (lane // ROWS) * 8
    r = lane % ROWS
    ws = []
    for b in range_constexpr(nb):
        row = fx.min(b * ROWS + r, nrows - 1)
        ws.append(_load(wr, (row * ldw + kk) // 2, CM_NT))
    if per * WAVES != n:
        live = wave + WAVES * i < n
        ws = [
            fx.Vector.from_elements(
                [live.select(v[d], fx.Int32(0)) for d in range(4)], fx.Int32
            )
            for v in ws
        ]
    xs = []
    for t in range_constexpr(m_tiles(m)):
        row = t * ROWS + r if t * ROWS + ROWS <= m else fx.min(t * ROWS + r, m - 1)
        xs.append(_load(xr, (row * ldx + kk) // 2, 0))
    return ws, xs


def tile_acc(lane, wave, x, ldx, m, w, ldw, nrows, nb, k):
    """This wave's K share of the task: a [m_tiles][nb] grid of 16x16 f32
    accumulators (lane l, element e: weight row 4 (l / 16) + e, token l % 16).
    ``w`` points at the task's first weight row, ``nrows`` rows are valid."""
    u, per, n = _units(k)
    xr, wr = rsrc(x), rsrc(w)
    mt = m_tiles(m)
    acc = [[fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(nb)] for _ in range(mt)]
    args = (lane, wave)
    tail = (xr, ldx, m, wr, ldw, nrows, nb, n, per)
    pend = [_unit_loads(*args, j, *tail) for j in range(min(DEPTH, u))]
    # sched_barrier pins each unit's prefetch ahead of the MFMAs: left alone, the
    # scheduler sinks every load to just before its use (vmcnt(1) per MFMA)
    rocdl.sched_barrier(0)
    for j in range_constexpr(u):
        if j + DEPTH < u:
            pend.append(_unit_loads(*args, j + DEPTH, *tail))
        rocdl.sched_barrier(0)
        ws, xs = pend.pop(0)
        for t in range_constexpr(mt):
            xb = xs[t].bitcast(fx.BFloat16)
            for b in range_constexpr(nb):
                acc[t][b] = mfma_bf16(ws[b].bitcast(fx.BFloat16), xb, acc[t][b])
        rocdl.sched_barrier(0)
    return acc


def tile_reduce(tid, lane, wave, acc, red, m, nb, emit):
    """Sum the waves' accumulators through LDS ``red`` (``red_words`` f32) and
    call ``emit(row, col, v, live)`` for 4 output columns col .. col + 3 of token
    row ``row`` (task-relative, row < m_tiles(m) * 16: the caller masks rows and
    columns; ``live`` is None or the Boolean that the slot is real)."""
    mt = m_tiles(m)
    tiles = mt * nb
    g = min(GROUP, tiles)
    for r0 in range_constexpr(0, tiles, g):
        cnt = min(g, tiles - r0)
        for f in range_constexpr(cnt):
            t, b = divmod(r0 + f, nb)
            fx.ptr_store(acc[t][b], red + ((wave * g + f) * 64 + lane) * 4)
        gpu.barrier()
        for e in range_constexpr((cnt * 64 + THREADS - 1) // THREADS):
            idx = tid + THREADS * e
            fl = fx.min(idx // 64, cnt - 1)
            ln = idx % 64
            v = fx.Vector.filled(4, 0.0, fx.Float32)
            for wv in range_constexpr(WAVES):
                v = v + fx.Vector(
                    fx.ptr_load(
                        red + ((wv * g + fl) * 64 + ln) * 4,
                        result_type=fx.Vector.make_type(4, fx.Float32),
                    )
                )
            f = r0 + fl
            live = None if cnt * 64 % THREADS == 0 else idx < cnt * 64
            emit(
                (f // nb) * ROWS + ln % ROWS,
                (f % nb) * ROWS + 4 * (ln // ROWS),
                v,
                live,
            )
        gpu.barrier()


def bf16_emit(out, ldo, n0, m, n):
    """``emit`` storing bf16 into the row-major [m, ldo] at ``out``, columns n0 + col."""
    r_o = rsrc(out, m * ldo * 2)  # sized: a masked store is an out-of-range offset

    def emit(row, col, v, live):
        words = fx.Vector.from_elements(
            [
                bf16_pair(v[0], v[1]).bitcast(fx.Int32),
                bf16_pair(v[2], v[3]).bitcast(fx.Int32),
            ],
            fx.Int32,
        )
        ok = (row < m) & (n0 + col < n)
        ok = ok if live is None else ok & live
        bo.buffer_store(words, r_o, (row * ldo + n0 + col) // 2, mask=ok)

    return emit


def f32_emit(out, ldo, n0, m, n):
    """``emit`` storing f32 partials into the row-major [m, ldo] at ``out``."""
    r_o = rsrc(out, m * ldo * 4)

    def emit(row, col, v, live):
        ok = (row < m) & (n0 + col < n)
        ok = ok if live is None else ok & live
        bo.buffer_store(v, r_o, row * ldo + n0 + col, mask=ok)

    return emit


def atomic_emit(out, ldo, n0, m, n, bf16):
    """``emit`` adding into the zeroed row-major [m, ldo] at ``out``: f32 adds,
    or with ``bf16`` packed bf16x2 adds (half the bytes, bf16 rounding a K slice)."""
    eb = 2 if bf16 else 4
    size = m * ldo * eb
    r_o = rsrc(out, size)
    z = fx.Int32(0)

    def emit(row, col, v, live):
        ok = (row < m) & (n0 + col < n)
        ok = ok if live is None else ok & live
        # a masked lane adds at the buffer's end: out of range for every + 4 j
        # (0x7FFFFFFF + 4 j would wrap)
        off = ok.select((row * ldo + n0 + col) * eb, fx.Int32(size))
        if const_expr(bf16):
            for j in range_constexpr(2):
                pair = fx.Vector.from_elements([v[2 * j], v[2 * j + 1]], fx.Float32).to(
                    fx.BFloat16
                )
                rocdl.raw_ptr_buffer_atomic_fadd(pair, r_o, off + 4 * j, z, z)
        else:
            for j in range_constexpr(4):
                rocdl.raw_ptr_buffer_atomic_fadd(v[j], r_o, off + 4 * j, z, z)

    return emit


def atomic_lin(out, ldo, n0, m, n, bf16):
    """Lane-contiguous adds for an LDS-shuffled epilogue: (elements a lane,
    ``add(row, col, vals)``); consecutive lanes take consecutive columns, so a
    wave's add covers whole 64 B lines."""
    eb = 2 if bf16 else 4
    size = m * ldo * eb
    r_o = rsrc(out, size)
    z = fx.Int32(0)

    def add(row, col, vals):
        ok = (row < m) & (n0 + col < n)
        off = ok.select((row * ldo + n0 + col) * eb, fx.Int32(size))
        if const_expr(bf16):
            pair = fx.Vector.from_elements(vals, fx.Float32).to(fx.BFloat16)
            rocdl.raw_ptr_buffer_atomic_fadd(pair, r_o, off, z, z)
        else:
            rocdl.raw_ptr_buffer_atomic_fadd(vals[0], r_o, off, z, z)

    return (2 if bf16 else 1), add


# AKA: the standalone row-tile GEMM builder (test/kernels/kimi_k31/tile_bench.py only) is not vendored;
# the layer launch uses this module's epilogues (``atomic_lin``, ``bf16_emit``) through ``tile_mma``.
