# SPDX-License-Identifier: Apache-2.0
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 68506823, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/models/deepseek_v41/mono/kernels/index_score_fp4.py, by way of
# sgl-project/sglang#43984 at 134919476c (``_keys`` reads SGLang's page-64 FP4 index
# plane through virtual 8-row pages)
# Vendored from sgl-amd-mono @ 52486703 (sgl_amd_mono/deepseek_v41/gfx950/indexer/score_fp4.py).
# Modified for AgentKernelArena (FlyDSL 0.3.2 runtime):
#  * imports are package-relative (vendored buffer_ops / mx / ops / plan / index_plan);
#  * ``_lds_barrier`` fences through ops.memory_fence (an llvm-dialect fence; fx.memory_fence and
#    fx.AtomicOrdering need FlyDSL >= 0.3.4), same orderings and scope;
#  * ``_stage_rows`` loads a row's 16-byte units through ops.global_load (an llvm-dialect global
#    load, 16-byte aligned) instead of fx.ptr_load on an fx.inttoptr global pointer;
#  * ``run_scores4`` keeps the FULL branch (the walk plan's task) only: the REINDEX branch, which
#    scores a producer layer's candidate blocks from the CTA index, and its row count argument
#    are removed (this task scores FULL index-source layers; the launch passed reindex = 0).
"""K2a's indexer logits on SGLang's FP4 index plane, bit for bit with aiter's row-group kernel.

    kv  SGLang's [pages64, 4 (K chunk), 64 (key), 16 B] E2M1, read as virtual 8-row pages
    ks  SGLang's [pages64, 4 (K chunk), 64 (key % 16 * 4 + key // 16)] E8M0, one per 32 dims
    iq  [S, HEADS, DIM / 2] E2M1, iqs [S, 4 (K chunk), 16 (head % 16), 4 (head // 16, padded)]
        E8M0, iw [S, HEADS] bf16 (widening to f32 is exact)

Lane (g, l) of tile nt holds key 4 l + nt, so a step's page is one lookup and its four tiles'
scales one dword. The logits match aiter ``pa_mqa_logits_fp4_rowgroup`` only in its arithmetic
order: relu as x + |x| against half the weight, the lane's heads in one fma chain, then the
permlane swaps that leave lane (g, l) key 4 l + g's logit.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import Int32, T, as_ir_value

from . import buffer_ops as bo
from . import index_plan as ip
from .mx import FP4, mfma_scaled
from .ops import (
    CM_DEV,
    global_load,
    ld_i32,
    memory_fence,
    permlane_swap,
    rsrc,
)
from .plan import WAVES

PAGE = ip.BLOCK_ROWS
PAGE_BYTES = PAGE * ip.DIM // 2
PAGE_SCALES = PAGE * ip.DIM // 32
# SGLang's FP4 index plane (``deepseek_v4_memory_pool``): 64-row pages, payload and scales apart
PLANE_ROWS = 64
PLANE_CHUNK_BYTES = PLANE_ROWS * 16
PLANE_PAGE_BYTES = PLANE_ROWS * ip.DIM // 2
PLANE_PAGE_SCALES = PLANE_ROWS * ip.DIM // 32
M_TILES = ip.HEADS // 16
# LDS (words): each row's E2M1 query, E8M0 scales and weights, then the rows' bounds
Q_WORDS, QS_WORDS = ip.HEADS * ip.DIM // 8, ip.HEADS * 2
ROW_WORDS = Q_WORDS + QS_WORDS + ip.HEADS
BOUNDS_AT = ip.WALK_ROWS * ROW_WORDS
ROWS_LDS_WORDS = BOUNDS_AT + ip.WALK_ROWS
# waves whose threads hold the rows' words
ROW_WAVES = -(-ROW_WORDS // (4 * 64))
DEPTH = 2  # steps of keys a wave has in flight
OLDER_SHARE = 55  # percent of a SIMD pair's steps the older wave walks
N_KV = 5  # a step's key buffer: four tiles and their scales


def _vop(asm, *args):
    """One unpacked f32 op in asm: the SLP vectorizer would pair it into a
    packed op, which does not co-issue in an MFMA's shadow."""
    return fx.Float32(
        llvm.inline_asm(
            T.f32,
            [fx.Float32(x).ir_value() for x in args],
            asm,
            "=v" + ",v" * len(args),
            has_side_effects=False,
        )
    )


def _sload4(addr):
    """Four words at a wave-uniform address as a scalar load (the constant address space).

    The scalar cache's round trip is shorter than the vector memory's."""
    ptr = llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<4>"), as_ir_value(addr)).result
    v = fx.Vector(llvm.LoadOp(T.vec(4, T.i32), ptr, alignment=16).result)
    return [v[k] for k in range(4)]


def _read_all(args):
    """Read every kernel argument up front.

    LLVM loads one where it is first used, so the walk's first reads would wait on a scalar load."""
    vals = [as_ir_value(v) for v in args if not isinstance(v, int)]
    llvm.inline_asm(
        T.i32, vals, "s_mov_b32 $0, 0", "=s" + ",s" * len(vals),
        has_side_effects=True,
    )  # fmt: skip


def _swap_add(off, x, y):
    a, b = permlane_swap(off, x.bitcast(Int32), y.bitcast(Int32))
    return _vop("v_add_f32 $0, $1, $2", a.bitcast(fx.Float32), b.bitcast(fx.Float32))


def _lds_barrier():
    """A barrier for an LDS hand-off only: ``gpu.barrier`` would also wait on
    the walk's first reads, issued ahead of it to overlap it."""
    memory_fence(syncscope="workgroup", ordering="release")
    rocdl.s_barrier()
    memory_fence(syncscope="workgroup", ordering="acquire")


@flyc.jit
def _stage_rows(c, lead, span, first_keys=None):
    """Stage the piece's rows and bounds (0 past the span) into LDS, once a CTA, from ROW_WAVES.

    Per-wave copies put the CTA's waves on the same lines at once and queue ahead of their key
    reads. ``first_keys`` go out between the reads and their stores, as a store waits on its read,
    all in one branch: a store after it lets LLVM sink the reads past the keys."""
    a, tid = c["args"], c["tid"]
    lds = c["rows_lds"]
    q_units, qs_units = ip.HEADS * ip.DIM // 32, ip.HEADS * 2 // 4
    w_units = q_units + qs_units  # from here, a unit holds 4 heads' f32 weights
    # the branch's result needs a value of its type before it
    zero4 = fx.Vector.filled(4, 0, fx.Int32)
    init = ([zero4] * 4 + [fx.Int32(0)]) * DEPTH + [fx.Int32(0)] * DEPTH
    if tid < ROW_WAVES * 64:
        u = fx.min(tid, ROW_WORDS // 4 - 1)
        # bf16 weights: two units read one 16 B line of 8 heads, each widening its half (exact)
        wu = fx.max(u - w_units, 0)
        is_w, low_half = u >= w_units, (wu % 2) == 0
        got = []
        for r in range_constexpr(ip.WALK_ROWS):
            tok = lead + fx.min(fx.Int32(r), span - 1)
            byte = (u < q_units).select(
                a["iq"] + fx.Int64(tok * (q_units * 16) + u * 16),
                (u < w_units).select(
                    a["iqs"] + fx.Int64(tok * (qs_units * 16) + (u - q_units) * 16),
                    a["iw"] + fx.Int64(tok * (ip.HEADS * 2) + (wu // 2) * 16),
                ),
            )
            got.append(global_load(byte, 4))
        bounds = [
            (fx.Int32(r) < span).select(
                ld_i32(a["ibound"], lead + fx.min(fx.Int32(r), span - 1)), fx.Int32(0)
            )
            for r in range(ip.WALK_ROWS)
        ]
        if const_expr(first_keys is not None):
            init = first_keys()
        for r in range_constexpr(ip.WALK_ROWS):
            v = got[r]
            wa = low_half.select(v[0], v[2])
            wb = low_half.select(v[1], v[3])
            wide = (wa << 16, wa & -65536, wb << 16, wb & -65536)
            words = [is_w.select(wide[e], v[e]) for e in range(4)]
            fx.ptr_store(
                fx.Vector.from_elements(words, fx.Int32),
                lds + (r * ROW_WORDS + u * 4),
            )
        for r in range_constexpr(ip.WALK_ROWS):
            fx.ptr_store(bounds[r], lds + (BOUNDS_AT + r))
    elif const_expr(first_keys is not None):
        init = first_keys()
    return init


def _rows(c, rows):
    """The piece's first ``rows`` rows from LDS: (bound, query [mi], scale dword, half-weights).

    Rows past the span repeat its last with bound 0, so they write nothing."""
    lane, lds = c["lane"], c["rows_lds"]
    l, g = lane % 16, lane // 16
    v4 = fx.Vector.make_type(4, fx.Int32)
    out = []
    for r in range_constexpr(rows):
        row = lds + r * ROW_WORDS
        bound = fx.ptr_load(lds + (BOUNDS_AT + r))
        q = [
            fx.ptr_load(row + ((mi * 16 + l) * 16 + g * 4), result_type=v4)
            for mi in range(M_TILES)
        ]
        qs = fx.ptr_load(row + (Q_WORDS + g * 16 + l))
        w = []
        for mi in range_constexpr(M_TILES):
            w4 = fx.ptr_load(
                row + (Q_WORDS + QS_WORDS + mi * 16 + g * 4), result_type=v4
            ).bitcast(fx.Float32)
            w.append([w4[e] * fx.Float32(0.5) for e in range(4)])
        out.append((bound, q, qs, w))
    return out


def _page(c, trow, st):
    """This lane's virtual page of step st (keys 4 l .. 4 l + 3), off table row ``trow``.

    An entry names 2 ** ishift pages: a request's page table, or a REINDEX token's candidates
    (ishift 0)."""
    a, l = c["args"], c["lane"] % 16
    shift = a["ishift"]
    blk = fx.min(st * (ip.STEP // PAGE) + l // 2, (a["itab_len"] << shift) - 1)
    entry = ld_i32(a["itab"], trow * a["itab_stride"] + (blk >> shift))
    return (entry << shift) | (blk & ((fx.Int32(1) << shift) - 1))


def _keys(c, page, nt):
    """A step's four key tiles (16 B a lane each) and their scales (a dword).

    ``nt`` loads nontemporal, for keys no other wave reads (a piece's rows on one wave); not for a
    REINDEX row, whose request's other rows read the same candidate pages."""
    a, lane = c["args"], c["lane"]
    l, g = lane % 16, lane // 16
    # ``page`` is virtual: 8 x SGLang's 64-row page + the 8-row sub page
    sub = page & 7
    phys = fx.Int64(page >> 3)
    # payload [chunk][slot][16 B]: the lane's keys (slots 8 sub + 4 (l % 2) + nt) are 64 B in a row
    # 64-bit addresses: a pool past 4 GiB is out of a buffer offset's reach
    base = (
        a["plane"]
        + phys * PLANE_PAGE_BYTES
        + fx.Int64(g * PLANE_CHUNK_BYTES + (sub * 8 + (l % 2) * 4) * 16)
    )
    kv = [global_load(base + t * 16, 4, nt) for t in range(4)]
    # scales [chunk][(slot % 16) * 4 + slot // 16]: key nt's byte is 4 nt + sub // 2 of one line
    line = global_load(
        a["pscale"]
        + phys * PLANE_PAGE_SCALES
        + fx.Int64(g * 64 + (sub % 2) * 32 + (l % 2) * 16),
        4,
    )
    shift = (sub >> 1) * 8
    kvs = fx.Int32(0)
    for t in range_constexpr(4):
        kvs = kvs | (((line[t] >> shift) & 0xFF) << (8 * t))
    return kv, kvs


def _first_keys(c, pages, nt):
    """The first DEPTH steps' keys, then the next DEPTH's pages: a walk's
    ``init``."""
    init = []
    for first in range_constexpr(DEPTH):
        first_kv, first_kvs = _keys(c, pages[first], nt)
        init += first_kv + [first_kvs]
    return init + pages[DEPTH:]


@flyc.jit
def _step(c, rows, outs, kv, kvs, st, after_mfmas):
    """Step st's logits of every row; ``after_mfmas`` once its keys are read."""
    lane = c["lane"]
    l, g = lane % 16, lane // 16
    zero = fx.Vector.filled(4, 0.0, fx.Float32)
    per_tile = []
    for nt in range_constexpr(4):
        # raised while a tile's MFMAs issue, so the SIMD's other wave does not interleave
        rocdl.s_setprio(1)
        accs = [
            [
                mfma_scaled(q[mi], kv[nt], zero, qs, kvs, FP4, FP4, mi, nt)
                for mi in range(M_TILES)
            ]
            for _, q, qs, _ in rows
        ]
        rocdl.s_setprio(0)
        if const_expr(nt == 3):
            rocdl.sched_barrier(0)
            after_mfmas()
            rocdl.sched_barrier(0)
        sums = []
        for r in range_constexpr(len(rows)):
            w = rows[r][3]
            total = None
            for mi in range_constexpr(M_TILES):
                for e in range_constexpr(4):
                    x = accs[r][mi][e]
                    t = x + fmath.absf(x)
                    total = (
                        _vop("v_mul_f32 $0, $1, $2", t, w[mi][e])
                        if total is None
                        else _vop("v_fma_f32 $0, $1, $2, $3", t, w[mi][e], total)
                    )
            sums.append(total)
        per_tile.append(sums)
    key = st * ip.STEP + l * 4 + g
    for r in range_constexpr(len(rows)):
        p = [per_tile[nt][r] for nt in range(4)]
        lo = _swap_add(32, p[0], p[2])
        hi = _swap_add(32, p[1], p[3])
        bo.buffer_store(_swap_add(16, lo, hi), outs[r], key, cache_modifier=CM_DEV)


@flyc.jit
def _walk(c, lead, trow, count, step_of, init, rows_n):
    """This wave's ``count`` steps ``step_of(k)`` of the piece's first ``rows_n`` rows.

    DEPTH key buffers start from ``init``; each refills with the keys DEPTH steps on right after
    its step's MFMAs, and pages are read a trip further ahead."""
    a = c["args"]
    rows = _rows(c, rows_n)
    # a row's buffer ends at its bound, so stores past it are dropped
    outs = [
        rsrc(
            a["ilog"] + fx.Int64(lead + r) * fx.Int64(a["lstride"]) * 4,
            rows[r][0] * fx.Int32(4),
        )
        for r in range(rows_n)
    ]
    # names bound out here must not be rebound in the loop, or the rewriter carries them as state
    trips = count // DEPTH
    for j, state in range(0, fx.Int64(trips), 1, init=init):
        k0 = fx.Int32(j) * DEPTH
        pages_in = [state[DEPTH * N_KV + i] for i in range(DEPTH)]
        bufs, pages = [], []
        for i in range_constexpr(DEPTH):
            k = k0 + i

            def refill(i=i, k=k, pages_in=pages_in, bufs=bufs, pages=pages):
                page = (k + DEPTH < count).select(pages_in[i], fx.Int32(0))
                kv, kvs = _keys(c, page, rows_n > 1)
                bufs.append(kv + [kvs])
                pages.append(_page(c, trow, step_of(k + 2 * DEPTH)))

            mine = [state[i * N_KV + e] for e in range(N_KV)]
            _step(c, rows, outs, mine[:4], mine[4], step_of(k), refill)
        out = []
        for b in bufs:
            out += b
        carried = yield out + pages

    tail = count - trips * DEPTH
    for i in range_constexpr(DEPTH - 1):
        if fx.Int32(i) < tail:
            mine = [carried[i * N_KV + e] for e in range(N_KV)]
            _step(
                c, rows, outs, mine[:4], mine[4], step_of(trips * DEPTH + i),
                lambda: None,
            )  # fmt: skip


@flyc.jit
def stage_iscore4(c, lead, span, ch0, ch1, rows):
    """Chunks [ch0, ch1) of the piece of ``span`` tokens from ``lead``, a wave holding ``rows``.

    A WALK_ROWS piece issues its first keys after the rows' barrier, since before it each SIMD
    pair's younger wave waits behind the older one's; a REINDEX token (``rows`` 1) issues before."""
    wave = c["wave"]
    s_lo = ch0 * (ip.CHUNK // ip.STEP)
    nt = rows > 1

    # SIMD pair w, w + WAVES / 2 splits its steps, the older taking more as issue goes to it first
    pair, older = wave % (WAVES // 2), wave < WAVES // 2
    n_pair = ch1 * (ip.CHUNK // ip.STEP) - s_lo - pair
    n_pair = (n_pair > 0).select((n_pair + WAVES // 2 - 1) // (WAVES // 2), 0)
    split = n_pair * OLDER_SHARE // 100
    first = older.select(fx.Int32(0), split)

    def step_of(k):
        return s_lo + pair + (first + k) * (WAVES // 2)

    # the table row: a REINDEX token's own, a FULL piece's request's
    if const_expr(rows > 1):
        trow = fx.max(ld_i32(c["args"]["ibatch"], lead), 0)
    else:
        trow = lead
    pages = [_page(c, trow, step_of(fx.Int32(k))) for k in range(2 * DEPTH)]
    if const_expr(rows > 1):
        _stage_rows(c, lead, span)
        _lds_barrier()
        init = _first_keys(c, pages, nt)
    else:
        init = _stage_rows(c, lead, span, lambda: _first_keys(c, pages, nt))
        _lds_barrier()
    lds = c["rows_lds"]
    bounds = [fx.ptr_load(lds + (BOUNDS_AT + r)) for r in range(ip.WALK_ROWS)]
    # steps up to the rows' last scored chunk: the first reads went out on the chunks alone
    top = ip.score_chunks(bounds[0])
    for r in range_constexpr(1, ip.WALK_ROWS):
        top = fx.max(top, ip.score_chunks(bounds[r]))
    rem = fx.min(ch1, top) * (ip.CHUNK // ip.STEP) - s_lo - pair
    steps = (rem > 0).select((rem + WAVES // 2 - 1) // (WAVES // 2), fx.Int32(0))
    count = older.select(fx.min(split, steps), fx.max(steps - split, 0))
    if count > 0:
        _walk(c, lead, trow, count, step_of, init, rows)


@flyc.jit
def run_scores4(c, bid):
    """The FP4 score stage: this CTA's task, if any.

    A FULL layer's is its ``iplan`` row (lead, span, ch0, ch1; span 0 for none)."""
    a = c["args"]
    _read_all(a.values())
    task = _sload4(a["iplan"] + fx.Int64(bid) * 16)
    if task[1] > 0:
        stage_iscore4(c, task[0], task[1], task[2], task[3], ip.WALK_ROWS)
