# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Indexer block scores for K4's sparse selection (``select_blocks`` polls them).

The unfused decode scorer's math, one 128-key block of index-K per task: bf16(fp8 K) x
bf16 index q on the MFMA, max over the causal keys, times fp32(sm_scale * log2 e). A
task is one block of one request, whose q tokens (a speculative verify's rows) are the
MFMA's B columns, so the block's index-K is read once for all of them. A context of
<= TOPK_BLOCKS blocks keeps every block and is not scored, nor are the blocks the
selector pins (the first ``init_blocks``, the last ``local_blocks``): their keys are
forced. Emitted at the end of K1 (``emit_index_scores``), which makes index q, so K4's
selector finds the scores in place.
"""

import math
from typing import NamedTuple

import flydsl.compiler as flyc
import flydsl.expr as fx
from . import buffer_ops as bo
from .act import LOG2E
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from .ops import (
    butterfly,
    fp8_to_bf16x8,
    lane_gather,
    mfma_bf16,
    permlane_swap,
    rsrc,
    uniform,
)
from .config import (
    BLOCKS,
    HEAD_DIM,
    INDEX_CP_FROM_BLOCKS,
    MAX_INDEX_BLOCKS,
    MAX_TOKENS,
    ONE_INDEX_HEAD,
    SPARSE_BLOCK,
    THREADS,
    TOPK_BLOCKS,
    TP,
    WAVES,
    IndexHeads,
)
from .device import readlane, unlikely

INDEX_BLOCK_BYTES = SPARSE_BLOCK * HEAD_DIM  # one fp8 index-K block
KEY_TILES = SPARSE_BLOCK // 16  # MFMA row tiles of a block


def _f32(x: float) -> float:
    """``x`` rounded to the nearest float32, ties to even (normal range: the scales here)."""
    if x == 0.0 or not math.isfinite(x):
        return x
    m, e = math.frexp(x)
    return math.ldexp(round(m * (1 << 24)), e - 24)


def index_scale_log2e(sm_scale: float) -> float:
    """fp32(sm_scale * log2 e), the scorer's scale."""
    return _f32(_f32(sm_scale) * _f32(LOG2E))


def scored_count(n_blk, init_blocks: int, local_blocks: int):
    """How many blocks (from block init_blocks on) a context of n_blk blocks scores."""
    return (n_blk > TOPK_BLOCKS).select(n_blk - (init_blocks + local_blocks), 0)


class StepRows(NamedTuple):
    """A decode step's rows as the indexer sees them (``step_rows``)."""

    seq: list  # each row's context, >= 1 (a graph's pad rows carry 0)
    n_blks: list  # each row's index blocks
    long: list  # each row's request is long

    def any_long(self):
        v = self.long[0]
        for lg in self.long[1:]:
            v = v | lg
        return v


def step_rows(seq_lens, tokens: int, q_len, heads: IndexHeads) -> StepRows:
    """The one place a request is decided long: past INDEX_CP_FROM_BLOCKS index
    blocks with every index head (its scores take the context-parallel layout,
    ``emit_index_scores``), past THREADS with one (more blocks than
    ``rank_blocks`` ranks); either way K4 selects it past the split stage.

    Decided by the request's last row (a request is ``q_len`` consecutive rows):
    a speculative verify's rows can sit either side of the threshold, and a
    request's rows must all read the layout its scores were written in."""
    seq = [
        fx.max(uniform(bo.buffer_load(rsrc(seq_lens), k, vec_width=1, dtype=T.i32)), 1)
        for k in range(tokens)
    ]
    n_blks = [(s + SPARSE_BLOCK - 1) // SPARSE_BLOCK for s in seq]
    req_blks = list(n_blks)
    for k in range(tokens - 2, -1, -1):
        req_blks[k] = (fx.Int32(k) % q_len == q_len - 1).select(
            n_blks[k], req_blks[k + 1]
        )
    past = INDEX_CP_FROM_BLOCKS if heads.count > 1 else THREADS
    return StepRows(seq, n_blks, [nb > past for nb in req_blks])


def _pick(vals, k):
    """vals[k] for a traced index k."""
    v = vals[0]
    for j in range_constexpr(1, len(vals)):
        v = (k == j).select(vals[j], v)
    return v


@flyc.jit
def emit_index_scores(
    slot, lane, wave, red, mb_put, iscore, q_frag, tokens, q_len, init_blocks,
    local_blocks, scale, index_cache, block_table, seq_lens, bt_width,
    wait_new_keys, prepare, heads=ONE_INDEX_HEAD, q_frag_head=None,
    prepare_heads=None,
):  # fmt: skip
    """This CTA's (``slot``) share of the step's score tasks: pair k * MAX_INDEX_BLOCKS
    + b of mailbox ``iscore`` <- the score of token k's block b.

    Up to BLOCKS tasks, task u runs on CTA u, else on wave (u // BLOCKS) % WAVES of CTA
    u % BLOCKS (``red``: WAVES * 16 floats of LDS). The tokens form requests of
    ``q_len`` consecutive rows sharing a block table; the last has the longest context
    and lists the request's scored blocks. ``q_frag(k, s)``: this lane's B operand for
    token k, k-chunk s (bf16x8 of index q). ``wait_new_keys()`` (wave-local) waits for
    the index keys this launch inserts, which a block within MAX_TOKENS keys of a
    token's end may hold (a speculative verify's earlier rows).
    ``prepare(first, last)`` (with CTA barriers) runs on a CTA with tasks before any
    index q is read: K1 makes rows first..last's there. Under indexer context
    parallelism (``heads.count`` > 1) a long request (``step_rows``) scores only this
    rank's blocks, b = heads.own (mod TP), for every index q head, a key block read
    once for all: pair k * MAX_INDEX_BLOCKS + h * (MAX_INDEX_BLOCKS / TP) + b / TP <-
    token k's block b for head h. ``q_frag_head(k, s, h)`` is head h's B operand
    (every lane its own h), and ``prepare_heads()`` (with CTA barriers, on every CTA)
    makes those requests' rows' heads first."""
    assert local_blocks >= 1, "the block of the current token must be pinned"
    l16 = lane % 16
    g4 = lane // 16
    rows = step_rows(seq_lens, tokens, q_len, heads)
    seq = rows.seq
    # one task list of every request's scored blocks, and one of CP requests' own blocks (pre_cp)
    cp = heads.count > 1
    first_own = init_blocks + (heads.own - init_blocks) % TP
    pre = [fx.Int32(0)]
    pre_cp = [fx.Int32(0)]
    for k in range_constexpr(tokens):
        last = fx.Int32(k) % q_len == q_len - 1
        n_scored = scored_count(rows.n_blks[k], init_blocks, local_blocks)
        if const_expr(cp):
            long = rows.long[k]
            n_own = fx.max(init_blocks + n_scored - first_own + TP - 1, 0) // TP
            pre.append(pre[-1] + (last & ~long).select(n_scored, fx.Int32(0)))
            pre_cp.append(pre_cp[-1] + (last & long).select(n_own, fx.Int32(0)))
        else:
            pre.append(pre[-1] + last.select(n_scored, fx.Int32(0)))
    total = pre[tokens]
    total_cp = pre_cp[-1]  # 0 without context parallelism

    r_bt = rsrc(block_table)

    def task_block(u, pre=pre, first=init_blocks, stride=1):
        """Task u of list ``pre`` -> (the request's last token, block b); per lane
        if u is."""
        tk = fx.Int32(0)
        for j in range_constexpr(1, tokens):
            tk = tk + (u >= pre[j]).select(fx.Int32(1), fx.Int32(0))
        return tk, first + stride * (u - _pick(pre[:tokens], tk))

    def page_of(tk, b):
        return fx.Int32(
            bo.buffer_load(r_bt, tk * bt_width + b, vec_width=1, dtype=T.i32)
        )

    def new_keys_in(tk, b):
        """Block b of the request ending at token tk may hold keys this launch
        inserts (a speculative verify's earlier rows)."""
        return (b + 1) * SPARSE_BLOCK > _pick(seq, tk) - MAX_TOKENS

    def cols_of(tk):
        """B column l16's token (the request's token l16, the last one repeated)
        and its context."""
        col = tk - (q_len - 1) + fx.min(l16, q_len - 1)
        return col, _pick(seq, col)

    def load_raw(page, tiles):
        """Key tiles ``tiles`` (16 keys each) of a block, in flight; ``split_tiles``
        makes them MFMA operands.

        Lane g4 = lane / 16 reads its key's bytes 32 g4 .. 32 g4 + 32 with two 16 B
        loads, half the load instructions of eight-byte operand reads scattered over
        the rows. ``page`` is uniform: a 64-bit base at it keeps the offsets 32-bit for
        a cache of any size."""
        r_ic = rsrc(index_cache + fx.Int64(page) * INDEX_BLOCK_BYTES)
        words = []
        for i in tiles:
            row = ((16 * i + l16) * HEAD_DIM + 32 * g4) // 4
            for h in range_constexpr(2):
                w = fx.Vector(
                    bo.buffer_load(r_ic, row + 4 * h, vec_width=4, dtype=T.i32)
                )
                words += [w[e] for e in range(4)]
        return fx.Vector.from_elements(words, fx.Int32)

    def split_tiles(raw, tiles):
        """``load_raw``'s words -> per tile the MFMA A operands: tile i holds keys
        16 i + lane % 16, k = 32 s + 8 (lane / 16). A 4 x 4 transpose of the 8 B
        parts across the key's four lanes (permlane swaps) gives each lane part
        g4 of every chunk s."""
        out = []
        for n in range_constexpr(len(tiles)):
            d = [raw[8 * n + e] for e in range(8)]
            # part j = dwords 2 j, 2 j + 1; lanes g4 and g4 ^ 2, then g4 ^ 1, trade
            for off, pairs in ((32, ((0, 2), (1, 3))), (16, ((0, 1), (2, 3)))):
                for a, b in pairs:
                    for w in range_constexpr(2):
                        d[2 * a + w], d[2 * b + w] = permlane_swap(
                            off, d[2 * a + w], d[2 * b + w]
                        )
            out.append(
                [
                    fx.Vector.from_elements([d[2 * s], d[2 * s + 1]], fx.Int32)
                    for s in range(HEAD_DIM // 32)
                ]
            )
        return out

    def tiles_max(b, qb, col_seq, tiles, kw):
        """Max over key tiles ``tiles`` of block b (``kw``: ``split_tiles``) of this
        lane's column's causal scores (its B operand ``qb``, per k-chunk; the
        column's context ``col_seq``), the same in every lane of the column."""
        v = fx.Float32(float("-inf"))
        for n, i in enumerate(tiles):
            c = fx.Vector.filled(4, 0.0, fx.Float32)
            for s in range_constexpr(HEAD_DIM // 32):
                c = mfma_bf16(fp8_to_bf16x8(*kw[n][s]), qb[s], c)
            # lane holds keys 16 i + 4 (lane / 16) + e of column lane % 16
            key0 = b * SPARSE_BLOCK + 16 * i + 4 * g4
            for e in range_constexpr(4):
                v = fx.max(v, (key0 + e < col_seq).select(c[e], float("-inf")))
        return butterfly(v, (32, 16), fx.max)

    def put(b, col, v):
        if (lane < 16) & (l16 < q_len):
            mb_put(iscore, col * MAX_INDEX_BLOCKS + b, v * scale)

    def emit_cp_tasks():
        """The context-parallel requests' tasks, a task a wave: B column f of pass
        p (f = 16 p + lane % 16) is head f / q_len's row f % q_len of the request,
        the key block's operands kept for every pass."""
        if const_expr(prepare_heads is not None):
            prepare_heads()
        tiles = range(KEY_TILES)
        seq_lane = _pick(seq, fx.min(lane, tokens - 1))  # lane k: token k's context
        n_cols = TP * q_len

        def score_cp(tk, b, page):
            """Block b of the request ending at token tk, at ``page``: every head."""
            if (b + 1) * SPARSE_BLOCK > readlane(seq_lane, tk) - MAX_TOKENS:
                wait_new_keys()
            kw = split_tiles(load_raw(page, tiles), tiles)
            # passes unrolled, unused ones skipped: a runtime pass loop slows every short context
            for p in range_constexpr(TP * MAX_TOKENS // 16):
                if p * 16 < n_cols:
                    f = p * 16 + l16
                    head = fx.min(f // q_len, TP - 1)
                    col = tk - (q_len - 1) + f % q_len
                    qb = [q_frag_head(col, s, head) for s in range(HEAD_DIM // 32)]
                    v = tiles_max(b, qb, lane_gather(seq_lane, col), tiles, kw)
                    if (lane < 16) & (f < n_cols):
                        mb_put(
                            iscore,
                            col * MAX_INDEX_BLOCKS
                            + head * (MAX_INDEX_BLOCKS // TP)
                            + b // TP,
                            v * scale,
                        )

        # the same task placement and per-lane decode as the wave mode below
        step = BLOCKS * WAVES
        for c0 in range(slot + BLOCKS * wave, total_cp, step * 64):
            c0 = fx.Int32(c0)
            tk_j, b_j = task_block(
                fx.min(c0 + step * lane, total_cp - 1), pre_cp, first_own, TP
            )
            pages = page_of(tk_j, b_j)
            for i in range(fx.min((total_cp - c0 + step - 1) // step, 64)):
                i = fx.Int32(i)
                score_cp(readlane(tk_j, i), readlane(b_j, i), readlane(pages, i))

    # a task a CTA, a key tile a wave; its keys load before prepare(), hiding under the index q's
    if total <= BLOCKS:
        for u in range(slot, total, BLOCKS):
            tk, b = task_block(fx.Int32(u))
            page = uniform(page_of(tk, b))
            if new_keys_in(tk, b):
                wait_new_keys()
            raw = load_raw(page, [wave])
            prepare(tk - (q_len - 1), tk)
            col, col_seq = cols_of(tk)
            qb = [q_frag(col, s) for s in range(HEAD_DIM // 32)]
            v = tiles_max(b, qb, col_seq, [wave], split_tiles(raw, [wave]))
            if lane < 16:
                fx.ptr_store(v, red + (wave * 16 + l16))
            gpu.barrier()
            if wave == 0:
                for w in range_constexpr(1, WAVES):
                    v = fx.max(v, fx.ptr_load(red + (w * 16 + l16)))
                put(b, col, v)
            gpu.barrier()
    # more: a task a wave, after every row's index q (wave task counts differ: no barrier inside)
    if total > BLOCKS:
        prepare(fx.Int32(0), fx.Int32(tokens - 1))
        tiles = range(KEY_TILES)
        seq_lane = _pick(seq, fx.min(lane, tokens - 1))  # lane k: token k's context

        def score_task(tk, b, page):
            """Block b of the request ending at token tk, at ``page``."""
            if (b + 1) * SPARSE_BLOCK > readlane(seq_lane, tk) - MAX_TOKENS:
                wait_new_keys()
            raw = load_raw(page, tiles)
            col = tk - (q_len - 1) + fx.min(l16, q_len - 1)
            col_seq = lane_gather(seq_lane, col)
            qb = [q_frag(col, s) for s in range(HEAD_DIM // 32)]
            put(b, col, tiles_max(b, qb, col_seq, tiles, split_tiles(raw, tiles)))

        # lane j decodes task c0 + step j once, read back per task: no select chain between tasks
        step = BLOCKS * WAVES
        for c0 in range(slot + BLOCKS * wave, total, step * 64):
            c0 = fx.Int32(c0)
            tk_j, b_j = task_block(fx.min(c0 + step * lane, total - 1))
            pages = page_of(tk_j, b_j)
            for i in range(fx.min((total - c0 + step - 1) // step, 64)):
                i = fx.Int32(i)
                score_task(readlane(tk_j, i), readlane(b_j, i), readlane(pages, i))
    if const_expr(cp):  # noqa: SIM102
        # cold: a step without context-parallel requests runs none of it
        if unlikely(total_cp > 0):
            emit_cp_tasks()
