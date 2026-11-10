# SPDX-License-Identifier: Apache-2.0
# AKA: vendored from sgl-amd-mono @ 52486703 (sgl_amd_mono/muse_spark/gfx950/stages/attention.py) with
# package-relative imports, RING_WAIT at its default (no environment read) and one correctness fix of
# the chunk loop's mask decision (``inner``, marked "AKA fix" below).
"""K-attn: one Muse Spark layer's attention over the fp8 KV cache, for S =
``tokens`` rows of R = S / QL requests, ``QL`` consecutive rows a request (a
DFlash verify's 8 draft tokens, or one for decode). Per request and KV head, the
rank's 8 query heads of each of its QL rows attend the request's keys, as the
ROCm model's verify kernels (``swa_verify_splitkv``, ``gqa_verify_splitkv_2kv``)
and aiter's unified attention compute them::

    S = scale log2(e) (q . k)   q bf16 (K1 already applied the query scale),
                                K E4M3 upcast to bf16, fp32
    masked: key < context + qpos + 1 (causal), context + qpos - key < window
    online softmax in the exp2 domain, P -> bf16, O += P V (V upcast)
    O / l -> bf16

Tasks. An attention task is one (request, KV head, ``nt`` row tiles, partition
of ``part`` keys; the partitions a sliding window masks whole are skipped): a
chunk's K / V loads and its V^T staging serve the task's tiles (QL = 8 is 64
rows = four tiles; two a task fit the registers, four spill). Each is a loop over 256-key chunks
with the waves splitting the keys: QK on wave-local key tiles, the row
statistics, P through LDS, V transposed through LDS, PV; the waves' key splits
merge in LDS as a tree (8 -> 4 -> 2 -> 1); the partial (m, l, O) a row goes out
as plain fp32 words with one flag a task. The partitions are sized to at most
``MAX_PARTS`` a request from the longest request's span (``part_in`` overrides),
so one 95k-key request's full layer is 128 tasks of three chunks a KV head and a
sliding layer eight tasks of one; the partition grows with the step's total
chunk work so a step stays near two equal tasks a CTA whatever the request mix.

Merge. A (request, KV head, row tile) with at most ``DIRECT_MAX`` partitions
folds them in one job; a longer one in two levels: ``MERGE_GROUPS`` level-1 jobs
each fold a contiguous eighth of the partitions (a thread a (row, 8 dims,
partition lane), the lanes combined in LDS) into a level-2 partial, and the
level-2 job folds those. The merge jobs sit on the last CTAs (the task loop's
extra tasks fall on the first); a level-2 job's CTA runs its level-1 job first,
so every wait points to work that is already running.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from . import buffer_ops as bo
from flydsl._mlir.dialects import rocdl as rocdl_raw
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from .ops import (
    CM_DEV,
    ballot,
    fp8_to_bf16x8,
    hw_exp2,
    mfma_bf16,
    permlane_swap,
    rsrc,
)
from .plan import BLOCKS, THREADS, WAVES
from .config import HEAD_DIM, LOCAL_KV_HEADS, LOG2E, Q_K
from .device import (
    lds_dma_16,
    poll_flags,
    release_stores,
    stamp,
)
from .layout import ATTN_CHUNK as CHUNK
from .layout import ATTN_IDS_CHUNKS as IDS_CHUNKS
from .layout import ATTN_KEY_SPLITS as KEY_SPLITS
from .layout import ATTN_KEY_TILES as KEY_TILES
from .layout import ATTN_KEYS_PER_WAVE as KEYS_PER_WAVE
from .layout import ATTN_MAX_PARTS as MAX_PARTS
from .layout import ATTN_MERGE_GROUPS as GROUPS
from .layout import ATTN_PART_ROW as PART_ROW
from .layout import ATTN_Q_PITCH as Q_PITCH
from .layout import ATTN_RING_DEPTH as RING_DEPTH
from .layout import ATTN_V_STEPS as V_STEPS
from .layout import (
    GQA,
)

TILE = 16  # keys an MFMA tile
NEG = -1.0e30  # a masked score, and "no key yet" of a row's running max
NO_MAX = -1.0e29  # a running max below this is "no key yet": exp2 then takes 0
O_OFF = 4  # a partial row: m, l, pad, pad, O[64] (O 16 B aligned)
COMB_ROW = PART_ROW  # a tile row's (m, l, pad, pad, O[64]) in the LDS tree
# the ring's wait at a chunk's head: chunk cc + 1's four DMAs may stay in flight
# (SGLANG_AMD_MONO_ATTN_RING_WAIT0=1: wait for both chunks, a diagnostic)
RING_WAIT = 4  # AKA: the default of SGLANG_AMD_MONO_ATTN_RING_WAIT0=0 (no environment access)
MERGE_LANES = 4  # partitions folded side by side a (row, 8 dims): 16 x 8 x 4 = THREADS
PAGE_WORDS = (
    16 * HEAD_DIM // 4
)  # a 16-key fp8 page of one head in LDS: 1 KB, chunk-major (dims 16 c .., key)
V_SKEW = 16  # words between the two V pages beyond a page: the transposed reads of both hit distinct banks
SLOT_WORDS = (
    2 * KEY_TILES * PAGE_WORDS + V_SKEW
)  # a ring slot: a wave's chunk, its K pages then its V pages
TAU = 8.0  # a row's reference max moves only when a chunk's max exceeds it by this (log2 units): P <= 256
DIRECT_MAX = (
    16  # partitions a level-2 job folds itself (four a lane); more go through level 1
)


def comb_floats(nt: int) -> int:
    """The key-split merge's LDS floats: every wave's state of ``nt`` tiles."""
    return WAVES * nt * 16 * COMB_ROW


def kattn_lds(nt: int):
    """K-attn's LDS as two aliasing views (a union of them, built by the launch): a
    task's Q tiles and the merge jobs' reduction, then either each wave's P tiles
    and V^T (the chunk loop) or the key splits' merge tree (after it)."""

    @fx.struct
    class Loop:
        qs: fx.Array[
            fx.BFloat16, nt * 16 * Q_PITCH, 16
        ]  # the task's Q rows (its row tiles)
        red: fx.Array[fx.Float32, THREADS * 10, 16]  # the merge jobs' lane combine
        ring: fx.Array[
            fx.Int32, WAVES * RING_DEPTH * SLOT_WORDS, 16
        ]  # each wave's K / V chunks (DMA ring)
        ids: fx.Array[
            fx.Int32, IDS_CHUNKS * 16, 16
        ]  # the task's page ids, a block of chunks

    @fx.struct
    class Tree:
        qs: fx.Array[fx.BFloat16, nt * 16 * Q_PITCH, 16]
        red: fx.Array[fx.Float32, THREADS * 10, 16]
        comb: fx.Array[fx.Float32, comb_floats(nt), 16]

    return Loop, Tree


def lds_views(loop, tree):
    """``c`` entries of the two ``kattn_lds`` views (``.peek()`` of union members)."""
    return {"qs": loop.qs.ptr, "red": loop.red.ptr, "ring": loop.ring.ptr, "ids": loop.ids.ptr, "comb": tree.comb.ptr}  # fmt: skip


@flyc.jit
def run_kattn(c, o_flag=None):
    """K-attn's work on this CTA for ``c["S"]`` rows, ``c["QL"]`` a request.
    ``o_flag`` (a mailbox region): each level-2 merge raises its (request, KV
    head, row tile)'s flag once its o rows are out (a launch that goes on to K2)."""
    S, QL = c["S"], c["QL"]
    tid, bid, lane, wave = c["tid"], c["bid"], c["lane"], c["wave"]
    put, poll = c["put"], c["poll"]
    part_w, pflag, l2_w, l2flag = c["part"], c["pflag"], c["l2"], c["l2flag"]
    a = c["args"]
    R = S // QL
    ROWS = QL * GQA  # rows a (request, KV head): 64 or 8
    ROW_TILES = (ROWS + 15) // 16  # 4 or 1
    NT = c["NT"]  # row tiles a task (the build's; 4 spills registers, 2 fits)
    TASK_TILES = ROW_TILES // NT
    l16 = lane % 16
    g4 = lane // 16
    V8B = fx.Vector.make_type(8, fx.BFloat16)
    V2I = fx.Vector.make_type(2, fx.Int32)
    V4F = fx.Vector.make_type(4, fx.Float32)
    qs, ring, ids, comb, red = c["qs"], c["ring"], c["ids"], c["comb"], c["red"]
    wring = ring + wave * (RING_DEPTH * SLOT_WORDS)  # this wave's K / V ring
    table_stride, extra, window, part_in = (
        a["table_stride"],
        a["extra"],
        a["window"],
        a["part_in"],
    )
    lens64 = a["lens64"]
    qk_scale = a["scale_bits"].bitcast(fx.Float32) * fx.Float32(LOG2E)
    r_k = rsrc(a["k_cache"])
    r_v = rsrc(a["v_cache"])
    r_pt = rsrc(a["page_table"], a["table_bytes"])
    r_q = rsrc(a["q"])
    r_part = rsrc(part_w.value)
    r_l2 = rsrc(l2_w.value)
    stamp(c, 0)

    def seq_len_of(r):
        """Request r's keys (int32, or the low word of its int64), the step's rows included."""
        return (
            fx.Int32(
                bo.buffer_load(
                    rsrc(a["seq_lens"]),
                    (lens64 != 0).select(2 * r, r),
                    vec_width=1,
                    dtype=T.i32,
                )
            )
            + extra
        )

    def chunk_range(keys, qpos_lo):
        """The chunks [lo, hi) of a request with ``keys`` keys a window from its
        first row's position leaves unmasked (every chunk of a full one)."""
        hi = (keys + CHUNK - 1) // CHUNK
        ctx = keys - QL
        first = fx.max(ctx + qpos_lo - window + 1, fx.Int32(0))
        lo = (window > 0).select(first // CHUNK, fx.Int32(0))
        return lo, hi

    # the partitions: a request's keys from its first unmasked chunk (a sliding
    # window's start; 0 for a full layer), the longest such span over at most
    # MAX_PARTS of them
    def span_of(r):
        lo_r, hi_r = chunk_range(seq_len_of(r), 0)
        return hi_r - lo_r

    max_span = span_of(fx.Int32(0))
    total_span = max_span
    for rr in range_constexpr(1, R):
        sp = span_of(fx.Int32(rr))
        max_span = fx.max(max_span, sp)
        total_span = total_span + sp

    # the partition (chunks a task) from the step's total chunk work, so that the
    # step has one task a CTA whatever the request mix (a task costs ~12 us beside
    # its chunks of ~1.8 us: its Q and page ids, the first chunk's exposed
    # latency, the key-split tree and the publish; eight 95k requests at 128
    # partitions each were 15 tasks a CTA, 212 us), the per-request rounding
    # checked so that no CTA gets a second task, and at least what keeps the
    # longest request within MAX_PARTS
    def tasks_of(pc_):
        n = fx.Int32(0)
        for rr in range_constexpr(R):
            lo_r, hi_r = chunk_range(seq_len_of(fx.Int32(rr)), 0)
            n = n + ((hi_r - lo_r) + pc_ - 1) // pc_
        return n * (LOCAL_KV_HEADS * TASK_TILES)

    per_cta = (total_span * (LOCAL_KV_HEADS * TASK_TILES) + BLOCKS - 1) // BLOCKS
    floor_ = (max_span + MAX_PARTS - 1) // MAX_PARTS
    pc = fx.max(fx.max(per_cta, floor_), fx.Int32(1))
    for _ in range_constexpr(3):
        pc = (tasks_of(pc) > BLOCKS).select(pc + 1, pc)
    part = (part_in > 0).select(part_in, pc * CHUNK)
    n_parts = (
        max_span * CHUNK + part - 1
    ) // part  # partitions a (request, head) at most

    def n_parts_of(r):
        """Request r's partitions (its span's, the rest of the task slots idle)."""
        lo, hi = chunk_range(seq_len_of(r), 0)
        return ((hi - lo) * CHUNK + part - 1) // part

    def pflag_at(r, h, rt, p):
        """Pair index of the flag of the task holding row tile ``rt`` of (request r,
        head h) over partition p: a task's own (its first tile's), not the
        partition's: the tasks of a partition's row tiles finish apart (a merge
        that took the first one's flag read the other tiles' rows of the last layer)."""
        return ((r * LOCAL_KV_HEADS + h) * ROW_TILES + (rt // NT) * NT) * MAX_PARTS + p

    def part_row(r, h, p, row):
        """Float index of (request r, head h, partition p)'s row record."""
        return (((r * LOCAL_KV_HEADS + h) * MAX_PARTS + p) * ROWS + row) * PART_ROW

    def l2_row(r, h, rt, g, row16):
        """Float index of (request r, head h, row tile rt, group g)'s level-2 row record."""
        return (
            (((r * LOCAL_KV_HEADS + h) * ROW_TILES + rt) * GROUPS + g) * 16 + row16
        ) * PART_ROW

    def fold(m, l, ov, m2, l2, v2, ok):
        """(m, l, O) with a partial folded in (``ok``: it exists)."""
        mn = fx.max(m, m2)
        mg = (mn > fx.Float32(NO_MAX)).select(mn, fx.Float32(0.0))
        a1 = ok.select(hw_exp2(m - mg), fx.Float32(1.0))
        a2 = ok.select(hw_exp2(m2 - mg), fx.Float32(0.0))
        return ok.select(mn, m), l * a1 + l2 * a2, ov * a1 + v2 * a2

    # ============================================ attention tasks
    def attn_task(r, h, rt0, p):
        """Row tiles rt0 .. rt0 + NT of (request r, KV head h) over partition p: one
        K / V load and one V^T staging a chunk serve every tile of the task."""
        keys = seq_len_of(r)
        ctx = keys - QL
        ks = wave  # this wave's key split
        # Q: rows 16 (rt0 + it) + i = 8 qpos + hg of (request r, head h) into LDS, 16 B a thread
        if tid < NT * 16 * HEAD_DIM // 8:
            i_t = tid // (16 * HEAD_DIM // 8)
            q_row = (tid % (16 * HEAD_DIM // 8)) // (HEAD_DIM // 8)
            row = (rt0 + i_t) * 16 + q_row
            d0 = (tid % (HEAD_DIM // 8)) * 8
            t = r * QL + fx.min(row, ROWS - 1) // GQA
            qv = fx.Vector(
                bo.buffer_load(
                    r_q,
                    (t * Q_K + (h * GQA + fx.min(row, ROWS - 1) % GQA) * HEAD_DIM + d0)
                    // 2,
                    vec_width=4,
                    dtype=T.i32,
                )  # fmt: skip
            )
            fx.ptr_store(
                qv.bitcast(fx.BFloat16), qs + ((i_t * 16 + q_row) * Q_PITCH + d0)
            )
        gpu.barrier()
        # each tile's A operands of this lane: row l16, dims 8 g4 + 32 step
        qa = [
            [
                fx.Vector(
                    fx.ptr_load(
                        qs + ((it * 16 + l16) * Q_PITCH + 32 * step + 8 * g4),
                        result_type=V8B,
                    )
                )  # fmt: skip
                for step in range(2)
            ]
            for it in range(NT)
        ]
        # S^T tiles: K is the MFMA's A operand (key l16, dims 8 g4 ..), Q^T its B
        # (row l16): a lane holds row l16 of each row tile and keys 4 g4 .. 4 g4 + 3 of
        # each key tile, so a row's running statistics are one lane's (four lanes
        # share a row), and its P values are packed in registers straight into the
        # PV MFMA's A operand (keys of tile 0 then tile 1; the V^T reads use the
        # same key order). The reference max of a row moves only when a chunk's max
        # exceeds it by TAU (lazy rescale, every row of the wave at once).
        r_lane = [(rt0 + it) * 16 + l16 for it in range(NT)]
        qpos = [fx.min(fx.min(r_lane[it], ROWS - 1) // GQA, QL - 1) for it in range(NT)]
        row_live = [r_lane[it] < ROWS for it in range(NT)]
        m_ref = [
            fx.Float32(NEG) for _ in range(NT)
        ]  # the row's reference max (its running max, lazily)
        l_lane = [
            fx.Float32(0.0) for _ in range(NT)
        ]  # this lane's keys' share of the row sum
        accs = [
            [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(4)] for _ in range(NT)
        ]
        lo, hi = chunk_range(keys, 0)
        c_lo = fx.max(lo, lo + p * (part // CHUNK))
        c_hi = fx.min(hi, lo + (p + 1) * (part // CHUNK))

        # The chunk loop: a wave's 32 keys of a chunk are two K pages and two V pages
        # (1 KB each, one head), landed in LDS by DMA (16 B a lane; chunk-major: dims
        # 16 c .. of key k at bytes 256 c + 16 k, conflict-free for both the K reads
        # and the transposed V reads) in the wave's ring slot; two chunks are in
        # flight while one computes. (A VGPR-destination software pipeline was
        # drained by the compiler at every loop head: 6 us a chunk at 300k keys.)
        # The page ids of a block of IDS_CHUNKS chunks are staged in LDS first.
        assert KEY_TILES == 2 and V_STEPS == 1
        c_last = fx.max(c_hi - 1, c_lo)
        l16d, c16d = (
            lane % 16,
            lane // 16,
        )  # the DMA lane: key l16d, dims chunk c16d of a page
        # the transposed V read: this lane addresses key 4 g4 + (l16 / 2) % 4 of page (l16 / 2) / 4
        v_page = (l16 // 2) // 4
        v_key = 4 * g4 + (l16 // 2) % 4

        def issue(cc, sl, b_lo):
            """Chunk cc's four pages -> ring slot sl (always four loads: the counts)."""
            slot = wring + sl * SLOT_WORDS
            for kt in range_constexpr(KEY_TILES):
                pg = fx.ptr_load(ids + ((cc - b_lo) * 16 + 2 * ks + kt))  # uniform
                voff = ((pg * 16 + l16d) * LOCAL_KV_HEADS + h) * HEAD_DIM + c16d * 16
                lds_dma_16(r_k, slot + kt * PAGE_WORDS, voff)
                lds_dma_16(
                    r_v, slot + ((KEY_TILES + kt) * PAGE_WORDS + kt * V_SKEW), voff
                )

        def quad_max(v):
            """The max over the four lanes sharing a row (l16, l16 + 16, + 32, + 48)."""
            for off in (32, 16):
                x, y = permlane_swap(off, v.bitcast(fx.Int32), v.bitcast(fx.Int32))
                v = fx.max(x.bitcast(fx.Float32), y.bitcast(fx.Float32))
            return v

        def quad_sum(v):
            for off in (32, 16):
                x, y = permlane_swap(off, v.bitcast(fx.Int32), v.bitcast(fx.Int32))
                v = x.bitcast(fx.Float32) + y.bitcast(fx.Float32)
            return v

        n_blk = (c_hi - c_lo + IDS_CHUNKS - 1) // IDS_CHUNKS
        stamp(c, 3)  # a task's chunk loop begins (the last task's stamps survive)
        for b in range(0, n_blk):
            b_lo = c_lo + b * IDS_CHUNKS
            b_hi = fx.min(b_lo + IDS_CHUNKS, c_hi)
            b_last = b_hi - 1
            gpu.barrier()  # the previous block's ids are consumed
            for i in range_constexpr(IDS_CHUNKS * 16 // THREADS):
                idx = tid + THREADS * i
                cc_i = fx.min(b_lo + idx // 16, c_last)
                pg = fx.Int32(
                    bo.buffer_load(
                        r_pt,
                        r * table_stride + cc_i * 16 + idx % 16,
                        vec_width=1,
                        dtype=T.i32,
                    )
                )
                fx.ptr_store(pg, ids + idx)
            gpu.barrier()
            issue(b_lo, 0, b_lo)
            issue(fx.min(b_lo + 1, b_last), 1, b_lo)
            for cc in range(b_lo, b_hi):
                sl = (cc - b_lo) % 2
                slot = wring + sl * SLOT_WORDS
                key0 = cc * CHUNK + ks * KEYS_PER_WAVE
                rocdl.s_waitcnt(
                    vmcnt=RING_WAIT
                )  # chunk cc's pages landed (cc + 1's in flight)
                # the chunk's K operands (tile kt, step: key l16, dims 32 step + 8 g4)
                kb = []
                for kt in range_constexpr(KEY_TILES):
                    for step in range_constexpr(2):
                        widx = (
                            slot
                            + kt * PAGE_WORDS
                            + ((2 * step + g4 // 2) * 256 + l16 * 16 + (g4 % 2) * 8)
                            // 4
                        )
                        w = fx.Vector(fx.ptr_load(widx, result_type=V2I))
                        kb.append(fp8_to_bf16x8(w[0], w[1]))
                # ---- S^T: tile (it, kt): row l16, keys 4 g4 + e of key tile kt; scaled and masked
                # AKA fix: a chunk with no masked key. Upstream took every chunk but the task's last
                # and the request's first as unmasked, but a verify step's causally masked keys (the
                # step's later rows) reach into the chunk before the last when keys % CHUNK is 1 .. QL - 2, and
                # when both chunks fall in one partition the earlier rows saw those keys. A chunk is
                # now unmasked only if it also ends at or before the step's first row's last key.
                inner = (cc != c_last) & (cc != lo) & ((cc + 1) * CHUNK <= ctx + 1)
                sc = []  # [it][kt * 4 + e]
                mx = []
                for it in range_constexpr(NT):
                    vals = []
                    for kt in range_constexpr(KEY_TILES):
                        st = fx.Vector.filled(4, 0.0, fx.Float32)
                        for step in range_constexpr(2):
                            st = mfma_bf16(kb[kt * 2 + step], qa[it][step], st)
                        for e in range_constexpr(4):
                            kidx = key0 + kt * TILE + 4 * g4 + e
                            ok = row_live[it] & (inner | ((kidx < ctx + qpos[it] + 1) & ((window == 0) | ((ctx + qpos[it] - kidx) < window))))  # fmt: skip
                            vals.append(
                                ok.select(fx.Float32(st[e]) * qk_scale, fx.Float32(NEG))
                            )
                    sc.append(vals)
                    m8 = vals[0]
                    for j in range_constexpr(1, 2 * KEY_TILES * 4 // 2):
                        m8 = fx.max(m8, vals[j])
                    mx.append(quad_max(m8))
                # ---- the lazy rescale: when any row's chunk max exceeds its reference by TAU,
                # every row takes its new max and its sum and O are rescaled (selects:
                # the step is wave-uniform, the values are loop-carried)
                need = mx[0] > m_ref[0] + TAU
                for it in range_constexpr(1, NT):
                    need = need | (mx[it] > m_ref[it] + TAU)
                move = ballot(need) != 0
                alphas_acc = []
                for it in range_constexpr(NT):
                    m_new = move.select(fx.max(m_ref[it], mx[it]), m_ref[it])
                    m_g = (m_new > fx.Float32(NO_MAX)).select(m_new, fx.Float32(0.0))
                    alpha = hw_exp2(
                        m_ref[it] - m_g
                    )  # 0 from NEG (a first chunk), 1 when nothing moved
                    l_lane[it] = l_lane[it] * alpha
                    m_ref[it] = m_new
                    # O rows 4 g4 + e take the alpha of row 4 g4 + e: from lane 4 g4 + e (its g4 = 0 copy)
                    a_rows = [
                        fx.Int32(
                            rocdl.ds_bpermute(
                                T.i32, (4 * g4 + e) * 4, alpha.bitcast(fx.Int32)
                            )
                        ).bitcast(fx.Float32)
                        for e in range(4)
                    ]
                    al = fx.Vector.from_elements(a_rows, fx.Float32)
                    accs[it] = [accs[it][dt] * al for dt in range(4)]
                # ---- P = exp2(S - m) as the PV A operand (eight bf16 a lane), the lane's row sums
                p_ops = []
                for it in range_constexpr(NT):
                    m_g = (m_ref[it] > fx.Float32(NO_MAX)).select(
                        m_ref[it], fx.Float32(0.0)
                    )
                    ps = [
                        hw_exp2(sc[it][j] - m_g) for j in range(2 * KEY_TILES * 4 // 2)
                    ]
                    acc_l = ps[0]
                    for j in range_constexpr(1, len(ps)):
                        acc_l = acc_l + ps[j]
                    l_lane[it] = l_lane[it] + acc_l
                    p_ops.append(
                        fx.Vector.from_elements(ps, fx.Float32).to(fx.BFloat16)
                    )
                # ---- PV: V^T operands by transposed LDS reads (dim tile dt: lane (g4, l16) gets
                # dim 16 dt + l16 at keys 4 g4 .. of page 0 then of page 1)
                b_v = []
                for dt in range_constexpr(4):
                    widx = slot + ((KEY_TILES + v_page) * PAGE_WORDS + v_page * V_SKEW) + (dt * 256 + v_key * 16 + (l16 % 2) * 8) // 4  # fmt: skip
                    w = fx.Vector(
                        rocdl_raw.ds_read_tr8_b64(
                            T.vec(2, T.i32), widx.llvm_ptr
                        ).result,
                        (2,),
                        fx.Int32,
                    )  # gfx950 ds_read_b64_tr_b8
                    b_v.append(fp8_to_bf16x8(w[0], w[1]))
                accs = [
                    [mfma_bf16(p_ops[it], b_v[dt], accs[it][dt]) for dt in range(4)]
                    for it in range(NT)
                ]
                rocdl.s_waitcnt(lgkmcnt=0)  # every read of the slot is done
                issue(fx.min(cc + 2, b_last), sl, b_lo)  # chunk cc + 2 into this slot
                if cc == b_lo:
                    stamp(c, 4)  # the first chunk done
            rocdl.s_waitcnt(
                vmcnt=0
            )  # the tail's repeated loads land before the LDS is reused
        stamp(c, 5)  # the chunk loop done
        gpu.barrier()  # every wave is done with the ring: the merge tree reuses the LDS

        # ---- the key splits merge in LDS in one round: every wave writes its
        # (m, l, O) rows (a (split, tile, row) record is COMB_ROW floats, O from
        # O_OFF); then each wave owns 2 NT of the task's rows, folds the eight splits
        # of them (a lane: one row, DPL dims) and publishes them with vector stores
        def comb_at(slot, it, row):
            return comb + ((slot * NT + it) * 16 + row) * COMB_ROW

        for it in range_constexpr(NT):
            l_row = quad_sum(l_lane[it])
            if g4 == 0:
                base = comb_at(ks, it, l16)
                fx.ptr_store(m_ref[it], base)
                fx.ptr_store(l_row, base + 1)
            for e in range_constexpr(4):
                base = comb_at(ks, it, 4 * g4 + e)
                for dt in range_constexpr(4):
                    fx.ptr_store(
                        fx.Float32(accs[it][dt][e]), base + (O_OFF + dt * 16 + l16)
                    )
        gpu.barrier()
        RPW = 2 * NT  # rows a wave: NT 16 / 8
        LPR = 64 // RPW  # lanes a row: 16 or 32
        DPL = HEAD_DIM // LPR  # dims a lane: 4 or 2
        VD = fx.Vector.make_type(DPL, fx.Float32)
        trow = wave * RPW + lane // LPR  # the task row (0 .. NT 16)
        it_f, row_f = trow // 16, trow % 16
        dd = (lane % LPR) * DPL
        m_f = fx.Float32(NEG)
        l_f = fx.Float32(0.0)
        o_f = fx.Vector.filled(DPL, 0.0, fx.Float32)
        for src in range_constexpr(KEY_SPLITS):
            base = comb_at(src, it_f, row_f)
            m2 = fx.ptr_load(base)
            l2 = fx.ptr_load(base + 1)
            v2 = fx.Vector(fx.ptr_load(base + (O_OFF + dd), result_type=VD))
            mn = fx.max(m_f, m2)
            mg = (mn > fx.Float32(NO_MAX)).select(mn, fx.Float32(0.0))
            a1 = hw_exp2(m_f - mg)
            a2 = hw_exp2(m2 - mg)
            m_f, l_f, o_f = mn, l_f * a1 + l2 * a2, o_f * a1 + v2 * a2
        # ---- the task's partial rows: plain fp32 words, then the flag
        row = rt0 * 16 + trow
        if row < ROWS:
            base = part_row(r, h, p, row)
            if lane % LPR == 0:
                bo.buffer_store(
                    fx.Vector.from_elements([m_f, l_f], fx.Float32),
                    r_part,
                    base,
                    cache_modifier=CM_DEV,
                )
            bo.buffer_store(o_f, r_part, base + O_OFF + dd, cache_modifier=CM_DEV)
        release_stores()
        gpu.barrier()
        if tid == 0:
            put(pflag, pflag_at(r, h, rt0, p), fx.Int32(1))

    # the tasks, request-major over the requests' own partition counts (a short
    # request has few): job -> (request, head, tile pair, partition) through the
    # requests' cumulative task counts, so a CTA iterates real tasks only
    counts = [
        n_parts_of(fx.Int32(rr)) * (LOCAL_KV_HEADS * TASK_TILES) for rr in range(R)
    ]
    starts = [fx.Int32(0)]
    for rr in range_constexpr(1, R):
        starts.append(starts[-1] + counts[rr - 1])
    total = starts[-1] + counts[-1]
    for job in range(bid, total, BLOCKS):
        r = fx.Int32(0)
        for rr in range_constexpr(1, R):
            r = (job >= starts[rr]).select(fx.Int32(rr), r)
        base = starts[0]
        for rr in range_constexpr(1, R):
            base = (r == rr).select(starts[rr], base)
        np_r = n_parts_of(r)
        local = job - base
        # a partition's TASK_TILES tasks (its row tiles' groups) sit 8 jobs apart, on
        # CTAs of one XCD (CTAs go to the XCDs in turn): they stream the same K / V
        # pages at about the same time, so the later reads hit that XCD's L2 (before,
        # the tasks were np_r apart, on different XCDs, each reading the pages from
        # HBM). Groups of 8 partitions x TASK_TILES jobs; the last np_r % 8 partitions
        # in a tail group.
        j = local % (TASK_TILES * np_r)
        h = (local // (TASK_TILES * np_r)) % LOCAL_KV_HEADS
        full = (np_r // 8) * 8
        g, w = j // (8 * TASK_TILES), j % (8 * TASK_TILES)
        jt = j - TASK_TILES * full
        tail = fx.max(np_r - full, fx.Int32(1))
        in_full = j < TASK_TILES * full
        p = in_full.select(g * 8 + w % 8, full + jt % tail)
        rt0 = in_full.select(w // 8, jt // tail) * NT
        attn_task(r, h, rt0, p)
    stamp(c, 1)  # the attention tasks done

    # ============================================ merge jobs
    # a thread: row 16 rt + tid / 32, dims 8 ((tid / 4) % 8) .., partition lane tid % 4
    row16 = tid // 32
    dch = (tid // 4) % 8
    pl = tid % 4
    d0 = dch * 8
    mbid = (
        BLOCKS - 1 - bid
    )  # the last CTAs: the task loop's extra tasks sit on the first
    n_l1 = R * LOCAL_KV_HEADS * ROW_TILES * GROUPS
    n_l2 = R * LOCAL_KV_HEADS * ROW_TILES

    def load_rec(r_src, base):
        """A record's (m, l) and this thread's 8 dims of O (plain words)."""
        ml = fx.Vector(
            bo.buffer_load(r_src, base, vec_width=2, dtype=T.f32, cache_modifier=CM_DEV)
        )
        o0 = fx.Vector(
            bo.buffer_load(
                r_src,
                base + O_OFF + d0,
                vec_width=4,
                dtype=T.f32,
                cache_modifier=CM_DEV,
            )
        )
        o1 = fx.Vector(
            bo.buffer_load(
                r_src,
                base + O_OFF + d0 + 4,
                vec_width=4,
                dtype=T.f32,
                cache_modifier=CM_DEV,
            )
        )
        return ml, o0, o1

    def fold_rec(m, l, o0, o1, rec, ok):
        ml, p0, p1 = rec
        m2, l2 = fx.Float32(ml[0]), fx.Float32(ml[1])
        mn = fx.max(m, m2)
        mg = (mn > fx.Float32(NO_MAX)).select(mn, fx.Float32(0.0))
        a1 = ok.select(hw_exp2(m - mg), fx.Float32(1.0))
        a2 = ok.select(hw_exp2(m2 - mg), fx.Float32(0.0))
        return ok.select(mn, m), l * a1 + l2 * a2, o0 * a1 + p0 * a2, o1 * a1 + p1 * a2

    def combine_lanes(m, l, o0, o1):
        """The four partition lanes of a (row, 8 dims) fold in LDS: every lane folds
        the four in lane order (the same result on each; lane 0 stores)."""
        slot = red + tid * 10
        fx.ptr_store(m, slot)
        fx.ptr_store(l, slot + 1)
        for j in range_constexpr(4):
            fx.ptr_store(fx.Float32(o0[j]), slot + (2 + j))
            fx.ptr_store(fx.Float32(o1[j]), slot + (6 + j))
        gpu.barrier()
        base = red + (tid - pl) * 10
        m = fx.Float32(NEG)
        l = fx.Float32(0.0)
        o0 = fx.Vector.filled(4, 0.0, fx.Float32)
        o1 = fx.Vector.filled(4, 0.0, fx.Float32)
        for src in range_constexpr(MERGE_LANES):
            s2 = base + src * 10
            m2 = fx.ptr_load(s2)
            l2 = fx.ptr_load(s2 + 1)
            p0 = fx.Vector.from_elements(
                [fx.ptr_load(s2 + (2 + j)) for j in range(4)], fx.Float32
            )
            p1 = fx.Vector.from_elements(
                [fx.ptr_load(s2 + (6 + j)) for j in range(4)], fx.Float32
            )
            mn = fx.max(m, m2)
            mg = (mn > fx.Float32(NO_MAX)).select(mn, fx.Float32(0.0))
            a1 = hw_exp2(m - mg)
            a2 = hw_exp2(m2 - mg)
            m, l, o0, o1 = mn, l * a1 + l2 * a2, o0 * a1 + p0 * a2, o1 * a1 + p1 * a2
        gpu.barrier()  # ``red`` is reused by the CTA's next job
        return m, l, o0, o1

    def merge_l1(r, h, rt, g):
        """Level-1 job: partitions [g PPG, (g + 1) PPG) of (request r, head h) for
        row tile rt, folded four lanes a (row, 8 dims) -> a level-2 record."""
        np_r = n_parts_of(r)
        ppg = (np_r + GROUPS - 1) // GROUPS
        p_lo = g * ppg
        cnt = fx.max(fx.min(np_r - p_lo, ppg), fx.Int32(0))
        poll_flags(c, [(pflag, pflag_at(r, h, rt, p_lo), cnt)])
        row = fx.min(rt * 16 + row16, ROWS - 1)
        m = fx.Float32(NEG)
        l = fx.Float32(0.0)
        o0 = fx.Vector.filled(4, 0.0, fx.Float32)
        o1 = fx.Vector.filled(4, 0.0, fx.Float32)
        # four partitions a lane in flight a round
        for k0 in range(0, (ppg + MERGE_LANES * 4 - 1) // (MERGE_LANES * 4)):
            ps = [p_lo + (k0 * 4 + k) * MERGE_LANES + pl for k in range(4)]
            recs = [
                load_rec(r_part, part_row(r, h, fx.min(pk, p_lo + cnt - 1), row))
                for pk in ps
            ]
            for k in range_constexpr(4):
                m, l, o0, o1 = fold_rec(m, l, o0, o1, recs[k], ps[k] < p_lo + cnt)
        m, l, o0, o1 = combine_lanes(m, l, o0, o1)
        if (pl == 0) & (rt * 16 + row16 < ROWS):
            base = l2_row(r, h, rt, g, row16)
            if dch == 0:
                bo.buffer_store(
                    fx.Vector.from_elements([m, l], fx.Float32),
                    r_l2,
                    base,
                    cache_modifier=CM_DEV,
                )
            bo.buffer_store(o0, r_l2, base + O_OFF + d0, cache_modifier=CM_DEV)
            bo.buffer_store(o1, r_l2, base + O_OFF + d0 + 4, cache_modifier=CM_DEV)
        release_stores()
        gpu.barrier()
        if tid == 0:
            put(
                l2flag,
                ((r * LOCAL_KV_HEADS + h) * ROW_TILES + rt) * GROUPS + g,
                fx.Int32(1),
            )

    def merge_l2(r, h, rt):
        """Level-2 job: the groups' records (or, with at most DIRECT_MAX partitions,
        the partitions themselves) of (request r, head h, row tile rt) -> o rows."""
        np_r = n_parts_of(r)
        direct = np_r <= DIRECT_MAX
        row = fx.min(rt * 16 + row16, ROWS - 1)
        m = fx.Float32(NEG)
        l = fx.Float32(0.0)
        o0 = fx.Vector.filled(4, 0.0, fx.Float32)
        o1 = fx.Vector.filled(4, 0.0, fx.Float32)
        # the source records: the partitions themselves (at most GROUPS of them), or
        # the level-1 records; one resource over the selected base address
        if direct:
            poll_flags(c, [(pflag, pflag_at(r, h, rt, 0), np_r)])
        else:
            poll_flags(
                c,
                [
                    (
                        l2flag,
                        ((r * LOCAL_KV_HEADS + h) * ROW_TILES + rt) * GROUPS,
                        GROUPS,
                    )
                ],
            )
        r_src = rsrc(direct.select(part_w.value, l2_w.value))
        for k in range_constexpr(DIRECT_MAX // MERGE_LANES):
            g = k * MERGE_LANES + pl  # a partition (direct) or a group (g < GROUPS)
            idx = direct.select(
                part_row(r, h, fx.min(g, np_r - 1), row),
                l2_row(r, h, rt, fx.min(g, GROUPS - 1), row16),
            )
            ok = direct.select(g < np_r, g < GROUPS)
            m, l, o0, o1 = fold_rec(m, l, o0, o1, load_rec(r_src, idx), ok)
        m, l, o0, o1 = combine_lanes(m, l, o0, o1)
        if (pl == 0) & (rt * 16 + row16 < ROWS):
            inv = fx.Float32(1.0) / l
            t = r * QL + (rt * 16 + row16) // GQA
            ov = fx.Vector.from_elements([fx.Float32(o0[j]) for j in range(4)] + [fx.Float32(o1[j]) for j in range(4)], fx.Float32)  # fmt: skip
            bo.buffer_store(
                (ov * inv).to(fx.BFloat16),
                rsrc(a["o"]),
                t * Q_K + (h * GQA + (rt * 16 + row16) % GQA) * HEAD_DIM + d0,
                cache_modifier=CM_DEV,
            )
        if const_expr(o_flag is not None):
            # the (request, head, row tile)'s o rows are out: its flag, for the og tasks
            release_stores()
            gpu.barrier()
            if tid == 0:
                put(o_flag, (r * LOCAL_KV_HEADS + h) * ROW_TILES + rt, fx.Int32(1))

    for j in range(mbid, n_l1, BLOCKS):
        g = j % GROUPS
        rt = (j // GROUPS) % ROW_TILES
        h = (j // (GROUPS * ROW_TILES)) % LOCAL_KV_HEADS
        r = j // (GROUPS * ROW_TILES * LOCAL_KV_HEADS)
        if (
            n_parts_of(r) > DIRECT_MAX
        ):  # else the level-2 job folds the partitions itself
            merge_l1(r, h, rt, g)
    if mbid < n_l2:
        rt = mbid % ROW_TILES
        h = (mbid // ROW_TILES) % LOCAL_KV_HEADS
        r = mbid // (ROW_TILES * LOCAL_KV_HEADS)
        merge_l2(r, h, rt)
    stamp(c, 2)  # the merge done (end)
