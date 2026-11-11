# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K4: one MiniMax-M3 sparse MoE layer, attention through FFN reduce.

One launch per layer and rank runs the layer's K1, then for each of S = 1..MAX_TOKENS
tokens the stages index (the unfused selector's top-k over K1's block scores, as a
page-16 table), split (the paged decode's 256-key partitions, FP8 QK / PV with its
scale placement), o (the paged decode's reduce, standalone per-token FP8, o_proj; the
rank-ordered sum s gives h_mid = bf16(bf16(s) + h) and mailbox ``a`` = s), router
(gemma RMSNorm in the fused all-reduce's reduction order, bf16 gate GEMV) and moe
(sigmoid + bias top-4 plus the fused shared expert, MXFP8 x MXFP4 on aiter's shuffled
layout, swiglu, route-weighted down, rank-ordered sum: ar_out = bf16(sum)). The next
layer's K1 takes (ar_out, h_mid) exactly as the unfused fused all-reduce + RMSNorm
takes (partials, residual). From ``WIDE_FROM`` tokens the MoE groups its work by
expert (``wide_moe``); from ``AG_RS_FROM`` both all-reduces reduce-scatter to the row
group's owner rank, which sums in rank order and all-gathers. Task t of a stage runs
on CTA (base + t) % BLOCKS, every CTA walks the stages in order and dependencies only
point to earlier stages, so with every CTA co-resident each spin wait makes progress.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from . import buffer_ops as bo
from .act import swiglu_mul_batch
from .act import LOG2E
from flydsl.expr import const_expr, gpu
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T, as_ir_value

from .compat import AtomicOrdering, memory_fence

from . import guard
from .plan import kernel_symbol
from .mx import scale_index
from .ops import (
    CM_DEV,
    bf2_f32,
    bf16_pair,
    bf16_round,
    butterfly,
    fp8_pack4,
    hw_exp2,
    hw_rcp,
    hw_rsq,
    rsrc,
    uniform,
    wave_max,
    xshfl,
)
from .plan import key_tuple
from .sync import preg
from .config import (
    BLOCKS,
    HEAD_DIM,
    HIDDEN,
    INDEX_CP_FROM_BLOCKS,
    INTER,
    MAX_INDEX_BLOCKS,
    MAX_TOKENS,
    MOE_SLOTS,
    N_ROUTED,
    ONE_INDEX_HEAD,
    PAGE16,
    SHARED_EXPERT,
    SPARSE_BLOCK,
    THREADS,
    TOP_K,
    TOPK_BLOCKS,
    TP,
    WAVES,
    IndexHeads,
)
from .device import StepMailbox as Mailbox
from .device import block_max as block_max_of
from .device import (
    release_stores,
    step_tag,
    unlikely,
    wave_sum,
)
from .index_score import index_scale_log2e, step_rows
from .layout import (
    AG_RS_FROM,
    DN_ROWS,
    MID_ROW,
    MIDSC_ROW,
    N_DN,
    N_O,
    N_ROUTER,
    N_SPLIT,
    O_K,
    O_ROW,
    O_ROWS,
    PAGES_PER_BLOCK,
    SCORE_COPIES,
    SCRATCH,
    SH_ROWS,
    SH_TASKS,
    SPLIT_KEYS,
    UG_PER_SLOT,
    UG_ROWS,
    XN8_ROW,
    XSC_ROW,
    H,
    pool_words,
    shared_after_routing,
    stage_bases,
    sym_layout,
    ug_ctas_per_token,
    ug_tasks_of,
    wide_moe,
)
from .pre_attn import N_HEAD_TASKS as K1_HEAD_TASKS
from .pre_attn import SCRATCH_HDONE as K1_SCRATCH_HDONE
from .pre_attn import SCRATCH_RDONE as K1_SCRATCH_RDONE
from .pre_attn import (
    emit_k1,
)
from .select import (
    KEY_LEAST,
    compact,
    count_beaten,
    group_floor,
    put_key,
)

FP8_MAX = 448.0
NEG = -3.4e38  # the paged decode's masked-score value
PAGE_BYTES = PAGE16 * HEAD_DIM
XN_SLICE = HIDDEN // N_ROUTER  # normalized-input elements each router task publishes
W13_ROWS = 2 * INTER  # gate rows then up rows (GGUU)
W13_BYTES = W13_ROWS * HIDDEN // 2  # one expert, fp4
W2_BYTES = HIDDEN * INTER // 2
W13_SCALE_COLS = HIDDEN // 32
W2_SCALE_COLS = INTER // 32
S2_BYTES = (N_ROUTED + 1) * HIDDEN * W2_SCALE_COLS  # e8m0, every expert
S13_BYTES = (N_ROUTED + 1) * W13_ROWS * W13_SCALE_COLS

# None: the caller's current stream, graph capture included
_CURRENT_STREAM = fx.Stream(None)


# blocks at or past the selection floor (ATOM's M3 select stage); more ties count every block
CAND_CAP = 256
# ``sel``: one put_key entry (two words) a thread; a long request's candidates reuse it
SEL_WORDS = 2 * THREADS
assert SEL_WORDS >= 2 * N_SPLIT * TOPK_BLOCKS
# no LDS spare at S = 16: the selection borrows ``p8`` / ``red`` between K1 and the split's P
SEL_P8_WORDS = 2 * CAND_CAP + 2 + WAVES * TOPK_BLOCKS
# blocks a thread keys in its split task's eighth of a context (<= MAX_INDEX_BLOCKS)
CAND_BATCH = 2
assert N_SPLIT * THREADS * CAND_BATCH >= MAX_INDEX_BLOCKS
# indexer CP: split task p scans head p // 2's scores of half p % 2 of this rank's blocks
assert N_SPLIT == 2 * TP
# a long request's rows end under a block apart, so a share holds >= TOPK_BLOCKS blocks
assert N_SPLIT * TOPK_BLOCKS <= INDEX_CP_FROM_BLOCKS <= THREADS
assert MAX_TOKENS <= SPARSE_BLOCK


def build_post_attn_kernel(
    npes: int,
    sm_scale: float,
    eps: float,
    route_scale: float,
    shared_weight: float,
    swiglu_limit: float,
    init_blocks: int,
    local_blocks: int,
    tokens: int = 1,
    heads: IndexHeads = ONE_INDEX_HEAD,
    fault_site: int = 0,
    fault_after: int = 0,
):
    """``@flyc.jit`` launcher of K4 for one rank of an ``npes``-way TP group.

    A step is ``tokens`` (<= MAX_TOKENS) rows, ``q_len`` consecutive rows a request (a
    speculative verify's), with contexts up to MAX_CONTEXT. The layer's K1 runs first in
    the same launch (its pointers ``ar_in`` .. ``scratch1``, ``positions``, ``slot_mapping``, ``res``) with a fused
    projection of index q ``heads``. The KV cache is separate page-16 SHUFFLE K and V
    planes with the layer's fixed K / V scales; the router weight is bf16."""
    # every build parameter (upstream: key_tuple(locals(), SOURCES), with a digest of the package's sources)
    keyed = key_tuple(dict(npes=npes, sm_scale=sm_scale, eps=eps, route_scale=route_scale,
                           shared_weight=shared_weight, swiglu_limit=swiglu_limit, init_blocks=init_blocks,
                           local_blocks=local_blocks, tokens=tokens, heads=heads, fault_site=fault_site,
                           fault_after=fault_after))
    assert 1 <= tokens <= MAX_TOKENS
    W = npes
    G = BLOCKS
    BASE = stage_bases(tokens)
    SY = sym_layout(W)
    # K1's outputs are read device-coherent (K1 stores them so)
    CM_K1 = CM_DEV
    WIDE = wide_moe(tokens)
    POOL = pool_words(tokens)
    U_MAX = TOP_K * tokens + 1  # distinct routed experts at most, then the shared

    @fx.struct
    class Smem:
        # bf16 pairs / fp8 quads of every token's xn, o_proj input or mid (wide: PV partials too)
        x: fx.Array[fx.Int32, POOL, 16]
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        p8: fx.Array[fx.Int32, H * SPLIT_KEYS // 4, 16]
        q8: fx.Array[fx.Int32, O_K // 4, 16]
        opart: fx.Array[fx.Float32, 1 if WIDE else WAVES * O_K, 16]  # PV partials
        # wide MoE: the step's distinct experts (the shared one last), token masks
        uexp: fx.Array[fx.Int32, U_MAX + 2, 16]
        umask: fx.Array[fx.Int32, U_MAX + 2, 16]
        # and per (expert, B column) the down stage's mid row and route weight
        urow: fx.Array[fx.Int32, U_MAX * 16 if WIDE else 1, 16]
        uwt: fx.Array[fx.Float32, U_MAX * 16 if WIDE else 1, 16]
        hst: fx.Array[
            fx.Float32, 3 * WAVES * H + WAVES, 16
        ]  # head max / sum, max, v_scale max
        misc: fx.Array[fx.Float32, O_ROWS * MAX_TOKENS, 16]  # a task's output rows
        route: fx.Array[fx.Int32, 8 * MAX_TOKENS, 16]  # token k's slots at 8 k
        rwt: fx.Array[fx.Float32, 8 * MAX_TOKENS, 16]  # and their routing weights
        # the indexer selection's entries and candidates (``SEL_WORDS``)
        sel: fx.Array[fx.Int32, SEL_WORDS, 16]
        blk: fx.Array[fx.Int32, 32, 16]  # this split's 16 pages, n_ctx, tail flag
        # K1's index q (every head's: in the pool)
        qs: fx.Array[fx.Float32, HEAD_DIM // 2 * tokens if heads.count == 1 else 1, 16]

    kernel_name = kernel_symbol(
        "minimax_m3_mono_layer",
        tp=npes,
        s=tokens,
        ib=init_blocks,
        lb=local_blocks,
        ih=heads.count,
        io=heads.own,
    )

    ih_count, ih_own = heads.count, heads.own
    # all heads' index q after K1's x8 rows in the pool; K4's stages use it only after K1
    QS_CP = tokens * HIDDEN // 4
    assert heads.count == 1 or (QS_CP + HEAD_DIM // 2 * tokens * heads.count <= POOL)

    @flyc.kernel(name=kernel_name, known_block_size=[THREADS, 1, 1])
    def post_attn_kernel(
        h_in: Int64,
        q: Int64,
        block_table: Int64,
        seq_lens: Int64,
        k_cache: Int64,
        v_cache: Int64,
        k_scale: Int64,
        v_scale: Int64,
        w_o: Int64,
        s_o: Int64,
        g_post: Int64,
        w_gate: Int64,
        bias: Int64,
        w13: Int64,
        s13: Int64,
        w2: Int64,
        s2: Int64,
        h_mid: Int64,
        ar_out: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        step: Int64,
        rank: Int32,
        layer: Int32,
        bt_width: Int32,
        q_len: Int32,
        ar_in: Int64,
        g_in: Int64,
        w_qkv: Int64,
        s_qkv: Int64,
        g_q: Int64,
        g_k: Int64,
        g_iq: Int64,
        g_ik: Int64,
        cos_sin: Int64,
        index_cache: Int64,
        iq_out: Int64,
        scratch1: Int64,
        positions: Int64,
        slot_mapping: Int64,
        res: Int64,
    ):
        _ = keyed
        # the guard's peer flags sit at the tail of each rank's peer buffer
        with guard.scope(
            guard.Guard(
                step,
                own=sym,
                peers=peers,
                rank=rank,
                npes=W,
                flags_offset=SY["_bytes"],
                fault_site=fault_site or None,
                fault_after=fault_after,
            )
        ):
            tid = fx.thread_idx.x
            bid = fx.block_idx.x
            lane = tid % 64
            wave = tid // 64
            g4 = lane // 16
            l16 = lane % 16
            lds = fx.SharedAllocator().allocate(Smem).peek()
            xs = lds.x.ptr
            red = lds.red.ptr
            p8 = lds.p8.ptr
            q8 = lds.q8.ptr
            # wide: the split's PV partials live in the pool, read and written as f32
            fpool = fx.recast_iter(fx.Float32, xs)
            opart = fpool if WIDE else lds.opart.ptr
            uexp = lds.uexp.ptr
            umask = lds.umask.ptr
            urow = lds.urow.ptr
            uwt = lds.uwt.ptr
            hst = lds.hst.ptr
            misc = lds.misc.ptr
            route = lds.route.ptr
            rwt = lds.rwt.ptr
            keys = lds.sel.ptr
            assert SEL_P8_WORDS <= H * SPLIT_KEYS // 4
            assert 2 * THREADS * CAND_BATCH <= WAVES * 64 * 4
            long_keys = fx.recast_iter(fx.Int32, red)
            cand = p8
            counter = cand + 2 * CAND_CAP
            floor_red = counter + 1
            blk = lds.blk.ptr
            v4f = fx.Vector.make_type(4, fx.Float32)
            v2i = fx.Vector.make_type(2, fx.Int32)

            mbox = Mailbox(tag=step_tag(step, layer))
            # flydsl carries a local whose method a dynamic if / for calls as loop state: bind them
            mb_put = mbox.put
            mb_put_bf = mbox.put_bf
            mb_put_words = mbox.put_words
            mb_poll = mbox.poll

            def mb(name):
                return scratch + fx.Int64(SCRATCH[name])

            def sy(base, region):
                """Region ``region`` of a rank's peer buffer ``base`` (system scope)."""
                return preg(base, SY[region], region)

            def acquire():
                memory_fence(
                    syncscope=rocdl.SyncScope.Workgroup,
                    ordering=AtomicOrdering.Acquire,
                )

            def scores_copy(c, k=0):
                """Copy c of token k's router outputs: expert e's (routing key, sigmoid)
                at pairs 2 e, 2 e + 1."""
                return mb("scores") + (fx.Int64(c) * MAX_TOKENS + fx.Int64(k)) * (
                    N_ROUTED * 16
                )

            def route_key(v, e):
                """Signed-orderable i32 of score ``v`` with its low 7 bits replaced by
                127 - expert ``e``: a pick is one integer max, a tie goes to the lower
                expert."""
                b = v.bitcast(fx.Int32)
                b = (b < 0).select(b ^ fx.Int32(0x7FFFFFFF), b)
                return (b & fx.Int32(-128)) | (127 - e)

            # each wave pushes to one peer
            pv = fx.Vector(
                bo.buffer_load(
                    rsrc(peers), fx.min(wave, W - 1) * 2, vec_width=2, dtype=T.i32
                )
            )
            peer_dst = (fx.Int64(uniform(pv[1])) << 32) | fx.Int64(
                fx.Uint32(uniform(pv[0]))
            )
            AG_RS = tokens >= AG_RS_FROM

            def peer_addr(w):
                pw = fx.Vector(
                    bo.buffer_load(rsrc(peers), w * 2, vec_width=2, dtype=T.i32)
                )
                return (fx.Int64(uniform(pw[1])) << 32) | fx.Int64(
                    fx.Uint32(uniform(pw[0]))
                )

            def start(name):
                return (bid + (G - BASE[name])) % G

            def ld_bf16x4(r, k, cm=0):
                w = fx.Vector(
                    bo.buffer_load(
                        r, k // 2, vec_width=2, dtype=T.i32, cache_modifier=cm
                    )
                )
                v = w.bitcast(fx.BFloat16).to(fx.Float32)
                return [v[j] for j in range(4)]

            def mfma_fp8(a64, b64, c):
                return fx.Vector(
                    rocdl.mfma_f32_16x16x32_fp8_fp8(
                        T.vec(4, T.f32), [a64, b64, c, 0, 0, 0]
                    )
                )

            def block_max(v):
                return block_max_of(v, lane, wave, red)

            def push_rows(region, row0, n_rows, only=None):
                """This task's bf16 partial rows (misc: token k's n_rows from k n_rows) to
                the peers' ``region``, a pair a lane, wave w -> peer w (itself too);
                ``only``: to that peer alone."""
                half = n_rows // 2
                to = (wave < W) if only is None else (wave == only)
                for c in range_constexpr((half * tokens + 63) // 64):
                    i = lane + 64 * c
                    if to & (i < half * tokens):
                        mb_put_bf(
                            sy(peer_dst, region),
                            (rank * MAX_TOKENS + i // half) * HIDDEN
                            + row0
                            + (i % half) * 2,
                            [
                                fx.ptr_load(misc + i * 2),
                                fx.ptr_load(misc + (i * 2 + 1)),
                            ],
                        )

            # this layer's K1 first: its GEMV / head / norm tasks and the scores. Its pointers are kernel
            # arguments here (upstream: the runner's per-layer device table of K1_ARGS)
            qs_k1 = lds.qs.ptr
            if const_expr(ih_count > 1):
                qs_k1 = fpool + QS_CP
            emit_k1(
                tid, bid, lane, wave, xs, red, qs_k1, step, layer,
                eps, index_scale_log2e(sm_scale), tokens, q_len, init_blocks,
                local_blocks, IndexHeads(ih_count, ih_own),
                (
                    ar_in, res, h_in, g_in, w_qkv, s_qkv, g_q, g_k, g_iq, g_ik, cos_sin,
                    positions, slot_mapping, k_cache, v_cache, k_scale, v_scale,
                    index_cache, q, iq_out, block_table, seq_lens, mb("iscore"), scratch1,
                ),
                bt_width,
            )  # fmt: skip
            gpu.barrier()
            k1_mbox = Mailbox(tag=step_tag(step, layer))
            k1_poll = k1_mbox.poll
            hdone_mb = scratch1 + fx.Int64(K1_SCRATCH_HDONE)
            rdone_mb = scratch1 + fx.Int64(K1_SCRATCH_RDONE)

            # ============================================================ 0. index
            # a graph pad row (seq_len 0, zero table row) is one key of page 0: finite, a real page
            seq_lens_k, n_blks, long_rows = step_rows(
                seq_lens, tokens, q_len, IndexHeads(ih_count, ih_own)
            )
            # all <= TOPK_BLOCKS blocks are kept, any order correct (index_topk._pack_score_key)
            scored = [nb > TOPK_BLOCKS for nb in n_blks]

            def bt_row(k):
                return block_table + fx.Int64(k) * fx.Int64(bt_width) * 4

            def pick(vals, k):
                """vals[k] for a traced token index k."""
                v = vals[0]
                for i in range_constexpr(1, tokens):
                    v = (k == i).select(vals[i], v)
                return v

            def index_key(b, word, n_blk):
                """Sort key of block b's raw score word: forced blocks pinned, then a
                signed int ordered like the float (-0 == 0, NaN below -inf)."""
                sc = word.bitcast(fx.Float32)
                sc = (b < init_blocks).select(fx.Float32(1e30), sc)
                sc = (b >= n_blk - local_blocks).select(fx.Float32(1e29), sc)
                bits = sc.bitcast(fx.Int32)
                bits = (bits == fx.Int32(-(2**31))).select(fx.Int32(0), bits)
                key = (bits < 0).select(bits ^ 0x7FFFFFFF, bits)
                return ((bits & 0x7FFFFFFF) > 0x7F800000).select(
                    fx.Int32(-(2**31)), key
                )

            def rank_in_cta(entries, m, ents):
                """Counts of the CTA's ``m`` entries that beat each of this thread's
                ``entries`` (key, block, live), exact for the TOPK_BLOCKS best.

                Entry i is entry tid + THREADS i of ``ents``. Only entries at or past
                the lane groups' floor are counted against each other, every entry
                only when more than CAND_CAP pass it. Contains barriers."""
                mines = [
                    put_key(ents, tid + THREADS * i, h, b, lv)
                    for i, (h, b, lv) in enumerate(entries)
                ]
                if tid < CAND_CAP:  # dead: past the candidates every count reads
                    fx.ptr_store(fx.Int32(0), cand + 2 * tid)
                    fx.ptr_store(fx.Int32(KEY_LEAST), cand + (2 * tid + 1))
                if tid == 0:
                    fx.ptr_store(fx.Int32(0), counter)
                floor = group_floor(
                    [lv.select(h, fx.Int32(KEY_LEAST)) for h, _, lv in entries],
                    TOPK_BLOCKS, floor_red, lane, wave,
                )  # fmt: skip
                compact(cand, counter, CAND_CAP, entries, floor)
                gpu.barrier()
                n_cand = fx.ptr_load(counter)
                every = n_cand > CAND_CAP
                # separate arrays: the count that does not apply runs no round
                in_cand = count_beaten(cand, mines, every.select(0, n_cand))
                in_all = count_beaten(ents, mines, every.select(m, 0))
                return [a + b for a, b in zip(in_cand, in_all)]

            def rank_blocks(k, iscore, n_blk, r_bt, scored_k):
                """Up to THREADS blocks: thread b keys block b, ranked among them
                (``rank_in_cta``). -> this thread's (key, block, rank, live, the tail
                block's key, page). ``scored_k``: the scores are read (else the blocks
                rank by id)."""
                b = tid
                live = b < n_blk
                bb = fx.min(b, n_blk - 1)
                page = fx.Int32(bo.buffer_load(r_bt, bb, vec_width=1, dtype=T.i32))
                # scorers run in this launch, so poll; a pinned block polls a scored block's slot
                word = fx.Int32(0)
                if scored_k:
                    word = mb_poll(
                        [
                            (
                                iscore,
                                fx.min(
                                    fx.max(tid, init_blocks),
                                    n_blk - 1 - local_blocks,
                                ),
                                1,
                            )
                        ]
                    )[0][0]
                key = scored_k.select(index_key(tid, word, n_blk), -tid)
                (rank,) = rank_in_cta([(key, b, live)], n_blk, keys)
                kt = fx.ptr_load(keys + (2 * (n_blk - 1) + 1))
                return key, bb, rank, live, kt, page

            def ballot(pred):
                return fx.Int64(rocdl.ballot(T.i64, pred))

            def place(key, bb, rank, live, kt, n_blk, seq_len, active):
                """The ranked blocks' slots, as the unfused selector emits them: full
                blocks by rank, the tail block (the one holding the current token)
                last. -> (selected, slot, sparse context length); ``active``: this
                CTA places (a uniform flag)."""
                n_sel = fx.min(n_blk, TOPK_BLOCKS)
                tail = n_blk - 1
                tail_first = (kt > key) | ((kt == key) & (tail > bb))
                sel = live & (rank < n_sel) & active
                is_tail = sel & (bb == tail)
                if is_tail:
                    fx.ptr_store(fx.Int32(1), blk + 17)
                gpu.barrier()
                tail_sel = fx.ptr_load(blk + 17) != 0
                n_full = n_sel - tail_sel.select(fx.Int32(1), fx.Int32(0))
                slot = is_tail.select(
                    n_full, rank - tail_first.select(fx.Int32(1), fx.Int32(0))
                )
                n_ctx = tail_sel.select(
                    n_full * SPARSE_BLOCK + seq_len - tail * SPARSE_BLOCK,
                    fx.min(n_sel * SPARSE_BLOCK, seq_len),
                )
                return sel, slot, n_ctx

            def select_blocks(k, t):
                """blk[0:16] := token k's pages of selection slots 2t, 2t+1, blk[16] :=
                the sparse context length.

                Exactly as the unfused selector emits them: top TOPK_BLOCKS by (score,
                block id) descending with the init blocks pinned to 1e30 and the local
                ones to 1e29, full blocks in that order and the tail block (the one
                holding the current token) last. For a long request
                ``stage_select_long`` filled blk: a one-block stand-in is ranked,
                placing nothing."""
                iscore = mb("iscore") + fx.Int64(k) * (MAX_INDEX_BLOCKS * 8)
                short = ~pick(long_rows, k)
                n_blk = short.select(pick(n_blks, k), fx.Int32(1))
                seq_len = pick(seq_lens_k, k)
                r_bt = rsrc(bt_row(k))
                if (tid < 2 * PAGES_PER_BLOCK + 2) & short:
                    fx.ptr_store(fx.Int32(0), blk + tid)
                key, bb, rank, live, kt, page = rank_blocks(
                    k, iscore, n_blk, r_bt, pick(scored, k) & short
                )
                sel, slot, n_ctx = place(key, bb, rank, live, kt, n_blk, seq_len, short)
                if sel & (slot // 2 == t):
                    for j in range_constexpr(PAGES_PER_BLOCK):
                        fx.ptr_store(
                            page * PAGES_PER_BLOCK + j,
                            blk + ((slot % 2) * PAGES_PER_BLOCK + j),
                        )
                if (tid == 0) & short:
                    fx.ptr_store(n_ctx, blk + 16)
                gpu.barrier()

            def select_long(k, part):
                """Split task (k, part)'s selection for a long request -> blk as
                ``select_blocks`` leaves it.

                The task ranks its share of the blocks (a thread keys CAND_BATCH) and
                mails the share's TOPK_BLOCKS best; the N_SPLIT shares' candidates,
                which hold the context's TOPK_BLOCKS best, are then ranked and placed.
                A share is 1 / N_SPLIT of the blocks, mailed to this rank's sel_long;
                with every index head (indexer context parallelism) it is head
                part // 2's scores of half part % 2 of this rank's blocks, mailed to
                the head's rank's sel_cp, where the shares of its own head come in."""
                n_blk = pick(n_blks, k)
                iscore = mb("iscore") + fx.Int64(k) * (MAX_INDEX_BLOCKS * 8)
                first = k * (N_SPLIT * TOPK_BLOCKS)  # token k's candidates (pair pairs)
                if tid == 0:
                    fx.ptr_store(fx.Int32(0), blk + 17)
                if const_expr(ih_count > 1):
                    # this rank's first block
                    own0 = init_blocks + (ih_own - init_blocks) % TP
                    n_own = (n_blk - own0 + TP - 1) // TP
                    # the last scored one: the local blocks' slots hold no score
                    last = own0 + TP * (
                        (n_blk - local_blocks - own0 + TP - 1) // TP - 1
                    )
                    span = (n_own + 1) // 2
                    lo = (part % 2) * span
                    hi = fx.min(lo + span, n_own)
                    js = [lo + tid + THREADS * i for i in range(CAND_BATCH)]
                    blocks = [own0 + TP * j for j in js]
                    live = [j < hi for j in js]
                    src = iscore + fx.Int64(part // 2) * (MAX_INDEX_BLOCKS // TP * 8)
                    slots = [fx.min(b, last) // TP for b in blocks]
                    out, out_slot = sy(peer_addr(part // 2), "sel_cp"), 2 * rank
                    out_slot = out_slot + part % 2
                    got_from = sy(sym, "sel_cp")
                else:
                    span = (n_blk + N_SPLIT - 1) // N_SPLIT
                    lo = part * span
                    hi = fx.min(lo + span, n_blk)
                    blocks = [lo + tid + THREADS * i for i in range(CAND_BATCH)]
                    live = [b < hi for b in blocks]
                    src = iscore
                    slots = [
                        fx.min(fx.max(b, init_blocks), n_blk - 1 - local_blocks)
                        for b in blocks
                    ]
                    out, out_slot = mb("sel_long"), part
                    got_from = out
                words = [w[0] for w in mb_poll([(src, sl, 1) for sl in slots])]
                # the share's entry tid + THREADS i is its block i of this thread's
                hs = [index_key(b, w, n_blk) for b, w in zip(blocks, words)]
                cnts = rank_in_cta(list(zip(hs, blocks, live)), hi - lo, long_keys)
                # a share holds >= TOPK_BLOCKS blocks: its 16 best have distinct ranks
                for h, b, lv, c in zip(hs, blocks, live, cnts):
                    if lv & (c < TOPK_BLOCKS):
                        mb_put_words(
                            out,
                            2 * (first + out_slot * TOPK_BLOCKS + c),
                            [h, b],
                        )
                gpu.barrier()
                # every candidate is live (distinct ranks); its page load issues before the count
                n_c = N_SPLIT * TOPK_BLOCKS
                if tid < n_c:
                    got = mb_poll([(got_from, 2 * (first + tid), 2)])[0]
                    fx.ptr_store(got[1], keys + 2 * tid)  # put_key's (block, key)
                    fx.ptr_store(got[0], keys + (2 * tid + 1))
                gpu.barrier()
                held = tid < n_c
                cl = fx.ptr_load(keys + 2 * fx.min(tid, n_c - 1))
                ch = fx.ptr_load(keys + (2 * fx.min(tid, n_c - 1) + 1))
                page = fx.Int32(
                    bo.buffer_load(
                        rsrc(bt_row(k)), fx.max(cl, 0), vec_width=1, dtype=T.i32
                    )
                )
                (cnt,) = count_beaten(keys, [(fx.Int64(ch) << 32) | fx.Int64(cl)], n_c)
                # the tail block is a local one: its key is forced
                kt = index_key(n_blk - 1, fx.Int32(0), n_blk)
                sel, slot, n_ctx = place(
                    ch,
                    cl,
                    cnt,
                    held,
                    kt,
                    n_blk,
                    pick(seq_lens_k, k),
                    pick(long_rows, k),
                )
                if sel & (slot // 2 == part):
                    for j in range_constexpr(PAGES_PER_BLOCK):
                        fx.ptr_store(
                            page * PAGES_PER_BLOCK + j,
                            blk + ((slot % 2) * PAGES_PER_BLOCK + j),
                        )
                if tid == 0:
                    fx.ptr_store(n_ctx, blk + 16)
                gpu.barrier()

            def stage_select_long():
                """``select_long`` for this CTA's split task (at most one) of a long request.

                Cold and out of the split stage: inlined there it slows the split for
                every context."""
                ts = start("split")
                k = ts // N_SPLIT
                if unlikely((ts < N_SPLIT * tokens) & pick(long_rows, k)):
                    select_long(k, ts % N_SPLIT)

            # a function a stage: flydsl carries a local reassigned in a dynamic if / for as state
            # ============================================================ 1. split
            def stage_split():
                r_ks, r_vs = rsrc(k_scale), rsrc(v_scale)
                PPW = SPLIT_KEYS // WAVES // PAGE16  # pages per wave
                for ts in range(start("split"), N_SPLIT * tokens, G):
                    ts = fx.Int32(ts)
                    tok = ts // N_SPLIT
                    t = ts % N_SPLIT
                    select_blocks(tok, t)
                    # every token's K / V too: a speculative step's tokens attend to each other
                    if wave == 0:
                        kv = lane - H
                        k1_poll(
                            [
                                (
                                    hdone_mb,
                                    (lane < H).select(
                                        tok * K1_HEAD_TASKS + lane,
                                        fx.min(kv // 2, tokens - 1) * K1_HEAD_TASKS
                                        + H
                                        + kv % 2,
                                    ),
                                    1,
                                )
                            ]
                        )
                    gpu.barrier()
                    acquire()
                    # independent loads first; a page past the context reads junk, masked below
                    n_ctx = uniform(fx.ptr_load(blk + 16))
                    pages = [
                        uniform(fx.ptr_load(blk + (wave * PPW + j))) for j in range(PPW)
                    ]
                    qv = ld_bf16x4(rsrc(q), tok * O_K + tid * 4, CM_K1)
                    # consecutive pages: a 64-bit base keeps lane offsets 32-bit, any cache size
                    page_base = fx.Int64(pages[0]) * PAGE_BYTES
                    r_k, r_v = rsrc(k_cache + page_base), rsrc(v_cache + page_base)
                    # K, the QK B operand: key l16 of page j
                    kw = []
                    for j in range_constexpr(PPW):
                        for s in range_constexpr(HEAD_DIM // 32):
                            kw.append(
                                fx.Vector(
                                    bo.buffer_load(
                                        r_k,
                                        (
                                            (pages[j] - pages[0]) * PAGE_BYTES
                                            + (2 * s + g4 // 2) * 256
                                            + l16 * 16
                                            + (g4 % 2) * 8
                                        )
                                        // 4,
                                        vec_width=2,
                                        dtype=T.i32,
                                        cache_modifier=CM_K1,
                                    )
                                ).bitcast(fx.Int64)[0]
                            )
                    # V, the PV B operand: dims 16 jd + l16, keys 8 g4 .. + 8
                    vpg = (g4 // 2 == 0).select(fx.Int32(0), pages[1] - pages[0])
                    vw = [
                        fx.Vector(
                            bo.buffer_load(
                                r_v,
                                (
                                    vpg * PAGE_BYTES
                                    + (16 * jd + l16) * PAGE16
                                    + (g4 % 2) * 8
                                )
                                // 4,
                                vec_width=2,
                                dtype=T.i32,
                                cache_modifier=CM_K1,
                            )
                        ).bitcast(fx.Int64)[0]
                        for jd in range(HEAD_DIM // 16)
                    ]
                    ks1 = fx.Float32(bo.buffer_load(r_ks, 0, vec_width=1, dtype=T.f32))
                    vs1 = fx.Float32(bo.buffer_load(r_vs, 0, vec_width=1, dtype=T.f32))
                    kss = [ks1] * PPW
                    vss = [vs1] * PPW
                    fx.ptr_store(fp8_pack4(qv[0], qv[1], qv[2], qv[3]), q8 + tid)
                    gpu.barrier()
                    # QK per page j: A = q (heads), B = K (keys); a lane: heads 4 g4 + e, key l16
                    key0 = t * SPLIT_KEYS + wave * (PPW * PAGE16)
                    valid = [key0 + j * PAGE16 + l16 < n_ctx for j in range(PPW)]
                    sc = []
                    for j in range_constexpr(PPW):
                        c = fx.Vector.filled(4, 0.0, fx.Float32)
                        for s in range_constexpr(HEAD_DIM // 32):
                            qw = fx.ptr_load(
                                q8 + (l16 * 32 + 8 * s + 2 * g4), result_type=v2i
                            )
                            c = mfma_fp8(
                                qw.bitcast(fx.Int64)[0], kw[j * (HEAD_DIM // 32) + s], c
                            )
                        qk = sm_scale * kss[j]
                        sc.append(
                            [
                                valid[j].select(qk * c[e], fx.Float32(NEG))
                                for e in range(4)
                            ]
                        )
                    # partition max per head and max v_scale: in-wave butterfly, then LDS
                    vloc = fx.max(
                        valid[0].select(vss[0], fx.Float32(0.0)),
                        valid[1].select(vss[1], fx.Float32(0.0)),
                    )
                    vmax_w = wave_max(vloc)
                    for e in range_constexpr(4):
                        mh = butterfly(fx.max(sc[0][e], sc[1][e]), (8, 4, 2, 1), fx.max)
                        if l16 == 0:
                            fx.ptr_store(mh, hst + (wave * H + g4 * 4 + e))
                    if lane == 0:
                        fx.ptr_store(vmax_w, hst + (3 * WAVES * H + wave))
                    gpu.barrier()
                    # read every partial before any p8 store: stores in between serialize the reads
                    vmax = fx.ptr_load(hst + 3 * WAVES * H)
                    for w in range_constexpr(1, WAVES):
                        vmax = fx.max(vmax, fx.ptr_load(hst + (3 * WAVES * H + w)))
                    mxs = []
                    for e in range_constexpr(4):
                        mx = fx.ptr_load(hst + (g4 * 4 + e))
                        for w in range_constexpr(1, WAVES):
                            mx = fx.max(mx, fx.ptr_load(hst + (w * H + g4 * 4 + e)))
                        mxs.append(mx)
                    fscale = FP8_MAX * hw_rcp(vmax + 1e-8)
                    pscale = vmax * (1.0 / FP8_MAX)
                    for e in range_constexpr(4):
                        head = g4 * 4 + e
                        mx = mxs[e]
                        ls = fx.Float32(0.0)
                        for j in range_constexpr(PPW):
                            pj = hw_exp2((sc[j][e] - mx) * LOG2E)
                            ls = ls + pj
                            b8 = (
                                fp8_pack4(
                                    (valid[j].select(vss[j], fx.Float32(0.0)) * fscale)
                                    * pj,
                                    0.0,
                                    0.0,
                                    0.0,
                                )
                                & 0xFF
                            )
                            w8 = b8 | (xshfl(b8, 1) << 8)
                            w8 = w8 | (xshfl(w8, 2) << 16)
                            if l16 % 4 == 0:
                                kloc = wave * (PPW * PAGE16) + j * PAGE16 + l16
                                fx.ptr_store(
                                    w8, p8 + (head * (SPLIT_KEYS // 4) + kloc // 4)
                                )
                        ls = butterfly(ls, (8, 4, 2, 1))
                        if l16 == 0:
                            fx.ptr_store(ls, hst + (WAVES * H + wave * H + head))
                            fx.ptr_store(mx, hst + (2 * WAVES * H + head))
                    gpu.barrier()
                    # PV over this wave's 32 keys: A = P (heads), B = V (dims)
                    pw = fx.ptr_load(
                        p8
                        + (
                            l16 * (SPLIT_KEYS // 4)
                            + (wave * (PPW * PAGE16) + 8 * g4) // 4
                        ),
                        result_type=v2i,
                    ).bitcast(fx.Int64)[0]
                    # keys past the context carry P = 0, but a NaN V byte would still poison
                    nv = n_ctx - (key0 + 8 * g4)
                    vmask = (nv >= 8).select(
                        fx.Int64(-1),
                        (nv <= 0).select(
                            fx.Int64(0), (fx.Int64(1) << (fx.Int64(nv) * 8)) - 1
                        ),
                    )
                    for jd in range_constexpr(HEAD_DIM // 16):
                        cv = mfma_fp8(
                            pw, vw[jd] & vmask, fx.Vector.filled(4, 0.0, fx.Float32)
                        )
                        for e in range_constexpr(4):
                            fx.ptr_store(
                                cv[e],
                                opart
                                + (
                                    wave * O_K + (g4 * 4 + e) * HEAD_DIM + 16 * jd + l16
                                ),
                            )
                    gpu.barrier()
                    # sum the waves' partials: acc = prob_scale * PV, then * (1 / exp_sum), bf16
                    e0 = tid * 4
                    head = e0 // HEAD_DIM
                    lsum = fx.ptr_load(hst + (WAVES * H + head))
                    for w in range_constexpr(1, WAVES):
                        lsum = lsum + fx.ptr_load(hst + (WAVES * H + w * H + head))
                    inv_l = 1.0 / (lsum > 0.0).select(lsum, fx.Float32(1.0))
                    ov = []
                    for k in range_constexpr(4):
                        acc = fx.ptr_load(opart + (e0 + k))
                        for w in range_constexpr(1, WAVES):
                            acc = acc + fx.ptr_load(opart + (w * O_K + e0 + k))
                        ov.append((pscale * acc) * inv_l)
                    mb_put_bf(mb("sp_o"), ts * O_K + e0, ov)
                    if tid < H:
                        mb_put(
                            mb("sp_m"),
                            ts * H + tid,
                            fx.ptr_load(hst + (2 * WAVES * H + tid)),
                        )
                        lt = fx.ptr_load(hst + (WAVES * H + tid))
                        for w in range_constexpr(1, WAVES):
                            lt = lt + fx.ptr_load(hst + (WAVES * H + w * H + tid))
                        mb_put(mb("sp_l"), ts * H + tid, lt)
                    gpu.barrier()

            stage_select_long()
            stage_split()

            def merge_token(k):
                """Token k's partitions merged (the paged decode's reduce; thread = 4 dims
                of one head) and quantized per token (standalone quant: scale = amax *
                (1 / FP8_MAX), x * rcp(scale)) -> (this thread's fp8 word, scale)."""
                hm = tid // (HEAD_DIM // 4)
                e0 = tid * 4
                sp0 = k * N_SPLIT  # token k's partitions
                # wave 0 waits alone: a whole CTA spinning loads the memory the split reads
                if wave == 0:
                    mb_poll(
                        [
                            (mb("sp_l"), sp0 * H + lane + 64 * i, 1)
                            for i in range(N_SPLIT * H // 64)
                        ]
                    )
                gpu.barrier()
                got = mb_poll(
                    [(mb("sp_m"), (sp0 + i) * H + hm, 1) for i in range(N_SPLIT)]
                    + [(mb("sp_l"), (sp0 + i) * H + hm, 1) for i in range(N_SPLIT)]
                    + [
                        (mb("sp_o"), ((sp0 + i) * O_K + e0) // 2, 2)
                        for i in range(N_SPLIT)
                    ],
                    batch=3 * N_SPLIT,
                )
                mls = got[: 2 * N_SPLIT]
                ovs = got[2 * N_SPLIT :]
                m_i = [mls[i][0].bitcast(fx.Float32) for i in range(N_SPLIT)]
                l_i = [mls[N_SPLIT + i][0].bitcast(fx.Float32) for i in range(N_SPLIT)]
                gm = m_i[0]
                for i in range_constexpr(1, N_SPLIT):
                    gm = fx.max(gm, m_i[i])
                sl = [l_i[i] * hw_exp2((m_i[i] - gm) * LOG2E) for i in range(N_SPLIT)]
                gs = sl[0]
                for i in range_constexpr(1, N_SPLIT):
                    gs = gs + sl[i]
                gs = (gs > 0.0).select(gs, fx.Float32(1.0))
                av = [fx.Float32(0.0) for _ in range(4)]
                for i in range_constexpr(N_SPLIT):
                    wi = sl[i] / gs
                    o01 = bf2_f32(ovs[i][0])
                    o23 = bf2_f32(ovs[i][1])
                    av = [
                        av[0] + wi * o01[0],
                        av[1] + wi * o01[1],
                        av[2] + wi * o23[0],
                        av[3] + wi * o23[1],
                    ]
                a0, a1, a2, a3 = (bf16_round(x) for x in av)
                amax = block_max(
                    fx.max(
                        fx.max(fmath.absf(a0), fmath.absf(a1)),
                        fx.max(fmath.absf(a2), fmath.absf(a3)),
                    )
                )
                x_scale = amax * (1.0 / FP8_MAX)
                # an all-zero row (a pad row) quantizes to zeros, not 0 * inf
                inv = (amax == 0.0).select(fx.Float32(0.0), hw_rcp(x_scale))
                return fp8_pack4(a0 * inv, a1 * inv, a2 * inv, a3 * inv), x_scale

            # ======================================= 2b. attention merge (S > 1)
            # a task per token: in the o tasks, every o task would run every token's merge in turn
            def stage_merge():
                if const_expr(tokens > 1):
                    for k in range(start("merge"), tokens, G):
                        k = fx.Int32(k)
                        word, x_scale = merge_token(k)
                        # plain words, then the scale: its pair is the flag
                        bo.buffer_store(
                            word,
                            rsrc(mb("attnq")),
                            k * (O_K // 4) + tid,
                            cache_modifier=CM_DEV,
                        )
                        release_stores()
                        gpu.barrier()
                        if tid == 0:
                            mb_put(mb("attnq_s"), k, x_scale)

            stage_merge()

            # ================================================ 3. o_proj + attn reduce
            def stage_o():
                r_wo = rsrc(w_o)
                O_CPW = O_K // 64 // 4  # 64-k chunks per wave: 4 waves per 16-row group
                for t in range(start("o"), N_O, G):
                    t = fx.Int32(t)
                    rg = t * 2 + wave // 4
                    wts = []
                    for cc in range_constexpr(O_CPW):
                        kc = (wave % 4) * O_CPW + cc
                        wts.append(
                            fx.Vector(
                                bo.buffer_load(
                                    r_wo,
                                    ((rg * (O_K // 32) + kc * 2) * 512 + lane * 16)
                                    // 4,
                                    vec_width=4,
                                    dtype=T.i32,
                                )
                            )
                        )
                    # this task's row scales, before the wait
                    ws = fx.Float32(
                        bo.buffer_load(
                            rsrc(s_o),
                            t * O_ROWS + tid % O_ROWS,
                            vec_width=1,
                            dtype=T.f32,
                        )
                    )

                    if const_expr(tokens == 1):
                        # the merge folded in: the partitions -> this CTA's fp8 input
                        word, x_scale = merge_token(0)
                        x_scales = [x_scale]
                        fx.ptr_store(word, xs + tid)
                    else:
                        # scales first: each is stored after its token's words, so it flags them
                        got = mb_poll([(mb("attnq_s"), k, 1) for k in range(tokens)])
                        x_scales = [
                            got[k][0].bitcast(fx.Float32) for k in range(tokens)
                        ]
                        acquire()
                        n_v = tokens * O_K // 16
                        for i in range_constexpr((n_v + THREADS - 1) // THREADS):
                            v = tid + THREADS * i
                            if v < n_v:
                                fx.ptr_store(
                                    fx.Vector(
                                        bo.buffer_load(
                                            rsrc(mb("attnq")),
                                            v * 4,
                                            vec_width=4,
                                            dtype=T.i32,
                                            cache_modifier=CM_DEV,
                                        )
                                    ),
                                    xs
                                    + (
                                        v * 4 // (O_K // 4) * O_ROW + v * 4 % (O_K // 4)
                                    ),
                                )
                    gpu.barrier()
                    # B column l16 = token l16 (clamped), so the weight stream is the same at any S
                    col_tok = fx.min(l16, tokens - 1)
                    c = fx.Vector.filled(4, 0.0, fx.Float32)
                    # 2 chunks a 16x16x128 MFMA (unit scales): k 16 g4 .. + 16 of each 64-k half
                    for p in range_constexpr(O_CPW // 2):
                        xb = [
                            fx.ptr_load(
                                xs
                                + (
                                    col_tok * O_ROW
                                    + ((wave % 4) * O_CPW + 2 * p + h) * 16
                                    + g4 * 4
                                ),
                                result_type=v4f,
                            ).bitcast(fx.Int32)
                            for h in range(2)
                        ]
                        c = fx.Vector(
                            rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                T.vec(4, T.f32),
                                [
                                    fx.Vector.from_elements(
                                        [
                                            wts[2 * p + h][e]
                                            for h in range(2)
                                            for e in range(4)
                                        ],
                                        fx.Int32,
                                    ),
                                    fx.Vector.from_elements(
                                        [xb[h][e] for h in range(2) for e in range(4)],
                                        fx.Int32,
                                    ),
                                    c,
                                    0,
                                    0,
                                    0,
                                    fx.Int32(127),
                                    0,
                                    fx.Int32(127),
                                ],
                            )
                        )
                    fx.ptr_store(c, red + (wave * 64 + lane) * 4)
                    gpu.barrier()
                    if tid < O_ROWS * tokens:
                        tk = tid // O_ROWS
                        grp = (tid % O_ROWS) // 16
                        rr = tid % 16
                        tot = fx.Float32(0.0)
                        for w in range_constexpr(4):
                            tot = tot + fx.ptr_load(
                                red
                                + (
                                    ((grp * 4 + w) * 64 + 16 * (rr // 4) + tk) * 4
                                    + rr % 4
                                )
                            )
                        fx.ptr_store(
                            bf16_round(tot * pick(x_scales, tk) * ws), misc + tid
                        )
                    gpu.barrier()
                    # with AG_RS only to the row group's owner rank
                    push_rows("attn", t * O_ROWS, O_ROWS, t % W if AG_RS else None)
                    gpu.barrier()
                    if const_expr(AG_RS):  # noqa: SIM102
                        # the owner's rank-ordered sum goes to every a_ag; the router writes h_mid
                        if (t % W == rank) & (tid < O_ROWS // 2 * tokens):
                            ptk = tid // (O_ROWS // 2)
                            row = t * O_ROWS + (tid % (O_ROWS // 2)) * 2
                            own = sy(sym, "attn")
                            prt = mb_poll(
                                [
                                    (
                                        own,
                                        ((src * MAX_TOKENS + ptk) * HIDDEN + row) // 2,
                                        1,
                                    )
                                    for src in range(W)
                                ],
                            )
                            t0 = fx.Float32(0.0)
                            t1 = fx.Float32(0.0)
                            for src in range_constexpr(W):
                                f0, f1 = bf2_f32(prt[src][0])
                                t0 = t0 + f0
                                t1 = t1 + f1
                            for w in range_constexpr(W):
                                mb_put_bf(
                                    sy(peer_addr(w), "a_ag"),
                                    ptk * HIDDEN + row,
                                    [t0, t1],
                                )
                    if const_expr(not AG_RS):  # noqa: SIM102
                        # as GLM's peer_reduce: routers then read device-scope bf16, not the peers'
                        if tid < O_ROWS // 2 * tokens:
                            ptk = tid // (O_ROWS // 2)
                            row = t * O_ROWS + (tid % (O_ROWS // 2)) * 2
                            # the residual (K1's norm / GEMV task)
                            k1_poll([(rdone_mb, ptk, 1)])
                            acquire()
                            hr = fx.Int32(
                                bo.buffer_load(
                                    rsrc(h_in),
                                    (ptk * HIDDEN + row) // 2,
                                    vec_width=1,
                                    dtype=T.i32,
                                    cache_modifier=CM_K1,
                                )
                            )
                            own = sy(sym, "attn")
                            prt = mb_poll(
                                [
                                    (
                                        own,
                                        ((src * MAX_TOKENS + ptk) * HIDDEN + row) // 2,
                                        1,
                                    )
                                    for src in range(W)
                                ],
                            )
                            t0 = fx.Float32(0.0)
                            t1 = fx.Float32(0.0)
                            for src in range_constexpr(W):
                                f0, f1 = bf2_f32(prt[src][0])
                                t0 = t0 + f0
                                t1 = t1 + f1
                            r0, r1 = bf2_f32(hr)
                            a0 = bf16_round(t0) + r0
                            a1 = bf16_round(t1) + r1
                            # sum only: the norm reads bf16(sum) + h unrounded, as the unfused path
                            mb_put_bf(mb("a"), ptk * HIDDEN + row, [t0, t1])
                            bo.buffer_store(
                                fx.Vector.from_elements([a0, a1], fx.Float32).to(
                                    fx.BFloat16
                                ),
                                rsrc(h_mid),
                                ptk * HIDDEN + row,
                            )

            stage_o()

            # ======================================================= MoE helpers
            def e8m0_load(r_s, byte_idx):
                """An e8m0 scale byte, still in flight (the MFMA takes it as is)."""
                return fx.Int32(bo.buffer_load(r_s, byte_idx, vec_width=1, dtype=T.i8))

            def group_scale_byte(expert, rows, rg, kc, cols):
                """scale_index(expert rows + 16 rg + l16, 4 kc + g4, cols): this lane's
                byte for row group rg, 128-k chunk kc, folded to wave-uniform terms plus
                the lane: the general form's per-lane arithmetic dominates the up /
                gate loop."""
                base = (expert * (rows // 32) + rg // 2) * (cols // 8) + kc // 2
                return (base * 64 + lane) * 4 + (kc % 2) * 2 + rg % 2

            def route_top4(k, k0, sc0, k1, sc1, sig, wave_mask=None, mask_bit=0):
                """route[8k : 8k+5] / rwt[..] := token k's sigmoid top-k gating, slot 4 the
                fused shared expert.

                Wave k % WAVES picks from the routing keys ``k0`` / ``k1`` (experts
                ``lane`` / ``lane + 64``, see ``route_key``), one wave max a pick; the
                weights are the picks' unbiased sigmoids, looked up in LDS at ``sig``,
                renormalized and scaled by route_scale. ``wave_mask``: this wave's
                expert -> token mask, ``mask_bit`` (bit k, or 0) or-ed in at the picks
                (distinct, and the wave's tokens run in program order). The caller
                syncs."""
                if wave == k % WAVES:
                    # the unbiased weights, looked up by expert once the picks are known
                    fx.ptr_store(sc0, sig + (k * N_ROUTED + lane))
                    fx.ptr_store(sc1, sig + (k * N_ROUTED + 64 + lane))
                    pid = fx.Int32(0)
                    for p in range_constexpr(TOP_K):
                        m = butterfly(fx.max(k0, k1), (32, 16, 8, 4, 2, 1), fx.max)
                        hit0 = k0 == m
                        k0 = hit0.select(fx.Int32(-(2**31)), k0)
                        k1 = ((k1 == m) & (k0 != m)).select(fx.Int32(-(2**31)), k1)
                        pid = (lane == p).select(127 - (m & 127), pid)
                    if lane < TOP_K:
                        w = fx.ptr_load(sig + (k * N_ROUTED + pid))
                        tot = butterfly(w, (1, 2))
                        fx.ptr_store(pid, route + (8 * k + lane))
                        fx.ptr_store(
                            w * (route_scale / fx.max(tot, 1e-20)), rwt + (8 * k + lane)
                        )
                        if const_expr(wave_mask is not None):
                            was = fx.ptr_load(wave_mask + pid)
                            fx.ptr_store(was | mask_bit, wave_mask + pid)
                    if lane == 0:
                        fx.ptr_store(fx.Int32(SHARED_EXPERT), route + (8 * k + TOP_K))
                        fx.ptr_store(fx.Float32(shared_weight), rwt + (8 * k + TOP_K))

            # a8w4 operands in xs as Int32 words, never f32: MXFP8 xn or mid, then E8M0 scales
            XN8_WORDS = HIDDEN // 4
            XSC_WORDS = HIDDEN // 32
            XSC_LDS = tokens * XN8_ROW
            MID16_WORDS = MOE_SLOTS * INTER // 2  # bf16 pairs, as the mids land
            MID8_WORDS = MOE_SLOTS * INTER // 4
            MID8_LDS = tokens * MID16_WORDS
            MIDSC_WORDS = MOE_SLOTS * INTER // 32
            MIDSC_LDS = MID8_LDS + tokens * MID8_WORDS

            def mx8_scale(vals, offsets):
                """E8M0 byte (>= 1) of the 32-block the lanes xor-``offsets`` apart share,
                aiter's MXFP8 RoundUp rule ceil_pow2(amax / 448), and its reciprocal."""
                amax = fx.max(vals[0], -vals[0])
                for v in vals[1:]:
                    amax = fx.max(amax, fx.max(v, -v))
                amax = butterfly(amax, offsets, fx.max)
                bits = (amax * fx.Float32(1.0 / FP8_MAX)).bitcast(fx.Int32)
                e = (bits >> 23) & 0xFF
                e = e + ((bits & 0x7FFFFF) != 0).select(fx.Int32(1), fx.Int32(0))
                e = fx.max(fx.min(e, fx.Int32(254)), fx.Int32(1))
                return e, hw_rcp((e << 23).bitcast(fx.Float32))

            def mx8_q(v, inv):
                return fx.max(
                    fx.min(v * inv, fx.Float32(FP8_MAX)), fx.Float32(-FP8_MAX)
                )

            def lds_b8(chunk_word):
                """This lane's B operand of the 128-k chunk at LDS word ``chunk_word``:
                fp8 k 16 g .. + 16 then 64 + 16 g .. + 16 (g = lane // 16), the
                f8f6f4 MFMA's B layout."""
                h = [
                    fx.ptr_load(
                        xs + (chunk_word + 16 * j + 4 * g4), result_type=v4f
                    ).bitcast(fx.Int32)
                    for j in range(2)
                ]
                return fx.Vector.from_elements(
                    [h[j][e] for j in range(2) for e in range(4)], fx.Int32
                )

            def bf16x2(word):
                return fx.Vector.from_elements([word], fx.Int32).bitcast(fx.BFloat16)

            def lds_i32(word):
                return fx.ptr_load(xs + word).bitcast(fx.Int32)

            def mfma_f4f8(w4, sa, b8, sb, c, sa_byte=None):
                """16x16x128 scaled MFMA: A = this lane's 32 fp4 (row l % 16, k block
                l / 16) with E8M0 ``sa``, B = its 32 fp8 with E8M0 ``sb``. ``sa_byte``:
                ``sa`` is a scale dword, the MFMA takes that byte of it."""
                if const_expr(sa_byte is None):
                    sa, sa_byte = sa & 0xFF, 0
                return fx.Vector(
                    rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        T.vec(4, T.f32), [w4, b8, c, 4, 0, sa_byte, sa, 0, sb]
                    )
                )

            # ============================================================ 4. router
            def stage_router():
                r_g = rsrc(g_post)
                r_gate = rsrc(w_gate)
                # 12 consecutive elements a thread: its poll is 3 specs, every thread busy
                RE = HIDDEN // THREADS
                rw0 = tid * (RE // 2)

                def ld_bf16x12(r, word, cm=0):
                    a8 = fx.Vector(
                        bo.buffer_load(
                            r, word, vec_width=4, dtype=T.i32, cache_modifier=cm
                        )
                    ).bitcast(fx.BFloat16)
                    a4 = fx.Vector(
                        bo.buffer_load(
                            r, word + 4, vec_width=2, dtype=T.i32, cache_modifier=cm
                        )
                    ).bitcast(fx.BFloat16)
                    return [a8[e] for e in range(8)] + [a4[e] for e in range(4)]

                # a task per (token, 8 experts): the tokens' norms run in parallel, not in series
                for tt in range(start("router"), N_ROUTER * tokens, G):
                    tt = fx.Int32(tt)
                    k = tt // N_ROUTER
                    t = tt % N_ROUTER
                    e_r = t * WAVES + wave
                    bias_r = fx.Float32(
                        bo.buffer_load(rsrc(bias), e_r, vec_width=1, dtype=T.f32)
                    )
                    gw = [  # bf16 pairs
                        fx.Vector(
                            bo.buffer_load(
                                r_gate,
                                (e_r * HIDDEN + (i * 64 + lane) * 8) // 2,
                                vec_width=4,
                                dtype=T.i32,
                            )
                        )
                        for i in range(HIDDEN // 512)
                    ]
                    gp = ld_bf16x12(r_g, rw0)
                    # the residual (K1's norm / GEMV task)
                    if wave == 0:
                        k1_poll([(rdone_mb, k, 1)])
                    gpu.barrier()
                    acquire()
                    hres = ld_bf16x12(rsrc(h_in), k * (HIDDEN // 2) + rw0, CM_K1)
                    # the all-reduced row: the o tasks' ``a``, or with AG_RS the owners' a_ag
                    a_src = sy(sym, "a_ag") if AG_RS else mb("a")
                    arow = mb_poll(
                        [
                            (a_src, (k * HIDDEN + tid * RE) // 2 + 2 * h, 2)
                            for h in range(RE // 4)
                        ],
                    )
                    v = [
                        f + fx.Float32(hres[4 * h + 2 * qq + i])
                        for h in range(RE // 4)
                        for qq in range(2)
                        for i, f in enumerate(bf2_f32(arow[h][qq]))
                    ]
                    if const_expr(AG_RS):  # noqa: SIM102
                        # h_mid = bf16(bf16(sum) + residual): task t writes slice t
                        if tid // (XN_SLICE // RE) == t:
                            hw = (k * HIDDEN + tid * RE) // 2
                            hb = fx.Vector.from_elements(v, fx.Float32).to(fx.BFloat16)
                            hi = hb.bitcast(fx.Int32)
                            bo.buffer_store(
                                fx.Vector.from_elements(
                                    [hi[e] for e in range(4)], fx.Int32
                                ),
                                rsrc(h_mid),
                                hw,
                            )
                            bo.buffer_store(
                                fx.Vector.from_elements([hi[4], hi[5]], fx.Int32),
                                rsrc(h_mid),
                                hw + 4,
                            )
                    # sum of squares: a wave sum each, one barrier, every lane adds the 8
                    ssw = fx.Float32(0.0)
                    for e in range_constexpr(RE):
                        ssw = fmath.fma(v[e], v[e], ssw)
                    ssw = wave_sum(ssw)
                    if lane == 0:
                        fx.ptr_store(ssw, red + wave)
                    gpu.barrier()
                    ss = fx.ptr_load(red)
                    for w in range_constexpr(1, WAVES):
                        ss = ss + fx.ptr_load(red + w)
                    rstd = hw_rsq(ss / float(HIDDEN) + eps)
                    for j in range_constexpr(RE // 2):
                        fx.ptr_store(
                            bf16_pair(
                                v[2 * j] * rstd * (fx.Float32(gp[2 * j]) + 1.0),
                                v[2 * j + 1] * rstd * (fx.Float32(gp[2 * j + 1]) + 1.0),
                            ),
                            xs + (rw0 + j),
                        )
                    gpu.barrier()
                    if tid < XN_SLICE // 4:  # 4 elements a lane, 8 lanes a 32-block
                        w0 = t * (XN_SLICE // 2) + tid * 2
                        f = [
                            v
                            for j in range(2)
                            for v in bf2_f32(
                                fx.ptr_load(xs + (w0 + j)).bitcast(fx.Int32)
                            )
                        ]
                        e, inv = mx8_scale(f, (1, 2, 4))
                        word = fp8_pack4(*[mx8_q(v, inv) for v in f])
                        xw = k * XN8_WORDS + t * (XN_SLICE // 4) + tid
                        xsw = k * XSC_WORDS + t * (XN_SLICE // 32) + tid // 8
                        if const_expr(WIDE):
                            # plain words, flagged once the task is done (below)
                            bo.buffer_store(
                                word, rsrc(mb("xn8p")), xw, cache_modifier=CM_DEV
                            )
                            if tid % 8 == 0:
                                bo.buffer_store(
                                    e, rsrc(mb("xscp")), xsw, cache_modifier=CM_DEV
                                )
                        else:
                            mb_put_words(mb("xn8"), xw, [word])
                            if tid % 8 == 0:
                                mb_put_words(mb("xsc"), xsw, [e])
                    # 4 accumulators: 12-long dot2 chains, not one 96-long FMA chain after widening
                    accs = [fx.Float32(0.0) for _ in range(4)]
                    for i in range_constexpr(HIDDEN // 512):
                        xv = fx.ptr_load(
                            xs + (i * 64 + lane) * 4, result_type=v4f
                        ).bitcast(fx.Int32)
                        for j in range_constexpr(4):
                            accs[j] = fx.Float32(
                                rocdl.fdot2_f32_bf16(
                                    T.f32,
                                    as_ir_value(bf16x2(gw[i][j])),
                                    as_ir_value(bf16x2(xv[j])),
                                    as_ir_value(accs[j]),
                                    clamp=False,
                                ).result
                            )
                    logit = bf16_round(
                        wave_sum((accs[0] + accs[1]) + (accs[2] + accs[3]))
                    )
                    # key and sigmoid go out, not the logit: the consumers start at the max rounds
                    sc = hw_rcp(1.0 + hw_exp2(-logit * LOG2E))
                    rk = route_key(sc + bias_r, e_r)
                    if lane < SCORE_COPIES:
                        mb_put_words(
                            scores_copy(lane, k), e_r * 2, [rk, sc.bitcast(fx.Int32)]
                        )
                    if const_expr(WIDE):
                        # the xn slice's flag, after the logits rather than in front of them
                        release_stores()
                        gpu.barrier()
                        if tid == 0:
                            mb_put(mb("xndone"), tt, fx.Int32(1))

            stage_router()

            # ================================ 5. experts: up / gate -> down -> FFN reduce
            UG_KC = HIDDEN // 128  # 128-k chunks
            UG_CPW = UG_KC // WAVES  # 128-k chunks per wave: each wave a k eighth
            DN_KC = INTER // 128  # 128-k chunks of one slot
            DN_UNITS = TOP_K * DN_KC  # routed (slot, 128-k chunk) units per row group
            DN_UPW = (DN_UNITS + 3) // 4  # 4 waves per row group
            SH_UPW = (DN_KC + 3) // 4  # the shared expert's chunks, once for all tokens
            # shared-expert tasks run during routing (stage "shared"): routed ones set the rounds
            ROUTED_TASKS = TOP_K * UG_PER_SLOT * tokens
            TPT = TOP_K * UG_PER_SLOT  # routed tasks per token
            CPT = ug_ctas_per_token(tokens)
            UG_ITERS = max(ug_tasks_of(c, tokens) for c in range(G))  # per CTA, at most

            def ug_task_id(it):
                """(this CTA's it-th routed task, clamped; whether it exists)."""
                if const_expr(CPT is not None):
                    u = bid % CPT + it * CPT
                    return (bid // CPT) * TPT + fx.min(u, TPT - 1), u < TPT
                t = start("ug") + it * G
                return fx.min(t, ROUTED_TASKS - 1), t < ROUTED_TASKS

            def ug_task(t):
                """(shared expert?, token, slot, row group) of up / gate task t."""
                shared = t >= ROUTED_TASKS
                r = fx.min(t, ROUTED_TASKS - 1)
                tk = r // (TOP_K * UG_PER_SLOT)
                slot = shared.select(
                    fx.Int32(TOP_K), (r % (TOP_K * UG_PER_SLOT)) // UG_PER_SLOT
                )
                return shared, tk, slot, t % UG_PER_SLOT

            def ug_load(t):
                """Up / gate weights and scales of task t."""
                shared, tk, slot, cg = ug_task(t)
                # the bound keeps a corrupt route from addressing past the expert table
                e_u = shared.select(
                    fx.Int32(SHARED_EXPERT),
                    fx.min(
                        uniform(fx.ptr_load(route + (8 * tk + slot))), SHARED_EXPERT
                    ),
                )
                r_w = rsrc(w13 + fx.Int64(e_u) * W13_BYTES)
                r_s = rsrc(s13)
                wts = []  # [gate, up][chunk]
                scs = []
                for gu in range_constexpr(2):
                    rg = gu * UG_PER_SLOT + cg
                    wts.append([])
                    scs.append([])
                    for i in range_constexpr(UG_CPW):
                        kc = wave * UG_CPW + i
                        wts[gu].append(
                            fx.Vector(
                                bo.buffer_load(
                                    r_w,
                                    ((rg * (HIDDEN // 64) + kc * 2) * 512 + lane * 16)
                                    // 4,
                                    vec_width=4,
                                    dtype=T.i32,
                                )
                            )
                        )
                        scs[gu].append(
                            e8m0_load(
                                r_s,
                                group_scale_byte(e_u, W13_ROWS, rg, kc, W13_SCALE_COLS),
                            )
                        )
                return wts, scs

            def ug_compute(t, wts, scs):
                """Up / gate GEMV of task t -> swiglu -> its 16 mid values (of every token
                for the shared expert: B column l % 16 is token l % 16)."""
                shared, tk, slot, cg = ug_task(t)
                col = shared.select(fx.min(l16, tokens - 1), tk)
                cg_ = fx.Vector.filled(4, 0.0, fx.Float32)
                cu_ = fx.Vector.filled(4, 0.0, fx.Float32)
                # every chunk's B scale up front: read at use, each MFMA waits out an LDS latency
                sbs = [
                    lds_i32(XSC_LDS + col * XSC_ROW + (wave * UG_CPW + i) * 4 + g4)
                    for i in range(UG_CPW)
                ]
                for i in range_constexpr(UG_CPW):
                    kc = wave * UG_CPW + i
                    xb = lds_b8(col * XN8_ROW + kc * 32)
                    cg_ = mfma_f4f8(wts[0][i], scs[0][i], xb, sbs[i], cg_)
                    cu_ = mfma_f4f8(wts[1][i], scs[1][i], xb, sbs[i], cu_)
                # gate partials in red, up partials in opart (free once the split is done)
                fx.ptr_store(cg_, red + (wave * 64 + lane) * 4)
                fx.ptr_store(cu_, opart + (wave * 64 + lane) * 4)
                gpu.barrier()
                # row tid % 16; a shared task's thread tid reads token tid / 16's column
                if tid < shared.select(fx.Int32(UG_ROWS * tokens), fx.Int32(UG_ROWS)):
                    r = tid % UG_ROWS
                    col = tid // UG_ROWS
                    kk = shared.select(col, tk)
                    gsum = fx.Float32(0.0)
                    usum = fx.Float32(0.0)
                    for w in range_constexpr(WAVES):
                        gsum = gsum + fx.ptr_load(
                            red + ((w * 64 + 16 * (r // 4) + col) * 4 + r % 4)
                        )
                        usum = usum + fx.ptr_load(
                            opart + ((w * 64 + 16 * (r // 4) + col) * 4 + r % 4)
                        )
                    y = swiglu_mul_batch([gsum], [usum], fx.Float32(-swiglu_limit))[0]
                    mb_put(
                        mb("mid"),
                        (kk * MOE_SLOTS + slot) * INTER + cg * UG_ROWS + r,
                        bf16_round(y),
                    )
                gpu.barrier()

            def sh_load(j, has=None):
                """The shared expert's half task j: gate rows 8 j .. + 8 on lanes l16 < 8,
                the up rows alike on the rest -- one MFMA tile; the waves split k."""
                row = (l16 < 8).select(SH_ROWS * j + l16, INTER + SH_ROWS * j + l16 - 8)
                rg = row // 16
                if const_expr(has is None):
                    r_w = rsrc(w13 + fx.Int64(SHARED_EXPERT) * W13_BYTES)
                    r_s = rsrc(s13)
                else:  # a CTA without the task reads through zero-sized buffers
                    r_w = rsrc(
                        w13 + fx.Int64(SHARED_EXPERT) * W13_BYTES,
                        has.select(fx.Int32(W13_BYTES), fx.Int32(0)),
                    )
                    r_s = rsrc(s13, has.select(fx.Int32(S13_BYTES), fx.Int32(0)))
                wts = []
                scs = []
                # the row's part of scale_index, once: a chunk adds 256 (kc / 2) + 2 (kc % 2)
                sc_row = scale_index(SHARED_EXPERT * W13_ROWS + row, g4, W13_SCALE_COLS)
                for i in range_constexpr(UG_CPW):
                    kc = wave * UG_CPW + i
                    wts.append(
                        fx.Vector(
                            bo.buffer_load(
                                r_w,
                                (
                                    (rg * (HIDDEN // 64) + kc * 2) * 512
                                    + (g4 * 16 + row % 16) * 16
                                )
                                // 4,
                                vec_width=4,
                                dtype=T.i32,
                            )
                        )
                    )
                    scs.append(e8m0_load(r_s, sc_row + kc // 2 * 256 + kc % 2 * 2))
                return wts, scs

            def sh_compute(j, wts, scs):
                """Half task j's GEMV (B column l % 16 = token l % 16) -> swiglu -> the
                shared slot's mid rows 8 j .. + 8 of every token."""
                col = fx.min(l16, tokens - 1)
                c = fx.Vector.filled(4, 0.0, fx.Float32)
                sbs = [
                    lds_i32(XSC_LDS + col * XSC_ROW + (wave * UG_CPW + i) * 4 + g4)
                    for i in range(UG_CPW)
                ]
                for i in range_constexpr(UG_CPW):
                    kc = wave * UG_CPW + i
                    c = mfma_f4f8(
                        wts[i], scs[i], lds_b8(col * XN8_ROW + kc * 32), sbs[i], c
                    )
                fx.ptr_store(c, red + (wave * 64 + lane) * 4)
                gpu.barrier()
                # mid row r = tid % 8 of token tid / 8: gate in tile row r, up in r + 8
                if tid < SH_ROWS * tokens:
                    r = tid % SH_ROWS
                    col = tid // SH_ROWS
                    gsum = fx.Float32(0.0)
                    usum = fx.Float32(0.0)
                    for w in range_constexpr(WAVES):
                        gsum = gsum + fx.ptr_load(
                            red + ((w * 64 + 16 * (r // 4) + col) * 4 + r % 4)
                        )
                        usum = usum + fx.ptr_load(
                            red + ((w * 64 + 16 * (r // 4 + 2) + col) * 4 + r % 4)
                        )
                    y = swiglu_mul_batch([gsum], [usum], fx.Float32(-swiglu_limit))[0]
                    mb_put(
                        mb("mid"),
                        (col * MOE_SLOTS + TOP_K) * INTER + j * SH_ROWS + r,
                        bf16_round(y),
                    )
                gpu.barrier()

            def down_load(k, td, has_dn):
                """Token k's route-weighted down units of row group td; a CTA without a
                down task reads through zero-sized buffers."""
                dn_bytes = has_dn.select(fx.Int32(W2_BYTES), fx.Int32(0))
                dn_scale_bytes = has_dn.select(fx.Int32(S2_BYTES), fx.Int32(0))
                rg_d = td * 2 + wave // 4
                units = []
                for i in range_constexpr(DN_UPW):
                    j = (wave % 4) + 4 * i
                    jj = fx.min(j, DN_UNITS - 1)
                    sl = jj // DN_KC
                    kc = jj % DN_KC
                    e_d = fx.min(
                        uniform(fx.ptr_load(route + (8 * k + sl))), SHARED_EXPERT
                    )
                    wv = fx.Vector(
                        bo.buffer_load(
                            rsrc(w2 + fx.Int64(e_d) * W2_BYTES, dn_bytes),
                            ((rg_d * (INTER // 64) + kc * 2) * 512 + lane * 16) // 4,
                            vec_width=4,
                            dtype=T.i32,
                        )
                    )
                    sc = e8m0_load(
                        rsrc(s2, dn_scale_bytes),
                        group_scale_byte(e_d, HIDDEN, rg_d, kc, W2_SCALE_COLS),
                    )
                    wgt = (j < DN_UNITS).select(
                        fx.ptr_load(rwt + (8 * k + sl)), fx.Float32(0.0)
                    )
                    units.append((wv, sc, sl, kc, wgt))
                return units

            def down_load_shared(td, has_dn):
                """The shared expert's down units of row group td (every token's)."""
                dn_bytes = has_dn.select(fx.Int32(W2_BYTES), fx.Int32(0))
                dn_scale_bytes = has_dn.select(fx.Int32(S2_BYTES), fx.Int32(0))
                rg_d = td * 2 + wave // 4
                units = []
                for i in range_constexpr(SH_UPW):
                    j = (wave % 4) + 4 * i
                    kc = fx.min(j, DN_KC - 1)
                    wv = fx.Vector(
                        bo.buffer_load(
                            rsrc(w2 + fx.Int64(SHARED_EXPERT) * W2_BYTES, dn_bytes),
                            ((rg_d * (INTER // 64) + kc * 2) * 512 + lane * 16) // 4,
                            vec_width=4,
                            dtype=T.i32,
                        )
                    )
                    sc = e8m0_load(
                        rsrc(s2, dn_scale_bytes),
                        group_scale_byte(
                            SHARED_EXPERT, HIDDEN, rg_d, kc, W2_SCALE_COLS
                        ),
                    )
                    wgt = (j < DN_KC).select(fx.Float32(shared_weight), fx.Float32(0.0))
                    units.append((wv, sc, kc, wgt))
                return units

            def ffn_finish(t):
                """misc (row group t's bf16 partial rows, token k's DN_ROWS from k DN_ROWS)
                -> every peer -> rank-ordered sum -> ar_out."""
                row_tok = tid // (DN_ROWS // 2)
                row_pair = tid % (DN_ROWS // 2)
                if const_expr(W > 1):
                    # with AG_RS only to the row group's owner rank
                    push_rows("ffn", t * DN_ROWS, DN_ROWS, t % W if AG_RS else None)
                    gpu.barrier()
                if tid < DN_ROWS // 2 * tokens:
                    row = t * DN_ROWS + row_pair * 2
                    if const_expr(W == 1):
                        t0 = fx.ptr_load(misc + tid * 2)
                        t1 = fx.ptr_load(misc + (tid * 2 + 1))
                    else:
                        own = sy(sym, "ffn")

                        def rank_sum():
                            parts = mb_poll(
                                [
                                    (
                                        own,
                                        ((src * MAX_TOKENS + row_tok) * HIDDEN + row)
                                        // 2,
                                        1,
                                    )
                                    for src in range(W)
                                ],
                            )
                            s0 = fx.Float32(0.0)
                            s1 = fx.Float32(0.0)
                            for src in range_constexpr(W):
                                f0, f1 = bf2_f32(parts[src][0])
                                s0 = s0 + f0
                                s1 = s1 + f1
                            return s0, s1

                        if const_expr(AG_RS):
                            t0 = fx.Float32(0.0)
                            t1 = fx.Float32(0.0)
                            if t % W == rank:  # the owner: sum, gather to every rank
                                t0, t1 = rank_sum()
                                for w in range_constexpr(W):
                                    mb_put_bf(
                                        sy(peer_addr(w), "ffn_ag"),
                                        row_tok * HIDDEN + row,
                                        [t0, t1],
                                    )
                            else:  # the owner's sums (bf16, as ar_out stores them)
                                got = mb_poll(
                                    [
                                        (
                                            sy(sym, "ffn_ag"),
                                            (row_tok * HIDDEN + row) // 2,
                                            1,
                                        )
                                    ],
                                )
                                t0, t1 = bf2_f32(got[0][0])
                        else:
                            t0, t1 = rank_sum()
                    bo.buffer_store(
                        fx.Vector.from_elements([t0, t1], fx.Float32).to(fx.BFloat16),
                        rsrc(ar_out),
                        row_tok * HIDDEN + row,
                    )

            # ======================================== 5w. experts, wide (S > 4)
            # pool: MXFP8 xn and scales, up partials, sigmoids; the down stage: mids and scales
            W_UPP = tokens * (XN8_ROW + XSC_ROW)
            W_SIG = W_UPP + WAVES * 64 * 4
            MID_ROWS = MOE_SLOTS * tokens  # the zero row's index
            MID_ROW_WORDS = INTER // 4
            MIDSC_ROW_WORDS = INTER // 32
            W_MIDSC = (MID_ROWS + 1) * MID_ROW
            # up / gate tasks go in pairs, a CTA runs both: an MXFP8 block is 32 rows
            UG_PAIRS = UG_PER_SLOT // 2
            # up / gate tasks a CTA runs
            UGW_ITERS = 2 * ((U_MAX * UG_PAIRS + G - 1) // G)
            DN_WITERS = (U_MAX * DN_KC + 3) // 4  # (expert, chunk) units a wave runs
            DN_DEPTH = 12  # units whose weights are in flight
            ROUTE_ITERS = (tokens + WAVES - 1) // WAVES  # tokens a wave routes

            def popc64(x):
                return fx.Int32(fx.ctpop(x))

            def slot_of(k, e):
                """Token k's slot routed to expert e (TOP_K: the shared expert)."""
                s = fx.Int32(TOP_K)
                for j in range_constexpr(TOP_K):
                    s = (fx.ptr_load(route + (8 * k + j)) == e).select(fx.Int32(j), s)
                return s

            def build_experts():
                """uexp / umask[0 .. n_u) := the tokens' distinct routed experts in
                expert order and their token masks, [n_u] := the shared expert (every
                token), uexp[U_MAX + 1] := n_u. The waves' masks must be in (synced)."""
                m = fx.Int32(0)
                if tid < N_ROUTED:
                    for w in range_constexpr(WAVES):
                        m = m | fx.ptr_load(xs + (W_UPP + w * N_ROUTED + tid))
                bal = ballot(m != 0)
                below = popc64(bal & ((fx.Int64(1) << fx.Int64(lane)) - 1))
                if tid == 0:
                    fx.ptr_store(popc64(bal), umask + (U_MAX + 1))
                gpu.barrier()
                if tid < N_ROUTED:
                    off = (wave == 0).select(
                        fx.Int32(0), fx.ptr_load(umask + (U_MAX + 1))
                    )
                    rank_ = off + below
                    if m != 0:
                        fx.ptr_store(tid, uexp + rank_)
                        fx.ptr_store(m, umask + rank_)
                    if tid == N_ROUTED - 1:
                        nu_ = off + popc64(bal)
                        fx.ptr_store(nu_, uexp + (U_MAX + 1))
                        fx.ptr_store(fx.Int32(SHARED_EXPERT), uexp + nu_)
                        fx.ptr_store(fx.Int32((1 << tokens) - 1), umask + nu_)
                gpu.barrier()

            def wug_load(t, has, scales=True):
                """Up / gate weights (and ``scales``: the scale dwords, which hold the
                pair's both halves) of task t = (distinct expert t / UG_PER_SLOT, row
                group t % UG_PER_SLOT); without the task, reads go through zero-sized
                buffers."""
                u = t // UG_PER_SLOT
                cg = t % UG_PER_SLOT
                e_u = fx.min(uniform(fx.ptr_load(uexp + u)), SHARED_EXPERT)
                r_w = rsrc(
                    w13 + fx.Int64(e_u) * W13_BYTES,
                    has.select(fx.Int32(W13_BYTES), fx.Int32(0)),
                )
                r_s = rsrc(s13, has.select(fx.Int32(S13_BYTES), fx.Int32(0)))
                wts = []
                scs = []
                for gu in range_constexpr(2):
                    rg = gu * UG_PER_SLOT + cg
                    wts.append([])
                    scs.append([])
                    for i in range_constexpr(UG_CPW):
                        kc = wave * UG_CPW + i
                        wts[gu].append(
                            fx.Vector(
                                bo.buffer_load(
                                    r_w,
                                    ((rg * (HIDDEN // 64) + kc * 2) * 512 + lane * 16)
                                    // 4,
                                    vec_width=4,
                                    dtype=T.i32,
                                )
                            )
                        )
                        if const_expr(scales and i % 2 == 0):
                            # scale dword: chunks kc, kc + 1 x both 16-row halves (byte 2 d + a)
                            scs[gu].append(
                                fx.Int32(
                                    bo.buffer_load(
                                        r_s,
                                        group_scale_byte(
                                            e_u, W13_ROWS, rg, kc, W13_SCALE_COLS
                                        )
                                        // 4,
                                        vec_width=1,
                                        dtype=T.i32,
                                    )
                                )
                            )
                return wts, scs

            def wug_y(wts, scs, half):
                """A task's up / gate GEMV for every token (B column l % 16 = token
                l % 16) -> swiglu: thread (row r, token k)'s mid value. ``half``: the
                task's 16-row half of its 32-row scale block (compile time)."""
                col = fx.min(l16, tokens - 1)
                cg_ = fx.Vector.filled(4, 0.0, fx.Float32)
                cu_ = fx.Vector.filled(4, 0.0, fx.Float32)
                sbs = [
                    lds_i32(XSC_LDS + col * XSC_ROW + (wave * UG_CPW + i) * 4 + g4)
                    for i in range(UG_CPW)
                ]
                for i in range_constexpr(UG_CPW):
                    kc = wave * UG_CPW + i
                    xb = lds_b8(col * XN8_ROW + kc * 32)
                    sel = (i % 2) * 2 + half
                    cg_ = mfma_f4f8(wts[0][i], scs[0][i // 2], xb, sbs[i], cg_, sel)
                    cu_ = mfma_f4f8(wts[1][i], scs[1][i // 2], xb, sbs[i], cu_, sel)
                fx.ptr_store(cg_, red + (wave * 64 + lane) * 4)
                fx.ptr_store(cu_, fpool + (W_UPP + (wave * 64 + lane) * 4))
                gpu.barrier()
                r = tid % UG_ROWS
                k = fx.min(tid // UG_ROWS, tokens - 1)
                gsum = fx.Float32(0.0)
                usum = fx.Float32(0.0)
                for w in range_constexpr(WAVES):
                    gsum = gsum + fx.ptr_load(
                        red + ((w * 64 + 16 * (r // 4) + k) * 4 + r % 4)
                    )
                    usum = usum + fx.ptr_load(
                        fpool + (W_UPP + (w * 64 + 16 * (r // 4) + k) * 4 + r % 4)
                    )
                gpu.barrier()  # red and the up partials are free for the next task
                return swiglu_mul_batch([gsum], [usum], fx.Float32(-swiglu_limit))[0]

            def fp8_lanes4(q):
                """Lane r (r % 4 == 0): the E4M3 dword of lanes r .. r + 3's values."""
                lo = fx.Int32(
                    rocdl.cvt_pk_fp8_f32(T.i32, q, xshfl(q, 1), fx.Int32(0), False)
                )
                return lo | (xshfl(lo, 2) << 16)

            def wug_publish(pr, y_lo, y_hi):
                """Task pair pr = (distinct expert pr / UG_PAIRS, 32-row block
                pr % UG_PAIRS): its mid rows (thread (r, k) holds rows r and 16 + r of
                the block) as the down stage's MXFP8 -- the rounding its bf16 staging
                did, one 32-block a 16-lane group -- then the pair's flag."""
                u = pr // UG_PAIRS
                blk_ = pr % UG_PAIRS
                r = tid % UG_ROWS
                k = fx.min(tid // UG_ROWS, tokens - 1)
                lo = bf16_round(y_lo)
                hi = bf16_round(y_hi)
                sc, inv = mx8_scale([lo, hi], (1, 2, 4, 8))
                words = [fp8_lanes4(mx8_q(v, inv)) for v in (lo, hi)]
                e_u = fx.ptr_load(uexp + u)
                m_u = fx.ptr_load(umask + u)
                row = k * MOE_SLOTS + slot_of(k, e_u)
                if (tid < UG_ROWS * tokens) & (((m_u >> k) & 1) != 0):
                    if r % 4 == 0:
                        for h in range_constexpr(2):
                            bo.buffer_store(
                                words[h],
                                rsrc(mb("mid8p")),
                                row * MID_ROW_WORDS + blk_ * 8 + h * 4 + r // 4,
                                cache_modifier=CM_DEV,
                            )
                    if r == 0:
                        bo.buffer_store(
                            sc,
                            rsrc(mb("midsp")),
                            row * MIDSC_ROW_WORDS + blk_,
                            cache_modifier=CM_DEV,
                        )
                release_stores()
                gpu.barrier()
                if tid == 0:
                    mb_put(mb("ugdone"), pr, fx.Int32(1))

            def down_tables():
                """urow / uwt := per (distinct expert, B column) the down stage's mid row
                (the zero row where the column's token is not routed there) and route
                weight. Routing only: built while the first up / gate weights land."""
                nu = uniform(fx.ptr_load(uexp + (U_MAX + 1)))
                for i in range_constexpr((U_MAX * 16 + THREADS - 1) // THREADS):
                    ix = tid + THREADS * i
                    if ix < U_MAX * 16:
                        xu = ix // 16
                        xk = ix % 16
                        xe = fx.ptr_load(uexp + fx.min(xu, nu))
                        xm = fx.ptr_load(umask + fx.min(xu, nu))
                        xt = fx.min(xk, tokens - 1)
                        xs_ = slot_of(xt, xe)
                        xh = (((xm >> xk) & 1) != 0) & (xu <= nu)
                        fx.ptr_store(
                            xh.select(xt * MOE_SLOTS + xs_, fx.Int32(MID_ROWS)),
                            urow + ix,
                        )
                        fx.ptr_store(
                            xh.select(
                                fx.ptr_load(rwt + (8 * xt + xs_)), fx.Float32(0.0)
                            ),
                            uwt + ix,
                        )

            def stage_moe_wide():
                td = start("down")  # this CTA's down task (the split tasks come first)
                has_dn = td < N_DN
                td = fx.min(td, N_DN - 1)
                mine = bid % SCORE_COPIES
                # expert -> token masks in the up partials' words, free before the first up / gate
                wave_mask = xs + (W_UPP + wave * N_ROUTED)
                fx.ptr_store(fx.Int32(0), wave_mask + lane)
                fx.ptr_store(fx.Int32(0), wave_mask + (lane + 64))
                wave_toks = [
                    fx.min(wave + WAVES * j, tokens - 1) for j in range(ROUTE_ITERS)
                ]
                sg = mb_poll(
                    [
                        (scores_copy(mine, k), (lane + 64 * h) * 2, 2)
                        for k in wave_toks
                        for h in range(2)
                    ]
                )
                # a pad row (seq_len <= 0) joins no expert, so cannot reorder the live rows' sums
                r_seq = rsrc(seq_lens)
                for j in range_constexpr(ROUTE_ITERS):
                    if wave + WAVES * j < tokens:
                        tk_seq = uniform(
                            bo.buffer_load(
                                r_seq, wave_toks[j], vec_width=1, dtype=T.i32
                            )
                        )
                        route_top4(
                            wave_toks[j],
                            sg[2 * j][0],
                            sg[2 * j][1].bitcast(fx.Float32),
                            sg[2 * j + 1][0],
                            sg[2 * j + 1][1].bitcast(fx.Float32),
                            fpool + W_SIG,
                            wave_mask,
                            (tk_seq > 0).select(
                                fx.Int32(1) << wave_toks[j], fx.Int32(0)
                            ),
                        )
                gpu.barrier()
                build_experts()
                n_pairs = (uniform(fx.ptr_load(uexp + (U_MAX + 1))) + 1) * UG_PAIRS

                def task(it):
                    """Iteration it's task: pair bid + (it / 2) G's half it % 2."""
                    pr = fx.Int32(bid + it // 2 * G)
                    return fx.min(pr, n_pairs - 1) * 2 + it % 2, pr < n_pairs

                t0, h0 = task(0)
                cur = wug_load(t0, h0)  # task 0: a lo half, with the pair's scales
                down_tables()
                # the xn once every router task flagged its slice, under the first task's weights
                n_rt = N_ROUTER * tokens
                if wave == 0:
                    mb_poll(
                        [
                            (mb("xndone"), fx.min(lane + 64 * c, n_rt - 1), 1)
                            for c in range((n_rt + 63) // 64)
                        ]
                    )
                gpu.barrier()
                acquire()
                for base_w, n_row, lds_w, lds_row in (
                    (mb("xn8p"), XN8_WORDS, 0, XN8_ROW),
                    (mb("xscp"), XSC_WORDS, XSC_LDS, XSC_ROW),
                ):
                    n_v = tokens * n_row // 4
                    for i in range_constexpr((n_v + THREADS - 1) // THREADS):
                        v = tid + THREADS * i
                        if v < n_v:
                            fx.ptr_store(
                                fx.Vector(
                                    bo.buffer_load(
                                        rsrc(base_w),
                                        v * 4,
                                        vec_width=4,
                                        dtype=T.i32,
                                        cache_modifier=CM_DEV,
                                    )
                                ),
                                xs + (lds_w + v * 4 // n_row * lds_row + v * 4 % n_row),
                            )
                gpu.barrier()
                y_lo = fx.Float32(0.0)  # the pair's first half, until the second's
                for it in range_constexpr(UGW_ITERS):
                    t, has_t = task(it)
                    if const_expr(it + 1 < UGW_ITERS):
                        tn, hn = task(it + 1)
                        # a hi half reads no scales: its lo half's dwords hold them
                        nxt = wug_load(tn, hn, scales=(it + 1) % 2 == 0)
                    if const_expr(it % 2 == 0):
                        pair_scs = cur[1]
                    if has_t:
                        # a pair: row groups cg, cg + 1
                        y = wug_y(cur[0], pair_scs, it % 2)
                        if const_expr(it % 2 == 0):
                            y_lo = y
                        else:
                            wug_publish(t // 2, y_lo, y)
                    if const_expr(it + 1 < UGW_ITERS):
                        cur = nxt
                if has_dn:
                    wide_down(td)

            def wide_down(td):
                """Row group td's down GEMV over every distinct expert (B column l % 16 =
                token l % 16's mid for it, the zero row where the token is not routed
                there), route-weighted per column -> ffn_finish."""
                nu = uniform(fx.ptr_load(uexp + (U_MAX + 1)))
                rg_d = td * 2 + wave // 4
                n_units = (nu + 1) * DN_KC

                def unit_load(i):
                    """(weights, scale, unit) of this wave's i-th (expert, 128-k chunk)
                    unit, zero-sized reads past the last."""
                    j = wave % 4 + 4 * i
                    ok = j < n_units
                    # clamped so a spare unit reads a real or zero row: garbage B gives 0 * NaN
                    du = fx.min(j // DN_KC, U_MAX - 1)
                    dkc = j % DN_KC
                    de = fx.min(uniform(fx.ptr_load(uexp + du)), SHARED_EXPERT)
                    dwv = fx.Vector(
                        bo.buffer_load(
                            rsrc(
                                w2 + fx.Int64(de) * W2_BYTES,
                                ok.select(fx.Int32(W2_BYTES), fx.Int32(0)),
                            ),
                            ((rg_d * (INTER // 64) + dkc * 2) * 512 + lane * 16) // 4,
                            vec_width=4,
                            dtype=T.i32,
                        )
                    )
                    dsc = e8m0_load(
                        rsrc(s2, ok.select(fx.Int32(S2_BYTES), fx.Int32(0))),
                        group_scale_byte(de, HIDDEN, rg_d, dkc, W2_SCALE_COLS),
                    )
                    return dwv, dsc, du, dkc, ok

                def with_rows(u):
                    """A loaded unit plus its B row and route weight (the tables are in),
                    read ahead so the unit's B loads wait on one LDS trip."""
                    dwv, dsc, du, dkc, ok = u
                    drow = fx.ptr_load(urow + (du * 16 + l16))
                    dw = ok.select(fx.ptr_load(uwt + (du * 16 + l16)), fx.Float32(0.0))
                    return dwv, dsc, drow, dkc, dw, ok

                # first weights out before the mids land; unit i + DN_DEPTH's before unit i's MFMA
                acc = [fx.Float32(0.0) for _ in range(4)]
                inflight = [unit_load(i) for i in range(min(DN_DEPTH, DN_WITERS))]
                # every up / gate pair's flag, then the plain-word mid rows and scales
                n_ug = (nu + 1) * UG_PAIRS
                mb_poll(
                    [
                        (mb("ugdone"), fx.min(tid + THREADS * i, n_ug - 1), 1)
                        for i in range((U_MAX * UG_PAIRS + THREADS - 1) // THREADS)
                    ]
                )
                gpu.barrier()
                acquire()
                for base_w, n_row, lds_w, lds_row in (
                    (mb("mid8p"), MID_ROW_WORDS, 0, MID_ROW),
                    (mb("midsp"), MIDSC_ROW_WORDS, W_MIDSC, MIDSC_ROW),
                ):
                    n_v = MID_ROWS * n_row // 4
                    for i in range_constexpr((n_v + THREADS - 1) // THREADS):
                        v = tid + THREADS * i
                        if v < n_v:
                            fx.ptr_store(
                                fx.Vector(
                                    bo.buffer_load(
                                        rsrc(base_w),
                                        v * 4,
                                        vec_width=4,
                                        dtype=T.i32,
                                        cache_modifier=CM_DEV,
                                    )
                                ),
                                xs + (lds_w + v * 4 // n_row * lds_row + v * 4 % n_row),
                            )
                if tid < MID_ROW_WORDS:
                    fx.ptr_store(fx.Int32(0), xs + (MID_ROWS * MID_ROW + tid))
                if tid < MIDSC_ROW_WORDS:
                    fx.ptr_store(
                        fx.Int32(0), xs + (W_MIDSC + MID_ROWS * MIDSC_ROW + tid)
                    )
                gpu.barrier()
                for i in range_constexpr(DN_WITERS):
                    if const_expr(i == 0):  # the tables are in: the early units' rows
                        inflight = [with_rows(u) for u in inflight]
                    dwv, dsc, drow, dkc, dw, ok = inflight.pop(0)
                    if const_expr(i + DN_DEPTH < DN_WITERS):
                        inflight.append(with_rows(unit_load(i + DN_DEPTH)))
                    dcu = mfma_f4f8(
                        dwv,
                        dsc,
                        lds_b8(drow * MID_ROW + dkc * 32),
                        lds_i32(W_MIDSC + drow * MIDSC_ROW + dkc * 4 + g4),
                        fx.Vector.filled(4, 0.0, fx.Float32),
                    )
                    # a unit past the last adds nothing, whatever its MFMA gave
                    acc = [
                        acc[e] + ok.select(dcu[e] * dw, fx.Float32(0.0))
                        for e in range(4)
                    ]
                a0, a1, a2, a3 = acc
                fx.ptr_store(
                    fx.Vector.from_elements([a0, a1, a2, a3], fx.Float32),
                    red + (wave * 64 + lane) * 4,
                )
                gpu.barrier()
                if tid < DN_ROWS * tokens:
                    k = tid // DN_ROWS
                    grp = (tid % DN_ROWS) // 16
                    rr = tid % 16
                    tot = fx.Float32(0.0)
                    for w in range_constexpr(4):
                        tot = tot + fx.ptr_load(
                            red
                            + (((grp * 4 + w) * 64 + 16 * (rr // 4) + k) * 4 + rr % 4)
                        )
                    fx.ptr_store(bf16_round(tot), misc + tid)
                gpu.barrier()
                ffn_finish(td)

            def stage_moe():
                td = bid - BASE["down"]  # this CTA's down task
                has_dn = (td >= 0) & (td < N_DN)
                td = fx.min(fx.max(td, 0), N_DN - 1)
                has_sh = start("shared") < SH_TASKS
                n_xw = XN8_WORDS // THREADS

                def xn_specs(k):
                    return [
                        (mb("xn8"), k * XN8_WORDS + tid + i * THREADS, 1)
                        for i in range(n_xw)
                    ] + [(mb("xsc"), k * XSC_WORDS + fx.min(tid, XSC_WORDS - 1), 1)]

                def stage_xn(k, got):
                    for i in range_constexpr(n_xw):
                        fx.ptr_store(got[i][0], xs + (k * XN8_ROW + tid + i * THREADS))
                    if tid < XSC_WORDS:
                        fx.ptr_store(got[n_xw][0], xs + (XSC_LDS + k * XSC_ROW + tid))

                def fetch_xn(tok=None):
                    """Token tok's (default: every token's) MXFP8 xn and scales -> xs. Wave
                    0 first polls the last word of each router task's slice: every
                    thread of every CTA spinning on the whole of it holds the
                    shared-expert CTAs' xn back."""
                    gate_tok = (
                        fx.min(lane // N_ROUTER, tokens - 1) if tok is None else tok
                    )
                    if wave == 0:
                        mb_poll(
                            [
                                (
                                    mb("xn8"),
                                    gate_tok * XN8_WORDS
                                    + (lane % N_ROUTER + 1) * (XN_SLICE // 4)
                                    - 1,
                                    1,
                                )
                            ]
                        )
                    gpu.barrier()
                    if tok is None:
                        got = mb_poll([sp for k in range(tokens) for sp in xn_specs(k)])
                        for k in range_constexpr(tokens):
                            stage_xn(k, got[k * (n_xw + 1) : (k + 1) * (n_xw + 1)])
                    else:
                        stage_xn(tok, mb_poll(xn_specs(tok)))

                sh_late = shared_after_routing(tokens)
                if const_expr(sh_late):
                    sh_j = fx.min(start("shared"), SH_TASKS - 1)
                    sh_w = sh_load(sh_j, has_sh)
                    if has_sh:
                        fetch_xn()
                elif has_sh:  # the shared expert needs xn, not the routing
                    sh_j0 = start("shared")
                    sh_w0 = sh_load(sh_j0)
                    fetch_xn()
                    gpu.barrier()
                    sh_compute(sh_j0, sh_w0[0], sh_w0[1])
                if ug_task_id(0)[1] | has_dn:
                    # xn lands before the logits: stage it now if no shared task fetched it
                    if start("shared") >= SH_TASKS:
                        fetch_xn(bid // CPT if CPT is not None else None)
                    # wave k polls token k's router logits and routes it right away
                    mine = bid % SCORE_COPIES
                    for k in range_constexpr(tokens):
                        if wave == k:
                            got = mb_poll(
                                [
                                    (scores_copy(mine, k), (lane + 64 * h) * 2, 2)
                                    for h in range(2)
                                ]
                            )
                            route_top4(
                                k,
                                got[0][0],
                                got[0][1].bitcast(fx.Float32),
                                got[1][0],
                                got[1][1].bitcast(fx.Float32),
                                opart,
                            )
                    gpu.barrier()
                    cur = ug_load(ug_task_id(0)[0])
                    if const_expr(sh_late):  # noqa: SIM102
                        # while the first routed task's weights are in flight
                        if has_sh:
                            sh_compute(sh_j, sh_w[0], sh_w[1])
                    # task i + 1's weights go out before task i's GEMV; a spare load is clamped
                    for it in range_constexpr(UG_ITERS):
                        t, has_t = ug_task_id(it)
                        if const_expr(it + 1 < UG_ITERS):
                            nxt = ug_load(ug_task_id(it + 1)[0])
                        if has_t:
                            ug_compute(t, cur[0], cur[1])
                        if const_expr(it == 0):
                            # after task 0, not with its up / gate loads, which they would slow
                            dn_units = {0: down_load(0, td, has_dn)}
                            sh_units = down_load_shared(td, has_dn)
                        if const_expr(tokens > 1 and it == UG_ITERS - 1):
                            # two tokens in flight when the down GEMVs start
                            dn_units[1] = down_load(1, td, has_dn)
                        if const_expr(it + 1 < UG_ITERS):
                            cur = nxt
                    if has_dn:
                        moe_down(td, has_dn, dn_units, sh_units)

            def moe_down(t, has_dn, dn_units, sh_units):
                """Every token's route-weighted down GEMV of row group ``t`` -> push bf16
                partial rows -> rank-ordered sum -> ar_out."""

                def stage_mids(k):
                    """Token k's mid rows (every slot) -> xs, as bf16 pairs."""
                    mids = mb_poll(
                        [
                            (
                                mb("mid"),
                                k * MOE_SLOTS * INTER
                                + fx.min(
                                    tid * 2 + i * 2 * THREADS, MOE_SLOTS * INTER - 2
                                ),
                                2,
                            )
                            for i in range(4)
                        ]
                    )
                    for i in range_constexpr(4):
                        e = tid * 2 + i * 2 * THREADS
                        if e < MOE_SLOTS * INTER:
                            fx.ptr_store(
                                bf16_pair(
                                    mids[i][0].bitcast(fx.Float32),
                                    mids[i][1].bitcast(fx.Float32),
                                ),
                                xs + (k * (MOE_SLOTS * INTER // 2) + e // 2),
                            )

                # every token's mid in one go: they land together
                for k in range_constexpr(tokens):
                    stage_mids(k)
                gpu.barrier()
                # MXFP8 in its own pass: inside stage_mids it would serialize behind each poll
                for k in range_constexpr(tokens):
                    if tid < MID8_WORDS // 2:  # 8 elements a lane, 4 lanes a 32-block
                        w = fx.ptr_load(
                            xs + (k * MID16_WORDS + tid * 4), result_type=v4f
                        ).bitcast(fx.Int32)
                        f = [v for j in range(4) for v in bf2_f32(w[j])]
                        sc, inv = mx8_scale(f, (1, 2))
                        qv = [mx8_q(v, inv) for v in f]
                        fx.ptr_store(
                            fp8_pack4(*qv[:4]),
                            xs + (MID8_LDS + k * MID8_WORDS + tid * 2),
                        )
                        fx.ptr_store(
                            fp8_pack4(*qv[4:]),
                            xs + (MID8_LDS + k * MID8_WORDS + tid * 2 + 1),
                        )
                        if tid % 4 == 0:
                            fx.ptr_store(
                                sc, xs + (MIDSC_LDS + k * MIDSC_WORDS + tid // 4)
                            )
                gpu.barrier()
                # shared expert once, B column l16 = token l16 -> opart (free after the split)
                col_tok = fx.min(l16, tokens - 1)
                sh = [fx.Float32(0.0) for _ in range(4)]
                for wv, sc, kc, wgt in sh_units:
                    cu = mfma_f4f8(
                        wv,
                        sc,
                        lds_b8(
                            MID8_LDS
                            + col_tok * MID8_WORDS
                            + (TOP_K * INTER + kc * 128) // 4
                        ),
                        lds_i32(
                            MIDSC_LDS
                            + col_tok * MIDSC_WORDS
                            + (TOP_K * INTER + kc * 128) // 32
                            + g4
                        ),
                        fx.Vector.filled(4, 0.0, fx.Float32),
                    )
                    sh = [sh[j4] + cu[j4] * wgt for j4 in range(4)]
                fx.ptr_store(
                    fx.Vector.from_elements(sh, fx.Float32),
                    opart + (wave * 64 + lane) * 4,
                )
                # all tokens' GEMVs first, partials past the shared ones: one barrier, one reduce
                for k in range_constexpr(tokens):
                    # token k + 2's weights go out before token k's GEMV
                    if const_expr(k + 2 < tokens):
                        dn_units[k + 2] = down_load(k + 2, t, has_dn)
                    acc = [fx.Float32(0.0) for _ in range(4)]
                    for wv, sc, sl, kc, wgt in dn_units.pop(k):
                        cu = mfma_f4f8(
                            wv,
                            sc,
                            lds_b8(
                                MID8_LDS + k * MID8_WORDS + (sl * INTER + kc * 128) // 4
                            ),
                            lds_i32(
                                MIDSC_LDS
                                + k * MIDSC_WORDS
                                + (sl * INTER + kc * 128) // 32
                                + g4
                            ),
                            fx.Vector.filled(4, 0.0, fx.Float32),
                        )
                        acc = [acc[j4] + cu[j4] * wgt for j4 in range(4)]
                    fx.ptr_store(
                        fx.Vector.from_elements(acc, fx.Float32),
                        opart + ((k + 1) * WAVES * 64 + wave * 64 + lane) * 4,
                    )
                gpu.barrier()
                if tid < DN_ROWS * tokens:
                    k = tid // DN_ROWS
                    grp = (tid % DN_ROWS) // 16
                    rr = tid % 16
                    tot = fx.Float32(0.0)
                    for w in range_constexpr(4):
                        tot = tot + fx.ptr_load(
                            opart
                            + (
                                (
                                    (k + 1) * WAVES * 64
                                    + (grp * 4 + w) * 64
                                    + 16 * (rr // 4)
                                )
                                * 4
                                + rr % 4
                            )
                        )
                    for w in range_constexpr(4):
                        tot = tot + fx.ptr_load(
                            opart
                            + (((grp * 4 + w) * 64 + 16 * (rr // 4) + k) * 4 + rr % 4)
                        )
                    fx.ptr_store(bf16_round(tot), misc + tid)
                gpu.barrier()
                ffn_finish(t)

            (stage_moe_wide if WIDE else stage_moe)()

    @flyc.jit
    def launch_post_attn(
        h_in: Int64,
        q: Int64,
        block_table: Int64,
        seq_lens: Int64,
        k_cache: Int64,
        v_cache: Int64,
        k_scale: Int64,
        v_scale: Int64,
        w_o: Int64,
        s_o: Int64,
        g_post: Int64,
        w_gate: Int64,
        bias: Int64,
        w13: Int64,
        s13: Int64,
        w2: Int64,
        s2: Int64,
        h_mid: Int64,
        ar_out: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        step: Int64,
        rank: Int32,
        layer: Int32,
        bt_width: Int32,
        q_len: Int32,
        ar_in: Int64,
        g_in: Int64,
        w_qkv: Int64,
        s_qkv: Int64,
        g_q: Int64,
        g_k: Int64,
        g_iq: Int64,
        g_ik: Int64,
        cos_sin: Int64,
        index_cache: Int64,
        iq_out: Int64,
        scratch1: Int64,
        positions: Int64,
        slot_mapping: Int64,
        res: Int64,
        stream: fx.Stream = _CURRENT_STREAM,
    ):
        _ = keyed
        post_attn_kernel(
            h_in,
            q,
            block_table,
            seq_lens,
            k_cache,
            v_cache,
            k_scale,
            v_scale,
            w_o,
            s_o,
            g_post,
            w_gate,
            bias,
            w13,
            s13,
            w2,
            s2,
            h_mid,
            ar_out,
            scratch,
            sym,
            peers,
            step,
            rank,
            layer,
            bt_width,
            q_len,
            ar_in,
            g_in,
            w_qkv,
            s_qkv,
            g_q,
            g_k,
            g_iq,
            g_ik,
            cos_sin,
            index_cache,
            iq_out,
            scratch1,
            positions,
            slot_mapping,
            res,
        ).launch(grid=(G,), block=(THREADS,), stream=stream)

    return launch_post_attn
