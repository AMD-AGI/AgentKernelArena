# SPDX-License-Identifier: Apache-2.0
"""K1: one Muse Spark layer's attention front, the normed input through the KV
cache write, for S = ``tokens`` rows (a token each; 8 consecutive rows a request)::

    x (bf16: the layer's attention input, the previous boundary's pre-norm)
      -> q | k | v | gate GEMV on the bf16 weight (fp32 MFMA accumulate, one
         bf16 rounding, as the split-K GEMM of ``qkv_proj_qk_norm_rope_cache``)
      -> q, k: per-head RMSNorm without weight (eps 1e-5, bf16); q then scaled by
         qk_scale_factor and rounded; then interleaved RoPE in the layers that
         have one, with bf16 cos / sin and each product rounded, as the fused
         kernel of the ROCm model does
      -> q, k, v, gate out (bf16); K / V -> the layer's page-16 NHD fp8 cache
         (unit KV scale, checked at bind; the sliding layers' slots through the
         SWA pool's full-to-SWA table)

Stage 1 is the GEMV tasks: 16 row groups of 144 rows (nine MFMA tiles) x 16 K
parts = 256 tasks, exactly one a CTA; a task streams 147 KB of weights (held in
registers over the token tiles, one token tile at a time) and S x 1 KB of x (its K
sixteenth of every token, straight from memory), its fp32 partials go to slot
``part`` of the ``k1_part`` mailbox (a pair a (part, token, row)). Stage 2 is a head
task a q, k or v head on CTAs ``HT0`` .. and a gate task a q head after them, each
summing a row's sixteen slots in order (one poll round) and rounding once, 32
tokens a pass on 16 lanes of 4 dims. Dependencies only point
from stage 2 to stage 1 and all CTAs are co-resident, so every wait makes
progress.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from . import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from .mx import clamp_fp8
from .ops import (
    CM_DEV,
    CM_NT,
    bf2_f32,
    bf16_pair,
    bf16_round,
    butterfly,
    fp8_pack4,
    hw_rsq,
    mfma_bf16,
    row_sum,
    rsrc,
)
from .plan import BLOCKS, THREADS
from .config import (
    G_OFF,
    HEAD_DIM,
    HIDDEN,
    K_OFF,
    KV_K,
    LOCAL_KV_HEADS,
    LOCAL_Q_HEADS,
    MAX_TOKENS,
    Q_K,
    QK_SCALE,
    QKVG_ROWS,
    RMS_EPS,
    V_OFF,
)
from .device import poll_flags, stamp
from .layout import (
    ELEMS,
    G_ROWS,
    G_TILES,
    GATE_TASKS,
    GEMV_TASKS,
    GT0,
    HEAD_TASKS,
    HT0,
    K_PART,
    K_PARTS,
    K_PER_WAVE,
    K_STEP,
    K_STEPS,
    PARTS,
    QT0,
    RED_FLOATS,
    ROWS,
    SECOND_TASKS,
    TILE,
    TOKEN_LANES,
    TOKENS_PER_PASS,
    token_passes,
    token_tiles,
)


def gemv_k(lane, wave, kpart, kc):
    """The k of step ``kc`` this lane streams, in K part ``kpart`` (each wave a
    contiguous K_PER_WAVE range of the part)."""
    return kpart * K_PART + wave * K_PER_WAVE + kc * K_STEP + 8 * (lane // 16)


def gemv_w_loads(c, rg, kpart, w):
    """GEMV task (``rg``, ``kpart``)'s weight loads, this lane's: row tile ``it``'s
    row l16, K_STEPS steps of 16 B, in flight (72 VGPRs). Row-major bf16, read once
    a layer (nontemporal). ``[tile][step]``. A row group past the last reads past
    the buffer: zeros, no traffic (a CTA without the task)."""
    lane, wave = c["lane"], c["wave"]
    r_w = rsrc(w, QKVG_ROWS * HIDDEN * 2)
    return [
        [
            fx.Vector(
                bo.buffer_load(
                    r_w,
                    (
                        (rg * G_ROWS + it * ROWS + lane % 16) * HIDDEN
                        + gemv_k(lane, wave, kpart, kc)
                    )
                    // 2,
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=CM_NT,
                )
            )
            for kc in range(K_STEPS)
        ]
        for it in range(G_TILES)
    ]


@flyc.jit
def stage_gemv(c, rg, kpart, first=None):
    """GEMV task: K part ``kpart`` of row group ``rg`` (64 rows, four MFMA tiles) for
    every token tile. Lane l of wave w holds row l % 16 of each tile and, per
    k-step, k = 1024 kpart + 128 w + 32 kc + 8 (l / 16) .. + 8 of the weight (A) and
    of token 16 t + l % 16 (B, 16 B straight from memory); the waves' accumulators
    of each (row tile, token tile) are summed in wave order through LDS and the
    part's fp32 partials go to slot ``kpart`` of the mailbox. ``first``: the task's
    weights, already in flight (``gemv_w_loads``)."""
    S, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    red, put, part = c["red"], c["put"], c["part"]
    a = c["args"]
    tiles = token_tiles(S)
    r_x = rsrc(a["x"])
    x_cm = c["x_cm"]
    if const_expr(first is not None):
        w = first
    else:
        w = gemv_w_loads(c, rg, kpart, a["w"])
    if const_expr(c.get("defer", False)):
        stage_gemv_deferred(c, rg, kpart, w)
        return
    # a token tile at a time: the weights stay in registers, the accumulators are a
    # tile's nine; the waves' accumulators sum through a reduction row set a row tile
    for t in range_constexpr(tiles):
        tok = fx.min(t * TILE + lane % 16, S - 1)
        xb = [
            fx.Vector(
                bo.buffer_load(
                    r_x,
                    (tok * HIDDEN + gemv_k(lane, wave, kpart, kc)) // 2,
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=x_cm,
                )  # fmt: skip
            )
            for kc in range(K_STEPS)
        ]
        for it in range_constexpr(G_TILES):
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for kc in range_constexpr(K_STEPS):
                acc = mfma_bf16(
                    w[it][kc].bitcast(fx.BFloat16), xb[kc].bitcast(fx.BFloat16), acc
                )
            fx.ptr_store(acc, red + (it * RED_FLOATS + (wave * 64 + lane) * 4))
        gpu.barrier()
        # this token tile's partials of the part: thread (row of the group, token of the tile)
        nt = min(TILE, S - t * TILE)
        n_out = G_ROWS * nt
        for p in range_constexpr(-(-n_out // THREADS)):
            idx = tid + p * THREADS
            if idx < n_out:
                r = idx % G_ROWS
                tk = t * TILE + idx // G_ROWS
                v = row_sum(red + (r // ROWS) * RED_FLOATS, r % ROWS, idx // G_ROWS)
                put(part, (kpart * MAX_TOKENS + tk) * QKVG_ROWS + rg * G_ROWS + r, v)
        gpu.barrier()  # ``red`` is reused by the next token tile / task


@flyc.jit
def stage_gemv_deferred(c, rg, kpart, w):
    """The GEMV task of a merged launch's K1 with the boundary's norms deferred: x =
    bf16(rstd2 npw (alpha st + beta bf16(pp rstd))), pp = bf16(u pfw), is rstd2 (c1 +
    rstd c2) with c1 = bf16(npw alpha st) (written by K2's boundary: ``x``) and c2 =
    bf16(npw beta pp) from the unpooled rows u of this part's columns (one rank's,
    polled from K3's ``unp`` under K3's tag); the c1 half's MFMAs and row sums run
    first (while u lands), then c2's; the token's two scalars (K3's boundary,
    ``nscal``) fold the halves into the partial: so the K1 waits for neither the
    boundary's x nor its flag."""
    S, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    red, put, part = c["red"], c["put"], c["part"]
    a = c["args"]
    d_poll, d_unp, d_ns = c["d_poll"], c["d_unp"], c["d_nscal"]
    tiles = token_tiles(S)
    r_x = rsrc(a["x"])
    r_fw, r_be, r_nw = rsrc(a["pfw"]), rsrc(a["beta2"]), rsrc(a["npw"])
    V8F = fx.Vector.make_type(8, fx.Float32)
    # this lane's per-k constants (8 values a step)
    fwv, bev, nwv = [], [], []
    for kc in range_constexpr(K_STEPS):
        k = gemv_k(lane, wave, kpart, kc)
        fwv.append(
            fx.Vector(bo.buffer_load(r_fw, k // 2, vec_width=4, dtype=T.i32))
            .bitcast(fx.BFloat16)
            .to(fx.Float32)
        )
        bev.append(
            [
                fx.Vector(bo.buffer_load(r_be, k + 4 * h, vec_width=4, dtype=T.f32))
                for h in range(2)
            ]
        )
        nwv.append(
            fx.Vector(bo.buffer_load(r_nw, k // 2, vec_width=4, dtype=T.i32))
            .bitcast(fx.BFloat16)
            .to(fx.Float32)
        )
    for t in range_constexpr(tiles):
        tok = fx.min(t * TILE + lane % 16, S - 1)
        nt = min(TILE, S - t * TILE)
        n_out = G_ROWS * nt
        rounds = -(-n_out // THREADS)
        # ---- c1: the first half's MFMAs and row sums
        xb = [
            fx.Vector(
                bo.buffer_load(
                    r_x,
                    (tok * HIDDEN + gemv_k(lane, wave, kpart, kc)) // 2,
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=CM_DEV,
                )
            )  # fmt: skip
            for kc in range(K_STEPS)
        ]
        for it in range_constexpr(G_TILES):
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for kc in range_constexpr(K_STEPS):
                acc = mfma_bf16(
                    w[it][kc].bitcast(fx.BFloat16), xb[kc].bitcast(fx.BFloat16), acc
                )
            fx.ptr_store(acc, red + (it * RED_FLOATS + (wave * 64 + lane) * 4))
        gpu.barrier()
        r1 = []
        for p in range_constexpr(rounds):
            idx = fx.min(tid + p * THREADS, n_out - 1)
            r1.append(
                row_sum(
                    red + ((idx % G_ROWS) // ROWS) * RED_FLOATS,
                    (idx % G_ROWS) % ROWS,
                    idx // G_ROWS,
                )
            )
        gpu.barrier()
        # ---- c2 from the unpooled rows of this lane's token and steps
        # (two pairs a spec: a poll load is at most four dwords)
        got = d_poll(
            [
                (d_unp, (tok * HIDDEN + gemv_k(lane, wave, kpart, kc)) // 2 + 2 * hf, 2)
                for kc in range(K_STEPS)
                for hf in range(2)
            ]
        )
        ub = []
        for kc in range_constexpr(K_STEPS):
            words = []
            for q in range_constexpr(4):
                lo, hi = bf2_f32(got[2 * kc + q // 2][q % 2])
                vals = []
                for h, u in ((0, lo), (1, hi)):
                    j = 2 * q + h
                    pp = bf16_round(u * fx.Float32(fwv[kc][j]))
                    vals.append(
                        fx.Float32(nwv[kc][j]) * fx.Float32(bev[kc][j // 4][j % 4]) * pp
                    )
                words.append(bf16_pair(vals[0], vals[1]).bitcast(fx.Int32))
            ub.append(fx.Vector.from_elements(words, fx.Int32))
        for it in range_constexpr(G_TILES):
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for kc in range_constexpr(K_STEPS):
                acc = mfma_bf16(
                    w[it][kc].bitcast(fx.BFloat16), ub[kc].bitcast(fx.BFloat16), acc
                )
            fx.ptr_store(acc, red + (it * RED_FLOATS + (wave * 64 + lane) * 4))
        gpu.barrier()
        # ---- the token's scalars, the fold, the partials
        for p in range_constexpr(rounds):
            idx = tid + p * THREADS
            if idx < n_out:
                r = idx % G_ROWS
                tk = t * TILE + idx // G_ROWS
                r2 = row_sum(red + (r // ROWS) * RED_FLOATS, r % ROWS, idx // G_ROWS)
                sc = d_poll([(d_ns, 2 * tk, 2)])
                rstd, rstd2 = sc[0][0].bitcast(fx.Float32), sc[0][1].bitcast(fx.Float32)
                put(
                    part,
                    (kpart * MAX_TOKENS + tk) * QKVG_ROWS + rg * G_ROWS + r,
                    rstd2 * (r1[p] + rstd * r2),
                )
        gpu.barrier()  # ``red`` is reused by the next token tile / task


def half_sum(poll, part, tk, pb):
    """Token ``tk``'s elements ``pb`` .. ``pb + ELEMS`` of the projection: the K
    parts' fp32 partials (``PARTS`` slots a row) from the mailbox, summed in slot
    order, bf16."""
    gq = poll(
        [
            (part, (hf * MAX_TOKENS + tk) * QKVG_ROWS + pb + 2 * p, 2)
            for hf in range(PARTS)
            for p in range(2)
        ],
        batch=2 * PARTS,  # the parts' pairs in one round
    )
    out = []
    for j in range_constexpr(ELEMS):
        v = gq[j // 2][j % 2].bitcast(fx.Float32)
        for hf in range_constexpr(1, PARTS):
            v = v + gq[hf * 2 + j // 2][j % 2].bitcast(fx.Float32)
        out.append(bf16_round(v))
    return out[0], out[1], out[2], out[3]


@flyc.jit
def stage_head(c, h):
    """Head task ``h`` (q heads, then k heads, then v heads): every token's head
    from the halves' partials, token ``tid / 16`` (+ 32 a pass) on 16 lanes of
    ``ELEMS`` dims each. q and k: the per-head norm (q then the query scale) and
    RoPE; k and v: the fp8 cache write."""
    S, tid, lane = c["S"], c["tid"], c["lane"]
    poll, part = c["poll"], c["part"]
    a = c["args"]
    is_q = h < LOCAL_Q_HEADS
    is_k = (h >= LOCAL_Q_HEADS) & (h < LOCAL_Q_HEADS + LOCAL_KV_HEADS)
    kv_head = is_k.select(h - LOCAL_Q_HEADS, h - LOCAL_Q_HEADS - LOCAL_KV_HEADS)
    col0 = is_q.select(h * HEAD_DIM, is_k.select(K_OFF, V_OFF) + kv_head * HEAD_DIM)
    normed = is_q | is_k
    d0 = (lane % 16) * ELEMS
    use_rope, use_swa = a["use_rope"], a["use_swa"]
    slot64, swa64 = a["slot64"], a["swa64"]
    r_cs = rsrc(a["cos_sin"])
    for p in range_constexpr(token_passes(S)):
        tk = tid // TOKEN_LANES + p * TOKENS_PER_PASS
        if tk < S:
            # positions are int64 and slots int64 or int32: this token's low word
            pos = fx.Int32(
                bo.buffer_load(rsrc(a["positions"]), 2 * tk, vec_width=1, dtype=T.i32)
            )
            slot = fx.Int32(
                bo.buffer_load(
                    rsrc(a["slots"]),
                    (slot64 != 0).select(2 * tk, tk),
                    vec_width=1,
                    dtype=T.i32,
                )
            )
            if (use_swa != 0) & (slot >= 0):
                slot = fx.Int32(
                    bo.buffer_load(
                        rsrc(a["swa_map"]),
                        (swa64 != 0).select(2 * slot, slot),
                        vec_width=1,
                        dtype=T.i32,
                    )
                )
            # this lane's two RoPE pairs: cos / sin of dims d0 / 2 and d0 / 2 + 1
            cs = fx.Vector(
                bo.buffer_load(r_cs, pos * HEAD_DIM + d0 // 2, vec_width=2, dtype=T.f32)
            )
            sn = fx.Vector(
                bo.buffer_load(
                    r_cs,
                    pos * HEAD_DIM + HEAD_DIM // 2 + d0 // 2,
                    vec_width=2,
                    dtype=T.f32,
                )
            )
            e0, e1, e2, e3 = half_sum(poll, part, tk, col0 + d0)
            e = [e0, e1, e2, e3]
            # the per-head norm: sum of squares over the head's 16 lanes
            ss = fx.Float32(0.0)
            for j in range_constexpr(ELEMS):
                ss = ss + e[j] * e[j]
            ss = butterfly(ss, (8, 4, 2, 1))
            rstd = hw_rsq(ss / float(HEAD_DIM) + RMS_EPS)
            n = [bf16_round(e[j] * rstd) for j in range(ELEMS)]
            # the query scale after the norm, rounded
            n = [is_q.select(bf16_round(n[j] * QK_SCALE), n[j]) for j in range(ELEMS)]
            # interleaved RoPE on (x, y) = dims (2i, 2i + 1), bf16 cos / sin, each
            # product rounded, then the sum rounded (the fused kernel's order)
            c0, c1 = bf16_round(cs[0]), bf16_round(cs[1])
            s0, s1 = bf16_round(sn[0]), bf16_round(sn[1])
            rot = [
                bf16_round(n[0] * c0) - bf16_round(n[1] * s0),
                bf16_round(n[1] * c0) + bf16_round(n[0] * s0),
                bf16_round(n[2] * c1) - bf16_round(n[3] * s1),
                bf16_round(n[3] * c1) + bf16_round(n[2] * s1),
            ]
            roped = normed & (use_rope != 0)
            out = [
                normed.select(roped.select(bf16_round(rot[j]), n[j]), e[j])
                for j in range(ELEMS)
            ]
            vec = fx.Vector.from_elements(out, fx.Float32).to(fx.BFloat16)
            if is_q:
                bo.buffer_store(vec, rsrc(a["q_out"]), tk * Q_K + h * HEAD_DIM + d0)
            else:
                if is_k:
                    bo.buffer_store(
                        vec, rsrc(a["k_out"]), tk * KV_K + kv_head * HEAD_DIM + d0
                    )
                else:
                    bo.buffer_store(
                        vec, rsrc(a["v_out"]), tk * KV_K + kv_head * HEAD_DIM + d0
                    )
                if slot >= 0:
                    packed = fp8_pack4(
                        clamp_fp8(out[0]),
                        clamp_fp8(out[1]),
                        clamp_fp8(out[2]),
                        clamp_fp8(out[3]),
                    )
                    # NHD page-16 fp8: slot s, head hh, dim d at byte (s KV_HEADS + hh) HEAD_DIM + d
                    idx = (slot * LOCAL_KV_HEADS + kv_head) * (HEAD_DIM // 4) + d0 // 4
                    if is_k:
                        bo.buffer_store(packed, rsrc(a["k_cache"]), idx)
                    else:
                        bo.buffer_store(packed, rsrc(a["v_cache"]), idx)


@flyc.jit
def stage_gate(c, g):
    """Gate task ``g``: rows ``G_OFF + 64 g`` .. of every token from the halves'
    partials, rounded once, out as bf16 (the o_gate of q head ``g``)."""
    S, tid, lane = c["S"], c["tid"], c["lane"]
    poll, part = c["poll"], c["part"]
    g_out = rsrc(c["args"]["g_out"])
    d0 = (lane % 16) * ELEMS
    for p in range_constexpr(token_passes(S)):
        tk = tid // TOKEN_LANES + p * TOKENS_PER_PASS
        if tk < S:
            e = half_sum(poll, part, tk, G_OFF + g * HEAD_DIM + d0)
            vec = fx.Vector.from_elements(list(e), fx.Float32).to(fx.BFloat16)
            bo.buffer_store(vec, g_out, tk * Q_K + g * HEAD_DIM + d0)


@flyc.jit
def run_k1(c, bid, x_flag=None):
    """K1's stages on this CTA: its GEMV task, then its head or gate task. ``x_flag`` (a mailbox region): x was written earlier in this
    launch (the previous layer's boundary), flagged a token each; the first task's
    weights go out before the wait. In that launch the boundary CTAs (0 .. S) reach
    K1 last (their weights would land ~7 us after everyone's x): their GEMV tasks
    run on the last S CTAs, idle since K3's experts, which issue both tasks' weights
    on arrival."""
    S = c["S"]
    stamp(c, 0)
    first = None
    extra = None
    if const_expr(x_flag is not None):
        moved = bid < S  # a boundary CTA: its task runs on CTA BLOCKS - S + bid
        spare = bid >= BLOCKS - S
        own = moved.select(fx.Int32(GEMV_TASKS), bid)  # past the last task: zero loads
        t2 = spare.select(bid - (BLOCKS - S), fx.Int32(GEMV_TASKS))
        first = gemv_w_loads(c, own // K_PARTS, own % K_PARTS, c["args"]["w"])
        extra = gemv_w_loads(c, t2 // K_PARTS, t2 % K_PARTS, c["args"]["w"])
        rocdl.sched_barrier(0)
        if const_expr(not c.get("defer", False)):
            poll_flags(
                c, [(x_flag, S)]
            )  # (deferred: the GEMV polls its unpooled rows itself)
    stamp(c, 1)  # x arrived (a merged launch) / entry
    if const_expr(x_flag is not None):
        if bid >= S:
            stage_gemv(c, bid // K_PARTS, bid % K_PARTS, first)
        if bid >= BLOCKS - S:
            t2 = bid - (BLOCKS - S)
            stage_gemv(c, t2 // K_PARTS, t2 % K_PARTS, extra)
    else:
        stage_gemv(c, bid // K_PARTS, bid % K_PARTS, first)
    if const_expr(SECOND_TASKS > 0):
        if (bid >= QT0) & (bid < QT0 + SECOND_TASKS):
            q = BLOCKS + bid - QT0  # the tasks past one a CTA, on idle CTAs
            stage_gemv(c, q // K_PARTS, q % K_PARTS)
    stamp(c, 2)  # the GEMV task(s) done
    if (bid >= HT0) & (bid < HT0 + HEAD_TASKS):
        stage_head(c, bid - HT0)
    if (bid >= GT0) & (bid < GT0 + GATE_TASKS):
        stage_gate(c, bid - GT0)
    stamp(c, 3)  # the head / gate task done (end)


def k1_lds(tokens: int):
    """K1's LDS: a reduction row set a row tile (a token tile at a time)."""

    @fx.struct
    class Lds:
        red: fx.Array[fx.Float32, G_TILES * RED_FLOATS, 16]

    return Lds
