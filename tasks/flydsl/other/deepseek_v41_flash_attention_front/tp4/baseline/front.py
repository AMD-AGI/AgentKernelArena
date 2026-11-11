# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Vendored from sgl-amd-mono @ 52486703 (sgl_amd_mono/deepseek_v41/gfx950/attention/front.py).
# Modified for AgentKernelArena: imports are package-relative; FrontBuild is a frozen dataclass
# instead of a msgspec Struct; the index-source index query (``index_query``, not vendored) is
# removed from run_front: no build of these tasks sets ``iq_chains``.
"""Run wqkv_a, the q / kv norms, RoPE, the V4 KV store and wq_b (K1's attention front)."""

import flydsl.compiler as flyc
import flydsl.expr as fx
import dataclasses
from . import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from .epoch import gstore
from .mx import clamp_fp8
from .ops import (
    bf16_round,
    butterfly,
    fp8_pack4,
    fresh,
    hw_rsq,
    row_sum,
    rsrc,
    xshfl,
)
from .plan import first_task
from .stamps import stamp
from .sync import preg
from .gemv import (
    ld_f32,
    lds_row,
    plus,
    poll_copy,
    tile_rows,
    token_tiles,
)
from .token_par import (
    KG_WORDS,
    QX_GROUPS,
    chunk_pairs,
    half_pair,
)

from .attn_device import (
    BLOCKS,
    CM_DEV,
    THREADS,
    WAVES,
    Mailbox,
    ceil_exp,
    div_rn,
    fma,
    ld_i32,
    mx_code,
    mx_mul,
)
from .attn_gemv import (
    RED,
    chain_loads,
    chain_mfmas,
    chain_slots,
    chain_store,
    chain_sum,
    gemv_loads,
    gemv_mfmas,
    rows_loads,
    rows_store,
)
from .attn_plan import (
    DATA,
    EPS,
    HALF,
    HEAD_DIM,
    HIDDEN,
    KEYS,
    KV_DIM,
    NOPE,
    Q_RANK,
    QKV_ROWS,
    ROPE,
    ROWS,
    TOPK,
    WINDOW,
    WQKV_KPARTS,
    WQKV_TASKS,
    Dims,
    front_scratch,
    wqkv_order,
    wqkv_split,
)

X_WORDS = HIDDEN // 4
X_GROUPS = HIDDEN // 32
QX_WORDS = Q_RANK // 4
KSTEPS = HIDDEN // 128 // WQKV_KPARTS  # an X8 K half's 128-steps
KWORDS = KSTEPS * 32
KGROUPS = KSTEPS * 4


@dataclasses.dataclass(frozen=True)
class FrontBuild:
    tokens: int
    tp: int
    kt: bool = False  # also write K2's key table
    # a row's window index slots: the window, plus the block's keys for the DSpark draft
    swa_width: int = WINDOW
    shuffled: bool = False  # wqkv and wq_b in aiter's (16, 16) preshuffle


@flyc.jit
def publish(c, region, i):
    """This CTA's stores visible device-wide, then the flag pair (region, i)."""
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if c["tid"] == 0:
        c["mb"].put(region, i, fx.Int32(1))


@flyc.jit
def await_flags(c, region, n, index):
    """Thread u < n polls flag ``index(u)``; then the CTA proceeds."""
    tid = c["tid"]
    if tid < n:
        c["mb"].poll([(region, index(fx.min(tid, n - 1)), 1)])
    gpu.barrier()


# ---------------------------------------------------------------- kt


@flyc.jit
def stage_kt(c, t):
    """Token t's keys for K2: ``KT[t][k] = slot | 1 << 31`` for its top-k compressed
    rows (k < ntopk), its window slots for k in [ntopk, kv_len), -1 past them or
    for an invalid slot; KLEN[t] = kv_len (0 for a pad row), KLEN[S + t] = ntopk.
    ``dev``: device scope, for a table the same launch reads (K2's)."""
    tid = c["tid"]
    a = c["args"]
    cm = CM_DEV if const_expr(c.get("dev", False)) else 0
    sw = c.get("swa_width", WINDOW)
    live = ld_i32(a["slot"], a["slot_words"] * t) >= 0
    nswa = live.select(fx.min(ld_i32(a["swa_lens"], t), fx.Int32(sw)), fx.Int32(0))
    ntopk = live.select(fx.min(ld_i32(a["topk_lens"], t), fx.Int32(TOPK)), fx.Int32(0))
    kv_len = ntopk + nswa
    for i in range_constexpr((KEYS + THREADS - 1) // THREADS):
        k = fresh(tid) + THREADS * i
        sidx = fx.min(fx.max(k - ntopk, 0), sw - 1)
        s_slot = ld_i32(a["swa_idx"], t * sw + sidx)
        val = ((k < kv_len) & (s_slot >= 0)).select(s_slot, fx.Int32(-1))
        local = ld_i32(a["topk"], t * TOPK + fx.min(k, TOPK - 1))
        c_val = (local >= 0).select(local | fx.Int32(-2147483648), fx.Int32(-1))
        val = (k < ntopk).select(c_val, val)
        if k < KEYS:
            bo.buffer_store(val, rsrc(a["kt"]), t * KEYS + k, cache_modifier=cm)
    if tid == 0:
        bo.buffer_store(kv_len, rsrc(a["klen"]), t, cache_modifier=cm)
        bo.buffer_store(ntopk, rsrc(a["klen"]), c["S"] + t, cache_modifier=cm)


# ---------------------------------------------------------------- wqkv_a


def stage_wqkv(c, ops, slots, n, halves):
    """Run the wqkv chains against a token tile of n rows."""
    accs = chain_mfmas(c, c["xl"], c["xsl"], ops, n, HIDDEN // 128, slots, halves)
    gpu.barrier()
    return accs


@flyc.jit
def wqkv_publish(c, task, accs, slots, order, t0, n, live, part=None, s=0):
    """Sum wqkv rows 16 task .. for tokens t0 .. t0 + n in SGLang's order into fp32 QKVP.
    Split (``part``): part 0 sums the first half of ``order`` into QKVP's second region
    (``s`` rows on), part 1 the rest onto that sum."""
    tid, red = c["tid"], c["red"]
    chain_store(c, red, accs, slots)
    gpu.barrier()
    if (tid < ROWS * n) & live:
        r = tid % ROWS
        tl = tid // ROWS
        at = plus(t0, tl) * QKV_ROWS + task * ROWS + r
        if const_expr(part is None):
            c["mb"].put(c["qkvp"], at, chain_sum(red, r, tl, order))
        else:
            h, mid = len(order) // 2, s * QKV_ROWS + at
            if part == 0:
                c["mb"].put(c["qkvp"], mid, chain_sum(red, r, tl, order[:h]))
            else:
                tot = c["mb"].poll([(c["qkvp"], mid, 1)])[0][0].bitcast(fx.Float32)
                c["mb"].put(c["qkvp"], at, chain_sum(red, r, tl, order[h:], tot))
    gpu.barrier()


# ---------------------------------------------------------------- q / kv norms


@flyc.jit
def stage_qkv(c, t):
    """Token t: q_norm to QX8, and kv_norm + RoPE to the fp8_ds_mla record at its slot."""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    a = c["args"]
    tt = fx.min(tid, 255)
    mine = tid < 256
    qc = [fx.min(tt * 8 + j, Q_RANK - 1) for j in range(8)]
    # independent loads first: they overlap the partials' wait
    qw = [c["ldbf"](a["qn_w"], qc[j]) for j in range(8)]
    kw = [c["ldbf"](a["kvn_w"], tt * 2 + j) for j in range(2)]
    # token parallel: t is this rank's row, tg the step's
    tok_par = c.get("tok_par", False)
    tg = plus(c["t0"], t) if const_expr(tok_par) else t
    pos = ld_i32(a["pos"], 2 * tg)
    slot = ld_i32(a["slot"], a["slot_words"] * tg)
    jr = tt - NOPE // 2
    is_rope = jr >= 0
    jj = fx.max(jr, 0)
    cs = is_rope.select(ld_f32(a["cos_sin"], pos * ROPE + jj), fx.Float32(1.0))
    sn = is_rope.select(ld_f32(a["cos_sin"], pos * ROPE + HALF + jj), fx.Float32(0.0))
    specs = [(c["qkvp"], t * QKV_ROWS + qc[j], 1) for j in range(8)]
    specs += [(c["qkvp"], t * QKV_ROWS + Q_RANK + tt * 2 + j, 1) for j in range(2)]
    got = c["mb"].poll(specs)
    vals = [bf16_round(got[j][0].bitcast(fx.Float32)) for j in range(10)]
    xq = [((tt * 8 + j) < Q_RANK).select(vals[j], fx.Float32(0.0)) for j in range(8)]
    xk = vals[8:10]
    ssq = xq[0] * xq[0]
    for j in range_constexpr(1, 8):
        ssq = ssq + xq[j] * xq[j]
    ssq = butterfly(ssq, (32, 16, 8, 4, 2, 1))
    # kv's sum of squares in the order of SGLang's K store (fused_k_norm_rope_flashmla, gfx950)
    ssk = butterfly(fma(xk[0], xk[0], xk[1] * xk[1]), (16, 8, 4, 2, 1))
    if (lane == 0) & mine:
        fx.ptr_store(ssq, red + wave)
    if (lane % 32 == 0) & mine:
        fx.ptr_store(ssk, red + 16 + tt // 32)
    gpu.barrier()
    tq = (fx.ptr_load(red + 0) + fx.ptr_load(red + 1)) + (
        fx.ptr_load(red + 2) + fx.ptr_load(red + 3)
    )
    pk = [fx.ptr_load(red + 16 + w) for w in range(8)]
    tk = ((pk[0] + pk[4]) + (pk[2] + pk[6])) + ((pk[1] + pk[5]) + (pk[3] + pk[7]))
    rq = hw_rsq(
        div_rn(tq, fx.Float32(float(Q_RANK)), fx.Float32(1.0 / Q_RANK))
        + fx.Float32(EPS)
    )
    rk = hw_rsq(
        div_rn(tk, fx.Float32(float(KV_DIM)), fx.Float32(1.0 / KV_DIM))
        + fx.Float32(EPS)
    )
    # q: bf16 normed row, then vLLM's MXFP8 (a 32-group = 4 threads)
    ys = [bf16_round((xq[j] * rq) * qw[j]) for j in range(8)]
    amax = abs(ys[0])
    for y in ys[1:]:
        amax = fx.max(amax, abs(y))
    code = mx_code(butterfly(amax, (1, 2), fx.max))
    mul = mx_mul(code)
    if mine & (tt * 8 < Q_RANK):
        w0 = fp8_pack4(*[clamp_fp8(ys[i] * mul) for i in range(4)])
        w1 = fp8_pack4(*[clamp_fp8(ys[4 + i] * mul) for i in range(4)])
        if const_expr(tok_par):
            # to every rank's QG / QGS, at the step's row: wq_b reads every row's
            for p in range_constexpr(c["d"].tp):
                words_dst, codes_dst = c["q_dsts"][p]
                c["mb"].put_words(words_dst, tg * QX_WORDS + 2 * tt, [w0, w1])
                if tt % 4 == 0:
                    c["mb"].put(codes_dst, tg * QX_GROUPS + tt // 4, code)
        else:
            bo.buffer_store(
                fx.Vector.from_elements([w0, w1], fx.Int32),
                rsrc(c["qx8"]),
                t * QX_WORDS + 2 * tt,
                cache_modifier=CM_DEV,
            )
            if tt % 4 == 0:
                bo.buffer_store(
                    code,
                    rsrc(c["qx8s"]),
                    t * (Q_RANK // 32) + tt // 4,
                    cache_modifier=CM_DEV,
                )
    if const_expr(not tok_par):
        # QRDY once QX8 landed: the qr and KV record stores after it are read only past
        # this launch, and the publish's vmcnt(0) would wait on them too
        publish(c, c["qrdy"], t)
    if const_expr(c.get("index", False)):  # noqa: SIM102
        # the bf16 qr, SGLang's indexer's input (an index source's K1)
        if mine & (tt * 8 < Q_RANK):
            words = [
                fx.Vector.from_elements([ys[2 * i], ys[2 * i + 1]], fx.Float32)
                .to(fx.BFloat16)
                .bitcast(fx.Int32)[0]
                for i in range(4)
            ]
            bo.buffer_store(
                fx.Vector.from_elements(words, fx.Int32),
                rsrc(a["qr_out"]),
                tg * (Q_RANK // 2) + 4 * tt,
            )
            if const_expr(tok_par):
                for p in range_constexpr(c["d"].tp):
                    if c["rank"] != p:
                        # two pairs a store: a store moves 16 B at most
                        for h in range_constexpr(2):
                            c["mb"].put_words(
                                c["qr_dsts"][p],
                                tg * (Q_RANK // 2) + half_pair(Q_RANK // 2, tt, h),
                                words[2 * h : 2 * h + 2],
                            )
    # kv: this thread's pair (2 tt, 2 tt + 1), fp32 rounding as SGLang's V4 K store
    kn = [(xk[j] * rk) * kw[j] for j in range(2)]
    e0 = is_rope.select(fma(kn[0], cs, -(kn[1] * sn)), kn[0])
    e1 = is_rope.select(fma(kn[0], sn, kn[1] * cs), kn[1])
    # record quant of the nope dims: a 64-block = 32 threads (half a wave)
    amax = butterfly(fx.max(abs(e0), abs(e1)), (1, 2, 4, 8, 16), fx.max)
    rcode = ceil_exp(
        div_rn(
            fx.max(amax, fx.Float32(1e-4)), fx.Float32(448.0), fx.Float32(1.0 / 448.0)
        )
    )
    inv = mx_mul(rcode)
    q0 = clamp_fp8(e0 * inv)
    q1 = clamp_fp8(e1 * inv)
    n0 = xshfl(q0, 1)
    n1 = xshfl(q1, 1)
    word = fp8_pack4(q0, q1, n0, n1)
    rope_word = (
        fx.Vector.from_elements([e0, e1], fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Int32)[0]
    )
    blk = a["swa_block"]
    sblock = slot // blk
    soff = slot % blk
    base = a["swa"] + fx.Int64(sblock) * fx.Int64(a["swa_stride"])
    rec = base + fx.Int64(soff * DATA)
    if mine & (tt % 32 == 0) & (tt < NOPE // 2):
        fx.ptr_store(rcode.bitcast(fx.Float32), red + 8 + tt // 32)
    gpu.barrier()
    if mine & (slot >= 0):
        if tt < NOPE // 2:
            if tt % 2 == 0:
                gstore(rec + fx.Int64(2 * tt), word, words=1)
        else:
            gstore(rec + fx.Int64(NOPE + 4 * (tt - NOPE // 2)), rope_word, words=1)
        if tt == 0:
            sc = base + fx.Int64(blk * DATA + soff * 8)
            gstore(sc, fx.Vector.from_elements(record_scales(red), fx.Int32), words=2)
    if const_expr(tok_par):
        # the record to every other rank's KG (``record_from``), a pad row's too
        if mine:
            nope = tt < NOPE // 2
            w = nope.select(word, rope_word)
            at = nope.select(tt // 2, DATA // 4 - ROPE // 2 + (tt - NOPE // 2))
            for p in range_constexpr(c["d"].tp):
                if c["rank"] != p:
                    if (tt % 2 == 0) | (tt >= NOPE // 2):
                        c["mb"].put_words(c["kv_dsts"][p], tg * KG_WORDS + at, [w])
                    if tt == 0:
                        c["mb"].put_words(
                            c["kv_dsts"][p],
                            tg * KG_WORDS + DATA // 4,
                            record_scales(red),
                        )
        gpu.barrier()


def record_scales(red):
    """The record's two scale words, from the 7 codes ``stage_qkv`` put in ``red``."""
    sb = [fx.ptr_load(red + 8 + b).bitcast(fx.Int32) for b in range(7)]
    w0 = sb[0] | (sb[1] << 8) | (sb[2] << 16) | (sb[3] << 24)
    w1 = sb[4] | (sb[5] << 8) | (sb[6] << 16)
    return [w0, w1]


@flyc.jit
def record_from(c, kg, t):
    """Row t, another rank's: its window record from KG -> its slot (``stage_qkv``'s
    stores)."""
    tid = c["tid"]
    a = c["args"]
    if tid < KG_WORDS:
        (w,) = c["mb"].poll([(kg, t * KG_WORDS + tid, 1)])
        slot = ld_i32(a["slot"], a["slot_words"] * t)
        if slot >= 0:
            blk = a["swa_block"]
            soff = slot % blk
            base = a["swa"] + fx.Int64(slot // blk) * fx.Int64(a["swa_stride"])
            data = tid < DATA // 4
            off = data.select(
                soff * DATA + 4 * tid, blk * DATA + soff * 8 + 4 * (tid - DATA // 4)
            )
            gstore(base + fx.Int64(off), w[0], words=1)


@flyc.jit
def qr_from(c, qrg, t):
    """Row t, another rank's: its bf16 qr from QRG -> ``qr_out``."""
    tid = c["tid"]
    for k in range_constexpr(-(-(Q_RANK // 2) // THREADS)):
        q = tid + THREADS * k
        if q < Q_RANK // 2:
            (w,) = c["mb"].poll([(qrg, t * (Q_RANK // 2) + q, 1)])
            pair = chunk_pairs(Q_RANK // 2, q // 2) + q % 2
            bo.buffer_store(w[0], rsrc(c["args"]["qr_out"]), t * (Q_RANK // 2) + pair)


def token_par_context(c, own, t0, rank, bases, sym, pl):
    """Point the front at token parallel (``token_par``): its rows t0 .. t0 + own,
    the peers' gather regions it pushes to, and QG / QGS, where wq_b reads every
    row's q latent."""
    c |= {
        "tok_par": True,
        "own": own,
        "t0": t0,
        "rank": rank,
        "poll": c["mb"].poll,
        "q_dsts": [(preg(b, pl["qg"], "qg"), preg(b, pl["qgs"], "qgs")) for b in bases],
        "kv_dsts": [preg(b, pl["kg"], "kg") for b in bases],
        "qr_dsts": [preg(b, pl["qrg"], "qrg") for b in bases],
        "qg": preg(sym, pl["qg"], "qg"),
        "qgs": preg(sym, pl["qgs"], "qgs"),
    }


# ---------------------------------------------------------------- wq_b


def wqb_rope(c, task0, pos_t, n):
    """Return (cos, sin) of this thread's rope pair for ``stage_wqb``'s epilogue."""
    a = c["args"]
    rp = c["tid"] % (ROWS // 2)
    dim = (task0 * ROWS) % HEAD_DIM + 2 * rp
    jj = fx.max((dim - NOPE) >> 1, 0)
    return ld_f32(a["cos_sin"], pos_t * ROPE + jj), ld_f32(
        a["cos_sin"], pos_t * ROPE + HALF + jj
    )


def wqb_pos(c, t0, n):
    """Return this thread's token position for ``stage_wqb``'s epilogue."""
    per = ROWS * n // 2
    tl = fx.min((c["tid"] % per) // (ROWS // 2), n - 1)
    return ld_i32(c["args"]["pos"], 2 * (t0 + tl))


@flyc.jit
def stage_wqb(c, task0, bops, t0, n, rope):
    """Run the CTA's wq_b tasks: GEMV, bf16, GPT-J RoPE of dims 448.., store q."""
    tid, red, d = c["tid"], c["red"], c["d"]
    a = c["args"]
    nt = len(bops)
    for k in range_constexpr(nt):
        gemv_mfmas(c, Q_RANK, c["qxl"], c["qxsl"], red + k * RED, bops[k], rows=n)
    gpu.barrier()
    per = ROWS * n // 2  # bf16 pairs a task
    if tid < nt * per:
        k = tid // per
        u = tid % per
        rp = u % (ROWS // 2)
        tl = u // (ROWS // 2)
        t = plus(t0, tl)
        r0 = (task0 + BLOCKS * k) * ROWS + 2 * rp
        rk = red + k * RED
        v0 = bf16_round(row_sum(rk, 2 * rp, tl))
        v1 = bf16_round(row_sum(rk, 2 * rp + 1, tl))
        is_rope = (task0 * ROWS) % HEAD_DIM + 2 * rp >= NOPE
        cs, sn = rope
        e0 = is_rope.select(v0 * cs - v1 * sn, v0)
        e1 = is_rope.select(v0 * sn + v1 * cs, v1)
        word = (
            fx.Vector.from_elements([e0, e1], fx.Float32)
            .to(fx.BFloat16)
            .bitcast(fx.Int32)[0]
        )
        bo.buffer_store(word, rsrc(a["q"]), (t * d.q_rows + r0) // 2)
    gpu.barrier()


# ---------------------------------------------------------------- kernel


def front_lds(s: int, d: Dims):
    """K1's LDS (the attention front's stages)."""
    rows = tile_rows(s)
    wqb_per_cta = d.wqb_tasks // BLOCKS

    @fx.struct
    class FrontLds:
        red: fx.Array[fx.Float32, wqb_per_cta * RED + 16, 16]
        qxl: fx.Array[fx.Int32, rows * lds_row(QX_WORDS), 16]
        qxsl: fx.Array[fx.Int32, rows * lds_row(Q_RANK // 32), 16]
        xl: fx.Array[fx.Int32, rows * lds_row(X_WORDS), 16]
        xsl: fx.Array[fx.Int32, rows * lds_row(X_GROUPS), 16]

    return FrontLds


def front_context(key, lds, args: dict, scratch, tag, layout=None) -> dict:
    """Return the front stages' context: LDS views, mailbox, scratch regions and arguments."""
    s, d = key.tokens, Dims(key.tp)
    tid = fx.thread_idx.x
    c = {
        "S": s,
        "tid": tid,
        "bid": fx.block_idx.x,
        "lane": tid % 64,
        "wave": tid // 64,
        "red": lds.red.ptr,
        "xl": lds.xl.ptr,
        "xsl": lds.xsl.ptr,
        "qxl": lds.qxl.ptr,
        "qxsl": lds.qxsl.ptr,
        "mb": Mailbox(tag),
        "d": d,
        "swa_width": key.swa_width,
        "shuffled": key.shuffled,
        "ldbf": lambda ptr, i: fx.Float32(
            fx.BFloat16(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.bf16))
        ),
        "args": args,
    }
    for region, (off, _) in (layout or front_scratch(s, d)).items():
        c[region] = scratch + fx.Int64(off)
    return c


@flyc.jit
def run_front(
    c,
    key,
    bid,
    x8_given=False,
    before_wqkv=None,
    after_wqkv=None,
    before_wqb=None,
    qkv_first=0,
):
    """Run the front's stages on this CTA; the seam publishes X8 before them.
    ``qkv_first``: the first CTA of the q / kv norms."""
    s = key.tokens
    # the wqkv GEMV takes all of K from X8, which the seam publishes
    assert x8_given
    wqb_tasks = Dims(key.tp).wqb_tasks
    a = c["args"]
    tiles = token_tiles(s)
    # wqkv_a and qkv on this rank's rows (token parallel), wq_b on the step's
    own = c.get("own", s)
    wtiles = token_tiles(own)
    wqb_per_cta = wqb_tasks // BLOCKS
    assert wqb_tasks % BLOCKS == 0
    n_wqkv = WQKV_TASKS  # a row tile a CTA, all of K in SGLang's order
    QKV0 = qkv_first
    WQB0 = 0
    waves, halves, order = wqkv_order(s)
    # a row tile's chains on two CTAs (``wqkv_split``): half its weights each land sooner
    split = len(wtiles) == 1 and wqkv_split(s, own)
    chains = waves // 2 if split else waves
    dead = -(-chains // WAVES) * WAVES - chains
    # the chains' partials, and a dead wave's, fit ``red``
    assert (waves + dead) * 64 * 4 <= wqb_per_cta * RED

    # loads complete in issue order: each CTA issues its latency-critical loads first
    wqb_task = first_task(bid, WQB0)
    if const_expr(len(wtiles) == 1):
        units = 2 * n_wqkv if split else n_wqkv
        wqkv_unit = first_task(bid, 0)
        has_wqkv = wqkv_unit < units
        unit = fx.min(wqkv_unit, units - 1)
        task, part = unit % n_wqkv, unit // n_wqkv
        # split: the key table past the norms', gates' and qkv's CTAs (``build_mono_k1``)
        kt_from = QKV0 + own if split else n_wqkv
        kt_step = BLOCKS - kt_from
    else:
        # a row tile's token tiles on CTAs task, task + n_wqkv, .. (ROCm/ATOM), not one in turn
        task = bid % n_wqkv
        group = bid // n_wqkv
        ctas = (BLOCKS - 1 - task) // n_wqkv + 1
        has_wqkv = group < fx.min(ctas, len(wtiles))
        # the key table on the CTAs without a tile, else past the first round
        kt_from = n_wqkv * len(wtiles) if n_wqkv * len(wtiles) < BLOCKS else n_wqkv
        kt_step = BLOCKS - kt_from

    def wqb_weights(pred=None):
        return [
            gemv_loads(
                c,
                a["wqb"],
                a["wqb_s"],
                Q_RANK,
                (wqb_task + BLOCKS * i) * ROWS,
                pred=pred,
                shuffled=c["shuffled"],
            )
            for i in range(wqb_per_cta)
        ]

    if const_expr(before_wqkv is not None):
        before_wqkv()
    slots = chain_slots(c, HIDDEN // 128, waves, part * chains if split else 0, chains)
    kops = chain_loads(
        c, a["wqkv"], a["wqkv_s"], HIDDEN, task * ROWS, slots, has_wqkv, c["shuffled"]
    )
    if const_expr(after_wqkv is not None):
        after_wqkv()
    # one token tile: the wq_b weights here on a CTA without a wqkv tile (the qkv stage's polls
    # behind them land by then); a wqkv CTA's past its GEMV, as its barriers wait on every load
    early_wqb = len(wtiles) == 1
    if const_expr(early_wqb):
        bops = wqb_weights(wqkv_unit >= units)
    if const_expr(key.kt):
        # K2's key table on CTAs past the wqkv units; K2 reads it a launch later
        kt_first = bid - kt_from
        for t in range((kt_first >= 0).select(kt_first, fx.Int32(s)), s, kt_step):
            stage_kt(c, t)
    stamp(c, 35)
    poss = [wqb_pos(c, t0, n) for t0, n in tiles]

    def x_loads(t0, n):
        return [
            rows_loads(
                c,
                c["x8"],
                c["x8s"],
                X_WORDS,
                p * KWORDS,
                KWORDS,
                X_GROUPS,
                p * KGROUPS,
                KGROUPS,
                t0,
                n,
                pred=has_wqkv,
            )
            for p in range(WQKV_KPARTS)
        ]

    def x_store(regs, n):
        for p in range_constexpr(WQKV_KPARTS):
            rows_store(
                c,
                regs[p],
                KWORDS,
                KGROUPS,
                c["xl"] + p * KWORDS,
                c["xsl"] + p * KGROUPS,
                n,
                lds_row(X_WORDS),
                lds_row(X_GROUPS),
            )

    if const_expr(len(wtiles) == 1):
        # wqkv_a: one row tile a CTA, all of K in LDS
        t0, n = wtiles[0]
        await_flags(c, c["xrdy"], WQKV_KPARTS * own, lambda u: u)
        regs = x_loads(t0, n)
        x_store(regs, n)
        accs = stage_wqkv(c, kops, slots, n, halves)
        wqkv_publish(
            c, task, accs, slots, order, t0, n, has_wqkv, part if split else None, s
        )
        if has_wqkv:
            bops = wqb_weights()
    else:
        # wqkv_a: this CTA's tiles of its row tile, each on its own tokens' flags
        for ti in range_constexpr(len(wtiles)):
            t0, n = wtiles[ti]
            if has_wqkv & (fx.Int32(ti) % ctas == group):
                await_flags(
                    c, c["xrdy"], WQKV_KPARTS * n, lambda u: WQKV_KPARTS * t0 + u
                )
                x_store(x_loads(t0, n), n)
                # nothing assigned in a traced branch: FlyDSL yields no None
                wqkv_publish(
                    c,
                    task,
                    stage_wqkv(c, kops, slots, n, halves),
                    slots,
                    order,
                    t0,
                    n,
                    has_wqkv,
                )
        # the wq_b weights in flight through the wait on q
        bops = wqb_weights()

    stamp(c, 36)
    for t in range(first_task(bid, QKV0), own, BLOCKS):
        stage_qkv(c, t)
    stamp(c, 37)

    # wq_b: every task's rows for one token tile at a time
    ropes = [wqb_rope(c, wqb_task, poss[ti], tiles[ti][1]) for ti in range(len(tiles))]
    if const_expr(before_wqb is not None):
        # work that does not wait on q: it runs while the qkv stage finishes
        before_wqb()
    if const_expr(c.get("tok_par", False)):
        # every row's q latent from QG / QGS, where its owner's qkv put it
        for ti in range_constexpr(len(tiles)):
            t0, n = tiles[ti]
            poll_copy(
                c,
                c["tid"],
                THREADS,
                [
                    (
                        c["qg"],
                        lambda u: plus(t0 * QX_WORDS, u),
                        n * QX_WORDS,
                        c["qxl"],
                        QX_WORDS,
                    ),
                    (
                        c["qgs"],
                        lambda u: plus(t0 * QX_GROUPS, u),
                        n * QX_GROUPS,
                        c["qxsl"],
                        QX_GROUPS,
                    ),
                ],
            )
            gpu.barrier()
            if const_expr(ti == 0):
                stamp(c, 38)
            stage_wqb(c, wqb_task, bops, t0, n, ropes[ti])
    else:
        await_flags(c, c["qrdy"], s, lambda u: u)
        stamp(c, 38)
        qx = (c["qx8"], c["qx8s"], QX_WORDS, 0, QX_WORDS, Q_RANK // 32, 0, Q_RANK // 32)
        regs = rows_loads(c, *qx, *tiles[0])
        for ti in range_constexpr(len(tiles)):
            t0, n = tiles[ti]
            rows_store(c, regs, QX_WORDS, Q_RANK // 32, c["qxl"], c["qxsl"], n)
            if const_expr(ti + 1 < len(tiles)):
                regs = rows_loads(c, *qx, *tiles[ti + 1])
            stage_wqb(c, wqb_task, bops, t0, n, ropes[ti])
    stamp(c, 39)
