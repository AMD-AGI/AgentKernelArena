# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Vendored from sgl-amd-mono @ 52486703 (sgl_amd_mono/deepseek_v41/gfx950/attention/back.py). Modified for
# AgentKernelArena: imports are package-relative; BackBuild is a frozen dataclass
# instead of a msgspec Struct.
"""K2 ``back``: one DeepSeek-V4.1 attention layer from the rotated query to the
rank's wo_b partial, adapted from vLLM's mono decode (``attention/back.py``).

One launch a layer and rank runs split (flash-decoding over a chunk of a token's
keys from K1's key table, every head of the rank), combine (the chunks merged with
the sink, inverse GPT-J RoPE -> bf16 XO), wo_a (-> MXFP8 X8B) and wo_b (-> the
caller's hook); the stages hand off device-scope data behind tagged flags.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import dataclasses
from . import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value

from .mx import clamp_fp8
from .ops import (
    CM_DEV,
    bf16_round,
    bf_hi,
    bf_lo,
    butterfly,
    fp8_pack4,
    fresh,
    global_load,
    hw_exp2,
    lane_gather,
    mfma_bf16,
    row_sum,
    rsrc,
    xshfl,
)
from .plan import (
    BLOCKS,
    THREADS,
    WAVES,
    first_task,
)
from .stamps import stamp
from .gemv import (
    ld_f32,
    lds_row,
)

from .attn_device import Mailbox, fp8x8_bf16, ld_i32, mx_code, mx_mul, pow2
from .attn_gemv import (
    RED,
    bf16_gemv_loads,
    bf16_gemv_mfmas,
    gemv_loads,
    gemv_mfmas,
    rows_loads,
    rows_store,
    words_loads,
    words_store,
)
from .attn_plan import (
    DATA,
    HALF,
    HEAD_DIM,
    HEAD_TILE,
    KEYS,
    NOPE,
    ROPE,
    ROWS,
    WOA_ROWS,
    Dims,
    cdiv,
    tile_rows,
    token_tiles,
)

_STREAM = fx.Stream(None)
NEG = -3.4028234663852886e38
LOG2E = 1.4426950408889634
REC = DATA + 8  # a raw record row in LDS: 576 data bytes + 8 scale bytes
# a q row in LDS, i32 words: 512 bf16 + 16 B of pad, so rows do not alias banks
Q_ROW = HEAD_DIM // 2 + 4
P_PAD = 2  # i32 words past a P row: 16 head rows read together hit distinct banks
# wo_a's K order: waves 0..3 / 4..7 take the two 16-row tiles, a K range each, summed in wave order
WOA_SPLIT_K = WAVES * ROWS // WOA_ROWS


@dataclasses.dataclass(frozen=True)
class BackBuild:
    tokens: int
    tp: int
    ratio: int  # 0: window only; 1 / 2: top-k over a compressed cache
    timeline: bool = False
    shuffled: bool = False  # wo_b in aiter's (16, 16) preshuffle (wo_a stays row-major)


def plus(t0, v):
    return v if isinstance(t0, int) and t0 == 0 else v + t0


def head_groups(d: Dims, s: int) -> int:
    """Wave groups a split unit's head tiles are spread over: two (a tile each,
    64-key chunks, twice the units) while the units fit the grid, else one (a
    wave's keys loaded and dequantized once for every head tile)."""
    return 2 if d.head_tiles == 2 and s * cdiv(KEYS, 64) <= BLOCKS else 1


def chunk_keys(d: Dims, s: int) -> int:
    """A split unit's keys: a 16-key tile a wave of a head group."""
    return 16 * (WAVES // head_groups(d, s))


def p_row(d: Dims, s: int) -> int:
    """A P row (one head's chunk of bf16 p) in LDS, i32 words."""
    return chunk_keys(d, s) // 2 + P_PAD


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


# ---------------------------------------------------------------- keys


def key_rec(c, t, k):
    """(data address, scale address, valid) of token t's key k from K1's key
    table: slot | 1 << 31 for a compressed row, a window slot otherwise, -1 for
    none (it reads a zero record: its scores are masked, its values zeros)."""
    a, ratio = c["args"], c["ratio"]
    # a table this launch wrote (``krdy``: an index source's) is read past the CU's cache
    cm = CM_DEV if const_expr("krdy" in c) else 0
    val = ld_i32(a["kt"], t * KEYS + fx.min(fx.max(k, 0), KEYS - 1), cm)
    ok = val != fx.Int32(-1)
    slot = ok.select(val & fx.Int32(0x7FFFFFFF), fx.Int32(0))
    if const_expr(ratio > 0):
        is_c = ok & (val < 0)
        blk = is_c.select(a["comp_block"], a["swa_block"])
        stride = is_c.select(a["comp_stride"], a["swa_stride"])
        base = is_c.select(a["comp"], a["swa"])
    else:
        blk, stride, base = a["swa_block"], a["swa_stride"], a["swa"]
    b = slot // blk
    off = slot % blk
    page = base + fx.Int64(b) * stride
    z = a["zrec"]
    data = ok.select(page + fx.Int64(off * DATA), z)
    sc = ok.select(page + fx.Int64(blk * DATA + off * 8), z + fx.Int64(DATA))
    return data, sc, ok


def _scale_byte(scw, blk64):
    return (scw[blk64 // 4] >> (blk64 % 4 * 8)) & 0xFF


def _fp8_pair_f32(word, scale, hi):
    v = rocdl.cvt_scalef32_pk_f32_fp8(
        fx.Vector.make_type(2, fx.Float32),
        as_ir_value(fx.Int32(word)),
        as_ir_value(fx.Float32(scale)),
        hi,
    )
    v = fx.Vector(v)
    return [v[0], v[1]]


# ---------------------------------------------------------------- split


@flyc.jit
def stage_split(c, unit):
    """Flash-decoding over chunk ``unit % nchunk`` of token ``unit // nchunk``,
    every head of the rank -> the chunk's partial and (max, sum) per head.

    Wave (head group hg, key tile kt) takes keys 16 kt .. of the chunk, loaded and
    dequantized once, against the group's head tiles; then wave w takes dims
    64 w .. of the PV over every key and head."""
    tid, lane, wave, d, s = c["tid"], c["lane"], c["wave"], c["d"], c["S"]
    a, sl = c["args"], c["sl"]
    nchunk = c["nchunk"]
    t = unit // nchunk
    ch = unit % nchunk
    ck = chunk_keys(d, s)
    ngrp = head_groups(d, s)
    nkt = WAVES // ngrp  # 16-key tiles a chunk
    nht = d.head_tiles // ngrp  # head tiles a wave
    hgi = wave // nkt
    kt = wave % nkt
    g = lane // 16
    r16 = lane % 16
    k0 = ch * ck
    # q does not depend on the key table: in flight before an index source's wait
    qrow = d.heads * HEAD_DIM // 8
    qv = []
    for i in range_constexpr(cdiv(qrow, THREADS)):
        u = fresh(tid) + THREADS * i
        qv.append(
            fx.Vector(
                bo.buffer_load(
                    rsrc(a["q"]),
                    (t * d.q_rows) // 2 + fx.min(u, qrow - 1) * 4,
                    vec_width=4,
                    dtype=T.i32,
                )
            )
        )
    if const_expr("krdy" in c):
        # an index source's table: this launch builds it after the indexer ran
        await_flags(c, c["krdy"], 1, lambda u: t)
    # kv_len and this lane's key load together: the table holds -1 past a token's keys
    kv_len = ld_i32(a["klen"], t, CM_DEV if const_expr("krdy" in c) else 0)
    key = k0 + kt * 16 + r16
    data, sc, okv = key_rec(c, t, key)
    if k0 < kv_len:
        # lane g's 16 B slices: e4m3 dims 64 c + 16 g .., bf16 rope dims 32 c + 8 g ..
        scw = global_load(sc, words=2)
        nope = [
            global_load(data + fx.Int64(64 * c4 + 16 * g), words=4)
            for c4 in range(NOPE // 64)
        ]
        rope = [
            global_load(data + fx.Int64(NOPE + (cc * 32 + 8 * g) * 2), words=4)
            for cc in range(ROPE // 32)
        ]
        for i in range_constexpr(cdiv(qrow, THREADS)):
            u = fresh(tid) + THREADS * i
            if u < qrow:
                fx.ptr_store(
                    qv[i],
                    sl["q"]
                    + (u // (HEAD_DIM // 8)) * Q_ROW
                    + (u % (HEAD_DIM // 8)) * 4,
                )
        # the raw rows to LDS for the PV: every head group loaded the same keys, group 0 stores
        if hgi == 0:
            rawb = sl["raw"] + (kt * 16 + r16) * (REC // 4)
            for c4 in range_constexpr(NOPE // 64):
                fx.ptr_store(nope[c4], rawb + (64 * c4 + 16 * g) // 4)
            for cc in range_constexpr(ROPE // 32):
                fx.ptr_store(rope[cc], rawb + (NOPE + (cc * 32 + 8 * g) * 2) // 4)
            if g == 0:
                fx.ptr_store(scw, rawb + DATA // 4)
        gpu.barrier()
        stamp(c, 5)
        # QK: MFMA K chunk 2 c + h is nope dims 64 c + 16 g + 8 h .., q read in the same order
        scale = a["qk_scale"].bitcast(fx.Float32)
        st = sl["st"]
        accs = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(nht)]
        for c4 in range_constexpr(NOPE // 64):
            ksc = pow2(_scale_byte(scw, c4))
            for h2 in range_constexpr(2):
                kw = fp8x8_bf16(nope[c4][2 * h2], nope[c4][2 * h2 + 1], ksc).bitcast(
                    fx.BFloat16
                )
                for ht in range_constexpr(nht):
                    qb = sl["q"] + ((hgi * nht + ht) * HEAD_TILE + r16) * Q_ROW
                    qw = fx.ptr_load(
                        qb + 32 * c4 + 8 * g + 4 * h2,
                        result_type=fx.Vector.make_type(4, fx.Int32),
                    )
                    accs[ht] = mfma_bf16(kw, qw.bitcast(fx.BFloat16), accs[ht])
        for cc in range_constexpr(ROPE // 32):
            kw = rope[cc].bitcast(fx.BFloat16)
            for ht in range_constexpr(nht):
                qb = sl["q"] + ((hgi * nht + ht) * HEAD_TILE + r16) * Q_ROW
                qw = fx.ptr_load(
                    qb + (NOPE + cc * 32 + 8 * g) // 2,
                    result_type=fx.Vector.make_type(4, fx.Int32),
                )
                accs[ht] = mfma_bf16(kw, qw.bitcast(fx.BFloat16), accs[ht])
        # scores of keys 4 g + i (C rows), head r16 of a tile (C column)
        ok = (key < kv_len) & okv
        okw = ok.select(fx.Int32(1), fx.Int32(0))
        valid = [lane_gather(okw, 4 * g + i) != 0 for i in range(4)]
        svs = []
        for ht in range_constexpr(nht):
            sv = [
                valid[i].select(accs[ht][i] * scale, fx.Float32(NEG)) for i in range(4)
            ]
            mw = butterfly(
                fx.max(fx.max(sv[0], sv[1]), fx.max(sv[2], sv[3])), (32, 16), fx.max
            )
            if g == 0:
                fx.ptr_store(mw, st + ((hgi * nht + ht) * nkt + kt) * HEAD_TILE + r16)
            svs.append(sv)
        gpu.barrier()
        lst = st + d.heads * nkt
        for ht in range_constexpr(nht):
            hti = hgi * nht + ht
            m = fx.ptr_load(st + (hti * nkt) * HEAD_TILE + r16)
            for j in range_constexpr(1, nkt):
                m = fx.max(m, fx.ptr_load(st + (hti * nkt + j) * HEAD_TILE + r16))
            sv = svs[ht]
            p = [valid[i].select(hw_exp2(sv[i] - m), fx.Float32(0.0)) for i in range(4)]
            lw = (p[0] + p[2]) + (p[1] + p[3])
            lw = lw + xshfl(lw, 32)
            lw = lw + xshfl(lw, 16)
            pw = (
                fx.Vector.from_elements(p, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
            )
            pbase = (
                sl["p"] + (hti * HEAD_TILE + r16) * p_row(d, s) + (kt * 16 + 4 * g) // 2
            )
            fx.ptr_store(fx.Vector.from_elements([pw[0], pw[1]], fx.Int32), pbase)
            if g == 0:
                fx.ptr_store(lw, lst + (hti * nkt + kt) * HEAD_TILE + r16)
        gpu.barrier()
        if tid < d.heads:
            hts = tid // HEAD_TILE
            hr = tid % HEAD_TILE
            mm = fx.ptr_load(st + (hts * nkt) * HEAD_TILE + hr)
            ll = fx.ptr_load(lst + (hts * nkt) * HEAD_TILE + hr)
            for j in range_constexpr(1, nkt):
                mm = fx.max(mm, fx.ptr_load(st + (hts * nkt + j) * HEAD_TILE + hr))
                ll = ll + fx.ptr_load(lst + (hts * nkt + j) * HEAD_TILE + hr)
            bo.buffer_store(
                fx.Vector.from_elements([mm, ll], fx.Float32),
                rsrc(c["pstat"]),
                ((t * nchunk + ch) * d.heads + tid) * 2,
                cache_modifier=CM_DEV,
            )
        # PV: wave w dims 64 w + 4 r16 + dt (an LDS word holds a key's four), lane g keys 4 g + j
        is_rope = wave == WAVES - 1
        dbase = 64 * wave
        sh = (wave % 4) * 8
        accs = [
            [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(4)]
            for _ in range(d.head_tiles)
        ]
        for kt2 in range_constexpr(nkt):
            rb0 = sl["raw"] + (kt2 * 16 + 4 * g) * (REC // 4)
            vv = fx.Vector.filled(16, 0.0, fx.Float32)  # [dt][j]
            if is_rope:
                rv = []
                for j in range_constexpr(4):
                    rw = fx.ptr_load(
                        rb0 + j * (REC // 4) + (NOPE + 8 * r16) // 4,
                        result_type=fx.Vector.make_type(2, fx.Int32),
                    )
                    rv.append([bf_lo(rw[0]), bf_hi(rw[0]), bf_lo(rw[1]), bf_hi(rw[1])])
                vv = fx.Vector.from_elements(
                    [rv[j][dt] for dt in range(4) for j in range(4)], fx.Float32
                )
            else:
                nv = []
                for j in range_constexpr(4):
                    rb = rb0 + j * (REC // 4)
                    w = fx.ptr_load(rb + (dbase + 4 * r16) // 4)
                    scb = fx.ptr_load(rb + DATA // 4 + wave // 4)
                    scl = pow2((scb >> sh) & 0xFF)
                    nv.append(
                        _fp8_pair_f32(w, scl, False) + _fp8_pair_f32(w, scl, True)
                    )
                vv = fx.Vector.from_elements(
                    [nv[j][dt] for dt in range(4) for j in range(4)], fx.Float32
                )
            for ht2 in range_constexpr(d.head_tiles):
                pb = fx.ptr_load(
                    sl["p"]
                    + (ht2 * HEAD_TILE + r16) * p_row(d, s)
                    + (kt2 * 16 + 4 * g) // 2,
                    result_type=fx.Vector.make_type(2, fx.Int32),
                ).bitcast(fx.Int16)
                for dt in range_constexpr(4):
                    av = (
                        fx.Vector.from_elements(
                            [vv[dt * 4 + j] for j in range(4)], fx.Float32
                        )
                        .to(fx.BFloat16)
                        .bitcast(fx.Int16)
                    )
                    accs[ht2][dt] = fx.Vector(
                        rocdl.mfma_f32_16x16x16bf16_1k(
                            T.vec(4, T.f32), [av, pb, accs[ht2][dt]]
                        )
                    )
        stamp(c, 6)
        # partial[t][ch][head][dim] bf16: lane (r16, g) owns 32 contiguous B, dims 64 w + 16 g ..
        for ht2 in range_constexpr(d.head_tiles):
            hh = ht2 * HEAD_TILE + r16
            words = []
            for i in range_constexpr(4):
                v4 = fx.Vector.from_elements(
                    [accs[ht2][dt][i] for dt in range(4)], fx.Float32
                )
                w2 = v4.to(fx.BFloat16).bitcast(fx.Int32)
                words += [w2[0], w2[1]]
            off = (((t * nchunk + ch) * d.heads + hh) * HEAD_DIM + dbase + 16 * g) // 2
            for hlf in range_constexpr(2):
                bo.buffer_store(
                    fx.Vector.from_elements(words[4 * hlf : 4 * hlf + 4], fx.Int32),
                    rsrc(c["part"]),
                    off + 4 * hlf,
                    cache_modifier=CM_DEV,
                )
        publish(c, c["srdy"], t * nchunk + ch)
        stamp(c, 7)


# ---------------------------------------------------------------- combine

COMB_HEADS = WAVES  # heads a combine unit: a wave each


@flyc.jit
def stage_combine(c, unit):
    """(token, 8 heads) ``unit``, a wave a head, 8 dims a lane: the chunks
    merged with the sink folded in, bf16, inverse RoPE, bf16 -> XO (the normal
    path's wo_a reads it unquantized), published (a flag a (token, head group))."""
    lane, wave, d, s = c["lane"], c["wave"], c["d"], c["S"]
    a = c["args"]
    hg_n = d.heads // COMB_HEADS
    t = unit // hg_n
    hg = unit % hg_n
    h = hg * COMB_HEADS + wave
    nchunk = c["nchunk"]
    ck = chunk_keys(d, s)
    pos = ld_i32(a["pos"], 2 * t)
    sink = ld_f32(a["sink"], h) * fx.Float32(LOG2E)
    d0 = lane * 8
    rope_lane = d0 >= NOPE
    jj = fx.max((d0 - NOPE) // 2, 0)  # 4 pairs from here
    cs = [
        rope_lane.select(ld_f32(a["cos_sin"], pos * ROPE + jj + i), fx.Float32(1.0))
        for i in range(4)
    ]
    sn = [
        rope_lane.select(
            ld_f32(a["cos_sin"], pos * ROPE + HALF + jj + i), fx.Float32(0.0)
        )
        for i in range(4)
    ]
    # the position, sink and RoPE above do not depend on the key table
    if const_expr("krdy" in c):
        # an index source's key count: written by this launch, past the CU's cache
        await_flags(c, c["krdy"], 1, lambda u: t)
    kv_len = ld_i32(a["klen"], t, CM_DEV if const_expr("krdy" in c) else 0)
    nlive = (kv_len + ck - 1) // ck
    await_flags(c, c["srdy"], nlive, lambda u: t * nchunk + u)
    m = fx.Float32(NEG)
    ms, ls, vs = [], [], []
    for cc in range_constexpr(nchunk):
        live = cc < nlive
        stv = fx.Vector(
            bo.buffer_load(
                rsrc(c["pstat"]),
                ((t * nchunk + cc) * d.heads + h) * 2,
                vec_width=2,
                dtype=T.f32,
            )
        )
        pv = fx.Vector(
            bo.buffer_load(
                rsrc(c["part"]),
                (((t * nchunk + cc) * d.heads + h) * HEAD_DIM + d0) // 2,
                vec_width=4,
                dtype=T.i32,
            )
        )
        mc = live.select(stv[0], fx.Float32(NEG))
        ms.append(mc)
        ls.append(live.select(stv[1], fx.Float32(0.0)))
        vals = []
        for e in range_constexpr(4):
            vals += [bf_lo(pv[e]), bf_hi(pv[e])]
        vs.append([live.select(v, fx.Float32(0.0)) for v in vals])
        m = fx.max(m, mc)
    m_final = fx.max(m, sink)
    lsum = hw_exp2(sink - m_final)
    tot = [fx.Float32(0.0)] * 8
    for cc in range_constexpr(nchunk):
        wgt = hw_exp2(ms[cc] - m_final)
        lsum = lsum + ls[cc] * wgt
        tot = [tot[e] + vs[cc][e] * wgt for e in range(8)]
    inv = fx.Float32(1.0) / lsum
    o = [bf16_round(tot[e] * inv) for e in range(8)]
    # inverse GPT-J RoPE of each pair: e' = e cos + o sin, o' = o cos - e sin
    y = []
    for i in range_constexpr(4):
        e, od = o[2 * i], o[2 * i + 1]
        y += [e * cs[i] + od * sn[i], od * cs[i] - e * sn[i]]
    words = [
        fx.Vector.from_elements([y[2 * i], y[2 * i + 1]], fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Int32)[0]
        for i in range(4)
    ]
    bo.buffer_store(
        fx.Vector.from_elements(words, fx.Int32),
        rsrc(c["xo"]),
        t * (d.q_rows // 2) + (h * HEAD_DIM + d0) // 2,
        cache_modifier=CM_DEV,
    )
    publish(c, c["crdy"], t * hg_n + hg)


# ---------------------------------------------------------------- wo_a / wo_b


@flyc.jit
def stage_woa(c, task, acc, t0, n):
    """Rows 32 task .. of wo_a from the waves' accumulators (``bf16_gemv_mfmas``)
    -> bf16 -> vLLM's MXFP8 of the 32 -> X8B, wo_b's input."""
    tid, red, d, lane = c["tid"], c["red"], c["d"], c["lane"]
    split = WOA_SPLIT_K
    fx.ptr_store(acc, red + (c["wave"] * 64 + lane) * 4)
    gpu.barrier()
    mine = tid < WOA_ROWS * n
    tl = fx.min(tid // WOA_ROWS, n - 1)
    t = plus(t0, tl)
    row = tid % WOA_ROWS
    tile = row // ROWS
    y = bf16_round(
        row_sum(red, row % ROWS, tl, [tile * split + w for w in range(split)])
    )
    amax = butterfly(abs(y), (1, 2, 4, 8, 16), fx.max)
    code = mx_code(amax)
    q = clamp_fp8(y * mx_mul(code))
    w = fp8_pack4(
        q,
        *[
            lane_gather(q.bitcast(fx.Int32), fx.min(lane + k, 63)).bitcast(fx.Float32)
            for k in (1, 2, 3)
        ],
    )
    if mine & (row % 4 == 0):
        bo.buffer_store(
            w,
            rsrc(c["x8b"]),
            t * (d.o_rows // 4) + (task * WOA_ROWS + row) // 4,
            cache_modifier=CM_DEV,
        )
    if mine & (row == 0):
        bo.buffer_store(
            code, rsrc(c["x8bs"]), t * (d.o_rows // 32) + task, cache_modifier=CM_DEV
        )
    gpu.barrier()


def wob_loads(c, task, pred=None):
    a = c["args"]
    return gemv_loads(
        c,
        a["wob"],
        a["wob_s"],
        c["d"].o_rows,
        task * ROWS,
        pred=pred,
        shuffled=c["shuffled"],
    )


def _wob_store(a, t, col, v0, v1):
    word = (
        fx.Vector.from_elements([v0, v1], fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Int32)[0]
    )
    bo.buffer_store(word, rsrc(a["out"]), (t * a["out_stride"] + col) // 2)


@flyc.jit
def stage_wob(c, task, ops, task2, ops2, has2, t0, n, mark=False):
    """Rows 16 task .. (and 16 task2 .., ``has2``) of wo_b -> bf16 -> out, or
    (``c["wob_out"]``, the mono layer) the caller's hook with each bf16 pair."""
    tid, red = c["tid"], c["red"]
    o_rows = c["d"].o_rows
    a = c["args"]
    gemv_mfmas(c, o_rows, c["xl"], c["xsl"], red, ops, rows=n)
    gemv_mfmas(c, o_rows, c["xl"], c["xsl"], red + RED, ops2, rows=n)
    gpu.barrier()
    if const_expr(mark):
        stamp(c, 15)
    per = ROWS * n // 2  # bf16 pairs a task
    k = tid // per
    emit = c.get("wob_out") or (lambda t, col, v0, v1: _wob_store(a, t, col, v0, v1))
    if (tid < 2 * per) & ((k == 0) | has2):
        u = tid % per
        rp = u % (ROWS // 2)
        tl = u // (ROWS // 2)
        t = plus(t0, tl)
        rk = red + k * RED
        col = (k == 0).select(task, task2) * ROWS + 2 * rp
        v0 = row_sum(rk, 2 * rp, tl)
        v1 = row_sum(rk, 2 * rp + 1, tl)
        emit(t, col, v0, v1)
    gpu.barrier()


# ---------------------------------------------------------------- kernel


def woa_units(s: int, d: Dims) -> int:
    """wo_a's units: past one token tile, a (task, tile) each where they fit the
    CTAs, else a task each.

    ROCm/ATOM's tile tasks: a task's tiles in series on one CTA would leave most
    CTAs idle through wo_a, and a tile's reread of its task's weights hits L2."""
    nt = len(token_tiles(s))
    return d.woa_tasks * nt if nt > 1 and d.woa_tasks * nt <= BLOCKS else d.woa_tasks


def back_plan(key):
    """(nchunk, n_split, n_comb, COMB0, WOA0, WOB0, early_wob) of a build."""
    s, d = key.tokens, Dims(key.tp)
    ck = chunk_keys(d, s)
    nchunk = cdiv(KEYS, ck)
    n_split = s * nchunk
    n_comb = s * (d.heads // COMB_HEADS)
    COMB0 = n_split % BLOCKS
    assert d.woa_tasks <= BLOCKS
    assert BLOCKS < d.wob_tasks <= 2 * BLOCKS
    # GEMV units past the attention units, so CTAs without attention load weights first
    WOA0 = (n_split + n_comb) % BLOCKS
    WOB0 = (WOA0 + woa_units(s, d)) % BLOCKS
    # loads complete in issue order, so wo_b's weights go early only if wo_a CTAs do no attention
    early_wob = n_split + n_comb + woa_units(s, d) <= BLOCKS
    return nchunk, n_split, n_comb, COMB0, WOA0, WOB0, early_wob


def back_lds_members(s: int, d: Dims) -> dict:
    """K2's attention LDS structs: the split stage's and the GEMVs' (a union's
    members: one at a time)."""
    rows = tile_rows(s)
    ck = chunk_keys(d, s)

    @fx.struct
    class SplitLds:
        q: fx.Array[fx.Int32, d.heads * Q_ROW, 16]
        raw: fx.Array[fx.Int32, ck * REC // 4, 16]
        p: fx.Array[fx.Int32, d.heads * p_row(d, s), 16]
        st: fx.Array[fx.Float32, 2 * d.heads * (WAVES // head_groups(d, s)), 16]

    @fx.struct
    class GemvLds:
        red: fx.Array[fx.Float32, 2 * RED, 16]
        xl: fx.Array[fx.Int32, rows * lds_row(d.group_k // 4), 16]
        xsl: fx.Array[fx.Int32, rows * lds_row(d.group_k // 32), 16]

    return {"split": SplitLds, "gemv": GemvLds}


def back_lds(s: int, d: Dims):
    """K2's attention LDS: the union of ``back_lds_members``."""
    m = back_lds_members(s, d)
    SplitLds, GemvLds = m["split"], m["gemv"]

    @fx.union
    class BackLds:
        split: SplitLds  # type: ignore[valid-type]
        gemv: GemvLds  # type: ignore[valid-type]

    return BackLds


def back_context(key, lds, args: dict, scratch, tag, layout: dict) -> dict:
    """The attention back's context: LDS views (``lds``: an allocated
    ``back_lds``, or a dict of the members' views), mailbox (tag ``tag``),
    scratch regions, arguments by name."""
    s, d = key.tokens, Dims(key.tp)
    tid = fx.thread_idx.x
    if isinstance(lds, dict):
        spl, gml = lds["split"], lds["gemv"]
    else:
        spl, gml = lds.split.peek(), lds.gemv.peek()
    c = {
        "S": s,
        "tid": tid,
        "bid": fx.block_idx.x,
        "lane": tid % 64,
        "wave": tid // 64,
        "red": gml.red.ptr,
        "xl": gml.xl.ptr,
        "xsl": gml.xsl.ptr,
        "d": d,
        "ratio": key.ratio,
        "shuffled": key.shuffled,
        "mb": Mailbox(tag),
        "nchunk": back_plan(key)[0],
        "sl": {"q": spl.q.ptr, "raw": spl.raw.ptr, "p": spl.p.ptr, "st": spl.st.ptr},
        "args": args,
    }
    for region, (off, _) in layout.items():
        c[region] = scratch + fx.Int64(off)
    return c


@flyc.jit
def run_back(c, key, bid):
    """The attention back's stages on this CTA: split, combine, wo_a, wo_b."""
    s = key.tokens
    dd = Dims(key.tp)
    # plain ints: a Python object read inside traced control flow is carried
    woa_tasks, groups, heads, group_k = dd.woa_tasks, dd.groups, dd.heads, dd.group_k
    q_rows, o_rows, wob_tasks = dd.q_rows, dd.o_rows, dd.wob_tasks
    a = c["args"]
    tiles = token_tiles(s)
    _, n_split, n_comb, COMB0, WOA0, WOB0, early_wob = back_plan(key)
    stamp(c, 0)
    for unit in range(first_task(bid, 0), n_split, BLOCKS):
        stage_split(c, unit)
    stamp(c, 1)
    for unit in range(first_task(bid, COMB0), n_comb, BLOCKS):
        stage_combine(c, unit)
    gpu.barrier()
    stamp(c, 2)
    # the GEMV weights right after the attention work: the wo_a unit's (if any), then wo_b's
    nt = len(tiles)
    units = woa_units(s, dd)
    woa_spread = units > woa_tasks
    woa_unit = first_task(bid, WOA0)
    has_woa = woa_unit < units
    woa_t = fx.min(woa_unit % woa_tasks, woa_tasks - 1)
    ops = bf16_gemv_loads(
        c, a["woa"], a["woa_s"], group_k, woa_t * WOA_ROWS, WOA_SPLIT_K, pred=has_woa
    )
    wob_unit = first_task(bid, WOB0)
    if const_expr(nt == 1):
        # one or two wo_b tasks a CTA, every one on the step's rows
        wob_tile = 0
        wob_ts = [wob_unit, wob_unit + BLOCKS]
    else:
        # a tile a CTA (ROCm/ATOM's tile tasks): rows read once, every task's weights in flight
        wob_tile = wob_unit % nt
        per_tile = (BLOCKS - wob_tile + nt - 1) // nt
        most = -(-wob_tasks // (BLOCKS // nt))
        wob_ts = [wob_unit // nt + k * per_tile for k in range(most)]

    def wob_weights():
        # one tile: the first task is every CTA's (BLOCKS < wob_tasks), unmasked
        return [
            wob_loads(c, t)
            if nt == 1 and k == 0
            else wob_loads(c, fx.min(t, wob_tasks - 1), pred=t < wob_tasks)
            for k, t in enumerate(wob_ts)
        ]

    if const_expr(early_wob):
        wops = wob_weights()
    if const_expr(c.get("tl") is not None):
        rocdl.s_waitcnt(vmcnt=0)
        gpu.barrier()
    stamp(c, 8)
    half_w = group_k // 4

    def woa_tile(ti, n_tokens):
        # wo_a on token tile ``ti`` (its first ``n_tokens`` rows' flags awaited)
        t0, n = tiles[ti]
        grp = woa_t // (woa_tasks // groups)
        await_flags(
            c,
            c["crdy"],
            n_tokens,
            lambda u: (t0 + u) * (heads // COMB_HEADS) + grp,
        )
        stamp(c, 9)
        # the group's XO rows a K half at a time, the next half's loads in flight
        off = [(grp * group_k + h * group_k // 2) // 2 for h in range(2)]
        regs = words_loads(c, c["xo"], q_rows // 2, off[0], half_w, t0, n)
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        for h in range_constexpr(2):
            words_store(c, regs, half_w, c["xl"], n)
            if const_expr(h == 0):
                regs = words_loads(c, c["xo"], q_rows // 2, off[1], half_w, t0, n)
            acc = bf16_gemv_mfmas(c, group_k, c["xl"], ops, acc, h, n)
            gpu.barrier()  # the next half's rows overwrite the LDS
        stage_woa(c, woa_t, acc, t0, n)

    if has_woa:
        if const_expr(woa_spread):
            # this unit's tile of its task
            for ti in range_constexpr(nt):
                if woa_unit // woa_tasks == ti:
                    woa_tile(ti, tiles[ti][1])
            stamp(c, 10)
            publish(c, c["ardy"], woa_unit)
        else:
            # every token tile in turn, the first waiting on every row's flags
            for ti in range_constexpr(nt):
                woa_tile(ti, s if ti == 0 else 0)
            stamp(c, 10)
            publish(c, c["ardy"], woa_t)
    stamp(c, 3)
    if const_expr(not early_wob):
        wops = wob_weights()
    stamp(c, 11)
    if const_expr(woa_spread):
        # this tile's wo_a units
        await_flags(c, c["ardy"], woa_tasks, lambda u: wob_tile * woa_tasks + u)
    else:
        await_flags(c, c["ardy"], woa_tasks, lambda u: u)
    stamp(c, 12)
    x8b = (
        c["x8b"],
        c["x8bs"],
        o_rows // 4,
        0,
        o_rows // 4,
        o_rows // 32,
        0,
        o_rows // 32,
    )

    def wob_tile_run(ti):
        # every wo_b task of this CTA on token tile ``ti``, two at a time
        t0, n = tiles[ti]
        rows_store(
            c,
            rows_loads(c, *x8b, t0, n),
            o_rows // 4,
            o_rows // 32,
            c["xl"],
            c["xsl"],
            n,
        )
        stamp(c, 14)
        for k in range_constexpr(0, len(wob_ts), 2):
            k2 = min(k + 1, len(wob_ts) - 1)
            # an odd last task runs alone: its pair is itself, masked off
            has2 = wob_ts[k2] < (wob_tasks if k2 > k else 0)
            run = (
                c,
                wob_ts[k],
                wops[k],
                fx.min(wob_ts[k2], wob_tasks - 1),
                wops[k2],
                has2,
                t0,
                n,
            )
            if const_expr(nt == 1 and k == 0):
                stage_wob(*run, mark=True)
            else:
                if wob_ts[k] < wob_tasks:
                    stage_wob(*run, mark=k == 0)
        stamp(c, 13)

    for ti in range_constexpr(nt):
        if wob_tile == ti:
            wob_tile_run(ti)
    stamp(c, 4)
