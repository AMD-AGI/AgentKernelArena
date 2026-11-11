# SPDX-License-Identifier: Apache-2.0
"""The MoE front's tail inside the layer launch: what the routed experts and the
next layer's MoE tail need from the front GEMM.

A rank's front GEMM rows (``chain``'s front, merged in f32 here) are its 112
router experts | 448 latent-down rows | 768 shared gate + 768 shared up. Its
finished tiles go three ways:

router  f32 logits -> every rank's gather region [m, 896] (expert e of rank r at
        112 r + e)
latent  bf16 -> every rank's gather region [m, 3584]
shared  bf16 gate / up -> local scratch

then, a row a task: ``route_row`` copies the gathered latent into the experts'
input and picks the top 16 (SGLang's noaux_tc: rank by sigmoid(logit) + bias,
weights sigmoid(logit) renormalized); ``situ_row`` applies SiTU to the shared
gate / up (bf16, as the shared MLP rounds), and the shared down GEMM follows.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from . import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from .compat import global_atomic_add, lds_atomic_or
from .mx import clamp_fp8, code_ceil, rcp_pow2
from .ops import (
    CM_DEV,
    CM_SYS,
    bf2_f32,
    bf16_pair,
    block_excl_scan,
    div_rn,
    fp8_pack4,
    hw_exp2,
    hw_rcp,
    ld_i32,
    popcount,
    read_lane_i32,
    rsrc,
    wave_umax,
    write_lane_i32,
    xshfl,
)
from .plan import THREADS
from .sync import _spin
from .kda import ld_f32
from . import coll

TP = 8
E, E_R = 896, 112  # experts, a rank's
LAT, L_R = 3584, 448
SI = 768  # shared intermediate a rank (2 shared experts x 3072 / TP)
R0, L0, S0 = 0, E_R, E_R + L_R  # column starts in a rank's front rows
NF = E_R + L_R + 2 * SI  # 2096
TOPK = 16
LOG2E = 1.4426950408889634


def gathered_tiles(bn):
    """Front tiles holding router / latent columns (0 .. S0)."""
    return (S0 + bn - 1) // bn


# ---- the routed experts' sort made in the launch (``sort`` builds), two kinds:
# "sgl"   bit for bit what SGLang's small-sort Triton kernel (sglang/kernels/ops/moe/moe_sorting_small.py,
#         _moe_sorting_small_kernel_sorted with EMIT_MX) hands aiter's 2-stage MoE: BM = 32 blocks, experts
#         ascending, a token's pair once per expert, plus the mxfp8 quant of the latent (pad id (TOPK << 24) | m).
# "aux16" aiter's BM = 16 aux sort (aiter.presorted_moe.register_presorted_moe, upstream mxmoe rows): the same
#         order in BM = 16 blocks, pad id m, plus m_indices (sorted row -> token, pad m) and reverse_sorted
#         ((token, slot) -> sorted row, = ``dest``); no quant (aiter's stage 1 quantizes the bf16 latent inline).
SORT_KINDS = ("sgl", "aux16")
SORT_BM = 32  # "sgl"
SORT_MAX_ROWS = 64  # a 64-bit token mask an expert (SGLang's patch covers M x 16 <= 1024 pairs: M <= 64)
SCALE_COLS = (
    (LAT // 32 + 7) // 8
) * 8  # 112 e8m0 bytes a sorted row (SGLang's SCALEN_PAD)


def sort_bm(kind):
    assert kind in SORT_KINDS, kind
    return 16 if kind == "aux16" else SORT_BM


def sort_pad(kind, m):
    """The sorted-id word of a pad row."""
    return m if kind == "aux16" else (TOPK << 24) | m


def sort_layout(m, kind="sgl"):
    """Byte offsets in the sort buffer of bucket m, and its size. "sgl": SGLang's buffer sizes for M = m;
    "aux16": aiter's (``aux_sort_capacity``: every pair + BM - 1 pad rows an active expert)."""
    if kind == "aux16":
        p = m * TOPK
        bm = sort_bm(kind)
        maxp = (p + min(E, p) * (bm - 1) + bm - 1) // bm * bm
        sizes = dict(
            ids=4 * maxp,  # int32 (slot << 24) | token, pad m
            w=4 * maxp,  # f32, pad 0
            eids=4 * (maxp // bm),  # int32 expert a block
            nv=8,  # int32 [blocks x 16, m]
            mi=4 * maxp,  # int32 m_indices: token, pad m
            dest=4
            * p,  # int32 reverse_sorted: sorted row of pair p (the next launch resets those rows)
        )
        offs, at = {}, 0
        for k, n in sizes.items():
            offs[k] = at
            at += (n + 255) // 256 * 256
        return dict(offs=offs, maxp=maxp, nblk=maxp // bm, nbytes=at, kind=kind)
    p = m * TOPK
    maxp = p + E * SORT_BM - TOPK  # max_num_tokens_padded
    nblk = (maxp + SORT_BM - 1) // SORT_BM
    srows = ((maxp + 31) // 32) * 32
    sizes = dict(
        ids=4 * maxp,  # int32 (slot << 24) | token, pad (TOPK << 24) | m
        w=4 * maxp,  # f32, pad 0
        eids=4 * nblk,  # int32 expert a block
        nv=8,  # int32 [blocks x 32, m]
        q=m * LAT,  # fp8 e4m3fn [m, 3584], token-major
        qs=srows
        * SCALE_COLS,  # e8m0 [srows, 112], aiter's shuffle_scale layout, by sorted row
        ts=m * SCALE_COLS,  # e8m0 [m, 112] by token (the launch's own staging)
        dest=4
        * p,  # int32 sorted row of pair p (this launch's; the next one resets those rows to pad)
    )
    offs, at = {}, 0
    for k, n in sizes.items():
        offs[k] = at
        at += (n + 255) // 256 * 256
    return dict(offs=offs, maxp=maxp, nblk=nblk, srows=srows, nbytes=at, kind=kind)


SORT_HELPERS = 32  # CTAs that place the e8m0 codes (21.5 K byte stores at m = 12: too many for one CTA)


def sort_init(buf, m, kind="sgl"):
    """The sort buffer's resting state (host, once): every sorted row a pad row (``sort_pad``), weight 0
    (aux16: m_indices m). Each launch then turns its pairs' rows real and the next one turns them back
    (``sort_rows``), so the rows below num_valid are always this launch's pairs or pads, with no pad pass."""
    import torch

    ly = sort_layout(m, kind)
    o = ly["offs"]
    buf.zero_()
    buf[o["ids"] : o["ids"] + 4 * ly["maxp"]].view(torch.int32).fill_(sort_pad(kind, m))
    if kind == "aux16":
        buf[o["mi"] : o["mi"] + 4 * ly["maxp"]].view(torch.int32).fill_(m)


def _quant8(v):
    """Eight bf16 (an i32 x4 vector) of a 32-column group whose 4 quarters sit in 4 consecutive lanes ->
    (two dwords of E4M3FN bytes, the group's e8m0 code), as SGLang's _sorted_mx_quant: amax >= 1e-10, sf = amax
    x (1 / 448), code rounded up to a power of two, q = clamp(x / 2^(code - 127), +-448), RNE."""
    xs = [x for w in (v[0], v[1], v[2], v[3]) for x in bf2_f32(w)]
    amax = fx.Float32(0.0)
    for x in xs:
        amax = fx.max(
            amax, (x.bitcast(fx.Int32) & fx.Int32(0x7FFFFFFF)).bitcast(fx.Float32)
        )
    amax = fx.max(amax, xshfl(amax, 1))
    amax = fx.max(amax, xshfl(amax, 2))
    amax = fx.max(amax, fx.Float32(1e-10))
    code = code_ceil(amax * fx.Float32(1.0 / 448.0))
    r = rcp_pow2(code)  # x / 2^k == x x 2^-k exactly
    q = [clamp_fp8(x * r) for x in xs]
    return fp8_pack4(q[0], q[1], q[2], q[3]), fp8_pack4(q[4], q[5], q[6], q[7]), code


@flyc.jit
def finish_front(c, fbuf, sgu, m, r0, rb, n0, bn, band, t, nbf, smark, tnf):
    """Rows r0 .. r0 + rb of finished front tile t (columns n0 ..): router / latent
    columns to every rank, shared columns to local scratch; zero the tile; flag
    (this rank, band, t) on every rank and / or mark (band, t) locally."""
    tid = c["tid"]
    rank = c["rank"]
    q4 = bn // 4
    rgs = [c["rgat_of"](p) for p in range(TP)]
    lgs = [c["lgat_of"](p) for p in range(TP)]
    for it in range(tid, rb * q4, THREADS):
        row = r0 + it // q4
        col = n0 + 4 * (it % q4)
        if (row < m) & (col < NF):
            v = fx.Vector(
                bo.buffer_load(
                    rsrc(fbuf),
                    row * NF + col,
                    vec_width=4,
                    dtype=T.f32,
                    cache_modifier=CM_DEV,
                )
            )
            bo.buffer_store(
                fx.Vector.filled(4, 0.0, fx.Float32),
                rsrc(fbuf),
                row * NF + col,
                cache_modifier=CM_DEV,
            )
            words = fx.Vector.from_elements(
                [
                    bf16_pair(v[0], v[1]).bitcast(fx.Int32),
                    bf16_pair(v[2], v[3]).bitcast(fx.Int32),
                ],
                fx.Int32,
            )
            if col < L0:
                for p in range_constexpr(TP):
                    bo.buffer_store(
                        v,
                        rsrc(rgs[p].value),
                        row * E + rank * E_R + col,
                        cache_modifier=CM_SYS,
                    )
            if (col >= L0) & (col < S0):
                for p in range_constexpr(TP):
                    bo.buffer_store(
                        words,
                        rsrc(lgs[p].value),
                        (row * LAT + rank * L_R + col - L0) // 2,
                        cache_modifier=CM_SYS,
                    )
            if col >= S0:
                bo.buffer_store(
                    words,
                    rsrc(sgu),
                    (row * 2 * SI + col - S0) // 2,
                    cache_modifier=CM_DEV,
                )
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    # a partial tile's pieces past row m hold no rows: their band (>= nbf) would flag another rank's
    if (n0 < S0) & (r0 < m):
        coll.put_each_peer(
            c, [("fgf", (rank * nbf + band) * gathered_tiles(bn) + t, fx.Int32(1))]
        )
    if (n0 + bn > S0) & (tid == 0) & (r0 < m):
        c["put"](smark, band * tnf + t, fx.Int32(1))


def _ld_sys(r, i):
    """One f32 of a peer region other ranks wrote (system scope: no stale cached line)."""
    return fx.Float32(
        bo.buffer_load(r, i, vec_width=1, dtype=T.f32, cache_modifier=CM_SYS)
    )


def _sigmoid(x):
    return hw_rcp(1.0 + hw_exp2(x * (-LOG2E)))


def _sigmoid_aiter(x):
    """aiter topk_reg_kernel's sigmoid bit for bit: ``rcpf(1.0f + exp2f(-C_LOG2E * x))`` with C_LOG2E a double
    literal, so the exponent is an fp64 product rounded once to f32 (an f32 product of the rounded constant
    differs by an ulp now and then: a weight off by ~1e-7 and, at a near tie, another expert)."""
    a = fx.Float32(fx.Float64(x) * fx.Float64(-1.44269504088896340736))
    return hw_rcp(1.0 + hw_exp2(a))


def _batcher(n):
    """Batcher's odd-even merge sort network on n (a power of two) slots: (a, b) pairs."""
    pairs = []
    p = 1
    while p < n:
        k = p
        while k >= 1:
            for j in range(k % p, n - k, 2 * k):
                for i in range(min(k, n - j - k)):
                    if (i + j) // (2 * p) == (i + j + k) // (2 * p):
                        pairs.append((i + j, i + j + k))
            k //= 2
        p *= 2
    return pairs


SLOTS = 16  # a lane's 14 candidates (896 / 64) padded to the network's width
NET = _batcher(SLOTS)


def _okey(v):
    """An order-preserving uint32 key of an f32 (larger float, larger key)."""
    b = v.bitcast(fx.Int32)
    return fx.Uint32((b < fx.Int32(0)).select(~b, b ^ fx.Int32(-2147483648)))


def _tie_rank(e):
    """aiter's ``legacy_topk_tie_rank`` (csrc/kernels/topk_softmax_kernels_group.cu): the order SGLang's top-k
    (``topk_reg_kernel``) breaks an exact score tie in, lower first; a bijection on 0 .. 895 into 0 .. 2047."""
    v4 = e >> 2
    ln = v4 & 63
    low2 = (ln & 3) ^ ((ln & 4) != 0).select(fx.Int32(3), fx.Int32(0))
    bit2 = (((ln >> 2) ^ (ln >> 3)) & 1) << 2
    lrank = ((~ln) & 0x38) | bit2 | low2
    return (lrank << 5) | (((v4 >> 6) << 2) | (e & 3))


@flyc.jit
def route_row(
    c,
    row,
    m,
    rbf,
    nbf,
    bn,
    bias,
    lat_out,
    topk_ids,
    topk_w,
    logits_out=None,
    qout=None,
    tscale=None,
    moe_out=None,
    cnt=None,
    cnt_word=0,
):
    """The row's gathered latent -> the experts' input; top-16 of the gathered logits
    (and the logits themselves -> ``logits_out`` f32 [m, 896], for SGLang's own top-k).
    ``moe_out`` (sort builds): also the row of ``moe_out`` bf16 [m, 3584] zeroed (stage 2 accumulates), ids /
    weights at device scope, then one count in ``cnt`` word ``cnt_word`` (``sort_rows`` waits for m).
    ``qout`` ("sgl" sort builds): also the row's mxfp8 quant -> ``qout`` fp8 [m, 3584] and its e8m0 codes ->
    ``tscale`` [m, 112] by token."""
    tid = c["tid"]
    lane, wave = tid % 64, tid // 64
    nt = gathered_tiles(bn)
    band = row // rbf
    if tid < TP * nt:
        c["poll"]([(c["fgf_own"], ((tid // nt) * nbf + band) * nt + tid % nt, 1)])
    gpu.barrier()
    lg = rsrc(c["lgat_own"].value)
    rg = rsrc(c["rgat_own"].value)
    sorted_build = moe_out is not None
    cm_rt = CM_DEV if const_expr(sorted_build) else 0
    for u in range(tid, LAT // 8, THREADS):
        v = fx.Vector(
            bo.buffer_load(
                lg,
                (row * LAT + 8 * u) // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_SYS,
            )
        )
        bo.buffer_store(v, rsrc(lat_out), (row * LAT + 8 * u) // 2)
        if const_expr(qout is not None):
            # LAT // 8 = 448 lanes, one round: a 32-column group is 4 consecutive lanes of one wave
            w0, w1, code = _quant8(v)
            bo.buffer_store(
                fx.Vector.from_elements([w0, w1], fx.Int32),
                rsrc(qout),
                (row * LAT + 8 * u) // 4,  # an i32 element index: 1 byte an fp8
            )
            if u % 4 == 0:
                bo.buffer_store(
                    fx.Int8(code),
                    rsrc(tscale),
                    row * SCALE_COLS + u // 4,
                    cache_modifier=CM_DEV,
                )
        if const_expr(sorted_build):
            z = fx.Int32(0)
            bo.buffer_store(
                fx.Vector.from_elements([z, z, z, z], fx.Int32),
                rsrc(moe_out),
                (row * LAT + 8 * u) // 2,
            )
    if const_expr(logits_out is not None):
        for e in range(tid, E, THREADS):
            bo.buffer_store(_ld_sys(rg, row * E + e), rsrc(logits_out), row * E + e)
    if wave == 0:
        # max-extraction (as the K3 K2 launch, mono/kimi-k31 05dfa48): each lane sorts its 14 candidates (expert
        # lane + 64 i; key descending, an exact tie by aiter's legacy tie rank, as SGLang's topk_reg_kernel), then
        # 16 rounds of a wave max pop the heads; pick k lands in lane k, ranked. Weights: sigmoid / sum in pick
        # order with an IEEE 1 / sum, as aiter's emit_legacy_ordered_topk.
        per = E // 64
        ks, ids, sg, rk = [], [], [], []
        for i in range_constexpr(per):
            e = lane + 64 * i
            s = _sigmoid_aiter(_ld_sys(rg, row * E + e))
            ks.append(_okey(s + ld_f32(bias, e)))
            ids.append(fx.Int32(e))
            sg.append(s.bitcast(fx.Int32))
            rk.append(_tie_rank(fx.Int32(e)))
        ks += [fx.Uint32(0)] * (SLOTS - per)
        ids += [fx.Int32(1023)] * (SLOTS - per)
        sg += [fx.Int32(0)] * (SLOTS - per)
        rk += [fx.Int32(4095)] * (SLOTS - per)
        for x, y in NET:
            first = (ks[x] > ks[y]) | ((ks[x] == ks[y]) & (rk[x] < rk[y]))
            kx, ky, ix, iy, sx, sy = ks[x], ks[y], ids[x], ids[y], sg[x], sg[y]
            rx, ry = rk[x], rk[y]
            ks[x], ks[y] = first.select(kx, ky), first.select(ky, kx)
            ids[x], ids[y] = first.select(ix, iy), first.select(iy, ix)
            sg[x], sg[y] = first.select(sx, sy), first.select(sy, sx)
            rk[x], rk[y] = first.select(rx, ry), first.select(ry, rx)
        ks = [fx.Int32(k) for k in ks] + [fx.Int32(0)]
        ids = ids + [fx.Int32(1023)]
        sg = sg + [fx.Int32(0)]
        rk = rk + [fx.Int32(4095)]
        tot, sel_i, sel_s = fx.Float32(0.0), fx.Int32(0), fx.Int32(0)
        for k in range_constexpr(TOPK):
            mx = wave_umax(ks[0])
            # among the heads at the max key, the lowest tie rank; rank and expert packed in one word
            best = wave_umax(
                (ks[0] == mx).select(
                    ((fx.Int32(4095) - rk[0]) << 10) | ids[0], fx.Int32(0)
                )
            )
            expert = best & 1023
            sig = read_lane_i32(
                sg[0], expert % 64
            )  # the winner's sigmoid, before its lane pops its head
            hit = (ks[0] == mx) & (ids[0] == expert)
            ks = [hit.select(ks[i + 1], ks[i]) for i in range(SLOTS)] + [ks[-1]]
            ids = [hit.select(ids[i + 1], ids[i]) for i in range(SLOTS)] + [ids[-1]]
            sg = [hit.select(sg[i + 1], sg[i]) for i in range(SLOTS)] + [sg[-1]]
            rk = [hit.select(rk[i + 1], rk[i]) for i in range(SLOTS)] + [rk[-1]]
            tot = tot + sig.bitcast(fx.Float32)
            sel_i = write_lane_i32(expert, k, sel_i)
            sel_s = write_lane_i32(sig, k, sel_s)
        inv = div_rn(
            fx.Float32(1.0), tot, hw_rcp(tot)
        )  # IEEE 1 / sum (aiter: routed_scaling_factor / total)
        if lane < TOPK:
            bo.buffer_store(
                sel_i, rsrc(topk_ids), row * TOPK + lane, cache_modifier=cm_rt
            )
            bo.buffer_store(
                sel_s.bitcast(fx.Float32) * inv,
                rsrc(topk_w),
                row * TOPK + lane,
                cache_modifier=cm_rt,
            )
    if const_expr(sorted_build):
        # the row's ids / weights / codes out of this CTA before it counts (sort_rows reads them on one CTA)
        rocdl.s_waitcnt(vmcnt=0)
        gpu.barrier()
        if tid == 0:
            global_atomic_add(cnt + fx.Int64(cnt_word * 4), fx.Int32(1), syncscope="agent")
    gpu.barrier()


def _situ(g, u):
    # SiTU(g, u) = 2 tanh(g / 4) (1 + tanh(g / 2)) x 25 tanh(u / 25)  (beta 4, linear beta 25)
    def tanh(x):
        e2 = hw_exp2(x * (2.0 * LOG2E))
        return (e2 - 1.0) * hw_rcp(e2 + 1.0)

    return (2.0 * tanh(g * 0.25)) * (1.0 + tanh(g * 0.5)) * (25.0 * tanh(u * 0.04))


@flyc.jit
def situ_row(c, row, m, rbf, tnf, bn, sgu, smark, hbuf, hmark):
    """SiTU of the row's shared gate / up (bf16 in, bf16 out) -> hbuf; mark the row."""
    tid = c["tid"]
    t0 = S0 // bn
    band = row // rbf
    if tid < tnf - t0:
        c["poll"]([(smark, band * tnf + t0 + tid, 1)])
    gpu.barrier()
    for u in range(tid, SI // 4, THREADS):
        gv = fx.Vector(
            bo.buffer_load(
                rsrc(sgu),
                (row * 2 * SI + 4 * u) // 2,
                vec_width=2,
                dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        )
        uv = fx.Vector(
            bo.buffer_load(
                rsrc(sgu),
                (row * 2 * SI + SI + 4 * u) // 2,
                vec_width=2,
                dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        )
        gs = [x for w in (gv[0], gv[1]) for x in bf2_f32(w)]
        us = [x for w in (uv[0], uv[1]) for x in bf2_f32(w)]
        h = [_situ(gs[e], us[e]) for e in range(4)]
        words = fx.Vector.from_elements(
            [
                bf16_pair(h[0], h[1]).bitcast(fx.Int32),
                bf16_pair(h[2], h[3]).bitcast(fx.Int32),
            ],
            fx.Int32,
        )
        bo.buffer_store(
            words, rsrc(hbuf), (row * SI + 4 * u) // 2, cache_modifier=CM_DEV
        )
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if tid == 0:
        c["put"](hmark, row, fx.Int32(1))


@flyc.jit
def sort_rows(
    c,
    tid,
    m,
    cnt,
    cnt_word,
    lc_word,
    topk_ids,
    topk_w,
    sbuf,
    lay,
    srtmk,
    masks,
    bbl,
    scan_a,
    scan_b,
):
    """One CTA (sort builds): once the m rows of this launch are routed (``route_row`` counts each in ``cnt``
    word ``cnt_word``; this CTA counts its launches in word ``lc_word``, so the target is launches x m), the
    routed experts' sort in ``sbuf`` (``sort_layout(m)``) exactly as SGLang's small-sort kernel: experts
    ascending, a block of SORT_BM rows one expert, an expert's pairs by token (a token picks an expert at most
    once, so pair order = token order) then pad rows (TOPK << 24) | m / weight 0; num_valid = [blocks x
    SORT_BM, m]. Pads are never written in bulk: the buffer rests all pad (``sort_init``), this launch first
    turns the previous launch's pair rows (``dest``) back to pad, then writes its own and their rows -> ``dest``,
    then puts ``srtmk`` for ``place_codes``. Every global word it reads is loaded up front (m is static).
    ``lay["kind"]`` "aux16": aiter's BM16 aux sort instead (``sort_layout``): blocks of 16, pad id m, and
    m_indices (``mi``) real / reset alongside the ids; ``dest`` is its reverse_sorted; no ``srtmk``.
    LDS: ``masks`` 2 x E words (an expert's 64-bit token mask), ``bbl`` E words, ``scan_a`` / ``scan_b``
    WAVES words each."""
    lane, wave = tid % 64, tid // 64
    p_n = m * TOPK
    o = lay["offs"]
    aux16 = lay["kind"] == "aux16"
    bm = sort_bm(lay["kind"])
    a_ids = sbuf + fx.Int64(o["ids"])
    a_w = sbuf + fx.Int64(o["w"])
    a_eids = sbuf + fx.Int64(o["eids"])
    a_nv = sbuf + fx.Int64(o["nv"])
    a_dest = sbuf + fx.Int64(o["dest"])
    a_mi = sbuf + fx.Int64(o.get("mi", 0))  # aux16 only
    pad = fx.Int32(sort_pad(lay["kind"], m))
    pr = (p_n + THREADS - 1) // THREADS  # pair rounds (static)
    # the previous launch's pair rows: loaded (and reset) while this launch's rows are still being routed
    old_d = [
        ld_i32(a_dest, fx.min(tid + k * THREADS, p_n - 1), CM_DEV)
        for k in range_constexpr(pr)
    ]
    for k in range_constexpr(pr):
        if tid + k * THREADS < p_n:
            bo.buffer_store(pad, rsrc(a_ids), old_d[k])
            bo.buffer_store(fx.Float32(0.0), rsrc(a_w), old_d[k])
            if const_expr(aux16):
                bo.buffer_store(fx.Int32(m), rsrc(a_mi), old_d[k])
    for i in range(tid, 2 * E, THREADS):
        fx.ptr_store(fx.Int32(0), masks + i)
    # 1. every row of this launch routed
    if tid == 0:
        old = global_atomic_add(cnt + fx.Int64(lc_word * 4), fx.Int32(1), syncscope="agent")
        target = (old + 1) * m
        _spin(lambda: ld_i32(cnt, cnt_word, CM_DEV), lambda v: (v - target) < 0)
    gpu.barrier()
    # 2. this launch's ids / weights, all at once
    es, ws = [], []
    for k in range_constexpr(pr):
        p = fx.min(tid + k * THREADS, p_n - 1)
        es.append(ld_i32(topk_ids, p, CM_DEV))
        ws.append(
            fx.Float32(
                bo.buffer_load(
                    rsrc(topk_w), p, vec_width=1, dtype=T.f32, cache_modifier=CM_DEV
                )
            )
        )
    # 3. an expert's tokens, as a 64-bit mask (order free: the bits are the same whoever sets them first)
    for k in range_constexpr(pr):
        p = tid + k * THREADS
        if p < p_n:
            t = p // TOPK
            lds_atomic_or(masks + es[k] * 2 + t // 32, fx.Int32(1) << (t % 32))
    gpu.barrier()

    # 4. pairs -> blocks an expert -> blocks before it (experts tid and THREADS + tid)
    def count(e):
        return popcount(fx.Uint32(fx.ptr_load(masks + 2 * e))) + popcount(
            fx.Uint32(fx.ptr_load(masks + 2 * e + 1))
        )

    e1 = tid + THREADS
    has1 = e1 < E
    c0 = count(tid)
    c1 = has1.select(count(fx.min(e1, E - 1)), fx.Int32(0))
    b0 = (c0 + bm - 1) // bm
    b1 = (c1 + bm - 1) // bm
    x0, n0 = block_excl_scan(b0, lane, wave, scan_a)
    x1, n1 = block_excl_scan(b1, lane, wave, scan_b)
    bb0, bb1 = x0, n0 + x1
    fx.ptr_store(bb0, bbl + tid)
    if has1:
        fx.ptr_store(bb1, bbl + e1)
    for k in range(b0):
        bo.buffer_store(fx.Int32(tid), rsrc(a_eids), bb0 + k)
    if has1:
        for k in range(b1):
            bo.buffer_store(fx.Int32(e1), rsrc(a_eids), bb1 + k)
    if tid == 0:
        bo.buffer_store((n0 + n1) * bm, rsrc(a_nv), 0)
        bo.buffer_store(fx.Int32(m), rsrc(a_nv), 1)
    # the resets above land before any of this launch's rows (one CTA: drained, then a barrier)
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    # 5. the real pairs: rank in the expert = its tokens below t
    for k in range_constexpr(pr):
        p = tid + k * THREADS
        if p < p_n:
            e = es[k]
            t = p // TOPK
            s = p % TOPK
            lo = fx.ptr_load(masks + 2 * e)
            hi = fx.ptr_load(masks + 2 * e + 1)
            low = t < 32
            ml = low.select((fx.Int32(1) << fx.min(t, 31)) - fx.Int32(1), fx.Int32(-1))
            mh = low.select(
                fx.Int32(0), (fx.Int32(1) << fx.max(t - 32, 0)) - fx.Int32(1)
            )
            rank = popcount(fx.Uint32(lo & ml)) + popcount(fx.Uint32(hi & mh))
            dest = fx.ptr_load(bbl + e) * bm + rank
            bo.buffer_store((s << 24) | t, rsrc(a_ids), dest)
            bo.buffer_store(ws[k], rsrc(a_w), dest)
            if const_expr(aux16):
                bo.buffer_store(t, rsrc(a_mi), dest)
            bo.buffer_store(dest, rsrc(a_dest), p, cache_modifier=CM_DEV)
    # 6. the rows out (device scope), then place_codes may read them ("sgl"; aux16 has no codes to place)
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if const_expr(not aux16):
        if tid == 0:
            c["put"](srtmk, 0, fx.Int32(1))


@flyc.jit
def place_codes(c, tid, h, m, sbuf, lay, srtmk):
    """Helper CTA h of SORT_HELPERS (sort builds): once ``sort_rows`` put ``srtmk``, its share of (pair, code
    word) items: pair p's sorted row d <- its token's e8m0 codes, at aiter's shuffle_scale address (SGLang's
    base_sw + (y // 8) * 256 + (y % 4) * 64 + ((y % 8) // 4) * 2, y = 4 q + j a code of word q)."""
    p_n = m * TOPK
    qw = SCALE_COLS // 4
    n = p_n * qw
    o = lay["offs"]
    a_qs = sbuf + fx.Int64(o["qs"])
    a_ts = sbuf + fx.Int64(o["ts"])
    a_dest = sbuf + fx.Int64(o["dest"])
    span = SORT_HELPERS * THREADS
    rounds = (n + span - 1) // span  # static
    if tid == 0:
        c["poll"]([(srtmk, 0, 1)])
    gpu.barrier()
    items, ds, words = [], [], []
    for k in range_constexpr(rounds):
        i = fx.min(h * THREADS + tid + k * span, n - 1)
        p = i // qw
        items.append(i)
        ds.append(ld_i32(a_dest, p, CM_DEV))
        words.append(ld_i32(a_ts, (p // TOPK) * qw + i % qw, CM_DEV))
    for k in range_constexpr(rounds):
        if h * THREADS + tid + k * span < n:
            q = items[k] % qw
            d = ds[k]
            base = (
                (d // 32) * (SCALE_COLS * 32)
                + (d % 16) * 4
                + (d % 32) // 16
                + (q // 2) * 256
                + (q % 2) * 2
            )
            for j in range_constexpr(4):
                bo.buffer_store(
                    fx.Int8((words[k] >> (8 * j)) & 0xFF), rsrc(a_qs), base + j * 64
                )
