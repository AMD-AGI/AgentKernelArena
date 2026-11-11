# SPDX-License-Identifier: Apache-2.0
"""One wide-row KDA layer launch: T1 .. T6 (``pre_attn`` -> KDA -> ``chain``).

    MoE tail (L-1) -> attention AttnRes seam -> in_proj -> KDA (a task a (request,
    head)) -> o_proj -> RS + residual -> MLP AttnRes seam -> MoE front (L)

The stages are ``pre_attn``'s and ``chain``'s, joined in one plan: the two halves
keep their region names in two contexts on one peer buffer (the post half's
regions after the pre half's), one tag. New hand-offs inside the launch, all at
device scope because the reader may sit on another XCD: KDA's core_out (each
(request, head) marked done in scratch; an o_proj tile waits for the heads of
its rows and K range), the prefix after the MoE tail (the RS stage's residual
input), and the bank row the attention seam writes (the MLP seam reads it).
"""

import dataclasses

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu
from flydsl.expr.typing import Int32, Int64

from . import guard
from .ops import CM_DEV, KernArgs, load_ptr64
from .plan import (
    BLOCKS,
    THREADS,
    WAVES,
    first_task,
    key_tuple,
)
from .sync import sreg
from . import kda as k1
from .stamps import stamp, stamp_begin, stamp_flush
from . import chain as post
from . import pre_attn as pre
from . import coll
from . import kda_wide as kdaw
from . import moe_front as mf
from .attnres import red_words
from .gemm_coll import (
    _arrive_all,
    _push_band_flag,
    dev_emit,
)
from .tile import atomic_lin, bf16_emit
from .tile_mma import DEPTH, RW, mma_tile
from .wide_plan import Plan

SOURCES = "aka-flat"  # AKA: no source digest (FlyDSL hashes the flat package's helpers)

TP, H, SL, LAT, LSL = coll.TP, pre.H, pre.SL, pre.LAT, pre.LSL
KO = post.KO
# timeline marks: a CTA passes mark k when it has finished its tasks of that stage
MARKS = (
    "start",
    "t1 push",
    "t1 up",
    "t2 seam",
    "t3 in_proj",
    "kda",
    "t4 oproj",
    "t5 rs",
    "t5 seam",
    "t6 front",
    "route",
    "situ",
    "sdown",
    "end",
)
TL_POINTS = len(MARKS)


@dataclasses.dataclass
class LayerCfg:
    """GEMM configs of the five tile GEMMs: latent up, in_proj, o_proj, MoE front, shared down."""

    up: post.Gemm
    inp: post.Gemm
    oproj: post.Gemm
    front: post.Gemm
    sdown: post.Gemm

    def key(self):
        return tuple(
            g.key() for g in (self.up, self.inp, self.oproj, self.front, self.sdown)
        )


def plan(m, cfg, L, nvb_a, nvb_m, nf, capture=False, timeline=False):
    """(combined Plan, pre plan, post plan, post region offset shift, post scratch shift)."""
    pp = pre.plan(m, cfg.up, cfg.inp, k1.NPROJ, nvb_a, L, True)
    po = post.plan(m, cfg.oproj, cfg.front, nf, nvb_m)
    p = Plan()
    for r in pp.regions:
        p.region(r.name, r.nbytes, r.peer)
    p.region("kdone", (m // L) * k1.NH * 8, peer=False)
    for r in po.regions:
        p.region("o_" + r.name, r.nbytes, r.peer)
    for s in pp.stages:
        p.stage(s.name, s.tasks, s.waits_on, s.all_ranks)
    for s in po.stages:
        waits = tuple(w for w in s.waits_on) + (("kda",) if s.name == "oproj" else ())
        p.stage(
            s.name if s.name != "seam" else "mseam",
            s.tasks,
            tuple(w if w != "seam" else "mseam" for w in waits),
            s.all_ranks,
        )
    gf, gs = cfg.front, cfg.sdown
    nbf = (m + gf.bm // gf.ks - 1) // (gf.bm // gf.ks)
    tnf = (nf + gf.bn - 1) // gf.bn
    p.region("rgat", m * mf.E * 4)
    p.region("lgat", m * mf.LAT * 2)
    p.region("fgf", TP * nbf * mf.gathered_tiles(gf.bn) * 8)
    p.region("fbuf", m * nf * 4, peer=False)
    p.region("sgu", m * 2 * mf.SI * 2, peer=False)
    p.region("smark", nbf * tnf * 8, peer=False)
    p.region("hbuf", m * mf.SI * 2, peer=False)
    p.region("hmark", m * 8, peer=False)
    p.region("srtmk", 8, peer=False)  # sort builds: sort_rows -> place_codes
    p.stage("route", m, waits_on=("front",), all_ranks=True)
    p.stage("situ", m, waits_on=("front",))
    p.stage(
        "sdown", ((m + gs.bm - 1) // gs.bm) * (H // gs.bn) * gs.ks, waits_on=("situ",)
    )
    if capture:
        p.region("cap", m * H * 2)
        p.region("capf", TP * m * 8)
        p.stage("capture", m, waits_on=("seam",), all_ranks=True)
    if timeline:
        p.region("tl", BLOCKS * TL_POINTS * 8, peer=False)
    return p.check(), pp, po


@flyc.jit
def _heads_wait(c, m0, bm, m, L, h0, nh):
    """Wait for the KDA (request, head) marks of rows m0 .. m0 + bm, heads h0 .. h0 + nh."""
    r0 = m0 // L
    nr = (fx.min(m0 + bm, m) - 1) // L - r0 + 1
    for i in range(c["tid"], nr * nh, THREADS):
        c["poll"]([(c["kdone"], (r0 + i // nh) * k1.NH + h0 + i % nh, 1)])
    gpu.barrier()


class _KdaKernArgs:
    """The KDA stage's arguments, each loaded from the kernarg segment where it is read.

    AKA: upstream passes these as ``kda_tab``, an int64 device table that the host builds once per
    (bucket, layer) with ``pre_attn.kda_table`` (``pre_attn._KdaArgs`` loads each entry where it is
    read). The arena's ``run()`` cannot build a device table (no host-to-device copies, no host-side
    state, the launch is captured into a graph), so the same pointers and strides are kernel
    arguments here, read through ``KernArgs`` at each use like the table entries were: the KDA
    stage's code is unchanged."""

    def __init__(self, ka):
        self.ka = ka

    def __getitem__(self, k):
        assert k in pre.KDA_PTRS or k in pre.KDA_INTS, k
        # AKA: ka(k) returns a namedtuple of the one argument k
        return self.ka(k)[0]


_BUILDS: dict = {}
CNT_STRIDE = 32  # int32s between tile counters: one 128 B line each (split-K pieces of many tiles arrive at once)


def kda_layer(
    m,
    cfg,
    L,
    nf,
    nvb_a=0,
    nvb_m=0,
    write_a=False,
    write_m=False,
    nbk=8,
    launches=1,
    eps=1e-5,
    eps_n=1e-5,
    kpt=16,
    capture=False,
    t1=True,
    restart=False,
    half_bytes=0,
    timeline=False,
    front_rows=None,
    up_norm=False,
    sort=False,
):
    """Build the layer launch; returns (launch, plan). Arguments: the pre half's
    (lat_p, sh_p, w_up (``pre.fold_up``), prefix, cw_a, w_in, w_ip, ip_buf (in_proj
    merge, bf16 [m, 6288] zeroed once), acc_u (bf16 [m, 896] zeroed once)), mid
    (the prefix after the MoE tail, [m, 896]), the KDA stage's pointers and strides (AKA: kernel arguments named as
    ``pre.KDA_PTRS`` / ``pre.KDA_INTS``, upstream the ``kda_tab`` table; its core_out is the o_proj input ``core``), the post half's (wo, cw_m, w_ln, wf,
    out (MoE front output), acc_o (o_proj merge, zeroed once)), prefix_out (after
    the attention add; prefix and prefix_out are this rank's columns of full-width
    rows: row stride H, the pointer at the rank's first column; ``pfx_bytes``: bytes
    readable from prefix, later rows read 0: SGLang's prefix holds the step's rows,
    not the bucket's), bank, the T6 tail's (bias f32 [896] (e_score_correction_bias),
    lat_out bf16 [m, 3584] + topk_ids int32 / topk_w f32 [m, 16] (the routed
    experts' input), w_sd [H, 768] (shared down rows of this rank), sh_out bf16
    [m, H] (the shared partial the next layer's MoE tail reads)), cnt (every tile
    counter, zeroed once), scratch, peers, rank, epoch, slot. linbf merges, the
    front in f32 (router logits). ``capture``: DSpark's capture of the previous
    layer (the attention seam's pre-norm mixture, gathered) -> cap_out [m, H] with
    row stride cap_ld; else cap_out is unused. ``t1=False`` (check builds): no MoE tail;
    ``mid`` is an input (the prefix after the previous layer, written by the caller).
    ``restart``: a block-write layer -- its prefix restarts at the attention output
    (SGLang passes no prefix_sum there). ``logits_out`` f32 [m, 896]: the gathered
    router logits, for SGLang's own top-k and experts. ``half_bytes``: lay the peer
    buffer out with at least this half size (every variant sharing one buffer must
    agree on its halves and on where the guard's flags are). ``timeline``: per-CTA clock marks (MARKS) -> the
    scratch region ``tl``, int64 [BLOCKS, TL_POINTS]. ``front_rows`` (router, latent down, shared gate_up): the
    rows of ``wf`` this rank's front rows start at, when ``wf`` is SGLang's merged front weight (no copy); None:
    ``wf`` holds this rank's [router | latent down | shared gate_up] rows contiguously. ``up_norm``: ``w_up`` is
    SGLang's up projection rows as is and ``bias`` continues with the latent norm weight (f32 [LAT]), which the
    MoE tail scales the gathered latent by (else ``w_up`` is ``pre.fold_up``'s copy)."""
    # sort: False / None (off), True or "sgl" (SGLang's BM32 small sort + mxfp8 quant), "aux16" (aiter's BM16 aux sort)
    sort = ("sgl" if sort is True else sort) or None
    assert sort is None or sort in mf.SORT_KINDS, sort
    # AKA: the build parameters by name (upstream ``{**locals(), ...}``; ``locals`` is not candidate code), the
    # same pairs in the same order
    params = dict(
        m=m, cfg=cfg, L=L, nf=nf, nvb_a=nvb_a, nvb_m=nvb_m, write_a=write_a, write_m=write_m, nbk=nbk,
        launches=launches, eps=eps, eps_n=eps_n, kpt=kpt, capture=capture, t1=t1, restart=restart,
        half_bytes=half_bytes, timeline=timeline, front_rows=str(front_rows), up_norm=up_norm, sort=sort,
    )
    keyed = key_tuple(params, SOURCES)
    key = (
        m,
        cfg.key(),
        L,
        nf,
        nvb_a,
        nvb_m,
        write_a,
        write_m,
        nbk,
        launches,
        eps,
        eps_n,
        kpt,
        capture,
        t1,
        restart,
        half_bytes,
        timeline,
        front_rows,
        up_norm,
        sort,
    )
    if key in _BUILDS:
        return _BUILDS[key]
    gu, gi, go, gf, gs = cfg.up, cfg.inp, cfg.oproj, cfg.front, cfg.sdown
    assert (
        nf == mf.NF
        and gf.ks > 1
        and gs.ks == 1
        and m % L == 0
        and (not write_a or nvb_m <= nvb_a + 1)
        and (not write_m or nvb_m < nbk)
    )
    # sort: the routed experts' sort (+ "sgl": mxfp8 quant) + output zeroing made here (mf.sort_rows), m <= 64
    # (a 64-bit token mask an expert); sort_buf holds mf.sort_layout(m, sort), the MoE output sits right before
    # sh_out (wide's flat ping)
    assert not sort or m <= mf.SORT_MAX_ROWS, m
    # the up and in_proj finishes read and zero linbf tiles
    assert gu.ks > 1 and gi.ks > 1
    pl, pp, po = plan(m, cfg, L, nvb_a, nvb_m, nf, capture, timeline)
    offs, half = pl.peer_offsets
    assert half_bytes == 0 or half <= half_bytes, (half, half_bytes)
    half = max(half, half_bytes)
    soffs, _ = pl.scratch_offsets
    pre_offs = {r.name: offs[r.name] for r in pp.regions if r.peer}
    if capture:
        pre_offs.update(cap=offs["cap"], capf=offs["capf"])
    post_offs = {r.name: offs["o_" + r.name] for r in po.regions if r.peer}
    fo = 2 * half
    state_len = L + k1.CONV_W - 1
    nin = k1.NPROJ
    # pre half
    rb_u, ct_u = gu.bm // gu.ks, SL // gu.bn
    ktu, kti = LAT // gu.ks, H // gi.ks
    nown_u, nown_i = max(ktu // LSL, 1), max(kti // SL, 1)
    tnu, tmu = SL // gu.bn, (m + gu.bm - 1) // gu.bm
    tni, tmi = (nin + gi.bn - 1) // gi.bn, (m + gi.bm - 1) // gi.bm
    n_u, n_i = tmu * tnu * gu.ks, tmi * tni * gi.ks
    # post half
    rb_o, ct_o = go.bm // go.ks, SL // go.bn
    kto, ktf = KO // go.ks, H // gf.ks
    assert kto % k1.HD == 0
    nown_f = max(ktf // SL, 1)
    tno, tmo = H // go.bn, (m + go.bm - 1) // go.bm
    tnf, tmf = (nf + gf.bn - 1) // gf.bn, (m + gf.bm - 1) // gf.bm
    n_o, n_f = tmo * tno * go.ks, tmf * tnf * gf.ks
    cnt_i = tmu * tnu
    cnt_o = cnt_i + tmi * tni
    cnt_f = cnt_o + tmo * tno
    rbf = gf.bm // gf.ks
    nbf = (m + rbf - 1) // rbf
    cnt_end = cnt_f + tmf * tnf
    w_rt, w_lc = (
        cnt_end * CNT_STRIDE,
        (cnt_end + 1) * CNT_STRIDE,
    )  # sort: rows routed / sort launches
    lay_s = mf.sort_layout(m, sort) if sort else None
    sort_q = sort == "sgl"  # the launch also quantizes (place_codes helpers)
    sort_cta = BLOCKS - m - 1
    help0 = sort_cta - mf.SORT_HELPERS  # place_codes CTAs: help0 .. sort_cta - 1
    wrow_f = None
    if front_rows is not None:
        n_r, n_l = mf.E_R, mf.L_R
        o_r, o_l, o_s = front_rows

        def wrow_f(row):
            return row + (row < n_r).select(
                fx.Int32(o_r),
                (row < n_r + n_l).select(
                    fx.Int32(o_l - n_r), fx.Int32(o_s - n_r - n_l)
                ),
            )

    tns, tms = H // gs.bn, (m + gs.bm - 1) // gs.bm
    n_s = tms * tns
    # sort builds: place_codes / sort_rows CTAs clear of route (0 .. m-1), sdown (0 .. n_s-1), situ (last m)
    assert not sort or BLOCKS - m - 1 - mf.SORT_HELPERS >= max(m, n_s), (m, n_s)
    gems = (gu, gi, go, gf, gs)
    bmx, bnx = max(g.bm for g in gems), max(g.bn for g in gems)
    cwords = max(g.bm * (g.bn + 4) for g in gems)
    kwl = kdaw.lds_floats(L)

    @fx.struct
    class AbLds:
        xs: fx.Array[fx.Int32, 2 * bmx * RW, 16]
        ws: fx.Array[fx.Int32, 2 * bnx * RW, 16]

    @fx.struct
    class CLds:
        c: fx.Array[fx.Float32, cwords, 16]

    @fx.struct
    class KdwLds:
        v: fx.Array[fx.Float32, kwl, 16]

    @fx.struct
    class SortLds:
        masks: fx.Array[fx.Int32, 2 * mf.E, 16]
        bb: fx.Array[fx.Int32, mf.E, 16]
        sa: fx.Array[fx.Int32, WAVES, 16]
        sb: fx.Array[fx.Int32, WAVES, 16]

    @fx.union
    class Smem:
        ab: AbLds
        c: CLds
        kw: KdwLds
        srt: SortLds

    @fx.struct
    class Small:
        slot: fx.Array[fx.Int32, 4, 16]
        rms: fx.Array[fx.Float32, gu.bm, 16]
        red: fx.Array[fx.Float32, max(red_words(nvb_a), red_words(nvb_m)), 16]

    @fx.struct
    class Tls:
        v: fx.Array[fx.Int64, TL_POINTS, 16]

    name = (
        f"k31_kda_layer_m{m}_{'_'.join('x'.join(map(str, g)) for g in cfg.key())}_L{L}_n{nf}_a{nvb_a}{int(write_a)}"
        f"_m{nvb_m}{int(write_m)}of{nbk}_l{launches}_p{kpt}_c{int(capture)}_t{int(t1)}_r{int(restart)}"
        + ("_tl" if timeline else "")
        + ("_fr" + "x".join(map(str, front_rows)) if front_rows else "")
        + ("_un" if up_norm else "")
        + ({"sgl": "_srt", "aux16": "_srt16"}.get(sort, ""))
        + "_cs_pH_rp"
    )

    def layer_body(
        lat_p: Int64,
        sh_p: Int64,
        w_up: Int64,
        prefix: Int64,
        mid: Int64,
        prefix_out: Int64,
        bank: Int64,
        cw_a: Int64,
        w_in: Int64,
        w_ip: Int64,
        ip_buf: Int64,
        acc_u: Int64,
        core: Int64,
        wo: Int64,
        cw_m: Int64,
        w_ln: Int64,
        wf: Int64,
        acc_o: Int64,
        bias: Int64,
        lat_out: Int64,
        topk_ids: Int64,
        topk_w: Int64,
        w_sd: Int64,
        sh_out: Int64,
        cap_out: Int64,
        cap_ld: Int32,
        logits_out: Int64,
        cnt: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        slot_i: Int32,
        pfx_bytes: Int64,
        sort_buf: Int64,
        # AKA: the KDA stage's arguments (upstream: the int64 ``kda_tab`` table)
        w_fb: Int64,
        conv_w: Int64,
        conv_st: Int64,
        a_log: Int64,
        dt_bias: Int64,
        on_w: Int64,
        rstate: Int64,
        st_idx: Int64,
        num_acc: Int64,
        ring_k: Int64,
        ring_v: Int64,
        ring_g: Int64,
        ring_b: Int64,
        core_out: Int64,
        conv_win: Int64,
        win_row: Int64,
        cs_seq: Int32,
        cs_dim: Int32,
        cs_tok: Int32,
        rs_seq: Int32,
        st_stride: Int32,
        rk_seq: Int32,
        rv_seq: Int32,
        rg_seq: Int32,
        rb_seq: Int32,
        ring_len: Int32,
        cw_seq: Int32,
        cw_tok: Int32,
        cw_q: Int32,
    ):
        own = load_ptr64(peers, rank)
        with guard.scope(
            guard.Guard(
                epoch, own=own, peers=peers, rank=rank, npes=TP, flags_offset=fo
            )
        ):
            tid, bid = fx.thread_idx.x, fx.block_idx.x
            alloc = fx.SharedAllocator()
            st = alloc.allocate(Smem)
            small = alloc.allocate(Small).peek()
            slot, rmsl, red = small.slot.ptr, small.rms.ptr, small.red.ptr
            tls = alloc.allocate(Tls).peek().v.ptr if timeline else None
            stamp_begin(timeline, tls, tid, TL_POINTS)
            lds, cl = st.ab.peek(), st.c.peek().c.ptr
            ca = coll.region_context(
                peers, rank, epoch, slot_i, pre_offs, half, tid, bid, launches
            )
            co = coll.region_context(
                peers, rank, epoch, slot_i, post_offs, half, tid, bid, launches
            )
            tail_offs = {k: offs[k] for k in ("rgat", "lgat", "fgf")}
            cf = coll.region_context(
                peers, rank, epoch, slot_i, tail_offs, half, tid, bid, launches
            )

            def sr(a, region):
                return sreg(a.scratch, soffs[region], region)

            def sp(a, region):
                return a.scratch + fx.Int64(soffs[region])

            # Each stage loads the arguments and scratch regions it uses (ka) instead of keeping all of them
            # live from the kernel entry: the launch is at the SGPR limit.
            # ---- T1: MoE tail
            a = ka(
                "lat_p",
                "sh_p",
                "w_up",
                "acc_u",
                "cnt",
                "prefix",
                "mid",
                "scratch",
                "pfx_bytes",
                *(("bias",) if up_norm else ()),
            )
            shs, pd_a = sp(a, "shs"), sr(a, "pdone")
            for t in range(bid, m * TP if t1 else 0, BLOCKS):
                pre._push(ca, t // TP, t % TP, m, a.lat_p, a.sh_p)
            for row in range(bid, m if t1 else 0, BLOCKS):
                pre._lred(
                    ca, row, m, shs, a.bias + fx.Int64(mf.E * 4) if up_norm else None
                )
            stamp(timeline, tls, tid, 1)
            lagx = ca["lag_own"].value
            for task in range(bid, n_u if t1 else 0, BLOCKS):
                sk = task % gu.ks
                tile = task // gu.ks
                m0 = (tile // tnu) * gu.bm
                n0 = (tile % tnu) * gu.bn
                pre._rows_wait(ca, "lagf_own", m0, gu.bm, m, (sk * ktu) // LSL, nown_u)
                pre._row_rms(ca, m0, gu.bm, m, eps_n, rmsl)
                e_, add = atomic_lin(a.acc_u, SL, n0, m, SL, True)

                def scaled(row, col, vals, add=add, m0=m0):
                    s = fx.ptr_load(rmsl + fx.min(row - m0, gu.bm - 1))
                    add(row, col, [v * s for v in vals])

                mma_tile(
                    tid,
                    lagx,
                    LAT,
                    m,
                    m0,
                    a.w_up,
                    LAT,
                    SL,
                    n0,
                    sk * ktu,
                    ktu,
                    gu.bm,
                    gu.bn,
                    gu.wm,
                    gu.wn,
                    lds.xs.ptr,
                    lds.ws.ptr,
                    None,
                    DEPTH,
                    (e_, scaled),
                    cl,
                )
                _arrive_all(tid, a.cnt, tile, gu.ks, slot, CNT_STRIDE)
                r0 = m0 + sk * rb_u
                pre._finish(
                    ca,
                    a.acc_u,
                    shs,
                    a.prefix,
                    a.mid,
                    m,
                    r0,
                    rb_u,
                    n0,
                    gu.bn,
                    gu.ks,
                    pd_a,
                    (r0 // rb_u) * ct_u + n0 // gu.bn,
                    H,
                    a.pfx_bytes,
                )
            stamp(timeline, tls, tid, 2)
            # ---- T2: attention seam
            for row in range(bid, m, BLOCKS):
                a2 = ka("mid", "bank", "cw_a", "w_in", "scratch")
                pre._seam(
                    ca,
                    row,
                    m,
                    rb_u,
                    ct_u,
                    nvb_a,
                    nbk,
                    a2.mid,
                    a2.bank,
                    a2.cw_a,
                    a2.w_in,
                    eps,
                    sr(a2, "pdone"),
                    red,
                    write_a,
                    capture,
                    t1,
                )
            stamp(timeline, tls, tid, 3)
            # ---- T3: in_proj -> KDA
            xa = ca["ag_own"].value
            f0 = first_task(bid, max(BLOCKS - n_i, 0))
            for task in range(f0, n_i, BLOCKS):
                sk = task % gi.ks
                tile = task // gi.ks
                m0 = (tile // tni) * gi.bm
                n0 = (tile % tni) * gi.bn
                pre._rows_wait(ca, "agf_own", m0, gi.bm, m, (sk * kti) // SL, nown_i)
                a3 = ka("w_ip", "ip_buf", "cnt", "scratch")
                mma_tile(
                    tid,
                    xa,
                    H,
                    m,
                    m0,
                    a3.w_ip,
                    H,
                    nin,
                    n0,
                    sk * kti,
                    kti,
                    gi.bm,
                    gi.bn,
                    gi.wm,
                    gi.wn,
                    lds.xs.ptr,
                    lds.ws.ptr,
                    None,
                    DEPTH,
                    atomic_lin(a3.ip_buf, nin, n0, m, nin, True),
                    cl,
                )
                _arrive_all(
                    tid,
                    a3.cnt + fx.Int64(4 * CNT_STRIDE * cnt_i),
                    tile,
                    gi.ks,
                    slot,
                    CNT_STRIDE,
                )
                rbi = gi.bm // gi.ks
                pre._finish_proj(
                    ca,
                    a3.ip_buf,
                    m,
                    nin,
                    m0 + sk * rbi,
                    rbi,
                    n0,
                    gi.bn,
                    gi.ks,
                    sr(a3, "proj"),
                )
            stamp(timeline, tls, tid, 4)
            a = ka("scratch")
            proj, kdone = sr(a, "proj"), sr(a, "kdone")
            ca.update(
                {
                    "S": m,
                    "L": L,
                    "SL": state_len,
                    "lane": tid % 64,
                    "wave": tid // 64,
                    "lb": -5.0,
                    "onorm_eps": eps_n,
                    "proj": proj,
                    "args": _KdaKernArgs(ka),
                    "kdone": kdone,
                }
            )
            kdaw.kda_heads(ca, st.kw.peek().v.ptr, 0, kpt)
            stamp(timeline, tls, tid, 5)
            # ---- T4: o_proj (core_out of this launch)
            for task in range(bid, n_o, BLOCKS):
                sk = task % go.ks
                tile = task // go.ks
                m0 = (tile // tno) * go.bm
                n0 = (tile % tno) * go.bn
                _heads_wait(ca, m0, go.bm, m, L, (sk * kto) // k1.HD, kto // k1.HD)
                a4 = ka("core", "wo", "acc_o", "cnt")
                mma_tile(
                    tid,
                    a4.core,
                    KO,
                    m,
                    m0,
                    a4.wo,
                    KO,
                    H,
                    n0,
                    sk * kto,
                    kto,
                    go.bm,
                    go.bn,
                    go.wm,
                    go.wn,
                    lds.xs.ptr,
                    lds.ws.ptr,
                    dev_emit(a4.acc_o, H, n0, m, H, False) if go.ks == 1 else None,
                    DEPTH,
                    None if go.ks == 1 else atomic_lin(a4.acc_o, H, n0, m, H, True),
                    cl,
                    CM_DEV,
                )
                _arrive_all(
                    tid,
                    a4.cnt + fx.Int64(4 * CNT_STRIDE * cnt_o),
                    tile,
                    go.ks,
                    slot,
                    CNT_STRIDE,
                )
                piece = ((m0 + sk * rb_o) // rb_o) * ct_o + (n0 % SL) // go.bn
                _push_band_flag(
                    co,
                    a4.acc_o,
                    m,
                    H,
                    m0 + sk * rb_o,
                    rb_o,
                    n0,
                    go.bn,
                    go.ks,
                    "linbf",
                    SL,
                    piece,
                )
            stamp(timeline, tls, tid, 6)
            # ---- T5: RS + residual, MLP seam
            for piece in range(bid, ((m + rb_o - 1) // rb_o) * ct_o, BLOCKS):
                a5 = ka("mid", "prefix_out", "scratch")
                post._rs_piece(
                    co,
                    piece,
                    m,
                    SL,
                    rb_o,
                    go.bn,
                    None if restart else a5.mid,
                    a5.prefix_out,
                    sr(a5, "o_pdone"),
                    False,
                    None,
                    False,
                    CM_DEV,
                    H,
                )
            stamp(timeline, tls, tid, 7)
            for row in range(bid, m, BLOCKS):
                a5r = ka("prefix_out", "bank", "cw_m", "w_ln", "scratch")
                post._seam_row(
                    co,
                    row,
                    m,
                    SL,
                    rb_o,
                    ct_o,
                    nvb_m,
                    nbk,
                    a5r.prefix_out,
                    a5r.bank,
                    a5r.cw_m,
                    a5r.w_ln,
                    eps,
                    sr(a5r, "o_pdone"),
                    red,
                    write_m,
                    False,
                    None,
                    False,
                    CM_DEV if write_a else 0,
                    H,
                )
            stamp(timeline, tls, tid, 8)
            # ---- T6: MoE front
            xm = co["ag_own"].value
            g0 = first_task(bid, max(BLOCKS - n_f, 0))
            for task in range(g0, n_f, BLOCKS):
                sk = task % gf.ks
                tile = task // gf.ks
                m0 = (tile // tnf) * gf.bm
                n0 = (tile % tnf) * gf.bn
                post._front_wait(
                    co, m0, gf.bm, m, (sk * ktf) // SL, nown_f, False, None, False
                )
                a6 = ka("wf", "cnt", "scratch")
                fbuf = sp(a6, "fbuf")
                mma_tile(
                    tid,
                    xm,
                    H,
                    m,
                    m0,
                    a6.wf,
                    H,
                    nf,
                    n0,
                    sk * ktf,
                    ktf,
                    gf.bm,
                    gf.bn,
                    gf.wm,
                    gf.wn,
                    lds.xs.ptr,
                    lds.ws.ptr,
                    None,
                    DEPTH,
                    atomic_lin(fbuf, nf, n0, m, nf, False),
                    cl,
                    wrow=wrow_f,
                )
                _arrive_all(
                    tid,
                    a6.cnt + fx.Int64(4 * CNT_STRIDE * cnt_f),
                    tile,
                    gf.ks,
                    slot,
                    CNT_STRIDE,
                )
                r0 = m0 + sk * rbf
                mf.finish_front(
                    cf,
                    fbuf,
                    sp(a6, "sgu"),
                    m,
                    r0,
                    rbf,
                    n0,
                    gf.bn,
                    r0 // rbf,
                    n0 // gf.bn,
                    nbf,
                    sr(a6, "smark"),
                    tnf,
                )
            stamp(timeline, tls, tid, 9)
            # ---- T6 tail: routing + latent for the experts, shared SiTU + down
            for row in range(bid, m, BLOCKS):
                if const_expr(sort):
                    a7 = ka(
                        "bias",
                        "lat_out",
                        "topk_ids",
                        "topk_w",
                        "logits_out",
                        "sh_out",
                        "sort_buf",
                        "cnt",
                    )
                    mf.route_row(
                        cf,
                        row,
                        m,
                        rbf,
                        nbf,
                        gf.bn,
                        a7.bias,
                        a7.lat_out,
                        a7.topk_ids,
                        a7.topk_w,
                        a7.logits_out,
                        qout=(
                            a7.sort_buf + fx.Int64(lay_s["offs"]["q"])
                            if sort_q
                            else None
                        ),
                        tscale=(
                            a7.sort_buf + fx.Int64(lay_s["offs"]["ts"])
                            if sort_q
                            else None
                        ),
                        moe_out=a7.sh_out - fx.Int64(m * mf.LAT * 2),
                        cnt=a7.cnt,
                        cnt_word=w_rt,
                    )
                else:
                    a7 = ka("bias", "lat_out", "topk_ids", "topk_w", "logits_out")
                    mf.route_row(
                        cf,
                        row,
                        m,
                        rbf,
                        nbf,
                        gf.bn,
                        a7.bias,
                        a7.lat_out,
                        a7.topk_ids,
                        a7.topk_w,
                        a7.logits_out,
                    )
            if const_expr(sort):
                sl = (
                    st.srt.peek()
                )  # outside the dynamic branch: the storage object is not an IR value
                if bid == sort_cta:
                    a7s = ka("cnt", "topk_ids", "topk_w", "sort_buf", "scratch")
                    mf.sort_rows(
                        cf,
                        tid,
                        m,
                        a7s.cnt,
                        w_rt,
                        w_lc,
                        a7s.topk_ids,
                        a7s.topk_w,
                        a7s.sort_buf,
                        lay_s,
                        sr(a7s, "srtmk"),
                        sl.masks.ptr,
                        sl.bb.ptr,
                        sl.sa.ptr,
                        sl.sb.ptr,
                    )
                if const_expr(sort_q):
                    if (bid >= help0) & (bid < sort_cta):
                        a7h = ka("sort_buf", "scratch")
                        mf.place_codes(
                            cf,
                            tid,
                            bid - help0,
                            m,
                            a7h.sort_buf,
                            lay_s,
                            sr(a7h, "srtmk"),
                        )
            stamp(timeline, tls, tid, 10)
            for row in range(first_task(bid, max(BLOCKS - m, 0)), m, BLOCKS):
                a8 = ka("scratch")
                mf.situ_row(
                    cf,
                    row,
                    m,
                    rbf,
                    tnf,
                    gf.bn,
                    sp(a8, "sgu"),
                    sr(a8, "smark"),
                    sp(a8, "hbuf"),
                    sr(a8, "hmark"),
                )
            stamp(timeline, tls, tid, 11)
            for task in range(bid, n_s, BLOCKS):
                a9 = ka("w_sd", "sh_out", "scratch")
                hbuf, hmark = sp(a9, "hbuf"), sr(a9, "hmark")
                m0 = (task // tns) * gs.bm
                n0 = (task % tns) * gs.bn
                rows = fx.min(m0 + gs.bm, m) - m0
                if tid < rows:
                    cf["poll"]([(hmark, m0 + tid, 1)])
                gpu.barrier()
                mma_tile(
                    tid,
                    hbuf,
                    mf.SI,
                    m,
                    m0,
                    a9.w_sd,
                    mf.SI,
                    H,
                    n0,
                    0,
                    mf.SI,
                    gs.bm,
                    gs.bn,
                    gs.wm,
                    gs.wn,
                    lds.xs.ptr,
                    lds.ws.ptr,
                    bf16_emit(a9.sh_out, H, n0, m, H),
                    DEPTH,
                    None,
                    cl,
                    CM_DEV,
                )
            stamp(timeline, tls, tid, 12)
            # ---- DSpark capture of the previous layer
            if const_expr(capture):
                a = ka("cap_out", "cap_ld")
                for row in range(first_task(bid, max(BLOCKS - m, 0)), m, BLOCKS):
                    pre.capture_row(ca, row, m, a.cap_out, a.cap_ld)
            stamp(timeline, tls, tid, 13)
            if const_expr(timeline):
                a = ka("scratch")
                stamp_flush(timeline, tls, sp(a, "tl"), tid, bid, TL_POINTS)

    ka = KernArgs(
        [
            (n, 8 if t is Int64 else 4)
            for n, t in layer_body.__annotations__.items()
            if n != "return"
        ]
    )
    layer_kernel = flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])(layer_body)

    @flyc.jit
    def launch(
        lat_p: Int64,
        sh_p: Int64,
        w_up: Int64,
        prefix: Int64,
        mid: Int64,
        prefix_out: Int64,
        bank: Int64,
        cw_a: Int64,
        w_in: Int64,
        w_ip: Int64,
        ip_buf: Int64,
        acc_u: Int64,
        core: Int64,
        wo: Int64,
        cw_m: Int64,
        w_ln: Int64,
        wf: Int64,
        acc_o: Int64,
        bias: Int64,
        lat_out: Int64,
        topk_ids: Int64,
        topk_w: Int64,
        w_sd: Int64,
        sh_out: Int64,
        cap_out: Int64,
        cap_ld: Int32,
        logits_out: Int64,
        cnt: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        slot_i: Int32,
        pfx_bytes: Int64,
        sort_buf: Int64,
        w_fb: Int64,
        conv_w: Int64,
        conv_st: Int64,
        a_log: Int64,
        dt_bias: Int64,
        on_w: Int64,
        rstate: Int64,
        st_idx: Int64,
        num_acc: Int64,
        ring_k: Int64,
        ring_v: Int64,
        ring_g: Int64,
        ring_b: Int64,
        core_out: Int64,
        conv_win: Int64,
        win_row: Int64,
        cs_seq: Int32,
        cs_dim: Int32,
        cs_tok: Int32,
        rs_seq: Int32,
        st_stride: Int32,
        rk_seq: Int32,
        rv_seq: Int32,
        rg_seq: Int32,
        rb_seq: Int32,
        ring_len: Int32,
        cw_seq: Int32,
        cw_tok: Int32,
        cw_q: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):  # noqa: B008
        _ = keyed
        layer_kernel(
            lat_p,
            sh_p,
            w_up,
            prefix,
            mid,
            prefix_out,
            bank,
            cw_a,
            w_in,
            w_ip,
            ip_buf,
            acc_u,
            core,
            wo,
            cw_m,
            w_ln,
            wf,
            acc_o,
            bias,
            lat_out,
            topk_ids,
            topk_w,
            w_sd,
            sh_out,
            cap_out,
            cap_ld,
            logits_out,
            cnt,
            scratch,
            peers,
            rank,
            epoch,
            slot_i,
            pfx_bytes,
            sort_buf,
            w_fb,
            conv_w,
            conv_st,
            a_log,
            dt_bias,
            on_w,
            rstate,
            st_idx,
            num_acc,
            ring_k,
            ring_v,
            ring_g,
            ring_b,
            core_out,
            conv_win,
            win_row,
            cs_seq,
            cs_dim,
            cs_tok,
            rs_seq,
            st_stride,
            rk_seq,
            rv_seq,
            rg_seq,
            rb_seq,
            ring_len,
            cw_seq,
            cw_tok,
            cw_q,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    _BUILDS[key] = (launch, pl)
    return launch, pl


def counters(m, cfg, nf):
    """int32 tile counters a layer build uses (up, in_proj, o_proj tiles)."""
    gu, gi, go, gf = cfg.up, cfg.inp, cfg.oproj, cfg.front
    return (
        ((m + gu.bm - 1) // gu.bm) * (SL // gu.bn)
        + ((m + gi.bm - 1) // gi.bm) * ((k1.NPROJ + gi.bn - 1) // gi.bn)
        + ((m + go.bm - 1) // go.bm) * (H // go.bn)
        + ((m + gf.bm - 1) // gf.bm) * ((nf + gf.bn - 1) // gf.bn)
        + 2  # sort builds: rows routed, sort launches
    ) * CNT_STRIDE
