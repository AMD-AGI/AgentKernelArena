# SPDX-License-Identifier: Apache-2.0
"""The pre-attention chain of a wide-row layer launch, one launch (T1 -> T2 -> in_proj).

SGLang's latent-MoE tail (``kimi_k3.KimiK3MoE._column_split_tail``) and the
attention-side AttnRes seam, on column-sharded state:

    latent = RMSNorm_n(AR(latent partials));  y = AR(shared partials + latent @ W_up^T)
    prefix' = prefix + y;  x = AttnRes(bank, prefix') -> input norm;  in_proj(x)

Stages (``plan.Plan``), joined by words only:

push    a (row, owner) a task: this rank's latent [448] and shared [896] columns of
        the owner pushed (plain stores), then (this rank, row) flagged at the owner.
lred    a row: the owner sums its latent and shared columns over the TP sources in
        rank order; the latent slice goes to every rank's gather region with the
        slice's sum of squares (tagged), the shared sum stays in local scratch.
up      latent up tiles of the owner's 896 columns over the gathered latent; the
        norm weight is folded into W_up's columns (``fold_up``) and the row's
        rsqrt(mean square) scales the epilogue. A tile's K slices merge, arrive,
        and each finishes its rows: y = up + shared, prefix' = prefix + y.
seam    a row: ``attnres_row`` with the attention weights -> x pushed to every rank.
inproj  in_proj tiles over the gathered x (as ``chain``'s front).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from . import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from . import guard
from .ops import (
    CM_DEV,
    CM_SYS,
    bf2_f32,
    bf16_pair,
    hw_rsq,
    ld_i32,
    load_ptr64,
    rsrc,
    wave_sum,
)
from .plan import (
    BLOCKS,
    THREADS,
    first_task,
    key_tuple,
)
from .sync import sreg
from . import kda as k1
from . import coll
from . import kda_wide as kdaw
from .attnres import attnres_row, red_words, scalars
from .gemm_coll import _arrive_all, dev_emit
from .tile import atomic_lin, bf16_emit, f32_emit
from .tile_mma import DEPTH, RW, mma_tile
from .wide_plan import Plan

SOURCES = "aka-flat"  # AKA: no source digest (FlyDSL hashes the flat package's helpers)

TP = coll.TP
H = 7168
LAT = 3584
SL, LSL = H // TP, LAT // TP


# AKA: ``fold_up`` (a host weight copy for builds without ``up_norm``) is not vendored: the task's
# launch reads the up projection rows as is (``up_norm``).


# the KDA stage's arguments, an int64 table (``kda_table``): pointers, then strides
KDA_PTRS = (
    "w_fb",
    "conv_w",
    "conv_st",
    "a_log",
    "dt_bias",
    "on_w",
    "rstate",
    "st_idx",
    "num_acc",
    "ring_k",
    "ring_v",
    "ring_g",
    "ring_b",
    "core_out",
    "conv_win",
    "win_row",
)
KDA_INTS = (
    "cs_seq",
    "cs_dim",
    "cs_tok",
    "rs_seq",
    "st_stride",
    "rk_seq",
    "rv_seq",
    "rg_seq",
    "rb_seq",
    "ring_len",
    "cw_seq",
    "cw_tok",
    "cw_q",
)


# AKA: ``kda_table`` / ``_KdaArgs`` (the host-built int64 table of these pointers and strides, read by
# the KDA stage through ``kda_tab``) are replaced by kernel arguments: see ``layer.py``.


def plan(m, gu, gi, nin, nvb, kda=None, wide=True):
    rb = gu.bm // gu.ks
    ct = SL // gu.bn
    nb = (m + rb - 1) // rb
    assert gu.bm % gu.ks == 0 and SL % gu.bn == 0 and BLOCKS % gu.ks == 0
    p = Plan()
    p.region("lrs", TP * m * LSL * 2)
    p.region("srs", TP * m * SL * 2)
    p.region("rsf", TP * m * 8)
    p.region("lag", m * LAT * 2)
    p.region("lss", TP * m * 8)
    p.region("lagf", TP * m * 8)
    p.region("xs", TP * m * scalars(nvb) * 8)
    p.region("ag", m * H * 2)
    p.region("agf", TP * m * 8)
    p.region("shs", m * SL * 4, peer=False)
    p.region("pdone", nb * ct * 8, peer=False)
    tasks_u = ((m + gu.bm - 1) // gu.bm) * (SL // gu.bn) * gu.ks
    tasks_i = ((m + gi.bm - 1) // gi.bm) * ((nin + gi.bn - 1) // gi.bn) * gi.ks
    p.stage("push", m * TP)
    p.stage("lred", m, waits_on=("push",), all_ranks=True)
    p.stage("up", tasks_u, waits_on=("lred", "up"), all_ranks=True)
    p.stage("seam", m, waits_on=("up", "seam"), all_ranks=True)
    p.stage(
        "inproj",
        tasks_i,
        waits_on=("seam", "inproj") if kda else ("seam",),
        all_ranks=True,
    )
    if kda:
        p.region("proj", m * k1.NPROJ * 8, peer=False)
        p.region("qk", m * k1.NH * 2 * k1.HD * 8, peer=False)
        p.region("onrm", m * k1.NH * k1.NVCH * 8, peer=False)
        p.stage(
            "kda",
            (m // kda) * k1.NH * (1 if wide else k1.NVCH),
            waits_on=("inproj",) if wide else ("inproj", "kda"),
        )
    return p.check()


@flyc.jit
def _finish_proj(c, out, m, nin, r0, rb, n0, bn, ks, proj):
    """Rows r0 .. r0 + rb of a finished in_proj tile -> the KDA stage's tagged f32
    pairs (two a store); zero the tile."""
    tid = c["tid"]
    q2 = bn // 2
    for it in range(tid, rb * q2, THREADS):
        row = r0 + it // q2
        col = n0 + 2 * (it % q2)
        if (row < m) & (col < nin):
            w = ld_i32(out, (row * nin + col) // 2, CM_DEV)
            lo, hi = bf2_f32(w)
            c["put_words"](
                proj, row * k1.NPROJ + col, [lo.bitcast(fx.Int32), hi.bitcast(fx.Int32)]
            )
            if const_expr(ks > 1):
                bo.buffer_store(
                    fx.Int32(0),
                    rsrc(out),
                    (row * nin + col) // 2,
                    cache_modifier=CM_DEV,
                )


@flyc.jit
def _push(c, row, j, m, lat_p, sh_p):
    """This rank's latent and shared columns of owner j for ``row`` -> the owner."""
    tid = c["tid"]
    rank = c["rank"]
    if tid < LSL // 8:
        v = fx.Vector(
            bo.buffer_load(
                rsrc(lat_p),
                (row * LAT + j * LSL + 8 * tid) // 2,
                vec_width=4,
                dtype=T.i32,
            )
        )
        bo.buffer_store(
            v,
            rsrc(c["lrs_of"](j).value),
            ((rank * m + row) * LSL + 8 * tid) // 2,
            cache_modifier=CM_SYS,
        )
    u = tid - 64
    if (u >= 0) & (u < SL // 8):
        v = fx.Vector(
            bo.buffer_load(
                rsrc(sh_p), (row * H + j * SL + 8 * u) // 2, vec_width=4, dtype=T.i32
            )
        )
        bo.buffer_store(
            v,
            rsrc(c["srs_of"](j).value),
            ((rank * m + row) * SL + 8 * u) // 2,
            cache_modifier=CM_SYS,
        )
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if tid == 0:
        c["put"](c["rsf_of"](j), rank * m + row, fx.Int32(1))


def _sum8(c, region, base, stride):
    got = [
        fx.Vector(
            bo.buffer_load(
                rsrc(c[region].value),
                (src * stride + base) // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_SYS,
            )
        )
        for src in range(TP)
    ]
    acc = None
    for src in range_constexpr(TP):
        cur = [v for e in range(4) for v in bf2_f32(got[src][e])]
        acc = cur if acc is None else [a + b for a, b in zip(acc, cur)]
    return acc


@flyc.jit
def _lred(c, row, m, shs, wn=None):
    """The owner's row: latent sum -> every rank's gather region + the slice's sum
    of squares (tagged) to every rank; shared sum (f32) -> local scratch. ``wn``: the
    latent norm weight (f32 [LAT]) the gathered latent is scaled by (the up projection
    then takes SGLang's weight as is); the sum of squares stays the unscaled row's."""
    tid = c["tid"]
    rank = c["rank"]
    lane = tid % 64
    if tid < TP:
        c["poll"]([(c["rsf_own"], tid * m + row, 1)])
    gpu.barrier()
    if tid < 64:
        live = tid < LSL // 8
        q = fx.min(tid, LSL // 8 - 1)
        acc = _sum8(c, "lrs_own", row * LSL + 8 * q, m * LSL)
        words = [
            bf16_pair(acc[2 * k], acc[2 * k + 1]).bitcast(fx.Int32) for k in range(4)
        ]
        vb = [x for w in words for x in bf2_f32(w)]
        part = vb[0] * vb[0]
        for x in vb[1:]:
            part = part + x * x
        ss = wave_sum(live.select(part, fx.Float32(0.0)))
        if const_expr(wn is not None):
            wr = rsrc(wn)
            wv = [
                fx.Vector(
                    bo.buffer_load(
                        wr, rank * LSL + 8 * q + 4 * h, vec_width=4, dtype=T.f32
                    )
                )
                for h in range(2)
            ]
            wl = [wv[e // 4][e % 4] for e in range(8)]
            words = [
                bf16_pair(vb[2 * k] * wl[2 * k], vb[2 * k + 1] * wl[2 * k + 1]).bitcast(
                    fx.Int32
                )
                for k in range(4)
            ]
        vec = fx.Vector.from_elements(words, fx.Int32)
        if live:
            for p in range_constexpr(TP):
                bo.buffer_store(
                    vec,
                    rsrc(c["lag_of"](p).value),
                    (row * LAT + rank * LSL + 8 * q) // 2,
                    cache_modifier=CM_SYS,
                )
        if lane == 0:
            for p in range_constexpr(TP):
                c["put"](c["lss_of"](p), rank * m + row, ss)
    u = tid - 64
    if (u >= 0) & (u < SL // 8):
        acc = _sum8(c, "srs_own", row * SL + 8 * u, m * SL)
        for k in range_constexpr(2):
            bo.buffer_store(
                fx.Vector.from_elements(acc[4 * k : 4 * k + 4], fx.Float32),
                rsrc(shs),
                row * SL + 8 * u + 4 * k,
                cache_modifier=CM_DEV,
            )
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    coll.put_each_peer(c, [("lagf", rank * m + row, fx.Int32(1))])


@flyc.jit
def _rows_wait(c, flags, m0, bm, m, j0, nown):
    """Wait for the (row, owner) flags in region ``flags`` of rows m0 .. m0 + bm and
    owners j0 .. j0 + nown."""
    rows = fx.min(m0 + bm, m) - m0
    for i in range(c["tid"], rows * nown, THREADS):
        c["poll"]([(c[flags], (j0 + i % nown) * m + m0 + i // nown, 1)])
    gpu.barrier()


@flyc.jit
def _row_rms(c, m0, bm, m, eps, rmsl):
    """rsqrt(mean square) of the gathered latent rows m0 .. m0 + bm -> LDS."""
    tid = c["tid"]
    if tid < fx.min(bm, m - m0):
        got = c["poll"]([(c["lss_own"], src * m + m0 + tid, 1) for src in range(TP)])
        tot = got[0][0].bitcast(fx.Float32)
        for src in range_constexpr(1, TP):
            tot = tot + got[src][0].bitcast(fx.Float32)
        fx.ptr_store(hw_rsq(tot * (1.0 / LAT) + eps), rmsl + tid)
    gpu.barrier()


@flyc.jit
def _finish(
    c,
    acc,
    shs,
    prefix,
    prefix_out,
    m,
    r0,
    rb,
    n0,
    bn,
    ks,
    pdone,
    piece,
    ldp=SL,
    pbytes=None,
):
    """Rows r0 .. r0 + rb of a finished up tile: y = up + shared, prefix' = prefix +
    y (bf16 roundings as SGLang's two adds), zero the up tile, mark the piece.
    ``ldp``: the prefix's row stride (H: this rank's columns of a full-width row);
    ``pbytes``: bytes readable from ``prefix`` (rows past it load 0)."""
    tid = c["tid"]
    q4 = bn // 4
    for it in range(tid, rb * q4, THREADS):
        row = r0 + it // q4
        col = n0 + 4 * (it % q4)
        if row < m:
            uv = fx.Vector(
                bo.buffer_load(
                    rsrc(acc),
                    (row * SL + col) // 2,
                    vec_width=2,
                    dtype=T.i32,
                    cache_modifier=CM_DEV,
                )
            )
            sv = fx.Vector(
                bo.buffer_load(
                    rsrc(shs),
                    row * SL + col,
                    vec_width=4,
                    dtype=T.f32,
                    cache_modifier=CM_DEV,
                )
            )
            pv = fx.Vector(
                bo.buffer_load(
                    rsrc(prefix, pbytes),
                    (row * ldp + col) // 2,
                    vec_width=2,
                    dtype=T.i32,
                )
            )
            up = [x for w in (uv[0], uv[1]) for x in bf2_f32(w)]
            pr = [x for w in (pv[0], pv[1]) for x in bf2_f32(w)]
            y = [
                fx.Float32(fx.Float32(up[e] + sv[e]).to(fx.BFloat16)) for e in range(4)
            ]
            o = [y[e] + pr[e] for e in range(4)]
            words = [
                bf16_pair(o[0], o[1]).bitcast(fx.Int32),
                bf16_pair(o[2], o[3]).bitcast(fx.Int32),
            ]
            bo.buffer_store(
                fx.Vector.from_elements(words, fx.Int32),
                rsrc(prefix_out),
                (row * SL + col) // 2,
                cache_modifier=CM_DEV,
            )
            if const_expr(ks > 1):
                bo.buffer_store(
                    fx.Vector.filled(2, 0, fx.Int32),
                    rsrc(acc),
                    (row * SL + col) // 2,
                    cache_modifier=CM_DEV,
                )
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    # a partial tile's pieces past row m: no piece of the region (the index would land past it)
    if (tid == 0) & (r0 < m):
        c["put"](pdone, piece, fx.Int32(1))


@flyc.jit
def _seam(
    c,
    row,
    m,
    rb,
    ct,
    nvb,
    nbk,
    prefix_out,
    bank,
    cw,
    w_in,
    eps,
    pdone,
    red,
    write,
    capture=False,
    wait=True,
):
    tid = c["tid"]
    if const_expr(wait):
        if tid < ct:
            c["poll"]([(pdone, (row // rb) * ct + tid, 1)])
        gpu.barrier()
    rank = c["rank"]
    ags = [c["ag_of"](p) for p in range(TP)]

    def emit(q, live, vec):
        if live:
            for p in range_constexpr(TP):
                bo.buffer_store(
                    vec,
                    rsrc(ags[p].value),
                    (row * H + rank * SL + 8 * q) // 2,
                    cache_modifier=CM_SYS,
                )

    emit_mixed = None
    if const_expr(capture):
        caps = [c["cap_of"](p) for p in range(TP)]

        def emit_mixed(q, live, vec):
            if live:
                for p in range_constexpr(TP):
                    bo.buffer_store(
                        vec,
                        rsrc(caps[p].value),
                        (row * H + rank * SL + 8 * q) // 2,
                        cache_modifier=CM_SYS,
                    )

    attnres_row(
        c,
        row,
        m,
        SL,
        nvb,
        nbk,
        prefix_out,
        bank,
        cw,
        w_in,
        eps,
        eps,
        red,
        emit,
        write,
        0,
        emit_mixed,
    )
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    coll.put_each_peer(
        c,
        [("agf", rank * m + row, fx.Int32(1))]
        + ([("capf", rank * m + row, fx.Int32(1))] if capture else []),
    )


@flyc.jit
def capture_row(c, row, m, cap_out, cap_ld):
    """The row's gathered pre-norm mixture -> the DSpark capture slot (row stride
    ``cap_ld`` elements, e.g. a column block of the packed aux buffer)."""
    tid = c["tid"]
    if tid < TP:
        c["poll"]([(c["capf_own"], tid * m + row, 1)])
    gpu.barrier()
    for u in range(tid, H // 8, THREADS):
        v = fx.Vector(
            bo.buffer_load(
                rsrc(c["cap_own"].value),
                (row * H + 8 * u) // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_SYS,
            )
        )
        bo.buffer_store(v, rsrc(cap_out), (row * cap_ld + 8 * u) // 2)


# AKA: the standalone ``pre_attn`` launch builder (test/kernels/kimi_k31/pre_attn_bench.py only) is not
# vendored; the layer launch (``layer.py``) uses the stage functions above.
