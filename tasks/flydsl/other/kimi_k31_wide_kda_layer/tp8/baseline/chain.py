# SPDX-License-Identifier: Apache-2.0
"""The post-attention chain of a wide-row layer launch, one launch (T4 -> T5 -> T6).

Stages (``plan.Plan``), every CTA walking them in order, joined by words only:

oproj  o_proj tiles (x the local attention output, K = 1536), K slices merged in
       scratch; each slice pushes its rows of the finished tile to the column
       owner (plain stores + a piece flag; ``gemm_coll``).
rsred  the owner sums a piece's TP sources in rank order, adds the residual,
       stores the new residual slice and marks the piece in local scratch.
seam   a row a task (remote stores need many CTAs): waits for its band's pieces,
       then the MLP AttnRes seam over the bank's nvb snapshots and the new prefix
       (``stages/attnres.py``, one scalar round with every rank), the normalized
       slice pushed to every rank's gather region, then (this rank, row) flagged.
front  MoE-front tiles (router + latent down + shared gate_up rows, K = 7168),
       a task waiting for the (row, owner) flags of its rows and K range and
       reading x' from the gather region.
"""

import dataclasses

import flydsl.compiler as flyc
import flydsl.expr as fx
from . import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from . import guard
from .ops import CM_DEV, CM_SYS, bf2_f32, bf16_pair, load_ptr64, rsrc
from .plan import (
    BLOCKS,
    THREADS,
    first_task,
    key_tuple,
)
from .sync import sreg
from .stamps import stamp, stamp_begin, stamp_flush
from . import coll
from .attnres import attnres_row, red_words, scalars
from .gemm_coll import (
    _arrive_all,
    _push_band_flag,
    dev_emit,
)
from .tile import atomic_lin, bf16_emit, f32_emit
from .tile_mma import DEPTH, RW, mma_tile
from .wide_plan import Plan

SOURCES = "aka-flat"  # AKA: no source digest (FlyDSL hashes the flat package's helpers)

TP = coll.TP
H = 7168
KO = 1536  # o_proj K a rank (KDA heads x head dim / TP)
# timeline marks (a CTA's first task in a stage marks its wait; the stage's end its last task)
MARKS = (
    "start",
    "oproj mma1",
    "oproj end",
    "rsred in1",
    "rsred end",
    "seam piece1",
    "seam pushed1",
    "seam end",
    "front in1",
    "front end",
)
TL_POINTS = len(MARKS)


@dataclasses.dataclass
class Gemm:
    """A tile GEMM's build config: bm, bn, wm x wn waves, ks K slices."""

    bm: int
    bn: int
    wm: int
    wn: int
    ks: int

    def key(self):
        return (self.bm, self.bn, self.wm, self.wn, self.ks)


def plan(m, go, gf, nf, nvb=0):
    """Regions and stages of a chain build (``go`` the o_proj config, ``gf`` the
    front's, ``nf`` the front's output rows a rank)."""
    sl = H // TP
    rb = go.bm // go.ks
    ct = sl // go.bn
    nb = (m + rb - 1) // rb
    assert (
        sl % go.bn == 0
        and go.bm % go.ks == 0
        and nb * ct <= BLOCKS
        and BLOCKS % go.ks == 0
    )
    p = Plan()
    p.region("rs", TP * m * sl * 2)
    p.region("fl", TP * BLOCKS * 8)
    p.region("xs", TP * m * scalars(nvb) * 8)
    p.region("ag", m * H * 2)
    p.region("agf", TP * m * 8)
    p.region("pdone", nb * ct * 8, peer=False)
    tasks_o = ((m + go.bm - 1) // go.bm) * (H // go.bn) * go.ks
    tasks_f = ((m + gf.bm - 1) // gf.bm) * ((nf + gf.bn - 1) // gf.bn) * gf.ks
    p.stage("oproj", tasks_o, waits_on=("oproj",))
    p.stage("rsred", nb * ct, waits_on=("oproj",), all_ranks=True)
    p.stage("seam", m, waits_on=("rsred", "seam"), all_ranks=True)
    p.stage("front", tasks_f, waits_on=("seam",), all_ranks=True)
    return p.check()


@flyc.jit
def _rs_piece(
    c, piece, m, sl, rb, bn, res, res_out, pdone, tl_on, tls, first, res_cm=0, ldo=0
):
    """Sum piece ``piece``'s TP sources in rank order, add the residual, store the
    new residual (device scope: the seam's CTA reads it), mark the piece. ``ldo``:
    ``res_out``'s row stride (0: sl)."""
    tid = c["tid"]
    q8 = bn // 8
    ct = sl // bn
    ldo = ldo or sl
    if tid < TP:
        c["poll"]([(c["fl_own"], tid * BLOCKS + piece, 1)])
    gpu.barrier()
    if first:
        stamp(tl_on, tls, tid, 3)
    band, cj = piece // ct, piece % ct
    # res None: a block-write layer, whose prefix restarts at the attention output
    r_res = rsrc(res) if res is not None else None
    r_out, r_rs = rsrc(res_out), rsrc(c["rs_own"].value)
    for u in range(tid, rb * q8, THREADS):
        row = band * rb + u // q8
        col = cj * bn + 8 * (u % q8)
        if row < m:
            got = [
                fx.Vector(
                    bo.buffer_load(
                        r_rs,
                        (src * m * sl + row * sl + col) // 2,
                        vec_width=4,
                        dtype=T.i32,
                        cache_modifier=CM_SYS,
                    )
                )
                for src in range(TP)
            ]
            if const_expr(res is not None):
                rv = fx.Vector(
                    bo.buffer_load(
                        r_res,
                        (row * sl + col) // 2,
                        vec_width=4,
                        dtype=T.i32,
                        cache_modifier=res_cm,
                    )
                )
            acc = None
            for src in range_constexpr(TP):
                cur = [v for e in range(4) for v in bf2_f32(got[src][e])]
                acc = cur if acc is None else [a + b for a, b in zip(acc, cur)]
            if const_expr(res is not None):
                r = [v for e in range(4) for v in bf2_f32(rv[e])]
                acc = [a + b for a, b in zip(acc, r)]
            words = [
                bf16_pair(acc[2 * j], acc[2 * j + 1]).bitcast(fx.Int32)
                for j in range(4)
            ]
            bo.buffer_store(
                fx.Vector.from_elements(words, fx.Int32),
                r_out,
                (row * ldo + col) // 2,
                cache_modifier=CM_DEV,
            )
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if tid == 0:
        c["put"](pdone, piece, fx.Int32(1))


@flyc.jit
def _seam_row(
    c,
    row,
    m,
    sl,
    rb,
    ct,
    nvb,
    nbk,
    res_out,
    bank,
    cw,
    w_o,
    eps,
    pdone,
    red,
    write,
    tl_on,
    tls,
    first,
    bank_cm=0,
    ldo=0,
):
    """The MLP AttnRes seam of one row (``attnres.attnres_row``) once its band's
    pieces are in; the normalized slice goes to every rank's gather region, then
    (this rank, row) is flagged on every rank."""
    tid = c["tid"]
    if tid < ct:
        c["poll"]([(pdone, (row // rb) * ct + tid, 1)])
    gpu.barrier()
    if first:
        stamp(tl_on, tls, tid, 5)
    rank = c["rank"]
    ags = [c["ag_of"](p) for p in range(TP)]

    def emit(q, live, vec):
        if live:
            for p in range_constexpr(TP):
                bo.buffer_store(
                    vec,
                    rsrc(ags[p].value),
                    (row * TP * sl + rank * sl + 8 * q) // 2,
                    cache_modifier=CM_SYS,
                )

    attnres_row(
        c,
        row,
        m,
        sl,
        nvb,
        nbk,
        res_out,
        bank,
        cw,
        w_o,
        eps,
        eps,
        red,
        emit,
        write,
        bank_cm,
        None,
        ldo,
    )
    if first:
        stamp(tl_on, tls, tid, 6)
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    coll.put_each_peer(c, [("agf", rank * m + row, fx.Int32(1))])


@flyc.jit
def _front_wait(c, m0, bm, m, j0, nown, tl_on, tls, first):
    """Wait for the (row, owner) flags of rows m0 .. m0 + bm and owners j0 .. j0 + nown."""
    rows = fx.min(m0 + bm, m) - m0
    for i in range(c["tid"], rows * nown, THREADS):
        c["poll"]([(c["agf_own"], (j0 + i % nown) * m + m0 + i // nown, 1)])
    gpu.barrier()
    if first:
        stamp(tl_on, tls, c["tid"], 8)


# AKA: ``_prefetch`` and the standalone ``post_attn`` launch builder (test/kernels/kimi_k31/chain_bench.py
# only) are not vendored; the layer launch (``layer.py``) uses the stage functions above.
