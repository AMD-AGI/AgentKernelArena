# SPDX-License-Identifier: Apache-2.0
# Vendored from sgl-amd-mono @ 52486703 (sgl_amd_mono/kimi_k3/gfx950/stages/seam.py; the kimi_k3
# package is adapted from vLLM PR #60683, whose shared stages come from ROCm/ATOM, MIT). Modified
# for AgentKernelArena: imports are package-relative; otherwise unchanged.
"""The MLP AttnRes seam, on the attn CTAs: slice, gate, mix.

These three stages run in this order wherever the seam is reached. In this
package that is the layer launch (``layer.py``) and the o_proj launch the
kernel checks use as its reference (``oproj.py``). The seam belongs to
neither, which is why it lives here.

A stage reads its state from the ``c`` dict its launch built.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from . import buffer_ops as bo
from .ops import (
    CM_DEV,
    bf_hi,
    bf_lo,
    butterfly,
    hw_exp2,
    hw_rsq,
    rsrc,
    wave_max,
)
from .plan import THREADS
from .sync import publish
from .dims import (
    COLS,
    GSL,
    HIDDEN,
    LOG2E,
    QC,
)


def ld_bf4(ptr, i):
    """bf16 elements i .. i + 4 (i a multiple of 4) as four f32."""
    w = fx.Vector(bo.buffer_load(rsrc(ptr), i // 2, vec_width=2, dtype=T.i32))
    return [bf_lo(w[0]), bf_hi(w[0]), bf_lo(w[1]), bf_hi(w[1])]


def bf4_words(v):
    """Four f32 -> two words of packed bf16."""
    return fx.Vector.from_elements(v, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)


def bf16r(v):
    return fx.Float32(fx.Float32(v).to(fx.BFloat16))


def f32_of(w):
    return w.bitcast(fx.Float32)


def sum_polled(vals):
    acc = f32_of(vals[0][0])
    for k in range(1, len(vals)):
        acc = acc + f32_of(vals[k][0])
    return acc


# ---------------------------------------------------------------- attn CTAs
def _raw4(ptr, i):
    """bf16 elements i .. i + 4 (i a multiple of 4) as two raw words."""
    return fx.Vector(bo.buffer_load(rsrc(ptr), i // 2, vec_width=2, dtype=T.i32))


def _f4(w):
    return [bf_lo(w[0]), bf_hi(w[0]), bf_lo(w[1]), bf_hi(w[1])]


def _slice_item(c, task, i):
    """Round i's work item of this thread: (live, row, token, source, column)."""
    s, ns, tid = c["S"], c["NS"], c["tid"]
    n = s * ns * QC
    e = fx.min(tid + THREADS * i, n - 1)
    row = e // QC  # (token, source)
    t = row // ns
    return (tid + THREADS * i) < n, e, row, t, row % ns, task * COLS + (e % QC) * 4


def slice_loads(c, task):
    """``stage_slice``'s loads that wait on nothing in this launch: every round's
    block and incoming-prefix words, and the norm / qk words (the same columns
    every round: THREADS % QC == 0). K2 issues them before the o_proj poll."""
    s, ns, nb = c["S"], c["NS"], c["NB"]
    a = c["args"]
    pre_in = a["prefix_in"] if "prefix_in" in a else a["prefix"]
    blk, pre = [], []
    for i in range((s * ns * QC + THREADS - 1) // THREADS):
        _, _, _, t, src, col = _slice_item(c, task, i)
        blk.append(
            _raw4(
                a["blocks"], t * a["blk_sm"] + fx.min(src, nb - 1) * a["blk_sr"] + col
            )
        )
        pre.append(_raw4(pre_in, t * HIDDEN + col))
    _, _, _, _, _, col = _slice_item(c, task, 0)
    return {
        "blk": blk,
        "pre": pre,
        "w": _raw4(a["ares_nw"], col),
        "q": _raw4(a["ares_qk"], col),
    }


@flyc.jit
def stage_slice(c, task, ld=None):
    """Columns 128 task ..: every token's sources into LDS ``vals`` (4 columns a
    thread), the prefix (+ delta) rounded to bf16 and stored back, per-(token,
    source) partial sum of squares and qk dot -> PART. ``ld``: the loads
    ``slice_loads`` issued earlier (else each round loads its own)."""
    s, ns, nb, lane = c["S"], c["NS"], c["NB"], c["lane"]
    a = c["args"]
    # the incoming prefix; K2 under ``fold`` reads it where the stream left it
    pre_in = a["prefix_in"] if "prefix_in" in a else a["prefix"]
    # under ``fold`` the reduce CTAs read the updated prefix back: store it
    # device-coherent (they may sit on another XCD)
    pre_cm = CM_DEV if c.get("fold", False) else 0
    n = s * ns * QC  # threads' work items
    for i in range_constexpr((n + THREADS - 1) // THREADS):
        live, e, row, t, src, col = _slice_item(c, task, i)
        is_pre = src == nb
        if const_expr(ld is None):
            blk = ld_bf4(
                a["blocks"], t * a["blk_sm"] + fx.min(src, nb - 1) * a["blk_sr"] + col
            )
            pre = ld_bf4(pre_in, t * HIDDEN + col)
        else:
            blk, pre = _f4(ld["blk"][i]), _f4(ld["pre"][i])
        if const_expr(c["delta"]):
            if const_expr(c.get("delta_lds") is not None):
                # the delta reduced into LDS first (K2a: the all-reduced o_proj)
                dl = [
                    fx.ptr_load(c["delta_lds"] + (t * COLS + (e % QC) * 4 + j))
                    for j in range(4)
                ]
            else:
                dl = ld_bf4(a["delta"], t * HIDDEN + col)
            if const_expr(c.get("reset", False)):
                # a block-write layer's MLP seam: the prefix restarts at the
                # attention's output (the input prefix is not read)
                pre = dl
            else:
                pre = [bf16r(pre[j] + dl[j]) for j in range(4)]
            if live & is_pre:
                bo.buffer_store(
                    bf4_words(pre),
                    rsrc(a["prefix"]),
                    (t * HIDDEN + col) // 2,
                    cache_modifier=pre_cm,
                )
        if const_expr(c.get("write_idx", -1) >= 0):  # noqa: SIM102 (traced: `and` does not combine traced values)
            # the block-write layer's attention seam: the updated prefix is
            # block write_idx
            if live & is_pre:
                bo.buffer_store(
                    bf4_words(pre),
                    rsrc(a["blocks"]),
                    (t * a["blk_sm"] + c["write_idx"] * a["blk_sr"] + col) // 2,
                )
        v = [
            live.select(is_pre.select(pre[j], blk[j]), fx.Float32(0.0))
            for j in range(4)
        ]
        if const_expr(ld is None):
            w = ld_bf4(a["ares_nw"], col)
            q = ld_bf4(a["ares_qk"], col)
        else:
            w, q = _f4(ld["w"]), _f4(ld["q"])
        # a thread past the end holds a clamped copy of the last element: it
        # must not store (it raced the owner with zeros)
        if live:
            for j in range_constexpr(4):
                fx.ptr_store(v[j], c["vals"] + (row * COLS + (e % QC) * 4 + j))
        sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]
        dt = (
            v[0] * (w[0] * q[0])
            + v[1] * (w[1] * q[1])
            + v[2] * (w[2] * q[2])
            + v[3] * (w[3] * q[3])
        )
        sq = butterfly(sq, (16, 8, 4, 2, 1))
        dt = butterfly(dt, (16, 8, 4, 2, 1))
        if live & (lane % 32 == 0):
            at = ((task * s + t) * ns + src) * 2
            c["put"](c["part"], at, sq)
            c["put"](c["part"], at + 1, dt)
    gpu.barrier()


@flyc.jit
def stage_gate(c, t):
    """Token t: the 56 slices' partials -> source weights e_s and their sum."""
    s, ns, tid, lane, wave, red = (
        c["S"],
        c["NS"],
        c["tid"],
        c["lane"],
        c["wave"],
        c["red"],
    )
    nq = ns * 2
    if tid < 8 * nq:
        g = tid // nq
        q = tid % nq
        vals = c["poll"](
            [(c["part"], (((g * GSL + k) * s + t) * ns) * 2 + q, 1) for k in range(GSL)]
        )
        fx.ptr_store(sum_polled(vals), red + (g * nq + q))
    gpu.barrier()
    if wave == 0:
        src = fx.min(lane, ns - 1)
        sq = fx.ptr_load(red + src * 2)
        dt = fx.ptr_load(red + src * 2 + 1)
        for g in range_constexpr(1, 8):
            sq = sq + fx.ptr_load(red + (g * nq + src * 2))
            dt = dt + fx.ptr_load(red + (g * nq + src * 2 + 1))
        rstd = hw_rsq(sq * fx.Float32(1.0 / HIDDEN) + fx.Float32(c["eps"]))
        logit = dt * rstd * fx.Float32(LOG2E)
        logit = (lane < ns).select(logit, fx.Float32(-3.0e38))
        m = wave_max(logit)
        ex = (lane < ns).select(hw_exp2(logit - m), fx.Float32(0.0))
        den = butterfly(ex, (32, 16, 8, 4, 2, 1))
        if lane < ns:
            c["put"](c["wgt"], t * (ns + 1) + lane, ex)
        if lane == 0:
            c["put"](c["wgt"], t * (ns + 1) + ns, den)
    gpu.barrier()


GATE_OFF = 512  # red words of stage_gate_local's group sums


@flyc.jit
def stage_gate_local(c):
    """Every token's source weights computed in this CTA (``stage_gate``'s sums
    in its order, so every CTA's are bit-identical): the 56 slices' partials
    polled in one round trip (no gate CTA, no WGT hand-off) -> red[t (ns + 1)
    + src], the sum at + ns."""
    s, ns, tid, lane, wave, red = (
        c["S"],
        c["NS"],
        c["tid"],
        c["lane"],
        c["wave"],
        c["red"],
    )
    nq = ns * 2
    n = 8 * s * nq  # (group, token, partial) items
    per = (n + THREADS - 1) // THREADS
    specs = []
    for r in range_constexpr(per):
        item = fx.min(tid + THREADS * r, n - 1)
        g = item // (s * nq)
        t = (item // nq) % s
        q = item % nq
        specs += [
            (c["part"], (((g * GSL + k) * s + t) * ns) * 2 + q, 1) for k in range(GSL)
        ]
    got = c["poll"](specs, per * GSL)
    for r in range_constexpr(per):
        acc = f32_of(got[r * GSL][0])
        for k in range_constexpr(1, GSL):
            acc = acc + f32_of(got[r * GSL + k][0])
        if tid + THREADS * r < n:
            fx.ptr_store(acc, red + (GATE_OFF + tid + THREADS * r))
    gpu.barrier()
    # wave t: token t's weights (stage_gate's math)
    if wave < s:
        t = wave
        src = fx.min(lane, ns - 1)
        base = GATE_OFF + t * nq
        sq = fx.ptr_load(red + (base + src * 2))
        dt = fx.ptr_load(red + (base + src * 2 + 1))
        for g in range_constexpr(1, 8):
            sq = sq + fx.ptr_load(red + (base + g * s * nq + src * 2))
            dt = dt + fx.ptr_load(red + (base + g * s * nq + src * 2 + 1))
        rstd = hw_rsq(sq * fx.Float32(1.0 / HIDDEN) + fx.Float32(c["eps"]))
        logit = dt * rstd * fx.Float32(LOG2E)
        logit = (lane < ns).select(logit, fx.Float32(-3.0e38))
        m = wave_max(logit)
        ex = (lane < ns).select(hw_exp2(logit - m), fx.Float32(0.0))
        den = butterfly(ex, (32, 16, 8, 4, 2, 1))
        if lane < ns:
            fx.ptr_store(ex, red + (t * (ns + 1) + lane))
        if lane == 0:
            fx.ptr_store(den, red + (t * (ns + 1) + ns))
    gpu.barrier()


@flyc.jit
def stage_mix(c, task):
    """Columns 128 task ..: the softmax mix of ``vals`` (4 columns a thread), its
    sum of squares (MSQ), every slice's -> rstd, then bf16 x = mixed rstd w ->
    the plain x rows, and the slice's XRDY flag."""
    s, ns, tid, lane, red = c["S"], c["NS"], c["tid"], c["lane"], c["red"]
    a = c["args"]
    # read before the traced branches (a dict lookup inside one is carried state)
    tag_x = c.get("tag_x", False)
    c0 = task * COLS
    nw = s * (ns + 1)
    if const_expr(c.get("gate_local", False)):
        stage_gate_local(c)
    else:
        if tid < nw:
            v = c["poll"]([(c["wgt"], tid, 1)])[0][0]
            fx.ptr_store(f32_of(v), red + tid)
        gpu.barrier()
    mine = tid < s * QC
    t = fx.min(tid // QC, s - 1)
    cc = (tid % QC) * 4
    mixed = []
    for j in range_constexpr(4):
        acc = fx.Float32(0.0)
        for src in range_constexpr(ns):
            acc = acc + fx.ptr_load(red + (t * (ns + 1) + src)) * fx.ptr_load(
                c["vals"] + ((t * ns + src) * COLS + cc + j)
            )
        mixed.append(acc / fx.ptr_load(red + (t * (ns + 1) + ns)))
    sq = (
        mixed[0] * mixed[0]
        + mixed[1] * mixed[1]
        + mixed[2] * mixed[2]
        + mixed[3] * mixed[3]
    )
    sq = butterfly(sq, (16, 8, 4, 2, 1))
    if mine & (lane % 32 == 0):
        c["put"](c["msq"], task * s + t, sq)
    gpu.barrier()
    if tid < 8 * s:
        g = tid // s
        tt = tid % s
        vals = c["poll"]([(c["msq"], (g * GSL + k) * s + tt, 1) for k in range(GSL)])
        fx.ptr_store(sum_polled(vals), red + (128 + g * s + tt))
    gpu.barrier()
    tot = fx.ptr_load(red + (128 + t))
    for g in range_constexpr(1, 8):
        tot = tot + fx.ptr_load(red + (128 + g * s + t))
    r = hw_rsq(tot * fx.Float32(1.0 / HIDDEN) + fx.Float32(c["out_eps"]))
    wn = ld_bf4(a["in_nw"], c0 + cc)
    xw = bf4_words([mixed[j] * r * wn[j] for j in range(4)])
    if mine:
        at = (t * HIDDEN + c0 + cc) // 2
        if const_expr(tag_x):
            # also as tagged pairs: the router and latent GEMVs poll these, not XRDY
            c["put_words"](c["xt"], at, [xw[0], xw[1]])
        bo.buffer_store(xw, rsrc(c["xbuf"]), at, cache_modifier=CM_DEV)
        if const_expr(c["debug"]):
            bo.buffer_store(xw, rsrc(a["x_dbg"]), at)
    publish(c["put"], c["xrdy"], task, fx.Int32(1), tid == 0)
    gpu.barrier()
