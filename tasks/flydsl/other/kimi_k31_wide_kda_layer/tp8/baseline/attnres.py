# SPDX-License-Identifier: Apache-2.0
"""The AttnRes seam of a token row on column-sharded state (one scalar round).

SGLang's aggregation (``attn_residual.aggregate_stream_torch`` + out_norm): the
row's R = nvb + 1 sources are the bank snapshots 0 .. nvb - 1 and the prefix;
score_r = rsqrt(mean(x_r^2) + eps) (x_r . cw) with cw = score_norm.weight x
score_proj.weight; p = softmax(score); mixed = sum_r p_r x_r; out = RMSNorm(mixed).

A rank holds sl = H / TP columns of every source, so every full-row sum is a sum
of TP partials. All of them travel in one round: the dots x_r . cw and the Gram
entries G_rs = x_r . x_s (r <= s), from which sum(mixed^2) = sum_rs p_r p_s G_rs
follows without a second exchange (it skips the bf16 rounding of mixed that
the reference squares; the difference is below bf16 output rounding).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from . import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from .ops import (
    CM_DEV,
    bf2_f32,
    bf16_pair,
    bf16_round,
    hw_exp2,
    hw_rcp,
    hw_rsq,
    rsrc,
    wave_sum,
)
from .plan import WAVES

LOG2E = 1.4426950408889634


def scalars(nvb):
    """Scalars a row exchanges: R dots + R (R + 1) / 2 Gram entries."""
    r = nvb + 1
    return r + r * (r + 1) // 2


def red_words(nvb):
    return (WAVES + 1) * scalars(nvb)


@flyc.jit
def attnres_row(
    c,
    row,
    m,
    sl,
    nvb,
    nbk,
    prefix,
    bank,
    cw,
    w_o,
    eps_s,
    eps_o,
    red,
    emit,
    write,
    bank_cm=0,
    emit_mixed=None,
    ldp=0,
):
    """One row: ``prefix`` [m, sl] (device scope, just written), ``bank`` [m, nbk, sl]
    (snapshots 0 .. nvb - 1), ``cw`` f32 [H], ``w_o`` bf16 [H]; ``c["xs_of"]`` /
    ``c["xs_own"]`` the scalar exchange region [src][m][V]. ``emit(q, live, vec)``
    gets the 8 normalized bf16 columns 8 q .. of the slice; with ``write`` the
    prefix becomes bank row nvb (device scope: a later seam of the launch may read
    it); ``bank_cm`` CM_DEV when an earlier seam of this launch wrote the bank.
    ``emit_mixed(q, live, vec)``: the pre-norm mixture (bf16), SGLang's
    ``aggregate_stream`` -- what DSpark captures after the previous layer. ``ldp``: the
    prefix's row stride (0: sl)."""
    tid = c["tid"]
    lane, wave = tid % 64, tid // 64
    rank = c["rank"]
    h = 8 * sl
    ldp = ldp or sl
    r_n = nvb + 1
    nv = scalars(nvb)
    q8 = sl // 8
    upw = (q8 + WAVES - 1) // WAVES
    q = fx.min(wave * upw + lane, q8 - 1)
    live = (lane < upw) & ((wave * upw + lane) < q8)
    xs = []
    for r in range_constexpr(nvb):
        v = fx.Vector(
            bo.buffer_load(
                rsrc(bank),
                ((row * nbk + r) * sl + 8 * q) // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=bank_cm,
            )
        )
        xs.append([x for e in range(4) for x in bf2_f32(v[e])])
    pv = fx.Vector(
        bo.buffer_load(
            rsrc(prefix),
            (row * ldp + 8 * q) // 2,
            vec_width=4,
            dtype=T.i32,
            cache_modifier=CM_DEV,
        )
    )
    xs.append([x for e in range(4) for x in bf2_f32(pv[e])])
    cwv = [
        fx.Vector(
            bo.buffer_load(
                rsrc(cw), rank * sl + 8 * q + 4 * k, vec_width=4, dtype=T.f32
            )
        )
        for k in range(2)
    ]
    cws = [cwv[k][e] for k in range(2) for e in range(4)]

    def dot(a, b):
        s = a[0] * b[0]
        for e in range_constexpr(1, 8):
            s = s + a[e] * b[e]
        return s

    vals = [dot(xs[r], cws) for r in range(r_n)]
    for r in range_constexpr(r_n):
        for s in range_constexpr(r, r_n):
            vals.append(dot(xs[r], xs[s]))
    z = fx.Float32(0.0)
    for j in range_constexpr(nv):
        tot = wave_sum(live.select(vals[j], z))
        if lane == 0:
            fx.ptr_store(tot, red + wave * nv + j)
    gpu.barrier()
    if tid < nv:
        mine = fx.ptr_load(red + tid)
        for w in range_constexpr(1, WAVES):
            mine = mine + fx.ptr_load(red + w * nv + tid)
        for p in range_constexpr(8):
            c["put"](c["xs_of"](p), (rank * m + row) * nv + tid, mine)
        got = c["poll"](
            [(c["xs_own"], (src * m + row) * nv + tid, 1) for src in range(8)]
        )
        total = got[0][0].bitcast(fx.Float32)
        for src in range_constexpr(1, 8):
            total = total + got[src][0].bitcast(fx.Float32)
        fx.ptr_store(total, red + WAVES * nv + tid)
    gpu.barrier()
    tot = [fx.ptr_load(red + WAVES * nv + j) for j in range(nv)]
    gram = {}
    j = r_n
    for r in range_constexpr(r_n):
        for s in range_constexpr(r, r_n):
            gram[(r, s)] = tot[j]
            j += 1
    if const_expr(nvb == 0):
        probs = [fx.Float32(1.0)]
    else:
        scores = [tot[r] * hw_rsq(gram[(r, r)] * (1.0 / h) + eps_s) for r in range(r_n)]
        mx = scores[0]
        for r in range_constexpr(1, r_n):
            mx = fx.max(mx, scores[r])
        ex = [hw_exp2((scores[r] - mx) * LOG2E) for r in range(r_n)]
        den = ex[0]
        for r in range_constexpr(1, r_n):
            den = den + ex[r]
        inv_den = hw_rcp(den)
        probs = [ex[r] * inv_den for r in range(r_n)]
    ssm = None
    for r in range_constexpr(r_n):
        for s in range_constexpr(r, r_n):
            t = probs[r] * probs[s] * gram[(r, s)]
            t = t if r == s else t + t
            ssm = t if ssm is None else ssm + t
    inv_o = hw_rsq(ssm * (1.0 / h) + eps_o)
    wv = fx.Vector(
        bo.buffer_load(rsrc(w_o), (rank * sl + 8 * q) // 2, vec_width=4, dtype=T.i32)
    )
    ws = [x for e in range(4) for x in bf2_f32(wv[e])]
    out, mix = [], []
    for e in range_constexpr(8):
        mixed = probs[0] * xs[0][e]
        for r in range_constexpr(1, r_n):
            mixed = mixed + probs[r] * xs[r][e]
        mix.append(bf16_round(mixed))
        out.append(mix[-1] * inv_o * ws[e])
    if const_expr(emit_mixed is not None):
        emit_mixed(
            q,
            live,
            fx.Vector.from_elements(
                [
                    bf16_pair(mix[2 * k], mix[2 * k + 1]).bitcast(fx.Int32)
                    for k in range(4)
                ],
                fx.Int32,
            ),
        )
    vec = fx.Vector.from_elements(
        [bf16_pair(out[2 * k], out[2 * k + 1]).bitcast(fx.Int32) for k in range(4)],
        fx.Int32,
    )
    emit(q, live, vec)
    if const_expr(write):
        if live:
            bo.buffer_store(
                pv,
                rsrc(bank),
                ((row * nbk + nvb) * sl + 8 * q) // 2,
                cache_modifier=CM_DEV,
            )
