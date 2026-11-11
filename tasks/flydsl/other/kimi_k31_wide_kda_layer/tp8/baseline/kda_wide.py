# SPDX-License-Identifier: Apache-2.0
"""KDA verify at wide rows: one task a (request, head), on the ReplaySSM ring.

The math, rounding and argument conventions of Kimi-K3's K1 KDA tasks
(sgl-project/sgl-amd-mono#13: conv + SiLU, f_b gate MFMA with the lower bound, q / k
l2norm, the gated delta-rule recurrence, gated RMSNorm), but K1 splits a head
into 8 tasks of 16 state rows and each recomputes the head's conv, gate and
l2norm, then exchanges partial norms. Here one CTA holds the whole head:

conv    thread < 384 owns channel (part, d) of the head: the carried window (the
        conv state, read only), the L new inputs (polled from the in_proj pairs),
        each token's window to SGLang's ``intermediate_conv_window``, ring k / v.
gates   f_a and beta polled once; wave w's MFMA gives gate rows 16 w .. for the
        L tokens (log-space gate to the ring, decay to LDS); q / k l2norm.
recur   thread -> 32 / KPT state rows x KPT K columns (a row's dot over 128 / KPT
        lanes): the 128 x 128 committed state in registers (ring: read only).
norm    wave j = token j: the head's 128 outputs, gated RMSNorm -> core_out.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from . import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from .ops import (
    CM_DEV,
    butterfly,
    hw_exp2,
    hw_rsq,
    ld_i32,
    mfma_bf16,
    rsrc,
    uniform,
)
from .plan import BLOCKS, THREADS, WAVES, first_task
from .kda import (
    BETA_0,
    CONV_W,
    FA_0,
    G2_0,
    NPROJ,
    ld_bf,
    ld_f32,
    sigmoid,
)
from .dims import HD, LOG2E, NH, PROJ


def bf16r(v):
    return fx.Float32(fx.Float32(v).to(fx.BFloat16))


def f32_of(w):
    return w.bitcast(fx.Float32)


KPT = 16  # K columns a thread holds in the recurrence (4, 8 or 16)


def _split(kpt):
    """(lanes a state row, rows a thread) for ``kpt`` K columns a thread."""
    lpr = HD // kpt
    return lpr, HD * HD // (THREADS * kpt)


def lds_floats(L):
    """f32 words of LDS a task uses: fa, q|k, decay, v, o (L x 128 each, q|k 2x), beta."""
    return 6 * L * HD + 16


@flyc.jit
def kda_head(c, req, h, lds, kpt=KPT):
    """One (request, head) task; ``lds`` = ``lds_floats(L)`` f32 words."""
    L, tid = c["L"], c["tid"]
    lane, wave = tid % 64, tid // 64
    a = c["args"]
    fa_l = lds
    qkl = lds + L * HD
    decay = lds + 3 * L * HD
    vloc = lds + 4 * L * HD
    ol = lds + 5 * L * HD
    beta = lds + 6 * L * HD
    acc_idx = uniform(ld_i32(a["num_acc"], req)) - 1
    st_idx, st_stride = a["st_idx"], a["st_stride"]
    slot_c = uniform(ld_i32(st_idx, req * st_stride))
    slot0 = uniform(ld_i32(st_idx, req * st_stride + acc_idx))
    live0 = slot0 >= 0
    ring_k = rsrc(a["ring_k"] + fx.Int64(fx.max(slot_c, 0)) * fx.Int64(a["rk_seq"]) * 2)
    ring_v = rsrc(a["ring_v"] + fx.Int64(fx.max(slot_c, 0)) * fx.Int64(a["rv_seq"]) * 2)
    ring_g = rsrc(a["ring_g"] + fx.Int64(fx.max(slot_c, 0)) * fx.Int64(a["rg_seq"]) * 4)
    ring_b = rsrc(a["ring_b"] + fx.Int64(fx.max(slot_c, 0)) * fx.Int64(a["rb_seq"]) * 4)
    ring_len = a["ring_len"]
    has_c = slot_c >= 0

    # inputs that need no hand-off, issued first: f_b rows, constants, the state
    fr = h * HD + wave * 16 + lane % 16
    fbw = (fr * HD + (lane // 16) * 32) // 2
    fb = [
        fx.Vector(
            bo.buffer_load(rsrc(a["w_fb"]), fbw + 4 * q, vec_width=4, dtype=T.i32)
        )
        for q in range(4)
    ]
    aa = hw_exp2(ld_f32(a["a_log"], h) * fx.Float32(LOG2E))
    dt_bias = a["dt_bias"]
    dtb = [ld_f32(dt_bias, h * HD + wave * 16 + 4 * (lane // 16) + i) for i in range(4)]
    lpr, rpt = _split(kpt)
    offs = tuple(1 << b for b in reversed(range(lpr.bit_length() - 1)))
    grp = tid // lpr
    k0 = (tid % lpr) * kpt
    base0 = rsrc(a["rstate"] + fx.Int64(fx.max(slot0, 0)) * fx.Int64(a["rs_seq"]) * 4)
    st = []
    for i in range_constexpr(rpt):
        row = []
        for q in range_constexpr(kpt // 4):
            v = fx.Vector(
                bo.buffer_load(
                    base0,
                    (h * HD + grp * rpt + i) * HD + k0 + 4 * q,
                    vec_width=4,
                    dtype=T.f32,
                )
            )
            row += [live0.select(v[e], fx.Float32(0.0)) for e in range(4)]
        st.append(row)

    # conv: channel (part, d); the conv state is read only, the verify windows go to SGLang's accept
    if tid < 3 * HD:
        part = tid // HD
        d = tid % HD
        chan = part * PROJ + h * HD + d
        cs = a["conv_st"] + fx.Int64(fx.max(slot_c, 0)) * fx.Int64(a["cs_seq"]) * 2
        co = chan * a["cs_dim"]
        cs_tok = a["cs_tok"]
        cw = fx.Vector(
            bo.buffer_load(rsrc(a["conv_w"]), chan * CONV_W, vec_width=4, dtype=T.f32)
        )
        win = [ld_bf(cs, co + (acc_idx + j) * cs_tok) for j in range(CONV_W - 1)]
        xs = c["poll"]([(c["proj"], (req * L + j) * NPROJ + chan, 1) for j in range(L)])
        xv = [f32_of(xs[j][0]) for j in range(L)]
        seq = win + xv
        w_row = uniform(ld_i32(a["win_row"], req))
        if has_c & (w_row >= 0):
            # token j's window: the CONV_W - 1 inputs up to and including token j
            wb = rsrc(
                a["conv_win"] + fx.Int64(fx.max(w_row, 0)) * fx.Int64(a["cw_seq"]) * 2
            )
            cw_tok, cw_q = a["cw_tok"], a["cw_q"]
            for j in range_constexpr(L):
                for q in range_constexpr(CONV_W - 1):
                    bo.buffer_store(
                        fx.Float32(seq[j + 1 + q]).to(fx.BFloat16),
                        wb,
                        j * cw_tok + q * cw_q + chan,
                    )
        for j in range_constexpr(L):
            acc = fx.Float32(0.0)
            for q in range_constexpr(CONV_W):
                acc = acc + cw[q] * seq[j + q]
            o = bf16r(acc * sigmoid(acc))
            at = (h * ring_len + j) * HD + d
            if has_c & (part == 1):
                bo.buffer_store(fx.Float32(o).to(fx.BFloat16), ring_k, at)
            if has_c & (part == 2):
                bo.buffer_store(fx.Float32(o).to(fx.BFloat16), ring_v, at)
            if part < 2:
                fx.ptr_store(o, qkl + ((2 * j + part) * HD + d))
            if part == 2:
                fx.ptr_store(o, vloc + (j * HD + d))

    # gates: f_a [L][128] and beta
    n_fa = L * HD
    for i in range_constexpr((n_fa + THREADS - 1) // THREADS):
        e = fx.min(tid + THREADS * i, n_fa - 1)
        got = c["poll"]([(c["proj"], (req * L + e // HD) * NPROJ + FA_0 + e % HD, 1)])
        if tid + THREADS * i < n_fa:
            fx.ptr_store(f32_of(got[0][0]), fa_l + e)
    if tid < L:
        b = sigmoid(
            f32_of(
                c["poll"]([(c["proj"], (req * L + tid) * NPROJ + BETA_0 + h, 1)])[0][0]
            )
        )
        fx.ptr_store(b, beta + tid)
        if has_c:
            bo.buffer_store(b, ring_b, h * ring_len + tid)
    gpu.barrier()
    t = fx.min(lane % 16, L - 1)
    gacc = fx.Vector.filled(4, 0.0, fx.Float32)
    for q in range_constexpr(4):
        kb = (lane // 16) * 32 + 8 * q
        xb = fx.Vector.from_elements(
            [fx.ptr_load(fa_l + (t * HD + kb + e)) for e in range(8)], fx.Float32
        ).to(fx.BFloat16)
        gacc = mfma_bf16(fb[q].bitcast(fx.BFloat16), xb, gacc)
    for i in range_constexpr(4):
        kk = wave * 16 + 4 * (lane // 16) + i
        g = bf16r(gacc[i]) + dtb[i]
        gate = fx.Float32(c["lb"]) * sigmoid(aa * g)
        if lane % 16 < L:
            fx.ptr_store(hw_exp2(gate * fx.Float32(LOG2E)), decay + (t * HD + kk))
            if has_c:
                bo.buffer_store(gate, ring_g, (h * ring_len + t) * HD + kk)
    # l2norm: q / k row r (2 L rows of 128) on wave r % 8
    for rr in range_constexpr((2 * L + WAVES - 1) // WAVES):
        row = fx.min(wave + WAVES * rr, 2 * L - 1)
        x0 = fx.ptr_load(qkl + (row * HD + lane))
        x1 = fx.ptr_load(qkl + (row * HD + 64 + lane))
        ss = butterfly(x0 * x0 + x1 * x1, (32, 16, 8, 4, 2, 1))
        sc = hw_rsq(ss + fx.Float32(1e-6))
        sc = (row % 2 == 0).select(sc * fx.Float32(HD**-0.5), sc)
        if wave + WAVES * rr < 2 * L:
            fx.ptr_store(x0 * sc, qkl + (row * HD + lane))
            fx.ptr_store(x1 * sc, qkl + (row * HD + 64 + lane))
    gpu.barrier()

    # recurrence: rows grp * rpt .., K columns k0 .. k0 + kpt
    def dot(x, y):
        acc = x[0] * y[0]
        for e in range_constexpr(1, kpt):
            acc = acc + x[e] * y[e]
        return acc

    def lds_vec(at):
        return [
            x
            for q in range(kpt // 4)
            for x in fx.Vector(
                fx.ptr_load(at + 4 * q, result_type=fx.Vector.make_type(4, fx.Float32))
            )
        ]

    for j in range_constexpr(L):
        qv = lds_vec(qkl + ((2 * j) * HD + k0))
        kv = lds_vec(qkl + ((2 * j + 1) * HD + k0))
        dk = lds_vec(decay + (j * HD + k0))
        bj = fx.ptr_load(beta + j)
        for i in range_constexpr(rpt):
            r = grp * rpt + i
            s_ = [st[i][e] * dk[e] for e in range(kpt)]
            p = butterfly(dot(s_, kv), offs)
            vn = (fx.ptr_load(vloc + (j * HD + r)) - p) * bj
            s_ = [s_[e] + vn * kv[e] for e in range(kpt)]
            o = butterfly(dot(s_, qv), offs)
            o = live0.select(bf16r(o), fx.Float32(0.0))
            st[i] = s_
            if lane % lpr == 0:
                fx.ptr_store(o, ol + (j * HD + r))
    gpu.barrier()

    # gated RMSNorm: wave j = token j
    for jj in range_constexpr((L + WAVES - 1) // WAVES):
        j = fx.min(wave + WAVES * jj, L - 1)
        x0 = fx.ptr_load(ol + (j * HD + lane))
        x1 = fx.ptr_load(ol + (j * HD + 64 + lane))
        ss = butterfly(x0 * x0 + x1 * x1, (32, 16, 8, 4, 2, 1))
        rr_ = hw_rsq(ss * fx.Float32(1.0 / HD) + fx.Float32(c["onorm_eps"]))
        g2 = c["poll"](
            [
                (c["proj"], (req * L + j) * NPROJ + G2_0 + h * HD + lane + 64 * u, 1)
                for u in range(2)
            ]
        )
        if wave + WAVES * jj < L:
            on_w, core_out = a["on_w"], rsrc(a["core_out"])
            for u, x in ((0, x0), (1, x1)):
                e = lane + 64 * u
                y = x * rr_ * ld_bf(on_w, e) * sigmoid(f32_of(g2[u][0]))
                bo.buffer_store(
                    y.to(fx.BFloat16),
                    core_out,
                    (req * L + j) * PROJ + h * HD + e,
                    cache_modifier=CM_DEV,
                )
    if const_expr("kdone" in c):
        # o_proj in the same launch reads this head's core_out (possibly on another XCD)
        rocdl.s_waitcnt(vmcnt=0)
        gpu.barrier()
        if tid == 0:
            c["put"](c["kdone"], req * NH + h, fx.Int32(1))
    gpu.barrier()


@flyc.jit
def kda_heads(c, lds, base=0, kpt=KPT):
    """Every (request, head) task of the launch, task t on CTA (t + base) % BLOCKS."""
    for task in range(first_task(c["bid"], base), (c["S"] // c["L"]) * NH, BLOCKS):
        kda_head(c, task // NH, task % NH, lds, kpt)
