# SPDX-License-Identifier: Apache-2.0
# Vendored from sgl-amd-mono @ 52486703 (sgl_amd_mono/kimi_k3/gfx950/attention/kda.py). Modified for
# AgentKernelArena: imports are package-relative (aiter's buffer_ops and sgl-amd-mono's common/
# helpers are vendored beside it); the build-key source tag is plan.py's constant. The kernel, its
# stages and the host launcher ``kda_pre`` are otherwise unchanged.
"""K1 ``kda_pre``: one Kimi-K3 KDA layer from the AttnRes seam to the gated core
output, for a spec-verify step (R requests x L tokens, S = R L rows).

One launch a layer, ``BLOCKS`` CTAs x ``THREADS`` threads, in three roles so that
no CTA that polls a mailbox has weight loads in flight (a poll's load retires
behind every earlier load):

    attn CTAs 0 .. 55   slice (128 hidden columns each): prefix (+ delta) ->
                        bf16 prefix out, per-source partial sum of squares and
                        qk dot (PART); gate (token t on CTA t): softmax weights
                        (WGT); mix: the softmax mix, its partial sum of squares
                        (MSQ), every slice's -> output RMSNorm -> bf16 x rows
                        (plain device-scope stores, an XRDY flag a slice)
    gemv CTAs 56 ..     in_proj, two 16-row groups each: the first group's
                        weights issued at kernel start, the second's as x lands
                        -> q|k|v|g|f_a|beta (PROJ)
    kda tasks           (R x 12 heads x 8 v-chunks) on CTAs 0 ..: inputs that
                        need no hand-off loaded first (metadata, conv window,
                        state rows, f_b rows), then conv update of the owned
                        channels (q / k out through QK), f_b MFMA + lower-bound
                        gate, l2norm, the recurrence on 16 state rows (a state a
                        token written), partial norms (ONRM), gated RMSNorm ->
                        core_out

Rounding follows the original path where it is elementwise (the AttnRes prefix
add, every bf16 hand-off between the original kernels); split-K reductions
differ only in their order.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from . import buffer_ops as bo
from .ops import (
    CM_DEV,
    CM_NT,
    butterfly,
    hw_exp2,
    hw_rsq,
    ld_i32,
    mfma_bf16,
    row_sum,
    rsrc,
    uniform,
)
from .plan import (
    BLOCKS,
    THREADS,
    WAVES,
    first_task,
    kernel_sources,
    key_tuple,
    pair_layout,
)
from .sync import Mailbox, sreg
from .dims import (
    COLS,
    GSL,
    HD,
    HIDDEN,
    LDS_PAD,
    LOG2E,
    NA,
    NH,
    PROJ,
    ROWS,
)
from .seam import (
    bf16r,
    f32_of,
    stage_gate,
    stage_mix,
    stage_slice,
    sum_polled,
)
from .stamps import (
    stamp,
    stamp_begin,
    stamp_flush,
)
from .tag import step_tag

SOURCES = kernel_sources(__name__)

QKV = 3 * PROJ
G2_0 = QKV
FA_0 = G2_0 + PROJ  # 6144
BETA_0 = FA_0 + HD  # 6272
NPROJ = 6288  # in_proj rows a rank holds (6284 + 4 pad)
GEMV_TASKS = NPROJ // ROWS  # 393
GEMV_CTAS = (GEMV_TASKS + 1) // 2  # 197
KCH = 128  # K a chunk: 4 MFMAs of 32
NKCH = HIDDEN // KCH  # 56
PER_WAVE = NKCH // WAVES  # 7
VCH = 16  # state rows a KDA task
NVCH = HD // VCH  # 8
CONV_W = 4
# timeline points: start, slice, gate, mix, load_x, gemv0, gemv1, k.stage,
# k.conv, k.gates, k.recur, k.norm
TL_POINTS = 15

assert NA + GEMV_CTAS <= BLOCKS and NA == 8 * GSL

_STREAM = fx.Stream(None)
# every mono source file: in every build's JIT cache key


@dataclass(frozen=True)
class KdaPreBuild:
    tokens: int  # S
    qlen: int  # L, tokens a request
    nblocks: int  # AttnRes block sources (the prefix is one more)
    delta: bool
    state_len: int  # conv state entries: CONV_W - 1 + num_spec
    eps: float = 1e-5  # AttnRes norm
    out_eps: float = 1e-5  # input_layernorm
    onorm_eps: float = 1e-5
    lower_bound: float = -5.0
    debug: bool = False
    timeline: bool = False
    write_idx: int = -1  # a block-write layer's block: the updated prefix stored there
    kda0: int = 0  # the first KDA task's CTA
    # the initial state read from the committed pool (``rstate0`` at the request's
    # ``init_slot``) instead of the speculative scratch's first row: no staging copy
    init_committed: bool = False
    xp: int = 0  # experiment switch (timing only)
    # ReplaySSM ring (--enable-linear-replayssm-spec): rstate is the committed
    # state, read and never written; each token's conv k / v, gate and beta go
    # to the ring rows SGLang's commit replays, instead of a state a token.
    ring: bool = False


def scratch_layout(key: KdaPreBuild) -> dict:
    s, ns = key.tokens, key.nblocks + 1
    assert s % key.qlen == 0
    pairs = {
        "part": NA * s * ns * 2,
        "wgt": s * (ns + 1),
        "msq": NA * s,
        "xrdy": NA,
        "proj": s * NPROJ,
        "qk": s * NH * 2 * HD,
        "onrm": s * NH * NVCH,
    }
    lay = pair_layout(pairs.items())
    end = max(o + n for o, n in lay.values())
    # the plain bf16 x rows (not pairs), 16 B aligned
    lay["x"] = ((end + 15) // 16 * 16, s * HIDDEN * 2)
    return lay


def scratch_bytes(key: KdaPreBuild) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


def ld_bf(ptr, i, cm=0):
    return fx.Float32(
        fx.BFloat16(
            bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.bf16, cache_modifier=cm)
        )
    )


def ld_f32(ptr, i):
    return fx.Float32(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.f32))


def sigmoid(z):
    return fx.Float32(1.0) / (fx.Float32(1.0) + hw_exp2(z * fx.Float32(-LOG2E)))


# ---------------------------------------------------------------- gemv CTAs
def gemv_loads(c, rg, chunks):
    """This wave's weight operands of rows 16 rg .. for its K chunks ``chunks``
    (indices into wave, wave + 8, ..): a lane 64 contiguous bytes of its row a
    chunk (4 dwordx4). ``rg`` past the last group loads the last group."""
    lane, wave = c["lane"], c["wave"]
    r_w = rsrc(c["args"]["w_in"])
    row = fx.min(rg, GEMV_TASKS - 1) * ROWS + lane % ROWS
    g = lane // ROWS
    out = []
    for i in chunks:
        k = (wave + WAVES * i) * KCH + g * 8
        base = (row * HIDDEN + k) // 2  # dwords
        out.append(
            [
                fx.Vector(
                    bo.buffer_load(
                        r_w,
                        base + 16 * q,
                        vec_width=4,
                        dtype=T.i32,
                        cache_modifier=CM_NT,
                    )
                )
                for q in range(4)
            ]
        )
    return out


@flyc.jit
def load_x(c):
    """Every slice's x ready -> the bf16 x rows into LDS (rows padded)."""
    s, tid = c["S"], c["tid"]
    if tid < NA:
        c["poll"]([(c["xrdy"], tid, 1)])
    gpu.barrier()
    words = s * HIDDEN // 8  # 16 B chunks
    row = HIDDEN + LDS_PAD
    for i in range_constexpr((words + THREADS - 1) // THREADS):
        e = fx.min(tid + THREADS * i, words - 1)
        v = fx.Vector(
            bo.buffer_load(
                rsrc(c["xbuf"]), e * 4, vec_width=4, dtype=T.i32, cache_modifier=CM_DEV
            )
        )
        t = e // (HIDDEN // 8)
        k = e % (HIDDEN // 8) * 8
        if tid + THREADS * i < words:
            fx.ptr_store(v, c["xl"] + (t * row + k) // 2)
    gpu.barrier()


@flyc.jit
def stage_gemv(c, rg, ops):
    """Rows 16 rg .. of in_proj against the LDS x -> bf16 PROJ (and proj_dbg)."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    g = lane // ROWS
    t = fx.min(lane % ROWS, s - 1)
    row = HIDDEN + LDS_PAD
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for i in range_constexpr(PER_WAVE):
        k = (wave + WAVES * i) * KCH + g * 8
        for q in range_constexpr(4):
            xb = fx.Vector(
                fx.ptr_load(
                    c["xl"] + (t * row + k + 32 * q) // 2,
                    result_type=fx.Vector.make_type(4, fx.Int32),
                )
            )
            acc = mfma_bf16(
                ops[i][q].bitcast(fx.BFloat16), xb.bitcast(fx.BFloat16), acc
            )
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if (tid < ROWS * s) & (rg < GEMV_TASKS):
        r = tid % ROWS
        tt = tid // ROWS
        v = bf16r(row_sum(red, r, tt))
        c["put"](c["proj"], tt * NPROJ + rg * ROWS + r, v)
        if const_expr(c["debug"]):
            bo.buffer_store(
                v.to(fx.BFloat16),
                rsrc(c["args"]["proj_dbg"]),
                tt * NPROJ + rg * ROWS + r,
            )
    gpu.barrier()


# ---------------------------------------------------------------- kda
def kda_stage(c, req, h, ch):
    """The task's inputs that need no hand-off, issued before its first poll:
    metadata, the owned channels' conv window / weights, the initial state rows,
    the f_b rows of this wave and the per-head constants."""
    L, sl, tid, lane, wave = c["L"], c["SL"], c["tid"], c["lane"], c["wave"]
    a = c["args"]
    k = {}
    # every value a buffer resource is built on wave-uniform (else a waterfall)
    k["acc_idx"] = uniform(ld_i32(a["num_acc"], req)) - 1
    k["slot_c"] = uniform(ld_i32(a["st_idx"], req * a["st_stride"]))
    k["slot0"] = uniform(ld_i32(a["st_idx"], req * a["st_stride"] + k["acc_idx"]))
    k["slots"] = [
        uniform(ld_i32(a["st_idx"], req * a["st_stride"] + j)) for j in range(L)
    ]
    # conv: thread < 48 owns channel (part, d)
    ct = fx.min(tid, 3 * VCH - 1)
    part = ct // VCH
    chan = part * PROJ + h * HD + ch * VCH + ct % VCH
    cs = a["conv_st"] + fx.Int64(fx.max(k["slot_c"], 0)) * fx.Int64(a["cs_seq"]) * 2
    co = chan * a["cs_dim"]  # the channel's first entry (bf16 elements)
    k["cs"], k["co"], k["chan"], k["part"] = cs, co, chan, part
    k["cw"] = fx.Vector(
        bo.buffer_load(rsrc(a["conv_w"]), chan * CONV_W, vec_width=4, dtype=T.f32)
    )
    k["win"] = [
        ld_bf(cs, co + (k["acc_idx"] + j) * a["cs_tok"]) for j in range(CONV_W - 1)
    ]
    k["old"] = [
        ld_bf(cs, co + (k["acc_idx"] + 1 + j) * a["cs_tok"]) for j in range(sl - L)
    ]
    # recurrence: thread -> state row v = tid / 32, k columns (tid % 32) 4 ..
    vr = ch * VCH + tid // 32
    k0 = (tid % 32) * 4
    k["head_off"] = (h * HD + vr) * HD + k0
    if const_expr(c["init_committed"]):
        # the committed state itself: nothing staged it into the scratch
        islot = uniform(ld_i32(a["init_slot"], req))
        base0 = a["rstate0"] + fx.Int64(fx.max(islot, 0)) * fx.Int64(a["rs0_seq"]) * 4
    else:
        base0 = (
            a["rstate"] + fx.Int64(fx.max(k["slot0"], 0)) * fx.Int64(a["rs_seq"]) * 4
        )
    k["st"] = fx.Vector(
        bo.buffer_load(rsrc(base0), k["head_off"], vec_width=4, dtype=T.f32)
    )
    # f_b: wave w's rows 16 w .. of head h, a lane 64 B of its row
    fr = h * HD + wave * 16 + lane % 16
    fb = (fr * HD + (lane // 16) * 32) // 2
    k["fb"] = [
        fx.Vector(bo.buffer_load(rsrc(a["w_fb"]), fb + 4 * q, vec_width=4, dtype=T.i32))
        for q in range(4)
    ]
    k["aa"] = hw_exp2(ld_f32(a["a_log"], h) * fx.Float32(LOG2E))
    k["dtb"] = [
        ld_f32(a["dt_bias"], h * HD + wave * 16 + 4 * (lane // 16) + i)
        for i in range(4)
    ]
    return k


@flyc.jit
def kda_conv(c, k, req, h, ch):
    """Owned channels: the spec conv update of request ``req``'s L tokens -> new
    conv state; q / k out through QK, v to LDS."""
    L, sl, tid = c["L"], c["SL"], c["tid"]
    a = c["args"]
    stamp(c["on"], c["tls"], tid, 12)
    if tid < 3 * VCH:
        xs = c["poll"](
            [(c["proj"], (req * L + j) * NPROJ + k["chan"], 1) for j in range(L)]
        )
        xv = [f32_of(xs[j][0]) for j in range(L)]
        stamp(c["on"], c["tls"], tid, 13)
        rs = rsrc(k["cs"])
        if const_expr(c["xp"] == 2):
            # force the staged conv window before timing the stores
            fx.ptr_store(k["old"][0] + k["win"][0], c["red"] + tid)
            stamp(c["on"], c["tls"], tid, 13)
        # A negative slot is the "no state" marker, matching SGLang's
        # null_block_id = -1 convention; row 0 is an ordinary slot. The address
        # above already clamps with max(slot, 0), so the two agree.
        if (k["slot_c"] >= 0) & (c["xp"] != 1):
            for j in range_constexpr(sl):
                nv = k["old"][j] if j < sl - L else xv[j - (sl - L)]
                bo.buffer_store(
                    fx.Float32(nv).to(fx.BFloat16), rs, k["co"] + j * a["cs_tok"]
                )
        stamp(c["on"], c["tls"], tid, 14)
        seq = k["win"] + xv
        for j in range_constexpr(L):
            acc = fx.Float32(0.0)
            for q in range_constexpr(CONV_W):
                acc = acc + k["cw"][q] * seq[j + q]
            o = bf16r(acc * sigmoid(acc))
            if const_expr(c["ring"]):
                # ring rows [slot][head][token][128]: k is stored before its l2norm
                at = (h * a["ring_len"] + j) * HD + ch * VCH + tid % VCH
                if (k["slot_c"] >= 0) & (k["part"] == 1):
                    bo.buffer_store(
                        fx.Float32(o).to(fx.BFloat16),
                        rsrc(
                            a["ring_k"]
                            + fx.Int64(k["slot_c"]) * fx.Int64(a["rk_seq"]) * 2
                        ),
                        at,
                    )
                if (k["slot_c"] >= 0) & (k["part"] == 2):
                    bo.buffer_store(
                        fx.Float32(o).to(fx.BFloat16),
                        rsrc(
                            a["ring_v"]
                            + fx.Int64(k["slot_c"]) * fx.Int64(a["rv_seq"]) * 2
                        ),
                        at,
                    )
            if k["part"] < 2:
                c["put"](
                    c["qk"],
                    ((req * NH + h) * L + j) * 2 * HD
                    + k["part"] * HD
                    + ch * VCH
                    + tid % VCH,
                    o,
                )
            if k["part"] == 2:
                fx.ptr_store(o, c["vloc"] + (j * VCH + tid % VCH))
    gpu.barrier()


@flyc.jit
def kda_gates(c, k, req, h, ch):
    """f_a, beta and the head's q / k (one poll batch a thread); g1 = f_b f_a by
    MFMA (a wave 16 rows) -> lower-bound gate -> decay in LDS; q / k l2-normed
    (q scaled). Every chunk of the head computes the whole gate; under the ring
    chunk 0 stores it."""
    L, tid, lane, wave = c["L"], c["tid"], c["lane"], c["wave"]
    a = c["args"]
    ring_out = (k["slot_c"] >= 0) & (ch == 0)
    n_fa = L * HD
    n = n_fa + L * 2 * HD
    per = (n + THREADS - 1) // THREADS
    specs = []
    for i in range_constexpr(per):
        e = fx.min(tid + THREADS * i, n - 1)
        fa_at = (req * L + e // HD) * NPROJ + FA_0 + e % HD
        qk_at = (req * NH + h) * L * 2 * HD + (e - n_fa)
        specs.append((e, fa_at, qk_at))
    # two regions, two batches: the f_a pairs, then the q / k pairs
    got_fa = c["poll"](
        [(c["proj"], fx.min(fa, (req * L + L) * NPROJ - 1), 1) for _, fa, _ in specs]
    )
    got_qk = c["poll"](
        [(c["qk"], fx.max(qk, (req * NH + h) * L * 2 * HD), 1) for _, _, qk in specs]
    )
    for i in range_constexpr(per):
        e = specs[i][0]
        if tid + THREADS * i < n_fa:
            fx.ptr_store(f32_of(got_fa[i][0]), c["fa"] + e)
        if (tid + THREADS * i >= n_fa) & (tid + THREADS * i < n):
            fx.ptr_store(f32_of(got_qk[i][0]), c["qkl"] + (e - n_fa))
    if tid < L:
        b = c["poll"]([(c["proj"], (req * L + tid) * NPROJ + BETA_0 + h, 1)])[0][0]
        fx.ptr_store(sigmoid(f32_of(b)), c["beta"] + tid)
        if const_expr(c["ring"]):
            if ring_out:
                bo.buffer_store(
                    sigmoid(f32_of(b)),
                    rsrc(
                        a["ring_b"] + fx.Int64(k["slot_c"]) * fx.Int64(a["rb_seq"]) * 4
                    ),
                    h * a["ring_len"] + tid,
                )
    gpu.barrier()
    # g1 rows 16 wave .. x tokens: A = f_b rows (staged), B = f_a [token][K]
    t = fx.min(lane % 16, L - 1)
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for q in range_constexpr(4):
        kb = (lane // 16) * 32 + 8 * q
        xs = [fx.ptr_load(c["fa"] + (t * HD + kb + e)) for e in range(8)]
        xb = fx.Vector.from_elements(xs, fx.Float32).to(fx.BFloat16)
        acc = mfma_bf16(k["fb"][q].bitcast(fx.BFloat16), xb, acc)
    for i in range_constexpr(4):
        kk = wave * 16 + 4 * (lane // 16) + i
        g = bf16r(acc[i]) + k["dtb"][i]
        gate = fx.Float32(c["lb"]) * sigmoid(k["aa"] * g)
        if lane % 16 < L:
            fx.ptr_store(hw_exp2(gate * fx.Float32(LOG2E)), c["decay"] + (t * HD + kk))
            if const_expr(c["ring"]):
                if ring_out:
                    bo.buffer_store(
                        gate,
                        rsrc(
                            a["ring_g"]
                            + fx.Int64(k["slot_c"]) * fx.Int64(a["rg_seq"]) * 4
                        ),
                        (h * a["ring_len"] + t) * HD + kk,
                    )
    # l2norm: q / k row r (2 L rows of 128) on wave r % 8
    for rr in range_constexpr((2 * L + WAVES - 1) // WAVES):
        row = fx.min(wave + WAVES * rr, 2 * L - 1)
        x0 = fx.ptr_load(c["qkl"] + (row * HD + lane))
        x1 = fx.ptr_load(c["qkl"] + (row * HD + 64 + lane))
        ss = butterfly(x0 * x0 + x1 * x1, (32, 16, 8, 4, 2, 1))
        sc = hw_rsq(ss + fx.Float32(1e-6))
        sc = (row % 2 == 0).select(sc * fx.Float32(HD**-0.5), sc)
        if wave + WAVES * rr < 2 * L:
            fx.ptr_store(x0 * sc, c["qkl"] + (row * HD + lane))
            fx.ptr_store(x1 * sc, c["qkl"] + (row * HD + 64 + lane))
    gpu.barrier()


@flyc.jit
def kda_recur(c, k, req, h, ch):
    """State rows 16 ch .. of head h: the recurrence over the L tokens, a state
    written a token, bf16 outputs to LDS ``ol`` and their partial norms (ONRM)."""
    L, tid, lane = c["L"], c["tid"], c["lane"]
    a = c["args"]
    v = tid // 32
    k0 = (tid % 32) * 4
    # See kda_conv: negative is the "no state" marker, row 0 is a real slot.
    live0 = k["slot0"] >= 0
    stv = [live0.select(k["st"][e], fx.Float32(0.0)) for e in range(4)]
    for j in range_constexpr(L):
        qv = [fx.ptr_load(c["qkl"] + ((2 * j) * HD + k0 + e)) for e in range(4)]
        kv = [fx.ptr_load(c["qkl"] + ((2 * j + 1) * HD + k0 + e)) for e in range(4)]
        dk = [fx.ptr_load(c["decay"] + (j * HD + k0 + e)) for e in range(4)]
        stv = [stv[e] * dk[e] for e in range(4)]
        p = stv[0] * kv[0] + stv[1] * kv[1] + stv[2] * kv[2] + stv[3] * kv[3]
        p = butterfly(p, (16, 8, 4, 2, 1))
        vn = (fx.ptr_load(c["vloc"] + (j * VCH + v)) - p) * fx.ptr_load(c["beta"] + j)
        stv = [stv[e] + vn * kv[e] for e in range(4)]
        o = stv[0] * qv[0] + stv[1] * qv[1] + stv[2] * qv[2] + stv[3] * qv[3]
        o = bf16r(butterfly(o, (16, 8, 4, 2, 1)))
        o = live0.select(o, fx.Float32(0.0))
        if const_expr(not c["ring"]):
            slot = k["slots"][j]
            if (slot >= 0) & live0:
                dst = rsrc(a["rstate"] + fx.Int64(slot) * fx.Int64(a["rs_seq"]) * 4)
                bo.buffer_store(
                    fx.Vector.from_elements(stv, fx.Float32), dst, k["head_off"]
                )
        if lane % 32 == 0:
            fx.ptr_store(o, c["ol"] + (j * VCH + v))
    gpu.barrier()
    if tid < L:
        ss = fx.Float32(0.0)
        for e in range_constexpr(VCH):
            x = fx.ptr_load(c["ol"] + (tid * VCH + e))
            ss = ss + x * x
        c["put"](c["onrm"], ((req * NH + h) * L + tid) * NVCH + ch, ss)
    gpu.barrier()


@flyc.jit
def kda_norm(c, req, h, ch):
    """Rows 16 ch ..: gated RMSNorm over the head's 128 (every chunk's partial)
    -> core_out."""
    L, tid = c["L"], c["tid"]
    a = c["args"]
    if tid < L * VCH:
        j = tid // VCH
        e = tid % VCH
        vr = ch * VCH + e
        got = c["poll"](
            [(c["onrm"], ((req * NH + h) * L + j) * NVCH + q, 1) for q in range(NVCH)]
            + [(c["proj"], (req * L + j) * NPROJ + G2_0 + h * HD + vr, 1)]
        )
        ss = sum_polled(got[:NVCH])
        r = hw_rsq(ss * fx.Float32(1.0 / HD) + fx.Float32(c["onorm_eps"]))
        y = (
            fx.ptr_load(c["ol"] + (j * VCH + e))
            * r
            * ld_bf(a["on_w"], vr)
            * sigmoid(f32_of(got[NVCH][0]))
        )
        bo.buffer_store(
            y.to(fx.BFloat16), rsrc(a["core_out"]), (req * L + j) * PROJ + h * HD + vr
        )
    gpu.barrier()


@flyc.jit
def kda_tasks(c, on, tls):
    s, L, tid, bid = c["S"], c["L"], c["tid"], c["bid"]
    for task in range(first_task(bid, c["kda0"]), (s // L) * NH * NVCH, BLOCKS):
        req = task // (NH * NVCH)
        h = (task // NVCH) % NH
        ch = task % NVCH
        k = kda_stage(c, req, h, ch)
        stamp(on, tls, tid, 7)
        kda_conv(c, k, req, h, ch)
        stamp(on, tls, tid, 8)
        kda_gates(c, k, req, h, ch)
        stamp(on, tls, tid, 9)
        kda_recur(c, k, req, h, ch)
        stamp(on, tls, tid, 10)
        kda_norm(c, req, h, ch)
        stamp(on, tls, tid, 11)


# ---------------------------------------------------------------- kernel
_BUILDS: dict = {}


def build(key: KdaPreBuild):
    if key in _BUILDS:
        return _BUILDS[key]
    s, L = key.tokens, key.qlen
    ns = key.nblocks + 1
    lay = scratch_layout(key)
    xrow = HIDDEN + LDS_PAD
    assert s * xrow * 2 <= 136 * 1024, "x rows exceed LDS"
    assert L <= 16 and key.nblocks >= 1

    @fx.struct
    class AttnLds:
        vals: fx.Array[fx.Float32, s * ns * COLS, 16]

    @fx.struct
    class GemvLds:
        xl: fx.Array[fx.Int32, s * xrow // 2, 16]

    @fx.struct
    class KdaLds:
        fa: fx.Array[fx.Float32, L * HD, 16]
        qkl: fx.Array[fx.Float32, L * 2 * HD, 16]
        decay: fx.Array[fx.Float32, L * HD, 16]
        vloc: fx.Array[fx.Float32, L * VCH, 16]
        ol: fx.Array[fx.Float32, L * VCH, 16]
        beta: fx.Array[fx.Float32, 16, 16]

    @fx.union
    class RoleLds:
        attn: AttnLds
        gemv: GemvLds
        kda: KdaLds

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        tls: fx.Array[fx.Int64, TL_POINTS, 16]

    keyed = key_tuple(key, SOURCES)
    name = (
        f"k3_mono_kda_pre_s{s}_l{L}_nb{key.nblocks}_d{int(key.delta)}"
        f"_g{int(key.debug)}_t{int(key.timeline)}_r{int(key.ring)}"
        + ("_ic" if key.init_committed else "")
    )

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kda_pre(
        prefix: Int64,
        delta: Int64,
        blocks: Int64,
        blk_sm: Int32,
        blk_sr: Int32,
        ares_nw: Int64,
        ares_qk: Int64,
        in_nw: Int64,
        w_in: Int64,
        w_fb: Int64,
        conv_w: Int64,
        conv_st: Int64,
        cs_seq: Int32,
        cs_dim: Int32,
        cs_tok: Int32,
        a_log: Int64,
        dt_bias: Int64,
        on_w: Int64,
        rstate: Int64,
        rs_seq: Int32,
        st_idx: Int64,
        st_stride: Int32,
        num_acc: Int64,
        ring_k: Int64,
        ring_v: Int64,
        ring_g: Int64,
        ring_b: Int64,
        rk_seq: Int32,
        rv_seq: Int32,
        rg_seq: Int32,
        rb_seq: Int32,
        ring_len: Int32,
        rstate0: Int64,
        rs0_seq: Int32,
        init_slot: Int64,
        core_out: Int64,
        x_dbg: Int64,
        proj_dbg: Int64,
        scratch: Int64,
        layer: Int32,
        epoch: Int64,
        tl: Int64,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        alloc = fx.SharedAllocator()
        lds = alloc.allocate(Smem).peek()
        role = alloc.allocate(RoleLds)
        al, gl, kl = role.attn.peek(), role.gemv.peek(), role.kda.peek()
        mb = Mailbox(step_tag(epoch, layer) - 1)
        c = {
            "S": s,
            "L": L,
            "NS": ns,
            "NB": key.nblocks,
            "SL": key.state_len,
            "tid": tid,
            "bid": bid,
            "lane": tid % 64,
            "wave": tid // 64,
            "delta": key.delta,
            "debug": key.debug,
            "eps": key.eps,
            "out_eps": key.out_eps,
            "onorm_eps": key.onorm_eps,
            "lb": key.lower_bound,
            "kda0": key.kda0,
            "xp": key.xp,
            "write_idx": key.write_idx,
            "ring": key.ring,
            "init_committed": key.init_committed,
            "gate_local": False,
            "put": mb.put,
            "poll": mb.poll,
            "red": lds.red.ptr,
            "vals": al.vals.ptr,
            "xl": gl.xl.ptr,
            "fa": kl.fa.ptr,
            "qkl": kl.qkl.ptr,
            "decay": kl.decay.ptr,
            "vloc": kl.vloc.ptr,
            "ol": kl.ol.ptr,
            "beta": kl.beta.ptr,
            "xbuf": scratch + fx.Int64(lay["x"][0]),
            "args": {
                "prefix": prefix,
                "delta": delta,
                "blocks": blocks,
                "blk_sm": blk_sm,
                "blk_sr": blk_sr,
                "ares_nw": ares_nw,
                "ares_qk": ares_qk,
                "in_nw": in_nw,
                "w_in": w_in,
                "w_fb": w_fb,
                "conv_w": conv_w,
                "conv_st": conv_st,
                "cs_seq": cs_seq,
                "cs_dim": cs_dim,
                "cs_tok": cs_tok,
                "a_log": a_log,
                "dt_bias": dt_bias,
                "on_w": on_w,
                "rstate": rstate,
                "rs_seq": rs_seq,
                "st_idx": st_idx,
                "st_stride": st_stride,
                "num_acc": num_acc,
                "ring_k": ring_k,
                "ring_v": ring_v,
                "ring_g": ring_g,
                "ring_b": ring_b,
                "rk_seq": rk_seq,
                "rv_seq": rv_seq,
                "rg_seq": rg_seq,
                "rb_seq": rb_seq,
                "ring_len": ring_len,
                "rstate0": rstate0,
                "rs0_seq": rs0_seq,
                "init_slot": init_slot,
                "core_out": core_out,
                "x_dbg": x_dbg,
                "proj_dbg": proj_dbg,
            },
        }
        for region in ("part", "wgt", "msq", "xrdy", "proj", "qk", "onrm"):
            c[region] = sreg(scratch, lay[region][0], region)
        on, tls = key.timeline, lds.tls.ptr
        c["on"], c["tls"] = on, tls
        stamp_begin(on, tls, tid, TL_POINTS)
        if bid < NA:
            stage_slice(c, bid)
            stamp(on, tls, tid, 1)
            if bid < s:
                stage_gate(c, bid)
            stamp(on, tls, tid, 2)
            stage_mix(c, bid)
            stamp(on, tls, tid, 3)
        if bid >= NA:
            gi = bid - NA
            rg0 = 2 * gi
            rg1 = 2 * gi + 1
            ops0 = gemv_loads(c, rg0, range(PER_WAVE))
            load_x(c)
            stamp(on, tls, tid, 4)
            # the second group's first half in flight under the first's MFMAs
            ops1 = gemv_loads(c, rg1, range(4))
            stage_gemv(c, rg0, ops0)
            stamp(on, tls, tid, 5)
            ops1 = ops1 + gemv_loads(c, rg1, range(4, PER_WAVE))
            stage_gemv(c, rg1, ops1)
            stamp(on, tls, tid, 6)
        kda_tasks(c, on, tls)
        stamp_flush(on, tls, tl, tid, bid, TL_POINTS)

    @flyc.jit
    def launch(
        prefix: Int64,
        delta: Int64,
        blocks: Int64,
        blk_sm: Int32,
        blk_sr: Int32,
        ares_nw: Int64,
        ares_qk: Int64,
        in_nw: Int64,
        w_in: Int64,
        w_fb: Int64,
        conv_w: Int64,
        conv_st: Int64,
        cs_seq: Int32,
        cs_dim: Int32,
        cs_tok: Int32,
        a_log: Int64,
        dt_bias: Int64,
        on_w: Int64,
        rstate: Int64,
        rs_seq: Int32,
        st_idx: Int64,
        st_stride: Int32,
        num_acc: Int64,
        ring_k: Int64,
        ring_v: Int64,
        ring_g: Int64,
        ring_b: Int64,
        rk_seq: Int32,
        rv_seq: Int32,
        rg_seq: Int32,
        rb_seq: Int32,
        ring_len: Int32,
        rstate0: Int64,
        rs0_seq: Int32,
        init_slot: Int64,
        core_out: Int64,
        x_dbg: Int64,
        proj_dbg: Int64,
        scratch: Int64,
        layer: Int32,
        epoch: Int64,
        tl: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        kda_pre(
            prefix,
            delta,
            blocks,
            blk_sm,
            blk_sr,
            ares_nw,
            ares_qk,
            in_nw,
            w_in,
            w_fb,
            conv_w,
            conv_st,
            cs_seq,
            cs_dim,
            cs_tok,
            a_log,
            dt_bias,
            on_w,
            rstate,
            rs_seq,
            st_idx,
            st_stride,
            num_acc,
            ring_k,
            ring_v,
            ring_g,
            ring_b,
            rk_seq,
            rv_seq,
            rg_seq,
            rb_seq,
            ring_len,
            rstate0,
            rs0_seq,
            init_slot,
            core_out,
            x_dbg,
            proj_dbg,
            scratch,
            layer,
            epoch,
            tl,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    _BUILDS[key] = launch
    return launch


def ptr(t):
    return 0 if t is None else t.data_ptr()


def kda_pre(
    key: KdaPreBuild,
    *,
    prefix,
    delta,
    blocks,
    ares_nw,
    ares_qk,
    in_nw,
    w_in,
    w_fb,
    conv_w,
    conv_state,
    a_log,
    dt_bias,
    on_w,
    rstate,
    st_idx,
    num_acc,
    core_out,
    scratch,
    layer,
    epoch,
    x_dbg=None,
    proj_dbg=None,
    tl=None,
    ring=None,
    rstate0=None,
    init_slot=None,
):
    """Host launcher. ``conv_state`` [slots, dim, state_len] or a transposed view
    of [slots, state_len, dim] (strides read); ``rstate`` [slots, 12, 128, 128]
    fp32; ``st_idx`` [R, >= L] int32; ``scratch`` zeroed before the step's first
    layer (tags are a layer's); ``epoch`` a device int32 bumped once a step and
    ``layer`` this launch's slot below ``TAG_SLOTS``."""
    assert w_in.shape == (NPROJ, HIDDEN) and w_in.is_contiguous()
    assert rstate.dtype == torch.float32 and rstate[0].numel() == NH * HD * HD
    assert rstate[0].is_contiguous(), (
        "a slot's state must be dense; slots may be strided"
    )
    assert conv_state.dtype == torch.bfloat16
    assert (ring is not None) == key.ring, (
        "a ring build takes the ring rows, and only it"
    )
    if ring is None:
        rk = rv = rg = rb = None
        seqs, ring_len = (0, 0, 0, 0), 0
    else:
        # [slots, heads, ring_len, 128] bf16 k / v and fp32 g; [slots, heads, ring_len] fp32 beta
        rk, rv, rg, rb = ring
        ring_len = rk.shape[2]
        assert (
            rk.dtype == rv.dtype == torch.bfloat16
            and rg.dtype == rb.dtype == torch.float32
        )
        assert ring_len >= key.qlen, f"a ring of {ring_len} rows for {key.qlen} tokens"
        for t in (rk, rv, rg):
            assert t.shape[1:] == (NH, ring_len, HD) and t[0].is_contiguous()
        assert rb.shape[1:] == (NH, ring_len) and rb[0].is_contiguous()
        seqs = (rk.stride(0), rv.stride(0), rg.stride(0), rb.stride(0))
    f = build(key)
    f(
        prefix.data_ptr(),
        ptr(delta),
        blocks.data_ptr(),
        blocks.stride(0),
        blocks.stride(1),
        ares_nw.data_ptr(),
        ares_qk.data_ptr(),
        in_nw.data_ptr(),
        w_in.data_ptr(),
        w_fb.data_ptr(),
        conv_w.data_ptr(),
        conv_state.data_ptr(),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        a_log.data_ptr(),
        dt_bias.data_ptr(),
        on_w.data_ptr(),
        rstate.data_ptr(),
        rstate.stride(0),
        st_idx.data_ptr(),
        st_idx.stride(0),
        num_acc.data_ptr(),
        ptr(rk),
        ptr(rv),
        ptr(rg),
        ptr(rb),
        *seqs,
        ring_len,
        ptr(rstate0),
        0 if rstate0 is None else rstate0.stride(0),
        ptr(init_slot),
        core_out.data_ptr(),
        ptr(x_dbg),
        ptr(proj_dbg),
        scratch.data_ptr(),
        layer,
        epoch.data_ptr(),
        ptr(tl),
        stream=torch.cuda.current_stream(),
    )
