# SPDX-License-Identifier: Apache-2.0
"""D-MLP: one DFlash2 draft layer's MLP half as one launch, for S = ``tokens`` rows
(a block of 8 consecutive rows a request), on a rank of a TP group::

    res' = bf16(res + attn_out)                     (the fused add of the post norm)
    h    = RMSNorm(res') w_post                      (eps 1e-5, bf16)
    delta = h W_kp^T                                 (2048: side x tap x group)
    hc   = conv0(h): (base[0][tap] + delta[token, 0, tap, g]) . h[token - tap] in the block
    g, u = hc W_gu^T (3584 gate rows, 3584 up rows a rank), a = silu(g) u (bf16)
    d    = a W_down^T                                (this rank's partial, 8192)
    d    = sum over the ranks (bf16 pairs through the peer regions, rank order)
    out  = conv1(d) with side 1's coefficients        (bf16)

Stages (every CTA; 256 CTAs, one task each): S0 norm statistics (the K parts'
sums of squares through the mailbox) and the coefficient GEMV (16 row groups of
128 rows x 16 K parts; fp32 partials); S1 the coefficients' reduce (8 rows a
CTA); S2 conv0 on this K part and the gate / up GEMV (16 x 448 rows x 16 K parts,
28 tiles in rounds of 7); S3 the reduce + silu (14 dims a CTA, bf16 pairs); S4 the
down GEMV (7 K parts of 512 x 32 row groups of 256 rows, 224 tasks); S5 the reduce
(32 rows a CTA), the push to every rank, the ranks' sum, conv1 and the output.
Seven mailbox hops; ~280 MB of weights a layer a rank stream once.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import dataclasses

from . import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from . import guard
from .epoch import kernel_symbol
from .ops import (
    CM_DEV,
    CM_NT,
    bf2_f32,
    bf16_round,
    butterfly,
    hw_rsq,
    mfma_bf16,
    peer_bases,
    row_sum,
    rsrc,
    sum_partials,
)
from .plan import (
    BLOCKS,
    THREADS,
    WAVES,
    key_tuple,
    pair_layout,
)
from .sync import Mailbox, preg, sreg
from .config import MAX_TOKENS
from .device import (
    TRACE_LAYERS,
    TRACE_SLOTS,
    stamp,
    step_tag,
)
from .layout import (
    RED_FLOATS,
    TILE,
    scratch_bytes,
    token_tiles,
)
from .post_attn import sigmoid

# AKA: FlyDSL hashes the helpers of a kernel's own directory, which in this flat package are
# all of them, so the upstream transitive source digest (plan.kernel_sources) is not needed;
# the FlyDSL 0.3.2 bridge (compat.py) is called directly by ops / epoch / guard instead of
# compat.install() patching them at import.
SOURCES = "aka-flat"
_STREAM = fx.Stream(None)

# ---------------------------------------------------------------- the draft's shapes (a rank of TP 8)
H = 8192  # hidden
I = 3584  # intermediate a rank (28672 / 8)
GU_ROWS = 2 * I  # the merged gate (first) and up rows
GROUP = 16  # channels a convolution group
GROUPS = H // GROUP  # 512
TAPS = 2
SIDES = 2
COEF_ROWS = (
    SIDES * TAPS * GROUPS
)  # 2048: the kernel projection's rows, (side, tap, group)
BLOCK = 8  # rows a request: the convolution shifts within it
EPS = 1e-5
ROWS = 16  # an MFMA tile's rows
K_PARTS, K_PART, K_PER_WAVE, K_STEP = 16, 512, 64, 32
K_STEPS = K_PER_WAVE // K_STEP  # 2
C_ROWS, C_TILES = 128, 8  # a coefficient GEMV task's rows
G_ROWS, G_TILES, G_ROUND = (
    448,
    28,
    4,
)  # a gate / up GEMV task's rows, in rounds of 4 tiles (32 + 32 VGPRs in flight)
D_KPARTS, D_RGS, D_ROWS, D_TILES, D_ROUND = (
    7,
    32,
    256,
    16,
    4,
)  # the down GEMV: 224 tasks, rounds of 4 tiles
D_TASKS = D_KPARTS * D_RGS
C_RED, G_RED, D_RED = (
    COEF_ROWS // BLOCKS,
    I // BLOCKS,
    H // BLOCKS,
)  # rows a reducer CTA: 8, 14, 32
RED_TILES = max(
    C_TILES, G_ROUND, D_ROUND
)  # reduction row sets in LDS (the coefficient task's 8)
assert K_PARTS * K_PART == H and C_TILES * ROWS == C_ROWS and 16 * C_ROWS == COEF_ROWS
assert G_TILES * ROWS == G_ROWS and 16 * G_ROWS == GU_ROWS and G_TILES % G_ROUND == 0
assert (
    D_KPARTS * K_PART == I
    and D_RGS * D_ROWS == H
    and D_TILES * ROWS == D_ROWS
    and D_TILES % D_ROUND == 0
)
assert D_TASKS <= BLOCKS and G_RED % 2 == 0 and D_RED % 2 == 0 and THREADS == 64 * WAVES


def dm_scratch_layout() -> dict:
    """D-MLP's scratch regions (pairs), sized for MAX_TOKENS: the K parts' sums of
    squares, the coefficient partials and sums, the gate / up partials, the
    activations (bf16 pairs), the down partials, the trace."""
    return pair_layout(
        [
            ("dm_ss", K_PARTS * MAX_TOKENS),
            ("dm_cpart", K_PARTS * MAX_TOKENS * COEF_ROWS),
            ("dm_coef", MAX_TOKENS * COEF_ROWS),
            ("dm_gpart", K_PARTS * MAX_TOKENS * GU_ROWS),
            ("dm_act", MAX_TOKENS * I // 2),
            ("dm_dpart", D_KPARTS * MAX_TOKENS * H),
            ("dm_trace", TRACE_LAYERS * BLOCKS * TRACE_SLOTS),
        ],
        align=256,
    )


def dm_sym_layout(npes: int) -> dict:
    """A rank's peer-symmetric region: ``dm`` [source rank][token][H / 2] bf16 pairs."""
    return pair_layout([("dm", npes * MAX_TOKENS * H // 2)], align=256)


def dm_sym_bytes(npes: int) -> int:
    return scratch_bytes(dm_sym_layout(npes))


def dm_lds(tokens: int):
    S = tokens

    @fx.struct
    class Lds:
        red: fx.Array[fx.Float32, RED_TILES * RED_FLOATS, 16]
        hbuf: fx.Array[fx.BFloat16, S * K_PART, 16]  # h, then hc, of this K part
        ssl: fx.Array[fx.Float32, S * K_PARTS, 16]  # the K parts' sums of squares
        nf: fx.Array[fx.Float32, S, 16]  # 1 / rms a token
        cf: fx.Array[
            fx.Float32, S * TAPS * (K_PART // GROUP), 16
        ]  # side 0 deltas of this K part's groups
        dbuf: fx.Array[fx.Float32, S * D_RED, 16]  # the summed down rows of this CTA
        cf1: fx.Array[
            fx.Float32, S * TAPS * (D_RED // GROUP), 16
        ]  # side 1 deltas of this CTA's groups

    return Lds


def dm_lds_bytes(tokens: int) -> int:
    S = tokens
    return (RED_TILES * RED_FLOATS + S * K_PARTS + S + S * TAPS * (K_PART // GROUP) + S * D_RED + S * TAPS * (D_RED // GROUP)) * 4 + S * K_PART * 2 + 7 * 16  # fmt: skip


# ---------------------------------------------------------------- shared pieces
def gemv_k(lane, wave, kpart, kc):
    return kpart * K_PART + wave * K_PER_WAVE + kc * K_STEP + 8 * (lane // 16)


def w_loads(c, w, width, row0, tiles, kpart):
    """Weight tiles ``row0 + 16 it + l16`` over this wave's K steps of part ``kpart`` of
    a row-major bf16 [*, width] matrix: ``[tile][step]`` 16 B a lane, in flight."""
    lane, wave = c["lane"], c["wave"]
    r_w = rsrc(w)
    return [
        [
            fx.Vector(
                bo.buffer_load(
                    r_w,
                    (
                        (row0 + it * ROWS + lane % 16) * width
                        + gemv_k(lane, wave, kpart, kc)
                    )
                    // 2,
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=CM_NT,
                )  # fmt: skip
            )
            for kc in range(K_STEPS)
        ]
        for it in range(tiles)
    ]


@flyc.jit
def gemv_tiles(c, w, xb_of, tiles, publish):
    """The MFMA tiles of a round: ``w[tile][step]`` weights, ``xb_of(t)`` the B
    operands of token tile t ([step]), the waves' accumulators summed through
    ``red`` and ``publish(tile, r, tk, v)`` called for every (row, token) of the
    round. Two barriers a token tile."""
    S, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    for t in range_constexpr(token_tiles(S)):
        xb = xb_of(t)
        for it in range_constexpr(tiles):
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for kc in range_constexpr(K_STEPS):
                acc = mfma_bf16(w[it][kc].bitcast(fx.BFloat16), xb[kc], acc)
            fx.ptr_store(acc, red + (it * RED_FLOATS + (wave * 64 + lane) * 4))
        gpu.barrier()
        nt = min(TILE, S - t * TILE)
        n_out = tiles * ROWS * nt
        for p in range_constexpr(-(-n_out // THREADS)):
            idx = tid + p * THREADS
            if idx < n_out:
                r = idx % (tiles * ROWS)
                tk = t * TILE + idx // (tiles * ROWS)
                v = row_sum(
                    red + (r // ROWS) * RED_FLOATS, r % ROWS, idx // (tiles * ROWS)
                )
                publish(r, tk, v)
        gpu.barrier()  # ``red`` is reused


def lds_b(c, base, tok_of, k_of):
    """B operands from LDS bf16 rows: token ``tok_of(t)`` of tile t, 8 values at ``k_of(kc)``."""
    V8B = fx.Vector.make_type(8, fx.BFloat16)

    def xb_of(t):
        tok = tok_of(t)
        return [
            fx.Vector(fx.ptr_load(base + (tok * K_PART + k_of(kc)), result_type=V8B))
            for kc in range(K_STEPS)
        ]

    return xb_of


# ---------------------------------------------------------------- S0: the norm, the coefficient GEMV
@flyc.jit
def stage_norm(c, rg, kp):
    """res' = bf16(res + attn_out) on this K part (row group 0 writes it out), the
    K part's sums of squares published (row group 0), every part's polled; h =
    RMSNorm(res') w_post -> ``hbuf``."""
    S, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    put, poll, hbuf, ssl, nf = c["put"], c["poll"], c["hbuf"], c["ssl"], c["nf"]
    a = c["args"]
    V8B = fx.Vector.make_type(8, fx.BFloat16)
    ch = kp * K_PART + 8 * lane  # this lane's eight channels
    r_res, r_attn, r_out = rsrc(a["res"]), rsrc(a["attn_out"]), rsrc(a["res_out"])
    wv = fx.Vector(
        bo.buffer_load(rsrc(a["w_post"]), ch // 2, vec_width=4, dtype=T.i32)
    ).bitcast(fx.BFloat16)
    TPW = S // WAVES  # tokens a wave
    rp = []
    for j in range_constexpr(TPW):
        t = wave + WAVES * j
        rv = fx.Vector(
            bo.buffer_load(r_res, (t * H + ch) // 2, vec_width=4, dtype=T.i32)
        ).bitcast(fx.BFloat16)
        av = fx.Vector(
            bo.buffer_load(r_attn, (t * H + ch) // 2, vec_width=4, dtype=T.i32)
        ).bitcast(fx.BFloat16)
        r8 = [bf16_round(fx.Float32(rv[e]) + fx.Float32(av[e])) for e in range(8)]
        ss = r8[0] * r8[0]
        for e in range_constexpr(1, 8):
            ss = ss + r8[e] * r8[e]
        ss = butterfly(ss, (1, 2, 4, 8, 16, 32))
        if rg == 0:
            bo.buffer_store(fx.Vector.from_elements(r8, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32), r_out, (t * H + ch) // 2)  # fmt: skip
            if lane == 0:
                put(c["ss"], kp * MAX_TOKENS + t, ss)
        rp.append(r8)
    # every part's sum of squares of every token (a pair a thread a pass; idle threads repeat one)
    N = S * K_PARTS
    for p in range_constexpr(-(-N // THREADS)):
        idx = fx.min(tid + p * THREADS, N - 1)
        got = poll([(c["ss"], (idx % K_PARTS) * MAX_TOKENS + idx // K_PARTS, 1)])
        if tid + p * THREADS < N:
            fx.ptr_store(got[0][0].bitcast(fx.Float32), ssl + idx)
    gpu.barrier()
    if tid < S:
        tot = fx.ptr_load(ssl + tid * K_PARTS)
        for p in range_constexpr(1, K_PARTS):
            tot = tot + fx.ptr_load(ssl + (tid * K_PARTS + p))
        fx.ptr_store(hw_rsq(tot / float(H) + EPS), nf + tid)
    gpu.barrier()
    for j in range_constexpr(TPW):
        t = wave + WAVES * j
        f = fx.ptr_load(nf + t)
        hv = fx.Vector.from_elements(
            [rp[j][e] * f * fx.Float32(wv[e]) for e in range(8)], fx.Float32
        ).to(fx.BFloat16)
        fx.ptr_store(hv, hbuf + (t * K_PART + 8 * lane))
    gpu.barrier()


@flyc.jit
def stage_coef(c, rg, kp, w):
    """The coefficient GEMV task (``rg``, ``kp``): rows 128 rg .. of W_kp over K part
    kp against h; fp32 partials to ``dm_cpart`` [kp][token][row]."""
    S, lane, wave, put = c["S"], c["lane"], c["wave"], c["put"]
    cpart = c["cpart"]

    def tok_of(t):
        return fx.min(t * TILE + lane % 16, S - 1)

    def k_of(kc):
        return wave * K_PER_WAVE + kc * K_STEP + 8 * (lane // 16)

    def publish(r, tk, v):
        put(cpart, (kp * MAX_TOKENS + tk) * COEF_ROWS + rg * C_ROWS + r, v)

    gemv_tiles(c, w, lds_b(c, c["hbuf"], tok_of, k_of), C_TILES, publish)


# ---------------------------------------------------------------- S1: the coefficients
@flyc.jit
def stage_coef_reduce(c, bid):
    """Rows 8 bid .. of the coefficients for every token: the K parts' partials summed
    in part order -> ``dm_coef`` [token][row] (fp32) and, with a ``coef_out`` argument
    (D-pre: D-post reads it), the plain fp32 [S, COEF_ROWS] tensor."""
    S, tid, poll, put = c["S"], c["tid"], c["poll"], c["put"]
    cpart, coef = c["cpart"], c["coef"]
    a = c["args"]
    N = C_RED * S
    for p in range_constexpr(-(-N // THREADS)):
        idx = fx.min(tid + p * THREADS, N - 1)
        j, t = idx % C_RED, idx // C_RED
        row = bid * C_RED + j
        got = poll(
            [
                (cpart, (kp * MAX_TOKENS + t) * COEF_ROWS + row, 1)
                for kp in range(K_PARTS)
            ],
            batch=K_PARTS,
        )
        v = got[0][0].bitcast(fx.Float32)
        for kp in range_constexpr(1, K_PARTS):
            v = v + got[kp][0].bitcast(fx.Float32)
        if tid + p * THREADS < N:
            put(coef, t * COEF_ROWS + row, v)
            if const_expr("coef_out" in a):
                bo.buffer_store(
                    v, rsrc(a["coef_out"]), t * COEF_ROWS + row, cache_modifier=CM_DEV
                )


# ---------------------------------------------------------------- S2: conv0, the gate / up GEMV
@flyc.jit
def stage_conv0(c, kp):
    """``hbuf`` <- conv0(h) on this K part: coefficient = base[0][tap][channel] +
    delta[token][0][tap][group], tap 1 reads the block's previous row."""
    S, tid, lane, wave, poll, hbuf, cf = (
        c["S"],
        c["tid"],
        c["lane"],
        c["wave"],
        c["poll"],
        c["hbuf"],
        c["cf"],
    )
    a = c["args"]
    GP = K_PART // GROUP  # groups a K part: 32
    V8B = fx.Vector.make_type(8, fx.BFloat16)
    # the deltas of this part's groups: (token, tap, group) a thread
    N = S * TAPS * GP
    for p in range_constexpr(-(-N // THREADS)):
        idx = fx.min(tid + p * THREADS, N - 1)
        t, tap, g = idx // (TAPS * GP), (idx // GP) % TAPS, idx % GP
        got = poll(
            [(c["coef"], t * COEF_ROWS + (0 * TAPS + tap) * GROUPS + kp * GP + g, 1)]
        )
        if tid + p * THREADS < N:
            fx.ptr_store(got[0][0].bitcast(fx.Float32), cf + idx)
    ch = kp * K_PART + 8 * lane
    r_base = rsrc(a["base"])
    b0 = [
        fx.Vector(
            bo.buffer_load(
                r_base, ((0 * TAPS + tap) * H + ch) // 2, vec_width=4, dtype=T.i32
            )
        ).bitcast(fx.BFloat16)
        for tap in range(TAPS)
    ]
    gpu.barrier()  # the deltas are in
    TPW = S // WAVES
    outs = []
    for j in range_constexpr(TPW):
        t = wave + WAVES * j
        tp = fx.max(t - 1, fx.Int32(0))
        hv = fx.Vector(fx.ptr_load(hbuf + (t * K_PART + 8 * lane), result_type=V8B))
        pv = fx.Vector(fx.ptr_load(hbuf + (tp * K_PART + 8 * lane), result_type=V8B))
        has_prev = (t % BLOCK) >= 1
        g = lane // 2  # the lane's group within the part (8 channels: half a group)
        d0 = fx.ptr_load(cf + ((t * TAPS + 0) * GP + g))
        d1 = fx.ptr_load(cf + ((t * TAPS + 1) * GP + g))
        vals = []
        for e in range_constexpr(8):
            c0 = fx.Float32(b0[0][e]) + d0
            c1 = fx.Float32(b0[1][e]) + d1
            v = c0 * fx.Float32(hv[e]) + has_prev.select(
                c1 * fx.Float32(pv[e]), fx.Float32(0.0)
            )
            vals.append(v)
        outs.append(fx.Vector.from_elements(vals, fx.Float32).to(fx.BFloat16))
    gpu.barrier()  # every read of h is done
    for j in range_constexpr(TPW):
        t = wave + WAVES * j
        fx.ptr_store(outs[j], hbuf + (t * K_PART + 8 * lane))
    gpu.barrier()


@flyc.jit
def stage_gu(c, rg, kp):
    """The gate / up GEMV task (``rg``, ``kp``): rows 448 rg .. of W_gu over K part kp
    against hc, in rounds of 7 tiles (the next round's weights in flight); fp32
    partials to ``dm_gpart`` [kp][token][row]."""
    S, lane, wave, put = c["S"], c["lane"], c["wave"], c["put"]
    gpart = c["gpart"]
    a = c["args"]

    def tok_of(t):
        return fx.min(t * TILE + lane % 16, S - 1)

    def k_of(kc):
        return wave * K_PER_WAVE + kc * K_STEP + 8 * (lane // 16)

    xb_of = lds_b(c, c["hbuf"], tok_of, k_of)
    cur = w_loads(c, a["w_gu"], H, rg * G_ROWS, G_ROUND, kp)
    for rd in range_constexpr(G_TILES // G_ROUND):
        if const_expr(rd + 1 < G_TILES // G_ROUND):
            nxt = w_loads(
                c, a["w_gu"], H, rg * G_ROWS + (rd + 1) * G_ROUND * ROWS, G_ROUND, kp
            )

        def publish(r, tk, v, rd=rd):
            put(
                gpart,
                (kp * MAX_TOKENS + tk) * GU_ROWS
                + rg * G_ROWS
                + rd * G_ROUND * ROWS
                + r,
                v,
            )

        gemv_tiles(c, cur, xb_of, G_ROUND, publish)
        if const_expr(rd + 1 < G_TILES // G_ROUND):
            cur = nxt


# ---------------------------------------------------------------- S3: the activations
@flyc.jit
def stage_act(c, bid):
    """Dims 14 bid .. of the intermediate for every token: the K parts' gate and up
    partials summed in part order, rounded to bf16, a = silu(g) u -> ``dm_act``
    [token][dim] bf16 pairs (two dims a thread)."""
    S, tid, poll, put_bf = c["S"], c["tid"], c["poll"], c["put_bf"]
    gpart, act = c["gpart"], c["act"]
    PAIRS = G_RED // 2  # 7 dim pairs a CTA
    N = PAIRS * S
    for p in range_constexpr(-(-N // THREADS)):
        idx = fx.min(tid + p * THREADS, N - 1)
        j, t = idx % PAIRS, idx // PAIRS
        d0 = bid * G_RED + 2 * j
        gu = poll(
            [(gpart, (kp * MAX_TOKENS + t) * GU_ROWS + d0, 2) for kp in range(K_PARTS)]
            + [
                (gpart, (kp * MAX_TOKENS + t) * GU_ROWS + I + d0, 2)
                for kp in range(K_PARTS)
            ],
            batch=2 * K_PARTS,
        )
        g0, g1, u0, u1 = (
            fx.Float32(0.0),
            fx.Float32(0.0),
            fx.Float32(0.0),
            fx.Float32(0.0),
        )
        for kp in range_constexpr(K_PARTS):
            g0 = g0 + gu[kp][0].bitcast(fx.Float32)
            g1 = g1 + gu[kp][1].bitcast(fx.Float32)
            u0 = u0 + gu[K_PARTS + kp][0].bitcast(fx.Float32)
            u1 = u1 + gu[K_PARTS + kp][1].bitcast(fx.Float32)
        g0, g1, u0, u1 = bf16_round(g0), bf16_round(g1), bf16_round(u0), bf16_round(u1)
        a0 = bf16_round(g0 * sigmoid(g0)) * u0
        a1 = bf16_round(g1 * sigmoid(g1)) * u1
        if tid + p * THREADS < N:
            put_bf(act, t * I + d0, [a0, a1])


# ---------------------------------------------------------------- S4: the down GEMV
@flyc.jit
def stage_down(c, rg, kp):
    """The down GEMV task (``rg``, ``kp``): rows 256 rg .. of W_down over K part kp (of
    7) against a (polled bf16 pairs), in two rounds of 8 tiles; fp32 partials to
    ``dm_dpart`` [kp][token][row]."""
    S, lane, wave, put, poll = c["S"], c["lane"], c["wave"], c["put"], c["poll"]
    dpart, act = c["dpart"], c["act"]
    a = c["args"]
    k_lane = [
        kp * K_PART + wave * K_PER_WAVE + kc * K_STEP + 8 * (lane // 16)
        for kc in range(K_STEPS)
    ]

    def xb_of(t):
        tok = fx.min(t * TILE + lane % 16, S - 1)
        # a spec is at most two pairs (a 16 B load): four pairs a k-step as two specs
        got = poll([(act, (tok * I + k_lane[kc]) // 2 + 2 * q, 2) for kc in range(K_STEPS) for q in range(2)], batch=2 * K_STEPS)  # fmt: skip
        return [
            fx.Vector.from_elements(
                [
                    fx.Int32(got[2 * kc][0]),
                    fx.Int32(got[2 * kc][1]),
                    fx.Int32(got[2 * kc + 1][0]),
                    fx.Int32(got[2 * kc + 1][1]),
                ],
                fx.Int32,
            ).bitcast(fx.BFloat16)
            for kc in range(K_STEPS)
        ]

    cur = w_loads(c, a["w_down"], I, rg * D_ROWS, D_ROUND, kp)
    for rd in range_constexpr(D_TILES // D_ROUND):
        if const_expr(rd + 1 < D_TILES // D_ROUND):
            nxt = w_loads(
                c, a["w_down"], I, rg * D_ROWS + (rd + 1) * D_ROUND * ROWS, D_ROUND, kp
            )

        def publish(r, tk, v, rd=rd):
            put(
                dpart,
                (kp * MAX_TOKENS + tk) * H + rg * D_ROWS + rd * D_ROUND * ROWS + r,
                v,
            )

        gemv_tiles(c, cur, xb_of, D_ROUND, publish)
        if const_expr(rd + 1 < D_TILES // D_ROUND):
            cur = nxt


# ---------------------------------------------------------------- S5: the reduce, the all-reduce, conv1, out
@flyc.jit
def stage_out(c, bid):
    """Rows 32 bid .. of d for every token: the K parts' partials summed, pushed as
    bf16 pairs to every rank's ``dm`` region, the ranks' summed in rank order (two
    rows a thread), conv1 with side 1's coefficients over the block, out (bf16)."""
    S, W, tid, poll, put_bf = c["S"], c["W"], c["tid"], c["poll"], c["put_bf"]
    dpart, dbuf, cf1 = c["dpart"], c["dbuf"], c["cf1"]
    a = c["args"]
    rank = c["rank"]
    GP = D_RED // GROUP  # groups a CTA: 2
    PAIRS = D_RED // 2  # 16 row pairs
    # side 1's deltas of this CTA's groups: (token, tap, group) a thread
    N1 = S * TAPS * GP
    for p in range_constexpr(-(-N1 // THREADS)):
        idx = fx.min(tid + p * THREADS, N1 - 1)
        t, tap, g = idx // (TAPS * GP), (idx // GP) % TAPS, idx % GP
        got = poll(
            [(c["coef"], t * COEF_ROWS + (1 * TAPS + tap) * GROUPS + bid * GP + g, 1)]
        )
        if tid + p * THREADS < N1:
            fx.ptr_store(got[0][0].bitcast(fx.Float32), cf1 + idx)
    N = PAIRS * S
    for p in range_constexpr(-(-N // THREADS)):
        idx = fx.min(tid + p * THREADS, N - 1)
        j, t = idx % PAIRS, idx // PAIRS
        col = bid * D_RED + 2 * j
        got = poll(
            [(dpart, (kp * MAX_TOKENS + t) * H + col, 2) for kp in range(D_KPARTS)],
            batch=D_KPARTS,
        )
        v0, v1 = fx.Float32(0.0), fx.Float32(0.0)
        for kp in range_constexpr(D_KPARTS):
            v0 = v0 + got[kp][0].bitcast(fx.Float32)
            v1 = v1 + got[kp][1].bitcast(fx.Float32)
        live = tid + p * THREADS < N
        if live:
            for pe in range_constexpr(W):
                put_bf(c["peer_dm"][pe], (rank * MAX_TOKENS + t) * H + col, [v0, v1])

        def pair_at(src):
            return ((src * MAX_TOKENS + t) * H + col) // 2

        d0, d1 = sum_partials(poll, c["own_dm"], pair_at, W)
        if live:
            fx.ptr_store(d0, dbuf + (t * D_RED + 2 * j))
            fx.ptr_store(d1, dbuf + (t * D_RED + 2 * j + 1))
    gpu.barrier()
    r_base, r_out = rsrc(a["base"]), rsrc(a["out"])
    for p in range_constexpr(-(-N // THREADS)):
        idx = fx.min(tid + p * THREADS, N - 1)
        j, t = idx % PAIRS, idx // PAIRS
        col = bid * D_RED + 2 * j
        g = (2 * j) // GROUP
        bw = [bf2_f32(fx.Int32(bo.buffer_load(r_base, ((1 * TAPS + tap) * H + col) // 2, vec_width=1, dtype=T.i32))) for tap in range(TAPS)]  # fmt: skip
        has_prev = (t % BLOCK) >= 1
        tp = fx.max(t - 1, fx.Int32(0))
        vals = []
        for e in range_constexpr(2):
            b0e, b1e = bw[0][e], bw[1][e]
            c0 = b0e + fx.ptr_load(cf1 + ((t * TAPS + 0) * GP + g))
            c1 = b1e + fx.ptr_load(cf1 + ((t * TAPS + 1) * GP + g))
            cur = fx.ptr_load(dbuf + (t * D_RED + 2 * j + e))
            prev = fx.ptr_load(dbuf + (tp * D_RED + 2 * j + e))
            vals.append(c0 * cur + has_prev.select(c1 * prev, fx.Float32(0.0)))
        if tid + p * THREADS < N:
            bo.buffer_store(fx.Vector.from_elements(vals, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0], r_out, (t * H + col) // 2)  # fmt: skip


# ---------------------------------------------------------------- the launch
@flyc.jit
def run_dmlp(c):
    """The stages in order; ``c["stages"]`` (a build's, for bisecting) stops after that many."""
    bid, n = c["bid"], c["stages"]
    rg, kp = bid // K_PARTS, bid % K_PARTS
    stamp(c, 0)
    w0 = w_loads(
        c, c["args"]["w_kp"], H, rg * C_ROWS, C_TILES, kp
    )  # the coefficient task's weights, in flight
    stage_norm(c, rg, kp)
    stamp(c, 1)
    if const_expr(n > 1):
        stage_coef(c, rg, kp, w0)
    stamp(c, 2)
    if const_expr(n > 2):
        stage_coef_reduce(c, bid)
    stamp(c, 3)
    if const_expr(n > 3):
        stage_conv0(c, kp)
    stamp(c, 4)
    if const_expr(n > 4):
        stage_gu(c, rg, kp)
    stamp(c, 5)
    if const_expr(n > 5):
        stage_act(c, bid)
    stamp(c, 6)
    if const_expr(n > 6):
        if bid < D_TASKS:
            stage_down(c, bid // D_KPARTS, bid % D_KPARTS)
    stamp(c, 7)
    if const_expr(n > 7):
        stage_out(c, bid)
    stamp(c, 8)


@dataclasses.dataclass(frozen=True)
class DMLPBuild:
    tokens: int
    npes: int = 8
    fault_site: int = 0
    fault_after: int = 0
    trace: bool = False
    stages: int = 8  # the stages run (fewer: a bisection build)


def dmlp_context(
    key: DMLPBuild,
    lds,
    args: dict,
    scratch,
    sym,
    peers,
    rank,
    tag,
    layout: dict,
    sym_layout: dict,
    layer=None,
) -> dict:
    s, npes = key.tokens, key.npes
    tid = fx.thread_idx.x
    mb = Mailbox(tag)
    bases = peer_bases(peers, npes)
    dm_off = sym_layout["dm"][0]
    c = {
        "S": s,
        "W": npes,
        "tid": tid,
        "bid": fx.block_idx.x,
        "lane": tid % 64,
        "wave": tid // 64,
        "red": lds.red.ptr,
        "hbuf": lds.hbuf.ptr,
        "ssl": lds.ssl.ptr,
        "nf": lds.nf.ptr,
        "cf": lds.cf.ptr,
        "dbuf": lds.dbuf.ptr,
        "cf1": lds.cf1.ptr,
        "put": mb.put,
        "put_bf": mb.put_bf,
        "poll": mb.poll,
        "args": args,
        "rank": rank,
        "trace_on": key.trace,
        "layer": layer,
        "stages": key.stages,
        "own_dm": preg(sym, dm_off, "dm"),
        "peer_dm": [preg(base, dm_off, "dm") for base in bases],
    }
    for region, (off, _) in layout.items():
        c[region[3:]] = sreg(scratch, off, region)  # "dm_cpart" -> c["cpart"]
    return c


def build_dmlp(key: DMLPBuild):
    """``@flyc.jit`` launcher of D-MLP for ``key.tokens`` rows on a rank of a
    ``key.npes``-way TP group. Arguments: ``res`` / ``attn_out`` bf16 [S, H],
    ``w_post`` bf16 [H], ``w_kp`` bf16 [2048, H], ``base`` bf16 [2, 2, H], ``w_gu``
    bf16 [7168, H], ``w_down`` bf16 [H, 3584], ``res_out`` / ``out`` bf16 [S, H],
    ``scratch`` (``dm_scratch_layout`` bytes), ``sym`` (this rank's peer buffer),
    ``peers`` (the ranks' peer bases, int64 [npes]), ``rank``, ``epoch``, ``layer``."""
    S, npes = key.tokens, key.npes
    assert S % WAVES == 0 and 1 <= S <= MAX_TOKENS
    name = kernel_symbol(
        "muse_spark_dmlp", s=S, w=npes, f=key.fault_site, t=int(key.trace), n=key.stages
    )
    keyed = key_tuple(key, SOURCES)
    fault_site, fault_after = key.fault_site, key.fault_after
    layout, sym_layout = dm_scratch_layout(), dm_sym_layout(npes)
    Lds = dm_lds(S)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def dmlp(
        res: Int64, attn_out: Int64, w_post: Int64, w_kp: Int64, base: Int64, w_gu: Int64, w_down: Int64,
        res_out: Int64, out: Int64, scratch: Int64, sym: Int64, peers: Int64, rank: Int32, epoch: Int64, layer: Int32,
    ):  # fmt: skip
        with guard.scope(
            guard.Guard(epoch, fault_site=fault_site or None, fault_after=fault_after)
        ):
            lds = fx.SharedAllocator().allocate(Lds).peek()
            args = {
                "res": res, "attn_out": attn_out, "w_post": w_post, "w_kp": w_kp, "base": base,
                "w_gu": w_gu, "w_down": w_down, "res_out": res_out, "out": out,
            }  # fmt: skip
            c = dmlp_context(
                key,
                lds,
                args,
                scratch,
                sym,
                peers,
                rank,
                step_tag(epoch, layer),
                layout,
                sym_layout,
                layer=layer,
            )
            run_dmlp(c)

    @flyc.jit
    def launch(
        res: Int64, attn_out: Int64, w_post: Int64, w_kp: Int64, base: Int64, w_gu: Int64, w_down: Int64,
        res_out: Int64, out: Int64, scratch: Int64, sym: Int64, peers: Int64, rank: Int32, epoch: Int64, layer: Int32,
        stream: fx.Stream = _STREAM,
    ):  # fmt: skip
        _ = keyed
        dmlp(
            res,
            attn_out,
            w_post,
            w_kp,
            base,
            w_gu,
            w_down,
            res_out,
            out,
            scratch,
            sym,
            peers,
            rank,
            epoch,
            layer,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    return launch
