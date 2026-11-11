# SPDX-License-Identifier: Apache-2.0
"""The D-MLP pieces D-pre shares (``draft_attn_half.py``): the draft's shapes, the GEMV
machinery and the coefficient / conv0 stages.

AKA: vendored from sgl-amd-mono @ 52486703 (sgl_amd_mono/muse_spark/gfx950/draft_mlp.py), keeping
only what D-pre imports: the shape constants, ``gemv_k``, ``w_loads``, ``gemv_tiles``, ``lds_b``,
``stage_coef``, ``stage_coef_reduce`` and ``stage_conv0``, unchanged apart from package-relative
imports. D-MLP's own launch, stages and scratch layout (and their ``post_attn.sigmoid`` import) are
not part of this task. Upstream module docstring::

    D-MLP: one DFlash2 draft layer's MLP half as one launch, for S = ``tokens`` rows
    (a block of 8 consecutive rows a request), on a rank of a TP group.
    Stages (every CTA; 256 CTAs, one task each): S0 norm statistics (the K parts'
    sums of squares through the mailbox) and the coefficient GEMV (16 row groups of
    128 rows x 16 K parts; fp32 partials); S1 the coefficients' reduce (8 rows a
    CTA); S2 conv0 on this K part and the gate / up GEMV ...
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from . import buffer_ops as bo
from .config import MAX_TOKENS
from .layout import RED_FLOATS, TILE, token_tiles
from .ops import CM_DEV, CM_NT, mfma_bf16, row_sum, rsrc
from .plan import BLOCKS, THREADS, WAVES

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

