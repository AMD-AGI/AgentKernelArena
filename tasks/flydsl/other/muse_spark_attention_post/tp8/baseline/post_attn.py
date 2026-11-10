# SPDX-License-Identifier: Apache-2.0
"""K2: one Muse Spark layer from the attention output to the experts' input,
for S = ``tokens`` rows; per token::

    o (bf16, the attention output), gate (K1's o_gate)
      -> per-head RMSNorm without weight (eps RMS_EPS) x sigmoid(gate), one bf16
         rounding (the fused norm-gate)
      -> o_proj GEMV (bf16 MFMA, fp32 accumulate, one rounding) -> the bf16
         partial rows pushed to every rank's ``attn`` region -> rank-ordered fp32
         sum, one rounding (the aiter one-stage all-reduce's numerics)
      -> attention boundary (the fused post-norm / residual / pre-norm with a
         weightless post norm): nb = bf16(attn_out rstd) (eps POST_EPS);
         stream = alpha stream + beta nb in fp32, in place; x = bf16(stream rstd2
         w_pre) (eps RMS_EPS), the FFN input
      -> router: the fp32 weight as bf16 hi + lo (the joint router / pooler's
         split): x @ hi + x @ lo in fp32, x ROUTER_MULT, sigmoid, + expert bias,
         top-8 by one integer max a pick (a tie to the lower expert), the picks'
         sigmoids / (sum + ROUTE_EPS) x ROUTE_SCALE -> ids (int32), scores (f32)
      -> pooler GEMV (bf16, one rounding) -> pre-expert norm (eps RMS_EPS,
         weight) -> expert input (bf16 [S, MOE_HIDDEN])

The work is task lists over the CTAs. Loads complete in order, so a CTA's
mailbox poll waits behind its own weight loads in flight: the last
``MAX_TOKENS`` CTAs (the chain CTAs, from ``CHAIN0``) issue their pooler rows'
weights at once, landed before their first poll, and run the per-token tasks
unqueued; the other CTAs (the stream CTAs) issue the pooler's weights after
their o tasks and wait on the tokens' flags behind them, as they must anyway::

    og      S tasks, stream CTA OG0 + t: token t's normed, gated o -> ``og``
    o       512 tasks (16 rows each), two a CTA on every CTA: o_proj partial
            rows -> every rank's ``attn`` region
    a_part  a_parts(S) tasks a token, chain CTAs AP0 ..: the all-reduce sum of
            the part's columns (the ranks' pairs polled in one round), the boundary
            with the row's sums of squares exchanged through ``ss``, x -> ``xbuf``
            (plain bf16 words), the stream in place, part 0 the token's ``xflag``
    router  128 tasks (16 expert tiles x an eighth of K), stream CTAs R0 ..:
            partial logits -> ``logits`` [part][token][expert]
    pool    256 tasks (128 rows x an eighth of K), one a CTA, the same eighth as
            the CTA's router task: partial pooled rows -> ``pool`` [part][token][row]
    route   S tasks, chain CTA CHAIN0 + t: top-8 -> ids / scores
    xn      S tasks, stream CTA t (beside the route): the pre-expert norm -> expert input

Width scaling (8 .. 64 rows): the o GEMV holds every token's og row in LDS
(S x 2 KB) and runs a token tile of 16 an MFMA accumulator, each tile's rows
summed through one reduction row set in turn; the router / pooler GEMVs load
each token tile's x of their K eighth straight into the B operands (16 B a lane a
step, no staging: a task reads S x 2 KB of x, not S x 16 KB) and sum their row
tiles through ``red_sets(S)`` row sets a round; the boundary runs ``a_parts(S)``
= 8 / 4 / 2 / 1 CTAs a token on the 64 chain CTAs.

The geometry constants, ``k2_scratch_layout`` and ``sym_layout`` belong in
``layout.py`` and live here until the K2 / K3 ports merge: ``sym_layout`` is the
rank's peer buffer for the whole layer (K3 pushes the FFN partials to ``ffn``
and the unpooled rows to ``unp``), so the K3 port imports it from here.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from . import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from .ops import (
    CM_DEV,
    CM_NT,
    bf2_f32,
    bf16_pair,
    bf16_round,
    butterfly,
    fp8_to_bf16x8,
    hw_exp2,
    hw_rsq,
    mfma_bf16,
    row_sum,
    rsrc,
)
from .plan import BLOCKS, PAIR_BYTES, THREADS, WAVES, pair_layout
from .sync import preg
from .config import (
    EXPERTS,
    HEAD_DIM,
    HIDDEN,
    LOCAL_KV_HEADS,
    LOG2E,
    MAX_TOKENS,
    MOE_HIDDEN,
    POST_EPS,
    Q_K,
    RMS_EPS,
    ROUTE_EPS,
    ROUTE_SCALE,
    ROUTER_MULT,
    TOP_K,
)
from .device import (
    acquire_loads,
    poll_flags,
    release_stores,
    stamp,
)
from .layout import (
    GQA,
    K_STEP,
    RED_FLOATS,
    ROWS,
    STEP_BATCH,
    TILE,
    TRACE_LAYERS,
    TRACE_SLOTS,
    scratch_bytes,
    token_tiles,
)

# ---------------------------------------------------------------- K2 geometry (for layout.py)
O_TASKS = HIDDEN // ROWS  # 512 o_proj row groups
O_K_PER_WAVE = Q_K // WAVES  # 128: the o GEMV's K over the waves
O_STEPS = O_K_PER_WAVE // K_STEP  # 4 MFMA steps a wave a token tile
OG_THREADS = Q_K // 4  # 256: the og task's threads, 4 o elements each
OG_POLL = 8  # tokens an og-row poll on a stream CTA (16 B a thread a token)
K_PER_WAVE = HIDDEN // WAVES  # 1024: the router / pooler GEMVs' K over the waves
K_STEPS = K_PER_WAVE // K_STEP  # 32
R_TILES = EXPERTS // ROWS  # 16 expert tiles
R_PARTS = WAVES  # the router's K split: a task takes one wave's share of K
R_TASKS = R_TILES * R_PARTS  # 128: partial logits, summed by the route tasks
P_ROWS = 128  # pooler rows a task: eight MFMA row tiles
P_SUB = (
    P_ROWS // ROWS
)  # 8: the largest pooler job (row tiles); the LDS reduction is sized for it
P_TILES = MOE_HIDDEN // P_ROWS  # 32 pooler row tiles
# The pooler's 256 row tiles a K eighth over the 32 CTAs of that eighth (bid % 8): the
# 24 stream CTAs take NS tiles each, the 8 chain CTAs NC. A chain CTA issues its
# weights only after its boundary task (loads complete in order: issued before, they
# would queue the boundary's polls), so its job is small and lands at once; the
# stream CTAs' bigger jobs stream during the o tasks and the boundary.
P_NS = 10  # 24 x 10 = 240 row tiles
P_NC = 2  # 8 x 2 = 16
assert 24 * P_NS + 8 * P_NC == MOE_HIDDEN // ROWS and P_NS <= P_SUB + 2
P_PARTS = R_PARTS  # the pooler's K split is the router's: a CTA's two tasks share a K eighth of x
from .ffn import POOL_PARTS  # noqa: E402

assert POOL_PARTS == P_PARTS
P_STEPS = (
    K_STEPS // WAVES
)  # 4 MFMA steps a wave a (row tile, token tile): a part's K over the waves
A_PARTS_MAX = 8  # CTAs a token's attention boundary at most
A_PAIRS_MAX = HIDDEN // (
    2 * THREADS
)  # 8: a thread's column pairs with the whole row on one CTA
CHAIN0 = BLOCKS - MAX_TOKENS  # 192: the chain CTAs, token bid - CHAIN0's route and xn
AP0 = CHAIN0  # the boundary tasks: the chain CTAs, before their pooler weights go out
A_TASKS_MAX = BLOCKS - CHAIN0  # 64 boundary tasks at most
R0 = 0  # the router tasks
assert O_TASKS == 2 * BLOCKS and R0 + R_TASKS <= CHAIN0  # two o tasks a CTA, every CTA
assert MAX_TOKENS <= CHAIN0 and MAX_TOKENS <= A_TASKS_MAX
assert K_STEPS % STEP_BATCH == 0 and K_STEPS % WAVES == 0 and O_K_PER_WAVE % K_STEP == 0
assert (
    EXPERTS == 4 * 64 and TOP_K <= 64
)  # a lane holds 4 routing keys; the picks on one wave
assert HIDDEN == 16 * THREADS and MOE_HIDDEN == 8 * THREADS and Q_K == 2 * THREADS
assert (
    P_TILES * P_PARTS == BLOCKS
    and MAX_TOKENS % TILE == 0
    and OG_THREADS <= THREADS
    and (BLOCKS - CHAIN0) == 8 * 8
    and CHAIN0 == 24 * 8
)


def a_parts(s: int) -> int:
    """CTAs a token's attention boundary at width ``s``: the largest power of two
    (at most A_PARTS_MAX) whose tasks fit the A_TASKS_MAX chain CTAs. (One task a
    token pulls 8 ranks x 16 KB of partials into one CTA: measured no better on one
    GPU and worse in the TP 8 server than eight parts with their three hops.)"""
    p = A_PARTS_MAX
    while p > 1 and p * s > A_TASKS_MAX:
        p //= 2
    assert p * s <= A_TASKS_MAX and A_PAIRS_MAX % p == 0, (s, p)
    return p


def k2_scratch_layout() -> dict:
    """K2's scratch regions, name -> (byte offset, bytes), sized for MAX_TOKENS:
    the tokens' normed gated o (bf16 pairs), a flag a token for its x, the
    boundary parts' norm sums (the flags of their x halves), the
    router parts' partial logits (fp32), the pooled rows (bf16 pairs), x as plain
    bf16 words in two halves (``k2_xbuf``, ``k2_xbuf2``, bytes given as pairs), a
    flag a boundary part once its stream columns are out, and the merged launch's
    hand-off flags: a token's routing / expert input out, a (request, KV head,
    row tile)'s o rows out."""
    return pair_layout(
        [
            ("k2_og", MAX_TOKENS * Q_K // 2),
            ("k2_xflag", MAX_TOKENS),
            ("k2_ss", 4 * MAX_TOKENS * A_PARTS_MAX),
            ("k2_logits", R_PARTS * MAX_TOKENS * EXPERTS),
            ("k2_pool", P_PARTS * MAX_TOKENS * MOE_HIDDEN // 2),
            ("k2_xbuf", MAX_TOKENS * HIDDEN * 2 // PAIR_BYTES),
            ("k2_xbuf2", MAX_TOKENS * HIDDEN * 2 // PAIR_BYTES),
            ("k2_c1", MAX_TOKENS * HIDDEN * 2 // PAIR_BYTES),
            # the expert input's MXFP4 blocks a token: MOE_HIDDEN / 8 code words, then MOE_HIDDEN / 32
            # scale words (K3's staged-column layout, ``ffn.stage_xq_all``)
            (
                "k2_xinq",
                MAX_TOKENS * (MOE_HIDDEN // 8 + MOE_HIDDEN // 32) * 4 // PAIR_BYTES,
            ),
            ("k2_resflag", MAX_TOKENS * A_PARTS_MAX),
            ("k2_rflag", MAX_TOKENS),
            ("k2_xinflag", MAX_TOKENS),
            ("k2_oflag", MAX_TOKENS * LOCAL_KV_HEADS * 4),
            ("k2_trace", TRACE_LAYERS * BLOCKS * TRACE_SLOTS),
        ],
        align=256,
    )


def sym_layout(npes: int) -> dict:
    """A rank's peer-symmetric buffer, name -> (byte offset, bytes): the attention
    and FFN partial rows, [source rank][token][width] as bf16 pairs, and the
    unpooled rows [token][HIDDEN / 2] (each rank's rows from that rank). Shared
    by the layer: K2 pushes ``attn``, K3 ``ffn`` and ``unp``. The guard's peer
    flags follow at ``sym_bytes(npes)``."""
    from .ffn import k3_sym_layout

    return k3_sym_layout(
        npes
    )  # one layout: K3's (attn, ffn, unp, then the TP-experts regions)


def sym_bytes(npes: int) -> int:
    """The peer buffer's data bytes: the guard flags' offset."""
    return scratch_bytes(sym_layout(npes))


def k2_xs_floats(tokens: int) -> int:
    """The ``xs`` LDS region: every token's og row (the o GEMV's B rows)."""
    return tokens * Q_K // 2


def red_sets(tokens: int) -> int:
    """Reduction row sets of the router / pooler passes (a row tile each a round):
    what the LDS leaves beside the og rows."""
    return (
        P_NS + 1 if tokens <= 16 else (4 if tokens <= 32 else 2)
    )  # one round of 9 tiles up to 16 rows


def k2_lds_bytes(tokens: int) -> int:
    return 4 * (
        k2_xs_floats(tokens)
        + red_sets(tokens) * RED_FLOATS
        + ROWS * MAX_TOKENS
        + EXPERTS
    )


def k2_lds(tokens: int):
    """K2's LDS for a step of ``tokens`` rows: the o GEMV's B rows, then the
    router / pooler GEMVs' staged x (the same region), one reduction row set, a
    task's output rows for every token, a token's sigmoids."""

    @fx.struct
    class Lds:
        xs: fx.Array[fx.Float32, k2_xs_floats(tokens), 16]
        red: fx.Array[fx.Float32, red_sets(tokens) * RED_FLOATS, 16]
        misc: fx.Array[fx.Float32, ROWS * MAX_TOKENS, 16]
        sig: fx.Array[fx.Float32, EXPERTS, 16]

    return Lds


# ---------------------------------------------------------------- device helpers
def sigmoid(z):
    """Triton's ``tl.sigmoid``: 1 / (1 + exp2(-z log2(e)))."""
    return fx.Float32(1.0) / (fx.Float32(1.0) + hw_exp2(z * fx.Float32(-LOG2E)))


def route_key(v, e):
    """Signed-orderable i32 of score ``v`` with its low 8 bits replaced by 255 -
    expert ``e``: a pick is one integer max, a tie goes to the lower expert."""
    b = v.bitcast(fx.Int32)
    b = (b < 0).select(b ^ fx.Int32(0x7FFFFFFF), b)
    return (b & fx.Int32(-256)) | (255 - e)


@flyc.jit
def block_sum(c, v, slot):
    """The CTA's sum of ``v`` over its threads, in wave order; ``slot`` picks the
    WAVES words of ``red`` (consecutive calls alternate slots)."""
    lane, wave, red = c["lane"], c["wave"], c["red"]
    ws = butterfly(v, (32, 16, 8, 4, 2, 1))
    if lane == 0:
        fx.ptr_store(ws, red + (slot * WAVES + wave))
    gpu.barrier()
    tot = fx.ptr_load(red + slot * WAVES)
    for w in range_constexpr(1, WAVES):
        tot = tot + fx.ptr_load(red + (slot * WAVES + w))
    return tot


@flyc.jit
def block_sum4(c, v0, v1, v2, v3):
    """Four CTA sums at once (one barrier): each wave's four sums to ``red`` slots 0
    .. 3, every thread adds the waves in wave order."""
    lane, wave, red = c["lane"], c["wave"], c["red"]
    ws = [butterfly(v, (32, 16, 8, 4, 2, 1)) for v in (v0, v1, v2, v3)]
    if lane == 0:
        for k in range_constexpr(4):
            fx.ptr_store(ws[k], red + (k * WAVES + wave))
    gpu.barrier()
    outs = []
    for k in range_constexpr(4):
        tot = fx.ptr_load(red + k * WAVES)
        for w in range_constexpr(1, WAVES):
            tot = tot + fx.ptr_load(red + (k * WAVES + w))
        outs.append(tot)
    return outs


def ld_bf(ptr, e, n, cm=0):
    """``n`` (2, 4, 8 or 16) bf16 at element ``e`` (even) of a bf16 buffer, as
    f32; at most 4 dwords a load."""
    out = []
    for e0 in range(0, n, 8):
        m = min(8, n - e0)
        w = bo.buffer_load(
            rsrc(ptr), (e + e0) // 2, vec_width=m // 2, dtype=T.i32, cache_modifier=cm
        )
        if m == 2:
            lo, hi = bf2_f32(fx.Int32(w))
            out += [lo, hi]
        else:
            v = fx.Vector(w).bitcast(fx.BFloat16).to(fx.Float32)
            out += [fx.Float32(v[j]) for j in range(m)]
    return out


def ld_f32(ptr, e, n, cm=0):
    """``n`` (2, 4, 8 or 16) f32 at element ``e`` of an f32 buffer; at most 4 a load."""
    out = []
    for e0 in range(0, n, 4):
        m = min(4, n - e0)
        w = bo.buffer_load(
            rsrc(ptr), e + e0, vec_width=m, dtype=T.f32, cache_modifier=cm
        )
        v = fx.Vector(w)
        out += [fx.Float32(v[j]) for j in range(m)]
    return out


def st_f32(ptr, e, vals, cm=0):
    """``vals`` (2, 4, 8 or 16 f32) at element ``e`` of an f32 buffer; at most 4 a store."""
    for e0 in range(0, len(vals), 4):
        chunk = vals[e0 : e0 + 4]
        bo.buffer_store(
            fx.Vector.from_elements(chunk, fx.Float32),
            rsrc(ptr),
            e + e0,
            cache_modifier=cm,
        )


def st_bf(ptr, e, vals, cm=0):
    """``vals`` (4 or 8 f32) as bf16 at element ``e`` of a bf16 buffer."""
    for e0 in range(0, len(vals), 4):
        bo.buffer_store(
            fx.Vector.from_elements(vals[e0 : e0 + 4], fx.Float32).to(fx.BFloat16),
            rsrc(ptr),
            e + e0,
            cache_modifier=cm,
        )


def st_words(ptr, d, words, cm=0):
    """``words`` (1 .. 8 i32) at dword ``d`` of a buffer; at most 4 a store."""
    for d0 in range(0, len(words), 4):
        chunk = words[d0 : d0 + 4]
        if len(chunk) == 1:
            bo.buffer_store(chunk[0], rsrc(ptr), d + d0, cache_modifier=cm)
        else:
            bo.buffer_store(
                fx.Vector.from_elements(chunk, fx.Int32),
                rsrc(ptr),
                d + d0,
                cache_modifier=cm,
            )


# ---------------------------------------------------------------- og tasks
# ---------------------------------------------------------------- o tasks
def o_w_loads(c, task):
    """The o_proj weight words of o task ``task``'s rows, this lane's: in flight
    (a task past the last reads past the buffer: zeros, no traffic). Row-major
    bf16, 16 B a lane a k-step, read once a layer (nontemporal)."""
    lane, wave = c["lane"], c["wave"]
    row = task * ROWS + lane % 16
    r_wo = rsrc(c["args"]["w_o"], HIDDEN * Q_K * 2)
    return [
        fx.Vector(
            bo.buffer_load(
                r_wo,
                (row * Q_K + wave * O_K_PER_WAVE + s * K_STEP + 8 * (lane // 16)) // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_NT,
            )
        )
        for s in range(O_STEPS)
    ]


@flyc.jit
def stage_og_rows(c, o_flag=None, QL=8):
    """Every token's normed gated o into LDS, computed here from o and the gate (16
    heads of 64: per-head RMSNorm without weight x sigmoid(gate), one bf16
    rounding; thread tid holds head (tid / 16) % 16, dims 4 (tid % 16) .. of
    token 2 round + tid / 256): the o tasks' B rows. It used to be an og task a
    token and a mailbox hop. ``o_flag`` (a merged launch): the requests' (KV head,
    row tile) flags first, then device-scope loads."""
    S, tid, lane, xs = c["S"], c["tid"], c["lane"], c["xs"]
    a = c["args"]
    o_cm = 0  # o from memory the host wrote: the default cache policy
    if const_expr(o_flag is not None):
        o_cm = CM_DEV  # the attention's device-scope stores, earlier in this launch
        rocdl.sched_barrier(0)
        RT = (QL * GQA + 15) // 16  # the attention's row tiles a (request, KV head)
        poll_flags(c, [(o_flag, (S // QL) * LOCAL_KV_HEADS * RT)])
        acquire_loads()
    TPR = THREADS // OG_THREADS  # tokens a round: 2
    e = ((tid % OG_THREADS) // 16) * HEAD_DIM + (lane % 16) * 4
    for t0 in range_constexpr(0, S, TPR):
        t = fx.min(t0 + tid // OG_THREADS, S - 1)
        ov = ld_bf(a["o"], t * Q_K + e, 4, o_cm)
        gv = ld_bf(a["g"], t * Q_K + e, 4)
        ss = fx.Float32(0.0)
        for j in range_constexpr(4):
            ss = ss + ov[j] * ov[j]
        ss = butterfly(ss, (8, 4, 2, 1))
        rstd = hw_rsq(ss / float(HEAD_DIM) + RMS_EPS)
        vals = [ov[j] * rstd * sigmoid(gv[j]) for j in range(4)]
        if t0 + tid // OG_THREADS < S:
            fx.ptr_store(bf16_pair(vals[0], vals[1]), xs + (t * (Q_K // 2) + e // 2))
            fx.ptr_store(
                bf16_pair(vals[2], vals[3]), xs + (t * (Q_K // 2) + e // 2 + 1)
            )
    gpu.barrier()


@flyc.jit
def stage_o(c, task, wts):
    """o task ``task``: rows ``16 task ..`` of o_proj for every token tile (lane l
    of wave w holds row l % 16 and k = 128 w + 32 s + 8 (l / 16) .. of token
    16 t + l % 16, from LDS), the waves' accumulators summed in wave order, one
    bf16 rounding, the partial rows pushed to every rank's ``attn`` region.
    ``wts``: the task's weights, in flight (``o_w_loads``)."""
    S, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    xs, red, misc, put_bf = c["xs"], c["red"], c["misc"], c["put_bf"]
    W, bases, rank, sy_attn = c["W"], c["bases"], c["rank"], c["sy_attn"]
    l16, g4 = lane % 16, lane // 16
    V4F = fx.Vector.make_type(4, fx.Float32)
    tiles = token_tiles(S)
    accs = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(tiles)]
    for s in range_constexpr(O_STEPS):
        k0 = wave * O_K_PER_WAVE + s * K_STEP + 8 * g4
        wa = wts[s].bitcast(fx.BFloat16)
        for t in range_constexpr(tiles):
            tok = fx.min(t * TILE + l16, S - 1)
            xb = fx.Vector(
                fx.ptr_load(xs + (tok * (Q_K // 2) + k0 // 2), result_type=V4F)
            )
            accs[t] = mfma_bf16(wa, xb.bitcast(fx.BFloat16), accs[t])
    # each tile's accumulators through the one reduction row set in turn (LDS)
    for t in range_constexpr(tiles):
        fx.ptr_store(accs[t], red + (wave * 64 + lane) * 4)
        gpu.barrier()
        if tid < ROWS * TILE:
            r = tid % ROWS
            tk = t * TILE + tid // ROWS
            if tk < S:
                fx.ptr_store(
                    bf16_round(row_sum(red, r, tid // ROWS)), misc + (tk * ROWS + r)
                )
        gpu.barrier()
    # pair i: token i / 8, rows 2 (i % 8), + 1; to every rank (itself too)
    n_pairs = (ROWS // 2) * S
    for rnd in range_constexpr(-(-n_pairs // THREADS)):
        i = tid + rnd * THREADS
        if i < n_pairs:
            tk = i // (ROWS // 2)
            rr = (i % (ROWS // 2)) * 2
            v0 = fx.ptr_load(misc + (tk * ROWS + rr))
            v1 = fx.ptr_load(misc + (tk * ROWS + rr + 1))
            for p in range_constexpr(W):
                put_bf(
                    preg(bases[p], sy_attn, "attn"),
                    (rank * MAX_TOKENS + tk) * HIDDEN + task * ROWS + rr,
                    [v0, v1],
                )
    gpu.barrier()  # ``misc`` is reused by the CTA's next task


# ---------------------------------------------------------------- boundary tasks
@flyc.jit
def stage_a_part(c, t, part):
    """Part ``part`` of token ``t``'s attention boundary (columns HIDDEN / A_PARTS
    part .., 2 PAIRS a thread): the ranks' partial o rows summed in rank order
    (``sum_partials``), one bf16 rounding; then, before any exchange, the pooler /
    router operands x = rstd2 (b1 + rstd b2) as its two halves b1 = bf16(pre_w alpha
    st), b2 = bf16(pre_w beta v) (plain bf16 words) and the part's four norm sums
    (``ss``, after the halves: the consumers' flag), so the GEMVs start without
    the sums' exchange, each consumer folding the scalars in itself. The stream's
    update is ``stage_a_part_res``, after the CTA's pooler job."""
    S, tid, W = c["S"], c["tid"], c["W"]
    put, poll = c["put"], c["poll"]
    own, ss_mb, xbuf, xbuf2 = c["own_attn"], c["ss"], c["xbuf"], c["xbuf2"]
    a = c["args"]
    A_PARTS = a_parts(S)
    PAIRS = A_PAIRS_MAX // A_PARTS
    NC = 2 * PAIRS  # columns a thread
    e0 = part * (HIDDEN // A_PARTS) + tid * NC
    # the operands first: after the poll each would be a dependent round trip
    prw = ld_bf(a["pre_w"], e0, NC)
    al = ld_f32(a["alpha"], e0, NC)
    be = ld_f32(a["beta"], e0, NC)
    st = ld_f32(a["res"], t * HIDDEN + e0, NC)
    rocdl.sched_barrier(0)
    # the all-reduce: this thread's PAIRS pairs from every rank in one poll round (two
    # adjacent pairs a spec), each pair summed in rank order, one bf16 rounding
    pb = t * (HIDDEN // 2) + e0 // 2
    PW = 2 if PAIRS % 2 == 0 else 1  # pairs a spec
    ws = poll(
        [
            (own, src * (MAX_TOKENS * HIDDEN // 2) + pb + PW * q, PW)
            for q in range(PAIRS // PW)
            for src in range(W)
        ],
        batch=max(8, (PAIRS // PW) * W),
    )
    stamp(c, 8)  # every rank's partial o rows arrived (thread 0's columns)
    v = []
    for j in range_constexpr(PAIRS):
        acc0, acc1 = bf2_f32(ws[(j // PW) * W][j % PW])
        for src in range_constexpr(1, W):
            lo, hi = bf2_f32(ws[(j // PW) * W + src][j % PW])
            acc0 = acc0 + lo
            acc1 = acc1 + hi
        v += [bf16_round(acc0), bf16_round(acc1)]
    # the operands' two halves: x = bf16(new rstd2 pre_w) with new = alpha st + beta
    # bf16(v rstd) is rstd2 (b1 + rstd b2) up to the roundings
    w1 = [bf16_pair(prw[2 * k] * al[2 * k] * st[2 * k], prw[2 * k + 1] * al[2 * k + 1] * st[2 * k + 1]).bitcast(fx.Int32) for k in range(PAIRS)]  # fmt: skip
    w2 = [bf16_pair(prw[2 * k] * be[2 * k] * v[2 * k], prw[2 * k + 1] * be[2 * k + 1] * v[2 * k + 1]).bitcast(fx.Int32) for k in range(PAIRS)]  # fmt: skip
    st_words(xbuf, (t * HIDDEN + e0) // 2, w1, CM_DEV)
    st_words(xbuf2, (t * HIDDEN + e0) // 2, w2, CM_DEV)
    # the two norms' statistics: the row's sum of squares of v and, for the pre-FFN
    # norm of new = alpha st + beta v rstd, the sums A = (alpha st)^2, B = alpha beta
    # st v, C = (beta v)^2 (its sum of squares is A + 2 rstd B + rstd^2 C up to the
    # bf16 rounding of v rstd: a 1e-5 relative difference)
    ssp, A_, B_, C_ = fx.Float32(0.0), fx.Float32(0.0), fx.Float32(0.0), fx.Float32(0.0)
    for k in range_constexpr(NC):
        ssp = ssp + v[k] * v[k]
        A_ = A_ + (al[k] * st[k]) * (al[k] * st[k])
        B_ = B_ + al[k] * be[k] * st[k] * v[k]
        C_ = C_ + (be[k] * v[k]) * (be[k] * v[k])
    ssp, A_, B_, C_ = block_sum4(c, ssp, A_, B_, C_)  # one barrier for the four
    release_stores()  # this wave's halves out ...
    gpu.barrier()  # ... every wave's, before the sums that flag them
    if tid < 4:
        put(
            ss_mb,
            (t * A_PARTS_MAX + part) * 4 + tid,
            (tid == 0).select(ssp, (tid == 1).select(A_, (tid == 2).select(B_, C_))),
        )


@flyc.jit
def stage_a_part_res(c, t, part):
    """The rest of boundary part ``part`` of token ``t``, after the CTA's pooler job
    (nothing reads it before K3's boundary): the ranks' partial o rows again (still in
    the mailbox, this layer's tag), the parts' sums of squares (``ss``), the post norm
    without weight, the stream updated in place (fp32), the next K1's first half
    (deferred K1), the (token, part) flag (``resflag``)."""
    S, tid, W = c["S"], c["tid"], c["W"]
    put, poll = c["put"], c["poll"]
    own, ss_mb, resflag = c["own_attn"], c["ss"], c["resflag"]
    a = c["args"]
    A_PARTS = a_parts(S)
    PAIRS = A_PAIRS_MAX // A_PARTS
    NC = 2 * PAIRS
    e0 = part * (HIDDEN // A_PARTS) + tid * NC
    al = ld_f32(a["alpha"], e0, NC)
    be = ld_f32(a["beta"], e0, NC)
    st = ld_f32(a["res"], t * HIDDEN + e0, NC)
    if const_expr(c["defer_k1"]):
        npw1 = ld_bf(a["npw"], e0, NC)
        al2 = ld_f32(a["alpha2"], e0, NC)
    pb = t * (HIDDEN // 2) + e0 // 2
    PW = 2 if PAIRS % 2 == 0 else 1
    ws = poll(
        [
            (own, src * (MAX_TOKENS * HIDDEN // 2) + pb + PW * q, PW)
            for q in range(PAIRS // PW)
            for src in range(W)
        ],
        batch=max(8, (PAIRS // PW) * W),
    )
    v = []
    for j in range_constexpr(PAIRS):
        acc0, acc1 = bf2_f32(ws[(j // PW) * W][j % PW])
        for src in range_constexpr(1, W):
            lo, hi = bf2_f32(ws[(j // PW) * W + src][j % PW])
            acc0 = acc0 + lo
            acc1 = acc1 + hi
        v += [bf16_round(acc0), bf16_round(acc1)]
    if const_expr(A_PARTS > 1):
        got = poll(
            [(ss_mb, (t * A_PARTS_MAX + q) * 4, 2) for q in range(A_PARTS)],
            batch=A_PARTS,
        )
        stamp(c, 9)  # the parts' norm sums arrived
        tot = fx.Float32(0.0)
        for q in range_constexpr(A_PARTS):
            tot = tot + got[q][0].bitcast(fx.Float32)
    else:
        tot = fx.Float32(0.0)
        for k in range_constexpr(NC):
            tot = tot + v[k] * v[k]
        tot = block_sum4(c, tot, tot, tot, tot)[0]
    rstd = hw_rsq(tot / float(HIDDEN) + POST_EPS)
    n = [bf16_round(v[k] * rstd) for k in range(NC)]
    new = [al[k] * st[k] + be[k] * n[k] for k in range(NC)]
    st_f32(a["res"], t * HIDDEN + e0, new, CM_DEV)
    if const_expr(c["defer_k1"]):
        c1w = [bf16_pair(npw1[2 * k] * al2[2 * k] * new[2 * k], npw1[2 * k + 1] * al2[2 * k + 1] * new[2 * k + 1]).bitcast(fx.Int32) for k in range(PAIRS)]  # fmt: skip
        st_words(c["c1"], (t * HIDDEN + e0) // 2, c1w, CM_DEV)
    release_stores()
    gpu.barrier()
    if tid == 0:
        put(
            resflag, t * A_PARTS + part, fx.Int32(1)
        )  # the stream's part (and c1) is out


# ---------------------------------------------------------------- router and pooler GEMVs
def pool_w_loads(c, rt0, part, nsub):
    """The pooler weights of the job (row tiles ``rt0`` .. + ``nsub``, K eighth
    ``part``) over this wave's P_STEPS k-steps, this lane's, in flight (16 VGPRs a
    row tile bf16, 8 FP8). Row-major, 8 values a lane a step (16 B bf16, 8 B of E4M3
    codes with ``c["fp8_pool"]``), read once a layer (nontemporal)."""
    lane, wave = c["lane"], c["wave"]
    l16, g4 = lane % 16, lane // 16
    r_w = rsrc(c["args"]["w_pool"])
    fp8 = c["fp8_pool"]
    out = []
    for sub in range(nsub):
        row = (rt0 + sub) * ROWS + l16
        if fp8:
            # the swizzled copy (``pack_fp8_rows_pooler``): a step pair's 16 codes of this
            # lane group sit together, one 16 B load a pair
            assert P_STEPS % 2 == 0
            steps = []
            for pr in range(P_STEPS // 2):
                v = fx.Vector(
                    bo.buffer_load(
                        r_w,
                        (
                            row * HIDDEN
                            + part * K_PER_WAVE
                            + wave * (P_STEPS * K_STEP)
                            + pr * (2 * K_STEP)
                            + 16 * g4
                        )
                        // 4,
                        vec_width=4,
                        dtype=T.i32,
                        cache_modifier=CM_NT,
                    )  # fmt: skip
                )
                steps.append(fx.Vector.from_elements([v[0], v[1]], fx.Int32))
                steps.append(fx.Vector.from_elements([v[2], v[3]], fx.Int32))
            out.append(steps)
            continue
        out.append(
            [
                fx.Vector(
                    bo.buffer_load(
                        r_w,
                        (
                            row * HIDDEN
                            + part * K_PER_WAVE
                            + (wave * P_STEPS + kc) * K_STEP
                            + 8 * g4
                        )
                        // 2,
                        vec_width=4,
                        dtype=T.i32,
                        cache_modifier=CM_NT,
                    )  # fmt: skip
                )
                for kc in range(P_STEPS)
            ]
        )
    return out


def pool_a(c, w):
    """A pooler weight step as the MFMA's bf16 A operand (FP8 codes decoded)."""
    if c["fp8_pool"]:
        return fp8_to_bf16x8(w[0], w[1])
    return w.bitcast(fx.BFloat16)


def pool_scale(c, row):
    """The pooler row's FP8 scale (fp32, after the codes), or 1."""
    if not c["fp8_pool"]:
        return fx.Float32(1.0)
    return fx.Float32(
        bo.buffer_load(
            rsrc(c["args"]["w_pool"]),
            (MOE_HIDDEN * HIDDEN) // 4 + row,
            vec_width=1,
            dtype=T.f32,
        )
    )


def router_w_loads(c, tile, part):
    """The split router's hi and lo rows of expert tile ``tile`` over this wave's
    steps of K eighth ``part``, this lane's: ([hi steps], [lo steps])."""
    lane, wave = c["lane"], c["wave"]
    l16, g4 = lane % 16, lane // 16
    row = tile * ROWS + l16
    a = c["args"]

    def loads(w):
        r_w = rsrc(w)
        return [
            fx.Vector(
                bo.buffer_load(
                    r_w,
                    (
                        row * HIDDEN
                        + part * K_PER_WAVE
                        + (wave * P_STEPS + kc) * K_STEP
                        + 8 * g4
                    )
                    // 2,
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=CM_NT,
                )  # fmt: skip
            )
            for kc in range(P_STEPS)
        ]

    return loads(a["w_hi"]), loads(a["w_lo"])


def x_loads(c, part, t0):
    """Token tile t0 .. t0 + 16's x over K eighth ``part``, this wave's steps: lane
    l16's token (clamped), 8 values a step, x = bf16(rstd2 (b1 + rstd b2)) from the
    boundary's two halves (device-scope: other CTAs wrote them) and the token's
    scalars (``wait_x``)."""
    S, lane, wave, misc = c["S"], c["lane"], c["wave"], c["misc"]
    l16, g4 = lane % 16, lane // 16
    tl = fx.min(t0 + l16, S - 1)
    r1, r2 = rsrc(c["xbuf"]), rsrc(c["xbuf2"])
    rstd = fx.ptr_load(misc + (SCAL0 + 2 * tl))
    rstd2 = fx.ptr_load(misc + (SCAL0 + 2 * tl + 1))
    out = []
    for kc in range_constexpr(P_STEPS):
        at = (
            tl * HIDDEN + part * K_PER_WAVE + (wave * P_STEPS + kc) * K_STEP + 8 * g4
        ) // 2
        h1 = (
            fx.Vector(
                bo.buffer_load(r1, at, vec_width=4, dtype=T.i32, cache_modifier=CM_DEV)
            )
            .bitcast(fx.BFloat16)
            .to(fx.Float32)
        )
        h2 = (
            fx.Vector(
                bo.buffer_load(r2, at, vec_width=4, dtype=T.i32, cache_modifier=CM_DEV)
            )
            .bitcast(fx.BFloat16)
            .to(fx.Float32)
        )
        words = [bf16_pair(rstd2 * (fx.Float32(h1[2 * j]) + rstd * fx.Float32(h2[2 * j])), rstd2 * (fx.Float32(h1[2 * j + 1]) + rstd * fx.Float32(h2[2 * j + 1]))).bitcast(fx.Int32) for j in range(4)]  # fmt: skip
        out.append(fx.Vector.from_elements(words, fx.Int32))
    return out


@flyc.jit
def stage_passes(c, rt0, part, pool_w, router_w, r_tile, is_router, nsub):
    """The pooler job (row tiles ``rt0`` .., ``part``) and, with ``router_w`` (a
    stream CTA), router task (``r_tile``, ``part``) over every token tile: x loaded
    straight into the B operands (16 B a lane a step, a token tile of 16), the
    waves' accumulators of each row tile summed through ``red`` (``red_sets`` row
    tiles a round) and published: pooled rows as bf16 pairs to ``pool``
    [part][token][row], logits as fp32 to ``logits`` [part][token][expert]."""
    S, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    put, put_bf, pool, logits = c["put"], c["put_bf"], c["pool"], c["logits"]
    RS = red_sets(S)
    NT_ = nsub + (
        1 if router_w is not None else 0
    )  # row tiles a pass: the job's and the router's
    for ps in range_constexpr(token_tiles(S)):
        t0 = ps * TILE
        nt = min(TILE, S - t0)
        xb = x_loads(c, part, t0)
        tiles = []
        for sub in range_constexpr(nsub):
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for kc in range_constexpr(P_STEPS):
                acc = mfma_bf16(
                    pool_a(c, pool_w[sub][kc]), xb[kc].bitcast(fx.BFloat16), acc
                )
            tiles.append(acc)
        if const_expr(router_w is not None):
            racc = fx.Vector.filled(4, 0.0, fx.Float32)
            for kc in range_constexpr(P_STEPS):
                racc = mfma_bf16(
                    router_w[0][kc].bitcast(fx.BFloat16),
                    xb[kc].bitcast(fx.BFloat16),
                    racc,
                )
                racc = mfma_bf16(
                    router_w[1][kc].bitcast(fx.BFloat16),
                    xb[kc].bitcast(fx.BFloat16),
                    racc,
                )
            tiles.append(racc)
        # the row sums over the waves (wave order), RS row tiles a round
        for r0 in range_constexpr(0, NT_, RS):
            n = min(RS, NT_ - r0)
            for j in range_constexpr(n):
                fx.ptr_store(
                    tiles[r0 + j], red + (j * RED_FLOATS + (wave * 64 + lane) * 4)
                )
            gpu.barrier()
            for j in range_constexpr(n):
                ti = r0 + j
                rb = red + j * RED_FLOATS
                if const_expr(ti < nsub):
                    # pooled rows: thread (token tk, row pair rp) of the tile, bf16 pairs
                    if tid < (ROWS // 2) * TILE:
                        tk = tid // (ROWS // 2)
                        rp = (tid % (ROWS // 2)) * 2
                        if tk < nt:
                            prow = (rt0 + ti) * ROWS + rp
                            if const_expr(c["pshard"]):
                                # the rank's rows of the pooled row: every rank's ``poolx``
                                grow = c["rank"] * (MOE_HIDDEN // c["W"]) + prow
                                vals = [row_sum(rb, rp, tk), row_sum(rb, rp + 1, tk)]
                                for pe in range_constexpr(c["W"]):
                                    put_bf(
                                        c["peer_poolx"][pe],
                                        (part * MAX_TOKENS + t0 + tk) * MOE_HIDDEN
                                        + grow,
                                        vals,
                                    )
                            else:
                                put_bf(
                                    pool,
                                    (part * MAX_TOKENS + t0 + tk) * MOE_HIDDEN + prow,
                                    [
                                        row_sum(rb, rp, tk) * pool_scale(c, prow),
                                        row_sum(rb, rp + 1, tk)
                                        * pool_scale(c, prow + 1),
                                    ],
                                )
                else:
                    # the router tile's partial logits: thread (token tk, expert row r)
                    if is_router & (tid < ROWS * TILE):
                        tk = tid // ROWS
                        r = tid % ROWS
                        if tk < nt:
                            put(
                                logits,
                                (part * MAX_TOKENS + t0 + tk) * EXPERTS
                                + r_tile * ROWS
                                + r,
                                row_sum(rb, r, tk),
                            )
            gpu.barrier()  # ``red`` is reused by the next round


SCAL0 = 0  # misc floats: a token's (rstd, rstd2) at 2 t
SUMS0 = 2 * MAX_TOKENS  # then the (token, part) norm sums, 4 each


@flyc.jit
def wait_x(c):
    """Every (token, part)'s four norm sums (stored after its x halves: their flag;
    one round, two words a thread), then each token's rstd (the post norm) and rstd2
    (the pre-FFN norm) into LDS (``misc``) for ``x_loads``, then device-scope loads."""
    S, tid, ss_mb, misc, poll = c["S"], c["tid"], c["ss"], c["misc"], c["poll"]
    A_PARTS = a_parts(S)
    NP = S * A_PARTS
    assert 2 * NP <= THREADS and SUMS0 + 4 * NP <= ROWS * MAX_TOKENS
    rocdl.sched_barrier(0)
    if tid < 2 * NP:
        tq = tid // 2
        got = poll(
            [
                (
                    ss_mb,
                    ((tq // A_PARTS) * A_PARTS_MAX + tq % A_PARTS) * 4 + 2 * (tid % 2),
                    2,
                )
            ]
        )
        fx.ptr_store(got[0][0].bitcast(fx.Float32), misc + (SUMS0 + 2 * tid))
        fx.ptr_store(got[0][1].bitcast(fx.Float32), misc + (SUMS0 + 2 * tid + 1))
    gpu.barrier()
    acquire_loads()
    if tid < S:
        tot, At, Bt, Ct = (
            fx.Float32(0.0),
            fx.Float32(0.0),
            fx.Float32(0.0),
            fx.Float32(0.0),
        )
        for q in range_constexpr(A_PARTS):
            b = misc + (SUMS0 + (tid * A_PARTS + q) * 4)
            tot = tot + fx.ptr_load(b)
            At = At + fx.ptr_load(b + 1)
            Bt = Bt + fx.ptr_load(b + 2)
            Ct = Ct + fx.ptr_load(b + 3)
        rstd = hw_rsq(tot / float(HIDDEN) + POST_EPS)
        tot2 = At + fx.Float32(2.0) * rstd * Bt + rstd * rstd * Ct
        fx.ptr_store(rstd, misc + (SCAL0 + 2 * tid))
        fx.ptr_store(
            hw_rsq(tot2 / float(HIDDEN) + RMS_EPS), misc + (SCAL0 + 2 * tid + 1)
        )
    gpu.barrier()


# ---------------------------------------------------------------- route and xn tasks
@flyc.jit
def stage_route(c, t, route_flag=None):
    """route task: token ``t``'s top-8 on wave 0 (lane l holds experts l, l + 64,
    ..): the parts' partial logits summed in part order, sigmoid(logit x
    ROUTER_MULT), + bias, a pick an integer max over ``route_key`` (a tie to the
    lower expert), the picks' sigmoids / (sum + ROUTE_EPS) x ROUTE_SCALE."""
    tid, lane, wave, sig = c["tid"], c["lane"], c["wave"], c["sig"]
    put, poll, logits = c["put"], c["poll"], c["logits"]
    a = c["args"]
    if wave == 0:
        got = poll(
            [
                (logits, (p * MAX_TOKENS + t) * EXPERTS + lane + 64 * h, 1)
                for p in range(R_PARTS)
                for h in range(4)
            ],
            batch=4 * R_PARTS,  # one round trip for the parts' logits
        )
        keys = []
        for h in range_constexpr(4):
            e = lane + 64 * h
            logit = got[h][0].bitcast(fx.Float32)
            for p in range_constexpr(1, R_PARTS):
                logit = logit + got[p * 4 + h][0].bitcast(fx.Float32)
            sc = sigmoid(logit * fx.Float32(ROUTER_MULT))
            fx.ptr_store(sc, sig + e)
            b = fx.Float32(bo.buffer_load(rsrc(a["bias"]), e, vec_width=1, dtype=T.f32))
            keys.append(route_key(sc + b, e))
        k0, k1, k2_, k3 = keys
        pid = fx.Int32(0)
        for p in range_constexpr(TOP_K):
            m = butterfly(fx.max(fx.max(k0, k1), fx.max(k2_, k3)), (32, 16, 8, 4, 2, 1), fx.max)  # fmt: skip
            k0 = (k0 == m).select(fx.Int32(-(2**31)), k0)
            k1 = (k1 == m).select(fx.Int32(-(2**31)), k1)
            k2_ = (k2_ == m).select(fx.Int32(-(2**31)), k2_)
            k3 = (k3 == m).select(fx.Int32(-(2**31)), k3)
            pid = (lane == p).select(255 - (m & 255), pid)
        if lane < TOP_K:
            w = fx.ptr_load(sig + pid)
            tot = butterfly(w, (1, 2, 4))
            bo.buffer_store(
                pid, rsrc(a["ids"]), t * TOP_K + lane, cache_modifier=CM_DEV
            )
            bo.buffer_store(
                (w / (tot + fx.Float32(ROUTE_EPS)) * fx.Float32(ROUTE_SCALE)).bitcast(
                    fx.Int32
                ),
                rsrc(a["scores"]),
                t * TOP_K + lane,
                cache_modifier=CM_DEV,
            )
    if const_expr(route_flag is not None):
        # the token's routing is out: its flag, for the experts' plan
        release_stores()
        gpu.barrier()
        if tid == 0:
            put(route_flag, t, fx.Int32(1))


@flyc.jit
def stage_xn(c, t, xin_flag=None):
    """xn task: token ``t``'s pre-expert norm of its pooled row (8 elements a
    thread, the K parts' partial rows summed), weight ``xn_w``, one bf16 rounding
    -> the expert input."""
    tid, poll, put, pool = c["tid"], c["poll"], c["put"], c["pool"]
    if const_expr(c["pshard"]):
        pool = c["own_poolx"]  # every rank's row slice, pushed here
    a = c["args"]
    e0 = tid * 8
    # the P_PARTS K-eighths' pooled rows (bf16 pairs), summed in part order, one round
    got = poll(
        [
            (pool, ((p * MAX_TOKENS + t) * MOE_HIDDEN + e0) // 2 + 2 * j, 2)
            for p in range(P_PARTS)
            for j in range(2)
        ],
        batch=2 * P_PARTS,
    )
    v = [fx.Float32(0.0) for _ in range(8)]
    for p in range_constexpr(P_PARTS):
        for j in range_constexpr(2):
            for q in range_constexpr(2):
                lo, hi = bf2_f32(got[p * 2 + j][q])
                v[4 * j + 2 * q] = v[4 * j + 2 * q] + lo
                v[4 * j + 2 * q + 1] = v[4 * j + 2 * q + 1] + hi
    ss = fx.Float32(0.0)
    for j in range_constexpr(8):
        ss = ss + v[j] * v[j]
    rstd = hw_rsq(block_sum(c, ss, 0) / float(MOE_HIDDEN) + RMS_EPS)
    xw = ld_bf(a["xn_w"], e0, 8)
    out = [v[j] * rstd * xw[j] for j in range(8)]
    st_bf(a["xin"], t * MOE_HIDDEN + e0, out, CM_DEV)
    if const_expr(c["xq_pre"]):
        # the same blocks K3 would quantize from the bf16 row (``ffn.stage_xq_cols``): a
        # thread's eight values one code word, four threads a 32-value block
        from .mxfp4 import e2m1_code, mxfp4_scale

        vals = [bf16_round(x) for x in out]
        amax = abs(vals[0])
        for k in range_constexpr(1, 8):
            amax = fx.max(amax, abs(vals[k]))
        amax = butterfly(amax, (1, 2), fx.max)
        byte, qs = mxfp4_scale(amax)
        word = e2m1_code(vals[0], qs)
        for k in range_constexpr(1, 8):
            word = word | (e2m1_code(vals[k], qs) << (4 * k))
        XQW_, XSW_ = MOE_HIDDEN // 8, MOE_HIDDEN // 32
        base = t * (XQW_ + XSW_)
        bo.buffer_store(word, rsrc(c["xinq"]), base + tid, cache_modifier=CM_DEV)
        if tid % 4 == 0:
            bo.buffer_store(
                byte, rsrc(c["xinq"]), base + XQW_ + tid // 4, cache_modifier=CM_DEV
            )
    if const_expr(xin_flag is not None):
        release_stores()
        gpu.barrier()
        if tid == 0:
            put(xin_flag, t, fx.Int32(1))


# ---------------------------------------------------------------- the CTAs' roles
@flyc.jit
def run_chain(c, route_flag=None, xin_flag=None, o_flag=None, QL=8):
    """A chain CTA (``bid >= CHAIN0``): its two o tasks (``bid`` and ``bid +
    BLOCKS``), its boundary part (the 64 boundary tasks sit here), then its pooler
    weights, the tokens' flags, the pooler passes, and token ``bid - CHAIN0``'s
    route and xn."""
    S, bid = c["S"], c["bid"]
    a = c["args"]
    A_PARTS = a_parts(S)
    tok = bid - CHAIN0
    # the o tasks and the boundary first, the pooler weights after: issued before
    # them, the weights (~15 us to land) delay the o tasks, and in the TP 8 server
    # every rank's boundary then waits for the slowest rank's o partials (measured
    # +1.1 ms a step)
    w_first = o_w_loads(c, bid)
    w_second = o_w_loads(c, bid + BLOCKS)
    stage_og_rows(c, o_flag, QL)
    stage_o(c, bid, w_first)
    stage_o(c, bid + BLOCKS, w_second)
    stamp(c, 2)  # the o tasks done
    if (bid >= AP0) & (bid < AP0 + A_PARTS * S):
        stage_a_part(c, (bid - AP0) // A_PARTS, (bid - AP0) % A_PARTS)
    stamp(c, 3)  # the boundary part done (boundary CTAs)
    # the small pooler job (P_NC row tiles past the stream CTAs' 24 P_NS), K eighth bid % 8
    part = bid % P_PARTS
    if const_expr(c["pshard"]):
        # the sharded pooler: a rank's 512 rows are 32 tiles a K part, one a CTA (24 stream, 8 chain)
        nsub = 1
        rt0 = 24 + (bid - CHAIN0) // P_PARTS
    else:
        nsub = P_NC
        rt0 = 24 * P_NS + P_NC * ((bid - CHAIN0) // P_PARTS)
    pool_w = pool_w_loads(c, rt0, part, nsub)
    rocdl.sched_barrier(0)
    wait_x(c)
    stamp(c, 4)  # every token's x arrived
    stage_passes(
        c, rt0, part, pool_w, None, fx.Int32(0), fx.Int32(0) == fx.Int32(1), nsub
    )
    stamp(c, 5)  # the pooler passes done
    if (bid >= AP0) & (bid < AP0 + A_PARTS * S):
        stage_a_part_res(c, (bid - AP0) // A_PARTS, (bid - AP0) % A_PARTS)
    stamp(c, 6)  # the boundary's stream update done (boundary CTAs)
    if tok < S:
        stage_route(c, tok, route_flag)  # the xn task runs on stream CTA tok meanwhile


@flyc.jit
def run_stream(c, o_flag=None, QL=8, xin_flag=None):
    """A stream CTA (``bid < CHAIN0``): its og task (CTAs 0 .. S), its two o tasks
    (``bid`` and ``bid + BLOCKS``), then its pooler weights, the tokens' flags, and
    the router / pooler passes (a router task's rows on CTAs R0 ..: every stream CTA
    loads a tile's rows, the index clamped; they stream while the first pass is
    staged)."""
    S, bid = c["S"], c["bid"]
    stamp(c, 1)  # entry (the og rows are computed below, no og task)
    w_first = o_w_loads(c, bid)
    w_second = o_w_loads(c, bid + BLOCKS)
    stage_og_rows(c, o_flag, QL)
    stage_o(c, bid, w_first)
    stage_o(c, bid + BLOCKS, w_second)
    stamp(c, 2)  # the o tasks done
    stamp(c, 3)  # (the boundary parts run on the chain CTAs)
    # the pooler job: P_NS row tiles of K eighth bid % 8 (the 24 stream CTAs of an eighth)
    part = bid % P_PARTS
    nsub = 1 if c["pshard"] else P_NS
    rt0 = (bid // P_PARTS) * nsub
    pool_w = pool_w_loads(c, rt0, part, nsub)
    rocdl.sched_barrier(0)
    wait_x(c)
    stamp(c, 4)  # every token's x arrived
    # a router task on CTAs R0 .. (every stream CTA loads a tile's rows, the index
    # clamped; they stream while the first token tile's x lands): the same K eighth
    r_tile = fx.min((bid - R0) // R_PARTS, fx.Int32(R_TILES - 1))
    router_w = router_w_loads(c, r_tile, part)
    stage_passes(
        c, rt0, part, pool_w, router_w, r_tile, (bid >= R0) & (bid < R0 + R_TASKS), nsub
    )
    stamp(c, 5)  # the router / pooler passes done
    stamp(c, 6)  # (the pooled rows go out with the passes)
    if bid < S:
        stage_xn(
            c, bid, xin_flag
        )  # token bid's expert input, beside the chain CTA's route


@flyc.jit
def run_k2(c, route_flag=None, xin_flag=None, o_flag=None, QL=8):
    """K2's stages on this CTA. ``route_flag`` / ``xin_flag`` (mailbox regions):
    in a launch that goes on to the experts, the route and xn tasks raise a
    token's flag once its routing / expert input is out. ``o_flag``: the
    attention ran earlier in this launch, a flag a (request of ``QL`` rows, KV
    head, row tile) once its o rows are out."""
    bid = c["bid"]
    stamp(c, 0)
    if bid >= CHAIN0:
        run_chain(c, route_flag, xin_flag, o_flag, QL)
    else:
        run_stream(c, o_flag, QL, xin_flag)
    stamp(c, 7)  # end
