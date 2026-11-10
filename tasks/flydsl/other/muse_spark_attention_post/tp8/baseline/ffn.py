# SPDX-License-Identifier: Apache-2.0
"""K3: one Muse Spark layer from the experts' input to the next layer's attention
input, for S = ``tokens`` rows on a rank of the TP 8 / EP 8 group. Per token and
each of its picks this rank holds, the compact-A4 chain of the ROCm model
(``mxfp4_compact_a4_moe`` + the FlyDSL finalize) on the Quark MXFP4 experts as
stored::

    x (bf16: K2's pre-expert-normed pooled input)
      -> per-32 MXFP4 (aiter's _mxfp4_quant_op rule) -> gate / up on the scaled
         E2M1 x E2M1 MFMA (fp32) -> bf16(gate sigmoid(gate) up) -> MXFP4 again
      -> down on the scaled MFMA -> y (bf16, a row a (token, slot))
      -> the finalize: v = bf16(y pen_w), out += score v / sqrt(mean v^2 + 1e-8)
         over the slots in order -> bf16 partial rows
      -> pushed to every rank, summed in rank order (the one-stage all-reduce) -> bf16
      -> the unpooler (rank r's rows r HIDDEN / 8 .., bf16 after the fp32 sum),
         gathered into every rank
      -> the FFN boundary: post_input = bf16(branch pfw), nb = bf16(post_input
         rstd) (eps 1e-8), stream = alpha stream + beta nb (fp32, in place), the
         next layer's pre-attention norm (eps 1e-5, weight npw) -> bf16; on the
         last layer the stream is rounded to bf16 before that norm, whose weight
         is then the final norm's

TP experts (``K3Build.tpx``, ATOM's TP-sharded MoE): every rank holds every
expert's 1/TP of the intermediate width (gate / up rows) and 1/TP of the down
projection's output columns (``weights.tp_expert_copies``), and works every pick
of the step: no rank waits on another's expert load (EP's slowest rank set the
layer's time). The per-(token, slot) post norm needs the slot's whole down row,
so the mids go to every rank (``midx``: ug -> dn), and each dn item pushes its
finalized values v = bf16(y pen_w) and their sum of squares to every rank
(``vx`` / ``ssx``): every rank then finalizes the whole row itself (no
all-reduce) and unpools its rows as before.
The experts' work is grouped by expert. Every CTA builds the step's plan in LDS
(wave 0: a ballot a local expert over rounds of 64 picks; a pick's column is its
rank among its expert's picks), then the picked experts' columns are cut into
column tiles of 16 (an MFMA B operand): an expert picked by every token of a
64-row step has four tiles. The ug and dn items are tile-major, (tile, 32-row
block), spread as contiguous ranges over the CTAs, so a CTA re-stages a tile's
quantized x (or mids) only when its range crosses a tile; the next item's weights
are issued before this item's MFMAs.

Stages (CTA map in ``k3_bases``; every wait points to an earlier stage, all CTAs
co-resident)::

    plan    every CTA, wave 0
    ug      n_tiles x 128 items of (gate pair, up pair): mids -> ``k3_mid`` blocks
    dn      n_tiles x 128 items: y rows -> ``k3_y``
    fin     S x AR_PARTS tasks on CTAs S ..: a part's columns of the token's slots,
            the finalize (slot sums of squares from the dn items' ``k3_ss``), the push to
            every rank's ``ffn`` region, the rank-ordered sum -> ``k3_fbuf`` + a flag
    unpool  the rank's 64 row tiles on the CTAs past the fin tasks (weights issued
            before the fin flags' wait): rows pushed to every rank's ``unp`` region
    bnd     S tasks on CTAs 0 ..: the whole row's boundary -> stream, attention input
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from . import buffer_ops as bo
from flydsl.expr import const_expr, gpu
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import T

from .mx import FP4, mfma_scaled
from .ops import (
    CM_DEV,
    ballot,
    bf2_f32,
    bf16_pair,
    bf16_round,
    butterfly,
    fp8_to_bf16x8,
    hw_exp2,
    hw_rsq,
    lane_gather,
    mfma_bf16,
    popcount,
    row_sum,
    rsrc,
    sum_partials,
    uniform,
)
from .plan import BLOCKS, THREADS, WAVES, cdiv, pair_layout
from .config import (
    EXPERTS,
    HIDDEN,
    INTER,
    LOCAL_EXPERTS,
    LOG2E,
    MAX_TOKENS,
    MOE_HIDDEN,
    MXFP4_GROUP,
    POST_EPS,
    RMS_EPS,
    TOP_K,
)
from .device import (
    global_atomic,
    poll_flags,
    release_stores,
    stamp,
)
from .layout import TRACE_LAYERS, TRACE_SLOTS
from .mxfp4 import (
    e2m1_code,
    lanes_before,
    mxfp4_scale,
)

POOL_PARTS = 8  # the pooler's K parts (post_attn.P_PARTS)

# ---------------------------------------------------------------- the plan
COLS = MAX_TOKENS  # picking columns an expert can have: a token picks it at most once
TILE_COLS = 16  # columns an expert item takes: the MFMA's N
TILES_PER_EXPERT = COLS // TILE_COLS  # 4
MAX_TILES = LOCAL_EXPERTS * TILES_PER_EXPERT  # 128
# ---------------------------------------------------------------- the experts
CHUNK = (
    128  # k of one scaled MFMA: a lane's 16 B of E2M1 (32 values) x the 4 lane groups
)
UG_CHUNKS = MOE_HIDDEN // CHUNK // WAVES  # 4 chunks a wave: the waves split K
DN_CHUNKS = INTER // CHUNK // WAVES  # 4
XS_LDS_MAX = 8  # tokens an unpool pass stages in LDS
XQW = MOE_HIDDEN // 8  # packed E2M1 words a staged column (eight codes a word)
XSW = MOE_HIDDEN // MXFP4_GROUP  # its E8M0 scales, an Int32 each
MID_W = 5  # mailbox pairs a mid block: four words of eight codes, the scale
UG_T = (
    INTER // 32
)  # ug items a column tile: 32 mid rows each (a gate pair + an up pair)
DN_T = MOE_HIDDEN // 32  # dn items a column tile: 32 y rows each (one tile pair)
MID_PAIRS = UG_T * MID_W  # a (token, slot)'s mid blocks
W13_ROW_BYTES = MOE_HIDDEN // 2  # a packed gate / up row of the Quark u8 layout
W13_SCALE_COLS = MOE_HIDDEN // MXFP4_GROUP
W2_ROW_BYTES = INTER // 2
W2_SCALE_COLS = INTER // MXFP4_GROUP
RED_SLOT_FLOATS = WAVES * 64 * 4  # one row set of MFMA accumulators
RED_SLOTS = 8  # two parities x (gate pair + up pair)
# ---------------------------------------------------------------- fin / unpool / bnd
AR_PARTS_MAX = 8
UNP_ROWS = 16  # unpooler rows an unpool task: the MFMA's M
UNP_TASKS = HIDDEN // UNP_ROWS  # 512 row tiles, 1 / npes of them a rank
UNP_STEPS = MOE_HIDDEN // WAVES // 32  # 16 bf16 MFMA steps a wave
BND_ELEMS = HIDDEN // THREADS  # 16 elements a thread of a token's row
assert XQW == THREADS and INTER == MOE_HIDDEN and UG_CHUNKS == 4 and DN_CHUNKS == 4
assert (
    TILE_COLS * (XQW + XSW) <= XS_LDS_MAX * MOE_HIDDEN // 2
)  # a staged tile fits ``xcol``
assert MOE_HIDDEN == 8 * THREADS and HIDDEN == 16 * THREADS and TOP_K == 8
assert LOCAL_EXPERTS <= 64 and MAX_TILES <= 256 and UNP_TASKS == 2 * BLOCKS
MAX_PICKS = (
    MAX_TOKENS * TOP_K
)  # 512: a TP-experts plan's picks (its sorted list) and at most its tiles


def tpx_dims(npes: int) -> dict:
    """A TP-experts build's per-rank slices: ug items a tile (32 intermediate rows of
    the rank's INTER / TP), dn items a tile (32 output columns of its MOE_HIDDEN /
    TP), the rows an expert's gate | up and down slices hold."""
    assert INTER % (32 * npes) == 0 and MOE_HIDDEN % (32 * npes) == 0, npes
    return {
        "ug_t": UG_T // npes,
        "dn_t": DN_T // npes,
        "up_row0": INTER // npes,
        "w13_rows": 2 * INTER // npes,
        "w2_rows": MOE_HIDDEN // npes,
    }


assert MAX_TOKENS % TILE_COLS == 0 and BND_ELEMS == 16


def tile_cols(tokens: int) -> int:
    """Columns a tile can hold at this width: a tile's rounds of x / mid staging."""
    return min(TILE_COLS, tokens)


def ar_parts(tokens: int) -> int:
    """fin / AR tasks a token (``MOE_HIDDEN / parts`` columns each): 8 ranks' partial
    rows are too many for one CTA's poll, and S x parts CTAs spread the pushes; fewer
    parts at the wide steps leave the unpool tasks their own CTAs."""
    return 8 if tokens <= 8 else (4 if tokens <= 32 else 2)


def fin_ppt(parts: int) -> int:
    """Column pairs a fin thread (1 or 2: a poll spec of at most two pairs)."""
    ppt = cdiv(MOE_HIDDEN // parts // 2, THREADS)
    assert ppt in (1, 2), ppt
    return ppt


def k3_bases(tokens: int, npes: int) -> dict:
    """K3's CTA map: the bnd tasks on CTAs 0 .. S, the fin tasks (``ar_parts`` a
    token) from ``fin0``, the unpool tasks from ``unp0`` with ``unp_stride`` CTAs
    taking ``ku`` tasks each (one at TP 8). Disjoint, so a fin CTA's polls wait
    behind no unpool weights and a bnd CTA carries nothing it waits on."""
    parts = ar_parts(tokens)
    fin0 = tokens
    unp0 = fin0 + tokens * parts
    stride = BLOCKS - unp0
    tasks = UNP_TASKS // npes
    passes = cdiv(tokens, XS_LDS_MAX)  # token passes of XS_LDS_MAX: a job each
    assert UNP_TASKS % npes == 0 and stride > 0, (tokens, npes)
    return {
        "ar_parts": parts,
        "fin0": fin0,
        "unp0": unp0,
        "unp_stride": stride,
        "unp_tasks": tasks,
        "unp_passes": passes,
        "unp_jobs": tasks
        * passes,  # (row tile, token pass) pairs, spread over the unpool CTAs
        "ku": cdiv(tasks * passes, stride),
    }


def k3_scratch_layout() -> dict:
    """K3's scratch regions, name -> (byte offset, bytes), sized for MAX_TOKENS: the
    mid blocks and y rows a (token, slot), a flag a (token, AR part), the dn items'
    sums of squares of the finalized values a (token, slot, item), and the
    all-reduce's result as plain bf16 words (``k3_fbuf``: 4 elements a "pair", read
    by LDS DMA, not polled)."""
    return pair_layout(
        [
            ("k3_mid", MAX_TOKENS * TOP_K * MID_PAIRS),
            ("k3_y", MAX_TOKENS * TOP_K * MOE_HIDDEN // 2),
            ("k3_fflag", MAX_TOKENS * AR_PARTS_MAX),
            ("k3_ss", MAX_TOKENS * TOP_K * DN_T),
            ("k3_fbuf", MAX_TOKENS * MOE_HIDDEN // 4),
            (
                "k3_nscal",
                MAX_TOKENS * 2,
            ),  # a token's (post rstd, pre rstd) for the next K1 (deferred)
            (
                "k3_ugq",
                2,
            ),  # the launch's ug queue: a word (zeroed at the merged launch's entry)
            ("k3_trace", TRACE_LAYERS * BLOCKS * TRACE_SLOTS),
        ],
        align=256,
    )


def k3_sym_layout(npes: int) -> dict:
    """A rank's peer-symmetric regions, name -> (byte offset, bytes): ``attn`` and
    ``ffn`` [source rank][token][width / 2] bf16 pairs, ``unp`` [token][HIDDEN / 2]
    (each rank's unpooled rows from that rank). Must equal post_attn.sym_layout
    (the same names, order and formula: attn = npes MAX_TOKENS HIDDEN / 2 pairs
    at 0, ffn = npes MAX_TOKENS MOE_HIDDEN / 2 after it, unp = MAX_TOKENS HIDDEN /
    2 after that); the two are to be unified in one place."""
    return pair_layout(
        [
            ("attn", npes * MAX_TOKENS * HIDDEN // 2),
            ("ffn", npes * MAX_TOKENS * MOE_HIDDEN // 2),
            ("unp", MAX_TOKENS * HIDDEN // 2),
            # TP experts: every rank's mid blocks a pick, the finalized values and the dn
            # items' sums of squares a pick (each written by the rank owning that part)
            ("midx", MAX_TOKENS * TOP_K * MID_PAIRS),
            ("vx", MAX_TOKENS * TOP_K * MOE_HIDDEN // 2),
            ("ssx", MAX_TOKENS * TOP_K * DN_T),
            # the sharded pooler: every rank's partial pooled rows of its row slice, [K part][token][row]
            ("poolx", POOL_PARTS * MAX_TOKENS * MOE_HIDDEN // 2),
        ],
        align=256,
    )


def k3_lds_floats() -> dict:
    """K3's LDS, name -> floats (ints for the plan), independent of the width."""
    return {
        "red": RED_SLOTS * RED_SLOT_FLOATS,
        "xcol": XS_LDS_MAX
        * MOE_HIDDEN
        // 2,  # a tile's MXFP4 columns; the unpool passes' rows
        "misc": max(
            UNP_ROWS * XS_LDS_MAX, 2 * WAVES * TILE_COLS
        ),  # an unpool pass's bf16 row sums; dn's wave partials
        "sums": 2 * WAVES,  # ``block_sum``'s wave sums
        # EP: expert e's column j: its pick (token TOP_K + slot); TP experts: the picks sorted
        # by expert (expert e's column j at start_of[e] + j)
        "pick_of": max(LOCAL_EXPERTS * COLS, MAX_PICKS),
        "cnt_of": EXPERTS,  # expert e's picks
        "start_of": EXPERTS,  # TP experts: expert e's first sorted pick
        "tile_of": max(
            MAX_TILES, MAX_PICKS
        ),  # column tile i: e COLS + its first column
        "nt": 4,
    }


def k3_lds_bytes(tokens: int) -> int:
    return 4 * sum(k3_lds_floats().values())


def k3_lds(tokens: int):
    """K3's LDS: the tiles' accumulators, a tile's staged columns (also the
    unpool passes' rows), the reductions, the step's plan."""
    f = k3_lds_floats()
    n_red, n_xcol, n_misc, n_sums, n_pick_of, n_cnt_of, n_start_of, n_tile_of, n_nt = (
        f[k]
        for k in (
            "red",
            "xcol",
            "misc",
            "sums",
            "pick_of",
            "cnt_of",
            "start_of",
            "tile_of",
            "nt",
        )
    )

    @fx.struct
    class Lds:
        red: fx.Array[fx.Float32, n_red, 16]
        xcol: fx.Array[fx.Float32, n_xcol, 16]
        misc: fx.Array[fx.Float32, n_misc, 16]
        sums: fx.Array[fx.Float32, n_sums, 16]
        pick_of: fx.Array[fx.Int32, n_pick_of, 16]
        cnt_of: fx.Array[fx.Int32, n_cnt_of, 16]
        start_of: fx.Array[fx.Int32, n_start_of, 16]
        tile_of: fx.Array[fx.Int32, n_tile_of, 16]
        nt: fx.Array[fx.Int32, n_nt, 16]

    return Lds


# ---------------------------------------------------------------- device helpers
def sigmoid(z):
    return fx.Float32(1.0) / (fx.Float32(1.0) + hw_exp2(z * fx.Float32(-LOG2E)))


def bor(x, y):
    return x | y


def lane_prefix(v, lane):
    """The wave's inclusive prefix sum of v, lanes ascending."""
    incl = v
    for d in (1, 2, 4, 8, 16, 32):
        moved = lane_gather(incl, fx.max(lane - d, 0))
        incl = incl + (lane >= d).select(moved, fx.Int32(0))
    return incl


@flyc.jit
def block_sum(c, v, slot):
    """The CTA's sum of ``v`` through ``sums`` row ``slot`` (one barrier)."""
    lane, wave, sums = c["lane"], c["wave"], c["sums"]
    ws = butterfly(v, (32, 16, 8, 4, 2, 1))
    if lane == 0:
        fx.ptr_store(ws, sums + (slot * WAVES + wave))
    gpu.barrier()
    tot = fx.ptr_load(sums + slot * WAVES)
    for w in range_constexpr(1, WAVES):
        tot = tot + fx.ptr_load(sums + (slot * WAVES + w))
    return tot


def pick(c, t, slot):
    """Token t's pick in ``slot`` as a local expert, or -1 (another rank's); TP experts:
    the expert (every rank works every pick)."""
    a = c["args"]
    e = fx.Int32(
        bo.buffer_load(
            rsrc(a["ids"]),
            t * TOP_K + slot,
            vec_width=1,
            dtype=T.i32,
            cache_modifier=CM_DEV,
        )
    )
    if const_expr(c["tpx"]):
        return fx.max(fx.min(e, fx.Int32(EXPERTS - 1)), fx.Int32(0))
    e = e - a["local_start"]
    return ((e >= 0) & (e < LOCAL_EXPERTS)).select(e, fx.Int32(-1))


def pick_at(c, e, col):
    """Expert e's column ``col``'s pick: token TOP_K + slot."""
    if const_expr(c["tpx"]):
        return fx.ptr_load(c["pick_of"] + (fx.ptr_load(c["start_of"] + e) + col))
    return fx.ptr_load(c["pick_of"] + (e * COLS + col))


def tile_info(c, ti, n_tiles):
    """Column tile ``ti``'s (local expert, first column, columns), clamped to valid
    values past the plan (a CTA with no item still issues loads: expert 0's)."""
    tc = fx.max(fx.min(ti, n_tiles - 1), fx.Int32(0))
    packed = uniform(fx.ptr_load(c["tile_of"] + tc))
    packed = fx.max(
        fx.min(packed, fx.Int32((EXPERTS if c["tpx"] else LOCAL_EXPERTS) * COLS - 1)),
        fx.Int32(0),
    )
    e = packed // COLS
    col0 = packed % COLS
    cnt = uniform(fx.ptr_load(c["cnt_of"] + e))
    n = fx.max(fx.min(cnt - col0, fx.Int32(TILE_COLS)), fx.Int32(1))
    return e, col0, n


def w13_rsrc(c, e):
    """Expert e's packed gate | up rows and their scales (the Quark layout as stored;
    TP experts: the rank's slice, gate rows then up rows, ``w13_rows`` an expert)."""
    a = c["args"]
    rows = c["w13_rows"]
    return (
        rsrc(a["w13"] + fx.Int64(e) * (rows * W13_ROW_BYTES)),
        rsrc(a["s13"] + fx.Int64(e) * (rows * W13_SCALE_COLS)),
    )


def w2_rsrc(c, e):
    """Expert e's down rows (TP experts: the rank's ``w2_rows`` output rows)."""
    a = c["args"]
    rows = c["w2_rows"]
    return (
        rsrc(a["w2"] + fx.Int64(e) * (rows * W2_ROW_BYTES)),
        rsrc(a["s2"] + fx.Int64(e) * (rows * W2_SCALE_COLS)),
    )


def fp4_tile_loads(c, r_w, r_s, row, chunks):
    """This lane's loads of one 16-row tile (``row``: its row) over the wave's
    ``chunks`` k chunks: (weight words, scale bytes) in flight. A lane's 16 B are
    32 E2M1 of the row at byte row K / 2 + chunk 64 + lane group 16; the wave's
    chunks' 16 scale bytes of the row are one load (byte 4 kc + g4: chunk kc's,
    lane group g4's)."""
    lane, wave = c["lane"], c["wave"]
    g4 = lane // 16
    assert chunks == 4
    wts = []
    for kc in range(chunks):
        k_chunk = wave * chunks + kc
        wts.append(
            fx.Vector(
                bo.buffer_load(
                    r_w,
                    (
                        row * (chunks * WAVES * CHUNK // 2)
                        + k_chunk * (CHUNK // 2)
                        + g4 * 16
                    )
                    // 4,
                    vec_width=4,
                    dtype=T.i32,
                )
            )
        )
    sc16 = fx.Vector(
        bo.buffer_load(
            r_s,
            (row * (chunks * WAVES * CHUNK // 32) + wave * 16) // 4,
            vec_width=4,
            dtype=T.i32,
        )
    )
    scs = [(fx.Int32(sc16[kc]) >> (8 * g4)) & 0xFF for kc in range(chunks)]
    return wts, scs


def tile_mfmas_a4(wts, scs, xb):
    """A tile's fp32 accumulator over the wave's chunks: the scaled E2M1 x E2M1
    MFMA on the weight words against the staged column blocks ``xb[kc]`` =
    (words, scale)."""
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for kc in range(len(wts)):
        acc = mfma_scaled(
            wts[kc],
            xb[kc][0],
            acc,
            fx.Int32(scs[kc]),
            fx.Int32(xb[kc][1]),
            a_fmt=FP4,
            b_fmt=FP4,
        )
    return acc


def tile_store(c, slot, acc):
    fx.ptr_store(acc, c["red"] + ((slot * WAVES + c["wave"]) * 64 + c["lane"]) * 4)


def xq_loads(c, e, col0, n):
    """The bf16 x of expert e's columns col0 .. col0 + n, every round's load at
    once (in flight): round i is column i, eight values a thread. A column past n
    reads the last one's (its store is skipped); the token is clamped to the
    step's rows, as a rank no pick reached stages its (unused) first tile from a
    plan it never wrote."""
    S, tid = c["S"], c["tid"]
    r_xin = rsrc(c["args"]["xin"])

    def x_unit(i):
        v = tid + THREADS * i  # eight values: column v / XQW, values 8 (v % XQW) ..
        tk = pick_at(c, e, col0 + fx.min(v // XQW, n - 1)) // TOP_K
        tk = fx.max(fx.min(tk, fx.Int32(S - 1)), fx.Int32(0))
        return fx.Vector(
            bo.buffer_load(
                r_xin,
                tk * (MOE_HIDDEN // 2) + (v % XQW) * 4,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        )

    return [x_unit(i) for i in range(tile_cols(S))]


def xq_all_loads(c):
    """Every token's MXFP4 blocks (K2's ``xinq``: XQW code words then XSW scale words
    a token), 16 B a thread a round, in flight (the staged-column layout with the
    column = the token; S <= TILE_COLS)."""
    S, tid = c["S"], c["tid"]
    W_TOK = XQW + XSW
    r = rsrc(c["args"]["xinq"])
    n4 = S * W_TOK // 4
    return [bo.buffer_load(r, 4 * fx.min(tid + THREADS * i, n4 - 1), vec_width=4, dtype=T.i32, cache_modifier=CM_DEV) for i in range(cdiv(n4, THREADS))]  # fmt: skip


@flyc.jit
def stage_xq_all(c, xv):
    """The blocks (``xq_all_loads``) into LDS: token t's words at xcol[t XQW ..], its
    scales at xcol[TILE_COLS XQW + t XSW ..]."""
    S, tid, xcol = c["S"], c["tid"], c["xcol"]
    W_TOK = XQW + XSW
    n4 = S * W_TOK // 4
    for i in range_constexpr(len(xv)):
        v = tid + THREADS * i
        if v < n4:
            w4 = fx.Vector(xv[i])
            for j in range_constexpr(4):
                d = 4 * v + j
                tk, o = d // W_TOK, d % W_TOK
                at = (o < XQW).select(
                    tk * XQW + o, TILE_COLS * XQW + tk * XSW + (o - XQW)
                )
                fx.ptr_store(fx.Int32(w4[j]).bitcast(fx.Float32), xcol + at)
    gpu.barrier()


def xq_b_tok(c, tok, chunks, kc):
    """This lane's B operand of chunk ``kc`` from token ``tok``'s staged blocks
    (``stage_xq_all``): its 32 codes of k block g4 and their scale."""
    lane, wave, xcol = c["lane"], c["wave"], c["xcol"]
    V4F = fx.Vector.make_type(4, fx.Float32)
    kb = (wave * chunks + kc) * (CHUNK // MXFP4_GROUP) + lane // 16
    q = fx.Vector(fx.ptr_load(xcol + (tok * XQW + kb * 4), result_type=V4F)).bitcast(
        fx.Int32
    )
    sc = fx.ptr_load(xcol + (TILE_COLS * XQW + tok * XSW + kb)).bitcast(fx.Int32)
    return q, sc


def lane_token(c, e, col0, n):
    """The token of this lane's column (lane % 16) of a tile (the last column's past n)."""
    tk = pick_at(c, e, col0 + fx.min(c["lane"] % 16, n - 1)) // TOP_K
    return fx.max(fx.min(tk, fx.Int32(c["S"] - 1)), fx.Int32(0))


def xq_b(c, n, chunks, kc):
    """This lane's B operand of chunk ``kc``: its column's 32 codes of k block g4
    and their scale, from the staged blocks (a column past n: the last)."""
    lane, wave, xcol = c["lane"], c["wave"], c["xcol"]
    V4F = fx.Vector.make_type(4, fx.Float32)
    col = fx.min(lane % 16, n - 1)
    kb = (wave * chunks + kc) * (CHUNK // MXFP4_GROUP) + lane // 16
    q = fx.Vector(fx.ptr_load(xcol + (col * XQW + kb * 4), result_type=V4F)).bitcast(
        fx.Int32
    )
    sc = fx.ptr_load(xcol + (TILE_COLS * XQW + col * XSW + kb)).bitcast(fx.Int32)
    return q, sc


def ug_pair_loads(c, q, pr, n_tiles):
    """ug item q's pair ``pr`` (0: gate rows, 1: up rows) of two 16-row tiles."""
    l16 = c["lane"] % 16
    e, _, _ = tile_info(c, q // c["ug_t"], n_tiles)
    r_w, r_s = w13_rsrc(c, e)
    return [
        [
            list(part)
            for part in fp4_tile_loads(
                c,
                r_w,
                r_s,
                pr * c["up_row0"] + (q % c["ug_t"]) * 32 + t * 16 + l16,
                UG_CHUNKS,
            )
        ]
        for t in range(2)
    ]


def ug_or_dn_pair_loads(c, q, pr, dq, n_tiles, use_dn):
    """ug item ``q``'s pair ``pr``, or (``use_dn``) dn item ``dq``'s tile pair: the
    same load shape (gate / up and down rows are both MOE_HIDDEN = INTER wide), the
    buffer and rows selected, so the ug loop's last prefetch fetches the dn stage's
    first weights."""
    assert (
        MOE_HIDDEN == INTER
        and W13_ROW_BYTES == W2_ROW_BYTES
        and W13_SCALE_COLS == W2_SCALE_COLS
    )
    l16 = c["lane"] % 16
    a = c["args"]
    e_u, _, _ = tile_info(c, q // c["ug_t"], n_tiles)
    e_d, _, _ = tile_info(c, dq // c["dn_t"], n_tiles)
    UGT, DNT, R13, R2, UP0 = (
        c["ug_t"],
        c["dn_t"],
        c["w13_rows"],
        c["w2_rows"],
        c["up_row0"],
    )
    w_addr = use_dn.select(a["w2"] + fx.Int64(e_d) * (R2 * W2_ROW_BYTES), a["w13"] + fx.Int64(e_u) * (R13 * W13_ROW_BYTES))  # fmt: skip
    s_addr = use_dn.select(a["s2"] + fx.Int64(e_d) * (R2 * W2_SCALE_COLS), a["s13"] + fx.Int64(e_u) * (R13 * W13_SCALE_COLS))  # fmt: skip
    r_w, r_s = rsrc(w_addr), rsrc(s_addr)
    return [
        [
            list(part)
            for part in fp4_tile_loads(
                c,
                r_w,
                r_s,
                use_dn.select(
                    (dq % DNT) * 32 + t * 16 + l16,
                    pr * UP0 + (q % UGT) * 32 + t * 16 + l16,
                ),
                UG_CHUNKS,
            )
        ]  # fmt: skip
        for t in range(2)
    ]


def dn_pair_loads(c, q, n_tiles):
    """dn item q's tile pair: 32 rows of the expert's down weight."""
    l16 = c["lane"] % 16
    e, _, _ = tile_info(c, q // c["dn_t"], n_tiles)
    r_w, r_s = w2_rsrc(c, e)
    return [
        [
            list(part)
            for part in fp4_tile_loads(
                c, r_w, r_s, (q % c["dn_t"]) * 32 + t * 16 + l16, DN_CHUNKS
            )
        ]
        for t in range(2)
    ]


def unp_gtask(c, k):
    """CTA ``bid``'s k-th unpool job: (its index among the rank's jobs, its row tile of
    the unpooler, its token pass; outside the rank's jobs: the tile past the buffer, no
    traffic). A job is one row tile over one pass of XS_LDS_MAX tokens, so a wide
    step's passes spread over the unpool CTAs instead of queuing on a tile's."""
    b = c["bases"]
    job = c["bid"] - b["unp0"] + b["unp_stride"] * k
    live = (job >= 0) & (job < b["unp_jobs"])
    tile = job // b["unp_passes"]
    ps = job % b["unp_passes"]
    return job, live.select(c["rank"] * b["unp_tasks"] + tile, fx.Int32(UNP_TASKS)), ps


def unp_w_loads(c, gtask):
    """A lane's loads of unpooler row tile ``gtask`` (16 rows x MOE_HIDDEN, row-major
    bf16, or E4M3 codes with ``c["fp8_unpool"]``): row l16, this wave's K eighth, 8
    values a step; 64 VGPRs (32 FP8), in flight."""
    lane, wave = c["lane"], c["wave"]
    l16, g4 = lane % 16, lane // 16
    fp8 = c["fp8_unpool"]
    # a tile past the last reads past the buffer: zeros, no traffic
    r_u = rsrc(c["args"]["w_unpool"], HIDDEN * MOE_HIDDEN * (1 if fp8 else 2))
    row = gtask * UNP_ROWS + l16
    return [
        fx.Vector(
            bo.buffer_load(
                r_u,
                (row * MOE_HIDDEN + wave * (MOE_HIDDEN // WAVES) + kc * 32 + 8 * g4)
                // (4 if fp8 else 2),
                vec_width=2 if fp8 else 4,
                dtype=T.i32,
            )
        )
        for kc in range(UNP_STEPS)
    ]


def unp_a(c, w):
    """An unpooler weight step as the MFMA's bf16 A operand (FP8 codes decoded)."""
    if c["fp8_unpool"]:
        return fp8_to_bf16x8(w[0], w[1])
    return w.bitcast(fx.BFloat16)


def unp_scale(c, row):
    """The unpooler row's FP8 scale (fp32, after the codes), or 1."""
    if not c["fp8_unpool"]:
        return fx.Float32(1.0)
    return fx.Float32(
        bo.buffer_load(
            rsrc(c["args"]["w_unpool"]),
            (HIDDEN * MOE_HIDDEN) // 4 + row,
            vec_width=1,
            dtype=T.f32,
        )
    )


# ---------------------------------------------------------------- 0. the plan
@flyc.jit
def stage_plan(c):
    """The step's plan in LDS, the same on every CTA: wave w owns local experts
    4 w .. 4 w + 4 and walks every pick (token p / TOP_K, slot p % TOP_K) in rounds
    of 64; a pick's column is its rank among its expert's picks in pick order (a
    ballot an expert a round, the count a uniform scalar); then wave 0 lists the
    experts' column tiles of 16 in expert order. Returns the tiles. (One wave over
    all 32 experts took 11 us at 64 rows; LDS atomics broke the multi-tile experts.)"""
    S, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    pick_of, cnt_of, tile_of, nt = c["pick_of"], c["cnt_of"], c["tile_of"], c["nt"]
    a = c["args"]
    local_start = a["local_start"]
    EPW = LOCAL_EXPERTS // WAVES  # experts a wave: 4
    r_ids = rsrc(a["ids"])
    PICKS = S * TOP_K
    ROUNDS = cdiv(PICKS, 64)
    # every round's ids, in flight at once (every wave loads them: 8 x 64 ids a round)
    p_ids = [
        fx.Int32(
            bo.buffer_load(
                r_ids,
                fx.min(rr * 64 + lane, PICKS - 1),
                vec_width=1,
                dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        )  # fmt: skip
        for rr in range(ROUNDS)
    ]
    cnts = [
        fx.Int32(0) for _ in range(EPW)
    ]  # this wave's experts' picks so far (uniform)
    for rr in range_constexpr(ROUNDS):
        p = rr * 64 + lane
        le = p_ids[rr] - local_start
        ok = (p < PICKS) & (le >= 0) & (le < LOCAL_EXPERTS)
        col = fx.Int32(0)
        mine = fx.Int32(0) == fx.Int32(1)
        for k in range_constexpr(EPW):
            hit = ok & (le == wave * EPW + k)
            m = ballot(hit)
            col = hit.select(cnts[k] + lanes_before(m, lane), col)
            cnts[k] = cnts[k] + uniform(popcount(m))
            mine = mine | hit
        if mine & (col < COLS):
            fx.ptr_store(p, pick_of + (le * COLS + col))
    for k in range_constexpr(EPW):
        if lane == 0:
            fx.ptr_store(cnts[k], cnt_of + (wave * EPW + k))
    gpu.barrier()
    if wave == 0:
        # lane e < LOCAL_EXPERTS: expert e's count, its tiles and their first index
        mine0 = lane < LOCAL_EXPERTS
        cnt = mine0.select(
            fx.ptr_load(cnt_of + fx.min(lane, LOCAL_EXPERTS - 1)), fx.Int32(0)
        )
        tiles = mine0.select(
            (fx.min(cnt, COLS) + TILE_COLS - 1) // TILE_COLS, fx.Int32(0)
        )
        incl = lane_prefix(tiles, lane)
        tile0 = incl - tiles
        n_tiles = lane_gather(incl, 63)
        if lane == 0:
            # an empty plan's first tile is expert 0, column 0: a CTA with no item
            # still issues its first item's loads, from the list's head, not the
            # LDS's stale word (a later store of a real tile 0 wins)
            fx.ptr_store(fx.Int32(0), tile_of)
            fx.ptr_store(n_tiles, nt)
        for j in range_constexpr(TILES_PER_EXPERT):
            if j < tiles:
                fx.ptr_store(lane * COLS + j * TILE_COLS, tile_of + (tile0 + j))
    gpu.barrier()
    return uniform(fx.ptr_load(nt))


@flyc.jit
def stage_plan_tpx(c):
    """TP experts' plan (every rank alike): wave 0 walks the picks in rounds of 64, a
    pick's column its rank among its expert's picks in pick order (the round's
    same-expert lanes from 8 ballots, the earlier rounds' count in LDS), then lays out
    the experts' starts in the sorted pick list (``pick_of``) and their column tiles
    of 16 in expert order. Returns the tiles."""
    S, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    sorted_, cnt_of, start_of, tile_of, nt = (
        c["pick_of"],
        c["cnt_of"],
        c["start_of"],
        c["tile_of"],
        c["nt"],
    )
    a = c["args"]
    r_ids = rsrc(a["ids"])
    PICKS = S * TOP_K
    ROUNDS = cdiv(PICKS, 64)
    EPL = EXPERTS // 64  # experts a lane: 4
    if tid < EXPERTS:
        fx.ptr_store(fx.Int32(0), cnt_of + tid)
    gpu.barrier()
    if wave == 0:
        ids = [
            fx.max(
                fx.min(
                    fx.Int32(
                        bo.buffer_load(
                            r_ids,
                            fx.min(rr * 64 + lane, PICKS - 1),
                            vec_width=1,
                            dtype=T.i32,
                            cache_modifier=CM_DEV,
                        )
                    ),
                    fx.Int32(EXPERTS - 1),
                ),
                fx.Int32(0),
            )  # fmt: skip
            for rr in range(ROUNDS)
        ]
        cols = []
        for rr in range_constexpr(ROUNDS):
            # a lane's same-expert lanes of the round by the expert id's 8 bits (a
            # ballot a bit: a fixed cost, where a turn a distinct expert took 6.8 us
            # at 64 picks); its column = the expert's count of the earlier rounds +
            # its same-expert lanes below it (pick order, alike on every rank); the
            # round's lowest lane of an expert moves the expert's count
            e = ids[rr]
            valid = rr * 64 + lane < PICKS
            m = ballot(valid)
            for bit in range_constexpr(8):
                on = (e >> bit) & 1
                m = m & (
                    ballot(on == 1) ^ (fx.Int64(on) - fx.Int64(1))
                )  # its own bit's lanes
            below = lanes_before(m, lane)
            base = fx.ptr_load(cnt_of + e)
            cols.append(base + below)
            if valid & (below == 0):
                fx.ptr_store(base + popcount(m), cnt_of + e)
        # the experts' starts and tiles: EPL experts a lane, an exclusive prefix over the lanes
        cn = [fx.ptr_load(cnt_of + (lane * EPL + k)) for k in range(EPL)]
        tl = [
            (fx.min(cn[k], fx.Int32(COLS)) + TILE_COLS - 1) // TILE_COLS
            for k in range(EPL)
        ]
        tot, tsum = cn[0], tl[0]
        for k in range_constexpr(1, EPL):
            tot = tot + cn[k]
            tsum = tsum + tl[k]
        incl = lane_prefix(tot, lane)
        tincl = lane_prefix(tsum, lane)
        n_tiles = lane_gather(tincl, 63)
        if lane == 0:
            # an empty plan's first tile is expert 0, column 0 (a later real tile 0 wins)
            fx.ptr_store(fx.Int32(0), tile_of)
            fx.ptr_store(n_tiles, nt)
        run, tile_run = incl - tot, tincl - tsum
        for k in range_constexpr(EPL):
            ek = lane * EPL + k
            fx.ptr_store(run, start_of + ek)
            for j in range_constexpr(TILES_PER_EXPERT):
                if j < tl[k]:
                    fx.ptr_store(ek * COLS + j * TILE_COLS, tile_of + (tile_run + j))
            run = run + cn[k]
            tile_run = tile_run + tl[k]
        # the sorted picks (a wave's LDS operations run in order: the starts are in)
        for rr in range_constexpr(ROUNDS):
            p = rr * 64 + lane
            if p < PICKS:
                fx.ptr_store(p, sorted_ + (fx.ptr_load(start_of + ids[rr]) + cols[rr]))
    gpu.barrier()
    return uniform(fx.ptr_load(nt))


# ---------------------------------------------------------------- 1 + 2. the experts
@flyc.jit
def stage_xq_cols(c, e, col0, n, xv=None):
    """Expert e's columns col0 .. col0 + n as MXFP4 blocks into LDS: column j's
    words at xcol[j XQW ..], its scales at xcol[TILE_COLS XQW + j XSW ..]; a
    32-value block is four lanes' (``xv``: the loads, already in flight)."""
    S, tid, xcol = c["S"], c["tid"], c["xcol"]
    xq_l = xcol
    xs_l = xcol + TILE_COLS * XQW
    if const_expr(xv is None):
        xv = xq_loads(c, e, col0, n)
    for i in range_constexpr(tile_cols(S)):
        v = tid + THREADS * i
        vb = xv[i].bitcast(fx.BFloat16).to(fx.Float32)
        vals = [fx.Float32(vb[k]) for k in range(8)]
        amax = abs(vals[0])
        for k in range_constexpr(1, 8):
            amax = fx.max(amax, abs(vals[k]))
        amax = butterfly(amax, (1, 2), fx.max)
        byte, qs = mxfp4_scale(amax)
        word = e2m1_code(vals[0], qs)
        for k in range_constexpr(1, 8):
            word = word | (e2m1_code(vals[k], qs) << (4 * k))
        if v < n * XQW:
            fx.ptr_store(word.bitcast(fx.Float32), xq_l + v)
            if (v % 4) == 0:
                fx.ptr_store(byte.bitcast(fx.Float32), xs_l + v // 4)
    gpu.barrier()


@flyc.jit
def stage_mid_q(c, e, col0, n):
    """Expert e's columns col0 .. col0 + n's mid blocks (MXFP4, from the ug items)
    into LDS in ``stage_xq_cols``' layout; every round polled in one spin (a
    column past n polls the last one's, its store skipped)."""
    S, tid, xcol, poll, mid = c["S"], c["tid"], c["xcol"], c["poll"], c["mid"]
    if const_expr(c["tpx"]):
        mid = c[
            "own_midx"
        ]  # every rank's blocks of the pick, pushed here (UG_T in all)
    xq_l = xcol
    xs_l = xcol + TILE_COLS * XQW
    XR = cdiv(tile_cols(S) * UG_T, THREADS)

    def specs(i):
        v = tid + THREADS * i  # block v % UG_T of column v / UG_T
        j = fx.min(v // UG_T, n - 1)
        base = (pick_at(c, e, col0 + j) * UG_T + v % UG_T) * MID_W
        return [(mid, base, 2), (mid, base + 2, 2), (mid, base + 4, 1)]

    got = poll([sp for i in range(XR) for sp in specs(i)])
    for i in range_constexpr(XR):
        v = tid + THREADS * i
        if v < n * UG_T:
            words = [got[3 * i][0], got[3 * i][1], got[3 * i + 1][0], got[3 * i + 1][1]]
            fx.ptr_store(
                fx.Vector.from_elements(words, fx.Int32).bitcast(fx.Float32),
                xq_l + ((v // UG_T) * XQW + (v % UG_T) * 4),
            )
            fx.ptr_store(got[3 * i + 2][0].bitcast(fx.Float32), xs_l + v)
    gpu.barrier()


@flyc.jit
def put_mid_block(c, blk, e, col0, n, mid_of):
    """Each column's 32 mids of block ``blk`` as one MXFP4 block to ``k3_mid``:
    thread (column tid / 32, row tid % 32), the block's lanes one half wave; the
    SwiGLU value rounded to bf16 first (the compact-A4 epilogue)."""
    tid, put, mid = c["tid"], c["put"], c["mid"]
    j = tid // 32
    r = tid % 32
    act = bf16_round(mid_of(r, fx.min(j, n - 1)))
    amax = butterfly(abs(act), (1, 2, 4, 8, 16), fx.max)
    byte, qs = mxfp4_scale(amax)
    word = butterfly(e2m1_code(act, qs) << (4 * (r % 8)), (1, 2, 4), bor)
    if const_expr(c["tpx"]):
        # the rank's block of the whole intermediate row, to every rank
        if (j < n) & ((r % 8) == 0):
            base = (
                pick_at(c, e, col0 + j) * UG_T + c["rank"] * c["ug_t"] + blk
            ) * MID_W
            for pe in range_constexpr(c["W"]):
                put(c["peer_midx"][pe], base + r // 8, word)
                if r == 0:
                    put(c["peer_midx"][pe], base + 4, byte)
    else:
        if (j < n) & ((r % 8) == 0):
            base = (pick_at(c, e, col0 + j) * UG_T + blk) * MID_W
            put(mid, base + r // 8, word)
            if r == 0:
                put(mid, base + 4, byte)


@flyc.jit
def put_rows(c, region, width, r, rows, e, col0, n, vals_of):
    """Thread (column j = tid % 16, row pair rp = tid / 16): rows 2 rp, + 1 of the
    item's ``rows`` for column j < n -> ``region`` (``width`` elements a row) at
    the column's pick's row, ``vals_of(row, j)`` the value, bf16 pairs."""
    tid, put_bf = c["tid"], c["put_bf"]
    j = tid % TILE_COLS
    for g in range_constexpr(cdiv(rows // 2, THREADS // TILE_COLS)):
        rp = tid // TILE_COLS + g * (THREADS // TILE_COLS)
        if (j < n) & (rp < rows // 2):
            p = pick_at(c, e, col0 + j)
            put_bf(
                region,
                p * width + r * rows + 2 * rp,
                [vals_of(2 * rp, j), vals_of(2 * rp + 1, j)],
            )


@flyc.jit
def stage_ug(c, n_tiles):
    """This CTA's contiguous range of the ug items (tile, 32-row block), each two
    pairs (gate rows, then up rows): the tile's x restaged at a tile boundary, the
    next pair's weights issued before this pair's MFMAs, and after the up pair the
    SwiGLU mids of the 32 rows leave as MXFP4 blocks. The reduction rows alternate
    between items: one barrier an item."""
    S, bid, red = c["S"], c["bid"], c["red"]
    UGT, DNT = c["ug_t"], c["dn_t"]  # items a tile (TP experts: the rank's slices)
    TU = n_tiles * UGT
    q0 = (bid * TU) // BLOCKS
    n_items = 2 * (((bid + 1) * TU) // BLOCKS - q0)
    # the first tile's x goes out before the first weights: loads complete in
    # order, so x lands (and is quantized) while the weights stream
    t0 = q0 // UGT
    e0, c0, n0 = tile_info(c, t0, n_tiles)
    pre = c["xq_pre"]
    xv0 = xq_all_loads(c) if pre else xq_loads(c, e0, c0, n0)
    # the dn stage's first item: its weights go out as the loop's last prefetch (a
    # duplicate of the last pair went out there before; a CTA without ug items
    # issues them first thing), so they land while the dn stage waits for its mids
    TD = n_tiles * DNT
    dq0 = (bid * TD) // BLOCKS
    cur = ug_or_dn_pair_loads(c, q0, 0, dq0, n_tiles, n_items == 0)
    if const_expr(pre):
        stage_xq_all(
            c, xv0
        )  # every token's blocks: a tile change is only the lanes' tokens
    else:
        stage_xq_cols(c, e0, c0, n0, xv0)
    cur_t = t0
    for i in range(0, n_items):
        q = q0 + i // 2
        pr = i % 2
        ti = q // UGT
        e, col0, n = tile_info(c, ti, n_tiles)
        if const_expr(not pre):
            if ti != cur_t:
                gpu.barrier()  # every thread is done with the previous tile's columns
                stage_xq_cols(c, e, col0, n)
                cur_t = ti
        i1 = fx.min(i + 1, n_items - 1)
        nxt = ug_or_dn_pair_loads(
            c, q0 + i1 // 2, i1 % 2, dq0, n_tiles, i + 1 >= n_items
        )
        if const_expr(pre):
            tok = lane_token(c, e, col0, n)
            xb = [xq_b_tok(c, tok, UG_CHUNKS, kc) for kc in range(UG_CHUNKS)]
        else:
            xb = [xq_b(c, n, UG_CHUNKS, kc) for kc in range(UG_CHUNKS)]
        par = (i // 2) % 2
        pbase = par * (4 * RED_SLOT_FLOATS)
        for t in range_constexpr(2):
            tile_store(c, 4 * par + 2 * pr + t, tile_mfmas_a4(cur[t][0], cur[t][1], xb))
        if pr == 1:
            gpu.barrier()

            def mid_of(row, j):
                gate = row_sum(red + pbase + (row // 16) * RED_SLOT_FLOATS, row % 16, j)
                up = row_sum(
                    red + pbase + (2 + row // 16) * RED_SLOT_FLOATS, row % 16, j
                )
                return gate * sigmoid(gate) * up

            put_mid_block(c, q % UGT, e, col0, n, mid_of)
        cur = nxt
    return cur  # the dn stage's first item's weights, in flight


@flyc.jit
def stage_ug_dyn(c, n_tiles):
    """The ug items from a queue (merged launch, pre-quantized x): CTA bid's first
    item is ``bid``, every later one ``BLOCKS`` + a ticket of the launch's queue
    word (``k3_ugq``, zeroed at the launch's entry by CTA 0 before its xn task's
    flag, which every CTA polls first), so a CTA that finishes early takes more
    (a fixed range left the CTAs ~11 us apart at S = 16). A ticket goes out at an
    item's head, before its up pair's weights (it returns ahead of them), and
    reaches the CTA through LDS (``misc``, a slot an item parity) behind the
    item's barrier. Returns the dn stage's first item's weights, in flight."""
    S, bid, red, tid, misc = c["S"], c["bid"], c["red"], c["tid"], c["misc"]
    UGT, DNT = c["ug_t"], c["dn_t"]
    TU = n_tiles * UGT
    TD = n_tiles * DNT
    dq0 = (bid * TD) // BLOCKS
    ctr = c["ugq"].value
    no = fx.Int32(0) == fx.Int32(1)
    xv = xq_all_loads(c)
    tk = fx.Int32(0)
    if tid == 0:
        tk = global_atomic(ctr, 1)  # the second item's
    q = bid
    cur = ug_or_dn_pair_loads(
        c, q, 0, dq0, n_tiles, q >= TU
    )  # (a CTA without an item: dn's first)
    stage_xq_all(c, xv)
    if tid == 0:
        fx.ptr_store(tk.bitcast(fx.Float32), misc + 2)
    gpu.barrier()
    qn = fx.Int32(BLOCKS) + fx.ptr_load(misc + 2).bitcast(fx.Int32)
    k = fx.Int32(0)
    while q < TU:
        par = k % 2
        t2 = fx.Int32(0)
        if tid == 0:
            t2 = global_atomic(ctr, 1)  # the item after next
        e, col0, n = tile_info(c, q // UGT, n_tiles)
        tok = lane_token(c, e, col0, n)
        xb = [xq_b_tok(c, tok, UG_CHUNKS, kc) for kc in range(UG_CHUNKS)]
        up = ug_or_dn_pair_loads(c, q, 1, dq0, n_tiles, no)
        pbase = par * (4 * RED_SLOT_FLOATS)
        for t in range_constexpr(2):
            tile_store(c, 4 * par + t, tile_mfmas_a4(cur[t][0], cur[t][1], xb))
        if tid == 0:
            fx.ptr_store(t2.bitcast(fx.Float32), misc + par)
        nxt = ug_or_dn_pair_loads(
            c, qn, 0, dq0, n_tiles, qn >= TU
        )  # the next item's gate pair, or dn's first
        for t in range_constexpr(2):
            tile_store(c, 4 * par + 2 + t, tile_mfmas_a4(up[t][0], up[t][1], xb))
        gpu.barrier()
        q_after = fx.Int32(BLOCKS) + fx.ptr_load(misc + par).bitcast(fx.Int32)

        def mid_of(row, j):
            gate = row_sum(red + pbase + (row // 16) * RED_SLOT_FLOATS, row % 16, j)
            upv = row_sum(red + pbase + (2 + row // 16) * RED_SLOT_FLOATS, row % 16, j)
            return gate * sigmoid(gate) * upv

        put_mid_block(c, q % UGT, e, col0, n, mid_of)
        q = qn
        qn = q_after
        cur = nxt
        k = k + 1
    return cur


@flyc.jit
def stage_dn(c, n_tiles, first=None):
    """This CTA's contiguous range of the dn items (tile, 32-row block): the tile's
    mid blocks restaged at a tile boundary, the next item's weights issued before
    this item's MFMAs, each item's 32 y rows of its columns to ``k3_y``. ``first``:
    the first item's weights, in flight (the ug stage's last prefetch)."""
    S, bid, red = c["S"], c["bid"], c["red"]
    DNT = c["dn_t"]
    TD = n_tiles * DNT
    q0 = (bid * TD) // BLOCKS
    n_items = ((bid + 1) * TD) // BLOCKS - q0
    cur = (
        first if first is not None else dn_pair_loads(c, q0, n_tiles)
    )  # (a trace-time choice)
    cur_t = fx.Int32(-1)
    # the finalize's sum of squares (v = bf16(bf16(y) pen_w), the fin stage's values)
    # of each item's 32 rows of each column: a wave partial a column in LDS (two
    # parities), summed and published an item later, behind that item's barrier
    tid, lane, wave, misc, put = c["tid"], c["lane"], c["wave"], c["misc"], c["put"]
    r_pw = rsrc(c["args"]["pen_w"])
    j16, rp = tid % TILE_COLS, tid // TILE_COLS
    prev_q, prev_e, prev_c0, prev_n = (
        fx.Int32(-1),
        fx.Int32(0),
        fx.Int32(0),
        fx.Int32(0),
    )

    def publish_ss(par, pq, pe, pc0, pn):
        if (pq >= 0) & (tid < TILE_COLS) & (tid < pn):
            tot = fx.ptr_load(misc + (par * (WAVES * TILE_COLS) + tid))
            for w in range_constexpr(1, WAVES):
                tot = tot + fx.ptr_load(
                    misc + (par * (WAVES * TILE_COLS) + w * TILE_COLS + tid)
                )
            if const_expr(c["tpx"]):
                # the item's part of the pick's sum, to every rank (DN_T parts in all)
                at = pick_at(c, pe, pc0 + tid) * DN_T + c["rank"] * DNT + pq % DNT
                for pr_ in range_constexpr(c["W"]):
                    put(c["peer_ssx"][pr_], at, tot)
            else:
                put(c["ss"], pick_at(c, pe, pc0 + tid) * DN_T + pq % DN_T, tot)

    for i in range(0, n_items):
        q = q0 + i
        ti = q // DNT
        e, col0, n = tile_info(c, ti, n_tiles)
        if ti != cur_t:
            gpu.barrier()  # every thread is done with the previous tile's columns
            stage_mid_q(c, e, col0, n)
            stamp(c, 12)  # a tile's mids arrived (the CTA's last tile's survives)
            cur_t = ti
        nxt = dn_pair_loads(c, q0 + fx.min(i + 1, n_items - 1), n_tiles)
        xb = [xq_b(c, n, DN_CHUNKS, kc) for kc in range(DN_CHUNKS)]
        par = i % 2
        pbase = par * (2 * RED_SLOT_FLOATS)
        for t in range_constexpr(2):
            tile_store(c, 2 * par + t, tile_mfmas_a4(cur[t][0], cur[t][1], xb))
        gpu.barrier()
        publish_ss((i + 1) % 2, prev_q, prev_e, prev_c0, prev_n)  # the previous item's

        def y_of(row, j):
            return row_sum(red + pbase + (row // 16) * RED_SLOT_FLOATS, row % 16, j)

        ya, yb = (
            y_of(2 * fx.min(rp, fx.Int32(15)), j16),
            y_of(2 * fx.min(rp, fx.Int32(15)) + 1, j16),
        )
        if const_expr(not c["tpx"]):
            put_rows(
                c,
                c["y"],
                MOE_HIDDEN,
                q % DN_T,
                32,
                e,
                col0,
                n,
                lambda row, j: (row % 2 == 0).select(ya, yb),
            )
        # (row pairs 16 .. 31 are no rows of the item: threads 256 .. 511 add 0)
        c0 = (q % DNT) * 32 + 2 * fx.min(rp, fx.Int32(15))
        if const_expr(c["tpx"]):
            c0 = c0 + c["rank"] * c["w2_rows"]  # the rank's output columns
        pw0, pw1 = bf2_f32(
            fx.Int32(bo.buffer_load(r_pw, c0 // 2, vec_width=1, dtype=T.i32))
        )
        v0 = bf16_round(bf16_round(ya) * pw0)
        v1 = bf16_round(bf16_round(yb) * pw1)
        if const_expr(c["tpx"]):
            # the finalized values v = bf16(y pen_w) to every rank: each finalizes the whole row
            if (j16 < n) & (rp < 16):
                at = pick_at(c, e, col0 + j16) * (MOE_HIDDEN // 2) + c0 // 2
                for pr_ in range_constexpr(c["W"]):
                    c["put_bf"](c["peer_vx"][pr_], 2 * at, [v0, v1])
        ss = ((j16 < n) & (rp < 16)).select(v0 * v0 + v1 * v1, fx.Float32(0.0))
        ss = butterfly(ss, (16, 32))  # the wave's four row pairs of column j16
        if lane < TILE_COLS:
            fx.ptr_store(
                ss, misc + (par * (WAVES * TILE_COLS) + wave * TILE_COLS + lane)
            )
        prev_q, prev_e, prev_c0, prev_n = q, e, col0, n
        cur = nxt
    gpu.barrier()  # ``red`` is reused by the fin and unpool stages; the last item's partials are in
    publish_ss(
        (n_items + 1) % 2, prev_q, prev_e, prev_c0, prev_n
    )  # the last item's (parity n_items - 1)


# ---------------------------------------------------------------- 3. fin + all-reduce
@flyc.jit
def stage_fin(c, t, part):
    """fin task (token t, part): the part's columns of the token's slots (two or
    four elements a thread), the finalize (v = bf16(y pen_w), each slot's sum of
    squares over its whole row summed from the dn items' publishes in ``k3_ss``,
    score / rms weighted sum in slot order), pushed as bf16 pairs to every rank's ``ffn``
    region, then the same columns of every rank's partial summed in rank order ->
    ``k3_fbuf`` (plain bf16) and the (token, part) flag. A token none of whose
    picks this rank holds pushes zeros and polls nothing."""
    S, W, tid, lane, wave, red = (
        c["S"],
        c["W"],
        c["tid"],
        c["lane"],
        c["wave"],
        c["red"],
    )
    put, put_bf, poll = c["put"], c["put_bf"], c["poll"]
    y_mb, ss_mb, fbuf, own_ffn = c["y"], c["ss"], c["fbuf"], c["own_ffn"]
    a = c["args"]
    rank = c["rank"]
    AP = c["ar_parts"]
    AR_COLS = MOE_HIDDEN // AP
    PPT = fin_ppt(AP)  # column pairs a thread
    NV = 2 * PPT
    act = NV * tid < AR_COLS
    col = part * AR_COLS + NV * fx.min(
        tid, AR_COLS // NV - 1
    )  # an idle thread loads the last columns
    # the post norm's weight of these columns (bf16 pairs)
    pwv = bo.buffer_load(rsrc(a["pen_w"]), col // 2, vec_width=PPT, dtype=T.i32)
    if const_expr(PPT == 1):
        pw_words = [fx.Int32(pwv)]
    else:
        pwv = fx.Vector(pwv)
        pw_words = [pwv[q] for q in range(PPT)]
    pw = [bf2_f32(w)[h] for w in pw_words for h in range(2)]
    es = [pick(c, t, sl) for sl in range(TOP_K)]
    scs = [
        fx.Float32(
            bo.buffer_load(
                rsrc(a["scores"]),
                t * TOP_K + sl,
                vec_width=1,
                dtype=T.f32,
                cache_modifier=CM_DEV,
            )
        )  # fmt: skip
        for sl in range(TOP_K)
    ]
    # the first slot this rank holds (a slot it does not hold polls that one's row)
    fb = fx.Int32(-1)
    for sl in range_constexpr(TOP_K):
        fb = ((fb < 0) & (es[sl] >= 0)).select(fx.Int32(sl), fb)
    a0 = fx.Float32(0.0)
    a1 = fx.Float32(0.0)
    a2 = fx.Float32(0.0)
    a3 = fx.Float32(0.0)
    # the slot this wave sums the squares of (a slot this rank does not hold: a held one's, taken as 0)
    psl = fx.Int32(fb)
    held_w = fx.Int32(0) == fx.Int32(1)
    for s_ in range_constexpr(TOP_K):
        hit = (wave == s_) & (es[s_] >= 0)
        psl = hit.select(fx.Int32(s_), psl)
        held_w = held_w | hit
    if fb >= 0:
        # the slots' y columns and this wave's slot's sum of squares in one poll round
        # (the ss publish trails its item's y by one item: polled apart it was a
        # second round trip)
        got = poll(
            [
                (y_mb, (t * TOP_K + (es[sl] >= 0).select(fx.Int32(sl), fb)) * (MOE_HIDDEN // 2) + col // 2, PPT)
                for sl in range(TOP_K)
            ]
            + [(ss_mb, (t * TOP_K + psl) * DN_T + 2 * lane, 2)]
        )  # fmt: skip
        got_ss = got[TOP_K]
        vs = []
        for sl in range_constexpr(TOP_K):
            held = es[sl] >= 0
            v = []
            for q in range_constexpr(PPT):
                lo, hi = bf2_f32(got[sl][q])
                # the post norm with input scale: y x weight -> bf16
                v.append(held.select(bf16_round(lo * pw[2 * q]), fx.Float32(0.0)))
                v.append(held.select(bf16_round(hi * pw[2 * q + 1]), fx.Float32(0.0)))
            vs.append(v)
        # every slot's sum of squares over its whole row: the dn items' DN_T publishes,
        # two a lane, a wave a slot; 1 / rms a slot to LDS
        tot = held_w.select(
            got_ss[0].bitcast(fx.Float32) + got_ss[1].bitcast(fx.Float32),
            fx.Float32(0.0),
        )
        tot = butterfly(tot, (1, 2, 4, 8, 16, 32))
        if lane == 0:
            rinv = fx.Float32(1.0) / fmath.sqrt(tot / float(MOE_HIDDEN) + POST_EPS)
            fx.ptr_store(rinv, red + (TOP_K * WAVES + wave))
        gpu.barrier()
        for sl in range_constexpr(TOP_K):
            sc = scs[sl] * fx.ptr_load(red + (TOP_K * WAVES + sl))
            a0 = a0 + sc * vs[sl][0]
            a1 = a1 + sc * vs[sl][1]
            if const_expr(NV == 4):
                a2 = a2 + sc * vs[sl][2]
                a3 = a3 + sc * vs[sl][3]
    if const_expr(NV == 2):
        vals = [a0, a1]
    else:
        vals = [a0, a1, a2, a3]
    if act:
        # this rank's partial of these columns to every rank (itself too)
        for pe in range_constexpr(W):
            put_bf(c["peer_ffn"][pe], (rank * MAX_TOKENS + t) * MOE_HIDDEN + col, vals)
    if act:
        # the ranks' partials of the same columns, summed in rank order, one rounding
        words = []
        for q in range_constexpr(PPT):

            def pair_at(src, q=q):
                return ((src * MAX_TOKENS + t) * MOE_HIDDEN + col) // 2 + q

            lo, hi = sum_partials(poll, own_ffn, pair_at, W)
            words.append(bf16_pair(lo, hi).bitcast(fx.Int32))
        r_f = rsrc(fbuf.value)
        if const_expr(PPT == 1):
            bo.buffer_store(
                words[0], r_f, (t * MOE_HIDDEN + col) // 2, cache_modifier=CM_DEV
            )
        else:
            bo.buffer_store(
                fx.Vector.from_elements(words, fx.Int32), r_f, (t * MOE_HIDDEN + col) // 2, cache_modifier=CM_DEV
            )  # fmt: skip
    release_stores()
    gpu.barrier()
    if tid == 0:
        put(c["fflag"], t * AP + part, fx.Int32(1))


@flyc.jit
def stage_fin_tpx(c, t, part):
    """TP experts' fin task (token t, part): the part's columns of the token's eight
    slots' finalized values v (``vx``, pushed by the rank that owns those columns)
    and each slot's sum of squares (``ssx``: DN_T parts from every rank, a wave a
    slot) in one poll round; score / rms weighted sum in slot order, one bf16
    rounding -> ``k3_fbuf`` and the (token, part) flag. No all-reduce: every rank
    finalizes the whole row from the same values, in the same order."""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    put, poll = c["put"], c["poll"]
    vx, ssx, fbuf = c["own_vx"], c["own_ssx"], c["fbuf"]
    a = c["args"]
    AP = c["ar_parts"]
    AR_COLS = MOE_HIDDEN // AP
    PPT = fin_ppt(AP)  # column pairs a thread
    NV = 2 * PPT
    act = NV * tid < AR_COLS
    col = part * AR_COLS + NV * fx.min(
        tid, AR_COLS // NV - 1
    )  # an idle thread loads the last columns
    scs = [
        fx.Float32(
            bo.buffer_load(
                rsrc(a["scores"]),
                t * TOP_K + sl,
                vec_width=1,
                dtype=T.f32,
                cache_modifier=CM_DEV,
            )
        )  # fmt: skip
        for sl in range(TOP_K)
    ]
    assert WAVES == TOP_K  # wave w sums slot w's squares
    got = poll(
        [(vx, (t * TOP_K + sl) * (MOE_HIDDEN // 2) + col // 2, PPT) for sl in range(TOP_K)]
        + [(ssx, (t * TOP_K + wave) * DN_T + 2 * lane, 2)]
    )  # fmt: skip
    stamp(c, 13)  # the token's v and sums of squares arrived
    tot = got[TOP_K][0].bitcast(fx.Float32) + got[TOP_K][1].bitcast(fx.Float32)
    tot = butterfly(tot, (1, 2, 4, 8, 16, 32))
    if lane == 0:
        fx.ptr_store(
            fx.Float32(1.0) / fmath.sqrt(tot / float(MOE_HIDDEN) + POST_EPS),
            red + (TOP_K * WAVES + wave),
        )
    gpu.barrier()
    acc = [fx.Float32(0.0) for _ in range(NV)]
    for sl in range_constexpr(TOP_K):
        sc = scs[sl] * fx.ptr_load(red + (TOP_K * WAVES + sl))
        for q in range_constexpr(PPT):
            lo, hi = bf2_f32(got[sl][q])
            acc[2 * q] = acc[2 * q] + sc * lo
            acc[2 * q + 1] = acc[2 * q + 1] + sc * hi
    if act:
        words = [
            bf16_pair(acc[2 * q], acc[2 * q + 1]).bitcast(fx.Int32) for q in range(PPT)
        ]
        r_f = rsrc(fbuf.value)
        if const_expr(PPT == 1):
            bo.buffer_store(
                words[0], r_f, (t * MOE_HIDDEN + col) // 2, cache_modifier=CM_DEV
            )
        else:
            bo.buffer_store(fx.Vector.from_elements(words, fx.Int32), r_f, (t * MOE_HIDDEN + col) // 2, cache_modifier=CM_DEV)  # fmt: skip
    release_stores()
    gpu.barrier()
    if tid == 0:
        put(c["fflag"], t * AP + part, fx.Int32(1))


# ---------------------------------------------------------------- 4. unpool
@flyc.jit
def stage_unpool(c, first_w):
    """This CTA's unpool tasks (one at TP 8; ``first_w``: the first's weights,
    issued before the fin flags' wait, which streams them): the summed rows are
    staged in LDS by DMA in passes of XS_LDS_MAX tokens, each pass's 16 x tokens
    bf16 sums pushed to every rank's ``unp`` region. A further task's weights go
    out before the previous task's passes."""
    S, W, tid, lane, wave = c["S"], c["W"], c["tid"], c["lane"], c["wave"]
    red, misc, xcol, poll, put_bf = (
        c["red"],
        c["misc"],
        c["xcol"],
        c["poll"],
        c["put_bf"],
    )
    b = c["bases"]
    AP, KU = c["ar_parts"], b["ku"]
    V4F = fx.Vector.make_type(4, fx.Float32)
    l16, g4 = lane % 16, lane // 16
    r_f = rsrc(c["fbuf"].value)
    assert S % XS_LDS_MAX == 0, S  # a job is one whole pass of XS_LDS_MAX tokens
    task0, _, _ = unp_gtask(c, 0)
    live = (task0 >= 0) & (task0 < b["unp_jobs"])
    if live:
        rocdl.sched_barrier(0)
        poll_flags(c, [(c["fflag"], S * AP)])
    stamp(c, 8)  # the fin flags arrived (unpool CTAs)
    if live:
        cur = first_w
        for k in range_constexpr(KU):
            task, gtask, job_ps = unp_gtask(c, k)
            live_k = (task >= 0) & (
                task < b["unp_jobs"]
            )  # a dead job computes on a clamped pass, publishes nothing
            t0 = fx.min(job_ps, b["unp_passes"] - 1) * XS_LDS_MAX
            nt = XS_LDS_MAX  # every served width is whole passes
            tl = fx.min(l16, nt - 1)
            # the pass's rows by LDS DMA: 16 B a lane a round at M0 + 16 lane
            for i in range_constexpr(nt * (MOE_HIDDEN // 8) // THREADS):
                v = tid + THREADS * i
                rocdl.raw_ptr_buffer_load_lds(
                    r_f,
                    (xcol + (i * THREADS + wave * 64) * 4).llvm_ptr,
                    16,
                    t0 * (MOE_HIDDEN * 2) + v * 16,
                    fx.Int32(0),
                    0,
                    CM_DEV,  # fmt: skip
                )
            # the next job's weights go out behind this job's rows: loads complete in
            # order, so the wait below counts them out (UNP_STEPS a lane still in flight)
            if const_expr(k + 1 < KU):
                nxt = unp_w_loads(c, unp_gtask(c, k + 1)[1])
                rocdl.s_waitcnt(vmcnt=UNP_STEPS)
            else:
                rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            if const_expr(k == 0):
                stamp(c, 9)  # the first job's rows landed in LDS
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for kc in range_constexpr(UNP_STEPS):
                k0 = wave * (MOE_HIDDEN // WAVES) + kc * 32 + 8 * g4
                fbv = fx.Vector(
                    fx.ptr_load(
                        xcol + (tl * (MOE_HIDDEN // 2) + k0 // 2), result_type=V4F
                    )
                )
                acc = mfma_bf16(unp_a(c, cur[kc]), fbv.bitcast(fx.BFloat16), acc)
            fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
            gpu.barrier()
            if const_expr(k == 0):
                stamp(c, 10)  # the first job's MFMAs done
            # branch = bf16(the fp32 sum over the waves, in wave order, x the row's FP8 scale)
            if tid < UNP_ROWS * nt:
                fx.ptr_store(bf16_round(row_sum(red, tid % UNP_ROWS, tid // UNP_ROWS) * unp_scale(c, gtask * UNP_ROWS + tid % UNP_ROWS)), misc + tid)  # fmt: skip
            gpu.barrier()
            if live_k & (tid < (UNP_ROWS // 2) * nt):
                tk = tid // (UNP_ROWS // 2)
                rr = (tid % (UNP_ROWS // 2)) * 2
                vals = [fx.ptr_load(misc + (tk * UNP_ROWS + rr)), fx.ptr_load(misc + (tk * UNP_ROWS + rr + 1))]  # fmt: skip
                for pe in range_constexpr(W):
                    put_bf(
                        c["peer_unp"][pe],
                        (t0 + tk) * HIDDEN + gtask * UNP_ROWS + rr,
                        vals,
                    )
            gpu.barrier()
            if const_expr(k == 0):
                stamp(c, 11)  # the first job's rows pushed
            if const_expr(k + 1 < KU):
                cur = nxt


# ---------------------------------------------------------------- 5. the boundary
@flyc.jit
def stage_bnd(c, t, x_flag=None, x_put=None):
    """bnd task t: the token's whole unpooled row (16 elements a thread, from every
    rank's pushes) through the post norm with input scale (eps 1e-8), the gated
    residual (the fp32 stream in place) and the next layer's pre-attention norm
    (eps 1e-5) -> ``attn_out``. On the last layer the stream is rounded to bf16
    before that norm (the model's final-norm input dtype), whose weight is then the
    final norm's: ``c["last"]`` compiles the rule in, else the launch's ``last``
    argument selects it. The operands go out before the wait on the row. ``x_flag``
    (with ``x_put``, the next layer's mailbox): the next layer's K1 runs later in
    this launch; the token's flag."""
    tid, poll, own_unp = c["tid"], c["poll"], c["own_unp"]
    a = c["args"]
    e0 = tid * BND_ELEMS
    fw_w = [fx.Vector(bo.buffer_load(rsrc(a["pfw"]), e0 // 2 + 4 * i, vec_width=4, dtype=T.i32)) for i in range(2)]  # fmt: skip
    al = [fx.Vector(bo.buffer_load(rsrc(a["alpha"]), e0 + 4 * i, vec_width=4, dtype=T.f32)) for i in range(4)]  # fmt: skip
    be = [fx.Vector(bo.buffer_load(rsrc(a["beta"]), e0 + 4 * i, vec_width=4, dtype=T.f32)) for i in range(4)]  # fmt: skip
    st = [fx.Vector(bo.buffer_load(rsrc(a["res"]), t * HIDDEN + e0 + 4 * i, vec_width=4, dtype=T.f32)) for i in range(4)]  # fmt: skip
    nw_w = [fx.Vector(bo.buffer_load(rsrc(a["npw"]), e0 // 2 + 4 * i, vec_width=4, dtype=T.i32)) for i in range(2)]  # fmt: skip
    rocdl.sched_barrier(0)
    got = poll([(own_unp, (t * HIDDEN + e0) // 2 + 2 * j, 2) for j in range(4)])
    u = [bf2_f32(got[j][q])[h] for j in range(4) for q in range(2) for h in range(2)]
    fw = [
        bf2_f32(fx.Int32(fw_w[k // 8][(k % 8) // 2]))[k % 2] for k in range(BND_ELEMS)
    ]
    # post_input = bf16(branch x weight), nb = bf16(post_input x rstd)
    pp = [bf16_round(u[k] * fw[k]) for k in range(BND_ELEMS)]
    ssp = pp[0] * pp[0]
    for k in range_constexpr(1, BND_ELEMS):
        ssp = ssp + pp[k] * pp[k]
    rstd = hw_rsq(block_sum(c, ssp, 0) / float(HIDDEN) + POST_EPS)
    nb = [bf16_round(pp[k] * rstd) for k in range(BND_ELEMS)]
    new = [
        fx.Float32(al[k // 4][k % 4]) * fx.Float32(st[k // 4][k % 4]) + fx.Float32(be[k // 4][k % 4]) * nb[k]
        for k in range(BND_ELEMS)
    ]  # fmt: skip
    for i in range_constexpr(4):
        bo.buffer_store(fx.Vector.from_elements(new[4 * i : 4 * i + 4], fx.Float32), rsrc(a["res"]), t * HIDDEN + e0 + 4 * i)  # fmt: skip
    if const_expr(c["last"]):
        pre = [bf16_round(v) for v in new]
    else:
        is_last = a["last"] != 0
        pre = [is_last.select(bf16_round(v), v) for v in new]
    ss2 = pre[0] * pre[0]
    for k in range_constexpr(1, BND_ELEMS):
        ss2 = ss2 + pre[k] * pre[k]
    rstd2 = hw_rsq(block_sum(c, ss2, 1) / float(HIDDEN) + RMS_EPS)
    if const_expr(c["defer_k1"]):
        # the next K1 folds its two operand halves with these (``pre_attn.stage_gemv``)
        if tid < 2:
            c["put"](c["nscal"], 2 * t + tid, (tid == 0).select(rstd, rstd2))
    nw = [
        bf2_f32(fx.Int32(nw_w[k // 8][(k % 8) // 2]))[k % 2] for k in range(BND_ELEMS)
    ]
    for i in range_constexpr(2):
        words = [
            bf16_pair(pre[k] * rstd2 * nw[k], pre[k + 1] * rstd2 * nw[k + 1]).bitcast(
                fx.Int32
            )
            for k in range(8 * i, 8 * i + 8, 2)
        ]
        bo.buffer_store(fx.Vector.from_elements(words, fx.Int32), rsrc(a["attn_out"]), (t * HIDDEN + e0) // 2 + 4 * i, cache_modifier=CM_DEV)  # fmt: skip
    if const_expr(x_flag is not None):
        # the next layer's K1 runs in this launch: the token's flag, under that
        # layer's tag (``x_put``)
        release_stores()
        gpu.barrier()
        if tid == 0:
            x_put(x_flag, t, fx.Int32(1))


# ---------------------------------------------------------------- the launch's order
@flyc.jit
def run_k3(c, route_flag=None, xin_flag=None, x_flag=None, x_put=None, res_flag=None):
    """K3's stages on this CTA. ``route_flag`` / ``xin_flag`` (mailbox regions):
    the routing and the expert input were written earlier in this launch, flagged a
    token each. ``x_flag`` / ``x_put``: see ``stage_bnd``."""
    S, bid = c["S"], c["bid"]
    b = c["bases"]
    AP = c["ar_parts"]
    stamp(c, 0)
    if const_expr(route_flag is not None):
        # (``res_flag`` = (region, count): K2's boundary parts flag their stream columns
        # after the x halves the routing came from; the boundary below reads them)
        poll_flags(
            c,
            [(route_flag, S), (xin_flag, S)]
            + ([res_flag] if res_flag is not None else []),
        )
    stamp(c, 1)  # the routing and expert input arrived (a merged launch) / entry
    if const_expr(c["tpx"]):
        n_tiles = stage_plan_tpx(c)
    else:
        n_tiles = stage_plan(c)
    stamp(c, 2)  # the plan built
    if const_expr(c["ug_dyn"]):
        dn_first = stage_ug_dyn(c, n_tiles)
    else:
        dn_first = stage_ug(c, n_tiles)
    stamp(c, 3)  # this CTA's ug items done
    stage_dn(c, n_tiles, dn_first)
    stamp(c, 4)  # this CTA's dn items done
    # the unpool CTAs' first task's weights go out now, behind nothing, so the fin
    # hops stream them; the fin CTAs carry none (their tile is past the buffer)
    first_w = unp_w_loads(c, unp_gtask(c, 0)[1])  # the first job's tile
    rocdl.sched_barrier(0)
    if (bid >= b["fin0"]) & (bid < b["fin0"] + S * AP):
        if const_expr(c["tpx"]):
            stage_fin_tpx(c, (bid - b["fin0"]) // AP, (bid - b["fin0"]) % AP)
        else:
            stage_fin(c, (bid - b["fin0"]) // AP, (bid - b["fin0"]) % AP)
    stamp(c, 5)  # the fin task done (fin CTAs)
    stage_unpool(c, first_w)
    stamp(c, 6)  # the unpool task(s) done (unpool CTAs)
    if bid < S:
        stage_bnd(c, bid, x_flag, x_put)
    stamp(c, 7)  # the boundary done (end)
