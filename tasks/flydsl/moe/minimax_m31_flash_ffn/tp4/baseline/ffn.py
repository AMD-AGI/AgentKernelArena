# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Vendored from sgl-amd-mono @ 52486703 (sgl_amd_mono/minimax_m31/ffn.py; the kernel comes from
# vllm-project/vllm@a50259bb vllm/models/minimax_m3/amd/specialized_decode/mono). Modified for
# AgentKernelArena (FlyDSL 0.3.2 runtime): imports are package-relative (aiter's buffer_ops and the
# common/ device library vendored beside it); the build key carries no source digest
# (plan.kernel_sources read the installed package's files); the two LDS atomics go through
# ffn_sync.lds_atomic_max / lds_atomic_or (fx.llvm.atomic_* needs FlyDSL >= 0.3.4). The device
# code is otherwise unchanged.
"""MiniMax-M3 FFN launch: one persistent kernel from the attention output to the
reduced MoE output (``MiniMaxM3DecoderLayer.forward`` after its attention, o_proj
included).

``BLOCKS`` x ``THREADS``, S rows, one launch per sparse layer and rank:

    push    (128 x 48 columns): o_proj as SGLang runs it (per-token FP8 rows x
            its rowwise-FP8 weight), the partial's bf16 triples to every rank
    reduce  (192): the ranks' pairs in rank order, f32 -> bf16; + residual
            in f32 -> residual out (bf16), the f32 sum, its sum of squares
    norm    (S): Gemma RMSNorm of the f32 sum -> X (bf16)
    router  (8 x 16 experts x 16 K parts): FP32 weight x f32(X), f32 MFMA
    route   (S): the parts summed in order, sigmoid, + bias for selection,
            top-4, renormalized x 2.0
    table   (every CTA): the union of picked experts in LDS, slot 0 the shared
            expert (expert 128) over all S rows
    up      (48 x slots): 16 gate + 16 up rows of a slot's expert, A16W4 MFMA,
            SwiGLU-OAI -> bf16 MID
    down    (384 x 16 rows): every slot's A16W4 GEMV of MID, routed weights
            and shared expert combined -> pushed to every rank
    final   (192): the ranks' partials in rank order -> ``out``
"""

from dataclasses import dataclass
from enum import Enum

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, gpu
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T, as_ir_value

from . import a16w8, guard
from . import buffer_ops as bo
from .a16w4 import scale_f32, scale_loads, step_mfma, weight_loads
from .epoch import EPOCH_MARKS, epoch_end
from .ffn_sync import (
    Mailbox,
    cta_sync,
    lds_atomic_max,
    lds_atomic_or,
    load32,
    load_dev,
    store16,
    store32,
    store64,
    store_dev,
    sync_release,
    wave_sync,
)
from .ops import (
    CM_NT,
    CM_SYS,
    ballot,
    bf2_f32,
    bf16_pair,
    bf16_round,
    butterfly,
    clock,
    fp8_pack4,
    hw_exp2,
    hw_rcp,
    lanes_below,
    peer_bases,
    popcount,
    rsrc,
    uniform,
)
from .plan import (
    BLOCKS,
    THREADS,
    WAVES,
    first_task,
    key_tuple,
    pair_layout,
)
from .sync import Space, preg, sreg

_STREAM = fx.Stream(None)
HIDDEN = 6144
INTER = 768  # a TP4 rank's expert intermediate
EXPERTS = 128
SHARED = 128  # the fused shared expert's index in w13 / w2
TOPK = 4
ROUTED_SCALE = 2.0
EPS = 1e-6
ALPHA, BETA, LIMIT = 1.702, 1.0, 7.0
LOG2E = 1.4426950408889634
MAX_TOKENS = 16

COLS = 32  # columns of a final task
FINAL_TASKS = HIDDEN // COLS  # 192
# the attention seam's pairs carry three bf16 under a 16-bit tag
TRIPLES = HIDDEN // 3  # 2048 a row
TASK_TRIPLES = 16  # triples of a push / reduce task (CTAs 0 .. 127)
ATTN_K = 2048  # a TP4 rank's o_proj input: 16 heads of 128
OP_TILES = 3 * TASK_TRIPLES // 16  # o_proj's 16-row tiles of a push task
INV_FP8_MAX = 1.0 / 448.0  # AITER's per-token FP8 scale: amax / 448 as one f32 multiply
# the push stage's LDS pool words: row amax, the task's partial, the waves' MFMA tiles
OP_AMAX = 0
OP_PART = OP_AMAX + MAX_TOKENS
OP_RED = OP_PART + MAX_TOKENS * 3 * TASK_TRIPLES
OP_WORDS = OP_RED + WAVES * OP_TILES * 256
SEAM_TASKS = TRIPLES // TASK_TRIPLES  # 128 (a token's SSQ parts)
ROUTER_PARTS = 16
ROUTER_K = HIDDEN // ROUTER_PARTS
ROUTER_TASKS = EXPERTS // 16 * ROUTER_PARTS

UP_TILES = INTER // 16  # 48 gate/up tile pairs an expert
DOWN_TASKS = HIDDEN // 16  # 384
XN_LDS_TOKENS = 8  # steps of up to this many rows keep XN in LDS (16 would not fit)
XN_ROW = HIDDEN // 2 + 4  # dwords of a token's XN in LDS, 16 bytes off the banks
XMID_ROW = INTER // 2 + 4  # dwords of a pick's MID in LDS, likewise
K1_STEPS = HIDDEN // 128
K2_STEPS = INTER // 128
UP_WAVE_STEPS = K1_STEPS // WAVES
assert UP_WAVE_STEPS * WAVES == K1_STEPS and UP_WAVE_STEPS % 2 == 0

# CTA bases of the stages placed by count
NORM0 = FINAL_TASKS
ROUTER0 = NORM0 + MAX_TOKENS
SHARED0 = (ROUTER0 + ROUTER_TASKS) % BLOCKS
ROUTED0 = SHARED0 + UP_TILES

# a triple pair's 16-bit tags over its parity's launches: 1 .. TAG16_CYCLE, never
# the 0 a cleared half holds
TAG16_CYCLE = (1 << 16) - 1


class Experts(Enum):
    """Which expert terms the combine sums: all of them in production; one side
    alone for stage parity."""

    ALL = "all"
    SHARED = "shared"
    ROUTED = "routed"


@dataclass(frozen=True)
class FfnBuild:
    tokens: int
    tp: int = 4
    experts: Experts = Experts.ALL
    timeline: bool = False
    # fault injection: this CTA never marks the epoch barrier
    drop_mark: int = -1
    # SGLANG_AMD_MONO_FAULT_INJECT (``guard.fault_injection``): the wait that stalls
    fault_site: int = 0
    fault_after: int = 0


def oproj_split(s: int) -> tuple[int, int]:
    """Waves and 64-wide K steps a wave of SGLang's o_proj GEMM on ``s`` rows
    (``smallm_fp8_gemm`` at (N, K) = (HIDDEN, ATTN_K)), so that the partial is its
    own to the bit."""
    return (8, 4) if s <= 8 else (4, 8)


def slots(s: int) -> int:
    return 1 + TOPK * s


def picks(s: int) -> int:
    """MID columns at most: a token's routed picks (t * 4 + k), then its shared one."""
    return (TOPK + 1) * s


def pool_words(s: int) -> int:
    """The LDS pool: the push stage's o_proj, XN for the up stage (up to
    ``XN_LDS_TOKENS`` tokens), then every pick's MID for the down stage."""
    xn = s * XN_ROW if s <= XN_LDS_TOKENS else 0
    return max(OP_WORDS, xn, picks(s) * XMID_ROW)


def peer_half_bytes(tp: int) -> int:
    """One parity's peer regions: the attention partials (triple pairs), then
    the MoE's (bf16-pair pairs)."""
    return tp * MAX_TOKENS * (TRIPLES + HIDDEN // 2) * 8


def flags_offset(tp: int) -> int:
    """The guard's peer flags, past the two parity halves."""
    return 2 * peer_half_bytes(tp)


def peer_bytes(tp: int) -> int:
    return flags_offset(tp) + guard.FLAG_BYTES


def scratch_layout(s: int) -> dict[str, tuple[int, int]]:
    tagged = pair_layout(
        [
            ("ssq", s * SEAM_TASKS),
            ("xrdy", s),
            ("rstd", s),
            ("logit", s * EXPERTS * ROUTER_PARTS),
            ("route", s * 2 * TOPK),
            ("upf", slots(s) * UP_TILES),
        ],
        align=256,
    )
    off = -(-max(o + n for o, n in tagged.values()) // 256) * 256
    out = dict(tagged)
    for name, nbytes in (
        ("sres", s * HIDDEN * 4),
        ("xn", s * HIDDEN * 2),
        ("mid", slots(s) * 16 * INTER * 2),
        ("tl", BLOCKS * TL_POINTS * 8),
    ):
        out[name] = (off, nbytes)
        off += -(-nbytes // 256) * 256
    return out


def scratch_bytes(s: int) -> int:
    return max(o + n for o, n in scratch_layout(s).values())


TL_STAGES = (
    "push",
    "reduce",
    "norm",
    "router",
    "route",
    "shared",
    "table",
    "up",
    "down",
    "final",
    "epoch",
)
TL_POINTS = 1 + len(TL_STAGES)
EPOCH_WORDS = EPOCH_MARKS + BLOCKS

# LDS route table (int32 words)
T_PICK_E = 0
T_PICK_W = T_PICK_E + TOPK * MAX_TOKENS
T_MASK = T_PICK_W + TOPK * MAX_TOKENS
T_ESLOT = T_MASK + EXPERTS
T_SLOT_E = T_ESLOT + EXPERTS
T_SLOT_MASK = T_SLOT_E + 80
T_SLOT_TOK = T_SLOT_MASK + 80
T_SLOT_PICK = T_SLOT_TOK + 80 * 16
T_MISC = T_SLOT_PICK + 80 * 16
T_WAVE = T_MISC + 8  # a wave's down-stage lane counter
T_WORDS = T_MISC + 16
assert T_WAVE + WAVES <= T_WORDS
assert slots(MAX_TOKENS) <= 80


# ---------------------------------------------------------------- helpers
def put_word(c, addr, i, word):
    """Pair i of ``addr`` := (word, tag) at its space's scope."""
    c["mb"].put_words(addr, i, [word])


@flyc.jit
def flag_after_stores(c, addr, i, who):
    """Pair i := 1 (thread ``who``), released after the CTA's earlier stores."""
    cta_sync(drain=True)
    if who:
        c["mb"].publish(addr, i, fx.Int32(1))


def attn_pair(c, src, t, triple):
    return (src * MAX_TOKENS + t) * TRIPLES + triple


def ffn_pair(c, src, t, colpair):
    attn = c["tp"] * MAX_TOKENS * TRIPLES
    return attn + (src * MAX_TOKENS + t) * (HIDDEN // 2) + colpair


def triple_bits(buf, t, triple):
    """Row t's bf16 at columns 3 triple .. of a [*, HIDDEN] bf16 buffer: one
    8-byte load from the dword holding the first, shifted to bits 0 .. 47."""
    v = fx.Vector(
        bo.buffer_load(
            rsrc(buf),
            t * (HIDDEN // 2) + (3 * triple) // 2,
            vec_width=2,
            dtype=T.i32,
        )
    )
    w = (fx.Int64(fx.Uint32(fx.Int32(v[1]))) << 32) | fx.Int64(
        fx.Uint32(fx.Int32(v[0]))
    )
    return w >> fx.Int64(16 * (triple & 1))


def bf16_bits_f32(bits):
    """A bf16's 16 bits (the low half of an int) as f32."""
    return ((bits & 0xFFFF) << 16).bitcast(fx.Float32)


@flyc.jit
def block_sum(c, v, slot):
    """``v`` summed over the CTA: each wave's butterfly, then waves in order
    (LDS ``red`` words ``slot * WAVES ..``). Every thread gets the sum."""
    lane, wave, red = c["lane"], c["wave"], c["red"]
    v = butterfly(v, (32, 16, 8, 4, 2, 1))
    if lane == 0:
        fx.ptr_store(v, red + (slot * WAVES + wave))
    gpu.barrier()
    tot = fx.Float32(0.0)
    for w in range_constexpr(WAVES):
        tot = tot + fx.ptr_load(red + (slot * WAVES + w))
    return tot


# ---------------------------------------------------------------- seam
@flyc.jit
def stage_push(c, task):
    """Triples 16 task ..: this rank's o_proj partial (``oproj``) as triple pairs
    to every rank's ATTN."""
    s, tid = c["S"], c["tid"]
    oproj(c, task)
    if tid < s * TASK_TRIPLES:
        t = tid // TASK_TRIPLES
        q = tid % TASK_TRIPLES
        tr = task * TASK_TRIPLES + q
        at = c["fpool"] + (OP_PART + t * 3 * TASK_TRIPLES + 3 * q)
        bits = fx.Int64(0)
        for k in range_constexpr(3):
            hi = fx.Int64((fx.ptr_load(at + k).bitcast(fx.Int32) >> 16) & 0xFFFF)
            bits = bits | (hi << (16 * k))
        for p in range_constexpr(c["tp"]):
            c["mb"].put_triple(c["peer"][p], attn_pair(c, c["rank"], t, tr), bits)


@flyc.jit
def oproj(c, task):
    """The rank's o_proj partial of every row at the task's 48 columns -> LDS
    ``OP_PART`` (bf16 values as f32), as SGLang computes it from the attention
    rows: AITER's per-token FP8 quant (scale amax / 448, its v_rcp_f32 inverse,
    clamped, RNE) and ``smallm_fp8_gemm`` on the rowwise-FP8 weight, with its waves,
    K steps, accumulator chains and wave order (``oproj_split``)."""
    s, tid, lane, wave, a = c["S"], c["tid"], c["lane"], c["wave"], c["args"]
    k = ATTN_K
    vw, steps = oproj_split(s)
    chains = 2  # smallm's accumulator chains: step pair p on chain p % 2
    row, kc = lane % 16, (lane // 16) * 16
    k0 = wave * steps * 64
    nb = task * OP_TILES
    pool, fpool = c["pool"], c["fpool"]
    # waves past the GEMM's run on zero-sized buffers: their tiles are never summed
    live = fx.Int64(uniform((wave < vw).select(fx.Int32(-1), fx.Int32(0))))
    rx = rsrc(a["attn"], fx.Int64(s * k * 2) & live)
    rw = rsrc(a["wo"], live)

    # the lane's 16 weight bytes of each tile and step (AITER's (16, 16) shuffle:
    # row l % 16, K bytes 16 (l / 16) .. of each 64-wide step), then its 16 bf16 of
    # row ``row`` at the same K (rows past s read 0)
    wvs = [
        [
            fx.Vector(
                bo.buffer_load(
                    rw,
                    ((nb + tt) * 16 * k + k0 * 16 + lane * 16 + i * 1024) // 4,
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=CM_NT,
                )
            )
            for i in range(steps)
        ]
        for tt in range(OP_TILES)
    ]
    xvs = [
        [
            fx.Vector(
                bo.buffer_load(
                    rx,
                    (row * k + k0 + i * 64 + kc) // 2 + 4 * h,
                    vec_width=4,
                    dtype=T.i32,
                )
            )
            for h in range(2)
        ]
        for i in range(steps)
    ]
    if tid < 16:
        fx.ptr_store(fx.Int32(0), pool + (OP_AMAX + tid))
    gpu.barrier()
    # every row's amax over K, as a nonnegative f32's bits
    mx = fx.Int32(0)
    for i in range_constexpr(steps):
        for h in range_constexpr(2):
            for d in range_constexpr(4):
                word = fx.Int32(xvs[i][h][d])
                mx = fx.max(mx, (word << 16) & 0x7FFF0000)
                mx = fx.max(mx, word & 0x7FFF0000)
    mx = butterfly(mx, (16, 32), fx.max)
    if lane < 16:
        lds_atomic_max(pool + (OP_AMAX + row), mx)
    gpu.barrier()
    scale = fx.ptr_load(pool + (OP_AMAX + row)).bitcast(fx.Float32) * fx.Float32(
        INV_FP8_MAX
    )
    inv = hw_rcp(scale)
    accs = [
        [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(chains)]
        for _ in range(OP_TILES)
    ]
    xq = []
    for i in range_constexpr(steps):
        vals = []
        for h in range_constexpr(2):
            for d in range_constexpr(4):
                word = fx.Int32(xvs[i][h][d])
                for half in range_constexpr(2):
                    bits = (word << 16) if half == 0 else (word & -65536)
                    v = bits.bitcast(fx.Float32) * inv
                    vals.append(
                        fx.min(fx.max(v, fx.Float32(-448.0)), fx.Float32(448.0))
                    )
        xq.append([fp8_pack4(*vals[4 * j : 4 * j + 4]) for j in range(4)])
    # a 16x16x128 MFMA takes two steps
    for p in range_constexpr(steps // 2):
        x8 = fx.Vector.from_elements(xq[2 * p] + xq[2 * p + 1], fx.Int32)
        for tt in range_constexpr(OP_TILES):
            w8 = fx.Vector.from_elements(
                [wvs[tt][2 * p + i][e] for i in range(2) for e in range(4)], fx.Int32
            )
            accs[tt][p % chains] = fx.Vector(
                rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                    T.vec(4, T.f32),
                    [x8, w8, accs[tt][p % chains], 0, 0, 0, 127, 0, 127],
                )
            )
    for tt in range_constexpr(OP_TILES):
        acc = accs[tt][0]
        for ch in range_constexpr(1, chains):
            acc = acc + accs[tt][ch]
        for r in range_constexpr(4):
            fx.ptr_store(
                acc[r], fpool + (OP_RED + (wave * OP_TILES + tt) * 256 + r * 64 + lane)
            )
    gpu.barrier()
    # MFMA C: element tt 256 + r 64 + l is row 4 (l / 16) + r, column 16 tt + l % 16
    for e0 in range_constexpr(0, OP_TILES * 256, THREADS):
        e = tid + e0
        m = (e % 64) // 16 * 4 + e % 256 // 64
        n = e // 256 * 16 + e % 16
        if (e < OP_TILES * 256) & (m < s):
            v = fx.Float32(0.0)
            for w in range_constexpr(vw):
                v = v + fx.ptr_load(fpool + (OP_RED + w * OP_TILES * 256 + e))
            amax = fx.ptr_load(pool + (OP_AMAX + m)).bitcast(fx.Float32)
            ws = fx.Float32(
                bo.buffer_load(rsrc(a["wos"]), nb * 16 + n, vec_width=1, dtype=T.f32)
            )
            sc = (amax * fx.Float32(INV_FP8_MAX)) * ws
            fx.ptr_store(
                bf16_round(v * sc), fpool + (OP_PART + m * 3 * TASK_TRIPLES + n)
            )
    gpu.barrier()


@flyc.jit
def stage_reduce(c, task):
    """Triples 16 task ..: the ranks' values summed in rank order (the
    all-reduce's bf16), + residual in f32 -> residual out, the f32 sum (SRES)
    and the task's sum of squares (SSQ, a pair a token)."""
    s, tid, a = c["S"], c["tid"], c["args"]
    sq = fx.Float32(0.0)
    if tid < s * TASK_TRIPLES:
        t = tid // TASK_TRIPLES
        tr = task * TASK_TRIPLES + tid % TASK_TRIPLES
        got = c["mb"].poll(
            [(c["own"], attn_pair(c, src, t, tr), 1) for src in range(c["tp"])],
            triples=True,
        )
        res = triple_bits(a["res_in"], t, tr)
        col = t * HIDDEN + 3 * tr
        for k in range_constexpr(3):
            word = [got[src][0][k // 2] for src in range(c["tp"])]
            acc = bf16_bits_f32(word[0] >> (16 * (k % 2)))
            for src in range_constexpr(1, c["tp"]):
                acc = acc + bf16_bits_f32(word[src] >> (16 * (k % 2)))
            v = bf16_round(acc) + bf16_bits_f32(fx.Int32(res >> (16 * k)))
            bo.buffer_store(v.to(fx.BFloat16), rsrc(a["res_out"]), col + k)
            store_dev(c["sres"].value, col + k, [v.bitcast(fx.Int32)])
            sq = sq + v * v
        sq = butterfly(sq, (8, 4, 2, 1))
    cta_sync(drain=True)
    if (tid < s * TASK_TRIPLES) & (tid % TASK_TRIPLES == 0):
        t = tid // TASK_TRIPLES
        c["mb"].publish(c["ssq"], t * SEAM_TASKS + task, sq)


@flyc.jit
def stage_norm(c, t):
    """Token t's Gemma RMSNorm of SRES -> XN, then XRDY."""
    tid = c["tid"]
    part = fx.Float32(0.0)
    if tid < SEAM_TASKS:
        got = c["mb"].poll([(c["ssq"], t * SEAM_TASKS + tid, 1)])
        part = got[0][0].bitcast(fx.Float32)
        c["mb"].acquire(Space.SCRATCH)
    cta_sync()
    var = block_sum(c, part, 0) / fx.Float32(HIDDEN)
    rstd = fx.Float32(1.0) / fx.Float32(fmath.sqrt(var + fx.Float32(EPS)))
    # released after the SSQ acquires: the router reads SRES on it, not on XRDY
    if tid == 0:
        c["mb"].publish(c["rstd"], t, rstd)
    per = HIDDEN // THREADS  # 12
    for q in range_constexpr(per // 2):
        col = (q * THREADS + tid) * 2
        store_dev(
            c["xn"].value, t * (HIDDEN // 2) + col // 2, xn_words(c, t, col, rstd, 2)
        )
    flag_after_stores(c, c["xrdy"], t, tid == 0)


def xn_words(c, t, col, rstd, n):
    """XN[t][col .. col + n) from SRES: bf16 pairs of the Gemma RMSNorm, one word
    a pair."""
    sv = load_dev(c["sres"].value, t * HIDDEN + col, n)
    words = []
    for h in range_constexpr(n // 2):
        ww = fx.Int32(
            bo.buffer_load(
                rsrc(c["args"]["norm_w"]), col // 2 + h, vec_width=1, dtype=T.i32
            )
        )
        wl, wh = bf2_f32(ww)
        ol = fx.Int32(sv[2 * h]).bitcast(fx.Float32) * rstd * (fx.Float32(1.0) + wl)
        oh = fx.Int32(sv[2 * h + 1]).bitcast(fx.Float32) * rstd * (fx.Float32(1.0) + wh)
        words.append(bf16_pair(ol, oh).bitcast(fx.Int32))
    return words


@flyc.jit
def await_x(c, tokens_n):
    """XRDY of the first ``tokens_n`` tokens (thread t polls token t); up to
    ``XN_LDS_TOKENS`` tokens, then their XN -> LDS (``xn_operands``)."""
    tid = c["tid"]
    if tid < tokens_n:
        c["mb"].poll([(c["xrdy"], tid, 1)])
        c["mb"].acquire(Space.SCRATCH)
    cta_sync()
    if const_expr(c["xn_lds"] is not None):
        row = HIDDEN // 8
        chunks = tokens_n * row
        n = -(-chunks // THREADS)
        for b in range_constexpr(0, n, 6):
            words, ats = [], []
            for i in range_constexpr(b, min(b + 6, n)):
                v = fx.min(tid + THREADS * i, chunks - 1)
                t, q = v // row, v % row
                words.append(load_dev(c["xn"].value, t * (HIDDEN // 2) + q * 4, 4))
                ats.append(t * XN_ROW + q * 4)
            for i in range_constexpr(len(words)):
                fx.ptr_store(words[i], c["xn_lds"] + ats[i])
        gpu.barrier()


def xn_operands(c, tok, st0):
    """The lane's B operands of up steps st0 .. (``x_operands`` of XN), from
    LDS when XN is there: a unit's global X loads would take most of the
    CTA's vector memory issue."""
    if const_expr(c["xn_lds"] is None):
        return x_operands(c, c["xn"].value, tok, HIDDEN // 2, st0, UP_WAVE_STEPS)
    v4 = fx.Vector.make_type(4, fx.Int32)
    at = tok * XN_ROW + (c["lane"] // 16) * 16
    return [
        [
            fx.ptr_load(c["xn_lds"] + (at + (st0 + i) * 64 + d * 4), result_type=v4)
            for d in range(4)
        ]
        for i in range(UP_WAVE_STEPS)
    ]


# ---------------------------------------------------------------- router
def mfma_f32(a, b, acc):
    from flydsl.expr.rocdl import _blgp_attr

    return fx.Vector(
        rocdl.mfma_f32_16x16x4f32(
            T.vec(4, T.f32),
            as_ir_value(a),
            as_ir_value(b),
            as_ir_value(acc),
            0,
            0,
            _blgp_attr(0),
        ).result
    )


@flyc.jit
def stage_router(c, task):
    """Experts 16 tile .. over K part ``part`` (384 columns, 48 a wave) for
    every token: FP32 weight x f32(X) on f32 MFMAs, the waves summed in order
    -> ``LOGIT[token][expert][part]``. A lane's float4 of weights and four bf16 of
    X feed four MFMAs (K slot = its lane group), the same K order on both."""
    s, tid, lane, wave, red, a = (
        c["S"],
        c["tid"],
        c["lane"],
        c["wave"],
        c["red"],
        c["args"],
    )
    tile, part = task // ROUTER_PARTS, task % ROUTER_PARTS
    k0 = part * ROUTER_K + wave * (ROUTER_K // WAVES) + (lane // 16) * 4
    chunks = ROUTER_K // WAVES // 16
    e = tile * 16 + lane % 16
    wv = [
        fx.Vector(
            bo.buffer_load(
                rsrc(a["router_w"]), e * HIDDEN + k0 + 16 * ch, vec_width=4, dtype=T.f32
            )
        )
        for ch in range(chunks)
    ]
    tok = fx.min(lane % 16, s - 1)
    # the norm's scale, published before XN: the router normalizes its own columns
    got = c["mb"].poll([(c["rstd"], tok, 1)])
    c["mb"].acquire(Space.SCRATCH)
    rstd = got[0][0].bitcast(fx.Float32)
    xv = [xn_words(c, tok, k0 + 16 * ch, rstd, 4) for ch in range(chunks)]
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for ch in range_constexpr(chunks):
        for h in range_constexpr(2):
            lo, hi = bf2_f32(fx.Int32(xv[ch][h]))
            acc = mfma_f32(wv[ch][2 * h], lo, acc)
            acc = mfma_f32(wv[ch][2 * h + 1], hi, acc)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < 16 * s:
        r = tid % 16
        t = tid // 16
        at = (16 * (r // 4) + t) * 4 + r % 4
        tot = fx.Float32(0.0)
        for w in range_constexpr(WAVES):
            tot = tot + fx.ptr_load(red + (w * 64 * 4 + at))
        c["mb"].publish(
            c["logit"], (t * EXPERTS + tile * 16 + r) * ROUTER_PARTS + part, tot
        )
    gpu.barrier()


# ---------------------------------------------------------------- route
def _sigmoid(x):
    return hw_rcp(fx.Float32(1.0) + hw_exp2(x * fx.Float32(-LOG2E)))


@flyc.jit
def stage_route(c, t):
    """Token t: a thread sums 4 of an expert's 16 logit parts, wave 0 the 4
    sums in order (two experts a lane); top-4 of sigmoid + bias (ties to the
    lower expert), weights the selected sigmoids x (2.0 / their sum)."""
    tid, lane, wave, red, a = c["tid"], c["lane"], c["wave"], c["red"], c["args"]
    quarter = ROUTER_PARTS // 4
    e = tid % EXPERTS
    q = tid // EXPERTS
    got = c["mb"].poll(
        [
            (c["logit"], (t * EXPERTS + e) * ROUTER_PARTS + q * quarter + p, 2)
            for p in range(0, quarter, 2)
        ],
    )
    c["mb"].acquire(Space.SCRATCH)
    part = fx.Float32(0.0)
    for p in range_constexpr(quarter):
        part = part + got[p // 2][p % 2].bitcast(fx.Float32)
    fx.ptr_store(part, red + tid)
    gpu.barrier()
    if wave == 0:
        logits = []
        for i in range_constexpr(2):
            tot = fx.Float32(0.0)
            for qq in range_constexpr(4):
                tot = tot + fx.ptr_load(red + (qq * EXPERTS + lane + 64 * i))
            logits.append(tot)
        sc = [_sigmoid(logits[i]) for i in range(2)]
        bias = [
            fx.Float32(
                bo.buffer_load(rsrc(a["bias"]), lane + 64 * i, vec_width=1, dtype=T.f32)
            )
            for i in range(2)
        ]
        neg = fx.Float32(float("-inf"))
        val = [sc[i] + bias[i] for i in range(2)]
        val = [fx.isnan(v).select(neg, v) for v in val]
        sc = [fx.isnan(v).select(fx.Float32(0.0), v) for v in sc]
        taken = [fx.Int32(0), fx.Int32(0)]
        total = fx.Float32(0.0)
        my_id, my_w = fx.Int32(0), fx.Float32(0.0)
        for k in range_constexpr(TOPK):
            v0 = (taken[0] == 0).select(val[0], neg)
            v1 = (taken[1] == 0).select(val[1], neg)
            pick1 = v1 > v0
            v = pick1.select(v1, v0)
            idx = pick1.select(fx.Int32(lane + 64), fx.Int32(lane))
            best = butterfly(v, (32, 16, 8, 4, 2, 1), fx.max)
            cand = (v == best).select(idx, fx.Int32(1 << 20))
            win = butterfly(cand, (32, 16, 8, 4, 2, 1), fx.min)
            mine0 = win == lane
            mine1 = win == lane + 64
            w_mine = mine0.select(sc[0], mine1.select(sc[1], fx.Float32(0.0)))
            w = butterfly(w_mine, (32, 16, 8, 4, 2, 1), fx.max)
            taken[0] = mine0.select(fx.Int32(1), taken[0])
            taken[1] = mine1.select(fx.Int32(1), taken[1])
            total = total + w
            my_id = (lane == k).select(win, my_id)
            my_w = (lane == k).select(w, my_w)
        scale = fx.Float32(ROUTED_SCALE) / fx.max(total, fx.Float32(1e-20))
        live = row_valid(c, t)
        my_id = live.select(my_id, fx.Int32(-1))
        if lane < TOPK:
            c["mb"].publish(c["route"], t * 2 * TOPK + lane, my_id)
            c["mb"].put(c["route"], t * 2 * TOPK + TOPK + lane, my_w * scale)


# ---------------------------------------------------------------- table
def tab_ld(c, i):
    return fx.ptr_load(c["tab"] + i)


def tab_st(c, v, i):
    fx.ptr_store(fx.Int32(v), c["tab"] + i)


@flyc.jit
def load_table(c):
    """Every valid token's picks -> the slots in LDS: slot 0 the shared expert
    over the valid rows, slots 1 .. n_u the picked experts ascending; a slot's
    columns its rows ascending, each column's pick (t * 4 + k; the shared
    expert's 4 S + t). A padded row's picks are -1: in no slot."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    n = s * TOPK
    if tid < n:
        t = tid // TOPK
        got = c["mb"].poll(
            [
                (c["route"], t * 2 * TOPK + tid % TOPK, 1),
                (c["route"], t * 2 * TOPK + TOPK + tid % TOPK, 1),
            ],
        )
        c["mb"].acquire(Space.SCRATCH)
        tab_st(c, got[0][0], T_PICK_E + tid)
        tab_st(c, got[1][0], T_PICK_W + tid)
    if tid < EXPERTS:
        tab_st(c, 0, T_MASK + tid)
    if tid < WAVES:
        tab_st(c, 0, T_WAVE + tid)
    gpu.barrier()
    if tid < n:
        e = tab_ld(c, T_PICK_E + tid)
        if e >= 0:
            lds_atomic_or(c["tab"] + (T_MASK + e), fx.Int32(1) << (tid // TOPK))
    gpu.barrier()
    e = tid
    m = fx.Int32(0)
    if tid < EXPERTS:
        m = tab_ld(c, T_MASK + e)
    ball = ballot(m != 0)
    if (wave == 0) & (lane == 0):
        tab_st(c, popcount(ball), T_MISC + 1)
    gpu.barrier()
    below = (wave == 1).select(tab_ld(c, T_MISC + 1), fx.Int32(0))
    slot = 1 + below + lanes_below(ball, lane)
    if tid < EXPERTS:
        tab_st(c, slot, T_ESLOT + e)
        if m != 0:
            tab_st(c, e, T_SLOT_E + slot)
            tab_st(c, m, T_SLOT_MASK + slot)
    if (wave == 1) & (lane == 0):
        tab_st(c, below + popcount(ball), T_MISC)
    shared_slot(c)
    if tid < n:
        t = tid // TOPK
        e = tab_ld(c, T_PICK_E + tid)
        if e >= 0:
            sl = tab_ld(c, T_ESLOT + e)
            col = popcount(tab_ld(c, T_SLOT_MASK + sl) & ((fx.Int32(1) << t) - 1))
            tab_st(c, t, T_SLOT_TOK + sl * 16 + col)
            tab_st(c, tid, T_SLOT_PICK + sl * 16 + col)
    gpu.barrier()


@flyc.jit
def shared_slot(c):
    """Slot 0: the shared expert over the valid rows (columns ascending; an
    unused column reads row 0)."""
    s, tid = c["S"], c["tid"]
    if tid == 0:
        tab_st(c, SHARED, T_SLOT_E)
        tab_st(c, c["vmask"], T_SLOT_MASK)
    if tid < 16:
        tab_st(c, 0, T_SLOT_TOK + tid)
    gpu.barrier()
    if (tid < s) & row_valid(c, fx.min(tid, s - 1)):
        col = popcount(c["vmask"] & ((fx.Int32(1) << tid) - 1))
        tab_st(c, tid, T_SLOT_TOK + col)
        tab_st(c, TOPK * s + tid, T_SLOT_PICK + col)
    gpu.barrier()


def row_valid(c, t):
    return ((c["vmask"] >> t) & 1) != 0


def valid_mask(c):
    """The step's real rows (a padded row's slot is -1), a bit a row: every
    wave's lane t < S reads row t's slot."""
    s, lane = c["S"], c["lane"]
    hi = fx.Int32(
        bo.buffer_load(
            rsrc(c["args"]["slots"]),
            2 * fx.min(lane, s - 1) + 1,
            vec_width=1,
            dtype=T.i32,
        )
    )
    ball = ballot((lane < s) & (hi >= 0))
    return fx.Int32(ball & fx.Int64((1 << s) - 1))


def n_slots(c):
    return 1 + tab_ld(c, T_MISC)


def slot_count(c, u):
    return popcount(tab_ld(c, T_SLOT_MASK + u))


def slot_col(c, cnt):
    """The lane's column of a slot of ``cnt`` columns (past them: the last)."""
    return fx.max(fx.min(c["lane"] % 16, cnt - 1), 0)


# ---------------------------------------------------------------- experts
def _swiglu(g, u):
    g = fx.min(g, fx.Float32(LIMIT))
    u = fx.max(fx.min(u, fx.Float32(LIMIT)), fx.Float32(-LIMIT))
    return g * _sigmoid(g * fx.Float32(ALPHA)) * (u + fx.Float32(BETA))


def x_operands(c, base, row, k_dwords, st0, steps):
    """The lane's B operands of steps st0 ..: row ``row`` of a bf16 [*, K]
    scratch (``k_dwords`` a row), 8 K values an MFMA."""
    lane = c["lane"]
    at = row * k_dwords + (lane // 16) * 16
    return [
        [load_dev(base, at + (st0 + i) * 64 + d * 4, 4) for d in range(4)]
        for i in range(steps)
    ]


@flyc.jit
def stage_up(c, u, j, weights, shared=False, nxt=None):
    """Slot u's tile pair j: gate rows 16 j .., up rows 768 + 16 j .. of its
    expert over this wave's K steps; waves summed in order; SwiGLU-OAI ->
    bf16 ``MID[u][column][16 j ..]``, then UPF. ``weights``: the pair's
    ``up_weights`` (``shared``: the MXFP8 shared expert's). ``nxt``: the
    routed unit after it; returns that unit's ``unit_weights``, issued past
    this pair's X reads (loads return in order: from global memory, the GEMV
    still waits on X only)."""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    e = tab_ld(c, T_SLOT_E + u)
    cnt = slot_count(c, u)
    st0 = wave * UP_WAVE_STEPS
    rgs = [e * (2 * UP_TILES) + j, e * (2 * UP_TILES) + UP_TILES + j]
    wvs, svs = weights
    tok = tab_ld(c, T_SLOT_TOK + u * 16 + slot_col(c, cnt))
    xs = xn_operands(c, tok, st0)
    ahead = None
    if const_expr(nxt is not None):
        ahead = unit_weights(c, nxt)
    accs = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(2)]
    for i in range_constexpr(UP_WAVE_STEPS):
        for h in range_constexpr(2):
            if const_expr(shared):
                accs[h] = a16w8.step_mfma(accs[h], wvs[h][i], svs[h][i], xs[i])
            else:
                accs[h] = step_mfma(
                    accs[h], wvs[h][i], scale_f32(svs[h], rgs[h], i), xs[i]
                )
    for h in range_constexpr(2):
        fx.ptr_store(accs[h], red + ((wave * 2 + h) * 64 + lane) * 4)
    gpu.barrier()
    if tid < 256:
        col = tid // 16
        r = tid % 16
        if col < cnt:
            at = (16 * (r // 4) + col) * 4 + r % 4
            g = fx.Float32(0.0)
            up = fx.Float32(0.0)
            for w in range_constexpr(WAVES):
                g = g + fx.ptr_load(red + ((w * 2) * 64 * 4 + at))
                up = up + fx.ptr_load(red + ((w * 2 + 1) * 64 * 4 + at))
            v = _swiglu(g, up).to(fx.BFloat16)
            store16(c["mid"].value, (u * 16 + col) * INTER + j * 16 + r, v, "agent")
    flag_after_stores(c, c["upf"], u * UP_TILES + j, tid == 0)
    return ahead


def unit_weights(c, unit):
    """Routed up unit ``unit``'s weights for this wave (``up_weights``); past
    the last unit, zero-sized reads."""
    total = (n_slots(c) - 1) * UP_TILES
    live = unit < total
    unit = fx.min(unit, total - 1)
    u, j = 1 + unit // UP_TILES, unit % UP_TILES
    e = tab_ld(c, T_SLOT_E + u)
    rgs = [e * (2 * UP_TILES) + j, e * (2 * UP_TILES) + UP_TILES + j]
    # a descriptor's size must be uniform: the unit is the CTA's
    nbytes = fx.Int64(uniform(live.select(fx.Int32(-1), fx.Int32(0))))
    return up_weights(c, rgs, c["wave"] * UP_WAVE_STEPS, nbytes=nbytes)


def up_weights(c, rgs, st0, shared=False, nbytes=None):
    a = c["args"]
    if const_expr(shared):
        wvs = [
            a16w8.weight_loads(a["w13s"], rg, HIDDEN, st0, UP_WAVE_STEPS) for rg in rgs
        ]
        svs = [
            a16w8.scale_loads(a["s13s"], rg, HIDDEN, st0, UP_WAVE_STEPS) for rg in rgs
        ]
        return wvs, svs
    wvs = [
        weight_loads(a["w13"], rg, HIDDEN, st0, UP_WAVE_STEPS, nbytes=nbytes)
        for rg in rgs
    ]
    svs = [
        scale_loads(a["s13"], rg, HIDDEN, st0, UP_WAVE_STEPS, nbytes=nbytes)
        for rg in rgs
    ]
    return wvs, svs


@flyc.jit
def down_slot(c, u, tasks, shared=False):
    """Slot u's down GEMV over every task's 16 rows -> LDS ``CONTRIB``."""
    lane, wave, a = c["lane"], c["wave"], c["args"]
    if const_expr(shared):
        rgs = list(tasks)
        wvs = [a16w8.weight_loads(a["w2s"], rg, INTER, 0, K2_STEPS) for rg in rgs]
        svs = [a16w8.scale_loads(a["s2s"], rg, INTER, 0, K2_STEPS) for rg in rgs]
    else:
        e = tab_ld(c, T_SLOT_E + u)
        rgs = [e * DOWN_TASKS + task for task in tasks]
        wvs = [weight_loads(a["w2"], rg, INTER, 0, K2_STEPS) for rg in rgs]
        svs = [scale_loads(a["s2"], rg, INTER, 0, K2_STEPS) for rg in rgs]
    cnt = slot_count(c, u)
    c["mb"].poll([(c["upf"], u * UP_TILES + fx.min(lane, UP_TILES - 1), 1)])
    c["mb"].acquire(Space.SCRATCH)
    k = (u - wave) // WAVES
    # a lane's MID loads span tiles other lanes polled
    wave_sync(c["tab"] + (T_WAVE + wave), 64 * (2 * k + 1))
    stage_slot_mid(c, u, cnt)
    # and its B reads span chunks other lanes stored
    wave_sync(c["tab"] + (T_WAVE + wave), 64 * (2 * k + 2))
    col = slot_col(c, cnt)
    pick = tab_ld(c, T_SLOT_PICK + u * 16 + col)
    v4 = fx.Vector.make_type(4, fx.Int32)
    at = fx.max(fx.min(pick, picks(c["S"]) - 1), 0) * XMID_ROW + (lane // 16) * 16
    accs = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in tasks]
    for i in range_constexpr(K2_STEPS):
        # a step's B at a time, each task's MFMA chain in its order
        xs = [
            fx.ptr_load(c["xmid"] + (at + i * 64 + d * 4), result_type=v4)
            for d in range(4)
        ]
        for ti in range_constexpr(len(tasks)):
            if const_expr(shared):
                accs[ti] = a16w8.step_mfma(accs[ti], wvs[ti][i], svs[ti][i], xs)
            else:
                sc = scale_f32(svs[ti], rgs[ti], i)
                accs[ti] = step_mfma(accs[ti], wvs[ti][i], sc, xs)
    for ti in range_constexpr(len(tasks)):
        acc = accs[ti]
        if lane % 16 < cnt:
            for q in range_constexpr(4):
                fx.ptr_store(
                    acc[q],
                    c["contrib"] + ((ti * 80 + pick) * 16 + 4 * (lane // 16) + q),
                )


@flyc.jit
def stage_slot_mid(c, u, cnt):
    """Slot u's ``cnt`` MID columns -> LDS ``XMID[pick]``, a 16-byte chunk a
    lane a load, every load out before a store: each lane loading its own B
    would repeat the last column's ``16 - cnt`` times."""
    s, lane = c["S"], c["lane"]
    row = INTER // 8
    n = cnt * row
    iters = -(-s * row // 64)
    # six loads out at a time: the slot's down weights hold most registers
    for b in range_constexpr(0, iters, 6):
        words = []
        for i in range_constexpr(b, min(b + 6, iters)):
            w = fx.Vector(_llvm.mlir_undef(T.vec(4, T.i32)))
            if i * 64 < n:
                v = fx.min(lane + 64 * i, n - 1)
                at = (u * 16 + v // row) * (INTER // 2) + v % row * 4
                w = load_dev(c["mid"].value, at, 4)
            words.append(w)
        for i in range_constexpr(b, min(b + 6, iters)):
            if i * 64 < n:
                v = fx.min(lane + 64 * i, n - 1)
                pick = tab_ld(c, T_SLOT_PICK + u * 16 + v // row)
                at = pick * XMID_ROW + v % row * 4
                fx.ptr_store(words[i - b], c["xmid"] + at)


@flyc.jit
def stage_down(c, tasks):
    """Rows 16 task .. of every task in ``tasks``: a slot a wave, round robin,
    its MID columns loaded once for every task; a column's result ->
    LDS ``CONTRIB[task][pick]``, then each token's routed picks (top-k order) x
    their weights + the shared expert -> bf16 pairs pushed to every rank's FFN
    region."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    nt = len(tasks)
    nu = n_slots(c)
    for u in range(wave, nu, WAVES):
        # slot 0 is the shared expert's (``shared_slot``), its weights MXFP8
        if u == 0:
            down_slot(c, u, tasks, shared=True)
        else:
            down_slot(c, u, tasks)
    gpu.barrier()
    if tid < nt * s * 8:
        ti = tid // (s * 8)
        t = tid % (s * 8) // 8
        r2 = tid % 8
        task = tasks[0]
        for q in range_constexpr(1, nt):
            task = (ti == q).select(tasks[q], task)
        # AITER's stage 2: bf16(weight x acc) a pick, added in bf16 (its
        # atomics' order is a race; here the shared expert, then top-k order)
        vals = []
        for h in range_constexpr(2):
            r = 2 * r2 + h
            tot = fx.Float32(0.0)
            if const_expr(c["experts"] != Experts.ROUTED):
                tot = bf16_round(
                    fx.ptr_load(c["contrib"] + ((ti * 80 + TOPK * s + t) * 16 + r))
                )
            if const_expr(c["experts"] != Experts.SHARED):
                for k in range_constexpr(TOPK):
                    w = tab_ld(c, T_PICK_W + t * TOPK + k).bitcast(fx.Float32)
                    v = fx.ptr_load(c["contrib"] + ((ti * 80 + t * TOPK + k) * 16 + r))
                    tot = bf16_round(tot + bf16_round(w * v))
            vals.append(row_valid(c, t).select(tot, fx.Float32(0.0)))
        word = bf16_pair(vals[0], vals[1]).bitcast(fx.Int32)
        cp = task * 8 + r2
        for p in range_constexpr(c["tp"]):
            put_word(c, c["peer"][p], ffn_pair(c, c["rank"], t, cp), word)
    gpu.barrier()


@flyc.jit
def stage_final(c, task):
    s, tid, a = c["S"], c["tid"], c["args"]
    if tid < s * COLS // 2:
        t = tid // (COLS // 2)
        cp = task * (COLS // 2) + tid % (COLS // 2)
        got = c["mb"].poll(
            [(c["own"], ffn_pair(c, src, t, cp), 1) for src in range(c["tp"])],
        )
        lo, hi = bf2_f32(got[0][0])
        for src in range_constexpr(1, c["tp"]):
            l2, h2 = bf2_f32(got[src][0])
            lo, hi = lo + l2, hi + h2
        bo.buffer_store(
            bf16_pair(lo, hi).bitcast(fx.Int32), rsrc(a["out"]), t * (HIDDEN // 2) + cp
        )


# ---------------------------------------------------------------- the launch
@flyc.jit
def stamp(c, k):
    # compile-time gate outside, traced condition inside
    if const_expr(c["timeline"]):  # noqa: SIM102
        if c["tid"] == 0:
            bo.buffer_store(
                clock(), rsrc(c["tl"].value), fx.block_idx.x * TL_POINTS + k
            )


@flyc.jit
def run_ffn(c, bid):
    s = c["S"]
    stamp(c, 0)
    for task in range(first_task(bid, 0), SEAM_TASKS, BLOCKS):
        stage_push(c, task)
    gpu.barrier()
    stamp(c, 1)
    for task in range(first_task(bid, 0), SEAM_TASKS, BLOCKS):
        stage_reduce(c, task)
    gpu.barrier()
    stamp(c, 2)
    for t in range(first_task(bid, NORM0), s, BLOCKS):
        stage_norm(c, t)
    stamp(c, 3)
    for task in range(first_task(bid, ROUTER0), ROUTER_TASKS, BLOCKS):
        stage_router(c, task)
    stamp(c, 4)
    for t in range(first_task(bid, NORM0), s, BLOCKS):
        stage_route(c, t)
    gpu.barrier()
    stamp(c, 5)
    sh = first_task(bid, SHARED0)
    if sh < UP_TILES:
        weights = up_weights(c, [sh, UP_TILES + sh], c["wave"] * UP_WAVE_STEPS, True)
        await_x(c, s)
        shared_slot(c)
        stage_up(c, 0, sh, weights, shared=True)
    stamp(c, 6)
    load_table(c)
    total = (n_slots(c) - 1) * UP_TILES
    first = first_task(bid, ROUTED0)
    weights = unit_weights(c, first)
    # the router no longer orders XN before the table; a shared CTA has it in LDS
    if sh >= UP_TILES:
        await_x(c, s)
    stamp(c, 7)
    for unit in range(first, total, BLOCKS):
        weights = stage_up(
            c, 1 + unit // UP_TILES, unit % UP_TILES, weights, nxt=unit + BLOCKS
        )
    stamp(c, 8)
    if bid < DOWN_TASKS - BLOCKS:
        stage_down(c, [bid, bid + BLOCKS])
    else:
        stage_down(c, [bid])
    stamp(c, 9)
    for task in range(first_task(bid, 0), FINAL_TASKS, BLOCKS):
        stage_final(c, task)
    stamp(c, 10)


def kernel_name(key: FfnBuild) -> str:
    name = f"minimax_m3_mono_ffn_s{key.tokens}_tp{key.tp}"
    if key.experts is not Experts.ALL:
        name += f"_{key.experts.value}"
    if key.timeline:
        name += "_tl"
    if key.drop_mark >= 0:
        name += f"_drop{key.drop_mark}"
    if key.fault_site:
        name += f"_fault{key.fault_site}_{key.fault_after}"
    return name


def build_ffn(key: FfnBuild):
    s, tp = key.tokens, key.tp
    assert 1 <= s <= MAX_TOKENS
    layout = scratch_layout(s)
    half = peer_half_bytes(tp)
    keyed = key_tuple(key)

    @fx.struct
    class Lds:
        red: fx.Array[fx.Float32, WAVES * 2 * 64 * 4, 16]
        tab: fx.Array[fx.Int32, T_WORDS, 16]
        contrib: fx.Array[fx.Float32, 2 * 80 * 16, 16]
        # XN while the up stage reads it (``XN_LDS_TOKENS``), then the down
        # stage's MID columns (``stage_slot_mid``)
        pool: fx.Array[fx.Int32, pool_words(s), 16]

    @flyc.kernel(name=kernel_name(key), known_block_size=[THREADS, 1, 1])
    def kffn(
        attn: Int64,
        res_in: Int64,
        res_out: Int64,
        out: Int64,
        slots: Int64,
        norm_w: Int64,
        router_w: Int64,
        bias: Int64,
        w13: Int64,
        s13: Int64,
        w2: Int64,
        s2: Int64,
        w13s: Int64,
        s13s: Int64,
        w2s: Int64,
        s2s: Int64,
        wo: Int64,
        wos: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        mode: Int32,
    ):
        _ = keyed
        with guard.scope(
            guard.Guard(
                epoch,
                own=sym,
                peers=peers,
                rank=rank,
                npes=tp,
                flags_offset=flags_offset(tp),
                fault_site=key.fault_site or None,
                fault_after=key.fault_after,
            )
        ):
            if mode != 0:
                bid = fx.block_idx.x
                tid = fx.thread_idx.x
                lds = fx.SharedAllocator().allocate(Lds).peek()
                ep = load32(epoch, 0, "agent")
                tag = ep + 1
                raw = peer_bases(peers, tp)
                par = fx.Int64(ep & 1) * fx.Int64(half)
                # a seam half's launches are those of one parity
                mb = Mailbox(tag, tag16_of(ep))
                c = {
                    "S": s,
                    "tp": tp,
                    "experts": key.experts,
                    "timeline": key.timeline,
                    "tid": tid,
                    "lane": tid % 64,
                    "wave": tid // 64,
                    "rank": rank,
                    "mb": mb,
                    "red": lds.red.ptr,
                    "tab": lds.tab.ptr,
                    "contrib": lds.contrib.ptr,
                    "xn_lds": lds.pool.ptr if s <= XN_LDS_TOKENS else None,
                    "xmid": lds.pool.ptr,
                    "pool": lds.pool.ptr,
                    "fpool": fx.recast_iter(fx.Float32, lds.pool.ptr),
                    "own": preg(sym + par, 0, "peer"),
                    "half_words": half // 8,
                    "peer": [preg(b + par, 0, "peer") for b in raw],
                    "args": {
                        "attn": attn,
                        "res_in": res_in,
                        "res_out": res_out,
                        "out": out,
                        "slots": slots,
                        "norm_w": norm_w,
                        "router_w": router_w,
                        "bias": bias,
                        "w13": w13,
                        "s13": s13,
                        "w2": w2,
                        "s2": s2,
                        "w13s": w13s,
                        "s13s": s13s,
                        "w2s": w2s,
                        "s2s": s2s,
                        "wo": wo,
                        "wos": wos,
                    },
                }
                for region, (off, _) in layout.items():
                    c[region] = sreg(scratch, off, region)
                c["vmask"] = valid_mask(c)
                run_ffn(c, bid)
                finish(c, key, epoch, ep, tag)
                stamp(c, 11)

    @flyc.jit
    def launch(
        attn: Int64,
        res_in: Int64,
        res_out: Int64,
        out: Int64,
        slots: Int64,
        norm_w: Int64,
        router_w: Int64,
        bias: Int64,
        w13: Int64,
        s13: Int64,
        w2: Int64,
        s2: Int64,
        w13s: Int64,
        s13s: Int64,
        w2s: Int64,
        s2s: Int64,
        wo: Int64,
        wos: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        mode: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        kffn(
            attn,
            res_in,
            res_out,
            out,
            slots,
            norm_w,
            router_w,
            bias,
            w13,
            s13,
            w2,
            s2,
            w13s,
            s13s,
            w2s,
            s2s,
            wo,
            wos,
            scratch,
            sym,
            peers,
            rank,
            epoch,
            mode,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    return launch


def tag16_of(ep):
    """Launch ``ep``'s triple-pair tag: its parity's launches count 1 .. TAG16_CYCLE."""
    return (ep >> 1) % TAG16_CYCLE + 1


@flyc.jit
def finish(c, key, epoch, ep, tag):
    """Every CTA marks the launch done after its work; CTA 0 waits for the marks
    and moves the epoch on (``common.epoch``), then ends a tag cycle."""
    tid, bid = c["tid"], fx.block_idx.x
    cta_sync(drain=True)
    if (tid == 0) & (bid != key.drop_mark):
        sync_release("agent")
        store32(epoch, EPOCH_MARKS + bid, tag, "agent")
    epoch_end({"bid": bid, "tid": tid}, epoch, ep)
    clear_cycle(c, ep)


@flyc.jit
def clear_cycle(c, ep):
    """CTA 0, past the epoch barrier of its parity's last launch of a tag cycle:
    zero this rank's receive half, so no stale pair there carries the next cycle's
    tags. Every read of it is done, and no peer writes it again before this rank's
    next launch has pushed its data."""
    if (fx.block_idx.x == 0) & (tag16_of(ep) == TAG16_CYCLE):
        for i in range(c["tid"], c["half_words"], THREADS):
            store64(c["own"].value, i, fx.Int64(0), None)


__all__ = [
    "BLOCKS",
    "CM_SYS",
    "EPOCH_WORDS",
    "Experts",
    "FfnBuild",
    "MAX_TOKENS",
    "build_ffn",
    "peer_bytes",
    "scratch_bytes",
]
