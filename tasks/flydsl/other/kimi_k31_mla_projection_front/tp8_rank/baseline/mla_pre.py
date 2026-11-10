# SPDX-License-Identifier: Apache-2.0
# Vendored from sgl-amd-mono @ 52486703 (sgl_amd_mono/kimi_k3/gfx950/attention/mla_pre.py). Modified
# for AgentKernelArena: imports are package-relative (aiter's buffer_ops and sgl-amd-mono's common/
# helpers are vendored beside it); the build-key source tag is plan.py's constant. The kernel, its
# stages and the host launcher ``mla_pre_launch`` are otherwise unchanged.
"""K1m ``mla_pre``: one Kimi-K3.1 MLA (DSA) layer's projection front, from the
AttnRes seam to every projection SGLang's MLA forward and its indexer read, for
a spec-verify step (S rows). One launch a layer, ``BLOCKS`` x ``THREADS``:

    attn CTAs 0 .. 55   the attention-side seam (K1's): slice -> gate -> mix,
                        input_layernorm -> the bf16 x rows (XRDY a slice)
    stage 1, CTAs 56 .. the projections of x, a 16-row group a task (weights
                        issued at launch, the second group's under the first's
                        MFMAs): fused_qkv_a (132 groups), the output gate's g_proj
                        (96), the indexer's wk (8) and weights_proj (2). The
                        latent rows go to LAT (and the raw kv/rope latent out);
                        a q / kv group also publishes its rows' sum of squares.
    stage 2, every CTA  from the normed q latent: q_b (144 groups) and the
                        indexer's wq_b (256). A task sums the 96 q groups' squares
                        in group order (rstd as aiter's fused_qk_rmsnorm: sum / n
                        + eps, then (x rstd) w), normalizes into LDS, GEMVs.
    CTA 255             also the kv norm -> k_nope, and the normed q latent out.

Outputs (SGLang reads them through ``_mono_pre``): q [S, 2304] (q_b), latent
[S, 576] (raw), k_nope [S, 512] (normed), q_lora [S, 1536] (normed), gate
[S, 1536] (g_proj), idx_q [S, 4096] (wq_b), idx_k [S, 128] (wk), idx_w [S, 32]
(weights_proj). Every rounding follows SGLang's bf16 hand-offs; the GEMV split-K
order differs.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from . import buffer_ops as bo
from .ops import (
    CM_DEV,
    bf16_round,
    bf_hi,
    bf_lo,
    hw_rsq,
    row_sum,
    rsrc,
)
from .plan import (
    BLOCKS,
    THREADS,
    WAVES,
    kernel_sources,
    key_tuple,
    pair_layout,
)
from .sync import Mailbox, sreg
from . import gemv
from .dims import COLS, HIDDEN, LDS_PAD, NA, ROWS
from .seam import (
    f32_of,
    stage_gate,
    stage_mix,
    stage_slice,
)
from .stamps import stamp, stamp_begin, stamp_flush
from .tag import step_tag

SOURCES = kernel_sources(__name__)

# The SGLang names this front and its glue read (test/unit/test_contract.py pins
# them); ``_mono_pre`` is where SGLang's MLA forward and indexer take its outputs.
SGLANG_READS = {
    "srt/models/deepseek_v2.py": (
        "fused_qkv_a_proj_with_mqa",
        "q_b_proj",
        "q_a_layernorm",
        "kv_a_layernorm",
        "indexer",
    ),
    "srt/models/kimi_k3.py": ("g_proj", '"_mono_pre"'),
    "srt/layers/attention/dsa/dsa_indexer.py": ("wk", "weights_proj", "wq_b"),
    # dev_amd's ROCm decode / verify prologue takes them
    "srt/layers/attention/dsa/indexer_prologue_hip.py": ('"_mono_pre"',),
    "srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py": (
        '"_mono_pre"',
        '"q_nope_out"',
    ),
}

QL = 1536  # q_lora_rank
KV = 512  # kv_lora_rank
ROPE = 64
LATW = QL + KV + ROPE  # 2112: fused_qkv_a's rows
GATE = 1536  # the output gate's rows a rank (96 heads x 128 / 8)
IK, IW, IQ = 128, 32, 32 * 128  # the indexer's wk, weights_proj, wq_b rows
QB = 12 * 192  # q_b's rows a rank (12 heads x (128 nope + 64 rope))
G_LAT = LATW // ROWS  # 132
G_Q = QL // ROWS  # 96 (the q part of the latent)
G_KV = KV // ROWS  # 32
S1 = (LATW + GATE + IK + IW) // ROWS  # 238 stage-1 groups
S1_GATE0, S1_WK0, S1_WP0 = G_LAT, G_LAT + GATE // ROWS, G_LAT + (GATE + IK) // ROWS
S2 = (QB + IQ) // ROWS  # 400 stage-2 groups
S2_IQ0 = QB // ROWS  # 144
NCTA1 = BLOCKS - NA  # 200 stage-1 CTAs
S1_EXTRA = S1 - NCTA1  # 38 stage-1 groups past the stage-1 CTAs
# stage 3 (qn): q_nope @ w_kc, a (head, 128 kv_lora columns) a task
NHQ, QHD, QN = 12, 192, 128  # q heads a rank, a head's q_b rows, its nope part
S3_TASKS = NHQ * (KV // 128)  # 48
S2_EXTRA = S2 - BLOCKS  # 144 stage-2 groups past one a CTA
KV_CTA = BLOCKS - 1  # the kv norm and the q_lora output
TL_POINTS = (
    8  # start, slice / x seen, gate / x in LDS, mix / dot, stage 1, stage 2, kv, end
)

S3_CTA0 = NA + S2_EXTRA  # CTAs with one stage-2 group
assert S1_EXTRA <= NA and NA + S2_EXTRA <= KV_CTA and S3_CTA0 + S3_TASKS <= KV_CTA


@dataclass(frozen=True)
class MlaPreBuild:
    tokens: int  # S
    nblocks: int  # AttnRes block sources (the prefix is one more)
    eps: float = 1e-5  # AttnRes norm
    out_eps: float = 1e-5  # input_layernorm
    q_eps: float = 1e-6  # q_a_layernorm
    kv_eps: float = 1e-6  # kv_a_layernorm
    write_idx: int = -1  # a block-write layer's block: the updated prefix stored there
    qn: bool = False  # stage 3: q_nope @ w_kc -> qn (the MLA absorb)
    timeline: bool = False


def scratch_layout(key: MlaPreBuild) -> dict:
    s, ns = key.tokens, key.nblocks + 1
    pairs = {
        "part": NA * s * ns * 2,
        "wgt": s * (ns + 1),
        "msq": NA * s,
        "xrdy": NA,
        "ss": (G_Q + G_KV) * s,  # a q / kv group's rows' sum of squares, a token
        "qrdy": S2_IQ0,  # a q_b group out (stage 3 polls its head's)
    }
    lay = pair_layout(pairs.items())
    end = max(o + n for o, n in lay.values())
    lay["x"] = ((end + 15) // 16 * 16, s * HIDDEN * 2)
    end = lay["x"][0] + lay["x"][1]
    lay["lat"] = ((end + 15) // 16 * 16, s * LATW * 2)
    return lay


def scratch_bytes(key: MlaPreBuild) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


_BUILDS: dict = {}
_STREAM = fx.Stream(None)


def _st_bf(ptr, i, v, cm=0):
    bo.buffer_store(fx.Float32(v).to(fx.BFloat16), rsrc(ptr), i, cache_modifier=cm)


@flyc.jit
def load_x(c):
    """Every slice's x ready -> the bf16 x rows into LDS (rows padded)."""
    s, tid = c["S"], c["tid"]
    if tid < NA:
        c["poll"]([(c["xrdy"], tid, 1)])
    gpu.barrier()
    stamp(c["on"], c["tls"], tid, 1)  # a stage-1 CTA: x seen
    gemv.rows_to_lds(tid, c["xbuf"], HIDDEN, s, c["xl"], HIDDEN + LDS_PAD, CM_DEV)
    stamp(c["on"], c["tls"], tid, 2)  # x in LDS


def s1_loads(c, g):
    """Stage-1 group ``g``'s weights for this wave (past S1: the last group's)."""
    a = c["args"]
    # past S1 (a CTA with one group): zero-range loads, no memory traffic -- the
    # barrier after the first group's dot waits on them (timeline: 9 us when
    # 162 CTAs re-read the last group)
    nbytes = (g < S1).select(fx.Int64(0xFFFFFFFF), fx.Int64(0))
    g = fx.min(g, S1 - 1)
    w = (g < S1_GATE0).select(
        a["w_qkv"],
        (g < S1_WK0).select(a["w_g"], (g < S1_WP0).select(a["w_k"], a["w_p"])),
    )
    r0 = (g < S1_GATE0).select(
        g,
        (g < S1_WK0).select(g - S1_GATE0, (g < S1_WP0).select(g - S1_WK0, g - S1_WP0)),
    )
    return gemv.loads(c["lane"], c["wave"], w, r0 * ROWS, HIDDEN, nbytes=nbytes)


@flyc.jit
def stage1(c, g, ops):
    """Rows 16 g .. of stage 1 against the LDS x: bf16 out; a latent group to LAT
    (the kv / rope part also to ``latent``), a q / kv group's squares to SS."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    a = c["args"]
    acc = gemv.mfmas(lane, wave, c["xl"], HIDDEN + LDS_PAD, HIDDEN, s, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    stamp(c["on"], c["tls"], tid, 3)  # the group's dot (the last group's, on two)
    mine = tid < ROWS * s
    r = tid % ROWS
    t = fx.min(tid // ROWS, s - 1)
    v = bf16_round(row_sum(red, r, t))
    is_lat = g < S1_GATE0
    col = g * ROWS + r
    if mine & is_lat:
        _st_bf(c["lat"], t * LATW + col, v, CM_DEV)
        if col >= QL:
            _st_bf(a["latent"], t * (KV + ROPE) + col - QL, v)
    if mine & (g >= S1_GATE0) & (g < S1_WK0):
        _st_bf(a["gate"], t * GATE + (g - S1_GATE0) * ROWS + r, v)
    if mine & (g >= S1_WK0) & (g < S1_WP0):
        _st_bf(a["idx_k"], t * IK + (g - S1_WK0) * ROWS + r, v)
    if mine & (g >= S1_WP0):
        _st_bf(a["idx_w"], t * IW + (g - S1_WP0) * ROWS + r, v)
    # a q / kv group: its 16 rows' squares a token, summed in row order, after
    # the rows are out (consumers read LAT once the sum is there)
    gpu.barrier()
    if mine:
        fx.ptr_store(v * v, red + tid)
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if (tid < s) & (g < G_Q + G_KV):
        tot = fx.ptr_load(red + tid * ROWS)
        for rr in range_constexpr(1, ROWS):
            tot = tot + fx.ptr_load(red + (tid * ROWS + rr))
        c["put"](c["ss"], g * s + tid, tot)
    gpu.barrier()


def s2_loads(c, h):
    """Stage-2 group ``h``'s weights (q_b, then wq_b; past S2: the last group's)."""
    a = c["args"]
    nbytes = (h < S2).select(fx.Int64(0xFFFFFFFF), fx.Int64(0))  # past S2: no traffic
    h = fx.min(h, S2 - 1)
    w = (h < S2_IQ0).select(a["w_qb"], a["w_iq"])
    r0 = (h < S2_IQ0).select(h, h - S2_IQ0)
    return gemv.loads(c["lane"], c["wave"], w, r0 * ROWS, QL, nbytes=nbytes)


@flyc.jit
def load_qlora(c):
    """The 96 q groups' squares (rank order of groups) -> rstd a token, then the
    normed q latent, bf16((x rstd) w) as aiter's fused_qk_rmsnorm, into LDS."""
    s, tid, red = c["S"], c["tid"], c["red"]
    a = c["args"]
    n = G_Q * s
    per = (n + THREADS - 1) // THREADS
    got = c["poll"](
        [(c["ss"], fx.min(tid + THREADS * i, n - 1), 1) for i in range(per)]
    )
    for i in range_constexpr(per):
        e = tid + THREADS * i
        if e < n:
            fx.ptr_store(f32_of(got[i][0]), red + e)  # red[g s + t]
    gpu.barrier()
    if tid < s:
        tot = fx.ptr_load(red + tid)
        for gg in range_constexpr(1, G_Q):
            tot = tot + fx.ptr_load(red + (gg * s + tid))
        fx.ptr_store(
            hw_rsq(tot / fx.Float32(QL) + fx.Float32(c["q_eps"])),
            red + (2048 - 8 + tid),
        )
    gpu.barrier()
    xrow = QL + LDS_PAD
    quads = s * QL // 8  # 4 words (8 bf16) a lane
    nq = (quads + THREADS - 1) // THREADS
    lw, nws = [], []
    for i in range_constexpr(nq):  # every load issued before the first use
        e = fx.min(tid + THREADS * i, quads - 1)
        t = e // (QL // 8)
        k = e % (QL // 8) * 8
        lw.append(
            fx.Vector(
                bo.buffer_load(
                    rsrc(c["lat"]),
                    (t * LATW + k) // 2,
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=CM_DEV,
                )
            )
        )
        nws.append(
            fx.Vector(bo.buffer_load(rsrc(a["q_nw"]), k // 2, vec_width=4, dtype=T.i32))
        )
    for i in range_constexpr(nq):
        e = fx.min(tid + THREADS * i, quads - 1)
        t = e // (QL // 8)
        k = e % (QL // 8) * 8
        rs = fx.ptr_load(red + (2048 - 8 + t))
        out = []
        for d in range_constexpr(4):
            w, nw = fx.Int32(lw[i][d]), fx.Int32(nws[i][d])
            y0 = bf16_round((bf_lo(w) * rs) * bf_lo(nw))
            y1 = bf16_round((bf_hi(w) * rs) * bf_hi(nw))
            out.append(
                fx.Vector.from_elements([y0, y1], fx.Float32)
                .to(fx.BFloat16)
                .bitcast(fx.Int32)[0]
            )
        quad = fx.Vector.from_elements(out, fx.Int32)
        if tid + THREADS * i < quads:
            fx.ptr_store(quad, c["xl"] + (t * xrow + k) // 2)
            if const_expr(c["qlora_out"]):
                bo.buffer_store(quad, rsrc(a["q_lora"]), (t * QL + k) // 2)
    gpu.barrier()


@flyc.jit
def stage2(c, h, ops):
    """Rows 16 h .. of q_b | wq_b against the normed q latent in LDS -> bf16 out."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    a = c["args"]
    acc = gemv.mfmas(lane, wave, c["xl"], QL + LDS_PAD, QL, s, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    mine = (tid < ROWS * s) & (h < S2)
    r = tid % ROWS
    t = fx.min(tid // ROWS, s - 1)
    v = bf16_round(row_sum(red, r, t))
    if mine & (h < S2_IQ0):
        _st_bf(a["q"], t * QB + h * ROWS + r, v, CM_DEV if c["qn"] else 0)
    if mine & (h >= S2_IQ0):
        _st_bf(a["idx_q"], t * IQ + (h - S2_IQ0) * ROWS + r, v)
    if const_expr(c["qn"]):
        # the group's rows out before its flag: stage 3 reads them from q
        rocdl.s_waitcnt(vmcnt=0)
        gpu.barrier()
        if (tid == 0) & (h < S2_IQ0):
            c["put"](c["qrdy"], h, fx.Float32(1.0))
    gpu.barrier()


@flyc.jit
def stage3(c, task):
    """q_nope @ w_kc (the MLA absorb) for head task // 4, kv_lora columns
    128 (task % 4) ..: the head's 8 nope q_b groups (stage 2's flags) into LDS,
    a 16-column tile a wave over K = 128 (fp32), bf16 -> qn [S, 12, 512]."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    a = c["args"]
    h = task // 4
    blk = task % 4
    w0 = fx.Int32(0)  # every wave the whole K: its own 16 rows (columns of the out)
    ops = gemv.loads(lane, w0, a["w_kc"], h * KV + blk * 128 + wave * ROWS, QN)
    c["poll"]([(c["qrdy"], h * (QHD // ROWS) + tid % (QN // ROWS), 1)])
    gpu.barrier()
    gemv.rows_to_lds(
        tid, a["q"], QN, s, c["xl"], QN + LDS_PAD, CM_DEV, ldk=QB, k0=h * QHD
    )
    acc = gemv.mfmas(lane, w0, c["xl"], QN + LDS_PAD, QN, s, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    n = WAVES * ROWS * s
    for i in range_constexpr((n + THREADS - 1) // THREADS):
        e = fx.min(tid + THREADS * i, n - 1)
        w = e // (ROWS * s)
        rem = e % (ROWS * s)
        r = rem % ROWS
        t = rem // ROWS
        # row_sum's indexing, for the one wave that owns the row
        v = bf16_round(fx.ptr_load(red + ((w * 64 + 16 * (r // 4) + t) * 4 + r % 4)))
        if tid + THREADS * i < n:
            _st_bf(a["qn"], t * (NHQ * KV) + h * KV + blk * 128 + w * ROWS + r, v)
    gpu.barrier()


@flyc.jit
def stage_kv(c):
    """kv_a_layernorm: the 32 kv groups' squares -> rstd -> k_nope (bf16)."""
    s, tid, red = c["S"], c["tid"], c["red"]
    a = c["args"]
    n = G_KV * s
    got = c["poll"]([(c["ss"], G_Q * s + fx.min(tid, n - 1), 1)])
    if tid < n:
        fx.ptr_store(f32_of(got[0][0]), red + tid)
    gpu.barrier()
    if tid < s:
        tot = fx.ptr_load(red + tid)
        for gg in range_constexpr(1, G_KV):
            tot = tot + fx.ptr_load(red + (gg * s + tid))
        fx.ptr_store(
            hw_rsq(tot / fx.Float32(KV) + fx.Float32(c["kv_eps"])),
            red + (2048 - 16 + tid),
        )
    gpu.barrier()
    words = s * KV // 2
    for i in range_constexpr((words + THREADS - 1) // THREADS):
        e = fx.min(tid + THREADS * i, words - 1)
        t = e // (KV // 2)
        k = e % (KV // 2) * 2
        w = fx.Int32(
            bo.buffer_load(
                rsrc(c["lat"]),
                (t * LATW + QL + k) // 2,
                vec_width=1,
                dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        )
        nw = fx.Int32(
            bo.buffer_load(rsrc(a["kv_nw"]), k // 2, vec_width=1, dtype=T.i32)
        )
        rs = fx.ptr_load(red + (2048 - 16 + t))
        y0 = bf16_round((bf_lo(w) * rs) * bf_lo(nw))
        y1 = bf16_round((bf_hi(w) * rs) * bf_hi(nw))
        pair = (
            fx.Vector.from_elements([y0, y1], fx.Float32)
            .to(fx.BFloat16)
            .bitcast(fx.Int32)[0]
        )
        if tid + THREADS * i < words:
            bo.buffer_store(pair, rsrc(a["k_nope"]), (t * KV + k) // 2)
    gpu.barrier()


ARGS = (
    "prefix", "blocks", "ares_nw", "ares_qk", "in_nw", "w_qkv", "w_g", "w_k", "w_p",
    "w_qb", "w_iq", "q_nw", "kv_nw", "q", "latent", "k_nope", "q_lora", "gate",
    "idx_q", "idx_k", "idx_w", "w_kc", "qn",
)  # fmt: skip


def build(key: MlaPreBuild):
    if key in _BUILDS:
        return _BUILDS[key]
    s = key.tokens
    ns = key.nblocks + 1
    lay = scratch_layout(key)
    assert s * (HIDDEN + LDS_PAD) * 2 <= 136 * 1024 and s <= 8

    @fx.struct
    class AttnLds:
        vals: fx.Array[fx.Float32, s * ns * COLS, 16]

    @fx.struct
    class GemvLds:
        xl: fx.Array[fx.Int32, s * (HIDDEN + LDS_PAD) // 2, 16]

    @fx.union
    class RoleLds:
        attn: AttnLds
        gemv: GemvLds

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        tls: fx.Array[fx.Int64, TL_POINTS, 16]

    keyed = key_tuple(key, SOURCES)
    name = (
        f"k3_mono_mla_pre_s{s}_nb{key.nblocks}_w{int(key.write_idx >= 0)}_t{int(key.timeline)}"
        + ("_qn" if key.qn else "")
    )

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def mla_pre(
        prefix: Int64, blocks: Int64, blk_sm: Int32, blk_sr: Int32, ares_nw: Int64,
        ares_qk: Int64, in_nw: Int64, w_qkv: Int64, w_g: Int64, w_k: Int64, w_p: Int64,
        w_qb: Int64, w_iq: Int64, q_nw: Int64, kv_nw: Int64, q: Int64, latent: Int64,
        k_nope: Int64, q_lora: Int64, gate: Int64, idx_q: Int64, idx_k: Int64,
        idx_w: Int64, w_kc: Int64, qn: Int64, scratch: Int64, layer: Int32,
        epoch: Int64, tl: Int64,
    ):  # fmt: skip
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        alloc = fx.SharedAllocator()
        lds = alloc.allocate(Smem).peek()
        role = alloc.allocate(RoleLds)
        al, gl = role.attn.peek(), role.gemv.peek()
        mb = Mailbox(step_tag(epoch, layer) - 1)
        c = {
            "S": s, "NS": ns, "NB": key.nblocks, "tid": tid, "bid": bid,
            "lane": tid % 64, "wave": tid // 64, "delta": False, "debug": False,
            "eps": key.eps, "out_eps": key.out_eps, "q_eps": key.q_eps,
            "kv_eps": key.kv_eps, "write_idx": key.write_idx, "gate_local": False,
            "qlora_out": False, "qn": key.qn, "put": mb.put, "poll": mb.poll, "red": lds.red.ptr,
            "vals": al.vals.ptr, "xl": gl.xl.ptr,
            "xbuf": scratch + fx.Int64(lay["x"][0]),
            "lat": scratch + fx.Int64(lay["lat"][0]),
            "args": {
                "prefix": prefix, "delta": 0, "blocks": blocks, "blk_sm": blk_sm,
                "blk_sr": blk_sr, "ares_nw": ares_nw, "ares_qk": ares_qk, "in_nw": in_nw,
                "x_dbg": 0, "w_qkv": w_qkv, "w_g": w_g, "w_k": w_k, "w_p": w_p,
                "w_qb": w_qb, "w_iq": w_iq, "q_nw": q_nw, "kv_nw": kv_nw, "q": q,
                "latent": latent, "k_nope": k_nope, "q_lora": q_lora, "gate": gate,
                "idx_q": idx_q, "idx_k": idx_k, "idx_w": idx_w, "w_kc": w_kc, "qn": qn,
            },
        }  # fmt: skip
        for region in ("part", "wgt", "msq", "xrdy", "ss", "qrdy"):
            c[region] = sreg(scratch, lay[region][0], region)
        on, tls = key.timeline, lds.tls.ptr
        c["on"], c["tls"] = on, tls
        stamp_begin(on, tls, tid, TL_POINTS)
        h0 = bid
        # stage 2's second round on stage-1 CTAs (one group each); the seam CTAs
        # took stage 1's extra groups
        h1 = ((bid >= NA) & (bid < NA + S2_EXTRA)).select(
            bid - NA + BLOCKS, fx.Int32(S2)
        )
        if bid < NA:
            stage_slice(c, bid)
            stamp(on, tls, tid, 1)
            if bid < s:
                stage_gate(c, bid)
            stamp(on, tls, tid, 2)
            stage_mix(c, bid)
            stamp(on, tls, tid, 3)
            # stage 1's 38 groups past the 200 stage-1 CTAs, on seam CTAs idle
            # until stage 2's polls
            if bid < S1_EXTRA:
                ge = bid + NCTA1
                opse = s1_loads(c, ge)
                load_x(c)
                stage1(c, ge, opse)
        if bid >= NA:
            # stage 1: the first group's weights at launch, the second's after x
            g0 = bid - NA
            ops = s1_loads(c, g0)
            load_x(c)
            stage1(c, g0, ops)
            stamp(on, tls, tid, 4)
        # stage 2 on every CTA, its weights issued once this CTA's polls are done
        ops2 = s2_loads(c, h0)
        ops3 = s2_loads(c, h1)  # the second round's too, ahead of the polls
        load_qlora(c)
        stage2(c, h0, ops2)
        if h1 < S2:
            stage2(c, h1, ops3)
        if const_expr(key.qn):
            if (bid >= S3_CTA0) & (bid < S3_CTA0 + S3_TASKS):
                stage3(c, bid - S3_CTA0)
        stamp(on, tls, tid, 5)
        if bid == KV_CTA:
            stage_kv(c)
        stamp(on, tls, tid, 7)
        stamp_flush(on, tls, tl, tid, bid, TL_POINTS)

    @flyc.jit
    def launch(
        prefix: Int64, blocks: Int64, blk_sm: Int32, blk_sr: Int32, ares_nw: Int64,
        ares_qk: Int64, in_nw: Int64, w_qkv: Int64, w_g: Int64, w_k: Int64, w_p: Int64,
        w_qb: Int64, w_iq: Int64, q_nw: Int64, kv_nw: Int64, q: Int64, latent: Int64,
        k_nope: Int64, q_lora: Int64, gate: Int64, idx_q: Int64, idx_k: Int64,
        idx_w: Int64, w_kc: Int64, qn: Int64, scratch: Int64, layer: Int32,
        epoch: Int64, tl: Int64, stream: fx.Stream = _STREAM,
    ):  # fmt: skip
        _ = keyed
        mla_pre(
            prefix, blocks, blk_sm, blk_sr, ares_nw, ares_qk, in_nw, w_qkv, w_g, w_k,
            w_p, w_qb, w_iq, q_nw, kv_nw, q, latent, k_nope, q_lora, gate, idx_q, idx_k,
            idx_w, w_kc, qn, scratch, layer, epoch, tl,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    _BUILDS[key] = launch
    return launch


def ptr(t):
    return 0 if t is None else t.data_ptr()


def mla_pre_launch(
    key: MlaPreBuild, *, prefix, blocks, w, out, scratch, layer, epoch, tl=None
):
    """Host launcher. ``w``: ares_nw, ares_qk, in_nw, w_qkv [2112, 7168], w_g
    [1536, 7168], w_k [128, 7168], w_p [32, 7168], w_qb [2304, 1536], w_iq
    [4096, 1536], q_nw [1536], kv_nw [512] (bf16, contiguous); ``out``: q,
    latent, k_nope, gate, idx_q, idx_k, idx_w (and q_lora, unused here) as the
    module docstring shapes them. ``prefix`` is read, not written, unless the
    layer is a block-write layer (its bank row)."""
    # A block-write layer stores its row at block write_idx; a bank without that
    # block puts the last token's row past the allocation.
    assert key.write_idx < blocks.shape[1], (
        f"mla_pre layer {layer} writes bank block {key.write_idx}, the bank has "
        f"{blocks.shape[1]}"
    )
    shapes = {
        "w_qkv": (LATW, HIDDEN), "w_g": (GATE, HIDDEN), "w_k": (IK, HIDDEN),
        "w_p": (IW, HIDDEN), "w_qb": (QB, QL), "w_iq": (IQ, QL),
    }  # fmt: skip
    if key.qn:  # w_kc as [heads x kv_lora, nope] (K contiguous), qn [S, heads, kv_lora]
        shapes["w_kc"] = (NHQ * KV, QN)
        assert (
            tuple(out["qn"].shape) == (key.tokens, NHQ, KV)
            and out["qn"].is_contiguous()
        )
    for n, shp in shapes.items():
        t = w[n]
        assert (
            tuple(t.shape) == shp and t.is_contiguous() and t.dtype == torch.bfloat16
        ), (n, t.shape, t.dtype)
    f = build(key)
    f(
        prefix.data_ptr(), blocks.data_ptr(), blocks.stride(0), blocks.stride(1),
        w["ares_nw"].data_ptr(), w["ares_qk"].data_ptr(), w["in_nw"].data_ptr(),
        w["w_qkv"].data_ptr(), w["w_g"].data_ptr(), w["w_k"].data_ptr(), w["w_p"].data_ptr(),
        w["w_qb"].data_ptr(), w["w_iq"].data_ptr(), w["q_nw"].data_ptr(), w["kv_nw"].data_ptr(),
        out["q"].data_ptr(), out["latent"].data_ptr(), out["k_nope"].data_ptr(),
        ptr(out.get("q_lora")), out["gate"].data_ptr(), out["idx_q"].data_ptr(),
        out["idx_k"].data_ptr(), out["idx_w"].data_ptr(), ptr(w.get("w_kc")),
        ptr(out.get("qn")), scratch.data_ptr(), layer,
        epoch.data_ptr(), ptr(tl), stream=torch.cuda.current_stream(),
    )  # fmt: skip
