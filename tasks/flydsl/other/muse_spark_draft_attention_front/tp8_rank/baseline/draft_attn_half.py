# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 draft layer's attention half around D-attn, as two launches on a
rank of a TP group, for S = ``tokens`` rows (a block of 8 consecutive rows a
request)::

  D-pre : res' = first ? x : bf16(res + x)         (the input norm's fused add)
          h    = RMSNorm(res') w_in                 (eps 1e-5, bf16)
          delta = h W_kp^T                           (2048: the attention conv's coefficients, both sides -> ``coef``)
          hc   = conv0(h)                            (side 0)
          qkv  = hc W_qkv^T                          (1280 rows a rank: 8 q heads, k, v of 128)
          q, k : per-head RMSNorm (weights, eps 1e-5) -> bf16 -> neox RoPE from the table
          q -> ``q`` bf16 [S, 1024]; k, v -> the layer's fp8 cache at the rows' slots
  D-post: d   = o W_o^T                              (this rank's partial, 8192)
          d   = sum over the ranks (bf16 pairs through the peer regions, rank order)
          attn_out = conv1(d) with side 1's coefficients (bf16 [S, 8192])

The stages, mailboxes and GEMV machinery are D-MLP's (``draft_mlp.py``).

AKA: vendored from sgl-amd-mono @ 52486703 (sgl_amd_mono/muse_spark/gfx950/draft_attn_half.py),
keeping only D-pre (its scratch layout, LDS, stages, ``DPreBuild``, ``_common`` and ``build_dpre``)
unchanged apart from these: package-relative imports; ``DPreBuild`` a frozen dataclass instead of a
``msgspec.Struct``; ``compat.install()`` not called (the modules call compat.py's llvm-dialect bridge
directly); the transitive source digest ``plan.kernel_sources`` replaced by a constant tag (FlyDSL
hashes the helpers of a kernel's own directory, which in this flat package are all of them). D-post
(the o projection, the TP all-reduce and conv1) is not part of this task.
"""

import dataclasses

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from . import buffer_ops as bo
from . import guard
from .config import MAX_TOKENS
from .device import TRACE_LAYERS, TRACE_SLOTS, stamp, step_tag
from .draft_mlp import (
    C_ROWS,
    C_TILES,
    COEF_ROWS,
    EPS,
    GROUP,
    K_PART,
    K_PARTS,
    K_PER_WAVE,
    K_STEP,
    ROWS,
    TAPS,
    H,
    gemv_tiles,
    lds_b,
    stage_coef,
    stage_coef_reduce,
    stage_conv0,
    w_loads,
)
from .epoch import kernel_symbol
from .layout import RED_FLOATS, TILE
from .mx import clamp_fp8
from .ops import bf2_f32, bf16_round, butterfly, fp8_pack4, hw_rsq, rsrc
from .plan import BLOCKS, THREADS, WAVES, key_tuple, pair_layout
from .sync import Mailbox, sreg

SOURCES = "aka-flat"
_STREAM = fx.Stream(None)

# the attention geometry (a rank of TP 8)
HEAD_DIM = 128
Q_HEADS = 8
Q_K = Q_HEADS * HEAD_DIM  # 1024
QKV_ROWS = Q_K + 2 * HEAD_DIM  # 1280: q heads, k, v
QKV_HEADS = Q_HEADS + 2  # head-rows a token of the qkv reduce
A_ROWS, A_TILES = 80, 5  # a qkv GEMV task's rows: 16 row groups x 16 K parts
O_ROWS, O_TILES = 32, 2  # an o GEMV task's rows: 256 tasks x full K (1024)
O_KPARTS = Q_K // K_PART  # 2 K parts of 512 a lane's wave split: the task sums them
assert (
    A_TILES * ROWS == A_ROWS
    and 16 * A_ROWS == QKV_ROWS
    and O_ROWS * BLOCKS == H
    and O_TILES * ROWS == O_ROWS
)
assert HEAD_DIM == 2 * 64


def dp_scratch_layout() -> dict:
    """D-pre's scratch regions (pairs): the K parts' sums of squares, the coefficient
    partials, the qkv partials, the trace."""
    return pair_layout(
        [
            ("dp_ss", K_PARTS * MAX_TOKENS),
            ("dp_cpart", K_PARTS * MAX_TOKENS * COEF_ROWS),
            ("dp_coef", MAX_TOKENS * COEF_ROWS),
            ("dp_apart", K_PARTS * MAX_TOKENS * QKV_ROWS),
            ("dp_trace", TRACE_LAYERS * BLOCKS * TRACE_SLOTS),
        ],
        align=256,
    )



def dp_lds(tokens: int):
    S = tokens

    @fx.struct
    class Lds:
        red: fx.Array[fx.Float32, C_TILES * RED_FLOATS, 16]
        hbuf: fx.Array[fx.BFloat16, S * K_PART, 16]
        ssl: fx.Array[fx.Float32, S * K_PARTS, 16]
        nf: fx.Array[fx.Float32, S, 16]
        cf: fx.Array[fx.Float32, S * TAPS * (K_PART // GROUP), 16]
        hrow: fx.Array[fx.Float32, HEAD_DIM, 16]  # a head-row of the qkv reduce
        hss: fx.Array[fx.Float32, WAVES, 16]

    return Lds



def _pick(pair, idx):
    """Element ``idx % 2`` of a (lo, hi) bf16 pair for a dynamic ``idx``."""
    return (idx % 2 == 0).select(pair[0], pair[1])


def _cs(r_cs, idx):
    """Element ``idx`` of the rotary table (fp32 [positions, 128]: cos then sin)."""
    return fx.Float32(bo.buffer_load(r_cs, idx, vec_width=1, dtype=T.f32))


# ---------------------------------------------------------------- D-pre stages
@flyc.jit
def stage_norm_in(c, rg, kp):
    """res' = first ? x : bf16(res + x) on this K part (row group 0 writes it out), the
    K part's sums of squares published (row group 0), every part's polled; h =
    RMSNorm(res') w_in -> ``hbuf``."""
    S, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    put, poll, hbuf, ssl, nf = c["put"], c["poll"], c["hbuf"], c["ssl"], c["nf"]
    a = c["args"]
    first = a["first"] != 0
    ch = kp * K_PART + 8 * lane
    r_res, r_x, r_out = rsrc(a["res"]), rsrc(a["x"]), rsrc(a["res_out"])
    wv = fx.Vector(
        bo.buffer_load(rsrc(a["w_in"]), ch // 2, vec_width=4, dtype=T.i32)
    ).bitcast(fx.BFloat16)
    TPW = S // WAVES
    rp = []
    for j in range_constexpr(TPW):
        t = wave + WAVES * j
        xv = fx.Vector(
            bo.buffer_load(r_x, (t * H + ch) // 2, vec_width=4, dtype=T.i32)
        ).bitcast(fx.BFloat16)
        rv = fx.Vector(
            bo.buffer_load(r_res, (t * H + ch) // 2, vec_width=4, dtype=T.i32)
        ).bitcast(fx.BFloat16)
        r8 = [
            first.select(
                fx.Float32(xv[e]), bf16_round(fx.Float32(rv[e]) + fx.Float32(xv[e]))
            )
            for e in range(8)
        ]
        ss = r8[0] * r8[0]
        for e in range_constexpr(1, 8):
            ss = ss + r8[e] * r8[e]
        ss = butterfly(ss, (1, 2, 4, 8, 16, 32))
        if rg == 0:
            bo.buffer_store(fx.Vector.from_elements(r8, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32), r_out, (t * H + ch) // 2)  # fmt: skip
            if lane == 0:
                put(c["ss"], kp * MAX_TOKENS + t, ss)
        rp.append(r8)
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
def stage_qkv(c, rg, kp):
    """The qkv GEMV task (``rg``, ``kp``): rows 80 rg .. of W_qkv over K part kp
    against hc; fp32 partials to ``dp_apart`` [kp][token][row]."""
    S, lane, wave, put = c["S"], c["lane"], c["wave"], c["put"]
    apart = c["apart"]
    a = c["args"]

    def tok_of(t):
        return fx.min(t * TILE + lane % 16, S - 1)

    def k_of(kc):
        return wave * K_PER_WAVE + kc * K_STEP + 8 * (lane // 16)

    def publish(r, tk, v):
        put(apart, (kp * MAX_TOKENS + tk) * QKV_ROWS + rg * A_ROWS + r, v)

    w = w_loads(c, a["w_qkv"], H, rg * A_ROWS, A_TILES, kp)
    gemv_tiles(c, w, lds_b(c, c["hbuf"], tok_of, k_of), A_TILES, publish)


@flyc.jit
def stage_heads(c, bid):
    """Head-rows (token, head) 10 S in all, one a CTA a pass: the K parts' qkv partials
    summed in part order (a dim a thread, 128 of them), the head's RMSNorm with its
    weight, bf16, neox RoPE from the table at the row's position; q heads -> ``q``
    (bf16), k -> the fp8 cache at the row's slot, v -> the cache unrotated."""
    S, tid, lane, wave, poll = c["S"], c["tid"], c["lane"], c["wave"], c["poll"]
    apart, hrow, hss = c["apart"], c["hrow"], c["hss"]
    a = c["args"]
    r_cs = rsrc(a["cos_sin"])
    r_q, r_k, r_v = rsrc(a["q"]), rsrc(a["k_cache"]), rsrc(a["v_cache"])
    N = QKV_HEADS * S
    for p in range_constexpr(-(-N // BLOCKS)):
        job = bid + p * BLOCKS
        live = job < N
        jb = fx.min(job, N - 1)
        t, hd = jb // QKV_HEADS, jb % QKV_HEADS  # token, head-row (0..7 q, 8 k, 9 v)
        row0 = hd * HEAD_DIM  # the head's first qkv row
        # the dim a thread (threads 128 .. idle): the K parts' partials summed in order
        d = fx.min(tid, HEAD_DIM - 1)
        got = poll(
            [
                (apart, (kp * MAX_TOKENS + t) * QKV_ROWS + row0 + d, 1)
                for kp in range(K_PARTS)
            ],
            batch=K_PARTS,
        )
        v = got[0][0].bitcast(fx.Float32)
        for kp in range_constexpr(1, K_PARTS):
            v = v + got[kp][0].bitcast(fx.Float32)
        v = bf16_round(v)  # the GEMM's bf16 output
        is_v = hd == QKV_HEADS - 1
        is_k = hd == QKV_HEADS - 2
        # the head's sum of squares (q and k): wave partials through LDS
        sq = (tid < HEAD_DIM).select(v * v, fx.Float32(0.0))
        sq = butterfly(sq, (1, 2, 4, 8, 16, 32))
        if lane == 0:
            fx.ptr_store(sq, hss + wave)
        gpu.barrier()
        tot = fx.ptr_load(hss)
        for w in range_constexpr(1, WAVES):
            tot = tot + fx.ptr_load(hss + w)
        inv = hw_rsq(tot / float(HEAD_DIM) + EPS)
        wn = fx.Float32(_pick(bf2_f32(fx.Int32(bo.buffer_load(rsrc(is_k.select(a["k_norm"], a["q_norm"])), d // 2, vec_width=1, dtype=T.i32))), d % 2))  # fmt: skip
        xn = bf16_round(v * inv * wn)  # the normed value, bf16 (q and k); v keeps v
        val = is_v.select(v, xn)
        if tid < HEAD_DIM:
            fx.ptr_store(val, hrow + d)
        gpu.barrier()
        # neox RoPE: dims (d, d + 64) rotate by the position's angle d % 64
        pos = fx.Int32(
            bo.buffer_load(
                rsrc(a["positions"]),
                (a["pos64"] != 0).select(2 * t, t),
                vec_width=1,
                dtype=T.i32,
            )
        )
        half = d % 64
        cs = _cs(r_cs, pos * HEAD_DIM + half)
        sn = _cs(r_cs, pos * HEAD_DIM + 64 + half)
        x1 = fx.ptr_load(hrow + half)
        x2 = fx.ptr_load(hrow + (half + 64))
        rot = (d < 64).select(x1 * cs - x2 * sn, x2 * cs + x1 * sn)
        out = is_v.select(val, bf16_round(rot))
        if live & (tid < HEAD_DIM):
            if hd < Q_HEADS:
                # q: bf16 pairs (even dims store the pair)
                if (d % 2) == 0:
                    nxt = (d < 64).select(
                        bf16_round(
                            fx.ptr_load(hrow + (half + 1))
                            * _cs(r_cs, pos * HEAD_DIM + half + 1)
                            - fx.ptr_load(hrow + (half + 65))
                            * _cs(r_cs, pos * HEAD_DIM + 64 + half + 1)
                        ),  # fmt: skip
                        bf16_round(
                            fx.ptr_load(hrow + (half + 65))
                            * _cs(r_cs, pos * HEAD_DIM + half + 1)
                            + fx.ptr_load(hrow + (half + 1))
                            * _cs(r_cs, pos * HEAD_DIM + 64 + half + 1)
                        ),  # fmt: skip
                    )
                    bo.buffer_store(fx.Vector.from_elements([out, nxt], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0], r_q, (t * Q_K + row0 + d) // 2)  # fmt: skip
            else:
                # k / v: fp8, four dims a dword, at the row's cache slot
                slot = fx.Int32(
                    bo.buffer_load(
                        rsrc(a["slots"]),
                        (a["slot64"] != 0).select(2 * t, t),
                        vec_width=1,
                        dtype=T.i32,
                    )
                )
                if (d % 4) == 0:
                    vals = [out]
                    for e in range_constexpr(1, 4):
                        de = d + e
                        he = de % 64
                        xe1 = fx.ptr_load(hrow + he)
                        xe2 = fx.ptr_load(hrow + (he + 64))
                        cse = _cs(r_cs, pos * HEAD_DIM + he)
                        sne = _cs(r_cs, pos * HEAD_DIM + 64 + he)
                        rote = (de < 64).select(
                            xe1 * cse - xe2 * sne, xe2 * cse + xe1 * sne
                        )
                        vals.append(
                            is_v.select(fx.ptr_load(hrow + de), bf16_round(rote))
                        )
                    packed = fp8_pack4(
                        clamp_fp8(vals[0]),
                        clamp_fp8(vals[1]),
                        clamp_fp8(vals[2]),
                        clamp_fp8(vals[3]),
                    )
                    bo.buffer_store(
                        packed,
                        rsrc(is_v.select(a["v_cache"], a["k_cache"])),
                        (slot * HEAD_DIM + d) // 4,
                    )
        gpu.barrier()  # ``hrow`` / ``hss`` are reused by the next head-row


@flyc.jit
def run_dpre(c):
    bid = c["bid"]
    rg, kp = bid // K_PARTS, bid % K_PARTS
    stamp(c, 0)
    w0 = w_loads(c, c["args"]["w_kp"], H, rg * C_ROWS, C_TILES, kp)
    stage_norm_in(c, rg, kp)
    stamp(c, 1)
    stage_coef(c, rg, kp, w0)
    stamp(c, 2)
    stage_coef_reduce(c, bid)
    stamp(c, 3)
    stage_conv0(c, kp)
    stamp(c, 4)
    stage_qkv(c, rg, kp)
    stamp(c, 5)
    stage_heads(c, bid)
    stamp(c, 6)



# ---------------------------------------------------------------- the launches
@dataclasses.dataclass(frozen=True)
class DPreBuild:
    tokens: int
    fault_site: int = 0
    fault_after: int = 0
    trace: bool = False



def _common(key, lds_members: dict, args: dict, tag, extra: dict, layer=None) -> dict:
    tid = fx.thread_idx.x
    mb = Mailbox(tag)
    c = {
        "S": key.tokens, "tid": tid, "bid": fx.block_idx.x, "lane": tid % 64, "wave": tid // 64,
        "put": mb.put, "put_bf": mb.put_bf, "poll": mb.poll, "args": args, "trace_on": key.trace,
        "layer": layer,
    }  # fmt: skip
    c.update(lds_members)
    c.update(extra)
    return c


def build_dpre(key: DPreBuild):
    """``@flyc.jit`` launcher of D-pre for ``key.tokens`` rows. Arguments: ``x`` /
    ``res`` bf16 [S, H] (``first``: res is x), ``w_in`` bf16 [H], ``w_kp`` bf16 [2048,
    H], ``base`` bf16 [2, 2, H], ``w_qkv`` bf16 [1280, H], ``q_norm`` / ``k_norm`` bf16
    [128], ``cos_sin`` fp32 [*, 128], ``positions`` (int32 / int64 [S]; ``pos64``),
    ``slots`` (int32 / int64 [S]: the rows' cache slots; ``slot64``), ``k_cache`` /
    ``v_cache`` fp8 [slots, 1, 128], ``res_out`` bf16 [S, H], ``q`` bf16 [S, 1024],
    ``coef`` fp32 [S, 2048], ``scratch``, ``epoch``, ``layer``."""
    S = key.tokens
    assert S % WAVES == 0 and 1 <= S <= MAX_TOKENS
    name = kernel_symbol("muse_spark_dpre", s=S, f=key.fault_site, t=int(key.trace))
    keyed = key_tuple(key, SOURCES)
    fault_site, fault_after = key.fault_site, key.fault_after
    layout = dp_scratch_layout()
    Lds = dp_lds(S)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def dpre(
        x: Int64, res: Int64, first: Int32, w_in: Int64, w_kp: Int64, base: Int64, w_qkv: Int64, q_norm: Int64, k_norm: Int64,
        cos_sin: Int64, positions: Int64, pos64: Int32, slots: Int64, slot64: Int32, k_cache: Int64, v_cache: Int64,
        res_out: Int64, q: Int64, coef: Int64, scratch: Int64, epoch: Int64, layer: Int32,
    ):  # fmt: skip
        with guard.scope(
            guard.Guard(epoch, fault_site=fault_site or None, fault_after=fault_after)
        ):
            lds = fx.SharedAllocator().allocate(Lds).peek()
            args = {
                "x": x, "res": res, "first": first, "w_in": w_in, "w_kp": w_kp, "base": base, "w_qkv": w_qkv,
                "q_norm": q_norm, "k_norm": k_norm, "cos_sin": cos_sin, "positions": positions, "pos64": pos64,
                "slots": slots, "slot64": slot64, "k_cache": k_cache, "v_cache": v_cache, "res_out": res_out,
                "q": q, "coef_out": coef,
            }  # fmt: skip
            members = {"red": lds.red.ptr, "hbuf": lds.hbuf.ptr, "ssl": lds.ssl.ptr, "nf": lds.nf.ptr, "cf": lds.cf.ptr, "hrow": lds.hrow.ptr, "hss": lds.hss.ptr}  # fmt: skip
            c = _common(key, members, args, step_tag(epoch, layer), {}, layer=layer)
            for region, (off, _) in layout.items():
                c[region[3:]] = sreg(scratch, off, region)  # "dp_cpart" -> c["cpart"]
            run_dpre(c)

    @flyc.jit
    def launch(
        x: Int64, res: Int64, first: Int32, w_in: Int64, w_kp: Int64, base: Int64, w_qkv: Int64, q_norm: Int64, k_norm: Int64,
        cos_sin: Int64, positions: Int64, pos64: Int32, slots: Int64, slot64: Int32, k_cache: Int64, v_cache: Int64,
        res_out: Int64, q: Int64, coef: Int64, scratch: Int64, epoch: Int64, layer: Int32, stream: fx.Stream = _STREAM,
    ):  # fmt: skip
        _ = keyed
        dpre(
            x, res, first, w_in, w_kp, base, w_qkv, q_norm, k_norm, cos_sin, positions, pos64, slots, slot64, k_cache,
            v_cache, res_out, q, coef, scratch, epoch, layer,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    return launch

