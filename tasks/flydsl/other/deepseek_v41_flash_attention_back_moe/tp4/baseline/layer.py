# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Vendored from sgl-amd-mono @ 52486703 (sgl_amd_mono/deepseek_v41/gfx950/layer.py). Modified for
# AgentKernelArena: imports are package-relative; MonoBuild is a frozen dataclass instead of a
# msgspec Struct; the build key closes over a constant source tag instead of
# plan.kernel_sources (which reads source files); scratch_layout keeps its running offset in a
# list instead of a nonlocal; the index-source index query (attention/index_query.py, a NumPy
# importer) is not vendored, so its iq branches (MonoBuild.iq, which these tasks never set)
# are removed.
"""Build K1 (attention seam and front), K2 (attention back, seam, MoE and TP
all-reduces) and the FFN launch (seam, MoE and TP all-reduce)."""

import dataclasses

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from . import buffer_ops as bo

from . import guard
from .epoch import (
    EPOCH_MARKS,
    epoch_begin,
    epoch_end,
    gstore,
    kernel_symbol,
)
from .mx import clamp_fp8
from .ops import (
    CM_DEV,
    butterfly,
    fp8_pack4,
    peer_bases,
    rsrc,
)
from .plan import (
    BLOCKS,
    THREADS,
    WAVES,
    SOURCE_TAG,
    first_task,
    key_tuple,
)
from .stamps import TL_POINTS, stamp, timeline_row
from .sync import Mailbox, preg, publish, sreg

from . import back as mb_back
from . import front as mb_front
from .attn_plan import (
    WINDOW,
    WQKV_TASKS,
    Dims,
    back_scratch,
    front_scratch,
    wqkv_split,
)
from . import moe, seam
from .dims import Dims as StageDims
from .gemv import plus
from .moe_shape import EXPERTS, SORT_NETS, TOPK, MoeBuild, route_shape
from .token_par import (
    chunk_pairs,
    half_pair,
    own_view,
    peer_layout,
    peer_offsets,
)

SOURCES = SOURCE_TAG

_STREAM = fx.Stream(None)
MAX_TOKENS = 48
HIDDEN = seam.HIDDEN
X_WORDS = HIDDEN // 4
X_GROUPS = HIDDEN // 32
SLICES = seam.SLICES
# K1 and the MoE share the MXFP8 x region (plain words in both); other names get one region each
SHARED_REGIONS = ("x8", "x8s")
COUNTER_WORDS = 2 * 256  # the MoE's ug queue and down counts, a word a slot
COUNTERS = ("ugq", "dq")
# epoch buffer words: the epoch, one mark per CTA, then the MoE counters on their own lines
EPOCH_COUNTERS = -(-(EPOCH_MARKS + BLOCKS) // 64) * 64
EPOCH_WORDS = EPOCH_COUNTERS + COUNTER_WORDS


@dataclasses.dataclass(frozen=True)
class MonoBuild:
    tokens: int
    tp: int
    kt: bool = False  # K1 also writes K2's key table
    # K1 folds the owed post into the residual; off on the first layer and after an Engram write
    fold: bool = True
    # an index source: K1 hands normed and qr to SGLang's indexer, K2 builds the key table
    index: bool = False
    # K2's keys include top-k compressed rows (compress ratio 1 or 2), not only the window
    compressed: bool = True
    # SGLANG_AMD_MONO_FAULT_INJECT: the wait to stall at (0: none) after fault_after launches
    fault_site: int = 0
    fault_after: int = 0
    # the route and window: the target's, or the DSpark draft's (``DSV41MonoRunner``)
    experts: int = EXPERTS
    topk: int = TOPK
    ug_groups: int = 2
    swa_width: int = WINDOW
    # SGLANG_AMD_MONO_TIMELINE: each CTA marks its stages in a scratch timeline
    timeline: bool = False
    # token parallel (``token_par``): each rank's seam, norm and qkv on its own rows
    tok_par: bool = False
    # K1 also computes an index source's index query, where ``index_query.fused``
    iq: bool = False
    # wqkv, wq_b and wo_b in SGLANG_ROCM_MXFP8_AITER_PRESHUFFLE's (16, 16) layout; wo_a row-major
    shuffled: bool = False


ROUTE = ("experts", "topk", "ug_groups", "swa_width")


def route_of(key: MonoBuild) -> dict:
    return {f: getattr(key, f) for f in ROUTE}


def _variant(key: MonoBuild) -> dict:
    """A kernel name's route parameters, where they are not the target's."""
    base = MonoBuild(key.tokens, key.tp)
    short = {"experts": "e", "topk": "k", "ug_groups": "u", "swa_width": "w"}
    out = {short[f]: v for f, v in route_of(key).items() if v != getattr(base, f)}
    return (
        out
        | ({"tl": 1} if key.timeline else {})
        | ({"p": 1} if key.tok_par else {})
        | ({"sh": 1} if key.shuffled else {})
    )


def _moe_key(key: MonoBuild) -> MoeBuild:
    return MoeBuild(key.tokens, key.tp, key.experts, key.topk, key.ug_groups)


def _front_key(key: MonoBuild):
    return mb_front.FrontBuild(
        key.tokens, key.tp, key.kt, key.swa_width, shuffled=key.shuffled
    )


def _back_key(key: MonoBuild):
    # 0: the window only (compress ratio 0); 1: also top-k compressed rows
    return mb_back.BackBuild(
        key.tokens, key.tp, 1 if key.compressed else 0, shuffled=key.shuffled
    )


def scratch_layout(s: int, tp: int, timeline: bool = False, **route) -> dict:
    """Return each scratch region of width s -> (byte offset, bytes), without overlap."""
    d = Dims(tp)
    out: dict = {}
    off = [0]  # the running byte offset

    def place(regions):
        base = off[0]
        for name, (o, n) in regions.items():
            assert name not in out or name in SHARED_REGIONS, name
            out[name] = (base + o, n)
        end = max(off[0], base + max(o + n for o, n in regions.values()))
        off[0] = -(-end // 256) * 256

    place(front_scratch(s, d))
    place(seam.scratch_layout(s))
    place(back_scratch(s, d, start=0))
    regions = moe.scratch_layout(_moe_key(MonoBuild(s, tp, **route))).items()
    mr = {n: v for n, v in regions if n not in COUNTERS}
    lo = min(o for o, _ in mr.values())
    place({n: (o - lo, n_) for n, (o, n_) in mr.items()})
    place({"normed": (0, s * HIDDEN * 2), "xrdy_moe": (s * HIDDEN * 2 + 256, s * 8)})
    place({"krdy": (0, s * 8)})  # an index source's key table, a flag a token
    if timeline:
        place({"timeline": (0, BLOCKS * TL_POINTS * 8)})
    return out


def scratch_bytes(
    s: int = MAX_TOKENS, tp: int = 2, timeline: bool = False, **route
) -> int:
    return max(o + n for o, n in scratch_layout(s, tp, timeline, **route).values())


def peer_half_bytes(tp: int) -> int:
    """One parity's peer regions (``token_par.peer_layout``)."""
    return max(o + n for o, n in peer_layout(MAX_TOKENS, tp).values())


def _epoch(epoch):
    return fx.Int32(bo.buffer_load(rsrc(epoch), 0, vec_width=1, dtype=T.i32))


# ---------------------------------------------------------------- K1


@flyc.jit
def store_x8(c, t, col, ys, live):
    """Store the seam's norm output at ``col`` as MXFP8 X8 / X8S for the front."""
    amax = abs(ys[0])
    for y in ys[1:]:
        amax = fx.max(amax, abs(y))
    code = moe.vllm_mx_code(butterfly(amax, (1, 2), fx.max))
    mul = ((254 - code) << 23).bitcast(fx.Float32)
    w0 = fp8_pack4(*[clamp_fp8(ys[i] * mul) for i in range(4)])
    w1 = fp8_pack4(*[clamp_fp8(ys[4 + i] * mul) for i in range(4)])
    if live:
        bo.buffer_store(
            fx.Vector.from_elements([w0, w1], fx.Int32),
            rsrc(c["x8"]),
            t * X_WORDS + col // 4,
            cache_modifier=CM_DEV,
        )
        if c["tid"] % 4 == 0:
            bo.buffer_store(
                code, rsrc(c["x8s"]), t * X_GROUPS + col // 32, cache_modifier=CM_DEV
            )


@flyc.jit
def publish_x8(c, t):
    """Token t's two XRDY flags (one a wqkv K half) once its X8 / X8S landed."""
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if c["tid"] < 2:
        c["mb"].put(c["xrdy"], 2 * t + c["tid"], fx.Int32(1))
    gpu.barrier()


def build_mono_k1(key: MonoBuild):
    s, tp = key.tokens, key.tp
    assert 1 <= s <= MAX_TOKENS
    layout = scratch_layout(s, tp, key.timeline, **route_of(key))
    fkey = _front_key(key)
    FrontLds = mb_front.front_lds(s, Dims(tp))
    # token parallel: the seam, wqkv and qkv on this rank's rows
    own_n = s // tp if key.tok_par else s
    # the norms past the slices' CTAs (the wqkv waits on them), and past a split wqkv's
    NORM0 = 2 * WQKV_TASKS if wqkv_split(s, own_n) else SLICES
    GATE0 = NORM0 + own_n
    assert GATE0 + own_n <= BLOCKS

    @fx.struct
    class SeamLds:
        red: fx.Array[fx.Float32, WAVES * 2 * 64 * 4, 16]
        rl: fx.Array[fx.Float32, s * seam.KT, 16]
        fl: fx.Array[fx.Float32, seam.MIX * seam.KT, 16]

    @fx.union
    class K1Lds:
        seam: SeamLds
        front: FrontLds  # type: ignore[valid-type]

    # the index query (``index_query``) is not vendored: no build here computes it
    assert not key.iq, key
    IW0 = (GATE0 + own_n) % BLOCKS  # the head weights' CTAs, past the gates'
    iw = 2 * -(-own_n // 16) if key.iq else 0
    QKV0 = IW0 + iw if GATE0 + 2 * own_n + iw <= BLOCKS else 0
    name = kernel_symbol(
        "dsv41_mono_k1",
        s=s,
        tp=tp,
        kt=key.kt,
        f=key.fold,
        i=key.index,
        **({"q": 1} if key.iq else {}),
        **_variant(key),
    )
    keyed = key_tuple(key, SOURCES)
    half = peer_half_bytes(tp)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k1(
        res_in: Int64,
        pend: Int64,
        post_in: Int64,
        comb_in: Int64,
        pre_in: Int64,
        hc_fn: Int64,
        hc_scale: Int64,
        hc_base: Int64,
        attn_w: Int64,
        res_out: Int64,
        aux_out: Int64,
        aux_stride: Int32,
        post_out: Int64,
        comb_out: Int64,
        pre_out: Int64,
        wqkv: Int64,
        wqkv_s: Int64,
        qn_w: Int64,
        kvn_w: Int64,
        wqb: Int64,
        wqb_s: Int64,
        cos_sin: Int64,
        pos: Int64,
        slot: Int64,
        slot_words: Int32,
        swa: Int64,
        swa_stride: Int64,
        swa_block: Int32,
        swa_idx: Int64,
        swa_lens: Int64,
        topk: Int64,
        topk_lens: Int64,
        kt: Int64,
        klen: Int64,
        q: Int64,
        normed_out: Int64,
        qr_out: Int64,
        iq_w: Int64,
        iq_ws: Int64,
        iw_w: Int64,
        iw_scale: Int32,
        num_pos: Int32,
        iq_out: Int64,
        iqs_out: Int64,
        iw_out: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
    ):
        _ = keyed
        # every wait runs under the guard, so a timeout flags every peer and stops the launch
        with guard.scope(
            guard.Guard(
                epoch,
                own=sym,
                peers=peers,
                rank=rank,
                npes=tp,
                flags_offset=2 * half,
                fault_site=key.fault_site or None,
                fault_after=key.fault_after,
            )
        ):
            bid = fx.block_idx.x
            lds = fx.SharedAllocator().allocate(K1Lds)
            sl = lds.seam.peek()
            ep = _epoch(epoch)
            tag = (ep << 1) + 1
            fargs = {
                "x": 0,
                "x_stride": 0,
                "wqkv": wqkv,
                "wqkv_s": wqkv_s,
                "qn_w": qn_w,
                "kvn_w": kvn_w,
                "wqb": wqb,
                "wqb_s": wqb_s,
                "cos_sin": cos_sin,
                "pos": pos,
                "slot": slot,
                "slot_words": slot_words,
                "swa": swa,
                "swa_stride": swa_stride,
                "swa_block": swa_block,
                "swa_idx": swa_idx,
                "swa_lens": swa_lens,
                "topk": topk,
                "topk_lens": topk_lens,
                "kt": kt,
                "klen": klen,
                "q": q,
                "qr_out": qr_out,
                "normed_out": normed_out,
                "iq_w": iq_w,
                "iq_ws": iq_ws,
                "iw_w": iw_w,
                "iw_scale": iw_scale.bitcast(fx.Float32),
                "num_pos": num_pos,
                "iq_out": iq_out,
                "iqs_out": iqs_out,
                "iw_out": iw_out,
            }
            flayout = {n: layout[n] for n in front_scratch(s, Dims(tp))}
            c = mb_front.front_context(
                fkey, lds.front.peek(), fargs, scratch, tag, flayout
            )
            c["index"] = key.index
            if const_expr(key.timeline):
                c["tl"] = timeline_row(scratch, layout["timeline"][0], bid)
            amb = Mailbox(tag)
            tid = fx.thread_idx.x
            ca = {
                "S": s,
                "tid": tid,
                "bid": bid,
                "lane": tid % 64,
                "wave": tid // 64,
                "rl": sl.rl.ptr,
                "fl": sl.fl.ptr,
                "red": sl.red.ptr,
                "put": amb.put,
                "put_bf": amb.put_bf,
                "put_words": amb.put_words,
                "poll": amb.poll,
                "args": {
                    "res_in": res_in,
                    "pend": pend,
                    "post_in": post_in,
                    "comb_in": comb_in,
                    "pre_in": pre_in,
                    "hc_fn": hc_fn,
                    "hc_scale": hc_scale,
                    "hc_base": hc_base,
                    "norm_w": attn_w,
                    "res_out": res_out,
                    "post_out": post_out,
                    "comb_out": comb_out,
                    "pre_out": pre_out,
                    # DSpark's capture of this layer: its streams' mean (0: none), rows
                    # ``aux_stride`` elements apart (a column block of the packed captures)
                    "aux_out": aux_out,
                    "aux_stride": aux_stride,
                },
                "d": StageDims(tp),
                # SGLang's Triton boundary + rmsnorm arithmetic (seam.slice_row, stage_norm)
                "sgl": True,
                "fold": key.fold,
                # before own_view moves the row arguments
                "aux_on": aux_out != 0,
            }
            for region in ("lin", "pmix"):
                ca[region] = sreg(scratch, layout[region][0], region)
            t0 = 0
            if const_expr(key.tok_par):
                t0 = rank * own_n
                ca = own_view(ca, t0, own_n)
                ca["rank"] = rank
                par = fx.Int64(ep & 1) * fx.Int64(half)
                bases = [b + par for b in peer_bases(peers, tp)]
                pl = peer_offsets(s, tp)
                mb_front.token_par_context(c, own_n, t0, rank, bases, sym + par, pl)

            def seam_slice():
                stamp(c, 32)
                for task in range(first_task(bid, 0), SLICES, BLOCKS):
                    seam.stage_slice(ca, task)
                gpu.barrier()
                stamp(c, 33)

            def store(t, col, ys, live):
                store_x8(c, t, col, ys, live)
                if const_expr(key.index):
                    # the normed input, SGLang's compressor and indexer's input
                    store_normed(normed_out, plus(t0, t), col, ys, live)
                    if const_expr(key.tok_par):
                        push_bf_row(ca, bases, pl["nbg"], plus(t0, t), col, ys, live)

            def seam_norm():
                for t in range(first_task(bid, NORM0), own_n, BLOCKS):
                    seam.stage_norm(ca, t, store)
                    publish_x8(c, t)
                gpu.barrier()
                stamp(c, 34)

            def seam_gate():
                # only K2 reads the mixes, so they fill q's wait; the gate's LDS aliases the front's
                gpu.barrier()
                for t in range(first_task(bid, GATE0), own_n, BLOCKS):
                    seam.stage_gate(ca, t)

            mb_front.run_front(
                c,
                fkey,
                bid,
                x8_given=True,
                before_wqkv=seam_slice,
                after_wqkv=seam_norm,
                before_wqb=seam_gate,
                # past the slices, norms and gates where the grid has room: a slice CTA
                # issues its weights after its slice, and the norms' publish (vmcnt(0))
                # waited for them to land
                qkv_first=QKV0,
            )
            if const_expr(key.tok_par):
                # K1's outputs hold every row: copy in the rows the other ranks computed
                own = sym + par
                for t in range(first_task(bid, 0), s, BLOCKS):
                    if t // own_n != rank:
                        mb_front.record_from(c, preg(own, pl["kg"], "kg"), t)
                        if const_expr(key.index):
                            gather_row(ca, preg(own, pl["nbg"], "nbg"), t, normed_out)
                            mb_front.qr_from(c, preg(own, pl["qrg"], "qrg"), t)
            stamp(c, 40)

    @flyc.jit
    def launch(
        res_in: Int64,
        pend: Int64,
        post_in: Int64,
        comb_in: Int64,
        pre_in: Int64,
        hc_fn: Int64,
        hc_scale: Int64,
        hc_base: Int64,
        attn_w: Int64,
        res_out: Int64,
        aux_out: Int64,
        aux_stride: Int32,
        post_out: Int64,
        comb_out: Int64,
        pre_out: Int64,
        wqkv: Int64,
        wqkv_s: Int64,
        qn_w: Int64,
        kvn_w: Int64,
        wqb: Int64,
        wqb_s: Int64,
        cos_sin: Int64,
        pos: Int64,
        slot: Int64,
        slot_words: Int32,
        swa: Int64,
        swa_stride: Int64,
        swa_block: Int32,
        swa_idx: Int64,
        swa_lens: Int64,
        topk: Int64,
        topk_lens: Int64,
        kt: Int64,
        klen: Int64,
        q: Int64,
        normed_out: Int64,
        qr_out: Int64,
        iq_w: Int64,
        iq_ws: Int64,
        iw_w: Int64,
        iw_scale: Int32,
        num_pos: Int32,
        iq_out: Int64,
        iqs_out: Int64,
        iw_out: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        k1(
            res_in,
            pend,
            post_in,
            comb_in,
            pre_in,
            hc_fn,
            hc_scale,
            hc_base,
            attn_w,
            res_out,
            aux_out,
            aux_stride,
            post_out,
            comb_out,
            pre_out,
            wqkv,
            wqkv_s,
            qn_w,
            kvn_w,
            wqb,
            wqb_s,
            cos_sin,
            pos,
            slot,
            slot_words,
            swa,
            swa_stride,
            swa_block,
            swa_idx,
            swa_lens,
            topk,
            topk_lens,
            kt,
            klen,
            q,
            normed_out,
            qr_out,
            iq_w,
            iq_ws,
            iw_w,
            iw_scale,
            num_pos,
            iq_out,
            iqs_out,
            iw_out,
            scratch,
            sym,
            peers,
            rank,
            epoch,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    return launch


# ---------------------------------------------------------------- the FFN launch


@flyc.jit
def store_normed(normed, t, col, ys, live):
    """Store the FFN seam's norm output as the MoE input row (bf16, device scope), the 8
    values in one 16 B store: as 8 stores of 2 B they held the norm's publish back."""
    words = [
        fx.Vector.from_elements([ys[2 * i], ys[2 * i + 1]], fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Int32)[0]
        for i in range(4)
    ]
    if live:
        bo.buffer_store(
            fx.Vector.from_elements(words, fx.Int32),
            rsrc(normed),
            (t * HIDDEN + col) // 2,
            cache_modifier=CM_DEV,
        )


@flyc.jit
def push_attn_pair(c, bases, own, s, t, col, vs):
    """Push this rank's wo_b partial pair of row t at ``col`` to every rank's ATTN region.

    The pairs are ``seam.stage_push``'s; token parallel (``own`` rows a rank), only the
    row's owner gets it, at its row there."""
    for p in range_constexpr(len(bases)):
        dst = preg(bases[p], 0, "attn")
        if const_expr(own is None):
            c["put_bf"](dst, 2 * seam.attn_region_pair(c["rank"], t, s, col), vs)
        else:
            if t // own == p:
                pair = seam.attn_region_pair(c["rank"], t - p * own, own, col)
                c["put_bf"](dst, 2 * pair, vs)


@flyc.jit
def push_bf_row(c, bases, off, t, col, ys, live):
    """Row t's 8 bf16 values at ``col`` to every other rank's peer region ``off``."""
    if live:
        for p in range_constexpr(c["d"].tp):
            if c["rank"] != p:
                dst = preg(bases[p], off, "gather")
                for h in range_constexpr(2):
                    at = t * HIDDEN + 2 * half_pair(HIDDEN // 2, col // 8, h)
                    c["put_bf"](dst, at, ys[4 * h : 4 * h + 4])


@flyc.jit
def gather_row(c, src, t, dst):
    """Row t of a bf16 [row][HIDDEN] peer region ``src`` -> ``dst`` (device scope)."""
    tid = c["tid"]
    for k in range_constexpr(-(-(HIDDEN // 4) // THREADS)):
        q = tid + THREADS * k
        if q < HIDDEN // 4:
            (w,) = c["poll"]([(src, t * (HIDDEN // 2) + 2 * q, 2)])
            bo.buffer_store(
                fx.Vector.from_elements([w[0], w[1]], fx.Int32),
                rsrc(dst),
                t * (HIDDEN // 2) + chunk_pairs(HIDDEN // 2, q),
                cache_modifier=CM_DEV,
            )


def _seam_lds(s):
    @fx.struct
    class SeamLds:
        red: fx.Array[fx.Float32, WAVES * 2 * 64 * 4, 16]
        rl: fx.Array[fx.Float32, s * seam.KT, 16]
        fl: fx.Array[fx.Float32, seam.MIX * seam.KT, 16]
        pl: fx.Array[fx.Float32, s * seam.COLS, 16]

    return SeamLds


@flyc.jit
def run_ffn(key: MonoBuild, lds, a: dict, amb, bases, own, rank, ep, part=None):
    """Run the FFN seam, the MoE, its all-reduce into ``a["out"]``, and the epoch update."""
    s, tp = key.tokens, key.tp
    layout = scratch_layout(s, tp, key.timeline, **route_of(key))
    mkey = _moe_key(key)
    rs = route_shape(mkey)
    # token parallel: the seam, norm and gates on this rank's rows
    own_n = s // tp if const_expr(key.tok_par) else s
    GATE0 = SLICES
    NORM0 = SLICES + own_n
    GATHER0 = NORM0 + own_n  # the other ranks' MoE input rows (token parallel)
    assert not key.tok_par or GATHER0 + s <= BLOCKS
    bid = fx.block_idx.x
    tid = fx.thread_idx.x
    sl = lds.seam.peek()
    scratch = a["scratch"]
    seam_args = (
        "res_in",
        "post_in",
        "comb_in",
        "pre_in",
        "hc_fn",
        "hc_scale",
        "hc_base",
        "res_out",
        "post_out",
        "comb_out",
        "pre_out",
    )
    ca = {
        "S": s,
        "tid": tid,
        "bid": bid,
        "lane": tid % 64,
        "wave": tid // 64,
        "rl": sl.rl.ptr,
        "fl": sl.fl.ptr,
        "red": sl.red.ptr,
        "pend_lds": sl.pl.ptr,
        "put": amb.put,
        "put_bf": amb.put_bf,
        "put_words": amb.put_words,
        "poll": amb.poll,
        "peer_addr": lambda p: bases[p],
        "rank": rank,
        "sym": own,
        "args": {n: a[n] for n in seam_args} | {"norm_w": a["ffn_w"]},
        "d": StageDims(tp),
        # SGLang's Triton boundary + rmsnorm arithmetic (seam.slice_row, stage_norm)
        "sgl": True,
    }
    if const_expr(key.timeline):
        ca["tl"] = timeline_row(scratch, layout["timeline"][0], bid)
    stamp(ca, 16)
    t0 = 0
    if const_expr(key.tok_par):
        t0 = rank * own_n
        ca = own_view(ca, t0, own_n)
        xg = peer_offsets(s, tp)["xg"]
    normed = scratch + fx.Int64(layout["normed"][0])
    for region in ("lin", "pmix"):
        ca[region] = sreg(scratch, layout[region][0], region)
    mlayout = {n: layout[n] for n in moe.scratch_layout(mkey) if n not in COUNTERS}
    mlayout["xrdy"] = layout["xrdy_moe"]
    margs = moe.moe_args(
        normed,
        a["gate_w"],
        a["bias"],
        a["w13"],
        a["w13_s"],
        a["w2"],
        a["w2_s"],
        a["sgu"],
        a["sgu_s"],
        a["sw2"],
        a["sw2_s"],
        a["out"],
    )
    cb = moe.moe_context(
        s, rs, lds.moe.peek(), amb, bases, mlayout, scratch, margs, rank, own
    )
    cb["tag"] = ep & 255
    if const_expr(key.timeline):
        cb["tl"] = ca["tl"]
    for i, region in enumerate(COUNTERS):
        cb[region] = sreg(a["epoch"], 4 * (EPOCH_COUNTERS + i * moe.TAGS), region)
    if const_expr(key.tok_par):
        # the down sums reduce-scattered: each rank sums its own rows into ``out``
        cb["own"] = own_n
        out_own = a["out"] + fx.Int64(t0) * fx.Int64(HIDDEN * 2)
        cb["own_view"] = {**cb, "S": own_n, "args": {**margs, "out": out_own}}
    for task in range(first_task(bid, 0), SLICES, BLOCKS):
        # the slice's own loads go out ahead of the wait on the ranks' partials
        pre = seam.slice_prefetch(ca, task)
        if const_expr(part is not None):
            seam.stage_push(ca, task, part)
        seam.stage_reduce(ca, task)
        seam.stage_slice(ca, task, pre)
    stamp(ca, 17)

    def store(t, col, ys, live):
        store_normed(normed, plus(t0, t), col, ys, live)
        if const_expr(key.tok_par):
            # the row for every other rank's MoE (``gather_row``)
            push_bf_row(ca, bases, xg, plus(t0, t), col, ys, live)

    for t in range(first_task(bid, NORM0), own_n, BLOCKS):
        seam.stage_norm(ca, t, store)
        publish(cb["put"], cb["xrdy"], plus(t0, t), 1, tid == 0)
    if const_expr(key.tok_par):
        for t in range(first_task(bid, GATHER0), s, BLOCKS):
            if t // own_n != rank:
                gather_row(ca, preg(own, xg, "xg"), t, normed)
                publish(cb["put"], cb["xrdy"], t, 1, tid == 0)
    gpu.barrier()
    stamp(ca, 18)

    def gates():
        # only the next launch reads the mixes; later MoE stages reload the LDS the gate clobbers
        gpu.barrier()
        for t in range(first_task(bid, GATE0), own_n, BLOCKS):
            seam.stage_gate(ca, t)
        stamp(ca, 23)

    moe.run_moe(cb, mkey, bid, before_ug=gates)
    stamp(ca, 27)

    def reset():
        # the MoE counter slot 128 launch pairs ahead: its last use is long done
        slot = (ep + 128) & 255
        for region in COUNTERS:
            gstore(cb[region].value + fx.Int64(slot * 4), fx.Int32(0), words=1)

    epoch_end({"bid": bid, "tid": tid}, a["epoch"], ep, reset)


def build_mono_ffn(key: MonoBuild):
    """Build the FFN launch: push this rank's wo_b partial, then ``run_ffn``."""
    s, tp = key.tokens, key.tp
    assert 1 <= s <= MAX_TOKENS
    assert not key.tok_par, "the FFN launch's partials arrive whole"
    rs = route_shape(_moe_key(key))
    assert rs.experts // 64 in SORT_NETS
    MoeLds = moe.moe_smem(s, rs)
    half = peer_half_bytes(tp)
    assert SLICES + 2 * s <= BLOCKS
    SeamLds = _seam_lds(s)

    @fx.union
    class FfnLds:
        seam: SeamLds  # type: ignore[valid-type]
        moe: MoeLds  # type: ignore[valid-type]

    name = kernel_symbol("dsv41_mono_ffn", s=s, tp=tp, **_variant(key))
    keyed = key_tuple(key, SOURCES)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kffn(
        part: Int64,
        res_in: Int64,
        post_in: Int64,
        comb_in: Int64,
        pre_in: Int64,
        hc_fn: Int64,
        hc_scale: Int64,
        hc_base: Int64,
        ffn_w: Int64,
        res_out: Int64,
        post_out: Int64,
        comb_out: Int64,
        pre_out: Int64,
        gate_w: Int64,
        bias: Int64,
        w13: Int64,
        w13_s: Int64,
        w2: Int64,
        w2_s: Int64,
        sgu: Int64,
        sgu_s: Int64,
        sw2: Int64,
        sw2_s: Int64,
        out: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
    ):
        _ = keyed
        # every wait runs under the guard, so a timeout flags every peer and stops the launch
        with guard.scope(
            guard.Guard(
                epoch,
                own=sym,
                peers=peers,
                rank=rank,
                npes=tp,
                flags_offset=2 * half,
                fault_site=key.fault_site or None,
                fault_after=key.fault_after,
            )
        ):
            lds = fx.SharedAllocator().allocate(FfnLds)
            ep = _epoch(epoch)
            par = fx.Int64(ep & 1) * fx.Int64(half)
            bases = [b + par for b in peer_bases(peers, tp)]
            epoch_begin({"bid": fx.block_idx.x, "tid": fx.thread_idx.x}, epoch, ep)
            a = dict(
                res_in=res_in,
                post_in=post_in,
                comb_in=comb_in,
                pre_in=pre_in,
                hc_fn=hc_fn,
                hc_scale=hc_scale,
                hc_base=hc_base,
                ffn_w=ffn_w,
                res_out=res_out,
                post_out=post_out,
                comb_out=comb_out,
                pre_out=pre_out,
                gate_w=gate_w,
                bias=bias,
                w13=w13,
                w13_s=w13_s,
                w2=w2,
                w2_s=w2_s,
                sgu=sgu,
                sgu_s=sgu_s,
                sw2=sw2,
                sw2_s=sw2_s,
                out=out,
                scratch=scratch,
                epoch=epoch,
            )
            amb = Mailbox((ep << 1) + 2)
            run_ffn(key, lds, a, amb, bases, sym + par, rank, ep, part=part)

    @flyc.jit
    def launch(
        part: Int64,
        res_in: Int64,
        post_in: Int64,
        comb_in: Int64,
        pre_in: Int64,
        hc_fn: Int64,
        hc_scale: Int64,
        hc_base: Int64,
        ffn_w: Int64,
        res_out: Int64,
        post_out: Int64,
        comb_out: Int64,
        pre_out: Int64,
        gate_w: Int64,
        bias: Int64,
        w13: Int64,
        w13_s: Int64,
        w2: Int64,
        w2_s: Int64,
        sgu: Int64,
        sgu_s: Int64,
        sw2: Int64,
        sw2_s: Int64,
        out: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        kffn(
            part,
            res_in,
            post_in,
            comb_in,
            pre_in,
            hc_fn,
            hc_scale,
            hc_base,
            ffn_w,
            res_out,
            post_out,
            comb_out,
            pre_out,
            gate_w,
            bias,
            w13,
            w13_s,
            w2,
            w2_s,
            sgu,
            sgu_s,
            sw2,
            sw2_s,
            out,
            scratch,
            sym,
            peers,
            rank,
            epoch,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    return launch


# ---------------------------------------------------------------- K2


def build_mono_k2(key: MonoBuild):
    """Build K2: the attention back on K1's q, its wo_b partial pushed to every rank,
    then ``run_ffn``."""
    s, tp = key.tokens, key.tp
    assert 1 <= s <= MAX_TOKENS
    layout = scratch_layout(s, tp, key.timeline, **route_of(key))
    bkey = _back_key(key)
    rs = route_shape(_moe_key(key))
    assert rs.experts // 64 in SORT_NETS
    back_m = mb_back.back_lds_members(s, Dims(tp))
    SplitLds, GemvLds = back_m["split"], back_m["gemv"]
    MoeLds = moe.moe_smem(s, rs)
    half = peer_half_bytes(tp)
    assert SLICES + 2 * s <= BLOCKS
    SeamLds = _seam_lds(s)

    @fx.union
    class K2Lds:
        split: SplitLds  # type: ignore[valid-type]
        gemv: GemvLds  # type: ignore[valid-type]
        seam: SeamLds  # type: ignore[valid-type]
        moe: MoeLds  # type: ignore[valid-type]

    name = kernel_symbol(
        "dsv41_mono_k2", s=s, tp=tp, c=key.compressed, i=key.index, **_variant(key)
    )
    keyed = key_tuple(key, SOURCES)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k2(
        q: Int64,
        swa: Int64,
        swa_stride: Int64,
        swa_block: Int32,
        comp: Int64,
        comp_stride: Int64,
        comp_block: Int32,
        kt: Int64,
        klen: Int64,
        slot: Int64,
        slot_words: Int32,
        swa_idx: Int64,
        swa_lens: Int64,
        topk: Int64,
        topk_lens: Int64,
        pos: Int64,
        sink: Int64,
        qk_scale: Int32,
        cos_sin: Int64,
        woa: Int64,
        woa_s: Int64,
        wob: Int64,
        wob_s: Int64,
        zrec: Int64,
        res_in: Int64,
        post_in: Int64,
        comb_in: Int64,
        pre_in: Int64,
        hc_fn: Int64,
        hc_scale: Int64,
        hc_base: Int64,
        ffn_w: Int64,
        res_out: Int64,
        post_out: Int64,
        comb_out: Int64,
        pre_out: Int64,
        gate_w: Int64,
        bias: Int64,
        w13: Int64,
        w13_s: Int64,
        w2: Int64,
        w2_s: Int64,
        sgu: Int64,
        sgu_s: Int64,
        sw2: Int64,
        sw2_s: Int64,
        out: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
    ):
        _ = keyed
        # every wait runs under the guard, so a timeout flags every peer and stops the launch
        with guard.scope(
            guard.Guard(
                epoch,
                own=sym,
                peers=peers,
                rank=rank,
                npes=tp,
                flags_offset=2 * half,
                fault_site=key.fault_site or None,
                fault_after=key.fault_after,
            )
        ):
            bid = fx.block_idx.x
            lds = fx.SharedAllocator().allocate(K2Lds)
            ep = _epoch(epoch)
            tag = (ep << 1) + 2
            par = fx.Int64(ep & 1) * fx.Int64(half)
            bases = [b + par for b in peer_bases(peers, tp)]
            own = sym + par
            bargs = {
                "q": q,
                "swa": swa,
                "swa_stride": swa_stride,
                "swa_block": swa_block,
                "comp": comp,
                "comp_stride": comp_stride,
                "comp_block": comp_block,
                "kt": kt,
                "klen": klen,
                "pos": pos,
                "sink": sink,
                "qk_scale": qk_scale,
                "cos_sin": cos_sin,
                "woa": woa,
                "woa_s": woa_s,
                "wob": wob,
                "wob_s": wob_s,
                "zrec": zrec,
            }
            blayout = {n: layout[n] for n in back_scratch(s, Dims(tp))}
            views = {"split": lds.split.peek(), "gemv": lds.gemv.peek()}
            c = mb_back.back_context(bkey, views, bargs, scratch, tag, blayout)
            if const_expr(key.timeline):
                c["tl"] = timeline_row(scratch, layout["timeline"][0], bid)
            amb = Mailbox(tag)

            cp = {"put_bf": amb.put_bf, "rank": rank}
            c["wob_out"] = lambda t, col, v0, v1: push_attn_pair(
                cp, bases, s // tp if key.tok_par else None, s, t, col, [v0, v1]
            )
            epoch_begin({"bid": bid, "tid": fx.thread_idx.x}, epoch, ep)
            if const_expr(key.index):
                # the indexer's top-k came after K1; each split task waits on its token's krdy
                c["krdy"] = scratch + fx.Int64(layout["krdy"][0])
                ckt = {
                    "S": s,
                    "tid": c["tid"],
                    "dev": True,
                    "args": {
                        "slot": slot,
                        "slot_words": slot_words,
                        "swa_idx": swa_idx,
                        "swa_lens": swa_lens,
                        "topk": topk,
                        "topk_lens": topk_lens,
                        "kt": kt,
                        "klen": klen,
                    },
                }
                for t in range(first_task(bid, BLOCKS - s), s, BLOCKS):
                    mb_front.stage_kt(ckt, t)
                    mb_back.publish(c, c["krdy"], t)
            mb_back.run_back(c, bkey, bid)
            gpu.barrier()
            a = dict(
                res_in=res_in,
                post_in=post_in,
                comb_in=comb_in,
                pre_in=pre_in,
                hc_fn=hc_fn,
                hc_scale=hc_scale,
                hc_base=hc_base,
                ffn_w=ffn_w,
                res_out=res_out,
                post_out=post_out,
                comb_out=comb_out,
                pre_out=pre_out,
                gate_w=gate_w,
                bias=bias,
                w13=w13,
                w13_s=w13_s,
                w2=w2,
                w2_s=w2_s,
                sgu=sgu,
                sgu_s=sgu_s,
                sw2=sw2,
                sw2_s=sw2_s,
                out=out,
                scratch=scratch,
                epoch=epoch,
            )
            run_ffn(key, lds, a, amb, bases, own, rank, ep)

    @flyc.jit
    def launch(
        q: Int64,
        swa: Int64,
        swa_stride: Int64,
        swa_block: Int32,
        comp: Int64,
        comp_stride: Int64,
        comp_block: Int32,
        kt: Int64,
        klen: Int64,
        slot: Int64,
        slot_words: Int32,
        swa_idx: Int64,
        swa_lens: Int64,
        topk: Int64,
        topk_lens: Int64,
        pos: Int64,
        sink: Int64,
        qk_scale: Int32,
        cos_sin: Int64,
        woa: Int64,
        woa_s: Int64,
        wob: Int64,
        wob_s: Int64,
        zrec: Int64,
        res_in: Int64,
        post_in: Int64,
        comb_in: Int64,
        pre_in: Int64,
        hc_fn: Int64,
        hc_scale: Int64,
        hc_base: Int64,
        ffn_w: Int64,
        res_out: Int64,
        post_out: Int64,
        comb_out: Int64,
        pre_out: Int64,
        gate_w: Int64,
        bias: Int64,
        w13: Int64,
        w13_s: Int64,
        w2: Int64,
        w2_s: Int64,
        sgu: Int64,
        sgu_s: Int64,
        sw2: Int64,
        sw2_s: Int64,
        out: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        k2(
            q,
            swa,
            swa_stride,
            swa_block,
            comp,
            comp_stride,
            comp_block,
            kt,
            klen,
            slot,
            slot_words,
            swa_idx,
            swa_lens,
            topk,
            topk_lens,
            pos,
            sink,
            qk_scale,
            cos_sin,
            woa,
            woa_s,
            wob,
            wob_s,
            zrec,
            res_in,
            post_in,
            comb_in,
            pre_in,
            hc_fn,
            hc_scale,
            hc_base,
            ffn_w,
            res_out,
            post_out,
            comb_out,
            pre_out,
            gate_w,
            bias,
            w13,
            w13_s,
            w2,
            w2_s,
            sgu,
            sgu_s,
            sw2,
            sw2_s,
            out,
            scratch,
            sym,
            peers,
            rank,
            epoch,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    return launch
