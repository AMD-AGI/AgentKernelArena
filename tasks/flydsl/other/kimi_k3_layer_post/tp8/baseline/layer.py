# SPDX-License-Identifier: Apache-2.0
# Vendored for AgentKernelArena from sgl-project/sgl-amd-mono at 52486703fbf27e77f05b573e4d931d4eec87118d,
# sgl_amd_mono/kimi_k3/gfx950/layer.py. AKA changes: relative imports; K2Build a frozen dataclass (no msgspec); SOURCES a constant
# (no plan.kernel_sources digest); tag_offset() sized over this launch's own scratch layouts (no K1 /
# MLA-front launches in this package); the host-side init_peer / SENT_READY sentinel arming and the
# k2_launch host wrapper removed (kernel.py launches the build). No kernel or stage code changed.
"""K2 ``layer_post``: one Kimi-K3 layer from the attention's core output (the
o_proj input) to the layer's MoE output, in one launch -- K2a's o_proj, its
all-reduce and the MLP AttnRes seam, then K2b's latent MoE (``moe``).

As ATOM's ``layer_post``, the two halves share the launch so that the MoE's
static weights stream while the AttnRes chain (partials -> reduce -> slice ->
gate -> mix) is still settling: the CTAs that will run the router, latent and
shared gate/up GEMVs issue their weights before they first wait.

    CTAs 0 .. 55     K2a's reduce + AttnRes (slice, gate, mix -> x rows + XRDY);
                     then route (0 .. S-1) and the shared SiTU (8 .. 8+S-1)
    CTAs 56 .. 159   o_proj (448 x 16 rows) -> partials to every rank; then
                     the shared gate/up (96 tasks, weights right after o_proj)
    CTAs 160 .. 243  router parts (28) and latent parts (56): weights first
    every CTA        ug; down (0 .. 223) or the shared down (224 .. 255);
                     lnorm (56 .. 83), up (0 .. 55), reduce (200 .. 255)
"""

import dataclasses

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Int32, Int64

from . import guard
from .ops import (
    ld_i32,
    load_ptr64,
    peer_bases,
    uniform,
)
from .plan import (
    BLOCKS,
    THREADS,
    WAVES,
    key_tuple,
    pair_layout,
)
from .sync import Mailbox, preg, shift, sreg
from .model import MAX_ROWS
from . import gemv as gemv
from . import moe as k2b
from . import oproj as k2a
from .dims import COLS, HIDDEN, NA
from .seam import (
    slice_loads,
    stage_gate,
    stage_mix,
    stage_slice,
)
from .stamps import (
    stamp,
    stamp_begin,
    stamp_flush,
)
from .tag import step_tag

# AKA: one flat package; FlyDSL hashes the kernel directory's own helpers
SOURCES = "aka-flat"

O0, O1 = NA, BLOCKS  # o_proj CTAs: every CTA past AttnRes's
O_N = O1 - O0  # 200
OTASKS = k2a.OTASKS  # 448
O_ROUNDS = (OTASKS + O_N - 1) // O_N  # 5
E0 = 160  # router / latent CTAs (their weights in flight under their o_proj)
H0 = 8  # the shared SiTU's CTAs (after their AttnRes work)
SD0 = k2b.DOWN_TASKS  # 224: the shared down's CTAs
# timeline points: 0 start, 1 o_proj done / AR reduced, 2 early done / slice,
# 3 ug, 4 down/sdown, 5 lnorm, 6 up, 7 reduce, 8 gate, 9 mix (x published),
# 10 route, 11 route table, 12 latent quant, 13 route starts
TL_POINTS = 25  # + 20-24 router: x / LDS / GEMV / parts / pushed; + 14 route logits in, 15 INTER loaded, 16 down done, 17-19 route keys / select / rank

assert E0 + k2b.R_TASKS + k2b.L_TASKS <= BLOCKS and k2b.G_TASKS <= E0 - O0

REGIONS = (
    ("part", "wgt", "msq", "xrdy", "ar")
    + k2b.REGIONS
    + (
        "ars",
        "latrs",
        "finals",
        "xt",
    )
)

# K2 options (``K2Build.opt``), each bit for bit the launch without it:
OPT_SENT = 1  # o_proj's all-reduce through a sentinel region (sync.put_sent)
OPT_PRELOAD = 2  # the seam's block / prefix / norm loads before that poll
OPT_TAG_X = 8  # x rows also as tagged pairs to the router / latent GEMVs: no XRDY
OPT_SENT_LATR = 16  # the latent all-reduce through a sentinel region
OPT_SENT_FINAL = 32  # the final all-reduce through a sentinel region
OPT_SDOWN_FIRST = 64  # the shared-down tasks up waits on before the others
OPT_RLOG_PARTS = 256  # every router part pushes its partial: no part-0 sum hop
OPT_BATCH = 512  # the route's, down's and up's polls in one batch each
OPT_SENTS = OPT_SENT | OPT_SENT_LATR | OPT_SENT_FINAL
K2_OPT = (
    OPT_SENTS | OPT_PRELOAD | OPT_TAG_X | OPT_SDOWN_FIRST | OPT_RLOG_PARTS | OPT_BATCH
)
NARROW_ROWS = 8  # this launch's widest step: the sentinel and tagged rows' size

_STREAM = fx.Stream(None)


@dataclasses.dataclass(frozen=True)
class K2Build:
    tokens: int
    nblocks: int  # the MLP AttnRes block sources
    eps: float = 1e-5  # mlp_res_norm
    out_eps: float = 1e-5  # post_attention_layernorm
    ln_eps: float = 1e-5  # the latent RMSNorm
    reduce: bool = True  # the final reduce here (else the next layer's K1)
    reset: bool = False  # a block-write layer: the prefix restarts at the o_proj output
    # the reduce writes out + the updated prefix (the layer's residual stream), and
    # the seam reads the incoming prefix from ``prefix_in``: no copy in, no add out.
    # False (the kernel checks): out is the MoE output alone
    fold: bool = True
    timeline: bool = False
    # fault injection (tests only): the wait to stall and how many launches
    # to let through first. Compile-time, so a serving build pays nothing.
    fault_site: int = 0
    fault_after: int = 0
    xp: int = 0  # timing experiments (7: route without its selection)
    guint: bool = False  # w13's AITER layout; see stages.moe.MoeBuild
    opt: int = 0  # the OPT_* bits


def peer_layout(s):
    """K2a's AR region, then K2b's: one parity's (name -> (byte offset, bytes))."""
    pairs = {"ar": k2a.ar_half_pairs(s), **k2b.peer_pairs(s)}
    return pair_layout(pairs.items())


def peer_bytes(s):
    lay = peer_layout(s)
    return 2 * max(o + n for o, n in lay.values())


def flags_offset() -> int:
    """Where every launch reads the guard flags in a rank's peer buffer.

    The runner allocates one peer buffer for all the widths. The flags follow
    the data, so they start after the widest data region of either launch.

    Both the runner and the kernel must use this function. A launch that
    computes the offset from its own width puts the flags inside the data
    region at every width below the widest one. The widths then disagree on
    where the flags are. A launch reads the bytes another width wrote, takes
    live data for an error word, and stops every rank.
    """
    return max(peer_bytes(MAX_ROWS), k2b.peer_bytes(MAX_ROWS))


def sent_layout() -> dict:
    """One layer parity's sentinel regions, name -> (byte offset, bytes): a word
    a (rank, bf16 pair, row), NARROW_ROWS rows."""
    sizes = {
        "ars": TP_ * NARROW_ROWS * HIDDEN // 2 * 4,
        "latrs": TP_ * NARROW_ROWS * k2b.LAT // 2 * 4,
        "finals": TP_ * NARROW_ROWS * HIDDEN // 2 * 4,
    }
    out, off = {}, 0
    for name, n in sizes.items():
        out[name] = (off, n)
        off += -(-n // 256) * 256
    return out


def sent_half_bytes() -> int:
    return max(o + n for o, n in sent_layout().values())


def sent_offset() -> int:
    """Where the sentinel regions start in a rank's peer buffer: past the guard
    flags, two layer parities. A word is pushed again two layers later, after
    a launch in between that its reader's rank finished first."""
    return -(-(flags_offset() + guard.FLAG_BYTES) // 4096) * 4096


def peer_alloc_bytes() -> int:
    """A rank's peer buffer: the parity halves, the guard flags, the sentinels."""
    return sent_offset() + 2 * sent_half_bytes()


def tag_offset() -> int:
    """Where the tagged x rows (OPT_TAG_X) start in the scratch: past every
    launch's layout at its widest, so no other launch's plain word sits where
    these keep tags."""
    from .model import MAX_BLOCKS

    # AKA: this package holds the layer launch alone, so the scratch it shares is K2's (and K2b's)
    # at their widest; upstream also takes Kimi-K3's K1 and MLA-front launches into the maximum.
    end = max(
        k2b.scratch_bytes(k2b.MoeBuild(tokens=MAX_ROWS)),
        scratch_bytes(K2Build(tokens=MAX_ROWS, nblocks=MAX_BLOCKS)),
    )
    return -(-end // 4096) * 4096


def scratch_alloc_bytes() -> int:
    """The scratch every Kimi-K3 launch on a rank shares, tagged x rows included."""
    return tag_offset() + NARROW_ROWS * HIDDEN // 2 * 8


def scratch_layout(key: K2Build) -> dict:
    s = key.tokens
    a = k2a.scratch_layout(k2a.AttnPostBuild(tokens=s, nblocks=key.nblocks))
    end = max(o + n for o, n in a.values())
    b = k2b.scratch_layout(k2b.MoeBuild(tokens=s))
    lay = dict(a)
    for name, (o, n) in b.items():
        lay[name] = ((end + 15) // 16 * 16 + o, n)
    end = max(o + n for o, n in lay.values())
    lay["xrow"] = ((end + 15) // 16 * 16, s * HIDDEN * 2)
    return lay


def scratch_bytes(key: K2Build) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


@flyc.jit
def stage_oproj(c, gi):
    """o_proj tasks gi, gi + 200, gi + 400 (< 448): every task's weights in
    flight at once (one memory latency, not three), then the MFMAs."""
    ops = [k2a.o_loads(c, fx.min(gi + O_N * j, OTASKS - 1)) for j in range(O_ROUNDS)]
    k2a.load_core(c)
    for j in range_constexpr(O_ROUNDS):
        t = gi + O_N * j
        if t < OTASKS:
            k2a.stage_o(c, t, ops[j])


_BUILDS: dict = {}


def build(key: K2Build):
    if key in _BUILDS:
        return _BUILDS[key]
    s, ns = key.tokens, key.nblocks + 1
    assert 1 <= s <= 8
    lay = scratch_layout(key)
    play = peer_layout(s)
    half = peer_bytes(s) // 2

    @fx.struct
    class RouteLds:
        route: fx.Array[fx.Int32, s * 2 * k2b.TOPK, 16]
        flag: fx.Array[fx.Int32, k2b.E, 16]
        expert: fx.Array[fx.Int32, k2b.MAX_U, 16]
        kof: fx.Array[fx.Int32, k2b.MAX_U * 8, 16]
        scan: fx.Array[fx.Int32, WAVES, 16]

    @fx.struct
    class EarlyLds:
        xl: fx.Array[fx.Int32, s * (HIDDEN + gemv.LDS_PAD) // 2, 16]

    @fx.struct
    class AttnLds:
        vals: fx.Array[fx.Float32, s * ns * COLS, 16]
        dl: fx.Array[fx.Float32, s * COLS, 16]

    @fx.struct
    class UgLds:
        latq: fx.Array[fx.Int32, s * k2b.LAT // 8, 16]
        latsc: fx.Array[fx.Int32, s * k2b.LAT // 32, 16]
        av: fx.Array[fx.Float32, THREADS, 16]

    @fx.struct
    class DownLds:
        interq: fx.Array[fx.Int32, s * k2b.TOPK * 48, 16]
        intsc: fx.Array[fx.Int32, s * k2b.TOPK * k2b.UG_GROUPS, 16]
        contrib: fx.Array[fx.Float32, s * k2b.TOPK * k2b.ROWS, 16]

    @fx.struct
    class PartsLds:
        rlp: fx.Array[fx.Float32, k2b.R_PARTS * k2b.E, 16]

    @fx.union
    class StageLds:
        early: EarlyLds
        attn: AttnLds
        ug: UgLds
        down: DownLds
        parts: PartsLds  # the route's router parts (OPT_RLOG_PARTS), after the mix

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        tls: fx.Array[fx.Int64, TL_POINTS, 16]

    keyed = key_tuple(key, SOURCES)
    name = (
        f"k3_mono_k2_s{s}_nb{key.nblocks}_x{int(key.reset)}_r{int(key.reduce)}"
        f"_t{int(key.timeline)}"
        + ("_f1" if key.fold else "")
        + (f"_o{key.opt}" if key.opt else "")
    )

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k2(
        core: Int64,
        w_o: Int64,
        prefix: Int64,
        prefix_in: Int64,
        blocks: Int64,
        blk_sm: Int32,
        blk_sr: Int32,
        ares_nw: Int64,
        ares_qk: Int64,
        in_nw: Int64,
        w_gate: Int64,
        bias: Int64,
        w_ld: Int64,
        w_sgu: Int64,
        w_sd: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        ln_w: Int64,
        w_up: Int64,
        out: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
        tl: Int64,
    ):
        _ = keyed
        own = load_ptr64(peers, rank)
        # Every wait of this launch runs under the guard. A wait past the
        # deadline sets this GPU's error word. It also flags every peer, so
        # their waits return too. The launch then stops on all of them.
        with guard.scope(
            guard.Guard(
                epoch,
                own=own,
                peers=peers,
                rank=rank,
                npes=TP_,
                # Not key.tokens: the offset is the same at every width. See
                # flags_offset().
                flags_offset=flags_offset(),
                fault_site=key.fault_site or None,
                fault_after=key.fault_after,
            )
        ):
            tid = fx.thread_idx.x
            bid = fx.block_idx.x
            alloc = fx.SharedAllocator()
            lds = alloc.allocate(Smem).peek()
            rl = alloc.allocate(RouteLds).peek()
            st = alloc.allocate(StageLds)
            el, al, ul, dl, pl = (
                st.early.peek(),
                st.attn.peek(),
                st.ug.peek(),
                st.down.peek(),
                st.parts.peek(),
            )
            mb = Mailbox(step_tag(epoch, layer))
            par = uniform(ld_i32(epoch, 0)) % 2
            bases = peer_bases(peers, TP_)
            peer = {}
            for region in ("rlog", "lat", "latq", "latr", "final", "ar", "rlogp"):
                off = play[region][0]
                peer[region] = [
                    shift(preg(b, off, region), fx.Int64(par) * half) for b in bases
                ]
                peer[region + "_own"] = shift(
                    preg(own, off, region), fx.Int64(par) * half
                )
            c = {
                "S": s,
                "NS": ns,
                "NB": key.nblocks,
                "tid": tid,
                "bid": bid,
                "lane": tid % 64,
                "wave": tid // 64,
                "rank": rank,
                "delta": True,
                "reset": key.reset,
                "fold": key.fold,
                # the top-16 by candidate rank; the latent quantized to MXFP4 by
                # its producers and pushed to every rank
                "route_v": 2,
                "latq_push": True,
                "gate_local": False,
                "debug": False,
                "eps": key.eps,
                "out_eps": key.out_eps,
                "ln_eps": key.ln_eps,
                "xp": key.xp,
                "guint": key.guint,
                "lazy_inter": True,
                "put": mb.put,
                "put_bf": mb.put_bf,
                "put_words": mb.put_words,
                "poll": mb.poll,
                "red": lds.red.ptr,
                "peer": peer,
                "ar_peer": peer["ar"],
                "ar_own": peer["ar_own"],
                "xl": el.xl.ptr,
                "vals": al.vals.ptr,
                "delta_lds": al.dl.ptr,
                "latq": ul.latq.ptr,
                "latsc": ul.latsc.ptr,
                "av": ul.av.ptr,
                "interq": dl.interq.ptr,
                "intsc": dl.intsc.ptr,
                "contrib": dl.contrib.ptr,
                "scan": rl.scan.ptr,
                "rt": {
                    "route": rl.route.ptr,
                    "flag": rl.flag.ptr,
                    "expert": rl.expert.ptr,
                    "kof": rl.kof.ptr,
                },
                "xbuf": scratch + fx.Int64(lay["xrow"][0]),
                "xflags": NA,
                "tl_on": key.timeline,
                "tls": lds.tls.ptr,
                "ln": scratch + fx.Int64(lay["ln"][0]),
                "shp": scratch + fx.Int64(lay["shp"][0]),
                "args": {
                    "core": core,
                    "w_o": w_o,
                    "prefix": prefix,
                    "prefix_in": prefix_in,
                    "blocks": blocks,
                    "blk_sm": blk_sm,
                    "blk_sr": blk_sr,
                    "ares_nw": ares_nw,
                    "ares_qk": ares_qk,
                    "in_nw": in_nw,
                    "x_dbg": 0,
                    "w_gate": w_gate,
                    "bias": bias,
                    "w_ld": w_ld,
                    "w_sgu": w_sgu,
                    "w_sd": w_sd,
                    "w13": w13,
                    "w13s": w13s,
                    "w2": w2,
                    "w2s": w2s,
                    "ln_w": ln_w,
                    "w_up": w_up,
                    "out": out,
                },
            }
            c["rlp"] = pl.rlp.ptr
            c["tag_x"] = bool(key.opt & OPT_TAG_X)
            c["rlog_parts"] = bool(key.opt & OPT_RLOG_PARTS)
            c["batch"] = bool(key.opt & OPT_BATCH)
            if const_expr(key.opt & OPT_TAG_X):
                c["xt"] = sreg(scratch, tag_offset(), "xt")
            c["sent_ar"] = bool(key.opt & OPT_SENT)
            c["sent_latr"] = bool(key.opt & OPT_SENT_LATR)
            # the K1 that takes a deferred reduce reads tagged pairs
            c["sent_final"] = bool(key.opt & OPT_SENT_FINAL) and key.reduce
            if const_expr(key.opt & OPT_SENTS):
                # this layer parity's copy
                lpar = fx.Int64(layer % 2) * sent_half_bytes()
                for region, (roff, _) in sent_layout().items():
                    off = sent_offset() + roff
                    c[region + "_peer"] = [
                        shift(preg(b, off, region), lpar) for b in bases
                    ]
                    c[region + "_own"] = shift(preg(own, off, region), lpar)
            for region in (
                "part",
                "wgt",
                "msq",
                "xrdy",
                "sgu",
                "h",
                "rpart",
                "lpart",
                "route",
                "inter",
                "isc",
                "lmsq",
                "lnrdy",
                "shrdy",
            ):
                c[region] = sreg(scratch, lay[region][0], region)
            on, tls = key.timeline, lds.tls.ptr
            stamp_begin(on, tls, tid, TL_POINTS)
            # ---- o_proj on every CTA past AttnRes's
            eb = bid - E0
            lb = eb - k2b.R_TASKS
            is_r = (bid >= E0) & (eb < k2b.R_TASKS)
            is_l = (bid >= E0 + k2b.R_TASKS) & (lb < k2b.L_TASKS)
            # o_proj first (every AttnRes CTA waits on the slowest partial; loads
            # retire in order), the early weights after: they land before x does
            if is_r:
                stage_oproj(c, bid - O0)
                stamp(on, tls, tid, 1)
                rops = k2b.early_loads(c, 0, eb // k2b.R_PARTS, eb % k2b.R_PARTS)
                k2b.early_compute(c, 0, eb // k2b.R_PARTS, eb % k2b.R_PARTS, rops)
            if is_l:
                stage_oproj(c, bid - O0)
                stamp(on, tls, tid, 1)
                lops = k2b.early_loads(c, 3, lb // k2b.LQ_PARTS, lb % k2b.LQ_PARTS)
                k2b.early_compute(c, 3, lb // k2b.LQ_PARTS, lb % k2b.LQ_PARTS, lops)
            if (bid >= O0) & ~is_r & ~is_l:
                stage_oproj(c, bid - O0)
                stamp(on, tls, tid, 1)
                gi = bid - O0
                if gi < k2b.G_TASKS:
                    k2b.early_compute(c, 2, gi, 0, k2b.early_loads(c, 2, gi, 0))
            # ---- AttnRes: the reduced o_proj -> x; then route / SiTU
            if bid < NA:
                # PRELOAD: the slice's loads in flight across the o_proj poll
                sld = slice_loads(c, bid) if key.opt & OPT_PRELOAD else None
                k2a.stage_reduce(c, bid)
                stamp(on, tls, tid, 1)
                stage_slice(c, bid, sld)
                stamp(on, tls, tid, 2)
                if bid < s:
                    stage_gate(c, bid)
                stamp(on, tls, tid, 8)
                stage_mix(c, bid)
                if const_expr(key.opt & OPT_SENT):
                    # the words stage_reduce took, off the path to x
                    k2a.rearm_ar(c, bid)
                stamp(on, tls, tid, 9)
                stamp(on, tls, tid, 13)
                if bid < s:
                    k2b.stage_route(c, bid)
                stamp(on, tls, tid, 10)
                if (bid >= H0) & (bid < H0 + s):
                    k2b.stage_h(c, bid - H0)
            if bid >= NA:
                stamp(on, tls, tid, 2)
            # ---- the routed experts
            # the latent (ready ~10 us before the route) quantized first
            k2b.load_latq(c)
            stamp(on, tls, tid, 12)
            # the route table from a bitmask's wave scan
            nu = k2b.load_route2(c)
            stamp(on, tls, tid, 11)
            total = nu * k2b.UG_GROUPS
            last = total - 1
            # reversed placement: the round's leftover tasks go to the high CTAs,
            # whose early work ends first (the AttnRes / route CTAs start ug last)
            task = BLOCKS - 1 - bid
            ops = k2b.ug_loads(c, fx.min(task, last))
            while task < total:
                nxt = task + BLOCKS
                ops_n = k2b.ug_loads(c, fx.min(nxt, last))
                k2b.stage_ug(c, task, ops)
                task = nxt
                ops = ops_n
            stamp(on, tls, tid, 3)
            if bid < k2b.DOWN_TASKS:
                dops = k2b.down_loads(c, bid, nu, fx.min(c["wave"], nu - 1))
                k2b.load_inter(c)
                stamp(on, tls, tid, 15)
                k2b.stage_down(c, bid, nu, dops)
                stamp(on, tls, tid, 16)
            if const_expr(key.opt & OPT_SDOWN_FIRST):
                k2b.run_sdown_first(c, bid, w_sd)
                k2b.run_sdown_rest(c, bid, w_sd)
            else:
                k2b.run_sdown_rest(c, bid, w_sd)
                k2b.run_sdown(c, bid, w_sd)
            stamp(on, tls, tid, 4)
            if (bid >= k2b.LN0) & (bid < k2b.LN0 + k2b.LN_TASKS):
                k2b.stage_lnorm(c, bid - k2b.LN0)
            stamp(on, tls, tid, 5)
            if bid < k2b.UP_TASKS:
                k2b.stage_up(c, bid)
            stamp(on, tls, tid, 6)
            if const_expr(key.reduce):  # noqa: SIM102 (traced: `and` does not combine traced values)
                if (bid >= k2b.RED0) & (bid < k2b.RED0 + k2b.RED_TASKS):
                    k2b.stage_reduce(c, bid - k2b.RED0)
            stamp(on, tls, tid, 7)
            stamp_flush(on, tls, tl, tid, bid, TL_POINTS)

    @flyc.jit
    def launch(
        core: Int64,
        w_o: Int64,
        prefix: Int64,
        prefix_in: Int64,
        blocks: Int64,
        blk_sm: Int32,
        blk_sr: Int32,
        ares_nw: Int64,
        ares_qk: Int64,
        in_nw: Int64,
        w_gate: Int64,
        bias: Int64,
        w_ld: Int64,
        w_sgu: Int64,
        w_sd: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        ln_w: Int64,
        w_up: Int64,
        out: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
        tl: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        k2(
            core,
            w_o,
            prefix,
            prefix_in,
            blocks,
            blk_sm,
            blk_sr,
            ares_nw,
            ares_qk,
            in_nw,
            w_gate,
            bias,
            w_ld,
            w_sgu,
            w_sd,
            w13,
            w13s,
            w2,
            w2s,
            ln_w,
            w_up,
            out,
            scratch,
            peers,
            rank,
            epoch,
            layer,
            tl,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    _BUILDS[key] = launch
    return launch


TP_ = k2b.TP
