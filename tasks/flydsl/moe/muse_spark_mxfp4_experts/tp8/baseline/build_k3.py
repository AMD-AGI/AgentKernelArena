# SPDX-License-Identifier: Apache-2.0
"""Build the Muse Spark K3 launch (``build_k3``): the MXFP4 experts, the finalize
and its all-reduce, the sharded unpooler and the FFN boundary of one layer,
standalone (the merged layer launch reuses ``stages.ffn.run_k3`` on its own
context)."""

import flydsl.compiler as flyc
import flydsl.expr as fx
import dataclasses
from flydsl.expr import const_expr
from flydsl.expr.typing import Int32, Int64

from . import guard
from .epoch import kernel_symbol
from .ops import peer_bases
from .plan import BLOCKS, THREADS, WAVES, key_tuple
from .sync import Mailbox, preg, sreg
from .config import MAX_TOKENS
from .device import step_tag
from .layout import scratch_bytes
from . import ffn

# AKA: FlyDSL hashes the helpers of a kernel's own directory, which in this flat package are
# all of them, so the upstream transitive source digest (plan.kernel_sources) is not needed.
SOURCES = "aka-flat"
_STREAM = fx.Stream(None)


@dataclasses.dataclass(frozen=True)
class K3Build:
    """A K3 build: the step's row count, the TP / EP group size (the ranks the
    partials are pushed to and summed over; 1 for the one-GPU tests), whether the
    boundary is the last layer's (the stream rounded to bf16 before the final norm:
    compiled in here, or selected by the launch's ``last`` argument), and the fault
    injection site (SGLANG_AMD_MONO_FAULT_INJECT: the wait this rank stalls at, once
    ``fault_after`` launches have run)."""

    tokens: int
    npes: int = 8
    last: bool = False
    fault_site: int = 0
    fault_after: int = 0
    trace: bool = False
    fp8_unpool: bool = (
        False  # w_unpool is a packed FP8 copy (codes, then fp32 row scales)
    )
    tpx: bool = False  # TP experts: w13 / s13 / w2 / s2 are the rank's slices of every expert (ffn module notes)
    defer_k1: bool = False  # the boundary publishes a token's two norm scalars (``k3_nscal``) for the next K1
    xq_pre: bool = False  # the expert input's MXFP4 blocks come from K2 (``xinq`` arg), staged once for every token
    ug_dyn: bool = False  # the ug items past each CTA's first from a launch-wide queue (``k3_ugq``; merged + xq_pre)


def k3_context(
    key: K3Build,
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
    """K3's context: LDS views, the mailbox, the scratch and peer regions (this
    rank's own ``ffn`` / ``unp`` for polling, every rank's for the pushes, the peer
    bases loaded once), the CTA map and the arguments."""
    s, npes = key.tokens, key.npes
    tid = fx.thread_idx.x
    mb = Mailbox(tag)
    bases = peer_bases(peers, npes)
    ffn_off, unp_off = sym_layout["ffn"][0], sym_layout["unp"][0]
    c = {
        "S": s,
        "W": npes,
        "tid": tid,
        "bid": fx.block_idx.x,
        "lane": tid % 64,
        "wave": tid // 64,
        "red": lds.red.ptr,
        "xcol": lds.xcol.ptr,
        "misc": lds.misc.ptr,
        "sums": lds.sums.ptr,
        "pick_of": lds.pick_of.ptr,
        "cnt_of": lds.cnt_of.ptr,
        "tile_of": lds.tile_of.ptr,
        "nt": lds.nt.ptr,
        "put": mb.put,
        "put_bf": mb.put_bf,
        "put_words": mb.put_words,
        "poll": mb.poll,
        "args": args,
        "rank": rank,
        "last": key.last,
        "trace_on": key.trace,
        "layer": layer,
        "fp8_unpool": key.fp8_unpool,
        "bases": ffn.k3_bases(s, npes),
        "ar_parts": ffn.ar_parts(s),
        "own_ffn": preg(sym, ffn_off, "ffn"),
        "own_unp": preg(sym, unp_off, "unp"),
        "peer_ffn": [preg(base, ffn_off, "ffn") for base in bases],
        "peer_unp": [preg(base, unp_off, "unp") for base in bases],
        "tpx": key.tpx,
        "defer_k1": key.defer_k1,
        "xq_pre": key.xq_pre,
        "ug_dyn": key.ug_dyn,
        "start_of": lds.start_of.ptr,
    }
    if key.tpx:
        c.update(ffn.tpx_dims(npes))
        for region in ("midx", "vx", "ssx"):
            off = sym_layout[region][0]
            c["own_" + region] = preg(sym, off, region)
            c["peer_" + region] = [preg(base, off, region) for base in bases]
    else:
        c.update(
            ug_t=ffn.UG_T,
            dn_t=ffn.DN_T,
            up_row0=ffn.INTER,
            w13_rows=2 * ffn.INTER,
            w2_rows=ffn.MOE_HIDDEN,
        )
    for region, (off, _) in layout.items():
        c[region[3:]] = sreg(scratch, off, region)  # "k3_mid" -> c["mid"]
    return c


def build_k3(key: K3Build):
    """``@flyc.jit`` launcher of K3 for a step of ``key.tokens`` rows on a rank of a
    ``key.npes``-way TP / EP group. Arguments: ``xin`` bf16 [S, MOE_HIDDEN], ``ids``
    int32 [S, TOP_K], ``scores`` fp32 [S, TOP_K], ``local_start`` (the rank's first
    expert), the Quark u8 experts ``w13`` [32, 2 INTER, MOE_HIDDEN / 2] / ``s13``
    [32, 2 INTER, MOE_HIDDEN / 32] / ``w2`` [32, MOE_HIDDEN, INTER / 2] / ``s2``,
    ``pen_w`` bf16 [MOE_HIDDEN], ``res`` fp32 [S, HIDDEN] (the stream, updated in
    place), ``pfw`` bf16 [HIDDEN], ``alpha`` / ``beta`` fp32 [HIDDEN], ``npw`` bf16
    [HIDDEN], ``w_unpool`` bf16 [HIDDEN, MOE_HIDDEN], ``attn_out`` bf16 [S, HIDDEN]
    (the next attention input), ``scratch`` (``k3_scratch_layout`` bytes), ``sym``
    (this rank's peer buffer: ``k3_sym_layout`` bytes + the guard's flags), ``peers``
    (every rank's base, int64), ``rank``, ``epoch``, ``layer``, ``last`` (1: the
    last-layer rule, when the build did not compile it in)."""
    S, npes = key.tokens, key.npes
    assert 1 <= S <= MAX_TOKENS and 1 <= npes <= WAVES, (S, npes)
    name = kernel_symbol(
        "muse_spark_k3",
        s=S,
        w=npes,
        l=int(key.last),
        f=key.fault_site,
        t=int(key.trace),
        p8=int(key.fp8_unpool),
        x=int(key.tpx),
        d=int(key.defer_k1),
        xq=int(key.xq_pre),
        u=int(key.ug_dyn),
    )
    keyed = key_tuple(key, SOURCES)
    fault_site, fault_after = key.fault_site, key.fault_after
    layout = ffn.k3_scratch_layout()
    sym_layout = ffn.k3_sym_layout(npes)
    sym_bytes = scratch_bytes(sym_layout)
    Lds = ffn.k3_lds(S)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k3(
        xin: Int64,
        ids: Int64,
        scores: Int64,
        local_start: Int32,
        w13: Int64,
        s13: Int64,
        w2: Int64,
        s2: Int64,
        pen_w: Int64,
        res: Int64,
        pfw: Int64,
        alpha: Int64,
        beta: Int64,
        npw: Int64,
        w_unpool: Int64,
        attn_out: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
        last: Int32,
    ):
        # every wait runs under the guard; with peers, a failure flags every rank
        if const_expr(npes > 1):
            g = guard.Guard(
                epoch, own=sym, peers=peers, rank=rank, npes=npes, flags_offset=sym_bytes,
                fault_site=fault_site or None, fault_after=fault_after,
            )  # fmt: skip
        else:
            g = guard.Guard(
                epoch, fault_site=fault_site or None, fault_after=fault_after
            )
        with guard.scope(g):
            lds = fx.SharedAllocator().allocate(Lds).peek()
            args = {
                "xin": xin, "ids": ids, "scores": scores, "local_start": local_start, "w13": w13,
                "s13": s13, "w2": w2, "s2": s2, "pen_w": pen_w, "res": res, "pfw": pfw,
                "alpha": alpha, "beta": beta, "npw": npw, "w_unpool": w_unpool,
                "attn_out": attn_out, "last": last,
            }  # fmt: skip
            c = k3_context(
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
            ffn.run_k3(c)

    @flyc.jit
    def launch(
        xin: Int64,
        ids: Int64,
        scores: Int64,
        local_start: Int32,
        w13: Int64,
        s13: Int64,
        w2: Int64,
        s2: Int64,
        pen_w: Int64,
        res: Int64,
        pfw: Int64,
        alpha: Int64,
        beta: Int64,
        npw: Int64,
        w_unpool: Int64,
        attn_out: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
        last: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        k3(
            xin, ids, scores, local_start, w13, s13, w2, s2, pen_w, res, pfw, alpha, beta,
            npw, w_unpool, attn_out, scratch, sym, peers, rank, epoch, layer, last,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    return launch
