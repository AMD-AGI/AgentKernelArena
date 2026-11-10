# SPDX-License-Identifier: Apache-2.0
"""Build the Muse Spark launches: K1 (``build_k1``) and K-attn (``build_kattn``) standalone,
and the whole layer (``build_layer``: attention, K2, K3 and the next layer's K1 in one launch)."""

import flydsl.compiler as flyc
import flydsl.expr as fx
import dataclasses
from flydsl.expr import const_expr, gpu
from flydsl.expr.typing import Int32, Int64

from . import guard
from .epoch import kernel_symbol
from .ops import CM_DEV
from .plan import BLOCKS, THREADS, key_tuple
from .sync import Mailbox, sreg
from .config import MAX_TOKENS
from .device import global_atomic, step_tag
from .layout import (
    attn_rows,
    k1_scratch_layout,
    kattn_scratch_layout,
)
from . import attention, pre_attn

# AKA: FlyDSL hashes the helpers of a kernel's own directory, which in this flat package are
# all of them, so the upstream transitive source digest (plan.kernel_sources) is not needed.
SOURCES = "aka-flat"
_STREAM = fx.Stream(None)


@dataclasses.dataclass(frozen=True)
class K1Build:
    """A K1 build: the step's row count and the fault injection site
    (SGLANG_AMD_MONO_FAULT_INJECT: the wait this rank stalls at, once
    ``fault_after`` launches have run)."""

    tokens: int
    fault_site: int = 0
    fault_after: int = 0
    trace: bool = False  # stage stamps into the ``k1_trace`` region (``device.stamp``)
    defer: bool = False  # a merged launch's K1 with the boundary's norms deferred (``stage_gemv_deferred``)


def k1_context(
    key: K1Build,
    lds,
    args: dict,
    scratch,
    tag,
    layout: dict,
    merged: bool = False,
    layer=None,
) -> dict:
    """K1's context: LDS views, the mailbox, the scratch regions and arguments.
    ``merged``: x comes from this launch's earlier stages (device-scope loads)."""
    s = key.tokens
    tid = fx.thread_idx.x
    mb = Mailbox(tag)
    c = {
        "S": s,
        "tid": tid,
        "bid": fx.block_idx.x,
        "lane": tid % 64,
        "wave": tid // 64,
        "red": lds.red.ptr,
        "put": mb.put,
        "poll": mb.poll,
        "args": args,
        "x_cm": CM_DEV
        if merged
        else 0,  # x from this launch: past the L2 (another XCD wrote it)
        "defer": key.defer,
        "trace_on": key.trace,
        "layer": layer,
    }
    for region, (off, _) in layout.items():
        c[region[3:]] = sreg(scratch, off, region)  # "k1_part" -> c["part"]
    return c


def build_k1(key: K1Build):
    """``@flyc.jit`` launcher of K1 for a step of ``key.tokens`` rows."""
    tokens = key.tokens
    assert 1 <= tokens <= MAX_TOKENS
    name = kernel_symbol("muse_spark_k1", s=tokens, f=key.fault_site, t=int(key.trace))
    keyed = key_tuple(key, SOURCES)
    fault_site, fault_after = key.fault_site, key.fault_after
    layout = k1_scratch_layout()
    Lds = pre_attn.k1_lds(tokens)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k1(
        x: Int64,
        w: Int64,
        cos_sin: Int64,
        positions: Int64,
        slots: Int64,
        swa_map: Int64,
        k_cache: Int64,
        v_cache: Int64,
        q_out: Int64,
        k_out: Int64,
        v_out: Int64,
        g_out: Int64,
        scratch: Int64,
        epoch: Int64,
        layer: Int32,
        use_rope: Int32,
        use_swa: Int32,
        slot64: Int32,
        swa64: Int32,
    ):
        # every wait runs under the guard (deadline, error words); K1 exchanges
        # nothing with its peers, so a failure stays on this GPU
        with guard.scope(
            guard.Guard(epoch, fault_site=fault_site or None, fault_after=fault_after)
        ):
            lds = fx.SharedAllocator().allocate(Lds).peek()
            args = {
                "x": x, "w": w, "cos_sin": cos_sin, "positions": positions, "slots": slots,
                "swa_map": swa_map, "k_cache": k_cache, "v_cache": v_cache, "q_out": q_out,
                "k_out": k_out, "v_out": v_out, "g_out": g_out, "use_rope": use_rope,
                "use_swa": use_swa, "slot64": slot64, "swa64": swa64,
            }  # fmt: skip
            c = k1_context(
                key, lds, args, scratch, step_tag(epoch, layer), layout, layer=layer
            )
            pre_attn.run_k1(c, c["bid"])

    @flyc.jit
    def launch(
        x: Int64,
        w: Int64,
        cos_sin: Int64,
        positions: Int64,
        slots: Int64,
        swa_map: Int64,
        k_cache: Int64,
        v_cache: Int64,
        q_out: Int64,
        k_out: Int64,
        v_out: Int64,
        g_out: Int64,
        scratch: Int64,
        epoch: Int64,
        layer: Int32,
        use_rope: Int32,
        use_swa: Int32,
        slot64: Int32,
        swa64: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        k1(
            x, w, cos_sin, positions, slots, swa_map, k_cache, v_cache, q_out, k_out,
            v_out, g_out, scratch, epoch, layer, use_rope, use_swa, slot64, swa64,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    return launch


@dataclasses.dataclass(frozen=True)
class KAttnBuild:
    """A K-attn build: the step's row count and rows a request, the row tiles a
    task shares its K / V loads over (clamped to the row tiles of ``q_len``; four
    spill registers, two fit), and the fault injection site."""

    tokens: int
    q_len: int = 8
    nt: int = 2
    fault_site: int = 0
    fault_after: int = 0
    trace: bool = False


def _attn_views(lds):
    """The K-attn LDS views of a union holding them as (loop, tree) or (aloop, atree)."""
    if hasattr(lds, "aloop"):
        return lds.aloop.peek(), lds.atree.peek()
    return lds.loop.peek(), lds.tree.peek()


def kattn_context(
    key: KAttnBuild, lds, args: dict, scratch, tag, layout: dict, layer=None
) -> dict:
    """K-attn's context: LDS views, the mailbox, the scratch region and arguments."""
    tid = fx.thread_idx.x
    mb = Mailbox(tag)
    c = {
        "S": key.tokens,
        "QL": key.q_len,
        "NT": min(key.nt, (attn_rows(key.q_len) + 15) // 16),
        "tid": tid,
        "bid": fx.block_idx.x,
        "lane": tid % 64,
        "wave": tid // 64,
        **attention.lds_views(*_attn_views(lds)),
        "put": mb.put,
        "poll": mb.poll,
        "args": args,
        "trace_on": key.trace,
        "layer": layer,
    }
    for region in ("attn_part", "attn_pflag", "attn_l2", "attn_l2flag", "attn_trace"):
        c[region[5:]] = sreg(
            scratch, layout[region][0], region
        )  # "attn_part" -> c["part"]
    return c


def build_kattn(key: KAttnBuild):
    """``@flyc.jit`` launcher of K-attn for ``key.tokens`` rows, ``key.q_len`` a request."""
    S, QL, NT = key.tokens, key.q_len, key.nt
    assert QL in (1, 8) and S % QL == 0 and 1 <= S <= MAX_TOKENS
    row_tiles = (attn_rows(QL) + 15) // 16
    NT = min(NT, row_tiles)
    assert row_tiles % NT == 0, (row_tiles, NT)
    name = kernel_symbol(
        "muse_spark_kattn", s=S, q=QL, n=NT, f=key.fault_site, t=int(key.trace)
    )
    keyed = key_tuple(key, SOURCES)
    fault_site, fault_after = key.fault_site, key.fault_after
    layout = kattn_scratch_layout(S, QL)
    Loop, Tree = attention.kattn_lds(NT)

    @fx.union
    class Lds:
        loop: Loop  # type: ignore[valid-type]
        tree: Tree  # type: ignore[valid-type]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kattn(
        q: Int64,
        o: Int64,
        k_cache: Int64,
        v_cache: Int64,
        page_table: Int64,
        table_stride: Int32,  # int32 entries a request's row of the table
        table_bytes: Int32,  # the table's size: a key past the row reads page 0
        seq_lens: Int64,  # int32 / int64: the request's keys before this step's rows
        lens64: Int32,  # seq_lens is int64
        extra: Int32,  # keys past seq_lens (the step's rows a verify adds: QL; 0 for decode)
        window: Int32,  # the sliding window (keys), 0 for full attention
        part_in: Int32,  # keys a partition (a multiple of CHUNK); 0: sized here
        scale_bits: Int32,  # the softmax scale, f32 bits
        scratch: Int64,
        epoch: Int64,
        layer: Int32,
    ):
        with guard.scope(
            guard.Guard(epoch, fault_site=fault_site or None, fault_after=fault_after)
        ):
            lds = fx.SharedAllocator().allocate(Lds)
            args = {
                "q": q, "o": o, "k_cache": k_cache, "v_cache": v_cache, "page_table": page_table,
                "table_stride": table_stride, "table_bytes": table_bytes, "seq_lens": seq_lens,
                "lens64": lens64, "extra": extra, "window": window, "part_in": part_in,
                "scale_bits": scale_bits,
            }  # fmt: skip
            c = kattn_context(
                key, lds, args, scratch, step_tag(epoch, layer), layout, layer=layer
            )
            attention.run_kattn(c)

    @flyc.jit
    def launch(
        q: Int64,
        o: Int64,
        k_cache: Int64,
        v_cache: Int64,
        page_table: Int64,
        table_stride: Int32,
        table_bytes: Int32,
        seq_lens: Int64,
        lens64: Int32,
        extra: Int32,
        window: Int32,
        part_in: Int32,
        scale_bits: Int32,
        scratch: Int64,
        epoch: Int64,
        layer: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        kattn(
            q, o, k_cache, v_cache, page_table, table_stride, table_bytes, seq_lens, lens64,
            extra, window, part_in, scale_bits, scratch, epoch, layer,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    return launch


@dataclasses.dataclass(frozen=True)
class LayerBuild:
    """A whole-layer build: the step's row count and rows a request, the TP / EP
    size, whether the next layer's K1 ends the launch, whether the FFN boundary is
    the last layer's, and the fault injection site."""

    tokens: int
    q_len: int = 8
    npes: int = 8
    next_k1: bool = True
    last: bool = False
    fault_site: int = 0
    fault_after: int = 0
    trace: bool = False
    fp8_dense: bool = False  # the pooler and unpooler as packed FP8 copies
    tpx: bool = False  # TP experts (``K3Build.tpx``)
    defer_k1: bool = (
        False  # the next K1 folds the FFN boundary's norms itself (needs next_k1)
    )
    xq_pre: bool = True  # K2's xn hands K3 the expert input as MXFP4 blocks (S <= 16; K3Build.xq_pre)
    ug_dyn: bool = (
        False  # K3's ug items from a launch-wide queue (with xq_pre; K3Build.ug_dyn)
    )
    pshard: bool = (
        False  # the pooler sharded by rows over the ranks (K2Build.pshard; npes == 8)
    )


def build_layer(key: LayerBuild):
    """``@flyc.jit`` launcher of a whole MXFP4 layer: the attention, K2, K3 and,
    with ``key.next_k1``, the next layer's K1, in one launch for ``key.tokens``
    rows, ``key.q_len`` a request, on a rank of a ``key.npes``-way TP / EP group.

    Every stage boundary is a mailbox flag: the attention's merge tasks flag each
    (request, KV head, row tile)'s o rows for K2's og tasks; K2's route / xn tasks
    flag each token for K3; K3's boundary task flags each token's x for the next
    layer's K1 (under that layer's tag). The LDS is the union of the stages' (a
    CTA is done with a stage's rows before it starts the next); the stages'
    scratch regions stay separate. A CTA runs the next layer's K1 only after its
    K3 stages, which every CTA's attention and K2 stages precede (K3's chain of
    flags runs through every token's route and expert input), so K1's q / g
    outputs are written after every read of this layer's.
    """
    from .build_k2 import K2Build, k2_context
    from .build_k3 import K3Build, k3_context
    from . import ffn, post_attn
    from .post_attn import (
        k2_scratch_layout,
        sym_bytes,
    )

    S, QL, W, next_k1, last = key.tokens, key.q_len, key.npes, key.next_k1, key.last
    assert (
        1 <= S <= MAX_TOKENS
        and S % QL == 0
        and QL in (1, 8)
        and 1 <= W <= guard.FLAG_WORDS
    )
    name = kernel_symbol("muse_spark_layer", s=S, q=QL, w=W, n=int(next_k1), l=int(last), f=key.fault_site, t=int(key.trace), p8=int(key.fp8_dense), x=int(key.tpx), d=int(key.defer_k1 and next_k1), xq=int(key.xq_pre and S <= 16), u=int(key.ug_dyn and key.xq_pre and S == 16), ps=int(key.pshard))  # fmt: skip
    keyed = key_tuple(key, SOURCES)
    fault_site, fault_after, tr = key.fault_site, key.fault_after, key.trace
    defer = key.defer_k1 and next_k1
    k1_key = K1Build(S, fault_site, fault_after, tr, defer)
    ka_key = KAttnBuild(S, QL, fault_site=fault_site, fault_after=fault_after, trace=tr)
    xq = key.xq_pre and S <= 16
    assert not key.pshard or (W == 8 and not key.fp8_dense), (
        "the sharded pooler: TP 8, bf16"
    )
    k2_key = K2Build(
        S, W, fault_site, fault_after, tr, key.fp8_dense, defer, xq, key.pshard
    )
    ugd = (
        key.ug_dyn and xq and S >= 16
    )  # (S = 8: neutral, 3.3 items a CTA; S = 16: -0.2 ms a step)
    k3_key = K3Build(
        S, W, last, fault_site, fault_after, tr, key.fp8_dense, key.tpx, defer, xq, ugd
    )
    k1_layout, ka_layout = k1_scratch_layout(), kattn_scratch_layout(S, QL)
    k2_layout, k3_layout = k2_scratch_layout(), ffn.k3_scratch_layout()
    sy = ffn.k3_sym_layout(W)
    flags_offset = sym_bytes(W)
    ALoop, ATree = attention.kattn_lds(min(ka_key.nt, (attn_rows(QL) + 15) // 16))
    K2Lds, K3Lds, K1Lds = post_attn.k2_lds(S), ffn.k3_lds(S), pre_attn.k1_lds(S)

    @fx.union
    class Lds:
        aloop: ALoop  # type: ignore[valid-type]
        atree: ATree  # type: ignore[valid-type]
        k2: K2Lds  # type: ignore[valid-type]
        k3: K3Lds  # type: ignore[valid-type]
        k1: K1Lds  # type: ignore[valid-type]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def layer_(
        # the attention
        q: Int64, o: Int64, k_cache: Int64, v_cache: Int64, page_table: Int64, table_stride: Int32,
        table_bytes: Int32, seq_lens: Int64, lens64: Int32, extra: Int32, window: Int32, part_in: Int32,
        scale_bits: Int32, scratch_a: Int64,
        # K2
        g: Int64, res: Int64, w_o: Int64, alpha: Int64, beta: Int64, pre_w: Int64, w_hi: Int64,
        w_lo: Int64, bias: Int64, w_pool: Int64, xn_w: Int64, ids: Int64, scores: Int64, xin: Int64,
        scratch2: Int64,
        # K3
        local_start: Int32, w13: Int64, s13: Int64, w2: Int64, s2: Int64, pen_w: Int64, pfw: Int64,
        alpha2: Int64, beta2: Int64, npw: Int64, w_unpool: Int64, attn_out: Int64, scratch3: Int64,
        # the next layer's K1 (its x: attn_out; its q / g outputs: this launch's q / g)
        w1: Int64, cos_sin: Int64, positions: Int64, slots: Int64, swa_map: Int64, k_cache1: Int64,
        v_cache1: Int64, k_out: Int64, v_out: Int64, scratch1: Int64, use_rope1: Int32, use_swa1: Int32,
        slot64: Int32, swa64: Int32,
        # the launch
        sym: Int64, peers: Int64, rank: Int32, epoch: Int64, layer: Int32, last_in: Int32,
    ):  # fmt: skip
        with guard.scope(
            guard.Guard(
                epoch,
                own=sym,
                peers=peers,
                rank=rank,
                npes=W,
                flags_offset=flags_offset,
                fault_site=fault_site or None,
                fault_after=fault_after,
            )  # fmt: skip
        ):
            lds = fx.SharedAllocator().allocate(Lds)
            tag = step_tag(epoch, layer)
            tag1 = step_tag(epoch, layer + 1)
            bid = fx.block_idx.x
            if const_expr(ugd):
                # K3's ug queue starts at 0: CTA 0 zeroes it before its xn task (token 0),
                # whose flag (after a release) every K3 CTA polls before its first ticket
                if (bid == 0) & (fx.thread_idx.x == 0):
                    global_atomic(
                        scratch3 + fx.Int64(k3_layout["k3_ugq"][0]), 0, xchg=True
                    )
            # the stages' contexts share the layer's tag; the next K1 takes the next layer's
            a_args = {
                "q": q, "o": o, "k_cache": k_cache, "v_cache": v_cache, "page_table": page_table,
                "table_stride": table_stride, "table_bytes": table_bytes, "seq_lens": seq_lens,
                "lens64": lens64, "extra": extra, "window": window, "part_in": part_in,
                "scale_bits": scale_bits,
            }  # fmt: skip
            k2_args = {
                "o": o, "g": g, "res": res, "w_o": w_o, "alpha": alpha, "beta": beta, "pre_w": pre_w,
                "w_hi": w_hi, "w_lo": w_lo, "bias": bias, "w_pool": w_pool, "xn_w": xn_w, "ids": ids,
                "scores": scores, "xin": xin, "npw": npw, "alpha2": alpha2,
            }  # fmt: skip
            k3_args = {
                "xin": xin, "ids": ids, "scores": scores, "local_start": local_start, "w13": w13,
                "s13": s13, "w2": w2, "s2": s2, "pen_w": pen_w, "res": res, "pfw": pfw, "alpha": alpha2,
                "beta": beta2, "npw": npw, "w_unpool": w_unpool, "attn_out": attn_out, "last": last_in,
            }  # fmt: skip
            c2 = k2_context(
                k2_key,
                lds.k2.peek(),
                k2_args,
                scratch2,
                sym,
                peers,
                rank,
                tag,
                k2_layout,
                layer=layer,
            )
            k3_args["xinq"] = c2["xinq"]  # (read by K3 only with xq_pre)
            oflag, rflag, xinflag = c2["oflag"], c2["rflag"], c2["xinflag"]
            ca = kattn_context(
                ka_key, lds, a_args, scratch_a, tag, ka_layout, layer=layer
            )
            attention.run_kattn(ca, o_flag=oflag)
            gpu.barrier()  # this CTA is done with the attention's LDS
            post_attn.run_k2(
                c2, route_flag=rflag, xin_flag=xinflag, o_flag=oflag, QL=QL
            )
            gpu.barrier()
            c3 = k3_context(
                k3_key,
                lds.k3.peek(),
                k3_args,
                scratch3,
                sym,
                peers,
                rank,
                tag,
                k3_layout,
                sy,
                layer=layer,
            )
            if const_expr(next_k1):
                k1_args = {
                    "x": c2["c1"] if defer else attn_out, "w": w1, "pfw": pfw, "beta2": beta2, "npw": npw, "cos_sin": cos_sin, "positions": positions, "slots": slots,
                    "swa_map": swa_map, "k_cache": k_cache1, "v_cache": v_cache1, "q_out": q,
                    "k_out": k_out, "v_out": v_out, "g_out": g, "use_rope": use_rope1,
                    "use_swa": use_swa1, "slot64": slot64, "swa64": swa64,
                }  # fmt: skip
                c1 = k1_context(
                    k1_key,
                    lds.k1.peek(),
                    k1_args,
                    scratch1,
                    tag1,
                    k1_layout,
                    merged=True,
                    layer=layer + 1,
                )
                if const_expr(defer):
                    # the unpooled rows and the scalars are this layer's (K3's tag)
                    c1.update(
                        d_poll=c3["poll"], d_unp=c3["own_unp"], d_nscal=c3["nscal"]
                    )
                ffn.run_k3(
                    c3,
                    route_flag=rflag,
                    xin_flag=xinflag,
                    x_flag=c1["xflag"],
                    x_put=c1["put"],
                    res_flag=(c2["resflag"], S * post_attn.a_parts(S)),
                )
                gpu.barrier()
                pre_attn.run_k1(c1, bid, x_flag=c1["xflag"])
            else:
                ffn.run_k3(
                    c3,
                    route_flag=rflag,
                    xin_flag=xinflag,
                    res_flag=(c2["resflag"], S * post_attn.a_parts(S)),
                )

    @flyc.jit
    def launch(
        q: Int64, o: Int64, k_cache: Int64, v_cache: Int64, page_table: Int64, table_stride: Int32,
        table_bytes: Int32, seq_lens: Int64, lens64: Int32, extra: Int32, window: Int32, part_in: Int32,
        scale_bits: Int32, scratch_a: Int64,
        g: Int64, res: Int64, w_o: Int64, alpha: Int64, beta: Int64, pre_w: Int64, w_hi: Int64,
        w_lo: Int64, bias: Int64, w_pool: Int64, xn_w: Int64, ids: Int64, scores: Int64, xin: Int64,
        scratch2: Int64,
        local_start: Int32, w13: Int64, s13: Int64, w2: Int64, s2: Int64, pen_w: Int64, pfw: Int64,
        alpha2: Int64, beta2: Int64, npw: Int64, w_unpool: Int64, attn_out: Int64, scratch3: Int64,
        w1: Int64, cos_sin: Int64, positions: Int64, slots: Int64, swa_map: Int64, k_cache1: Int64,
        v_cache1: Int64, k_out: Int64, v_out: Int64, scratch1: Int64, use_rope1: Int32, use_swa1: Int32,
        slot64: Int32, swa64: Int32,
        sym: Int64, peers: Int64, rank: Int32, epoch: Int64, layer: Int32, last_in: Int32,
        stream: fx.Stream = _STREAM,
    ):  # fmt: skip
        _ = keyed
        layer_(
            q, o, k_cache, v_cache, page_table, table_stride, table_bytes, seq_lens, lens64, extra,
            window, part_in, scale_bits, scratch_a, g, res, w_o, alpha, beta, pre_w, w_hi, w_lo, bias,
            w_pool, xn_w, ids, scores, xin, scratch2, local_start, w13, s13, w2, s2, pen_w, pfw, alpha2,
            beta2, npw, w_unpool, attn_out, scratch3, w1, cos_sin, positions, slots, swa_map, k_cache1,
            v_cache1, k_out, v_out, scratch1, use_rope1, use_swa1, slot64, swa64, sym, peers, rank,
            epoch, layer, last_in,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    return launch
