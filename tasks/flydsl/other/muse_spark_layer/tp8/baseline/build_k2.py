# SPDX-License-Identifier: Apache-2.0
"""Build the Muse Spark K2 launch standalone (``build_k2``): the attention
output through the experts' input, with the attention all-reduce embedded
(every rank's bf16 partial rows pushed into every rank's peer buffer, summed in
rank order). To be merged into ``layer.py`` with the other builders."""

import flydsl.compiler as flyc
import flydsl.expr as fx
import dataclasses
from flydsl.expr.typing import Int32, Int64

from . import guard
from .epoch import kernel_symbol
from .ops import peer_bases
from .plan import BLOCKS, THREADS, key_tuple
from .sync import Mailbox, preg, sreg
from .config import MAX_TOKENS
from .device import step_tag
from . import post_attn
from .post_attn import (
    k2_scratch_layout,
    sym_bytes,
    sym_layout,
)

# AKA: FlyDSL hashes the helpers of a kernel's own directory, which in this flat package are
# all of them, so the upstream transitive source digest (plan.kernel_sources) is not needed.
SOURCES = "aka-flat"
_STREAM = fx.Stream(None)

# the launch's pointer arguments, in order (the scalars follow: rank, epoch, layer)
K2_ARGS = (
    "o",  # bf16 [S, Q_K]: the attention output
    "g",  # bf16 [S, Q_K]: K1's o_gate
    "res",  # fp32 [S, HIDDEN]: the residual stream, updated in place
    "w_o",  # bf16 [HIDDEN, Q_K]: o_proj (this rank's K shard)
    "alpha",  # fp32 [HIDDEN]: the attention residual gate's stream coefficient
    "beta",  # fp32 [HIDDEN]: its branch coefficient
    "pre_w",  # bf16 [HIDDEN]: pre_feedforward_layernorm's effective weight
    "w_hi",  # bf16 [EXPERTS, HIDDEN]: the fp32 router's high part (packed rows 4096 ..)
    "w_lo",  # bf16 [EXPERTS, HIDDEN]: its low part (packed rows 4352 ..)
    "bias",  # fp32 [EXPERTS]: expert_bias
    "w_pool",  # bf16 [MOE_HIDDEN, HIDDEN]: pre_expert_proj
    "xn_w",  # bf16 [MOE_HIDDEN]: pre_expert_norm's effective weight
    "ids",  # int32 [S, TOP_K] out
    "scores",  # fp32 [S, TOP_K] out
    "xin",  # bf16 [S, MOE_HIDDEN] out: the expert input
    "scratch",  # this rank's scratch (k2_scratch_layout)
    "sym",  # this rank's peer buffer base (sym_layout, then the guard flags)
    "peers",  # int64 [npes]: every rank's peer buffer base
)


@dataclasses.dataclass(frozen=True)
class K2Build:
    """A K2 build: the step's row count, the TP size (the ranks of the embedded
    all-reduce) and the fault injection site (SGLANG_AMD_MONO_FAULT_INJECT: the
    wait this rank stalls at, once ``fault_after`` launches have run)."""

    tokens: int
    npes: int = 8
    fault_site: int = 0
    fault_after: int = 0
    trace: bool = False
    fp8_pool: bool = False  # w_pool is a packed FP8 copy (codes, then fp32 row scales)
    defer_k1: bool = (
        False  # the boundary also writes the next K1's first operand half (``k2_c1``)
    )
    xq_pre: bool = (
        False  # xn also writes the expert input as MXFP4 blocks (``k2_xinq``) for K3
    )
    pshard: bool = False  # w_pool is the rank's 1 / npes of the pooler rows; partials to every rank (``poolx``)


def k2_context(
    key: K2Build,
    lds,
    args: dict,
    scratch,
    sym,
    peers,
    rank,
    tag,
    layout: dict,
    layer=None,
) -> dict:
    """K2's context: LDS views, the mailbox, the scratch and peer regions, the
    ranks' peer buffer bases (loaded once a CTA) and the arguments."""
    tid = fx.thread_idx.x
    mb = Mailbox(tag)
    sy = sym_layout(key.npes)
    c = {
        "S": key.tokens,
        "W": key.npes,
        "tid": tid,
        "bid": fx.block_idx.x,
        "lane": tid % 64,
        "wave": tid // 64,
        "xs": lds.xs.ptr,
        "red": lds.red.ptr,
        "misc": lds.misc.ptr,
        "sig": lds.sig.ptr,
        "put": mb.put,
        "put_bf": mb.put_bf,
        "put_words": mb.put_words,
        "poll": mb.poll,
        "args": args,
        "rank": rank,
        "sym": sym,
        "bases": peer_bases(peers, key.npes),
        "sy_attn": sy["attn"][0],
        "own_attn": preg(sym, sy["attn"][0], "attn"),
        "trace_on": key.trace,
        "layer": layer,
        "fp8_pool": key.fp8_pool,
        "defer_k1": key.defer_k1,
        "xq_pre": key.xq_pre,
        "pshard": key.pshard,
        "own_poolx": preg(sym, sy["poolx"][0], "poolx"),
    }
    c["peer_poolx"] = [preg(base, sy["poolx"][0], "poolx") for base in c["bases"]]
    for region, (off, _) in layout.items():
        if region in ("k2_xbuf", "k2_xbuf2", "k2_c1", "k2_xinq"):
            c[region[3:]] = scratch + fx.Int64(
                off
            )  # plain bf16 words, not a mailbox region
        else:
            c[region[3:]] = sreg(scratch, off, region)  # "k2_og" -> c["og"]
    return c


def build_k2(key: K2Build):
    """``@flyc.jit`` launcher of K2 for a step of ``key.tokens`` rows on a rank
    of a ``key.npes``-way TP group."""
    S, W = key.tokens, key.npes
    assert 1 <= S <= MAX_TOKENS and 1 <= W <= guard.FLAG_WORDS
    name = kernel_symbol(
        "muse_spark_k2",
        s=S,
        w=W,
        f=key.fault_site,
        t=int(key.trace),
        p8=int(key.fp8_pool),
        d=int(key.defer_k1),
        xq=int(key.xq_pre),
        ps=int(key.pshard),
    )
    keyed = key_tuple(key, SOURCES)
    fault_site, fault_after = key.fault_site, key.fault_after
    flags_offset = sym_bytes(W)
    layout = k2_scratch_layout()
    Lds = post_attn.k2_lds(S)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k2(
        o: Int64,
        g: Int64,
        res: Int64,
        w_o: Int64,
        alpha: Int64,
        beta: Int64,
        pre_w: Int64,
        w_hi: Int64,
        w_lo: Int64,
        bias: Int64,
        w_pool: Int64,
        xn_w: Int64,
        ids: Int64,
        scores: Int64,
        xin: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
    ):
        # every wait runs under the guard: a deadline, the error words, and the
        # peers' flags at the tail of each rank's peer buffer
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
            )
        ):
            lds = fx.SharedAllocator().allocate(Lds).peek()
            args = {
                "o": o, "g": g, "res": res, "w_o": w_o, "alpha": alpha, "beta": beta,
                "pre_w": pre_w, "w_hi": w_hi, "w_lo": w_lo, "bias": bias, "w_pool": w_pool,
                "xn_w": xn_w, "ids": ids, "scores": scores, "xin": xin,
            }  # fmt: skip
            c = k2_context(
                key,
                lds,
                args,
                scratch,
                sym,
                peers,
                rank,
                step_tag(epoch, layer),
                layout,
                layer=layer,
            )
            post_attn.run_k2(c)

    @flyc.jit
    def launch(
        o: Int64,
        g: Int64,
        res: Int64,
        w_o: Int64,
        alpha: Int64,
        beta: Int64,
        pre_w: Int64,
        w_hi: Int64,
        w_lo: Int64,
        bias: Int64,
        w_pool: Int64,
        xn_w: Int64,
        ids: Int64,
        scores: Int64,
        xin: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        k2(
            o, g, res, w_o, alpha, beta, pre_w, w_hi, w_lo, bias, w_pool, xn_w, ids, scores,
            xin, scratch, sym, peers, rank, epoch, layer,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    return launch
