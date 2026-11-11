# SPDX-License-Identifier: Apache-2.0
"""Build the Muse Spark K-attn launch (``build_kattn``) standalone.

AKA: vendored from sgl-amd-mono @ 52486703 (sgl_amd_mono/muse_spark/gfx950/layer.py), keeping only
the K-attn build (``KAttnBuild``, ``_attn_views``, ``kattn_context``, ``build_kattn``) unchanged
apart from these: imports are package-relative; the build key is a frozen dataclass instead of a
``msgspec.Struct``; ``compat.install()`` is not called (the modules call compat.py's llvm-dialect
bridge directly); the transitive source digest ``plan.kernel_sources`` is replaced by a constant
tag (FlyDSL hashes the helpers of a kernel's own directory, which in this flat package are all of
them). The K1 and whole-layer builds are not part of this task."""

import dataclasses

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import Int32, Int64

from . import attention, guard
from .config import MAX_TOKENS
from .device import step_tag
from .epoch import kernel_symbol
from .layout import attn_rows, kattn_scratch_layout
from .plan import BLOCKS, THREADS, key_tuple
from .sync import Mailbox, sreg

SOURCES = "aka-flat"
_STREAM = fx.Stream(None)


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
