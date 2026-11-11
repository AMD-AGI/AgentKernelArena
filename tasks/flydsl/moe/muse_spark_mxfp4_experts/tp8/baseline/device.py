# SPDX-License-Identifier: Apache-2.0
"""Muse Spark's device helpers that ``sgl_amd_mono.common`` does not hold: the
step's mailbox tag and the fences of a plain-words hand-off (on the llvm dialect
directly, which every FlyDSL this family runs on provides)."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from . import buffer_ops as bo
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, rocdl
from flydsl.expr.typing import T, as_ir_value

from .ops import clock, rsrc, uniform
from .plan import THREADS
from .config import LAYER_SLOTS


def step_tag(epoch_ptr, layer):
    """The layer's mailbox tag this step, ``step * LAYER_SLOTS + layer + 1``: the
    step is word 0 of the epoch buffer, which the host moves on after a step's last
    launch. Graph-safe, and no mailbox is cleared between steps."""
    step = uniform(bo.buffer_load(rsrc(epoch_ptr), 0, vec_width=1, dtype=T.i32))
    return step * LAYER_SLOTS + layer + 1


def release_stores():
    """Order this wave's earlier device-scope stores before a flag that other CTAs
    poll: wait for them (a workgroup release fence alone does not wait for global
    stores), then the fence."""
    rocdl.s_waitcnt(vmcnt=0)
    llvm.FenceOp(llvm.AtomicOrdering.release, syncscope="workgroup")


def acquire_loads():
    """Order this thread's later loads after a flag it polled."""
    llvm.FenceOp(llvm.AtomicOrdering.acquire, syncscope="workgroup")


# ---------------------------------------------------------------- stage trace
TRACE_SLOTS = 16  # stamps a CTA: int64 clock ticks (100 MHz) at trace[((layer % TRACE_LAYERS) BLOCKS + bid) TRACE_SLOTS + idx]
TRACE_LAYERS = 64  # a layer's own stamp slots (``c["layer"]``): a server snapshot keeps every layer's last launch
TRACE_BLOCKS = 256  # CTAs a launch (``common.plan.BLOCKS``)


@flyc.jit
def stamp(c, idx: int):
    """Trace stamp ``idx`` of this CTA, compiled in by a build with ``trace``: the
    100 MHz realtime clock, an int64 in the launch's ``*_trace`` scratch region
    (``c["trace"]``, a pair a stamp), written by thread 0."""
    if const_expr(not c.get("trace_on", False)):
        return
    layer = c.get("layer")
    if const_expr(layer is None):
        base = c["bid"] * TRACE_SLOTS
    else:
        base = (layer % TRACE_LAYERS) * (TRACE_BLOCKS * TRACE_SLOTS) + c[
            "bid"
        ] * TRACE_SLOTS
    if c["tid"] == 0:
        now = clock()
        addr = c["trace"].value + fx.Int64((base + idx) * 8)
        ptr = llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<1>"), as_ir_value(addr)).result
        llvm.StoreOp(as_ir_value(now), ptr, alignment=8)


@flyc.jit
def poll_flags(c, regions):
    """Wait for the flags ``regions`` = [(region, n) | (region, base, n), ...] (flags
    base .. base + n of each), one flag a thread, then a barrier and an acquire: one
    round trip whatever the count (a thread polling every flag takes a round trip
    per ``POLL_MAX`` specs). Every thread of the CTA must call it (the barrier)."""
    tid = c["tid"]
    off = 0
    for spec in regions:
        region, base, n = spec if len(spec) == 3 else (spec[0], 0, spec[1])
        if (tid >= off) & (tid < off + n):
            c["poll"]([(region, base + tid - off, 1)])
        off += n
    if isinstance(off, int):
        assert off <= THREADS, off
    gpu.barrier()
    acquire_loads()


def lds_dma_16(rsrc_v, lds_ptr, voffset, policy: str = ""):
    """``buffer_load_dwordx4 ... lds``: 16 B a lane from the buffer resource
    ``rsrc_v`` at byte ``voffset`` into LDS at ``lds_ptr`` (wave-uniform) + 16 lane,
    by inline asm so the compiler's wait counting does not see it (it waits for a
    visible LDS DMA before every LDS read, which serializes a ring): the caller
    counts them (``s_waitcnt vmcnt``, loads retire in order). ``policy``: cache
    modifiers, e.g. ``"sc1"`` for data another CTA stored in this launch."""
    r = as_ir_value(rsrc_v)
    if "ptr<8>" in str(r.type):
        r = bo._ptr8_to_v4i32(r)
    # the SGPR operands as values the compiler knows are uniform (readfirstlane each
    # word): a VGPR-held 128-bit resource has no legal copy into an SGPR quad
    rv = fx.Vector(r, (4,), fx.Int32)
    rs = fx.Vector.from_elements([uniform(fx.Int32(rv[i])) for i in range(4)], fx.Int32)
    base = uniform(fx.Int32(llvm.ptrtoint(T.i32, lds_ptr.llvm_ptr)))
    llvm.inline_asm(
        None,
        [
            as_ir_value(rs),
            as_ir_value(base),
            as_ir_value(fx.Int32(voffset)),
            as_ir_value(uniform(fx.Int32(0))),
        ],
        "s_mov_b32 m0, $1\n\ts_nop 0\n\tbuffer_load_dwordx4 $2, $0, $3 offen lds"  # codespell:ignore offen
        + (" " + policy if policy else ""),
        "s,s,v,s,~{m0},~{memory}",
        has_side_effects=True,
    )


def global_atomic(addr, val, xchg=False):
    """Device-scope atomic add (or exchange) of ``val`` on the int32 at global
    address ``addr`` (Int64); the word's previous value."""
    ptr = llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<1>"), as_ir_value(addr)).result
    op = llvm.AtomicRMWOp(
        llvm.AtomicBinOp.xchg if xchg else llvm.AtomicBinOp.add, ptr, as_ir_value(fx.Int32(val)),
        llvm.AtomicOrdering.monotonic, syncscope="agent",
    )  # fmt: skip
    return fx.Int32(op.result)


def lds_atomic_add(ptr, val):
    """Atomic add of ``val`` to the LDS word at ``ptr`` (workgroup scope); the word's
    previous value."""
    op = llvm.AtomicRMWOp(
        llvm.AtomicBinOp.add,
        ptr.llvm_ptr,
        as_ir_value(fx.Int32(val)),
        llvm.AtomicOrdering.monotonic,
        syncscope="workgroup",
    )
    return fx.Int32(op.result)
