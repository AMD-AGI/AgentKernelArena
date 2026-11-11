# SPDX-License-Identifier: Apache-2.0
# Adapted from sgl-project/sgl-amd-mono at 52486703fbf27e77f05b573e4d931d4eec87118d
# (sgl_amd_mono/muse_spark/gfx950/compat.py, Apache-2.0): the FlyDSL 0.3.2 bridge, as plain
# functions instead of attributes installed on flydsl.expr (a candidate may not mutate an
# imported module).
"""FlyDSL 0.3.2 stand-ins for the newer FlyDSL entry points the upstream kernel calls.

FlyDSL 0.3.2 (the AgentKernelArena runtime) has no ``fx.memory_fence`` / ``fx.AtomicOrdering``,
no pointer ``fx.atomic_add`` and no ``fx.generic_store``; these emit the same llvm-dialect
operations directly.
"""
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr.typing import as_ir_value

_ORDERINGS = {
    "acquire": llvm.AtomicOrdering.acquire,
    "release": llvm.AtomicOrdering.release,
    "acq_rel": llvm.AtomicOrdering.acq_rel,
    "seq_cst": llvm.AtomicOrdering.seq_cst,
}


def memory_fence(syncscope="workgroup", ordering="acquire"):
    """``fence syncscope(<scope>) <ordering>`` (``fx.memory_fence`` of FlyDSL >= 0.3.4)."""
    llvm.FenceOp(_ORDERINGS[ordering], syncscope=syncscope)


def lds_atomic_add(ptr, value, syncscope="workgroup"):
    """Monotonic ``atomicrmw add`` on an LDS ``fx`` pointer (``fx.atomic_add`` of FlyDSL >= 0.3.4);
    returns the previous value."""
    v = as_ir_value(fx.Int32(value))
    return llvm.AtomicRMWOp(llvm.AtomicBinOp.add, fx.to_llvm_ptr(ptr), v,
                            llvm.AtomicOrdering.monotonic, syncscope=syncscope, alignment=4).res


def null_page_store():
    """A volatile store of 1 to address 0: ``guard.stop_launch``'s deliberate fault."""
    zero = llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<1>"), as_ir_value(fx.Int64(0))).result
    llvm.StoreOp(as_ir_value(fx.Int32(1)), zero, alignment=4, volatile_=True)
