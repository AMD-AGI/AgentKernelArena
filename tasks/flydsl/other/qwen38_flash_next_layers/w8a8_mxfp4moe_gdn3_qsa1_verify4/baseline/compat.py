# SPDX-License-Identifier: Apache-2.0
"""FlyDSL 0.3.2 forms of the FlyDSL 0.4 entry points the kernel uses.

FlyDSL 0.3.2 has no ``fx.generic_store`` / ``fx.generic_load`` / ``fx.atomic_add``,
and its ``fx.Pointer[...]`` carries no ``ir_type``; these build the same operations
on the llvm dialect directly (sgl-amd-mono's muse_spark FlyDSL 0.3.2 bridge, as
plain functions instead of patches of ``flydsl.expr``)."""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr.typing import as_ir_value

_PTR1 = "!llvm.ptr<1>"


def gptr(addr):
    """An llvm global pointer (address space 1) at the 64-bit address ``addr``."""
    return llvm.IntToPtrOp(ir.Type.parse(_PTR1), as_ir_value(fx.Int64(addr))).result


def store(ptr, value, words=None, volatile=False):
    v = as_ir_value(value)
    if words is None:
        words = 1 if isinstance(value, fx.Int32) else len(value)
    llvm.StoreOp(v, ptr, alignment=4 * words, volatile_=volatile)


def load(ptr, count=1, nontemporal=False):
    i32 = ir.IntegerType.get_signless(32)
    ty = i32 if count == 1 else ir.VectorType.get([count], i32)
    r = llvm.LoadOp(
        ty, ptr, alignment=4 * count, nontemporal=nontemporal or None
    ).result
    return fx.Int32(r) if count == 1 else fx.Vector(r)


def lds_atomic_add(ptr, val, syncscope="workgroup"):
    """Atomic add of ``val`` to the LDS word at ``ptr``; the word's previous value."""
    op = llvm.AtomicRMWOp(
        llvm.AtomicBinOp.add,
        ptr.llvm_ptr,
        as_ir_value(fx.Int32(val)),
        llvm.AtomicOrdering.monotonic,
        syncscope=syncscope,
    )
    return fx.Int32(op.result)
