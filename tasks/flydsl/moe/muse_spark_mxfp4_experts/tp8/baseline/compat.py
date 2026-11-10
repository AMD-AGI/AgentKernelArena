# SPDX-License-Identifier: Apache-2.0
"""FlyDSL 0.3.2 bridge for the upstream device library.

The Muse Spark 1020 image ships FlyDSL 0.3.2 (aiter 4ad998328), which has no
``fx.generic_store`` / ``fx.generic_load`` / ``fx.memory_fence`` / ``fx.AtomicOrdering``
and whose ``fx.Pointer[...]`` carries no ``ir_type``: ``common.guard.stop_launch``,
``common.epoch.gstore`` and ``common.ops.global_load`` do not trace on it. Importing
this module installs those entry points on the llvm dialect directly (the form the
plugin's fork used on 0.3.2) where they are missing, and leaves a FlyDSL that has
them alone. Retire it with the image's FlyDSL >= 0.3.4.
"""

import enum

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr.typing import as_ir_value

_PTR1 = "!llvm.ptr<1>"


def _gptr(addr):
    """An llvm global pointer (address space 1) at the 64-bit address ``addr``."""
    return llvm.IntToPtrOp(ir.Type.parse(_PTR1), as_ir_value(fx.Int64(addr))).result


def _store(ptr, value, words=None, volatile=False):
    v = as_ir_value(value)
    if words is None:
        words = 1 if isinstance(value, fx.Int32) else len(value)
    llvm.StoreOp(v, ptr, alignment=4 * words, volatile_=volatile)


def _load(ptr, count=1, nontemporal=False):
    i32 = ir.IntegerType.get_signless(32)
    ty = i32 if count == 1 else ir.VectorType.get([count], i32)
    r = llvm.LoadOp(
        ty, ptr, alignment=4 * count, nontemporal=nontemporal or None
    ).result
    return fx.Int32(r) if count == 1 else fx.Vector(r)


class AtomicOrdering(enum.Enum):
    Acquire = "acquire"
    Release = "release"
    AcquireRelease = "acq_rel"
    SequentiallyConsistent = "seq_cst"


_ORDERINGS = {
    "acquire": llvm.AtomicOrdering.acquire,
    "release": llvm.AtomicOrdering.release,
    "acq_rel": llvm.AtomicOrdering.acq_rel,
    "seq_cst": llvm.AtomicOrdering.seq_cst,
}


def _scope_name(scope) -> str:
    name = getattr(scope, "name", None) or getattr(scope, "value", None) or str(scope)
    return str(name).lower()


def memory_fence(syncscope="workgroup", ordering="release"):
    """``llvm.fence`` at ``syncscope`` (workgroup / agent / system) with ``ordering``."""
    order = (
        ordering
        if isinstance(ordering, str)
        else getattr(ordering, "value", str(ordering))
    )
    llvm.FenceOp(_ORDERINGS[str(order).lower()], syncscope=_scope_name(syncscope))


def global_ptr(addr, words=1):
    return _gptr(addr)


def global_load(addr, words=1, nt=False):
    return _load(_gptr(addr), count=words, nontemporal=nt)


def gstore(addr, value, words=4):
    _store(_gptr(addr), fx.Int32(value) if words == 1 else value, words=words)


def stop_launch(page=0):
    _store(_gptr(page << 12), fx.Int32(1), words=1, volatile=True)
