# SPDX-License-Identifier: Apache-2.0
"""FlyDSL 0.3.2 bridge for the upstream device library.

The AgentKernelArena runtime ships FlyDSL 0.3.2, which has no ``fx.generic_store`` /
``fx.generic_load`` / ``fx.memory_fence`` / ``fx.AtomicOrdering`` / ``fx.llvm`` and whose
``fx.Pointer[...]`` carries no ``ir_type``: ``common.guard.stop_launch``, ``common.ops.global_ptr`` /
``global_load`` and the MoE route table's LDS ``atomic_or`` do not trace on it. These are the same
entry points on the llvm dialect directly.

Vendored from sgl-project/sgl-amd-mono at 52486703 (Apache-2.0),
``sgl_amd_mono/muse_spark/gfx950/compat.py``, as adapted by the AgentKernelArena task
``tasks/flydsl/other/muse_spark_layer/tp8``: upstream installs the functions into ``flydsl.expr`` and
the common modules at import (``compat.install()``); candidate code may not patch modules, so the
vendored ops / guard call these functions directly. ``lds_atomic_or`` (below) is new here.
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


def lds_atomic_or(ptr, val):
    """Atomic OR of ``val`` into the LDS word at ``ptr`` (workgroup scope); the word's previous value.

    AKA: the upstream route table calls ``fx.llvm.atomic_or`` (FlyDSL >= 0.3.4); this is the same
    ``atomicrmw or`` on the llvm dialect, in the form of the upstream ``muse_spark/gfx950/device.py``
    ``lds_atomic_add``."""
    op = llvm.AtomicRMWOp(
        llvm.AtomicBinOp._or,
        ptr.llvm_ptr,
        as_ir_value(fx.Int32(val)),
        llvm.AtomicOrdering.monotonic,
        syncscope="workgroup",
    )
    return fx.Int32(op.result)
