# SPDX-License-Identifier: Apache-2.0
"""FlyDSL 0.3.2 bridge for the upstream device library.

The AgentKernelArena runtime ships FlyDSL 0.3.2, which has no ``fx.generic_store`` / ``fx.memory_fence`` /
``fx.AtomicOrdering`` / ``fx.llvm.atomic_add`` and whose ``fx.Pointer[...]`` carries no ``ir_type``, so the
upstream ``guard.stop_launch``, the mailbox fences and the selection's LDS slot counter do not trace on it.
These are the same entry points on the llvm dialect directly.

Adapted from sgl-project/sgl-amd-mono at 52486703 (Apache-2.0): sgl_amd_mono/muse_spark/gfx950/compat.py
(the store, the fence and the ordering enum) and the ``lds_atomic_add`` of
sgl_amd_mono/muse_spark/gfx950/device.py. Upstream installs them into ``flydsl.expr`` at import; candidate code
may not patch modules, so the vendored kernels call these functions instead.
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
    """``llvm.fence`` at ``syncscope`` (workgroup / agent; FlyDSL's ``rocdl.SyncScope`` strings) with
    ``ordering``."""
    order = ordering if isinstance(ordering, str) else getattr(ordering, "value", str(ordering))
    llvm.FenceOp(_ORDERINGS[str(order).lower()], syncscope=_scope_name(syncscope))


def stop_launch(page=0):
    """``guard.stop_launch``: a volatile store to the null page stops a failed launch."""
    _store(_gptr(page << 12), fx.Int32(1), words=1, volatile=True)


def lds_atomic_add(ptr, val):
    """Atomic add of ``val`` to the LDS word at ``ptr`` (workgroup scope); the word's previous value."""
    op = llvm.AtomicRMWOp(
        llvm.AtomicBinOp.add,
        ptr.llvm_ptr,
        as_ir_value(fx.Int32(val)),
        llvm.AtomicOrdering.monotonic,
        syncscope="workgroup",
    )
    return fx.Int32(op.result)
