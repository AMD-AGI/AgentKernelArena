# SPDX-License-Identifier: Apache-2.0
"""FlyDSL 0.3.2 bridge for the upstream device library.

The AgentKernelArena runtime ships FlyDSL 0.3.2, which has no ``fx.generic_store`` and no ``fx.llvm``
(its ``fx.Pointer[...]`` carries no ``ir_type`` either): ``common.guard.stop_launch`` and the
``fx.llvm.atomic_add`` counter of ``gemm_coll._ag_push`` do not trace on it. These functions emit the same
operations on the llvm dialect directly (the form sgl-amd-mono's ``muse_spark/gfx950/compat.py`` bridge
installs). Retire them with a FlyDSL >= 0.3.4 runtime.

AKA: only the entry points this launch calls are kept (the global pointer, the store and the launch stop
of the upstream bridge, unchanged), plus ``global_atomic_add`` (new, below).
"""

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


def stop_launch(page=0):
    _store(_gptr(page << 12), fx.Int32(1), words=1, volatile=True)


def global_atomic_add(addr, value, syncscope="agent"):
    """Monotonic atomic add of an Int32 at the 64-bit global address ``addr``; the word's previous value.

    AKA: upstream ``fx.llvm.atomic_add(fx.inttoptr(fx.PointerType.get(i32, Global, 4), addr), ...)``
    (FlyDSL >= 0.3.4: FlyDSL 0.3.2 has no ``fx.llvm``); the same ``atomicrmw add`` on the llvm dialect,
    as sgl-amd-mono's ``muse_spark/gfx950/device.py::global_atomic`` emits it."""
    op = llvm.AtomicRMWOp(
        llvm.AtomicBinOp.add,
        _gptr(addr),
        as_ir_value(fx.Int32(value)),
        llvm.AtomicOrdering.monotonic,
        syncscope=syncscope,
    )
    return fx.Int32(op.result)
