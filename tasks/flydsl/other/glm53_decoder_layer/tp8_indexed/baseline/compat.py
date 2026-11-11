# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# global_ptr_i32 / global_load_words / volatile_store_i32 adapted from sgl-amd-mono 52486703
# sgl_amd_mono/muse_spark/gfx950/compat.py (Apache-2.0, the FlyDSL 0.3.2 bridge); lds_atomic_add adapted from
# sgl_amd_mono/muse_spark/gfx950/device.py (MIT, Copyright (C) 2026, Advanced Micro Devices, Inc.).
"""FlyDSL 0.3.2 forms of the FlyDSL >= 0.3.4 entry points the vendored kernel uses.

The AgentKernelArena runtime ships FlyDSL 0.3.2, which has no ``fx.atomic_add``, ``fx.generic_load`` or
``fx.generic_store`` and whose ``fx.Pointer[...]`` carries no ``ir_type``. These helpers emit the same LLVM
operations directly (sgl-amd-mono's own 0.3.2 bridge does this by patching FlyDSL; the candidate rules forbid
module patching, so the vendored files call these functions instead).
"""
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr.typing import as_ir_value


def global_ptr_i32(addr):
    """An llvm global pointer (address space 1) at the 64-bit address ``addr``."""
    return llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<1>"), as_ir_value(fx.Int64(addr))).result


def global_load_words(addr, words=1, nt=False):
    """``words`` i32 (1, 2 or 4) at global address ``addr``; ``nt``: nontemporal."""
    i32 = ir.IntegerType.get_signless(32)
    ty = i32 if words == 1 else ir.VectorType.get([words], i32)
    r = llvm.LoadOp(ty, global_ptr_i32(addr), alignment=4 * words, nontemporal=nt or None).result
    return fx.Int32(r) if words == 1 else fx.Vector(r)


def volatile_store_i32(addr, value):
    """A volatile i32 store at global address ``addr`` (the guard's stop: a store to the null page)."""
    llvm.StoreOp(as_ir_value(fx.Int32(value)), global_ptr_i32(addr), alignment=4, volatile_=True)


def lds_atomic_add(ptr, val, syncscope="workgroup"):
    """Atomic add of ``val`` to the LDS word at ``ptr`` (a FlyDSL LDS iterator); the word's previous value."""
    op = llvm.AtomicRMWOp(
        llvm.AtomicBinOp.add,
        ptr.llvm_ptr,
        as_ir_value(fx.Int32(val)),
        llvm.AtomicOrdering.monotonic,
        syncscope=syncscope,
    )
    return fx.Int32(op.result)
