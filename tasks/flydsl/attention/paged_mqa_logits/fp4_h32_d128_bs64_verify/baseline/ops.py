# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 97359d6, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/device/ops.py
# (``global_load`` and ``ld_i32`` from ATOM at 454d0e3023, by way of
# sgl-project/sglang#43984)
# Vendored from sgl-amd-mono @ 52486703 (sgl_amd_mono/common/ops.py). Modified for
# AgentKernelArena (FlyDSL 0.3.2 runtime): only the primitives the K2i scorer and its walk plan
# use are kept; imports are package-relative; ``global_ptr`` / ``global_load`` are written on
# llvm-dialect ops, and ``memory_fence`` is an llvm-dialect fence, as
# sgl_amd_mono/muse_spark/gfx950/compat.py bridges them on FlyDSL 0.3.2 (which has no
# fx.generic_load, fx.memory_fence or fx.Pointer.ir_type), without patching flydsl.
"""Device primitives of the mono kernels: lane swaps and buffer / global access."""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import rocdl
from flydsl.expr.typing import T, as_ir_value

from . import buffer_ops as bo

# gfx95x cache policy bits: SC1 = device scope, SC0 | SC1 = system scope (peers over XGMI)
CM_DEV = 16

_ORDERINGS = {
    "acquire": llvm.AtomicOrdering.acquire,
    "release": llvm.AtomicOrdering.release,
    "acq_rel": llvm.AtomicOrdering.acq_rel,
    "seq_cst": llvm.AtomicOrdering.seq_cst,
}


def rsrc(addr, nbytes=None):
    return bo.create_buffer_resource_from_addr(addr, num_records_bytes=nbytes)


def permlane_swap(off, x, y):
    """v_permlane{32,16}_swap on registers (x, y); returns (x', y')."""
    swap = rocdl.permlane32_swap if off == 32 else rocdl.permlane16_swap
    pr = swap(
        llvm.StructType.get_literal([T.i32, T.i32]),
        as_ir_value(fx.Int32(x)),
        as_ir_value(fx.Int32(y)),
        False,
        False,
    )
    return tuple(fx.Int32(llvm.extractvalue(T.i32, pr, [j])) for j in range(2))


def memory_fence(syncscope="workgroup", ordering="release"):
    """``fence syncscope(<scope>) <ordering>``: fx.memory_fence of FlyDSL >= 0.3.4 (compat.py)."""
    llvm.FenceOp(_ORDERINGS[ordering], syncscope=syncscope)


def global_ptr(addr, words=1):
    """An llvm global pointer (address space 1) at the 64-bit address ``addr``."""
    return llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<1>"), as_ir_value(fx.Int64(addr))).result


def global_load(addr, words=1, nt=False):
    """``words`` i32 (1, 2 or 4) at the 64-bit global address ``addr``: a pool past
    4 GiB is reached only this way (a buffer load's offset is 32 bits). ``nt``:
    nontemporal."""
    i32 = ir.IntegerType.get_signless(32)
    ty = i32 if words == 1 else ir.VectorType.get([words], i32)
    r = llvm.LoadOp(ty, global_ptr(addr, words), alignment=4 * words, nontemporal=nt or None).result
    return fx.Int32(r) if words == 1 else fx.Vector(r)


def ld_i32(ptr, i, cm=0):
    """Word ``i`` of a buffer, an Int32 load (cache policy ``cm``)."""
    return fx.Int32(
        bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.i32, cache_modifier=cm)
    )
