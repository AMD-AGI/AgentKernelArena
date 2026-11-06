# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.
"""Buffer resource builders required by this task's FlyDSL HGEMM kernel.

Adapted from ROCm/FlyDSL ``kernels/common/buffer_ops.py`` for FlyDSL 0.3.2.
The installed FlyDSL wheel omits that source-tree module.  This task uses only
the operations below; all GEMM arithmetic and launches remain in ``kernel.py``.
"""

from flydsl._mlir import ir
from flydsl._mlir.dialects import arith, fly, llvm, rocdl
from flydsl._mlir.extras import types as T
from flydsl.expr.typing import as_ir_value
from flydsl.runtime.device import is_rdna_arch


def _constant(type_, value):
    return arith.ConstantOp(type_, ir.IntegerAttr.get(type_, value)).result


def _i32(value):
    if value > 0x7FFFFFFF:
        value -= 1 << 32
    return _constant(T.i32(), value)


def _i64(value):
    return _constant(T.i64(), value)


def _to_i32(value):
    if isinstance(value, int):
        return _i32(value)
    value = as_ir_value(value)
    if isinstance(value.type, ir.IntegerType):
        if value.type.width == 32:
            return value
        if value.type.width > 32:
            return arith.TruncIOp(T.i32(), value).result
        return arith.ExtSIOp(T.i32(), value).result
    return arith.IndexCastOp(T.i32(), value).result


def _buffer_flags():
    flags = (7 << 12) | (4 << 15)
    if is_rdna_arch(None):
        flags |= (1 << 24) | (2 << 28)
    return flags


def _resource(base_ptr):
    return rocdl.MakeBufferRsrcOp(
        ir.Type.parse("!llvm.ptr<8>"),
        base_ptr,
        _constant(T.i16(), 0),
        _i64(0xFFFFFFFF),
        _i32(_buffer_flags()),
    ).result


def create_buffer_resource_from_addr(addr_i64):
    base_ptr = llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr"), as_ir_value(addr_i64)).result
    return _resource(base_ptr)


def create_buffer_resource(memref, *, max_size=True):
    if not max_size:
        raise ValueError("HGEMM requires a maximum-size buffer resource")
    base_ptr = fly.extract_aligned_pointer_as_index(
        ir.Type.parse("!llvm.ptr"), as_ir_value(memref)
    )
    return _resource(base_ptr)


def buffer_load(rsrc, offset, *, vec_width, dtype):
    dtype = dtype.ir_type if hasattr(dtype, "ir_type") else dtype
    byte_offset = arith.MulIOp(_to_i32(offset), _i32(dtype.width // 8)).result
    result_type = dtype if vec_width == 1 else ir.VectorType.get([vec_width], dtype)
    return rocdl.RawPtrBufferLoadOp(
        result_type, as_ir_value(rsrc), byte_offset, _i32(0)
    ).result


def buffer_store(data, rsrc, offset, *, cache_modifier=0):
    data = as_ir_value(data)
    element_type = data.type.element_type if isinstance(data.type, ir.VectorType) else data.type
    byte_offset = arith.MulIOp(_to_i32(offset), _i32(element_type.width // 8)).result
    aux = ir.IntegerAttr.get(T.i32(), cache_modifier) if cache_modifier else None
    rocdl.RawPtrBufferStoreOp(data, as_ir_value(rsrc), byte_offset, _i32(0), aux=aux)


def create_llvm_ptr(value, *, address_space=0):
    value = as_ir_value(value)
    if isinstance(value.type, ir.IndexType):
        value = arith.IndexCastOp(T.i64(), value).result
    return llvm.IntToPtrOp(ir.Type.parse(f"!llvm.ptr<{address_space}>"), value).result


def get_element_ptr(base_ptr, byte_offset=None, *, static_byte_offset=0):
    base_ptr = as_ir_value(base_ptr)
    if byte_offset is None:
        dynamic_indices = []
        static_indices = [int(static_byte_offset)]
    elif isinstance(byte_offset, int):
        dynamic_indices = []
        static_indices = [int(byte_offset) + int(static_byte_offset)]
    else:
        byte_offset = as_ir_value(byte_offset)
        if isinstance(byte_offset.type, ir.IndexType):
            byte_offset = arith.IndexCastOp(T.i64(), byte_offset).result
        if static_byte_offset:
            byte_offset = arith.AddIOp(
                byte_offset, _constant(byte_offset.type, int(static_byte_offset))
            ).result
        dynamic_indices = [byte_offset]
        static_indices = [-(1 << 31)]
    return llvm.GEPOp(
        base_ptr.type, base_ptr, dynamic_indices, static_indices, T.i8(), None
    ).result
