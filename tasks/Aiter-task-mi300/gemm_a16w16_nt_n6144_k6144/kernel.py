# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""BF16 GEMM out = a @ b.T in FlyDSL: a split-K HGEMM kernel with MFMA, LDS staging and a per-M launch config.

`build_gemm_a16w16_nt_n6144_k6144_module(m, n, k)` compiles the kernel for one shape and returns `launch(a, b) -> out`.
"""
from flydsl._mlir import ir
from flydsl._mlir.dialects import vector as _vector
from flydsl._mlir.dialects.vector import *
from flydsl.expr.meta import dsl_loc_tracing
from flydsl.expr.typing import (  # noqa: F401
    ReductionOp,
    Vector,
    as_ir_value,
    empty_like,
    full,
    full_like,
    ones_like,
    zeros_like,
)
from flydsl._mlir.dialects import arith as std_arith
from flydsl._mlir.dialects import llvm, rocdl as _buffer_ops_rocdl
from flydsl._mlir.extras import types as _buffer_ops_T
from flydsl.runtime.device import is_rdna_arch
from abc import ABC, abstractmethod
from itertools import product
import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import fly, llvm
from flydsl.compiler.protocol import extract_to_ir_values
from flydsl.expr import ptrtoint, range_constexpr
from flydsl.expr.typing import T
import functools
from flydsl._mlir.dialects import llvm, scf
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.runtime.device import get_rocm_arch
import types

def _suffix_products(shape):
    out, acc = [], 1
    for dim in reversed(list(shape)):
        acc *= int(dim)
        out.append(acc)
    return out[::-1]


# ---- kernels/vector.py -----------------------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""Vector dialect wrappers, vendored into aiter.

flydsl deleted ``flydsl.expr.vector``; callers are now expected to use the raw
``flydsl._mlir.dialects.vector`` and wrap every operand in ``as_ir_value``.
aiter has 500+ vector call sites, so the auto-unwrapping layer is kept here
instead — a missing ``as_ir_value`` or ``kDynamic`` sentinel only surfaces at
trace time.

Re-exports the whole raw dialect (``broadcast``, ``shuffle``, ``insert``,
``extract_strided_slice``, ``reduction``, ...) and rebuilds the five wrappers
that unwrap DSL values: ``from_elements``, ``store``, ``extract``,
``load_op``, ``bitcast``. ``extract`` also supplies the ``kDynamic`` sentinel
for dynamic positions. Only published flydsl APIs are used.

Upstream: FlyDSL ``python/flydsl/expr/vector.py``, deleted before
ROCm/FlyDSL#880; the wrappers use FlyDSL's canonical ``as_ir_value``
converter.
"""



# Re-export the raw dialect so vector.broadcast / vector.shuffle work directly

# Vector and related types, which flydsl.expr.vector also re-exported

# ═══════════════════════════════════════════════════════════════════════
# Dialect helper wrappers (legacy, will be deprecated)
# Prefer using Vector methods or _mlir.dialects.vector directly.
# ═══════════════════════════════════════════════════════════════════════


def _as_index_ir_value(value):
    if isinstance(value, int):
        from flydsl.expr import arith as _arith_ext

        return _arith_ext.constant(value, index=True)
    v = as_ir_value(value)
    # vector/memref indices must be index-typed; cast integer offsets
    # (e.g. fx.Int64/i32) so callers can pass explicit-width offsets.
    if isinstance(v.type, ir.IntegerType):
        from flydsl._mlir.dialects import arith as _std_arith

        v = _std_arith.IndexCastOp(ir.IndexType.get(), v).result
    return v


@dsl_loc_tracing
def from_elements(*args, **kwargs):
    """Construct a vector from scalar elements, auto-unwrapping ArithValue wrappers."""
    if len(args) >= 2:
        args = list(args)
        elems = args[1]
        if isinstance(elems, (list, tuple)):
            args[1] = [as_ir_value(v) for v in elems]
        return _vector.from_elements(*args, **kwargs)

    return _vector.from_elements(*args, **kwargs)


@dsl_loc_tracing
def store(value, memref, indices, **kwargs):
    """Vector store wrapper that accepts ArithValue/wrappers for value/indices."""
    return _vector.store(
        as_ir_value(value),
        as_ir_value(memref),
        [_as_index_ir_value(i) for i in indices],
        **kwargs,
    )


# -----------------------------------------------------------------------------
# Thin wrappers for common op classes that otherwise require `.result` access.
# -----------------------------------------------------------------------------


@dsl_loc_tracing
def extract(vector, static_position=None, dynamic_position=None):
    """Wrapper around `vector.ExtractOp(...).result`.

    When only ``dynamic_position`` is supplied (without explicit
    ``static_position``), each dynamic index needs a corresponding
    ``kDynamic`` sentinel in the static attribute so the ODS builder
    pairs them correctly.  This wrapper fills in the sentinels
    automatically.
    """
    if static_position is None:
        static_position = []
    if dynamic_position is None:
        dynamic_position = []
    dynamic_position = [_as_index_ir_value(i) for i in dynamic_position]

    n_static = len(static_position)
    n_dynamic = len(dynamic_position)
    if n_dynamic > 0 and n_static < n_dynamic:
        kDynamic = ir.ShapedType.get_dynamic_size()
        static_position = list(static_position) + [kDynamic] * (n_dynamic - n_static)

    return _vector.ExtractOp(
        as_ir_value(vector),
        static_position=static_position,
        dynamic_position=dynamic_position,
    ).result


@dsl_loc_tracing
def load_op(result_type, memref, indices):
    """Wrapper around `vector.LoadOp(...).result`."""
    return _vector.LoadOp(
        result_type,
        as_ir_value(memref),
        [_as_index_ir_value(i) for i in indices],
    ).result


@dsl_loc_tracing
def bitcast(result_type, source):
    """Wrapper around `vector.BitCastOp(...).result`."""
    return _vector.BitCastOp(
        result_type,
        as_ir_value(source),
    ).result
vector = types.SimpleNamespace(**{_n: globals()[_n] for _n in ['_as_index_ir_value', 'from_elements', 'store', 'extract', 'load_op', 'bitcast']})


# ---- kernels/buffer_ops.py -------------------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""AMD buffer load/store operations, vendored into aiter.

flydsl moved these from ``flydsl.expr.buffer_ops`` to its repo-level
``kernels/common/``, which its wheel does not ship, so aiter keeps an
equivalent copy here. Only published flydsl APIs are used.

Buffer instructions are an AMD hardware feature (buffer resource descriptor
plus ROCDL intrinsics) providing out-of-bounds protection and better memory
throughput; plain memref load/store is not a substitute.

Upstream: FlyDSL ``kernels/common/buffer_ops.py`` @ ROCm/FlyDSL#880.
Behavior matches upstream; formatting and type annotations differ only to
satisfy aiter's lint.

Example:
    >>> from aiter.ops.flydsl.kernels import buffer_ops
    >>> from flydsl.expr import arith
    >>>
    >>> rsrc = buffer_ops.create_buffer_resource(A)
    >>> offset = row * arith.index(4096) + col
    >>> data = buffer_ops.buffer_load(rsrc, offset, vec_width=4)
    >>> buffer_ops.buffer_store(data, rsrc, offset)
"""




def _get_buffer_flags(arch=None):
    """Get AMD buffer resource descriptor (V#) flags word (bits 127:96).

    Constructs the 32-bit flags field for rocdl.make.buffer.rsrc, following the
    same logic as LLVM's AMDGPUToROCDL makeBufferRsrc():
      https://github.com/llvm/llvm-project/blob/main/mlir/lib/Conversion/AMDGPUToROCDL/AMDGPUToROCDL.cpp

    Bit layout (common to all architectures):
      bits [11:0]  - DST_SEL: ignored by raw buffer intrinsics
      bits [14:12] - DATA_FORMAT: must be nonzero, 7 = float
      bits [18:15] - NUM_FORMAT:  must be nonzero, 4 = 32-bit
      bit  [19]    - In nested heap (0)
      bit  [20]    - Behavior on unmap (0 = return 0 / ignore)
      bits [22:21] - Index stride for swizzles (0)
      bit  [23]    - Add thread ID (0)
      bit  [24]    - Reserved: must be 1 on RDNA, 0 on CDNA
      bits [26:25] - Reserved (0)
      bit  [27]    - Non-volatile (CDNA only, 0)
      bits [29:28] - OOB_SELECT (RDNA only): 0=structured, 2=none, 3=check offset
      bits [31:30] - Type (must be 0)

    CDNA (gfx9xx):    (7 << 12) | (4 << 15)                         = 0x20070
    RDNA (gfx10+):    (7 << 12) | (4 << 15) | (1 << 24) | (2 << 28) = 0x21020070
      - bit 24 set to 1 (required on RDNA)
      - OOB_SELECT=2 (no bounds checking, matching LLVM boundsCheck=false)
    """

    if arch is None:
        arch = None
    flags = (7 << 12) | (4 << 15)
    if is_rdna_arch(arch):
        flags |= 1 << 24  # reserved bit, must be 1 on RDNA
        flags |= 2 << 28  # OOB_SELECT = 2 (no bounds checking)
    return flags


__all__ = [
    "BufferResourceDescriptor",
    "buffer_load",
    "buffer_store",
    "create_buffer_resource",
    "create_buffer_resource_from_addr",
    "create_llvm_ptr",
    "extract_base_index",
    "get_element_ptr",
]


def _unwrap_value(value):
    """Recursively unwrap ArithValue or similar wrappers to get the actual MLIR value.

    Handles:
    - FlyDSL ArithValue (has ._value)
    - flyc DSL Numeric like fx.Int32 (has .ir_value() method)
    - flyc ArithValue (is already ir.Value subclass)
    """
    # DSL Numeric (Int32, Float32, etc.) — use ir_value() to materialize
    if hasattr(value, "ir_value") and not isinstance(value, ir.Value):
        return value.ir_value()
    max_depth = 10  # Safety limit
    depth = 0
    while depth < max_depth and not isinstance(value, ir.Value):
        if hasattr(value, "_value"):
            value = value._value
        elif hasattr(value, "value"):
            value = value.value
        else:
            break
        depth += 1
    return value


@dsl_loc_tracing
def _create_i32_constant(value: 'int') -> 'ir.Value':
    """Create i32 constant using standard MLIR arith dialect."""
    i32_type = _buffer_ops_T.i32()
    if value > 0x7FFFFFFF:
        value = int(value - 2**32)
    attr = ir.IntegerAttr.get(i32_type, value)
    op = std_arith.ConstantOp(i32_type, attr)
    return _unwrap_value(op.result)


def _to_i32_offset(offset: 'ir.Value') -> 'ir.Value':
    """Normalize an already-unwrapped offset value to i32.

    Accepts index (index_cast), i32 (as-is), wider ints e.g. i64 (trunc), and
    narrower ints (sign-extend). Lets callers pass fx.Int64 offsets (element or
    byte) without hitting the ``index_cast i64<->i32`` incompatibility.
    """
    ot = offset.type
    if isinstance(ot, ir.IntegerType):
        if ot.width == 32:
            return offset
        if ot.width > 32:
            return _unwrap_value(std_arith.TruncIOp(_buffer_ops_T.i32(), offset).result)
        return _unwrap_value(std_arith.ExtSIOp(_buffer_ops_T.i32(), offset).result)
    return _unwrap_value(std_arith.IndexCastOp(_buffer_ops_T.i32(), offset).result)


@dsl_loc_tracing
def _create_i16_constant(value: 'int') -> 'ir.Value':
    """Create i16 constant using standard MLIR arith dialect."""
    i16_type = _buffer_ops_T.i16()
    attr = ir.IntegerAttr.get(i16_type, value)
    op = std_arith.ConstantOp(i16_type, attr)
    return _unwrap_value(op.result)


@dsl_loc_tracing
def _create_i64_constant(value: 'int') -> 'ir.Value':
    """Create i64 constant using standard MLIR arith dialect."""
    i64_type = _buffer_ops_T.i64()
    attr = ir.IntegerAttr.get(i64_type, value)
    op = std_arith.ConstantOp(i64_type, attr)
    return _unwrap_value(op.result)


@dsl_loc_tracing
def _ptr8_to_v4i32(ptr8_val) -> 'ir.Value':
    """Reinterpret a buffer resource (!llvm.ptr<8>) as a <4 x i32> vector.

    Required by the scalar ``s.buffer.load`` intrinsic, whose resource operand is
    a v4i32 rather than the opaque buffer pointer used by the vector path.
    """
    i128_ty = ir.IntegerType.get_signless(128)
    v4i32_ty = ir.VectorType.get([4], ir.IntegerType.get_signless(32))
    i128_val = llvm.ptrtoint(i128_ty, _unwrap_value(ptr8_val))
    return llvm.bitcast(v4i32_ty, i128_val)


@dsl_loc_tracing
def create_llvm_ptr(value, address_space: 'int' = 0) -> 'ir.Value':
    """Create an LLVM pointer from an integer or index value."""
    value = _unwrap_value(value)
    if isinstance(value.type, ir.IndexType):
        i64_type = _buffer_ops_T.i64()
        value = _unwrap_value(std_arith.IndexCastOp(i64_type, value).result)
    ptr_type = ir.Type.parse(f"!llvm.ptr<{address_space}>")
    return llvm.IntToPtrOp(ptr_type, value).result


@dsl_loc_tracing
def extract_base_index(tensor, address_space: 'int' = 1) -> 'ir.Value':
    """Extract the base address of a fly.memref as an index value.

    Inverse of :func:`create_llvm_ptr` (index -> ptr). Useful when ISA
    requires a raw pointer instead of a buffer resource descriptor
    (e.g. global_atomic_pk_add_bf16 on gfx942).
    """
    from flydsl._mlir.dialects import fly as _fly
    from flydsl._mlir.dialects import memref as _memref

    raw = _unwrap_value(tensor)
    try:
        ir.MemRefType(raw.type)
        return _memref.extract_aligned_pointer_as_index(raw)
    except ValueError:
        pass

    ptr_type = ir.Type.parse(f"!llvm.ptr<{address_space}>")
    ptr = _fly.extract_aligned_pointer_as_index(ptr_type, raw)
    i64_val = llvm.PtrToIntOp(ir.IntegerType.get_signless(64), ptr).result
    return _unwrap_value(std_arith.IndexCastOp(ir.IndexType.get(), i64_val).result)


@dsl_loc_tracing
def get_element_ptr(
    base_ptr,
    byte_offset: 'int | ir.Value | None' = None,
    static_byte_offset: 'int' = 0,
    elem_type: 'ir.Type | None' = None,
    no_wrap_flags=None,
) -> 'ir.Value':
    """Build an LLVM GEP from a base pointer plus byte offsets."""
    _gep_dynamic_index_sentinel = -(2**31)

    base_ptr = _unwrap_value(base_ptr)
    if not isinstance(static_byte_offset, int):
        raise TypeError(
            f"static_byte_offset must be int, got {type(static_byte_offset).__name__}"
        )
    if elem_type is None:
        elem_type = _buffer_ops_T.i8()
    elif callable(elem_type):
        elem_type = elem_type()

    if byte_offset is None:
        dynamic_indices = []
        raw_constant_indices = [int(static_byte_offset)]
    elif isinstance(byte_offset, int):
        dynamic_indices = []
        raw_constant_indices = [int(byte_offset) + int(static_byte_offset)]
    else:
        offset_val = _unwrap_value(byte_offset)
        if isinstance(offset_val.type, ir.IndexType):
            i64_type = _buffer_ops_T.i64()
            offset_val = _unwrap_value(
                std_arith.IndexCastOp(i64_type, offset_val).result
            )
        elif not isinstance(offset_val.type, ir.IntegerType):
            raise TypeError(
                "byte_offset must be int, index, or integer-typed MLIR value; "
                f"got {offset_val.type}"
            )

        if static_byte_offset != 0:
            static_type = offset_val.type
            static_attr = ir.IntegerAttr.get(static_type, int(static_byte_offset))
            static_const = _unwrap_value(
                std_arith.ConstantOp(static_type, static_attr).result
            )
            offset_val = _unwrap_value(
                std_arith.AddIOp(offset_val, static_const).result
            )

        dynamic_indices = [offset_val]
        raw_constant_indices = [_gep_dynamic_index_sentinel]

    return llvm.GEPOp(
        base_ptr.type,
        base_ptr,
        dynamic_indices,
        raw_constant_indices,
        elem_type,
        no_wrap_flags,
    ).result


class BufferResourceDescriptor:
    """AMD Buffer Resource Descriptor

    A buffer resource descriptor contains:
    - base_pointer: Scalar base pointer (wave-uniform, stored in SGPRs)
    - stride: Stride for structured buffers (typically 0 for contiguous)
    - num_records: Buffer size in bytes
    - flags: Data format and access flags

    The descriptor is stored in a special LLVM pointer type (!llvm.ptr<8>)
    """

    def __init__(self, rsrc: 'ir.Value'):
        """Initialize with ROCDL resource descriptor value."""
        self.rsrc = rsrc

    @staticmethod
    @dsl_loc_tracing
    def from_memref(
        memref_val: 'ir.Value',
        stride: 'int' = 0,
        max_size: 'bool' = True,
        data_format: 'str' = "f32",
        num_records_bytes: 'int | ir.Value | None' = None,
        base_byte_offset: 'int | ir.Value | None' = None,
    ) -> 'BufferResourceDescriptor':
        """Create buffer resource descriptor from memref.

        Args:
            memref_val: Memref value to create descriptor for
            stride: Stride in elements (0 for contiguous)
            max_size: If True, use max buffer size for flexibility
            num_records_bytes: Override buffer size (in BYTES) used by hardware OOB checking.
                              If provided, this takes precedence over `max_size`.
            base_byte_offset: Optional byte offset added to the descriptor base pointer.
            data_format: Data format ('f32', 'f16', 'i32', etc.)

        Returns:
            BufferResourceDescriptor instance

        Example:
            >>> rsrc = BufferResourceDescriptor.from_memref(A)
        """
        # Extract raw pointer from fly.memref.
        raw_val = _unwrap_value(memref_val)
        from flydsl._mlir.dialects import fly as _fly

        ptr_type = ir.Type.parse("!llvm.ptr")
        base_ptr = _fly.extract_aligned_pointer_as_index(ptr_type, raw_val)
        if base_byte_offset is not None:
            base_ptr = get_element_ptr(base_ptr, byte_offset=base_byte_offset)

        # Create buffer resource descriptor
        flags_val = _get_buffer_flags()
        flags = _create_i32_constant(flags_val)
        stride_val = _create_i16_constant(stride)

        def _num_records_from_memref_type() -> 'int | None':
            """Best-effort: derive logical buffer size (in bytes) from static memref type."""
            try:
                mt = ir.MemRefType(_unwrap_value(memref_val).type)
                shape = list(mt.shape)
                if any(int(d) < 0 for d in shape):
                    return None
                # Compute element size in bytes (scalar element type).
                elem_t = mt.element_type
                elem_bits = getattr(elem_t, "width", None)
                if elem_bits is None:
                    return None
                elem_bytes = int(elem_bits) // 8
                if elem_bytes <= 0:
                    return None
                num_elems = 1
                for d in shape:
                    num_elems *= int(d)
                return int(num_elems) * int(elem_bytes)
            # best-effort size probe: any failure just means "unknown"
            except Exception:  # noqa: BLE001
                return None

        if num_records_bytes is not None:
            # Caller-provided size in BYTES (preferred for exact hardware OOB behavior).
            if isinstance(num_records_bytes, int):
                nbytes = int(num_records_bytes)
                nbytes = max(0, nbytes)
                # Descriptor uses i32 bytes; clamp to the max representable.
                nbytes = min(nbytes, 0xFFFFFFFF)
                num_records = _create_i64_constant(nbytes)
            else:
                v = _unwrap_value(num_records_bytes)
                i64_type = _buffer_ops_T.i64()
                if not isinstance(v.type, ir.IntegerType) or v.type.width != 64:
                    if isinstance(v.type, ir.IndexType):
                        op = std_arith.IndexCastOp(i64_type, v)
                    else:
                        op = std_arith.ExtSIOp(i64_type, v)
                    v = _unwrap_value(op.result)
                num_records = v
        elif max_size:
            # Use max for flexibility (hardware will check actual bounds)
            # Note: FlyDSL's rocdl.make.buffer.rsrc requires i32, not i64
            num_records = _create_i64_constant(0xFFFFFFFF)  # FALLBACK_MAX_SIZE
        else:
            # Use the logical memref size (in bytes) for hardware OOB checking.
            nbytes = _num_records_from_memref_type()
            if nbytes is None:
                # Fall back to max-size if we can't infer statically.
                num_records = _create_i64_constant(0xFFFFFFFF)
            else:
                nbytes = min(nbytes, 0xFFFFFFFF)
                num_records = _create_i64_constant(int(nbytes))

        # Create resource descriptor (returns !llvm.ptr<8>)
        rsrc_type = ir.Type.parse("!llvm.ptr<8>")
        rsrc = _buffer_ops_rocdl.MakeBufferRsrcOp(
            rsrc_type, base_ptr, stride_val, num_records, flags
        ).result

        return BufferResourceDescriptor(rsrc)


@dsl_loc_tracing
def create_buffer_resource_from_addr(
    addr_i64: 'ir.Value',
    *,
    num_records_bytes: 'int | ir.Value | None' = None,
) -> 'ir.Value':
    """Create AMD buffer resource descriptor from a raw i64 device address.

    Useful when working with runtime pointer arrays (e.g. IPC-mapped addresses
    or device-side pointer tables) where no fly.memref is available.
    The full address is encoded as the buffer base; callers should pass
    byte offset 0 to buffer_load / buffer_store.

    Args:
        addr_i64: Raw 64-bit device address (i64 MLIR value).
        num_records_bytes: Optional buffer size in bytes for hardware OOB checking.

    Returns:
        ROCDL buffer resource descriptor (!llvm.ptr<8>).

    Example:
        >>> rsrc = create_buffer_resource_from_addr(raw_addr_i64)
        >>> data = buffer_load(rsrc, i32_zero, vec_width=4, dtype=T.i32)
    """
    addr_i64 = _unwrap_value(addr_i64)
    ptr_type = ir.Type.parse("!llvm.ptr")
    base_ptr = llvm.IntToPtrOp(ptr_type, addr_i64).result
    flags = _create_i32_constant(_get_buffer_flags())
    stride = _create_i16_constant(0)
    if num_records_bytes is None:
        num_records = _create_i64_constant(0xFFFFFFFF)
    elif isinstance(num_records_bytes, int):
        nbytes = max(0, min(int(num_records_bytes), 0xFFFFFFFF))
        num_records = _create_i64_constant(nbytes)
    else:
        num_records = _unwrap_value(num_records_bytes)
        i64_type = _buffer_ops_T.i64()
        if (
            not isinstance(num_records.type, ir.IntegerType)
            or num_records.type.width != 64
        ):
            if isinstance(num_records.type, ir.IndexType):
                num_records = _unwrap_value(
                    std_arith.IndexCastOp(i64_type, num_records).result
                )
            else:
                num_records = _unwrap_value(
                    std_arith.ExtSIOp(i64_type, num_records).result
                )
    rsrc_type = ir.Type.parse("!llvm.ptr<8>")
    return _buffer_ops_rocdl.MakeBufferRsrcOp(
        rsrc_type, base_ptr, stride, num_records, flags
    ).result


@dsl_loc_tracing
def create_buffer_resource(
    memref_val: 'ir.Value',
    stride: 'int' = 0,
    max_size: 'bool' = True,
    *,
    num_records_bytes: 'int | ir.Value | None' = None,
    base_byte_offset: 'int | ir.Value | None' = None,
) -> 'ir.Value':
    """Create AMD buffer resource descriptor from memref.

    This is a simplified wrapper around BufferResourceDescriptor.from_memref()
    that returns the raw ROCDL resource value.

    Args:
        memref_val: Memref value
        stride: Buffer stride (0 for contiguous)
        max_size: Use maximum buffer size
        num_records_bytes: Override buffer size in bytes.
        base_byte_offset: Optional byte offset added to the descriptor base pointer.

    Returns:
        ROCDL buffer resource descriptor (!llvm.ptr<8>)

    Example:
        >>> rsrc = create_buffer_resource(A)
        >>> data = buffer_load(rsrc, offset)
    """
    desc = BufferResourceDescriptor.from_memref(
        memref_val,
        stride,
        max_size,
        num_records_bytes=num_records_bytes,
        base_byte_offset=base_byte_offset,
    )
    return desc.rsrc


@dsl_loc_tracing
def buffer_load(
    rsrc: 'ir.Value',
    offset: 'ir.Value',
    vec_width: 'int' = 4,
    dtype=None,
    mask: 'ir.Value | None' = None,
    cache_modifier: 'int' = 0,
    soffset_bytes: 'int | ir.Value | None' = None,
    is_scalar: 'bool' = False,
) -> 'ir.Value':
    """AMD buffer load operation.

    Load data from global memory using buffer descriptor and offset.
    Uses hardware-level bounds checking and vectorization.

    Args:
        rsrc: Buffer resource descriptor (!llvm.ptr<8>)
        offset: Offset in elements (i32 type)
        vec_width: Vector width (1, 2, or 4)
        dtype: Element data type (None for f32, or ir.F32Type, etc.)
        mask: Optional mask for predicated load (i1 type)
        cache_modifier: Cache control flags (0 for default)
        soffset_bytes: Optional scalar offset (in BYTES) added by the buffer instruction (soffset).
                      Use this to fold small constant deltas into the instruction instead of emitting
                      extra VGPR address arithmetic.
        is_scalar: Emit a uniform/SGPR scalar load (llvm.amdgcn.s.buffer.load) instead of the
                      vector buffer load. Use only for wave-uniform addresses to route through the
                      SMEM cache and land the result directly in SGPRs. Restricted to vec_width 1 or 4;
                      dtype is forced to i32 (the result is raw i32 dwords). mask and soffset_bytes
                      are not supported in this mode and raise ValueError if provided.

    Returns:
        Loaded data (scalar or vector depending on vec_width)

    Example:
        >>> # Load 4xf32
        >>> data = buffer_load(rsrc, offset, vec_width=4)
        >>>
        >>> # Load with mask
        >>> data = buffer_load(rsrc, offset, vec_width=4, mask=valid)
    """
    # Scalar (uniform) loads return raw i32 dwords; force the element type so the
    # element->byte offset math below uses 4 and the result type is i32 / v4i32.
    if is_scalar:
        if vec_width not in (1, 4):
            raise ValueError(
                f"buffer_load(is_scalar=True): unsupported vec_width={vec_width}"
            )
        if mask is not None or soffset_bytes is not None:
            raise ValueError(
                "buffer_load(is_scalar=True) does not support mask or soffset_bytes"
            )
        dtype = _buffer_ops_T.i32()
    # Default dtype to f32
    elif dtype is None:
        dtype = _buffer_ops_T.f32()
    # Accept DSL Numeric class (e.g. fx.Int32) as dtype: unwrap to ir.Type
    elif hasattr(dtype, "ir_type"):
        dtype = dtype.ir_type

    # Unwrap offset first (accept Python ints and DSL Numeric values).
    if isinstance(offset, int):
        offset = _create_i32_constant(offset)
    elif hasattr(offset, "ir_value"):
        offset = offset.ir_value()
    offset = _unwrap_value(offset)

    # Convert offset to i32 if needed (accepts index/i64/i32)
    offset = _to_i32_offset(offset)

    # IMPORTANT: Buffer load offset is in BYTES, not elements!
    # For vec4xf32, each element is 4 bytes, so multiply offset by 4
    element_bytes = dtype.width // 8
    bytes_const = _create_i32_constant(element_bytes)
    op = std_arith.MulIOp(offset, bytes_const)
    offset = _unwrap_value(op.result)

    # Apply mask by setting invalid offsets to max
    if mask is not None:
        mask = _unwrap_value(mask)
        max_offset = _create_i32_constant(0x7FFFFFFF)
        op = std_arith.SelectOp(mask, offset, max_offset)
        offset = _unwrap_value(op.result)

    # Create vector type
    if vec_width == 1:
        result_type = dtype
    else:
        result_type = ir.VectorType.get([vec_width], dtype)

    # Scalar/uniform load path: emit s.buffer.load with a v4i32 resource and the
    # byte offset computed above. Returns i32 (vec_width 1) or v4i32 (vec_width 4).
    if is_scalar:
        rsrc_v4 = _ptr8_to_v4i32(rsrc)
        cache_policy = _create_i32_constant(cache_modifier)
        suffix = "i32" if vec_width == 1 else "v4i32"
        return llvm.call_intrinsic(
            result_type,
            f"llvm.amdgcn.s.buffer.load.{suffix}",
            [rsrc_v4, offset, cache_policy],
            [],
            [],
        )

    # Create instruction offset and aux flags
    if soffset_bytes is None:
        soffset = _create_i32_constant(0)
    else:
        if isinstance(soffset_bytes, int):
            soffset = _create_i32_constant(soffset_bytes)
        else:
            soffset = _to_i32_offset(_unwrap_value(soffset_bytes))
    aux_attr = (
        ir.IntegerAttr.get(ir.IntegerType.get_signless(32), cache_modifier)
        if cache_modifier
        else None
    )

    # Emit buffer load
    load_op = _buffer_ops_rocdl.RawPtrBufferLoadOp(
        result_type,
        rsrc,
        offset,
        soffset,
        aux=aux_attr,
    )

    return load_op.result


@dsl_loc_tracing
def buffer_store(
    data: 'ir.Value',
    rsrc: 'ir.Value',
    offset: 'ir.Value',
    mask: 'ir.Value | None' = None,
    cache_modifier: 'int' = 0,
    *,
    soffset_bytes: 'int | ir.Value | None' = None,
    offset_is_bytes: 'bool' = False,
):
    """AMD buffer store operation.

    Store data to global memory using buffer descriptor and offset.

    Args:
        data: Data to store (scalar or vector)
        rsrc: Buffer resource descriptor (!llvm.ptr<8>)
        offset: Offset in elements (i32 type)
        mask: Optional mask for predicated store (i1 type)
        cache_modifier: Cache control flags (0 for default)

    Example:
        >>> buffer_store(data, rsrc, offset)
        >>>
        >>> # Store with mask
        >>> buffer_store(data, rsrc, offset, mask=valid)
    """
    # Unwrap all inputs (accept DSL Numeric values via ir_value())
    if hasattr(data, "ir_value"):
        data = data.ir_value()
    if isinstance(offset, int):
        offset = _create_i32_constant(offset)
    elif hasattr(offset, "ir_value"):
        offset = offset.ir_value()
    data = _unwrap_value(data)
    rsrc = _unwrap_value(rsrc)
    offset = _unwrap_value(offset)

    # Convert offset to i32 if needed (accepts index/i64/i32)
    offset = _to_i32_offset(offset)

    # IMPORTANT: RawPtrBufferStoreOp offset is in BYTES.
    # For backward compat, `buffer_store()` accepts element offsets by default
    # and scales them to bytes. Set `offset_is_bytes=True` to skip scaling.
    if not offset_is_bytes:
        # Get element size from data type
        data_type = data.type
        if hasattr(data_type, "element_type"):  # Vector type
            element_type = data_type.element_type
        else:  # Scalar type
            element_type = data_type
        element_bytes = element_type.width // 8
        bytes_const = _create_i32_constant(element_bytes)
        op = std_arith.MulIOp(offset, bytes_const)
        offset = _unwrap_value(op.result)

    # Apply mask by setting invalid offsets to max
    if mask is not None:
        mask = _unwrap_value(mask)
        max_offset = _create_i32_constant(0x7FFFFFFF)
        op = std_arith.SelectOp(mask, offset, max_offset)
        offset = _unwrap_value(op.result)

    # Create instruction offset (soffset) and aux flags
    if soffset_bytes is None:
        soffset = _create_i32_constant(0)
    else:
        if isinstance(soffset_bytes, int):
            soffset = _create_i32_constant(int(soffset_bytes))
        else:
            soffset = _to_i32_offset(_unwrap_value(soffset_bytes))
    aux_attr = (
        ir.IntegerAttr.get(ir.IntegerType.get_signless(32), cache_modifier)
        if cache_modifier
        else None
    )

    # Emit buffer store
    _buffer_ops_rocdl.RawPtrBufferStoreOp(
        data,
        rsrc,
        offset,
        soffset,
        aux=aux_attr,
    )
buffer_ops = types.SimpleNamespace(**{_n: globals()[_n] for _n in ['_get_buffer_flags', '__all__', '_unwrap_value', '_create_i32_constant', '_to_i32_offset', '_create_i16_constant', '_create_i64_constant', '_ptr8_to_v4i32', 'create_llvm_ptr', 'extract_base_index', 'get_element_ptr', 'BufferResourceDescriptor', 'create_buffer_resource_from_addr', 'create_buffer_resource', 'buffer_load', 'buffer_store']})


# ---- kernels/tensor_shim.py ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.




# Global toggle for the amdgpu-kernarg-preload compile hint used by the flydsl
# kernels. Enabled by default; set AITER_FLYDSL_KERNARG_PRELOAD=0 to disable it
# globally for all kernels. AITER_FLYDSL_KERNARG_PRELOAD_COUNT overrides the
# number of kernel arguments to preload.
AITER_FLYDSL_KERNARG_PRELOAD = bool(
    1
)
AITER_FLYDSL_KERNARG_PRELOAD_COUNT = int(
    "32"
)

# Toggle for the amdgpu-expert-scheduling-mode compile hint on the MoE GEMM
# kernels. Disabled by default; set AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1
# to enable it.
AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE = bool(
    0
)


def ptr_rsrc(ptr):
    """Convert an fx.Pointer kernel arg to a buffer resource for buffer_load/store."""
    return buffer_ops.create_buffer_resource_from_addr(fx.Int64(ptrtoint(ptr)))


_BUF_COPY_ATOM = {
    16: fx.rocdl.BufferCopy128b,
    8: fx.rocdl.BufferCopy64b,
    4: fx.rocdl.BufferCopy32b,
    1: fx.rocdl.BufferCopy8b,
}

# Nominal extent of a raw-pointer buffer view. The real bound is the V#
# num_records field, which make_buffer_tensor(max_size=True) pins to 0xFFFFFFFF
# bytes -- the same descriptor ptr_rsrc builds -- so this only has to be large
# enough not to constrain any caller's indices.
BUF_VIEW_MAX_ELEMS = 0xFFFFFFFF


def ptr_buf_tensor(
    ptr, elem=fx.Int32, n=BUF_VIEW_MAX_ELEMS, unit_elems=1, num_records_bytes=None
):
    """Buffer-resource (V#) view of *ptr*, so ``t[i]`` / ``fx.slice`` index it.

    Keeps the addressing `buffer_ops` used: descriptor in SGPRs, 32-bit voffset
    per access. Indexing a plain typed pointer instead builds a full 64-bit
    address in VGPRs on every access.

    ``unit_elems`` sets the access width and hence the rank:
      1  -> flat ``(n,)``; ``t[i]`` is one element. No atom, no fragment.
      >1 -> ``(n, unit_elems)``; ``fx.slice(t, (u, None))`` is one wide access
            and ``fx.copy`` over it emits a single ``buffer_load_dwordx{2,4}``.
            A vector *element type* would be the obvious spelling for this but
            MLIR rejects it (``AlignAttr`` takes integer/float only), so the
            width lives in the layout.

    ``num_records_bytes`` may be a runtime value, for a hardware OOB check that
    zero-fills rather than reading stale bytes.
    """
    layout = (
        fx.make_layout((n,), (1,))
        if unit_elems == 1
        else fx.make_layout((n, unit_elems), (unit_elems, 1))
    )
    pt = fx.PointerType.get(
        elem.ir_type,
        address_space=fx.AddressSpace.Global,
        alignment=unit_elems * (elem.width // 8),
    )
    view = fx.make_view(fx.inttoptr(pt, fx.Int64(ptrtoint(ptr))), layout)
    return fx.rocdl.make_buffer_tensor(view, num_records_bytes=num_records_bytes)


def buf_copy_atom(unit_bytes, elem=fx.Int32):
    """Copy atom for a ``unit_bytes``-wide buffer access."""
    return fx.make_copy_atom(_BUF_COPY_ATOM[unit_bytes](), elem)


def ptr_arg(t: torch.Tensor, dtype=None):
    """Wrap a torch.Tensor as an fx.Pointer (PointerJitArg) for kernel launch."""
    if dtype is None:
        dtype = fx.Uint8
    type_name = type(t).__name__
    module_name = type(t).__module__
    if type_name == "FakeTensor" or "fake_tensor" in module_name:
        return flyc.from_c_void_p(dtype, 0)
    return flyc.from_c_void_p(dtype, t.data_ptr())


def _run_compiled(exe, *args):
    """First call: ``flyc.compile(exe, *args)`` compiles **and** executes the kernel.
    Subsequent calls: fast dispatch via the cached ``CompiledFunction``.
    """
    cf = getattr(exe, "_cf", None)
    if cf is not None:
        cf(*args)
        return
    try:
        cf = flyc.compile(exe, *args)
        exe._cf = cf
    except Exception:
        # flyc.compile leaks ir.Context on failure; pop it so a retry takes the right path.
        try:
            while ir.Context.current is not None:
                ir.Context.current.__exit__(None, None, None)
        except Exception:  # noqa: BLE001, S110
            pass
        raise


def _to_raw(v):
    """Convert ArithValue / Numeric (Int32, Boolean, …) to raw ir.Value."""
    if isinstance(v, ir.Value):
        return v
    if hasattr(v, "ir_value"):
        return _to_raw(v.ir_value())
    return ir.Value._CAPICreate(v._CAPIPtr)


def get_dtype_str(dtype):
    if dtype == torch.float:
        return "f32"
    elif dtype == torch.half:
        return "f16"
    elif dtype == torch.bfloat16:
        return "bf16"


def get_dtype_in_kernel(dtype: str):
    if dtype == "f32":
        return T.f32
    elif dtype == "f16":
        return T.f16
    elif dtype == "bf16":
        return T.bf16


def get_dtype_vec_size(dtype: str):
    if dtype == "f32":
        return 4
    elif dtype == "f16" or dtype == "bf16":
        return 8


def get_dtype_bytes(dtype: str):
    if dtype == "f32":
        return 4
    elif dtype == "f16" or dtype == "bf16":
        return 2


class TensorView:
    def __init__(self, dtype, shape, stride, base_offset, load_impl, store_impl):
        self.dtype = dtype
        self.shape = shape
        if stride is None:
            self.stride = tuple(
                (
                    _suffix_products(shape)
                    + [
                        1,
                    ]
                )[1:]
            )
        else:
            self.stride = stride
        self.base_offset = base_offset
        self.load_impl = load_impl
        self.store_impl = store_impl

    def _linear_offset(self, idxs):
        slice_shape = []
        slice_stride = []
        d_offset = self.base_offset
        for i in range_constexpr(len(idxs)):
            md_id = idxs[i]
            if md_id is None:
                slice_shape.append(self.shape[i])
                slice_stride.append(self.stride[i])
            elif isinstance(md_id, int):
                d_offset = d_offset + md_id * self.stride[i]
            else:
                d_offset = d_offset + md_id * self.stride[i]
        if len(slice_shape) > 0:
            return d_offset, tuple(slice_shape), tuple(slice_stride)
        else:
            return (d_offset,)

    def _lazy_init(self):
        pass

    def __repr__(self):
        return f"TensorView(offset={self.base_offset}, shape={self.shape}, stride={self.stride}, dtype={self.dtype})"

    def __getitem__(self, idxs):
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        offset = self._linear_offset(idxs)
        if len(offset) == 1:
            return self.load_impl(offset[0])
        else:
            return TensorView(
                self.dtype,
                offset[1],
                offset[2],
                offset[0],
                self.load_impl,
                self.store_impl,
            )

    def __setitem__(self, idxs, value):
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        offset = self._linear_offset(idxs)
        assert len(offset) == 1
        self.store_impl(offset[0], value)

    def vec_load(self, idxs, vec_size):
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        offset = self._linear_offset(idxs)
        assert len(offset) == 1
        return self.load_impl(offset[0], vec_size=vec_size)

    def vec_store(self, idxs, value, vec_size):
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        offset = self._linear_offset(idxs)
        assert len(offset) == 1
        self.store_impl(offset[0], value, vec_size=vec_size)

    def linear_offset(self, idxs):
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        offset = self._linear_offset(idxs)
        assert len(offset) == 1
        return offset[0]

    def local_tile(self, tile_shape, tile_idxs):
        d_offset = self.base_offset
        stride = []
        for i in range_constexpr(len(tile_idxs)):
            d_offset = d_offset + tile_idxs[i] * tile_shape[i] * self.stride[i]
            stride.append(self.stride[i])
        return TensorView(
            self.dtype,
            tile_shape,
            tuple(stride),
            d_offset,
            self.load_impl,
            self.store_impl,
        )

    def copy_(self, src_tensor, thread_layout, value_layout, thread_idxs, vec_size):
        src_tensor._lazy_init()
        ndim = len(thread_layout)
        src_offset = src_tensor.base_offset
        dst_offset = self.base_offset
        for d in range_constexpr(ndim):
            src_offset = (
                src_offset + thread_idxs[d] * value_layout[d] * src_tensor.stride[d]
            )
            dst_offset = dst_offset + thread_idxs[d] * value_layout[d] * self.stride[d]
        value_layout_v = value_layout[:-1] + (value_layout[-1] // vec_size,)
        coords = tuple(product(*(range_constexpr(s) for s in value_layout_v)))
        for coord in coords:
            src_vec_offset = src_offset
            dst_vec_offset = dst_offset
            for d in range_constexpr(len(coord)):
                if d == len(coord) - 1:
                    src_vec_offset = (
                        src_vec_offset + coord[d] * src_tensor.stride[d] * vec_size
                    )
                    dst_vec_offset = (
                        dst_vec_offset + coord[d] * self.stride[d] * vec_size
                    )
                else:
                    src_vec_offset = src_vec_offset + coord[d] * src_tensor.stride[d]
                    dst_vec_offset = dst_vec_offset + coord[d] * self.stride[d]
            value = src_tensor.load_impl(src_vec_offset, vec_size=vec_size)
            self.store_impl(dst_vec_offset, value, vec_size=vec_size)


class TensorBase(ABC):
    def __init__(self, dtype, shape, stride=None, base_offset=0):
        self.tensor_view = None
        self.dtype = dtype
        self.shape = shape
        self.stride = stride
        self.base_offset = base_offset

    @abstractmethod
    def load(self, offset):
        return None

    @abstractmethod
    def store(self, offset, value):
        pass

    def _lazy_init(self):
        if self.tensor_view is None:
            self.tensor_view = TensorView(
                self.dtype,
                self.shape,
                self.stride,
                self.base_offset,
                self.load,
                self.store,
            )
            self.stride = self.tensor_view.stride
            self.load_impl = self.tensor_view.load_impl
            self.store_impl = self.tensor_view.store_impl

    def __repr__(self):
        self._lazy_init()
        return self.tensor_view.__repr__()

    def __getitem__(self, idxs):
        self._lazy_init()
        return self.tensor_view[idxs]

    def __setitem__(self, idxs, value):
        self._lazy_init()
        self.tensor_view[idxs] = value

    def vec_load(self, idxs, vec_size):
        self._lazy_init()
        return self.tensor_view.vec_load(idxs, vec_size)

    def vec_store(self, idxs, value, vec_size):
        self._lazy_init()
        self.tensor_view.vec_store(idxs, value, vec_size)

    def linear_offset(self, idxs):
        self._lazy_init()
        return self.tensor_view.linear_offset(idxs)

    def local_tile(self, tile_shape, tile_idxs):
        self._lazy_init()
        return self.tensor_view.local_tile(tile_shape, tile_idxs)

    def copy_(self, src_tensor, thread_layout, value_layout, thread_idxs, vec_size):
        self._lazy_init()
        self.tensor_view.copy_(
            src_tensor, thread_layout, value_layout, thread_idxs, vec_size
        )


class TorchTensor(TensorBase):
    def __init__(self, torch_tensor, dtype, shape, stride=None, base_offset=0):
        super().__init__(dtype, shape, stride, base_offset)
        self.torch_tensor = torch_tensor

    def load(self, offset, vec_size=1):
        return self.torch_tensor.view(-1)[offset : offset + vec_size]

    def store(self, offset, value, vec_size=1):
        self.torch_tensor.view(-1)[offset : offset + vec_size] = value


class GTensor(TensorBase):
    def __init__(
        self,
        memref,
        dtype,
        shape,
        stride=None,
        base_offset=0,
        cache_modifier=0,
        static_bytes_offset_i64=None,
    ):
        super().__init__(dtype, shape, stride, base_offset)
        raw = extract_to_ir_values(memref)[0]
        if static_bytes_offset_i64 is None:
            if str(raw.type).startswith("!fly.ptr"):
                base_i64 = fx.Int64(ptrtoint(memref))
                self.rsrc = buffer_ops.create_buffer_resource_from_addr(base_i64)
            else:
                self.rsrc = buffer_ops.create_buffer_resource(memref, max_size=True)
        else:
            array_base_i64 = self.get_llvm_ptr(memref, (static_bytes_offset_i64))
            self.rsrc = buffer_ops.create_buffer_resource_from_addr(array_base_i64)
        self.cache_modifier = cache_modifier

    def load(self, offset, vec_size=1):
        return buffer_ops.buffer_load(
            self.rsrc, offset, vec_width=vec_size, dtype=self.dtype
        )

    def store(self, offset, value, vec_size=1):
        buffer_ops.buffer_store(
            value, self.rsrc, offset, cache_modifier=self.cache_modifier
        )

    def get_llvm_ptr(self, ptr, bytes_offset_i64, ptr_type="!llvm.ptr<1>"):
        # fx.Int64 coerces index / i32 / i64 byte offsets to i64.
        bytes_offset_i64 = _to_raw(fx.Int64(bytes_offset_i64))
        _ptr_type = ir.Type.parse(ptr_type)
        raw = extract_to_ir_values(ptr)[0]
        if str(raw.type).startswith("!fly.ptr"):
            base_ptr = _to_raw(fx.Int64(ptrtoint(ptr)))
        else:
            base_ptr = fly.extract_aligned_pointer_as_index(_ptr_type, raw)
            base_ptr = llvm.PtrToIntOp(T.i64, base_ptr).result
        llvm_ptr = llvm.AddOp(
            base_ptr, bytes_offset_i64, llvm.IntegerOverflowFlags(0)
        ).result
        return llvm_ptr


class STensor(TensorBase):
    def __init__(self, memptr, dtype, shape, stride=None, base_offset=0):
        super().__init__(dtype, shape, stride, base_offset)
        self.memptr = memptr.get()

    def load(self, offset, vec_size=1):
        vec_t = T.vec(vec_size, self.dtype)
        x = vector.load_op(vec_t, self.memptr, [offset])
        if vec_size > 1:
            return x
        else:
            x = vector.extract(x, static_position=[0], dynamic_position=[])
            return x

    def store(self, offset, value, vec_size=1):
        if vec_size > 1:
            vector.store(value, self.memptr, [offset], alignment=16)
        else:
            vec_t = T.vec(1, self.dtype)
            vec = vector.from_elements(vec_t, [value])
            vector.store(vec, self.memptr, [offset], alignment=16)
tensor_shim = types.SimpleNamespace(**{_n: globals()[_n] for _n in ['AITER_FLYDSL_KERNARG_PRELOAD', 'AITER_FLYDSL_KERNARG_PRELOAD_COUNT', 'AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE', 'ptr_rsrc', '_BUF_COPY_ATOM', 'BUF_VIEW_MAX_ELEMS', 'ptr_buf_tensor', 'buf_copy_atom', 'ptr_arg', '_run_compiled', '_to_raw', 'get_dtype_str', 'get_dtype_in_kernel', 'get_dtype_vec_size', 'get_dtype_bytes', 'TensorView', 'TensorBase', 'TorchTensor', 'GTensor', 'STensor']})


# ---- kernels/splitk_hgemm.py -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.





SPLIT_K_SEMAPHORE_MAX_LEN = 256


def swizzle_xor16(row, col_in_bytes, k_blocks16):
    return col_in_bytes ^ ((row % k_blocks16) * 16)


class WmmaHalfBase(ABC):
    @abstractmethod
    def __init__(self, dtype: str):
        pass

    @abstractmethod
    def __call__(self, a_frag, b_frag, c_frag):
        pass


class WmmaHalf_m16n16k16(WmmaHalfBase):
    WMMA_M = 16
    WMMA_N = 16
    WMMA_K = 16
    WMMA_A_FRAG_VALUES = 4
    WMMA_B_FRAG_VALUES = 4
    WMMA_C_FRAG_VALUES = 4

    def __init__(self, dtype: str):
        self.dtype = dtype

    def __call__(self, a_frag, b_frag, c_frag):
        if self.dtype == "bf16":
            a_frag_vi16 = vector.bitcast(T.vec(self.WMMA_A_FRAG_VALUES, T.i16), a_frag)
            b_frag_vi16 = vector.bitcast(T.vec(self.WMMA_B_FRAG_VALUES, T.i16), b_frag)
            return rocdl.mfma_f32_16x16x16bf16_1k(
                T.f32x4, [a_frag_vi16, b_frag_vi16, c_frag, 0, 0, 0]
            )
        return rocdl.mfma_f32_16x16x16f16(
            T.vec(self.WMMA_C_FRAG_VALUES, T.f32), [a_frag, b_frag, c_frag, 0, 0, 0]
        )


class WmmaHalf_m16n16k32(WmmaHalfBase):
    WMMA_M = 16
    WMMA_N = 16
    WMMA_K = 32
    WMMA_A_FRAG_VALUES = 8
    WMMA_B_FRAG_VALUES = 8
    WMMA_C_FRAG_VALUES = 4

    def __init__(self, dtype: str):
        self.dtype = dtype

    def __call__(self, a_frag, b_frag, c_frag):
        if self.dtype == "bf16":
            return rocdl.mfma_f32_16x16x32_bf16(
                T.vec(self.WMMA_C_FRAG_VALUES, T.f32), [a_frag, b_frag, c_frag, 0, 0, 0]
            )
        return rocdl.mfma_f32_16x16x32_f16(
            T.vec(self.WMMA_C_FRAG_VALUES, T.f32), [a_frag, b_frag, c_frag, 0, 0, 0]
        )


class OnlineScheduler:
    def __init__(self, total_signals: int, init_count: int = 0):
        self.total_signals = total_signals
        self.current_signal_id = init_count
        self.remaining = init_count

    def release(self, count: int):
        count = min(count, self.total_signals - self.current_signal_id)
        self.current_signal_id += count
        self.remaining += count

    def consume(self, count: int):
        count = min(count, self.remaining)
        self.remaining -= count
        return count


@functools.lru_cache(maxsize=16384)
def compile_hgemm_kernel(
    dtype: str,
    n: int,
    k: int,
    TILE_M: int = 128,
    TILE_N: int = 128,
    TILE_K: int = 64,
    STAGES: int = 2,
    SPLIT_K: int = 1,
    BLOCK_M_WARPS: int = 2,
    BLOCK_N_WARPS: int = 2,
    BLOCK_K_WARPS: int = 1,
    B_TO_LDS: bool = False,
    HAS_BIAS: bool = False,
):
    assert BLOCK_M_WARPS * BLOCK_N_WARPS * BLOCK_K_WARPS <= 16
    assert TILE_M * TILE_N * TILE_K <= 256 * 256 * 64
    if (TILE_M == 256) and (TILE_N == 256):
        assert (TILE_K == 64) and (SPLIT_K == 1) and (STAGES == 2)
    assert STAGES >= 2
    N_BLOCKS = n // TILE_N
    assert (N_BLOCKS >= 1) and (n % TILE_N == 0)
    IS_SPLIT_K = SPLIT_K > 1
    IS_SLICE_K = BLOCK_K_WARPS > 1
    BLOCK_K = TILE_K
    assert (k % SPLIT_K == 0) and (k // SPLIT_K >= 1)
    ks = k // SPLIT_K
    assert (ks % BLOCK_K == 0) and (ks // BLOCK_K >= 1)
    assert BLOCK_K >= 32
    GPU_ARCH = get_rocm_arch()
    if GPU_ARCH == "gfx942":
        WMMA_IMPL = WmmaHalf_m16n16k16(dtype)
        DMA_BYTES = 4
        MFMA_PER_WARP_K = 2
        ASYNC_COPY = True
    else:
        WMMA_IMPL = WmmaHalf_m16n16k32(dtype)
        DMA_BYTES = 16
        MFMA_PER_WARP_K = 1
        ASYNC_COPY = True

    # Fixed parameters:
    WARP_SIZE = 64
    DTYPE_BYTES = 2
    LDG_VEC_SIZE = 8

    # Propagated parameters:
    WMMA_M = WMMA_IMPL.WMMA_M
    WMMA_N = WMMA_IMPL.WMMA_N
    WMMA_K = WMMA_IMPL.WMMA_K
    WMMA_A_FRAG_VALUES = WMMA_IMPL.WMMA_A_FRAG_VALUES
    WMMA_B_FRAG_VALUES = WMMA_IMPL.WMMA_B_FRAG_VALUES
    WMMA_C_FRAG_VALUES = WMMA_IMPL.WMMA_C_FRAG_VALUES
    WARP_ATOM_M = WMMA_M
    WARP_ATOM_N = WMMA_N
    WARP_ATOM_K = WMMA_K * MFMA_PER_WARP_K
    BLOCK_K_LOOPS = ks // BLOCK_K
    assert BLOCK_K_LOOPS >= STAGES
    WARP_GROUP_K = BLOCK_K_WARPS * WARP_ATOM_K
    WARP_K_STEPS = BLOCK_K // WARP_GROUP_K
    assert (BLOCK_K % WARP_GROUP_K == 0) and (WARP_K_STEPS >= 1)
    K_SLICE = BLOCK_K // BLOCK_K_WARPS
    assert K_SLICE % WARP_ATOM_K == 0
    BLOCK_THREADS = BLOCK_M_WARPS * BLOCK_N_WARPS * BLOCK_K_WARPS * WARP_SIZE
    BLOCK_MN_WARPS = BLOCK_M_WARPS * BLOCK_N_WARPS
    WARP_M_STEPS = TILE_M // BLOCK_M_WARPS // WARP_ATOM_M
    WARP_N_STEPS = TILE_N // BLOCK_N_WARPS // WARP_ATOM_N
    assert (WARP_M_STEPS >= 1) and (WARP_N_STEPS >= 1)
    assert TILE_M % (BLOCK_M_WARPS * WARP_ATOM_M) == 0
    assert TILE_N % (BLOCK_N_WARPS * WARP_ATOM_N) == 0
    WARP_M = WARP_M_STEPS * WARP_ATOM_M
    WARP_N = WARP_N_STEPS * WARP_ATOM_N
    BLOCK_M = BLOCK_M_WARPS * WARP_M
    BLOCK_N = BLOCK_N_WARPS * WARP_N
    assert (n >= BLOCK_N) and (n % BLOCK_N == 0)
    BLOCK_MK_SIZE = BLOCK_M * BLOCK_K
    BLOCK_NK_SIZE = BLOCK_N * BLOCK_K
    BLOCK_MN_SIZE = BLOCK_M * BLOCK_N
    LDG_A_X_THREADS = BLOCK_K // LDG_VEC_SIZE
    # LDG_B_X_THREADS = BLOCK_K // LDG_VEC_SIZE
    LDG_C_X_THREADS = BLOCK_N // LDG_VEC_SIZE
    BLOCK_VECS = LDG_VEC_SIZE * BLOCK_THREADS
    LDG_REG_A_COUNT = BLOCK_MK_SIZE // BLOCK_VECS
    LDG_REG_B_COUNT = BLOCK_NK_SIZE // BLOCK_VECS
    LDG_REG_C_COUNT = BLOCK_MN_SIZE // BLOCK_VECS
    assert (LDG_REG_A_COUNT >= 1) and (LDG_REG_B_COUNT >= 1) and (LDG_REG_C_COUNT >= 1)
    assert BLOCK_MK_SIZE % BLOCK_VECS == 0
    assert BLOCK_NK_SIZE % BLOCK_VECS == 0
    assert BLOCK_MN_SIZE % BLOCK_VECS == 0
    BLOCK_K_BYTES = BLOCK_K * DTYPE_BYTES

    # LDS parameters: the A/B pipeline and C scratch are live at disjoint times,
    # so model their storage overlap explicitly with a union. SharedAllocator
    # emits one static LDS symbol for a union, preserving the old contiguous
    # A-then-B layout without relying on separate struct leaves being adjacent.
    AS_ELEMS = STAGES * BLOCK_M * BLOCK_K
    BS_ELEMS = STAGES * BLOCK_N * BLOCK_K
    CMN_ELEMS = BLOCK_K_WARPS * BLOCK_M * BLOCK_N
    fx_dtype = fx.Float16 if dtype == "f16" else fx.BFloat16
    if B_TO_LDS:
        assert ASYNC_COPY

        @fx.struct
        class PipelineStorage:
            a_lds: fx.Array[fx_dtype, AS_ELEMS, 16]
            b_lds: fx.Array[fx_dtype, BS_ELEMS, 16]

    else:

        @fx.struct
        class PipelineStorage:
            a_lds: fx.Array[fx_dtype, AS_ELEMS, 16]

    @fx.union
    class SharedStorage:
        pipeline: PipelineStorage
        c_lds: fx.Array[fx_dtype, CMN_ELEMS, 16]

    LDG_ASYNC_VEC_SIZE = DMA_BYTES // DTYPE_BYTES
    LDG_A_X_THREADS_AS = BLOCK_K // LDG_ASYNC_VEC_SIZE
    LDG_REG_A_COUNT_AS = BLOCK_MK_SIZE // LDG_ASYNC_VEC_SIZE // BLOCK_THREADS
    LDG_B_X_THREADS_AS = BLOCK_K // LDG_ASYNC_VEC_SIZE
    LDG_REG_B_COUNT_AS = BLOCK_NK_SIZE // LDG_ASYNC_VEC_SIZE // BLOCK_THREADS
    LDG_WAIT_COUNT = LDG_REG_B_COUNT_AS + LDG_REG_A_COUNT_AS
    assert ((STAGES - 2) * LDG_WAIT_COUNT) < 63

    USE_8WAVE_PIPE = (
        ASYNC_COPY
        and B_TO_LDS
        and BLOCK_M == 256
        and BLOCK_N == 256
        and BLOCK_K == 64
        and LDG_REG_A_COUNT_AS == 4
        and LDG_REG_B_COUNT_AS == 4
        and MFMA_PER_WARP_K == 1
    )
    USE_8WAVE_PIPE = (
        USE_8WAVE_PIPE
        and BLOCK_M_WARPS == 2
        and BLOCK_N_WARPS == 4
        and BLOCK_K_WARPS == 1
    )

    KERNEL_NAME = f"hgemm_{dtype}_{BLOCK_M}x{BLOCK_N}x{BLOCK_K}x{STAGES}_SPK{SPLIT_K}_W{BLOCK_M_WARPS}x{BLOCK_N_WARPS}x{BLOCK_K_WARPS}_BLDS{int(B_TO_LDS)}_TN"
    KERNEL_NAME += "_AS0" if not ASYNC_COPY else "_AS1"
    if HAS_BIAS:
        KERNEL_NAME += "_BIAS"

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def hgemm_kernel(
        C: fx.Pointer,
        A: fx.Pointer,
        B: fx.Pointer,
        BIAS: fx.Pointer,
        m: fx.Int32,
        semaphore: fx.Pointer,
        signal: fx.Pointer,
    ):
        dtype_ = get_dtype_in_kernel(dtype)
        c_zero_d = arith.constant(0.0, type=dtype_)
        acc_init = arith.constant_vector(0.0, T.vec(WMMA_C_FRAG_VALUES, T.f32))

        A_ = GTensor(A, dtype=dtype_, shape=(-1, k))
        B_ = GTensor(B, dtype=dtype_, shape=(n, k))
        C_ = GTensor(C, dtype=dtype_, shape=(-1, n))
        if const_expr(HAS_BIAS):
            BIAS_ = GTensor(BIAS, dtype=dtype_, shape=(n,))
        lds = fx.SharedAllocator().allocate(SharedStorage)
        a_lds_ptr = lds.pipeline.a_lds.peek().ptr
        c_lds_ptr = lds.c_lds.peek().ptr
        a_lds_i64 = fx.Int64(fx.ptrtoint(a_lds_ptr))
        if const_expr(B_TO_LDS):
            b_lds_ptr = lds.pipeline.b_lds.peek().ptr
            b_lds_i64 = fx.Int64(fx.ptrtoint(b_lds_ptr))

        def _lds_a3_ptr(base_i64, elem_off):
            off_i64 = arith.index_cast(
                T.i64, fx.Index(elem_off) * fx.Index(DTYPE_BYTES)
            )
            return buffer_ops.create_llvm_ptr(
                base_i64 + fx.Int64(off_i64), address_space=3
            )

        # LDS accessors: linear element offsets mirroring the old STensor shapes
        # as_/bs_ = (stage, row, col) over (STAGES, BLOCK*, BLOCK_K);
        # cs_ = (k_slice, row, col) over (BLOCK_K_WARPS, BLOCK_M, BLOCK_N),
        # using the C variant of the pipeline/C union.
        def as_store(stage, row, col, value):
            elem_off = (
                fx.Int64(stage) * (BLOCK_M * BLOCK_K)
                + fx.Int64(row) * BLOCK_K
                + fx.Int64(col)
            )
            fx.ptr_store(value, a_lds_ptr + elem_off)

        def as_load(stage, row, col, vec_size):
            elem_off = (
                fx.Int64(stage) * (BLOCK_M * BLOCK_K)
                + fx.Int64(row) * BLOCK_K
                + fx.Int64(col)
            )
            return fx.ptr_load(
                a_lds_ptr + elem_off,
                result_type=fx.Vector.make_type(vec_size, fx_dtype),
            )

        def bs_store(stage, row, col, value):
            elem_off = (
                fx.Int64(stage) * (BLOCK_N * BLOCK_K)
                + fx.Int64(row) * BLOCK_K
                + fx.Int64(col)
            )
            fx.ptr_store(value, b_lds_ptr + elem_off)

        def bs_load(stage, row, col, vec_size):
            elem_off = (
                fx.Int64(stage) * (BLOCK_N * BLOCK_K)
                + fx.Int64(row) * BLOCK_K
                + fx.Int64(col)
            )
            return fx.ptr_load(
                b_lds_ptr + elem_off,
                result_type=fx.Vector.make_type(vec_size, fx_dtype),
            )

        def cs_store_scalar(k_slice, row, col, value):
            elem_off = (
                fx.Int64(k_slice) * (BLOCK_M * BLOCK_N)
                + fx.Int64(row) * BLOCK_N
                + fx.Int64(col)
            )
            fx.ptr_store(value, c_lds_ptr + elem_off)

        def cs_load_vec(k_slice, row, col, vec_size):
            elem_off = (
                fx.Int64(k_slice) * (BLOCK_M * BLOCK_N)
                + fx.Int64(row) * BLOCK_N
                + fx.Int64(col)
            )
            return fx.ptr_load(
                c_lds_ptr + elem_off,
                result_type=fx.Vector.make_type(vec_size, fx_dtype),
            )

        if const_expr(IS_SPLIT_K):
            semaphore_ = GTensor(semaphore, dtype=T.i32, shape=(-1,))
            signal_ = GTensor(signal, dtype=T.i32, shape=(-1,))
            signal_idx = fx.Int32(fx.block_idx.x)

        tid = fx.thread_idx.x
        wid = tid // WARP_SIZE
        wid_mn = wid % BLOCK_MN_WARPS
        wid_k = wid // BLOCK_MN_WARPS
        w_tid = tid % WARP_SIZE

        def swizzle_for_cache_reuse(pid):
            # Do nothing currently
            return pid // N_BLOCKS, pid % N_BLOCKS

        block_m_idx, block_n_idx = swizzle_for_cache_reuse(fx.block_idx.x)
        ks_idx = fx.Index(fx.block_idx.y)
        ks_begin = arith.index_cast(T.i32, ks_idx * ks)

        m_offset = fx.Index(block_m_idx * BLOCK_M)
        n_offset = fx.Index(block_n_idx * BLOCK_N)
        k_blocks16 = fx.Int32(BLOCK_K_BYTES // 16)

        warp_m_idx = wid_mn // BLOCK_N_WARPS * WARP_M
        warp_n_idx = wid_mn % BLOCK_N_WARPS * WARP_N
        ldmatrix_a_m_idx = w_tid % WMMA_M
        ldmatrix_a_k_vec_idx = w_tid // WMMA_M * WMMA_A_FRAG_VALUES * MFMA_PER_WARP_K
        ldmatrix_b_n_idx = w_tid % WMMA_N
        ldmatrix_b_k_vec_idx = w_tid // WMMA_N * WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K
        warp_k_slice_base = wid_k * K_SLICE
        C_FRAGS_LEN = WARP_M_STEPS * WARP_N_STEPS
        c_frags = [acc_init] * C_FRAGS_LEN

        def __barrier(vmcnt=0, use_s_barrier=True):
            if const_expr(use_s_barrier):
                asm = f"s_waitcnt vmcnt({vmcnt})\n\ts_barrier"
            else:
                asm = f"s_waitcnt vmcnt({vmcnt})"
            llvm.InlineAsmOp(None, [], asm, "", has_side_effects=True)

        def get_llvm_ptr(
            ptr,
            offset,
            dtype_bytes,
            ptr_type=ir.Type.parse("!llvm.ptr<1>"),  # noqa: B008
        ):
            base_ptr = arith.index_cast(T.i64, fx.ptrtoint(ptr))
            byte_offset = arith.index_cast(
                T.i64, fx.Index(offset) * fx.Index(dtype_bytes)
            )
            llvm_ptr = llvm.AddOp(
                base_ptr, byte_offset, llvm.IntegerOverflowFlags(0)
            ).result
            llvm_ptr = llvm.IntToPtrOp(ptr_type, llvm_ptr).result
            ptr_v = (
                llvm_ptr._value if const_expr(hasattr(llvm_ptr, "_value")) else llvm_ptr
            )
            return ptr_v

        def zero_c():
            # zero c if current block is the first block
            is_t0_cond = arith.cmpi(arith.CmpIPredicate.eq, fx.Index(tid), fx.Index(0))
            cond_ks0 = arith.cmpi(arith.CmpIPredicate.eq, ks_idx, fx.Index(0))
            cond_ks0_if = scf.IfOp(cond_ks0, results_=[], has_else=False)
            with ir.InsertionPoint(cond_ks0_if.then_block):
                zero_vec = vector.broadcast(T.vec(LDG_VEC_SIZE, dtype_), c_zero_d)
                for i in range_constexpr(LDG_REG_C_COUNT):
                    global_tid = BLOCK_THREADS * i + tid
                    m_local_idx = global_tid // LDG_C_X_THREADS
                    n_local_idx = global_tid % LDG_C_X_THREADS * LDG_VEC_SIZE
                    row_idx = m_offset + fx.Index(m_local_idx)
                    init_vec = zero_vec
                    if const_expr(HAS_BIAS):
                        init_vec = BIAS_.vec_load(
                            (n_offset + n_local_idx,), LDG_VEC_SIZE
                        )
                    cond_boundary = arith.cmpi(
                        arith.CmpIPredicate.ult, row_idx, fx.Index(m)
                    )
                    cond_boundary_if = scf.IfOp(
                        cond_boundary, results_=[], has_else=False
                    )
                    with ir.InsertionPoint(cond_boundary_if.then_block):
                        bytes_offset = C_.linear_offset(
                            (row_idx, n_offset + n_local_idx)
                        )
                        bytes_offset_i32 = arith.index_cast(T.i32, bytes_offset)
                        c_ptr = get_llvm_ptr(C, bytes_offset_i32, DTYPE_BYTES)
                        llvm.InlineAsmOp(
                            None,
                            [c_ptr, init_vec],
                            "global_store_dwordx4 $0, $1, off sc0 sc1",
                            "v,v",
                            has_side_effects=True,
                        )
                        scf.YieldOp([])
                gpu.barrier()
                # trigger signal when zeroc is done by the first arrived block
                is_t0_cond_if = scf.IfOp(is_t0_cond, results_=[], has_else=False)
                with ir.InsertionPoint(is_t0_cond_if.then_block):
                    signal_ptr = get_llvm_ptr(signal, signal_idx, 4)
                    llvm.InlineAsmOp(
                        None,
                        [signal_ptr, arith.constant(1, type=T.i32)],
                        "global_store_dword $0, $1, off sc0 sc1",
                        "v,v",
                        has_side_effects=True,
                    )
                    scf.YieldOp([])
                gpu.barrier()
                scf.YieldOp([])

        def split_k_barrier():
            # spin-wait until signal triggered
            is_t0_cond = arith.cmpi(arith.CmpIPredicate.eq, fx.Index(tid), fx.Index(0))
            is_t0_cond_if = scf.IfOp(is_t0_cond, results_=[], has_else=False)
            with ir.InsertionPoint(is_t0_cond_if.then_block):
                init_cur = arith.constant(0, type=T.i32)
                w = scf.WhileOp([T.i32], [init_cur])
                before = ir.Block.create_at_start(w.before, [T.i32])
                after = ir.Block.create_at_start(w.after, [T.i32])
                with ir.InsertionPoint(before):
                    cur = before.arguments[0]
                    need_wait = arith.CmpIOp(
                        arith.CmpIPredicate.eq, cur, arith.constant(0, type=T.i32)
                    ).result
                    scf.ConditionOp(need_wait, [cur])
                with ir.InsertionPoint(after):
                    signal_ptr = get_llvm_ptr(signal, signal_idx, 4)
                    data = llvm.InlineAsmOp(
                        T.i32,
                        [signal_ptr],
                        "global_load_dword $0, $1, off sc1",
                        "=v,v",
                        has_side_effects=True,
                    ).result
                    rocdl.s_waitcnt(0)
                    scf.YieldOp([data])
                scf.YieldOp([])
            rocdl.sched_barrier(0)
            gpu.barrier()
            # clean semaphore and signal if this is the last block within split-k group
            is_t0_cond_if = scf.IfOp(is_t0_cond, results_=[], has_else=False)
            with ir.InsertionPoint(is_t0_cond_if.then_block):
                semaphore_ptr = get_llvm_ptr(semaphore, signal_idx, 4)
                arrive_idx = llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.add,
                    semaphore_ptr,
                    arith.constant(1, type=T.i32),
                    llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                    alignment=4,
                ).result
                cond_ksl = arith.cmpi(
                    arith.CmpIPredicate.eq, fx.Index(arrive_idx), fx.Index(SPLIT_K - 1)
                )
                cond_ksl_if = scf.IfOp(cond_ksl, results_=[], has_else=False)
                with ir.InsertionPoint(cond_ksl_if.then_block):
                    semaphore_[signal_idx] = arith.constant(0, type=T.i32)
                    signal_[signal_idx] = arith.constant(0, type=T.i32)
                    scf.YieldOp([])
                scf.YieldOp([])
            gpu.barrier()

        def ldg_a(k_offset):
            vecs = []
            for i in range_constexpr(LDG_REG_A_COUNT):
                global_tid = BLOCK_THREADS * i + tid
                m_local_idx = global_tid // LDG_A_X_THREADS
                k_local_idx = global_tid % LDG_A_X_THREADS * LDG_VEC_SIZE
                row_idx = m_offset + fx.Index(m_local_idx)
                safe_row_idx = arith.select(
                    arith.cmpi(arith.CmpIPredicate.ult, row_idx, fx.Index(m)),
                    row_idx,
                    fx.Index(0),
                )
                col_idx = fx.Index(k_offset + k_local_idx)
                vec = A_.vec_load((safe_row_idx, col_idx), LDG_VEC_SIZE)
                vecs.append(vec)
            return vecs

        def sts_a(vecs, lds_stage):
            for i in range_constexpr(LDG_REG_A_COUNT):
                global_tid = BLOCK_THREADS * i + tid
                m_local_idx = global_tid // LDG_A_X_THREADS
                k_local_idx = global_tid % LDG_A_X_THREADS * LDG_VEC_SIZE
                col_in_bytes = k_local_idx * DTYPE_BYTES
                col_in_bytes = swizzle_xor16(m_local_idx, col_in_bytes, k_blocks16)
                as_store(lds_stage, m_local_idx, col_in_bytes // DTYPE_BYTES, vecs[i])

        def get_dma_copy_warp_offset():
            warp_offset = rocdl.readfirstlane(
                T.i64,
                arith.index_cast(
                    T.i64,
                    fx.Index(wid) * arith.constant(WARP_SIZE * DMA_BYTES, index=True),
                ),
            )
            return warp_offset

        def buffer_load_lds_inline(rsrc, lds_ptr, global_offset):
            if const_expr(DMA_BYTES == 16):
                asm = "s_mov_b32 m0, $0\n\tbuffer_load_dwordx4 $1, $2, 0 offen sc0 lds"
            elif const_expr(DMA_BYTES == 8):
                asm = "s_mov_b32 m0, $0\n\tbuffer_load_dwordx2 $1, $2, 0 offen sc0 lds"
            elif const_expr(DMA_BYTES == 4):
                asm = "s_mov_b32 m0, $0\n\tbuffer_load_dword $1, $2, 0 offen sc0 lds"
            else:
                raise NotImplementedError(f"DMA_BYTES={DMA_BYTES} not supported")
            llvm.InlineAsmOp(
                None,
                [lds_ptr, global_offset, rsrc],
                asm,
                "s,v,s",
                has_side_effects=True,
            )

        def ldg_sts_a_async_one(ii, k_offset, write_stage, lds_ptr=None):
            global_tid = BLOCK_THREADS * ii + tid
            m_local_idx = global_tid // LDG_A_X_THREADS_AS
            k_local_idx = global_tid % LDG_A_X_THREADS_AS * LDG_ASYNC_VEC_SIZE
            col_in_bytes = k_local_idx * DTYPE_BYTES
            col_in_bytes = swizzle_xor16(m_local_idx, col_in_bytes, k_blocks16)
            row_idx = m_offset + fx.Index(m_local_idx)
            safe_row_idx = arith.select(
                arith.cmpi(arith.CmpIPredicate.ult, row_idx, fx.Index(m)),
                row_idx,
                fx.Index(0),
            )
            col_idx = fx.Index(k_offset + col_in_bytes // DTYPE_BYTES)
            global_offset = A_.linear_offset((safe_row_idx, col_idx)) * DTYPE_BYTES
            global_offset = arith.index_cast(T.i32, global_offset)
            if const_expr(lds_ptr is None):
                lds_ptr_base = _lds_a3_ptr(
                    a_lds_i64, fx.Index(write_stage) * (BLOCK_M * BLOCK_K)
                )
                lds_ptr = buffer_ops.get_element_ptr(lds_ptr_base, warp_offset)
            else:
                lds_ptr = buffer_ops.get_element_ptr(
                    lds_ptr, static_byte_offset=BLOCK_THREADS * DMA_BYTES
                )
            buffer_load_lds_inline(A_.rsrc, lds_ptr, global_offset)
            return lds_ptr

        def ldg_sts_a_async(k_offset, lds_stage):
            lds_ptr = None
            for i in range_constexpr(LDG_REG_A_COUNT_AS):
                lds_ptr = ldg_sts_a_async_one(
                    i, k_offset, lds_stage, lds_ptr if i > 0 else None
                )

        def ldg_sts_b_async_one(ii, k_offset, write_stage, lds_ptr=None):
            global_tid = BLOCK_THREADS * ii + tid
            n_local_idx = global_tid // LDG_B_X_THREADS_AS
            k_local_idx = global_tid % LDG_B_X_THREADS_AS * LDG_ASYNC_VEC_SIZE
            col_in_bytes = k_local_idx * DTYPE_BYTES
            col_in_bytes = swizzle_xor16(n_local_idx, col_in_bytes, k_blocks16)
            row_idx = n_offset + fx.Index(n_local_idx)
            safe_row_idx = arith.select(
                arith.cmpi(arith.CmpIPredicate.ult, row_idx, fx.Index(n)),
                row_idx,
                fx.Index(0),
            )
            col_idx = fx.Index(k_offset + col_in_bytes // DTYPE_BYTES)
            global_offset = B_.linear_offset((safe_row_idx, col_idx)) * DTYPE_BYTES
            global_offset = arith.index_cast(T.i32, global_offset)
            if const_expr(lds_ptr is None):
                lds_ptr_base = _lds_a3_ptr(
                    b_lds_i64, fx.Index(write_stage) * (BLOCK_N * BLOCK_K)
                )
                lds_ptr = buffer_ops.get_element_ptr(lds_ptr_base, warp_offset)
            else:
                lds_ptr = buffer_ops.get_element_ptr(
                    lds_ptr, static_byte_offset=BLOCK_THREADS * DMA_BYTES
                )
            buffer_load_lds_inline(B_.rsrc, lds_ptr, global_offset)
            return lds_ptr

        def ldg_sts_b_async(k_offset, lds_stage):
            lds_ptr = None
            for i in range_constexpr(LDG_REG_B_COUNT_AS):
                lds_ptr = ldg_sts_b_async_one(
                    i, k_offset, lds_stage, lds_ptr if i > 0 else None
                )

        def ldg_matrix_b(k_offset):
            vecs = []
            for kk in range_constexpr(WARP_K_STEPS):
                for ii in range_constexpr(WARP_N_STEPS):
                    warp_atom_n_idx = warp_n_idx + ii * WARP_ATOM_N
                    warp_atom_k_idx = warp_k_slice_base + kk * WARP_ATOM_K
                    n_idx = n_offset + warp_atom_n_idx + ldmatrix_b_n_idx
                    k_idx = k_offset + warp_atom_k_idx + ldmatrix_b_k_vec_idx
                    vec = B_.vec_load(
                        (n_idx, k_idx), WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K
                    )
                    vecs.append(vec)
            return vecs

        def ldmatrix_compute_tile_streaming(lds_stage, c_frags, initial_b_frags=None):
            s = fx.Index(lds_stage)
            c_frags_new = [cx for cx in c_frags]
            for kk in range_constexpr(WARP_K_STEPS):
                warp_atom_k_idx = warp_k_slice_base + kk * WARP_ATOM_K
                if const_expr(initial_b_frags is None):
                    b_frags = [0] * WARP_N_STEPS
                    for ii in range_constexpr(WARP_N_STEPS):
                        warp_atom_n_idx = warp_n_idx + ii * WARP_ATOM_N
                        row = warp_atom_n_idx + ldmatrix_b_n_idx
                        col_in_bytes = (
                            warp_atom_k_idx + ldmatrix_b_k_vec_idx
                        ) * DTYPE_BYTES
                        col_in_bytes = swizzle_xor16(row, col_in_bytes, k_blocks16)
                        vec = bs_load(
                            s,
                            row,
                            col_in_bytes // DTYPE_BYTES,
                            WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K,
                        )
                        b_frags[ii] = vec
                else:
                    b_frags = [
                        initial_b_frags[i]
                        for i in range_constexpr(
                            kk * WARP_N_STEPS, (kk + 1) * WARP_N_STEPS
                        )
                    ]
                a_frags = [0] * WARP_M_STEPS
                for ii in range_constexpr(WARP_M_STEPS):
                    warp_atom_m_idx = warp_m_idx + ii * WARP_ATOM_M
                    row = warp_atom_m_idx + ldmatrix_a_m_idx
                    col_in_bytes = (
                        warp_atom_k_idx + ldmatrix_a_k_vec_idx
                    ) * DTYPE_BYTES
                    col_in_bytes = swizzle_xor16(row, col_in_bytes, k_blocks16)
                    vec = as_load(
                        s,
                        row,
                        col_in_bytes // DTYPE_BYTES,
                        WMMA_A_FRAG_VALUES * MFMA_PER_WARP_K,
                    )
                    a_frags[ii] = vec
                rocdl.sched_barrier(0)
                for ii in range_constexpr(WARP_M_STEPS):
                    a_frag = a_frags[ii]
                    for jj in range_constexpr(WARP_N_STEPS):
                        b_frag = b_frags[jj]
                        if const_expr(MFMA_PER_WARP_K == 2):
                            # split a
                            a_i64x2 = vector.bitcast(T.i64x2, a_frag)
                            a0_i64 = vector.extract(
                                a_i64x2, static_position=[0], dynamic_position=[]
                            )
                            a1_i64 = vector.extract(
                                a_i64x2, static_position=[1], dynamic_position=[]
                            )
                            a_v0 = vector.bitcast(
                                T.f16x4, vector.from_elements(T.vec(1, T.i64), [a0_i64])
                            )
                            a_v1 = vector.bitcast(
                                T.f16x4, vector.from_elements(T.vec(1, T.i64), [a1_i64])
                            )
                            # split b
                            b_i64x2 = vector.bitcast(T.i64x2, b_frag)
                            b0_i64 = vector.extract(
                                b_i64x2, static_position=[0], dynamic_position=[]
                            )
                            b1_i64 = vector.extract(
                                b_i64x2, static_position=[1], dynamic_position=[]
                            )
                            b_v0 = vector.bitcast(
                                T.f16x4, vector.from_elements(T.vec(1, T.i64), [b0_i64])
                            )
                            b_v1 = vector.bitcast(
                                T.f16x4, vector.from_elements(T.vec(1, T.i64), [b1_i64])
                            )
                            # wmma
                            c_idx = ii * WARP_N_STEPS + jj
                            acc_in = c_frags_new[c_idx]
                            acc_mid = WMMA_IMPL(a_v0, b_v0, acc_in)
                            c_frags_new[c_idx] = WMMA_IMPL(a_v1, b_v1, acc_mid)
                        elif const_expr(MFMA_PER_WARP_K == 1):
                            c_idx = ii * WARP_N_STEPS + jj
                            c_frags_new[c_idx] = WMMA_IMPL(
                                a_frag, b_frag, c_frags_new[c_idx]
                            )
                        else:
                            raise NotImplementedError(
                                f"MFMA_PER_WARP_K={MFMA_PER_WARP_K} not supported"
                            )
            return c_frags_new

        def async_copy_ldmatrix_compute_tile_streaming(
            lds_stage,
            c_frags,
            k_offset_next_tile,
            write_stage,
        ):
            assert LDG_REG_A_COUNT_AS == 4
            assert LDG_REG_B_COUNT_AS == 4
            assert MFMA_PER_WARP_K == 1
            assert WARP_M_STEPS % 2 == 0
            assert WARP_N_STEPS % 2 == 0

            M_HALF_STEPS = WARP_M_STEPS // 2
            N_HALF_STEPS = WARP_N_STEPS // 2

            s = fx.Index(lds_stage)
            c_frags_new = [cx for cx in c_frags]
            lds_ptr_a = None
            lds_ptr_b = None

            def load_b_frag(n_step, warp_atom_k_idx):
                warp_atom_n_idx = warp_n_idx + n_step * WARP_ATOM_N
                row = warp_atom_n_idx + ldmatrix_b_n_idx
                col_in_bytes = (warp_atom_k_idx + ldmatrix_b_k_vec_idx) * DTYPE_BYTES
                col_in_bytes = swizzle_xor16(row, col_in_bytes, k_blocks16)
                return bs_load(
                    s,
                    row,
                    col_in_bytes // DTYPE_BYTES,
                    WMMA_B_FRAG_VALUES * MFMA_PER_WARP_K,
                )

            def load_a_frag(m_step, warp_atom_k_idx):
                warp_atom_m_idx = warp_m_idx + m_step * WARP_ATOM_M
                row = warp_atom_m_idx + ldmatrix_a_m_idx
                col_in_bytes = (warp_atom_k_idx + ldmatrix_a_k_vec_idx) * DTYPE_BYTES
                col_in_bytes = swizzle_xor16(row, col_in_bytes, k_blocks16)
                return as_load(
                    s,
                    row,
                    col_in_bytes // DTYPE_BYTES,
                    WMMA_A_FRAG_VALUES * MFMA_PER_WARP_K,
                )

            for kk in range_constexpr(WARP_K_STEPS):
                warp_atom_k_idx = kk * WARP_ATOM_K
                b0_frags = [0] * N_HALF_STEPS  # 2
                b1_frags = [0] * N_HALF_STEPS
                a0_frags = [0] * M_HALF_STEPS  # 4
                a1_frags = [0] * M_HALF_STEPS
                if const_expr(kk == 0):
                    lds_ptr_b = ldg_sts_b_async_one(
                        0, k_offset_next_tile, write_stage, lds_ptr_b
                    )
                    lds_ptr_b = ldg_sts_b_async_one(
                        1, k_offset_next_tile, write_stage, lds_ptr_b
                    )
                for ni in range_constexpr(N_HALF_STEPS):
                    b0_frags[ni] = load_b_frag(ni, warp_atom_k_idx)
                for mi in range_constexpr(M_HALF_STEPS):
                    a0_frags[mi] = load_a_frag(mi, warp_atom_k_idx)
                if const_expr(kk == 0):
                    rocdl.s_setprio(1)
                for mi in range_constexpr(M_HALF_STEPS):
                    for ni in range_constexpr(N_HALF_STEPS):
                        c_idx = mi * WARP_N_STEPS + ni
                        c_frags_new[c_idx] = WMMA_IMPL(
                            a0_frags[mi],
                            b0_frags[ni],
                            c_frags_new[c_idx],
                        )
                if const_expr(kk == 0):
                    rocdl.s_setprio(0)
                if const_expr(kk == 0):
                    lds_ptr_a = ldg_sts_a_async_one(
                        0, k_offset_next_tile, write_stage, lds_ptr_a
                    )
                    lds_ptr_a = ldg_sts_a_async_one(
                        1, k_offset_next_tile, write_stage, lds_ptr_a
                    )
                for ni in range_constexpr(N_HALF_STEPS):
                    b1_frags[ni] = load_b_frag(N_HALF_STEPS + ni, warp_atom_k_idx)
                if const_expr(kk == 0):
                    rocdl.s_setprio(1)
                for mi in range_constexpr(M_HALF_STEPS):
                    for ni in range_constexpr(N_HALF_STEPS):
                        c_idx = mi * WARP_N_STEPS + N_HALF_STEPS + ni
                        c_frags_new[c_idx] = WMMA_IMPL(
                            a0_frags[mi],
                            b1_frags[ni],
                            c_frags_new[c_idx],
                        )
                if const_expr(kk == 0):
                    rocdl.s_setprio(0)
                    lds_ptr_b = ldg_sts_b_async_one(
                        2, k_offset_next_tile, write_stage, lds_ptr_b
                    )
                    lds_ptr_b = ldg_sts_b_async_one(
                        3, k_offset_next_tile, write_stage, lds_ptr_b
                    )
                for mi in range_constexpr(M_HALF_STEPS):
                    a1_frags[mi] = load_a_frag(M_HALF_STEPS + mi, warp_atom_k_idx)
                if const_expr(kk == 0):
                    rocdl.s_setprio(1)
                for mi in range_constexpr(M_HALF_STEPS):
                    for ni in range_constexpr(N_HALF_STEPS):
                        c_idx = (M_HALF_STEPS + mi) * WARP_N_STEPS + ni
                        c_frags_new[c_idx] = WMMA_IMPL(
                            a1_frags[mi],
                            b0_frags[ni],
                            c_frags_new[c_idx],
                        )
                if const_expr(kk == 0):
                    rocdl.s_setprio(0)
                    lds_ptr_a = ldg_sts_a_async_one(
                        2, k_offset_next_tile, write_stage, lds_ptr_a
                    )
                    lds_ptr_a = ldg_sts_a_async_one(
                        3, k_offset_next_tile, write_stage, lds_ptr_a
                    )
                    rocdl.s_setprio(1)
                for mi in range_constexpr(M_HALF_STEPS):
                    for ni in range_constexpr(N_HALF_STEPS):
                        c_idx = (M_HALF_STEPS + mi) * WARP_N_STEPS + N_HALF_STEPS + ni
                        c_frags_new[c_idx] = WMMA_IMPL(
                            a1_frags[mi],
                            b1_frags[ni],
                            c_frags_new[c_idx],
                        )
                if const_expr(kk == 0):
                    rocdl.s_setprio(0)
            return c_frags_new

        warp_offset = get_dma_copy_warp_offset()

        if const_expr(IS_SPLIT_K):
            zero_c()

        if const_expr(B_TO_LDS):

            for s in range_constexpr(STAGES - 1):
                ldg_sts_b_async(ks_begin + s * BLOCK_K, s)
                ldg_sts_a_async(ks_begin + s * BLOCK_K, s)
            rocdl.sched_barrier(0)

            def hot_loop_scheduler():
                # ================ Ordered ================
                if const_expr(USE_8WAVE_PIPE):
                    for ki in range_constexpr(WARP_K_STEPS):
                        if const_expr(ki == 0):
                            rocdl.sched_vmem(2)
                        rocdl.sched_dsrd(2)
                        rocdl.sched_dsrd(4)
                        rocdl.sched_mfma(8)
                        if const_expr(ki == 0):
                            rocdl.sched_vmem(2)
                        rocdl.sched_dsrd(2)
                        rocdl.sched_mfma(8)
                        if const_expr(ki == 0):
                            rocdl.sched_vmem(2)
                        rocdl.sched_dsrd(4)
                        rocdl.sched_mfma(8)
                        if const_expr(ki == 0):
                            rocdl.sched_vmem(2)
                        rocdl.sched_mfma(8)
                else:
                    for i in range_constexpr(LDG_REG_B_COUNT_AS):
                        rocdl.sched_vmem(1)  # ldg_sts_b_async next
                    for i in range_constexpr(LDG_REG_A_COUNT_AS):
                        rocdl.sched_vmem(1)  # ldg_sts_a_async next
                    for ki in range_constexpr(WARP_K_STEPS):
                        for i in range_constexpr(WARP_N_STEPS):
                            rocdl.sched_dsrd(1)  # lds_matrix_b current
                        for i in range_constexpr(WARP_M_STEPS):
                            rocdl.sched_dsrd(1)  # lds_matrix_a current
                        for i in range_constexpr(WARP_M_STEPS):
                            rocdl.sched_mfma(WARP_N_STEPS)
                # ================ Reordered ================
                rocdl.sched_barrier(0)

            init_state = [ks_begin, arith.constant(0, index=True)] + c_frags
            for bki, state in range(
                0, BLOCK_K_LOOPS - (STAGES - 1), 1, init=init_state
            ):
                k_offset = state[0]
                current_stage = fx.Index(state[1])
                c_frags = state[2:]
                next_stage = (current_stage + 1) % STAGES
                write_stage = (current_stage + STAGES - 1) % STAGES
                __barrier((STAGES - 2) * LDG_WAIT_COUNT)
                if const_expr(USE_8WAVE_PIPE):
                    c_frags_new = async_copy_ldmatrix_compute_tile_streaming(
                        current_stage,
                        c_frags,
                        k_offset + (STAGES - 1) * BLOCK_K,
                        write_stage,
                    )
                else:
                    ldg_sts_b_async(k_offset + (STAGES - 1) * BLOCK_K, write_stage)
                    ldg_sts_a_async(k_offset + (STAGES - 1) * BLOCK_K, write_stage)
                    c_frags_new = ldmatrix_compute_tile_streaming(
                        current_stage, c_frags
                    )
                k_offset_next = k_offset + fx.Int32(BLOCK_K)
                hot_loop_scheduler()
                results = yield [k_offset_next, next_stage] + c_frags_new
            current_stage = fx.Index(results[1])
            c_frags = results[2:]
            for s in range_constexpr(0, STAGES - 1):
                __barrier((STAGES - 2 - s) * LDG_WAIT_COUNT)
                c_frags = ldmatrix_compute_tile_streaming(current_stage, c_frags)
                current_stage = (current_stage + 1) % STAGES

        else:

            assert STAGES == 2
            sts_a(ldg_a(ks_begin), 0)
            b_frags_next = ldg_matrix_b(ks_begin)
            rocdl.sched_barrier(0)
            __barrier()

            def hot_loop_scheduler():
                LDG_REG_A_COUNT_ = (
                    LDG_REG_A_COUNT_AS if const_expr(ASYNC_COPY) else LDG_REG_A_COUNT
                )
                LDG_TOTAL = LDG_REG_A_COUNT_ + WARP_K_STEPS * WARP_N_STEPS
                # ================ Ordered ================
                for i in range_constexpr(LDG_TOTAL):
                    rocdl.sched_vmem(1)
                for ki in range_constexpr(WARP_K_STEPS):
                    for i in range_constexpr(WARP_M_STEPS):
                        rocdl.sched_dsrd(1)
                    for i in range_constexpr(WARP_M_STEPS):
                        rocdl.sched_mfma(WARP_N_STEPS)
                # ================ Reordered ================
                rocdl.sched_barrier(0)

            init_state = (
                [ks_begin, arith.constant(0, index=True)] + c_frags + b_frags_next
            )
            for bki, state in range(1, BLOCK_K_LOOPS, init=init_state):
                k_offset = state[0]
                current_stage = fx.Index(state[1])
                next_stage = 1 - current_stage
                c_frags = state[2 : 2 + C_FRAGS_LEN]
                b_frags = state[2 + C_FRAGS_LEN :]
                if const_expr(ASYNC_COPY):
                    ldg_sts_a_async(k_offset + BLOCK_K, next_stage)
                else:
                    a_regs_next = ldg_a(k_offset + BLOCK_K)
                b_frags_next = ldg_matrix_b(k_offset + BLOCK_K)
                c_frags_new = ldmatrix_compute_tile_streaming(
                    current_stage, c_frags, b_frags
                )
                if const_expr(not ASYNC_COPY):
                    sts_a(a_regs_next, next_stage)
                k_offset = k_offset + fx.Int32(BLOCK_K)
                hot_loop_scheduler()
                __barrier()
                results = yield [k_offset, next_stage] + c_frags_new + b_frags_next
            current_stage = fx.Index(results[1])
            c_frags = results[2 : 2 + C_FRAGS_LEN]
            b_frags = results[2 + C_FRAGS_LEN :]
            c_frags = ldmatrix_compute_tile_streaming(current_stage, c_frags, b_frags)

        # write to lds
        stmatrix_c_m_vec_idx = w_tid // WMMA_N * WMMA_C_FRAG_VALUES
        stmatrix_c_n_idx = w_tid % WMMA_N
        gpu.barrier()
        for ii in range_constexpr(WARP_M_STEPS):
            warp_atom_m_idx = warp_m_idx + ii * WARP_ATOM_M
            for jj in range_constexpr(WARP_N_STEPS):
                warp_atom_n_idx = warp_n_idx + jj * WARP_ATOM_N
                for kk in range_constexpr(WMMA_C_FRAG_VALUES):
                    lds_m_idx = fx.Index(warp_atom_m_idx + stmatrix_c_m_vec_idx + kk)
                    lds_n_idx = fx.Index(warp_atom_n_idx + stmatrix_c_n_idx)
                    val = vector.extract(
                        c_frags[ii * WARP_N_STEPS + jj],
                        static_position=[kk],
                        dynamic_position=[],
                    )
                    val = val.truncf(dtype_)
                    if const_expr(IS_SLICE_K):
                        cs_store_scalar(wid_k, lds_m_idx, lds_n_idx, val)
                    else:
                        cs_store_scalar(0, lds_m_idx, lds_n_idx, val)

        # write back to global
        if const_expr(IS_SPLIT_K):
            split_k_barrier()
            for i in range_constexpr(LDG_REG_C_COUNT):
                global_tid = BLOCK_THREADS * i + tid
                m_local_idx = fx.Index(global_tid // LDG_C_X_THREADS)
                n_local_idx = fx.Index(global_tid % LDG_C_X_THREADS * LDG_VEC_SIZE)
                m_global_idx = m_offset + m_local_idx
                n_global_idx = n_offset + n_local_idx
                cond_boundary = arith.cmpi(
                    arith.CmpIPredicate.ult, m_global_idx, fx.Index(m)
                )
                cond_boundary_if = scf.IfOp(cond_boundary, results_=[], has_else=False)
                with ir.InsertionPoint(cond_boundary_if.then_block):
                    pk_val = cs_load_vec(0, m_local_idx, n_local_idx, LDG_VEC_SIZE)
                    for ksi in range_constexpr(1, BLOCK_K_WARPS):
                        pk_val += cs_load_vec(
                            ksi, m_local_idx, n_local_idx, LDG_VEC_SIZE
                        )
                    linear_offset_c = C_.linear_offset((m_global_idx, n_global_idx))
                    # split to vec2s
                    vec2_ty = T.vec(2, dtype_)
                    for vec_idx in range_constexpr(LDG_VEC_SIZE // 2):
                        e0 = vector.extract(
                            pk_val, static_position=[vec_idx * 2], dynamic_position=[]
                        )
                        e1 = vector.extract(
                            pk_val,
                            static_position=[vec_idx * 2 + 1],
                            dynamic_position=[],
                        )
                        pair = vector.from_elements(vec2_ty, [e0, e1])
                        pair_v = (
                            pair._value if const_expr(hasattr(pair, "_value")) else pair
                        )
                        pair_ptr_v = get_llvm_ptr(
                            C, fx.Int32(linear_offset_c + vec_idx * 2), DTYPE_BYTES
                        )
                        llvm.AtomicRMWOp(
                            llvm.AtomicBinOp.fadd,
                            pair_ptr_v,
                            pair_v,
                            llvm.AtomicOrdering.monotonic,
                            syncscope="agent",
                            alignment=4,
                        )
                    scf.YieldOp([])
        else:
            gpu.barrier()
            for i in range_constexpr(LDG_REG_C_COUNT):
                global_tid = BLOCK_THREADS * i + tid
                m_local_idx = fx.Index(global_tid // LDG_C_X_THREADS)
                n_local_idx = fx.Index(global_tid % LDG_C_X_THREADS * LDG_VEC_SIZE)
                m_global_idx = m_offset + m_local_idx
                cond_boundary = arith.cmpi(
                    arith.CmpIPredicate.ult, m_global_idx, fx.Index(m)
                )
                cond_boundary_if = scf.IfOp(cond_boundary, results_=[], has_else=False)
                with ir.InsertionPoint(cond_boundary_if.then_block):
                    vec = cs_load_vec(0, m_local_idx, n_local_idx, LDG_VEC_SIZE)
                    for ksi in range_constexpr(1, BLOCK_K_WARPS):
                        vec += cs_load_vec(ksi, m_local_idx, n_local_idx, LDG_VEC_SIZE)
                    if const_expr(HAS_BIAS):
                        bias_vec = BIAS_.vec_load(
                            (n_offset + n_local_idx,), LDG_VEC_SIZE
                        )
                        vec = vec + bias_vec
                    C_.vec_store(
                        (m_global_idx, n_offset + n_local_idx), vec, LDG_VEC_SIZE
                    )
                    scf.YieldOp([])

    @flyc.jit
    def launch_hgemm_kernel(
        C: fx.Pointer,
        A: fx.Pointer,
        B: fx.Pointer,
        BIAS: fx.Pointer,
        m: fx.Int32,
        semaphore: fx.Pointer,
        signal: fx.Pointer,
        stream: fx.Stream,
    ):
        bm = (m + BLOCK_M - 1) // BLOCK_M
        hgemm_kernel._func.__name__ = KERNEL_NAME
        value_attrs = (
            {
                "rocdl.waves_per_eu": 2,
                "rocdl.flat_work_group_size": f"{BLOCK_THREADS},{BLOCK_THREADS}",
            }
            if USE_8WAVE_PIPE
            else None
        )
        hgemm_kernel(C, A, B, BIAS, m, semaphore, signal).launch(
            grid=(bm * N_BLOCKS, SPLIT_K, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
            value_attrs=value_attrs,
        )

    return launch_hgemm_kernel
splitk_hgemm = types.SimpleNamespace(**{_n: globals()[_n] for _n in ['SPLIT_K_SEMAPHORE_MAX_LEN', 'swizzle_xor16', 'WmmaHalfBase', 'WmmaHalf_m16n16k16', 'WmmaHalf_m16n16k32', 'OnlineScheduler', 'compile_hgemm_kernel']})

# ---- task entry point ------------------------------------------------------------------------------------------

SPLIT_K_COUNTERS = 256


def _config(m: int, n: int, k: int) -> dict:
    """One launch configuration per M bucket; every value must satisfy compile_hgemm_kernel's tiling checks."""
    if m <= 32:
        return dict(TILE_M=32, TILE_N=64, TILE_K=64, STAGES=2, SPLIT_K=1,
                    BLOCK_M_WARPS=1, BLOCK_N_WARPS=2, BLOCK_K_WARPS=1, B_TO_LDS=True)
    return dict(TILE_M=128, TILE_N=128 if n % 128 == 0 else 64, TILE_K=64, STAGES=2, SPLIT_K=1,
                BLOCK_M_WARPS=2, BLOCK_N_WARPS=2, BLOCK_K_WARPS=1, B_TO_LDS=True)


def build_gemm_a16w16_nt_n6144_k6144_module(m: int, n: int, k: int):
    cfg = _config(m, n, k)
    kernel = compile_hgemm_kernel("bf16", n, k, HAS_BIAS=False, **cfg)
    if cfg["SPLIT_K"] > 1 and ((m + cfg["TILE_M"] - 1) // cfg["TILE_M"]) * (n // cfg["TILE_N"]) > SPLIT_K_COUNTERS:
        raise ValueError("split-K needs more tile counters than SPLIT_K_COUNTERS")
    semaphore = torch.zeros((SPLIT_K_COUNTERS,), dtype=torch.int32, device="cuda")
    signal = torch.zeros((SPLIT_K_COUNTERS,), dtype=torch.int32, device="cuda")

    def launch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
        stream = torch.cuda.current_stream(a.device)
        _run_compiled(kernel, ptr_arg(out), ptr_arg(a), ptr_arg(b), ptr_arg(b), m,
                      ptr_arg(semaphore), ptr_arg(signal), fx.Stream(stream))
        return out

    return launch
