# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""AMD buffer operations used by the flash attention kernel (trimmed from FlyDSL buffer_ops).

Only the helpers the kernel calls are kept: LLVM pointer creation, element pointers, the base index
of a tensor and buffer resource descriptors.
"""

from typing import Optional, Union

from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as std_arith
from flydsl._mlir.dialects import llvm, rocdl
from flydsl._mlir.extras import types as T
from flydsl.runtime.device import get_rocm_arch, is_rdna_arch


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
        arch = get_rocm_arch()
    flags = (7 << 12) | (4 << 15)
    if is_rdna_arch(arch):
        flags |= 1 << 24  # reserved bit, must be 1 on RDNA
        flags |= 2 << 28  # OOB_SELECT = 2 (no bounds checking)
    return flags


__all__ = [
    "create_llvm_ptr",
    "get_element_ptr",
    "create_buffer_resource",
    "BufferResourceDescriptor",
    "extract_base_index",
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


def _create_i32_constant(value: int) -> ir.Value:
    """Create i32 constant using standard MLIR arith dialect."""
    i32_type = T.i32()
    if value > 0x7FFFFFFF:
        value = int(value - 2**32)
    attr = ir.IntegerAttr.get(i32_type, value)
    op = std_arith.ConstantOp(i32_type, attr)
    return _unwrap_value(op.result)


def _create_i16_constant(value: int) -> ir.Value:
    """Create i16 constant using standard MLIR arith dialect."""
    i16_type = T.i16()
    attr = ir.IntegerAttr.get(i16_type, value)
    op = std_arith.ConstantOp(i16_type, attr)
    return _unwrap_value(op.result)


def _create_i64_constant(value: int) -> ir.Value:
    """Create i64 constant using standard MLIR arith dialect."""
    i64_type = T.i64()
    attr = ir.IntegerAttr.get(i64_type, value)
    op = std_arith.ConstantOp(i64_type, attr)
    return _unwrap_value(op.result)


def create_llvm_ptr(value, address_space: int = 0) -> ir.Value:
    """Create an LLVM pointer from an integer or index value."""
    value = _unwrap_value(value)
    if isinstance(value.type, ir.IndexType):
        i64_type = T.i64()
        value = _unwrap_value(std_arith.IndexCastOp(i64_type, value).result)
    ptr_type = ir.Type.parse(f"!llvm.ptr<{address_space}>")
    return llvm.IntToPtrOp(ptr_type, value).result


def extract_base_index(tensor, address_space: int = 1) -> ir.Value:
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

    # FlyDSL 0.3 uses a typed iterator instead of the removed extract op.
    from flydsl.expr import get_iter, to_llvm_ptr
    ptr = to_llvm_ptr(get_iter(raw))
    i64_val = llvm.PtrToIntOp(ir.IntegerType.get_signless(64), ptr).result
    return _unwrap_value(std_arith.IndexCastOp(ir.IndexType.get(), i64_val).result)


def get_element_ptr(
    base_ptr,
    byte_offset: Union[int, ir.Value, None] = None,
    static_byte_offset: int = 0,
    elem_type: Optional[ir.Type] = None,
    no_wrap_flags=None,
) -> ir.Value:
    """Build an LLVM GEP from a base pointer plus byte offsets."""
    _gep_dynamic_index_sentinel = -(2**31)

    base_ptr = _unwrap_value(base_ptr)
    if not isinstance(static_byte_offset, int):
        raise TypeError(f"static_byte_offset must be int, got {type(static_byte_offset).__name__}")
    if elem_type is None:
        elem_type = T.i8()
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
            i64_type = T.i64()
            offset_val = _unwrap_value(std_arith.IndexCastOp(i64_type, offset_val).result)
        elif not isinstance(offset_val.type, ir.IntegerType):
            raise TypeError("byte_offset must be int, index, or integer-typed MLIR value; " f"got {offset_val.type}")

        if static_byte_offset != 0:
            static_type = offset_val.type
            static_attr = ir.IntegerAttr.get(static_type, int(static_byte_offset))
            static_const = _unwrap_value(std_arith.ConstantOp(static_type, static_attr).result)
            offset_val = _unwrap_value(std_arith.AddIOp(offset_val, static_const).result)

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

    def __init__(self, rsrc: ir.Value):
        """Initialize with ROCDL resource descriptor value."""
        self.rsrc = rsrc

    @staticmethod
    def from_memref(
        memref_val: ir.Value,
        stride: int = 0,
        max_size: bool = True,
        data_format: str = "f32",
        num_records_bytes: Optional[Union[int, ir.Value]] = None,
        base_byte_offset: Optional[Union[int, ir.Value]] = None,
    ) -> "BufferResourceDescriptor":
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

        # Preserve byte offsets and descriptor policy with the current pointer API.
        from flydsl.expr import get_iter, to_llvm_ptr
        base_ptr = to_llvm_ptr(get_iter(raw_val))
        if base_byte_offset is not None:
            base_ptr = get_element_ptr(base_ptr, byte_offset=base_byte_offset)

        # Create buffer resource descriptor
        flags_val = _get_buffer_flags()
        flags = _create_i32_constant(flags_val)
        stride_val = _create_i16_constant(stride)

        def _num_records_from_memref_type() -> Optional[int]:
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
            except Exception:
                return None

        if num_records_bytes is not None:
            # Caller-provided size in BYTES (preferred for exact hardware OOB behavior).
            if isinstance(num_records_bytes, int):
                nbytes = int(num_records_bytes)
                if nbytes <= 0:
                    nbytes = 0
                # Descriptor uses i32 bytes; clamp to the max representable.
                if nbytes > 0xFFFFFFFF:
                    nbytes = 0xFFFFFFFF
                num_records = _create_i64_constant(nbytes)
            else:
                v = _unwrap_value(num_records_bytes)
                i64_type = T.i64()
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
                if nbytes > 0xFFFFFFFF:
                    nbytes = 0xFFFFFFFF
                num_records = _create_i64_constant(int(nbytes))

        # Create resource descriptor (returns !llvm.ptr<8>)
        rsrc_type = ir.Type.parse("!llvm.ptr<8>")
        rsrc = rocdl.MakeBufferRsrcOp(rsrc_type, base_ptr, stride_val, num_records, flags).result

        return BufferResourceDescriptor(rsrc)


def create_buffer_resource(
    memref_val: ir.Value,
    stride: int = 0,
    max_size: bool = True,
    *,
    num_records_bytes: Optional[Union[int, ir.Value]] = None,
    base_byte_offset: Optional[Union[int, ir.Value]] = None,
) -> ir.Value:
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
