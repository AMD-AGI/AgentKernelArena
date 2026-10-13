# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Self-contained FlyDSL a8w8 b-preshuffle GEMM.

Computes ``out = (XQ @ WQ.T) * x_scale * w_scale`` in bf16/f16, where ``XQ`` is
fp8/int8 ``[M, K]`` (per-token scaled), ``WQ`` is fp8/int8 ``[N, K]`` stored in
the CK/aiter ``(16, 16)`` pre-shuffled layout (per-channel scaled), and
accumulation is in fp32. MFMA wave-level matmul with double-buffered LDS,
XOR-swizzled tiles and optional async copy / XCD swizzle on CDNA3/CDNA4.

Host entry points:

  * ``preshuffle_weight_a8(wq)``                 -> (16, 16) pre-shuffled weight
  * ``build_gemm_a8w8_bpreshuffle_module(...)``  -> compiled FlyDSL launcher
  * ``flydsl_gemm_a8w8_bpreshuffle(xq, wq, ...)``-> runs the kernel, returns [M, N]

The device kernel and its on-device shims are inlined verbatim from the FlyDSL
preshuffle-GEMM source; only host-side data prep (weight pre-shuffle) touches a
PyTorch-level layout transform.
"""
from __future__ import annotations

import functools
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Optional

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as mlir_arith
from flydsl._mlir.dialects import llvm, rocdl as mlir_rocdl
from flydsl._mlir.dialects import vector
from flydsl._mlir.dialects.arith import CmpIPredicate
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith as _arith
from flydsl.expr import const_expr, gpu, math, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch, is_rdna_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr


# The installed FlyDSL 0.3.2 package excludes its repository-level
# kernels.common.buffer_ops. Keep only the raw buffer operations this kernel
# uses here. Descriptor fields and byte offsets follow the Apache-2.0 FlyDSL
# kernels/common/buffer_ops.py definitions (ROCm/FlyDSL v0.3.2).
# Copyright (c) 2025 FlyDSL Project Contributors (Apache-2.0).
class _RawBufferOps:
    @staticmethod
    def _value(value):
        return value.ir_value() if hasattr(value, "ir_value") else value

    @staticmethod
    def _constant(bits, value):
        ty = ir.IntegerType.get_signless(bits)
        attr = ir.IntegerAttr.get(ty, int(value))
        return mlir_arith.ConstantOp(ty, attr).result

    @classmethod
    def _i32(cls, value):
        value = cls._value(value)
        if isinstance(value, int):
            return cls._constant(32, value)
        if isinstance(value.type, ir.IntegerType):
            if value.type.width == 32:
                return value
            if value.type.width > 32:
                return mlir_arith.TruncIOp(ir.IntegerType.get_signless(32), value).result
            return mlir_arith.ExtSIOp(ir.IntegerType.get_signless(32), value).result
        return mlir_arith.IndexCastOp(ir.IntegerType.get_signless(32), value).result

    @classmethod
    def create_buffer_resource_from_addr(cls, address, *, num_records_bytes=None):
        address = cls._value(address)
        base = llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr"), address).result
        flags = (7 << 12) | (4 << 15)
        if is_rdna_arch(get_hip_arch()):
            flags |= (1 << 24) | (2 << 28)
        records = 0xFFFFFFFF if num_records_bytes is None else num_records_bytes
        records = cls._value(records)
        if isinstance(records, int):
            records = cls._constant(64, records)
        elif not isinstance(records.type, ir.IntegerType) or records.type.width != 64:
            records = mlir_arith.IndexCastOp(ir.IntegerType.get_signless(64), records).result
        return mlir_rocdl.MakeBufferRsrcOp(
            ir.Type.parse("!llvm.ptr<8>"),
            base,
            cls._constant(16, 0),
            records,
            cls._constant(32, flags),
        ).result

    @classmethod
    def create_llvm_ptr(cls, address, *, address_space=0):
        address = cls._value(address)
        if isinstance(address.type, ir.IndexType):
            address = mlir_arith.IndexCastOp(ir.IntegerType.get_signless(64), address).result
        return llvm.IntToPtrOp(ir.Type.parse(f"!llvm.ptr<{address_space}>"), address).result

    @classmethod
    def get_element_ptr(cls, base, byte_offset=None, *, static_byte_offset=0):
        base = cls._value(base)
        if byte_offset is None or isinstance(byte_offset, int):
            delta = static_byte_offset + (byte_offset or 0)
            dynamic = []
            raw_offsets = [int(delta)]
        else:
            offset = cls._value(byte_offset)
            if isinstance(offset.type, ir.IndexType):
                offset = mlir_arith.IndexCastOp(ir.IntegerType.get_signless(64), offset).result
            if static_byte_offset:
                offset = mlir_arith.AddIOp(
                    offset, cls._constant(offset.type.width, static_byte_offset)
                ).result
            dynamic = [offset]
            raw_offsets = [-(2**31)]
        return llvm.GEPOp(
            base.type, base, dynamic, raw_offsets, ir.IntegerType.get_signless(8), None
        ).result

    @classmethod
    def buffer_load(cls, resource, offset, *, vec_width=4, dtype=None, cache_modifier=0):
        dtype = ir.F32Type.get() if dtype is None else getattr(dtype, "ir_type", dtype)
        dtype = dtype() if callable(dtype) else dtype
        byte_offset = mlir_arith.MulIOp(
            cls._i32(offset), cls._constant(32, dtype.width // 8)
        ).result
        result_type = dtype if vec_width == 1 else ir.VectorType.get([vec_width], dtype)
        aux = (
            ir.IntegerAttr.get(ir.IntegerType.get_signless(32), cache_modifier)
            if cache_modifier
            else None
        )
        return mlir_rocdl.RawPtrBufferLoadOp(
            result_type, cls._value(resource), byte_offset, cls._constant(32, 0), aux=aux
        ).result

    @classmethod
    def buffer_store(cls, data, resource, offset, *, offset_is_bytes=False):
        data = cls._value(data)
        byte_offset = cls._i32(offset)
        if not offset_is_bytes:
            data_type = data.type
            elem_type = data_type.element_type if hasattr(data_type, "element_type") else data_type
            byte_offset = mlir_arith.MulIOp(
                byte_offset, cls._constant(32, elem_type.width // 8)
            ).result
        mlir_rocdl.RawPtrBufferStoreOp(
            data, cls._value(resource), byte_offset, cls._constant(32, 0), aux=None
        )


buffer_ops = _RawBufferOps


# ===========================================================================
# Inlined on-device shims (mfma_preshuffle_pipeline)
# ===========================================================================
def crd2idx(crd, layout):
    """crd2idx returning an index-type scalar (unwraps fly.int_tuple)."""
    result = fx.crd2idx(crd, layout)
    scalar = fx.get_scalar(result)
    if isinstance(scalar, ir.Value) and not isinstance(scalar.type, ir.IndexType):
        scalar = _arith.IndexCastOp(T.index, scalar).result
    return scalar


def swizzle_xor16(row, col, k_blocks16):
    """XOR-with-row swizzle on the K dimension at 16B granularity.

    Computes: col XOR ((row & (k_blocks16 - 1)) * 16)

    k_blocks16 is always a power of 2 (tile_k_bytes / 16), so use
    bitwise AND instead of remui to save ~10 VALU cycles on CDNA.
    """
    from flydsl.expr import arith as _swz_arith

    mask = k_blocks16 - _swz_arith.index(1)
    rem = _swz_arith.andi(row, mask)
    return col ^ (rem * 16)


def lds_row_major_idx(row, col, row_stride, base=None):
    """Linearize a 2D LDS coordinate with explicit index arithmetic."""
    idx = row * row_stride + col
    return idx if base is None else idx + base


def split_row_major_2d(index, minor_extent):
    """Split a linear row-major index into (major, minor)."""
    return index // minor_extent, index % minor_extent


def _buffer_load_vec(
    buffer_ops,
    vector,
    rsrc,
    idx,
    *,
    elem_type,
    vec_elems,
    elem_bytes,
    offset_in_bytes,
    cache_modifier=0,
):
    """Load vec_elems elements via buffer_load dwordx[1,2,4] + bitcast."""
    from flydsl.expr import arith as _ld_arith

    elem_size = int(elem_bytes)
    load_bytes = int(vec_elems) * elem_size
    vec_width = load_bytes // 4

    if offset_in_bytes:
        idx_i32 = _ld_arith.shrui(idx, _ld_arith.index(2))
    elif elem_bytes == 2:
        idx_i32 = _ld_arith.shrui(idx, _ld_arith.index(1))
    else:
        idx_i32 = idx

    i32_val = buffer_ops.buffer_load(
        rsrc,
        idx_i32,
        vec_width=vec_width,
        dtype=T.i32,
        cache_modifier=cache_modifier,
    )
    if vec_width == 1:
        i32_vec = vector.from_elements(T.vec(1, T.i32), [i32_val])
    else:
        i32_vec = i32_val
    return vector.bitcast(T.vec(int(vec_elems), elem_type), i32_vec)


@dataclass(frozen=True)
class PreshuffleScaleLayout:
    """Container returned by `make_preshuffle_scale_layout`.

    The scale layout is ``(c_mn1, c_k1, 4, 16) : (stride_n0, stride_k0, stride_klane, 1)``.
    Callers compute flat index directly with plain arith::

        idx = mni * stride_n0 + ku * stride_k0 + k_lane * stride_klane + n_lane
    """

    layout_scale: object
    stride_n0: object
    stride_k0: object
    stride_klane: object


def make_preshuffle_scale_layout(
    arith,
    *,
    c_mn: ir.Value,
    c_k: ir.Value,
    mn_pack: int = 2,
    k_pack: int = 2,
    elem_bytes: int = 4,
    scale_block_size: int = 32,
) -> PreshuffleScaleLayout:
    """Build scale layout matching aiter/CK preshuffle for FP4/FP8 microscale.

    Layout shape: ``(c_mn1, c_k1, 4, 16)`` where
    ``c_mn1 = c_mn / 16 / mn_pack`` and ``c_k1 = (c_k / scale_block_size) / 4 / k_pack``.
    """
    c16 = fx.Index(16)
    c4 = fx.Index(4)
    c_k_scale = c_k // fx.Index(scale_block_size)

    c_mn1 = (c_mn // c16) // fx.Index(mn_pack)
    c_k1 = (c_k_scale // c4) // fx.Index(k_pack)
    if elem_bytes != mn_pack * k_pack:
        raise ValueError(
            f"elem_bytes of scale must be {mn_pack} * {k_pack}, got {elem_bytes!r}"
        )

    stride_klane = c16
    stride_k0 = c4 * stride_klane
    stride_n0 = c_k1 * stride_k0

    c_mn1_i32 = arith.index_cast(T.i32, c_mn1)
    c_k1_i32 = arith.index_cast(T.i32, c_k1)
    stride_n0_i32 = arith.index_cast(T.i32, stride_n0)
    stride_k0_i32 = arith.index_cast(T.i32, stride_k0)
    stride_klane_i32 = arith.index_cast(T.i32, stride_klane)

    layout_scale = fx.make_layout(
        (c_mn1_i32, c_k1_i32, 4, 16),
        stride=(stride_n0_i32, stride_k0_i32, stride_klane_i32, 1),
    )

    return PreshuffleScaleLayout(
        layout_scale=layout_scale,
        stride_n0=stride_n0,
        stride_k0=stride_k0,
        stride_klane=stride_klane,
    )


@dataclass(frozen=True)
class PreshuffleBLayout:
    """Container returned by `make_preshuffle_b_layout`."""

    layout_b: object
    kpack_bytes: int


def make_preshuffle_b_layout(
    arith,
    *,
    c_n: ir.Value,
    c_k: ir.Value,
    kpack_bytes: int = 16,
    elem_bytes: int = 1,
    k_major: bool = False,
) -> PreshuffleBLayout:
    """Build B layout matching aiter/CK preshuffle for A8 MFMA kernels.

    When *k_major* is True the block-level order is K-major (``k_blk`` outermost),
    matching the ``(0,3,1,4,2,5)`` shuffle permutation.  The default N-major
    order (``k_major=False``) matches the legacy ``(0,1,3,4,2,5)`` permutation.
    """
    if kpack_bytes not in (8, 16):
        raise ValueError(f"kpack_bytes must be 8 or 16, got {kpack_bytes!r}")

    c16 = fx.Index(16)
    c_kpack = fx.Index(kpack_bytes)

    if elem_bytes not in (1, 2):
        raise ValueError(f"elem_bytes must be 1 or 2, got {elem_bytes!r}")
    c_k_bytes = c_k * arith.constant(int(elem_bytes), index=True)
    n0 = c_n // c16

    c_kpack_elems = (
        c_kpack
        if elem_bytes == 1
        else (c_kpack // arith.constant(int(elem_bytes), index=True))
    )

    stride_nlane = c_kpack_elems

    if k_major:
        c32 = fx.Index(32)
        c2 = fx.Index(2)
        c_k0 = c_k_bytes // c32
        klane_dim = 2
        stride_klane = c16 * stride_nlane
        stride_n0 = c2 * stride_klane
        stride_k0 = n0 * stride_n0
    else:
        c64 = fx.Index(64)
        c4 = fx.Index(4)
        c_k0 = c_k_bytes // c64
        klane_dim = 4
        stride_klane = c16 * stride_nlane
        stride_k0 = c4 * stride_klane
        stride_n0 = c_k0 * stride_k0

    kpack_elems_static = kpack_bytes if elem_bytes == 1 else kpack_bytes // elem_bytes
    n0_i32 = arith.index_cast(T.i32, n0)
    c_k0_i32 = arith.index_cast(T.i32, c_k0)
    stride_n0_i32 = arith.index_cast(T.i32, stride_n0)
    stride_k0_i32 = arith.index_cast(T.i32, stride_k0)
    stride_klane_i32 = arith.index_cast(T.i32, stride_klane)
    stride_nlane_i32 = arith.index_cast(T.i32, stride_nlane)

    stride_b = (stride_n0_i32, stride_k0_i32, stride_klane_i32, stride_nlane_i32, 1)
    layout_b = fx.make_layout(
        (n0_i32, c_k0_i32, klane_dim, 16, kpack_elems_static), stride_b
    )
    return PreshuffleBLayout(layout_b=layout_b, kpack_bytes=kpack_bytes)


def _unpack_int4_to_int8_pair(packed32):
    """Split packed int4 dword into two int8 dwords (even/odd nibbles).

    7-op bit manipulation shared by all int4 unpack paths (W4A8, W4A16, W4A_FP8).
    """
    c_08 = fx.Int32(0x08080808)
    c_0f = fx.Int32(0x0F0F0F0F)
    c_1e = fx.Int32(0x1E)
    c_4 = fx.Int32(4)
    s0 = (packed32 & c_08) * c_1e
    even = (packed32 & c_0f) | s0
    t = packed32 >> c_4
    s1 = (t & c_08) * c_1e
    odd = (t & c_0f) | s1
    return even, odd


def _pack_i32_pair_to_i64(lo, hi, vector):
    """Pack two i32 values into one i64 via vector bitcast."""
    v2 = vector.from_elements(T.vec(2, T.i32), [lo, hi])
    v64 = vector.bitcast(T.vec(1, T.i64), v2)
    return vector.extract(v64, static_position=[0], dynamic_position=[])


def _i8x4_in_i32_to_bf16x4_i64(val_i32, arith, vector, scale_val=None):
    """Convert one i32 (4 signed int8 bytes) to 4 bf16 packed as i64.

    Uses shift-based f32->bf16 truncation (lshr 16) instead of arith.truncf
    which on gfx942 expands to ~5 VALU per element. The shift is exact for
    unscaled int8 values and introduces <0.5 ULP error for scaled values.
    """
    vec1_i32_t = T.vec(1, T.i32)
    vec2_i32 = T.i32x2
    vec4_i8 = T.i8x4
    vec1_i64 = T.vec(1, T.i64)

    v1 = vector.from_elements(vec1_i32_t, [val_i32])
    i8x4 = vector.bitcast(vec4_i8, v1)

    f32_vals = []
    for i in range(4):
        val_i8 = vector.extract(i8x4, static_position=[i], dynamic_position=[])
        v = arith.sitofp(T.f32, val_i8)
        if scale_val is not None:
            v = v * scale_val
        f32_vals.append(v)

    c16 = fx.Int32(16)
    c_ffff0000 = fx.Int32(0xFFFF0000)
    bits0 = arith.bitcast(T.i32, f32_vals[0])
    bits1 = arith.bitcast(T.i32, f32_vals[1])
    bits2 = arith.bitcast(T.i32, f32_vals[2])
    bits3 = arith.bitcast(T.i32, f32_vals[3])
    i32_lo = (bits0 >> c16) | (bits1 & c_ffff0000)
    i32_hi = (bits2 >> c16) | (bits3 & c_ffff0000)

    v2 = vector.from_elements(vec2_i32, [i32_lo, i32_hi])
    v64 = vector.bitcast(vec1_i64, v2)
    return vector.extract(v64, static_position=[0], dynamic_position=[])


def load_b_raw_w4a16(
    buffer_ops,
    arith,
    vector,
    *,
    arg_b,
    b_rsrc,
    layout_b,
    base_k: ir.Value,
    ku: int,
    n_blk: ir.Value,
    n_intra: ir.Value,
    lane_div_16: ir.Value,
    elem_type: ir.Type,
    kpack_bytes: int = 8,
):
    """Phase 1 of W4A16 B load: issue buffer_load_dword, return raw packed i32.

    Same address calculation as the int4 unpack path in load_b_pack_k32
    but using ku-based indexing for 2-phase latency hiding.
    """
    if kpack_bytes != 8:
        raise ValueError(f"W4A16 requires kpack_bytes=8, got {kpack_bytes!r}")

    c64 = fx.Index(64)
    half_bytes = kpack_bytes // 2
    c2_idx = fx.Index(2)
    c4_idx = fx.Index(4)

    k0_base = base_k // c64

    k1_layout_offset = ku * 2
    lane_div_32 = lane_div_16 // c2_idx
    total_k1 = fx.Index(k1_layout_offset) + lane_div_32
    k0 = k0_base + (total_k1 // c4_idx)
    k1_local = total_k1 % c4_idx
    lane_odd = lane_div_16 % c2_idx
    k2_base = lane_odd * fx.Index(half_bytes)

    coord_pack = (n_blk, k0, k1_local, n_intra, fx.Index(0))
    idx_pack = crd2idx(coord_pack, layout_b)
    idx_bytes = idx_pack + k2_base

    b4 = _buffer_load_vec(
        buffer_ops,
        vector,
        b_rsrc,
        idx_bytes,
        elem_type=elem_type,
        vec_elems=4,
        elem_bytes=1,
        offset_in_bytes=True,
    )
    packed32 = vector.extract(
        vector.bitcast(T.vec(1, T.i32), b4),
        static_position=[0],
        dynamic_position=[],
    )
    return packed32


def _int4_to_bf16x4_i64_gfx950(
    packed32, nibble_offsets, arith, vector, scale_val=None, defer_scale16=False
):
    """Convert 4 int4 nibbles to 4 bf16 packed as i64 using gfx950 instructions.

    Uses v_cvt_off_f32_i4_sdwa with byte_sel to avoid per-nibble shifts.
    Even nibbles (0,2,4,6) → SDWA BYTE_0/1/2/3 on original src.
    Odd nibbles (1,3,5,7)  → SDWA BYTE_0/1/2/3 on (src >> 4).
    Only 1 shift total instead of 7.

    When defer_scale16=True, the ×16 correction factor for v_cvt_off_f32_i4 is
    omitted and must be applied later (e.g. in the epilogue).  This saves VALU
    in the hot loop and uses v_cvt_pk_bf16_f32 for proper f32→bf16 conversion.
    """
    from flydsl.expr import rocdl
    from flydsl._mlir.dialects._arith_ops_gen import MulFOp as _MulFOp

    _uw = _arith._to_raw
    _av = _arith.ArithValue

    src_even = packed32
    src_odd = packed32 >> fx.Int32(4)

    f32_vals = []
    for nib in nibble_offsets:
        byte_idx = nib // 2
        src = src_odd if (nib % 2) else src_even
        v = rocdl.cvt_off_f32_i4(src, byte_sel=byte_idx)
        f32_vals.append(v)

    if defer_scale16:
        # Skip ×16; multiply by scale_val only if groupwise.
        if scale_val is not None:
            raw_scale = _uw(scale_val)
            f32_vals = [_MulFOp(v, raw_scale).result for v in f32_vals]
        # Use v_cvt_pk_bf16_f32 for proper f32→bf16 (no bit-shift trick needed).
        i32_lo = rocdl.cvt_pk_bf16_f32(f32_vals[0], f32_vals[1])
        i32_hi = rocdl.cvt_pk_bf16_f32(f32_vals[2], f32_vals[3])
    else:
        c16 = fx.Float32(16.0)
        if scale_val is not None:
            effective_scale = scale_val * c16
        else:
            effective_scale = c16
        raw_scale = _uw(effective_scale)
        f32_vals = [_MulFOp(v, raw_scale).result for v in f32_vals]
        # Truncate f32→bf16 via bit-shift (exact for scaled int values).
        c16_shift = fx.Int32(16)
        c_ffff0000 = fx.Int32(0xFFFF0000)
        bf16_vals = [arith.bitcast(T.i32, _av(v)) for v in f32_vals]
        i32_lo = (bf16_vals[0] >> c16_shift) | (bf16_vals[1] & c_ffff0000)
        i32_hi = (bf16_vals[2] >> c16_shift) | (bf16_vals[3] & c_ffff0000)

    v2 = vector.from_elements(T.vec(2, T.i32), [i32_lo, i32_hi])
    v64 = vector.bitcast(T.vec(1, T.i64), v2)
    return vector.extract(v64, static_position=[0], dynamic_position=[])


def unpack_b_w4a16(
    packed32, arith, vector, scale_val=None, use_gfx950_cvt=False, defer_scale16=False
):
    """Phase 2 of W4A16 B load: unpack int4->int8 + convert int8->bf16.

    Takes raw packed32 from load_b_raw_w4a16 and produces (b0, b1) --
    two i64 values each containing 4 bf16 for one MFMA.

    When use_gfx950_cvt=True, uses v_cvt_off_f32_i4 + v_cvt_pk_bf16_f32
    for ~2x fewer VALU instructions.

    When defer_scale16=True (requires use_gfx950_cvt=True), the ×16
    correction for v_cvt_off_f32_i4 is omitted; caller must apply it
    in the epilogue.
    """
    if use_gfx950_cvt:
        b0 = _int4_to_bf16x4_i64_gfx950(
            packed32,
            [0, 2, 4, 6],
            arith,
            vector,
            scale_val,
            defer_scale16=defer_scale16,
        )
        b1 = _int4_to_bf16x4_i64_gfx950(
            packed32,
            [1, 3, 5, 7],
            arith,
            vector,
            scale_val,
            defer_scale16=defer_scale16,
        )
        return (b0, b1)
    even, odd = _unpack_int4_to_int8_pair(packed32)
    b0 = _i8x4_in_i32_to_bf16x4_i64(even, arith, vector, scale_val=scale_val)
    b1 = _i8x4_in_i32_to_bf16x4_i64(odd, arith, vector, scale_val=scale_val)
    return (b0, b1)


def load_b_pack_k32(
    buffer_ops,
    arith,
    vector,
    *,
    arg_b,
    b_rsrc,
    layout_b,
    base_k: ir.Value,
    ki_step: int,
    n_blk: ir.Value,
    n_intra: ir.Value,
    lane_div_16: ir.Value,
    elem_type: ir.Type,
    kpack_bytes: int = 16,
    elem_bytes: int = 1,
    unpack_int4: bool = False,
) -> ir.Value:
    """Load one B pack for one MFMA(x32) micro-step.

    Returns an i64 Value containing 8 bytes consumed by MFMA.
    """
    if kpack_bytes not in (8, 16):
        raise ValueError(f"kpack_bytes must be 8 or 16, got {kpack_bytes!r}")
    if unpack_int4 and kpack_bytes != 8:
        raise ValueError("unpack_int4 requires kpack_bytes=8 (packed int4 layout)")
    if elem_bytes not in (1, 2):
        raise ValueError(f"elem_bytes must be 1 or 2, got {elem_bytes!r}")

    c64 = fx.Index(64)
    base_k_bytes = base_k * arith.constant(int(elem_bytes), index=True)
    k0_base = base_k_bytes // c64
    k0 = k0_base + arith.constant(ki_step // 2, index=True)
    k1 = lane_div_16
    half_bytes = kpack_bytes // 2
    k2_base = arith.constant((ki_step % 2) * half_bytes, index=True)

    coord_pack = (n_blk, k0, k1, n_intra, fx.Index(0))
    idx_pack = crd2idx(coord_pack, layout_b)

    if unpack_int4:
        idx_bytes = idx_pack + k2_base
        b4 = _buffer_load_vec(
            buffer_ops,
            vector,
            b_rsrc,
            idx_bytes,
            elem_type=elem_type,
            vec_elems=4,
            elem_bytes=1,
            offset_in_bytes=True,
        )
        packed32 = vector.extract(
            vector.bitcast(T.vec(1, T.i32), b4),
            static_position=[0],
            dynamic_position=[],
        )
        even, odd = _unpack_int4_to_int8_pair(packed32)
        return _pack_i32_pair_to_i64(even, odd, vector)

    vec_elems = kpack_bytes // int(elem_bytes)
    b16 = _buffer_load_vec(
        buffer_ops,
        vector,
        b_rsrc,
        idx_pack,
        elem_type=elem_type,
        vec_elems=vec_elems,
        elem_bytes=elem_bytes,
        offset_in_bytes=(elem_bytes == 1),
    )

    b_i32x4 = vector.bitcast(T.i32x4, b16)

    half = ki_step % 2
    if half == 0:
        d0 = vector.extract(b_i32x4, static_position=[0], dynamic_position=[])
        d1 = vector.extract(b_i32x4, static_position=[1], dynamic_position=[])
    else:
        d0 = vector.extract(b_i32x4, static_position=[2], dynamic_position=[])
        d1 = vector.extract(b_i32x4, static_position=[3], dynamic_position=[])

    v2 = vector.from_elements(T.vec(2, T.i32), [d0, d1])
    v64 = vector.bitcast(T.vec(1, T.i64), v2)
    return vector.extract(v64, static_position=[0], dynamic_position=[])


def tile_chunk_coord_i32(
    arith,
    *,
    tx_i32_base: ir.Value,
    i: int,
    total_threads: int,
    layout_tile_div4,
    chunk_i32: int = 4,
):
    """Map (thread, chunk_id) -> (row_local, col_local_i32) for X/A loads."""
    if chunk_i32 not in (1, 2, 4):
        raise ValueError(f"chunk_i32 must be one of (1,2,4), got {chunk_i32!r}")
    chunk_off_i32 = arith.constant(i * total_threads * chunk_i32, index=True)
    tile_idx_i32 = tx_i32_base + chunk_off_i32
    # flydsl 0.2.2 widens an `index` idx2crd input to i64, which makes the
    # extracted i32 coordinate lower to an invalid `arith.extsi (i64)->i32`.
    tile_idx_i32 = arith.index_cast(T.i32, tile_idx_i32)
    coord_local = fx.idx2crd(tile_idx_i32, layout_tile_div4)
    row_local = fx.get(coord_local, 0)
    col_local_i32 = fx.get(coord_local, 1)
    return row_local, col_local_i32


def buffer_copy_gmem16_dwordx4(
    buffer_ops,
    vector,
    *,
    elem_type,
    idx_i32: ir.Value,
    rsrc,
    vec_elems: int = 16,
    elem_bytes: int = 1,
):
    """Copy 16 bytes from global memory into regs via buffer-load dwordx4 lowering."""
    if int(vec_elems) <= 0:
        raise ValueError(f"vec_elems must be > 0, got {vec_elems!r}")
    return _buffer_load_vec(
        buffer_ops,
        vector,
        rsrc,
        idx_i32,
        elem_type=elem_type,
        vec_elems=vec_elems,
        elem_bytes=elem_bytes,
        offset_in_bytes=False,
    )


def lds_store_16b_xor16(
    arith,
    vector,
    *,
    lds_memref,
    vec16_ty,
    layout_lds,
    row_local: ir.Value,
    col_local_i32: ir.Value,
    tx_c4: ir.Value,
    k_blocks16: ir.Value,
    lds_base: ir.Value,
    vec_part_i32x4: ir.Value,
    elem_bytes: int = 1,
):
    """Store one 16B chunk into LDS with CK-style XOR16 swizzle on the K dimension."""
    if elem_bytes not in (1, 2):
        raise ValueError(f"elem_bytes must be 1 or 2, got {elem_bytes!r}")
    col_local_bytes = col_local_i32 * tx_c4
    col_swz_bytes = swizzle_xor16(row_local, col_local_bytes, k_blocks16)
    col_swz = col_swz_bytes if elem_bytes == 1 else col_swz_bytes // 2
    coord_store = (row_local, col_swz)
    idx0 = crd2idx(coord_store, layout_lds) + lds_base
    v16 = vector.bitcast(vec16_ty, vec_part_i32x4)
    vector.store(v16, lds_memref, [idx0])


def lds_store_8b_xor16(
    arith,
    vector,
    *,
    lds_memref,
    vec8_ty,
    layout_lds,
    row_local: ir.Value,
    col_local_i32: ir.Value,
    tx_c4: ir.Value,
    k_blocks16: ir.Value,
    lds_base: ir.Value,
    vec_part_i32x2: ir.Value,
    elem_bytes: int = 1,
):
    """Store one 8B chunk into LDS with CK-style XOR16 swizzle on the K dimension."""
    if elem_bytes not in (1, 2):
        raise ValueError(f"elem_bytes must be 1 or 2, got {elem_bytes!r}")
    col_local_bytes = col_local_i32 * tx_c4
    col_swz_bytes = swizzle_xor16(row_local, col_local_bytes, k_blocks16)
    col_swz = col_swz_bytes if elem_bytes == 1 else col_swz_bytes // 2
    coord_store = (row_local, col_swz)
    idx0 = crd2idx(coord_store, layout_lds) + lds_base
    v8 = vector.bitcast(vec8_ty, vec_part_i32x2)
    vector.store(v8, lds_memref, [idx0])


def lds_store_4b_xor16(
    arith,
    vector,
    *,
    lds_memref,
    vec4_ty,
    layout_lds,
    row_local: ir.Value,
    col_local_i32: ir.Value,
    tx_c4: ir.Value,
    k_blocks16: ir.Value,
    lds_base: ir.Value,
    vec_part_i32x1: ir.Value,
    elem_bytes: int = 1,
):
    """Store one 4B chunk into LDS with CK-style XOR16 swizzle on the K dimension."""
    if elem_bytes not in (1, 2):
        raise ValueError(f"elem_bytes must be 1 or 2, got {elem_bytes!r}")
    col_local_bytes = col_local_i32 * tx_c4
    col_swz_bytes = swizzle_xor16(row_local, col_local_bytes, k_blocks16)
    col_swz = col_swz_bytes if elem_bytes == 1 else col_swz_bytes // 2
    coord_store = (row_local, col_swz)
    idx0 = crd2idx(coord_store, layout_lds) + lds_base
    v4 = vector.bitcast(vec4_ty, vec_part_i32x1)
    vector.store(v4, lds_memref, [idx0])


def lds_load_pack_k32(
    arith,
    vector,
    *,
    lds_memref,
    layout_lds,
    k_blocks16: ir.Value,
    curr_row_a_lds: ir.Value,
    col_base: ir.Value,
    half: int,
    lds_base: ir.Value,
    ck_lds128: bool,
    vec16_ty,
    vec8_ty,
    vec2_i64_ty,
    vec1_i64_ty,
):
    """Load one i64 A-pack for an MFMA K32 micro-step from LDS."""
    col_base_swz = swizzle_xor16(curr_row_a_lds, col_base, k_blocks16)
    if ck_lds128:
        coord_a16 = (curr_row_a_lds, col_base_swz)
        idx_a16 = crd2idx(coord_a16, layout_lds) + lds_base
        loaded_a16 = vector.load(vec16_ty, lds_memref, [idx_a16])
        a_vec128 = vector.bitcast(vec2_i64_ty, loaded_a16)
        return vector.extract(a_vec128, static_position=[half], dynamic_position=[])
    else:
        col_swizzled = col_base_swz + (half * 8)
        coord_a = (curr_row_a_lds, col_swizzled)
        idx_a = crd2idx(coord_a, layout_lds) + lds_base
        loaded_a8 = vector.load(vec8_ty, lds_memref, [idx_a])
        a_vec64 = vector.bitcast(vec1_i64_ty, loaded_a8)
        return vector.extract(a_vec64, static_position=[0], dynamic_position=[])


def xcd_remap_bx_by(
    bx,
    by,
    c_m,
    *,
    tile_m: int,
    tile_n: int,
    N: int,
    xcd_swizzle: int,
    num_xcds: int = 8,
):
    """Remap (bx, by) for L2-cache reuse via XCD swizzle.

    No-op when ``xcd_swizzle <= 0``. Otherwise:
      1. Linearize the original (bx, by) grid round-robin across ``num_xcds``
         XCDs so that contiguous workgroup ids stay on the same XCD.
      2. Re-tile that 1-D order with an M-major group of size ``xcd_swizzle``,
         folding the tail group when ``gy`` does not divide evenly.

    Designed to be called inside a ``@flyc.kernel`` immediately after::

        bx = gpu.block_id("x")
        by = gpu.block_id("y")
        bx, by = xcd_remap_bx_by(bx, by, c_m, tile_m=..., tile_n=..., N=...,
                                 xcd_swizzle=xcd_swizzle)

    ``c_m`` is the dynamic ``fx.Index`` for runtime ``M``; ``tile_m``,
    ``tile_n``, ``N`` and ``xcd_swizzle`` are compile-time Python ints.
    """
    if xcd_swizzle <= 0:
        return bx, by

    _c1 = fx.arith.constant(1, index=True)
    _c_tm = fx.arith.constant(tile_m, index=True)
    _gx = fx.arith.constant(N // tile_n, index=True)
    _gy = (c_m + _c_tm - _c1) / _c_tm

    _linear_id = bx * _gx + by
    _num_wgs = _gx * _gy

    _c_xcds = fx.arith.constant(num_xcds, index=True)
    _q = _num_wgs / _c_xcds
    _r = _num_wgs % _c_xcds
    _xcd = _linear_id % _c_xcds
    _in_xcd = _linear_id / _c_xcds
    _xcd_lt_r = fx.arith.cmpi(CmpIPredicate.ult, _xcd, _r)
    _clip = fx.arith.select(_xcd_lt_r, _xcd, _r)
    _wgid = _xcd * _q + _clip + _in_xcd

    _c_wgm = fx.arith.constant(xcd_swizzle, index=True)
    _num_wgid_in_group = _c_wgm * _gx
    _group_id = _wgid / _num_wgid_in_group
    _first_pid_m = _group_id * _c_wgm
    _remaining_m = _gy - _first_pid_m
    _cmp_m = fx.arith.cmpi(CmpIPredicate.ult, _remaining_m, _c_wgm)
    _group_size_m = fx.arith.select(_cmp_m, _remaining_m, _c_wgm)

    _wgid_in_group = _wgid % _num_wgid_in_group
    new_bx = _first_pid_m + (_wgid_in_group % _group_size_m)
    new_by = _wgid_in_group / _group_size_m
    return new_bx, new_by


# ===========================================================================
# Inlined MFMA epilogues (mfma_epilogues)
# ===========================================================================
@contextmanager
def _if_then(if_op, scf):
    """Compat helper for SCF IfOp then-region across old/new Python APIs."""
    with ir.InsertionPoint(if_op.then_block):
        try:
            yield if_op.then_block
        finally:
            blk = if_op.then_block
            if (not blk.operations) or not isinstance(blk.operations[-1], scf.YieldOp):
                scf.YieldOp([])


def default_epilog(
    *,
    arith,
    range_constexpr,
    m_repeat: int,
    lane_div_16,
    bx_m,
    body_row: Callable,
):
    """Iterate the standard MFMA 16x16 row mapping and call `body_row(...)`.

    The mapping matches the common MFMA fragment layout used across kernels in this repo.

    Args:
      arith: flydsl arith ext module.
      range_constexpr: compile-time unrolled range helper.
      m_repeat: tile_m // 16 (python int).
      lane_div_16: index Value (0..3).
      bx_m: base row (index Value). For MoE, this is the base sorted-row for the tile.
      body_row: callback invoked as:
        body_row(mi=<int>, ii=<int>, row_in_tile=<index>, row=<index>)
    """
    bx_m_v = bx_m
    lane_div_16_mul4 = lane_div_16 * 4
    ii_idx_list = [fx.Index(ii) for ii in range(4)]

    for mi in range_constexpr(m_repeat):
        mi_base = arith.constant(mi * 16, index=True)
        for ii in range_constexpr(4):
            row_off = lane_div_16_mul4 + ii_idx_list[ii]
            row_in_tile = mi_base + row_off
            row = bx_m_v + row_in_tile
            body_row(mi=mi, ii=ii, row_in_tile=row_in_tile, row=row)


def c_shuffle_epilog(
    *,
    arith,
    vector,
    gpu,
    scf=None,
    range_constexpr,
    # Tile params
    tile_m: int,
    tile_n: int,
    e_vec: int = 2,
    cshuffle_nlane: int = 32,
    block_size: int = 256,
    m_repeat: int,
    num_acc_n: int,
    # Thread mapping inputs
    tx,
    lane_div_16,
    lane_mod_16,
    bx_m,
    by_n,
    n_tile_base,
    # LDS buffer (f16 view, row-major [tile_m, tile_n] flattened)
    lds_out,
    # Element type for LDS loads (defaults to f16). Pass bf16 to support bf16 epilogues.
    frag_elem_type: ir.Type | None = None,
    # Callbacks
    write_row_to_lds: Callable,
    precompute_row: Callable | None = None,
    store_pair: Callable,
    # When LDS overflows, split lds_out across two buffers by wave-group.
    # Pass the second buffer here; first buffer is `lds_out`.
    lds_out_split=None,
    # Row offset in lds_out for 8-wave mode (MLIR index value).
    # Shifts both write and read LDS indices by lds_row_offset * tile_n elements.
    lds_row_offset=None,
):
    """LDS CShuffle epilogue skeleton.

    Call pattern:
      - `write_row_to_lds(...)` is called once per MFMA row produced by this thread.
        It is responsible for writing all ni columns for that row into `lds_out`.
      - `store_pair(...)` is called for each (row_local, col_pair0) half2 after shuffle.

    `store_pair` can implement either global stores or atomics.
    """
    if int(block_size) <= 0 or (int(block_size) % int(cshuffle_nlane)) != 0:
        raise ValueError(
            f"block_size ({block_size}) must be divisible by cshuffle_nlane ({cshuffle_nlane})"
        )
    cshuffle_mlane = int(block_size) // int(cshuffle_nlane)
    if (int(tile_m) % cshuffle_mlane) != 0:
        raise ValueError(
            f"tile_m must be divisible by CShuffleMLane ({cshuffle_mlane}), got tile_m={tile_m}"
        )
    if int(e_vec) <= 0:
        raise ValueError(f"e_vec must be positive, got {e_vec}")
    if (int(tile_n) % (int(cshuffle_nlane) * int(e_vec))) != 0:
        raise ValueError(
            f"tile_n must be divisible by (CShuffleNLane*EVec) = {cshuffle_nlane*e_vec}, got tile_n={tile_n}"
        )

    # ===================== Split-LDS mode (early return) =====================
    # When lds_out_split is provided, waves are divided into two groups:
    #   Group A (waves 0..N/2-1) uses lds_out,  columns [0, tile_n/2)
    #   Group B (waves N/2..N-1) uses lds_out_split, columns [tile_n/2, tile_n)
    # Each group writes/reads independently; same barriers synchronise all waves.
    if lds_out_split is not None:
        if scf is None:
            raise ValueError("scf module is required for split-LDS cshuffle")

        _half_n = int(tile_n) // 2
        _half_threads = int(block_size) // 2
        EVec = int(e_vec)

        CShuffleNLane_s = min(int(cshuffle_nlane), _half_n // EVec)
        if _half_threads % CShuffleNLane_s != 0:
            raise ValueError(
                f"half_threads={_half_threads} not divisible by CShuffleNLane_split={CShuffleNLane_s}"
            )
        CShuffleMLane_s = _half_threads // CShuffleNLane_s
        if int(tile_m) % CShuffleMLane_s != 0:
            raise ValueError(
                f"tile_m={tile_m} not divisible by CShuffleMLane_split={CShuffleMLane_s}"
            )
        m_reps_s = int(tile_m) // CShuffleMLane_s
        n_reps_s = _half_n // (CShuffleNLane_s * EVec)

        _half_n_idx = arith.constant(_half_n, index=True)
        _half_thr_idx = arith.constant(_half_threads, index=True)
        _zero_idx = arith.constant(0, index=True)

        _is_group_b = arith.cmpi(CmpIPredicate.uge, tx, _half_thr_idx)

        # -- write phase (all waves, each to its group's LDS buffer) --
        n_tile_base_v = n_tile_base
        col_base_local_a = n_tile_base_v + lane_mod_16
        col_base_local_b = col_base_local_a - _half_n_idx

        def _write_row_split(mi: int, ii: int, row_in_tile, row):
            row_base_lds = row_in_tile * _half_n_idx
            _if_g = scf.IfOp(_is_group_b, has_else=True)
            with ir.InsertionPoint(_if_g.then_block):
                write_row_to_lds(
                    mi=mi,
                    ii=ii,
                    row_in_tile=row_in_tile,
                    row=row,
                    row_base_lds=row_base_lds,
                    col_base_local=col_base_local_b,
                    num_acc_n=num_acc_n,
                    lds_out=lds_out_split,
                )
                scf.YieldOp([])
            with ir.InsertionPoint(_if_g.else_block):
                write_row_to_lds(
                    mi=mi,
                    ii=ii,
                    row_in_tile=row_in_tile,
                    row=row,
                    row_base_lds=row_base_lds,
                    col_base_local=col_base_local_a,
                    num_acc_n=num_acc_n,
                    lds_out=lds_out,
                )
                scf.YieldOp([])

        gpu.barrier()
        default_epilog(
            arith=arith,
            range_constexpr=range_constexpr,
            m_repeat=m_repeat,
            lane_div_16=lane_div_16,
            bx_m=bx_m,
            body_row=_write_row_split,
        )
        gpu.barrier()

        # -- read phase (each group reads from its own LDS buffer) --
        tx_local = tx - arith.select(_is_group_b, _half_thr_idx, _zero_idx)
        c_nlane_s = arith.constant(CShuffleNLane_s, index=True)
        m_lane_s = tx_local / c_nlane_s
        n_lane_s = tx_local % c_nlane_s
        c_evec = arith.constant(EVec, index=True)

        if frag_elem_type is None:
            frag_elem_type = T.f16
        vec_frag = T.vec(EVec, frag_elem_type)
        bx_m_v = bx_m
        by_n_v = by_n

        _precomputed_rows_s = []
        for mr in range_constexpr(m_reps_s):
            row_base_m = arith.constant(mr * CShuffleMLane_s, index=True)
            row_local = row_base_m + m_lane_s
            row = bx_m_v + row_local
            row_ctx_raw = (
                precompute_row(row_local=row_local, row=row)
                if precompute_row is not None
                else None
            )
            row_ctx = row_ctx_raw
            row_pred = None
            if (
                scf is not None
                and row_ctx_raw is not None
                and isinstance(row_ctx_raw, tuple)
                and len(row_ctx_raw) == 2
            ):
                row_ctx, row_pred = row_ctx_raw
            _precomputed_rows_s.append((row_local, row, row_ctx, row_pred))

        for mr in range_constexpr(m_reps_s):
            row_local, row, row_ctx, row_pred = _precomputed_rows_s[mr]

            def _do_store_row_split():
                row_base_lds = row_local * _half_n_idx
                for nr in range_constexpr(n_reps_s):
                    col_base_nr = arith.constant(
                        nr * (CShuffleNLane_s * EVec), index=True
                    )
                    col_pair0_local = col_base_nr + (n_lane_s * c_evec)
                    lds_idx = row_base_lds + col_pair0_local

                    _if_ld = scf.IfOp(_is_group_b, [vec_frag], has_else=True)
                    with ir.InsertionPoint(_if_ld.then_block):
                        fb = vector.load(vec_frag, lds_out_split, [lds_idx])
                        scf.YieldOp([fb])
                    with ir.InsertionPoint(_if_ld.else_block):
                        fa = vector.load(vec_frag, lds_out, [lds_idx])
                        scf.YieldOp([fa])
                    frag = _if_ld.results[0]

                    col_pair0 = col_pair0_local + arith.select(
                        _is_group_b, _half_n_idx, _zero_idx
                    )
                    store_pair(
                        row_local=row_local,
                        row=row,
                        row_ctx=row_ctx,
                        col_pair0=col_pair0,
                        col_g0=by_n_v + col_pair0,
                        frag=frag,
                    )

            if row_pred is not None:
                _if_row = scf.IfOp(row_pred)
                with _if_then(_if_row, scf):
                    _do_store_row_split()
            else:
                _do_store_row_split()

        return  # split path complete

    # ===================== Standard (non-split) path below =====================

    # ---------------- Step 1: write C tile to LDS (row-major, fp16) ----------------
    tile_n_idx = arith.constant(int(tile_n), index=True)
    n_tile_base_v = n_tile_base
    col_base_local = n_tile_base_v + lane_mod_16  # index within [0,tile_n)

    _lds_row_base_offset = (
        lds_row_offset * tile_n_idx if lds_row_offset is not None else None
    )

    def _write_row(mi: int, ii: int, row_in_tile, row):
        row_base_lds = row_in_tile * tile_n_idx
        if _lds_row_base_offset is not None:
            row_base_lds = row_base_lds + _lds_row_base_offset
        write_row_to_lds(
            mi=mi,
            ii=ii,
            row_in_tile=row_in_tile,
            row=row,
            row_base_lds=row_base_lds,
            col_base_local=col_base_local,
            num_acc_n=num_acc_n,
            lds_out=lds_out,
        )

    # Ensure all LDS reads finished before the lds write.
    gpu.barrier()
    default_epilog(
        arith=arith,
        range_constexpr=range_constexpr,
        m_repeat=m_repeat,
        lane_div_16=lane_div_16,
        bx_m=bx_m,
        body_row=_write_row,
    )

    # Ensure all LDS writes are visible before the shuffle-read.
    gpu.barrier()

    # ---------------- Step 2: shuffle mapping + half2 store/atomic ----------------
    CShuffleNLane = int(cshuffle_nlane)
    CShuffleMLane = int(cshuffle_mlane)
    EVec = int(e_vec)

    m_reps_shuffle = int(tile_m) // CShuffleMLane
    n_reps_shuffle = int(tile_n) // (CShuffleNLane * EVec)

    c_nlane = fx.Index(CShuffleNLane)
    m_lane = tx // c_nlane
    n_lane = tx % c_nlane
    c_evec = fx.Index(EVec)

    if frag_elem_type is None:
        frag_elem_type = T.f16
    vec_frag = T.vec(EVec, frag_elem_type)
    bx_m_v = bx_m
    by_n_v = by_n

    # Batch-precompute all row contexts (sorted_idx loads) before the store loop.
    # This issues all buffer_load instructions upfront so the compiler can pipeline
    # them instead of serializing each load with s_waitcnt vmcnt(0).
    _precomputed_rows = []
    for mr in range_constexpr(m_reps_shuffle):
        row_base_m = arith.constant(mr * CShuffleMLane, index=True)
        row_local = row_base_m + m_lane
        row = bx_m_v + row_local

        row_ctx_raw = (
            precompute_row(row_local=row_local, row=row)
            if precompute_row is not None
            else None
        )

        # Optional row-level predicate: if `precompute_row` returns `(ctx, pred_i1)` and `scf`
        # is provided, we can skip the entire N-loop for invalid rows (cheaper than per-store checks).
        row_ctx = row_ctx_raw
        row_pred = None
        if (
            scf is not None
            and row_ctx_raw is not None
            and isinstance(row_ctx_raw, tuple)
            and len(row_ctx_raw) == 2
        ):
            row_ctx, row_pred = row_ctx_raw

        _precomputed_rows.append((row_local, row, row_ctx, row_pred))

    # Now perform LDS reads and stores using the pre-fetched row contexts.
    for mr in range_constexpr(m_reps_shuffle):
        row_local, row, row_ctx, row_pred = _precomputed_rows[mr]

        def _do_store_row():
            row_base_lds = row_local * tile_n_idx
            if _lds_row_base_offset is not None:
                row_base_lds = row_base_lds + _lds_row_base_offset
            for nr in range_constexpr(n_reps_shuffle):
                col_base_nr = arith.constant(nr * (CShuffleNLane * EVec), index=True)
                col_pair0 = col_base_nr + (n_lane * c_evec)  # even col within tile

                lds_idx_pair = row_base_lds + col_pair0
                frag = vector.load(vec_frag, lds_out, [lds_idx_pair])

                store_pair(
                    row_local=row_local,
                    row=row,
                    row_ctx=row_ctx,
                    col_pair0=col_pair0,
                    col_g0=by_n_v + col_pair0,
                    frag=frag,
                )

        if row_pred is not None:
            _if_row = scf.IfOp(row_pred)
            with _if_then(_if_row, scf):
                _do_store_row()
        else:
            _do_store_row()


def mfma_epilog(
    *,
    use_cshuffle: bool,
    # Common (always required)
    arith,
    range_constexpr,
    m_repeat: int,
    lane_div_16,
    bx_m,
    # Default epilog (required when use_cshuffle=False)
    body_row: Callable | None = None,
    # CShuffle epilog (required when use_cshuffle=True)
    vector=None,
    gpu=None,
    scf=None,
    tile_m: int | None = None,
    tile_n: int | None = None,
    e_vec: int = 2,
    cshuffle_nlane: int = 32,
    block_size: int = 256,
    num_acc_n: int | None = None,
    tx=None,
    lane_mod_16=None,
    by_n=None,
    n_tile_base=None,
    lds_out=None,
    write_row_to_lds: Callable | None = None,
    precompute_row: Callable | None = None,
    store_pair: Callable | None = None,
    frag_elem_type: ir.Type | None = None,
):
    if not use_cshuffle:
        if body_row is None:
            raise ValueError("mfma_epilog(use_cshuffle=False) requires `body_row`.")
        return default_epilog(
            arith=arith,
            range_constexpr=range_constexpr,
            m_repeat=m_repeat,
            lane_div_16=lane_div_16,
            bx_m=bx_m,
            body_row=body_row,
        )

    return c_shuffle_epilog(
        arith=arith,
        vector=vector,
        gpu=gpu,
        scf=scf,
        range_constexpr=range_constexpr,
        tile_m=int(tile_m),
        tile_n=int(tile_n),
        e_vec=int(e_vec),
        cshuffle_nlane=int(cshuffle_nlane),
        block_size=int(block_size),
        m_repeat=m_repeat,
        num_acc_n=int(num_acc_n),
        tx=tx,
        lane_div_16=lane_div_16,
        lane_mod_16=lane_mod_16,
        bx_m=bx_m,
        by_n=by_n,
        n_tile_base=n_tile_base,
        lds_out=lds_out,
        frag_elem_type=frag_elem_type,
        write_row_to_lds=write_row_to_lds,
        precompute_row=precompute_row,
        store_pair=store_pair,
    )


# ===========================================================================
# Inlined device kernel + compiler (preshuffle_gemm)
# ===========================================================================
_TILE_PRELOAD_TABLE = {
    # (tile_m, tile_n, tile_k): (dsrd_preload, dvmem_preload)
    # ── tile_m = 16 ──
    (16, 64, 256): (2, 2),
    (16, 64, 512): (8, 8),
    (16, 128, 256): (2, 2),
    (16, 128, 512): (2, 2),
    (16, 192, 256): (2, 2),
    (16, 256, 256): (2, 2),
    (16, 256, 512): (2, 2),
    (16, 512, 256): (2, 2),
    # ── tile_m = 32 ──
    (32, 64, 128): (6, 6),
    (32, 64, 256): (6, 6),
    (32, 64, 512): (2, 2),
    (32, 128, 128): (6, 6),
    (32, 128, 256): (6, 6),
    (32, 192, 128): (6, 6),
    (32, 192, 256): (6, 6),
    (32, 256, 128): (6, 6),
    (32, 256, 256): (6, 6),
    # ── tile_m = 48 ──
    (48, 64, 128): (8, 8),
    (48, 64, 256): (2, 2),
    (48, 128, 256): (6, 6),
    (48, 192, 256): (6, 6),
    (48, 256, 256): (6, 6),
    # ── tile_m = 64 ──
    (64, 64, 128): (4, 4),
    (64, 64, 256): (4, 4),
    (64, 128, 128): (8, 8),
    (64, 128, 256): (8, 8),
    (64, 192, 128): (8, 8),
    (64, 192, 256): (8, 8),
    (64, 256, 64): (8, 8),
    (64, 256, 128): (8, 8),
    (64, 256, 256): (8, 8),
    # ── tile_m = 80 ──
    (80, 64, 256): (4, 4),
    (80, 128, 256): (8, 8),
    (80, 192, 256): (8, 8),
    (80, 256, 256): (8, 8),
    # ── tile_m = 96 ──
    (96, 64, 128): (6, 6),
    (96, 64, 256): (6, 6),
    (96, 128, 128): (8, 8),
    (96, 128, 256): (6, 6),
    (96, 192, 128): (8, 8),
    (96, 192, 256): (8, 8),
    (96, 256, 128): (8, 8),
    (96, 256, 256): (8, 8),
    # ── tile_m = 112 ──
    (112, 64, 256): (8, 8),
    (112, 128, 256): (4, 4),
    (112, 192, 256): (8, 8),
    (112, 256, 256): (8, 8),
    # ── tile_m = 128 ──
    (128, 64, 128): (6, 6),
    (128, 64, 256): (8, 8),
    (128, 128, 64): (4, 4),
    (128, 128, 128): (8, 8),
    (128, 128, 256): (4, 4),
    (128, 192, 128): (8, 8),
    (128, 192, 256): (8, 8),
    (128, 256, 128): (6, 6),
    (128, 256, 256): (4, 4),
    # ── tile_m = 160 ──
    (160, 192, 128): (8, 8),
    # ── tile_m = 192 ──
    (192, 64, 128): (6, 6),
    (192, 128, 128): (6, 6),
    # ── tile_m = 224 ──
    (224, 64, 128): (4, 4),
    (224, 128, 128): (6, 6),
    (224, 192, 128): (6, 6),
    # ── tile_m = 256 ──
    (256, 64, 128): (4, 4),
    (256, 128, 128): (6, 6),
    (256, 192, 128): (6, 6),
    (256, 256, 128): (4, 4),
}

_TILE_PRELOAD_DEFAULT = (0, 0)


def _get_preload(tile_m, tile_n, tile_k):
    """Look up (dsrd_preload, dvmem_preload) from the tile table."""
    return _TILE_PRELOAD_TABLE.get(
        (int(tile_m), int(tile_n), int(tile_k)), _TILE_PRELOAD_DEFAULT
    )


@functools.lru_cache(maxsize=1024)
def compile_preshuffle_gemm_a8(
    *,
    M: int = 0,
    N: int = 0,
    K: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    in_dtype: str = "fp8",
    out_dtype: str = "fp16",
    lds_stage: int = 2,
    use_cshuffle_epilog: bool = False,
    waves_per_eu: Optional[int] = None,
    use_async_copy: bool = False,
    dsrd_preload: int = -1,
    dvmem_preload: int = -1,
    epilogue: str = "none",  # "none", "bias", "bias_relu", "bias_silu", "bias_gelu"
    xcd_swizzle: int = 0,
):
    """Compile the preshuffle GEMM kernel using the @flyc.kernel API.

    Returns a JitFunction that auto-compiles and executes when called.
    Signature:  launch_fn(arg_c, arg_a, arg_b, arg_bias, arg_scale_a, arg_scale_b, M, N, stream)

    Compile-time constants: K, tile_m/n/k, in_dtype, out_dtype (determine loop structure).
    Runtime parameters: M, N (passed as i32 kernel args).

    Args:
        out_dtype: Output element type, "fp16" or "bf16" (default: "fp16").
        waves_per_eu: Occupancy hint (None = default, 1-4 = limit occupancy).
        use_async_copy: Use async DMA for A tile global-to-LDS transfer.
        dsrd_preload: Initial LDS-read preload count (-1 = auto from _TILE_PRELOAD_TABLE).
        dvmem_preload: Initial global-load preload count (-1 = auto from _TILE_PRELOAD_TABLE).
    """
    if dsrd_preload < 0 or dvmem_preload < 0:
        if in_dtype in ("fp8", "int8") and str(get_hip_arch()) == "gfx950":
            computed_dsrd, computed_dvmem = _get_preload(tile_m, tile_n, tile_k)
        else:
            computed_dsrd, computed_dvmem = _TILE_PRELOAD_DEFAULT
        if dsrd_preload < 0:
            dsrd_preload = computed_dsrd
        if dvmem_preload < 0:
            dvmem_preload = computed_dvmem
    if in_dtype not in ("fp8", "int8", "int4", "fp16", "bf16", "fp4"):
        raise ValueError(
            "in_dtype must be one of ('fp8','int8','int4','fp16','bf16','fp4'), "
            f"got {in_dtype!r}"
        )
    if out_dtype not in ("fp16", "bf16"):
        raise ValueError(f"out_dtype must be 'fp16' or 'bf16', got {out_dtype!r}")
    _out_is_bf16 = out_dtype == "bf16"
    is_fp4 = in_dtype == "fp4"
    is_int4 = in_dtype == "int4"
    is_int8 = (in_dtype == "int8") or is_int4
    is_f16 = in_dtype == "fp16"
    is_bf16 = in_dtype == "bf16"
    is_f16_or_bf16 = is_f16 or is_bf16
    elem_bytes = 1 if (in_dtype in ("fp8", "int8", "int4", "fp4")) else 2
    a_elem_vec_pack = 2 if is_fp4 else 1
    b_elem_vec_pack = 2 if is_fp4 else 1

    KERNEL_NAME = (
        f"preshuffle_gemm_{in_dtype}_{out_dtype}"
        f"_t{tile_m}x{tile_n}x{tile_k}"
        f"_lds{lds_stage}"
        f"_pl{dsrd_preload}x{dvmem_preload}"
    )
    if use_cshuffle_epilog:
        KERNEL_NAME += "_csh"
    if use_async_copy:
        KERNEL_NAME += "_async"
    if waves_per_eu is not None:
        KERNEL_NAME += f"_wpe{waves_per_eu}"
    if epilogue != "none":
        KERNEL_NAME += f"_ep_{epilogue}"
    if xcd_swizzle > 0:
        KERNEL_NAME += f"_xcd{xcd_swizzle}"

    tile_k_bytes = int(tile_k) * int(elem_bytes)

    if (tile_k_bytes % 64) != 0:
        raise ValueError(
            f"tile_k_bytes must be divisible by 64, got tile_k_bytes={tile_k_bytes} "
            f"(tile_k={tile_k}, elem_bytes={elem_bytes})"
        )

    _min_k_unroll = tile_k_bytes // a_elem_vec_pack // 64
    if is_fp4 and _min_k_unroll < 2 and int(tile_k) != 128:
        raise ValueError(
            f"FP4 requires tile_k=128 or tile_k >= {64 * 2 * a_elem_vec_pack} "
            f"(mfma_scale_f32_16x16x128 needs k_unroll >= 1), "
            f"got tile_k={tile_k} (k_unroll={_min_k_unroll})"
        )
    if is_fp4 and int(tile_k) == 128 and lds_stage != 2:
        raise NotImplementedError("FP4 tile_k=128 currently only supports lds_stage=2")

    mfma_i32_k32 = None
    if is_int8:
        mfma_i32_k32 = getattr(rocdl, "mfma_i32_16x16x32i8", None) or getattr(
            rocdl, "mfma_i32_16x16x32_i8", None
        )
        if mfma_i32_k32 is None:
            raise AttributeError(
                "INT8 K32 MFMA op not found: expected `rocdl.mfma_i32_16x16x32i8` "
                "(or `rocdl.mfma_i32_16x16x32_i8`)."
            )

    gpu_arch = get_hip_arch()

    allocator_pong = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem0")
    allocator_ping = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem1")

    total_threads = 256
    bytes_a_per_tile = int(tile_m) * int(tile_k) * int(elem_bytes) // a_elem_vec_pack
    if bytes_a_per_tile % total_threads != 0:
        raise ValueError(
            "tile_m*tile_k*elem_bytes/a_elem_vec_pack must be divisible by "
            f"{total_threads}: tile_m={tile_m}, tile_k={tile_k}, elem_bytes={elem_bytes}, pack={a_elem_vec_pack}"
        )
    bytes_per_thread_a = bytes_a_per_tile // total_threads

    a_load_bytes = 16
    if bytes_per_thread_a % a_load_bytes != 0:
        raise ValueError(
            f"bytes_per_thread_a ({bytes_per_thread_a}) must be divisible by {a_load_bytes}"
        )
    a_async_load_bytes = 4 if gpu_arch == "gfx942" else 16
    a_async_load_dword = a_async_load_bytes // 4

    bytes_b_per_tile = int(tile_n) * int(tile_k) * int(elem_bytes) // b_elem_vec_pack
    bytes_per_thread_b = bytes_b_per_tile // total_threads
    b_load_bytes = 16
    num_b_loads = bytes_per_thread_b // b_load_bytes

    wave_size = 64
    num_a_lds_load = bytes_a_per_tile // wave_size // a_load_bytes

    _is_gfx950 = str(gpu_arch).startswith("gfx950")
    _is_gfx942 = str(gpu_arch).startswith("gfx942")
    use_mfma_k32 = _is_gfx950 and is_f16_or_bf16

    lds_stride_bytes = tile_k_bytes

    Vec = fx.Vector

    def _fp8_dtype():
        return (
            fx.Float8E4M3FN
            if (_is_gfx950 or str(gpu_arch).startswith("gfx12"))
            else fx.Float8E4M3FNUZ
        )

    def _elem_dtype():
        if is_f16:
            return fx.Float16
        if is_bf16:
            return fx.BFloat16
        if is_fp4:
            return fx.Int8
        return fx.Int8 if is_int8 else _fp8_dtype()

    def _elem_type():
        return _elem_dtype().ir_type

    def _vec16_type():
        if is_f16:
            return Vec.make_type(8, fx.Float16)
        if is_bf16:
            return Vec.make_type(8, fx.BFloat16)
        if is_fp4:
            return Vec.make_type(16, fx.Int8)
        return Vec.make_type(16, fx.Int8 if is_int8 else _fp8_dtype())

    def _mfma_pack_ty():
        if is_f16:
            return Vec.make_type(4, fx.Float16)
        if is_bf16:
            return Vec.make_type(4, fx.Int16)
        return fx.Int64.ir_type

    def _out_dtype():
        return fx.BFloat16 if _out_is_bf16 else fx.Float16

    def _out_elem():
        return _out_dtype().ir_type

    # ── LDS sizing (pure Python, no MLIR ops) ────────────────────────────────
    lds_tile_bytes = int(tile_m) * int(lds_stride_bytes) // a_elem_vec_pack
    lds_out_bytes = 2 * int(tile_m) * int(tile_n) if use_cshuffle_epilog else 0

    lds_pong_offset = 0
    lds_ping_offset = 0
    lds_alloc_offset = 0
    if int(lds_stage) == 2:
        assert lds_out_bytes % 2 == 0, "lds_out_bytes should be multiple of 2"
        buffer_size_bytes = max(lds_tile_bytes, lds_out_bytes // lds_stage)
        buffer_size_elems = (
            buffer_size_bytes if elem_bytes == 1 else (buffer_size_bytes // 2)
        )

        lds_pong_offset = allocator_pong._align(allocator_pong.ptr, 16)
        allocator_pong.ptr = lds_pong_offset + buffer_size_elems * elem_bytes

        lds_ping_offset = allocator_ping._align(allocator_ping.ptr, 16)
        allocator_ping.ptr = lds_ping_offset + buffer_size_elems * elem_bytes
    else:
        lds_total_bytes = max(lds_tile_bytes, lds_out_bytes)
        lds_total_elems = lds_total_bytes if elem_bytes == 1 else (lds_total_bytes // 2)

        lds_alloc_offset = allocator_pong._align(allocator_pong.ptr, 16)
        allocator_pong.ptr = lds_alloc_offset + lds_total_elems * elem_bytes

    # ── Kernel function ────────────────────────────────────────────────────
    _has_epilogue = epilogue != "none"
    _has_bias = epilogue in ("bias", "bias_relu", "bias_silu", "bias_gelu")
    _has_relu = epilogue == "bias_relu"
    _has_silu = epilogue == "bias_silu"
    _has_gelu = epilogue == "bias_gelu"

    # Fused epilogue is implemented inside body_row (the direct store path).
    # When use_cshuffle_epilog=True, the epilogue path goes through
    # write_row_to_lds -> store_pair and returns before body_row, which would
    # silently drop the bias + activation. Reject the unsupported combination.
    if _has_epilogue and use_cshuffle_epilog:
        raise ValueError(
            "Fused epilogue (epilogue != 'none') is not supported with "
            "use_cshuffle_epilog=True; the cshuffle path bypasses body_row "
            "where the bias/activation fusion lives."
        )

    @flyc.kernel
    def kernel_gemm(
        arg_c: fx.Pointer,
        arg_a: fx.Pointer,
        arg_b: fx.Pointer,
        arg_scale_a: fx.Pointer,
        arg_scale_b: fx.Pointer,
        arg_bias: fx.Pointer,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
    ):
        c_m = fx.Index(i32_m)
        c_n = fx.Index(i32_n)

        # ---- Types ----
        acc_init = (
            Vec.filled(4, 0, fx.Int32) if is_int8 else Vec.filled(4, 0.0, fx.Float32)
        )

        # ---- Layouts ----

        _k_div4_factor = (K * elem_bytes) // 4 // a_elem_vec_pack

        kpack_bytes = 8 if is_int4 else 16
        kpack_elems = kpack_bytes if elem_bytes == 1 else kpack_bytes // elem_bytes
        k_bytes_b = K * elem_bytes // b_elem_vec_pack
        n0_val = N // 16
        k0_val = k_bytes_b // 64
        _stride_nlane = kpack_elems
        _stride_klane = 16 * _stride_nlane
        _stride_k0 = 4 * _stride_klane
        _stride_n0 = k0_val * _stride_k0
        layout_b = fx.make_layout(
            (n0_val, k0_val, 4, 16, kpack_elems),
            (_stride_n0, _stride_k0, _stride_klane, _stride_nlane, 1),
        )

        lds_k_dim = tile_k // a_elem_vec_pack
        k_blocks16 = fx.Index(tile_k_bytes // a_elem_vec_pack // 16)

        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        by = gpu.block_id("y")

        bx, by = xcd_remap_bx_by(
            bx,
            by,
            c_m,
            tile_m=tile_m,
            tile_n=tile_n,
            N=N,
            xcd_swizzle=xcd_swizzle,
        )

        # ---- LDS (separate ping/pong buffers for no-alias guarantee) ----
        base_ptr_pong = allocator_pong.get_base()
        base_ptr_ping = allocator_ping.get_base()

        lds_a_pong_ptr = SmemPtr(
            base_ptr_pong, lds_alloc_offset, _elem_type(), shape=(1,)
        )
        lds_a_ping_ptr = lds_a_pong_ptr
        lds_out_ptr = SmemPtr(base_ptr_pong, lds_alloc_offset, _out_elem(), shape=(1,))

        if const_expr(lds_stage == 2):
            lds_a_pong_ptr = SmemPtr(
                base_ptr_pong, lds_pong_offset, _elem_type(), shape=(tile_m * tile_k,)
            )
            lds_a_ping_ptr = SmemPtr(
                base_ptr_ping, lds_ping_offset, _elem_type(), shape=(tile_m * tile_k,)
            )

            if const_expr(use_cshuffle_epilog):
                lds_out_ptr = SmemPtr(
                    base_ptr_pong,
                    lds_pong_offset,
                    _out_elem(),
                    shape=(tile_m * tile_n,),
                )
            else:
                lds_out_ptr = SmemPtr(
                    base_ptr_pong, lds_pong_offset, _out_elem(), shape=(1,)
                )
        else:
            lds_a_pong_ptr = SmemPtr(
                base_ptr_pong, lds_alloc_offset, _elem_type(), shape=(lds_total_elems,)
            )
            lds_a_ping_ptr = lds_a_pong_ptr
            if const_expr(use_cshuffle_epilog):
                lds_out_ptr = SmemPtr(
                    base_ptr_pong,
                    lds_alloc_offset,
                    _out_elem(),
                    shape=(tile_m * tile_n,),
                )
            else:
                lds_out_ptr = SmemPtr(
                    base_ptr_pong, lds_alloc_offset, _out_elem(), shape=(1,)
                )

        lds_a_pong = lds_a_pong_ptr.get()
        lds_a_ping = lds_a_ping_ptr.get()
        lds_out = lds_out_ptr.get()

        # ---- Buffer resources (runtime byte sizes for OOB protection) ----
        _a_nrec = fx.Int64(c_m * (K * elem_bytes // a_elem_vec_pack))
        _c_nrec = fx.Int64(c_m * c_n * 2)

        def _ptr_buffer_resource(ptr, num_records_bytes=None):
            addr = fx.ptrtoint(ptr)
            addr_i64 = fx.arith.index_cast(T.i64, addr)
            if num_records_bytes is None:
                return buffer_ops.create_buffer_resource_from_addr(addr_i64)
            return buffer_ops.create_buffer_resource_from_addr(
                addr_i64, num_records_bytes=num_records_bytes
            )

        a_rsrc = _ptr_buffer_resource(arg_a, _a_nrec)
        c_rsrc = _ptr_buffer_resource(arg_c, _c_nrec)
        _needs_per_token_scale = not is_f16_or_bf16 and not is_fp4
        scale_a_rsrc = None
        if const_expr(not is_f16_or_bf16):
            if const_expr(is_fp4):
                _scale_a_rows = (c_m + fx.Index(31)) // fx.Index(32)
                _scale_a_stride_elems = fx.Index((K // (32 * 4 * 2)) * 64)
                _scale_a_nrec = fx.Int64(
                    _scale_a_rows * _scale_a_stride_elems * fx.Index(4)
                )
            else:
                _scale_a_nrec = fx.Int64(c_m * fx.Index(4))
            scale_a_rsrc = _ptr_buffer_resource(arg_scale_a, _scale_a_nrec)

        # ---- Bias buffer resource (for fused epilogue) ----
        # Use max_size=True so the buffer descriptor's size is taken from the
        # actual arg_bias tensor; this avoids hardcoding the output element
        # size (was c_n * 2, which broke if out_dtype became fp32 etc.).
        bias_rsrc = None
        if const_expr(_has_bias):
            bias_rsrc = _ptr_buffer_resource(arg_bias)
        b_rsrc = _ptr_buffer_resource(arg_b)
        scale_b_rsrc = None if (is_f16_or_bf16) else _ptr_buffer_resource(arg_scale_b)

        bx_m = bx * tile_m
        by_n = by * tile_n

        # ---- Wave / lane decomposition ----
        wave_size = 64
        layout_wave_lane = fx.make_layout((4, wave_size), (64, 1))
        # flydsl 0.2.2 casts an `index` idx2crd input to i64, so extracting a
        # scalar coordinate (get_scalar -> i32) lowers to an invalid
        # `arith.extsi (i64)->i32`. Feed idx2crd an i32 thread id (the same
        # convention as tile_chunk_coord_i32) so coordinates stay i32.
        tx_i32 = fx.arith.index_cast(T.i32, tx)
        coord_wave_lane = fx.idx2crd(tx_i32, layout_wave_lane)
        wave_id = fx.get(coord_wave_lane, 0)
        lane_id = fx.get(coord_wave_lane, 1)

        layout_lane16 = fx.make_layout((4, 16), (16, 1))
        # Keep the idx2crd input i32 (see the tx_i32 note above); fx.get returns
        # an `index` that flydsl 0.2.2 would otherwise widen to i64.
        lane_id_i32 = fx.arith.index_cast(T.i32, lane_id)
        coord_lane16 = fx.idx2crd(lane_id_i32, layout_lane16)
        lane_div_16 = fx.get(coord_lane16, 0)
        lane_mod_16 = fx.get(coord_lane16, 1)

        row_a_lds = lane_mod_16
        kpack_elems = 16 if elem_bytes == 1 else 8
        col_offset_base = lane_div_16 * kpack_elems
        col_offset_base_bytes = (
            col_offset_base if elem_bytes == 1 else col_offset_base * elem_bytes
        )

        m_repeat = tile_m // 16
        k_unroll = tile_k_bytes // a_elem_vec_pack // 64

        num_waves = 4
        n_per_wave = tile_n // num_waves
        num_acc_n = n_per_wave // 16

        n_tile_base = wave_id * n_per_wave

        n_intra_list = []
        n_blk_list = []
        for i in range_constexpr(num_acc_n):
            global_n = by_n + n_tile_base + (i * 16) + lane_mod_16
            n_blk_list.append(global_n // 16)
            n_intra_list.append(global_n % 16)

        # ── B load helpers ────────────────────────────────────────────────
        def load_b_pack(base_k, ki_step, ni):
            return load_b_pack_k32(
                buffer_ops,
                fx.arith,
                vector,
                arg_b=arg_b,
                b_rsrc=b_rsrc,
                layout_b=layout_b,
                base_k=base_k,
                ki_step=ki_step,
                n_blk=n_blk_list[ni],
                n_intra=n_intra_list[ni],
                lane_div_16=lane_div_16,
                elem_type=_elem_type(),
                kpack_bytes=kpack_bytes,
                elem_bytes=elem_bytes,
                unpack_int4=is_int4,
            )

        c64_b = 64

        _b_stride_n0_c = fx.Index(_stride_n0)
        _b_stride_k0_c = fx.Index(_stride_k0)
        _b_stride_klane_c = fx.Index(_stride_klane)
        _b_stride_nlane_c = fx.Index(_stride_nlane)

        _b_dword_stride_n0 = _stride_n0 // 4
        _b_dword_stride_k0 = _stride_k0 // 4
        _b_dword_stride_klane = _stride_klane // 4
        _b_dword_stride_nlane = _stride_nlane // 4

        _b_n_full_dword_list = []
        for _ni in range_constexpr(num_acc_n):
            _n_dword = (
                n_blk_list[_ni] * fx.Index(_b_dword_stride_n0)
                + n_intra_list[_ni] * fx.Index(_b_dword_stride_nlane)
                + lane_div_16 * fx.Index(_b_dword_stride_klane)
            )
            _b_n_full_dword_list.append(_n_dword)

        _b_dword_stride_k0_c = fx.Index(_b_dword_stride_k0)
        _c64_elem = fx.Index(64 // elem_bytes * b_elem_vec_pack)

        def _extract_b_packs(b16):
            b_i64x2 = Vec(b16).bitcast(fx.Int64)
            b0_i64 = b_i64x2[0]
            b1_i64 = b_i64x2[1]
            if const_expr(not is_f16_or_bf16 or use_mfma_k32):
                return b0_i64.ir_value(), b1_i64.ir_value()
            b0_v1 = Vec.from_elements([b0_i64], fx.Int64)
            b1_v1 = Vec.from_elements([b1_i64], fx.Int64)
            if const_expr(is_f16):
                return b0_v1.bitcast(fx.Float16), b1_v1.bitcast(fx.Float16)
            return b0_v1.bitcast(fx.Int16), b1_v1.bitcast(fx.Int16)

        def _load_b_single(k_dword_offset, ni):
            """Load one 16B B vector using pre-computed k dword offset."""
            dword_idx = _b_n_full_dword_list[ni] + k_dword_offset
            dword_idx_i32 = fx.Int32(dword_idx)
            b_vec4 = buffer_ops.buffer_load(
                b_rsrc, dword_idx_i32, vec_width=4, dtype=fx.Int32
            )
            b16 = Vec(b_vec4).bitcast(_elem_dtype())
            return _extract_b_packs(b16)

        def load_b_packs_k64(base_k, ku: int, ni: int):
            if const_expr(is_int4):
                ki0 = (ku * 2) + 0
                ki1 = (ku * 2) + 1
                return load_b_pack(base_k, ki0, ni), load_b_pack(base_k, ki1, ni)

            base_k_bytes = base_k * elem_bytes
            k0 = base_k_bytes // c64_b + ku
            idx_pack = (
                n_blk_list[ni] * _b_stride_n0_c
                + k0 * _b_stride_k0_c
                + lane_div_16 * _b_stride_klane_c
                + n_intra_list[ni] * _b_stride_nlane_c
            )
            vec_elems = 16 if elem_bytes == 1 else 8
            b16 = _buffer_load_vec(
                buffer_ops,
                vector,
                b_rsrc,
                idx_pack,
                elem_type=_elem_type(),
                vec_elems=vec_elems,
                elem_bytes=elem_bytes,
                offset_in_bytes=(elem_bytes == 1),
            )
            return _extract_b_packs(b16)

        def load_b_tile(base_k):
            if const_expr((not is_int4) and (not is_f16_or_bf16)):
                base_k_bytes = base_k * elem_bytes
                k0_base = base_k_bytes // c64_b
                k_dwords = []
                for ku in range_constexpr(k_unroll):
                    k_dwords.append((k0_base + ku) * _b_dword_stride_k0_c)
                packs0_per_ku = [[] for _ in range(k_unroll)]
                packs1_per_ku = [[] for _ in range(k_unroll)]
                for ni in range_constexpr(num_acc_n):
                    for ku in range_constexpr(k_unroll):
                        b0, b1 = _load_b_single(k_dwords[ku], ni)
                        packs0_per_ku[ku].append(b0)
                        packs1_per_ku[ku].append(b1)
                b_tile = []
                for ku in range_constexpr(k_unroll):
                    b_tile.append((packs0_per_ku[ku], packs1_per_ku[ku]))
                return b_tile

            packs0_per_ku = [[] for _ in range(k_unroll)]
            packs1_per_ku = [[] for _ in range(k_unroll)]
            for ni in range_constexpr(num_acc_n):
                for ku in range_constexpr(k_unroll):
                    b0, b1 = load_b_packs_k64(base_k, ku, ni)
                    packs0_per_ku[ku].append(b0)
                    packs1_per_ku[ku].append(b1)
            b_tile = []
            for ku in range_constexpr(k_unroll):
                b_tile.append((packs0_per_ku[ku], packs1_per_ku[ku]))
            return b_tile

        # ── A LDS load/store helpers (now take lds_buffer memref directly) ──
        lds_base_zero = fx.Index(0)

        _lds_k_dim_c = fx.Index(lds_k_dim)

        def lds_load_16b(curr_row_a_lds, col_base, lds_buffer):
            col_base_swz_bytes = swizzle_xor16(curr_row_a_lds, col_base, k_blocks16)
            col_base_swz = (
                col_base_swz_bytes if elem_bytes == 1 else (col_base_swz_bytes // 2)
            )
            idx_a16 = curr_row_a_lds * _lds_k_dim_c + col_base_swz
            return Vec.load(_vec16_type(), lds_buffer, [idx_a16])

        def lds_load_packs_k64(curr_row_a_lds, col_base, lds_buffer):
            loaded_a16 = lds_load_16b(curr_row_a_lds, col_base, lds_buffer)
            a_i64x2 = Vec(loaded_a16).bitcast(fx.Int64)
            a0_i64 = a_i64x2[0]
            a1_i64 = a_i64x2[1]

            if const_expr(not is_f16_or_bf16 or use_mfma_k32):
                return a0_i64.ir_value(), a1_i64.ir_value()

            a0_v1 = Vec.from_elements([a0_i64], fx.Int64)
            a1_v1 = Vec.from_elements([a1_i64], fx.Int64)
            if const_expr(is_f16):
                return a0_v1.bitcast(fx.Float16), a1_v1.bitcast(fx.Float16)
            return a0_v1.bitcast(fx.Int16), a1_v1.bitcast(fx.Int16)

        # ── A global→reg load ─────────────────────────────────────────────
        num_a_loads = bytes_per_thread_a // a_load_bytes
        tile_k_dwords = (
            (tile_k * 2) // 4 if elem_bytes == 2 else tile_k // 4 // a_elem_vec_pack
        )
        layout_a_tile_div4 = fx.make_layout((tile_m, tile_k_dwords), (tile_k_dwords, 1))
        c4 = fx.Index(4)
        tx_i32_base = tx * c4

        def load_a_16(idx_elem):
            return buffer_copy_gmem16_dwordx4(
                buffer_ops,
                vector,
                elem_type=_elem_type(),
                idx_i32=idx_elem,
                rsrc=a_rsrc,
                vec_elems=(16 if elem_bytes == 1 else 8),
                elem_bytes=elem_bytes,
            )

        def a_tile_chunk_coord_i32(i: int):
            return tile_chunk_coord_i32(
                fx.arith,
                tx_i32_base=tx_i32_base,
                i=i,
                total_threads=total_threads,
                layout_tile_div4=layout_a_tile_div4,
            )

        def load_a_tile(base_k_div4):
            parts = []
            for i in range_constexpr(num_a_loads):
                row_a_local, col_a_local_i32 = a_tile_chunk_coord_i32(i)
                row_a_global = bx_m + row_a_local
                idx_i32 = row_a_global * _k_div4_factor + (
                    base_k_div4 + col_a_local_i32
                )
                idx_elem = idx_i32 if elem_bytes == 1 else idx_i32 * 2
                a_16B = load_a_16(idx_elem)
                parts.append(Vec(a_16B).bitcast(fx.Int32))
            return parts

        def store_a_tile_to_lds(vec_a_parts, lds_buffer):
            for i in range_constexpr(num_a_loads):
                row_a_local, col_a_local_i32 = a_tile_chunk_coord_i32(i)
                col_local_bytes = col_a_local_i32 * c4
                col_swz_bytes = swizzle_xor16(row_a_local, col_local_bytes, k_blocks16)
                col_swz = col_swz_bytes if elem_bytes == 1 else col_swz_bytes // 2
                idx0 = row_a_local * _lds_k_dim_c + col_swz + lds_base_zero
                v16 = Vec(vec_a_parts[i]).bitcast(_elem_dtype())
                v16.store(lds_buffer, [idx0])

        # ── A DMA async: direct global→LDS transfer ─────────────────────
        num_a_async_loads = bytes_per_thread_a // a_async_load_bytes
        tx_i32_async_base = tx * a_async_load_dword
        k_bytes_factor = K * elem_bytes // a_elem_vec_pack

        def a_tile_chunk_coord_i32_async(i: int):
            return tile_chunk_coord_i32(
                fx.arith,
                tx_i32_base=tx_i32_async_base,
                i=i,
                total_threads=total_threads,
                layout_tile_div4=layout_a_tile_div4,
                chunk_i32=a_async_load_dword,
            )

        def dma_a_tile_to_lds(
            base_k_div4,
            lds_buffer,
            *,
            wave_id_v,
            wave_size_v,
            dma_bytes_v,
            num_a_async_loads_v,
            a_tile_chunk_coord_i32_async_fn,
            c4_v,
            k_blocks16_v,
            bx_m_v,
            k_bytes_factor_v,
            total_threads_v,
            a_rsrc_v,
        ):
            from flydsl._mlir.dialects import memref as memref_dialect

            wave_offset = rocdl.readfirstlane(
                fx.Int64.ir_type,
                fx.Int64(wave_id_v * fx.Index(wave_size_v * dma_bytes_v)),
            )
            lds_base = memref_dialect.extract_aligned_pointer_as_index(lds_buffer)
            lds_ptr_base = buffer_ops.create_llvm_ptr(
                fx.Int64(lds_base), address_space=3
            )
            lds_ptr = buffer_ops.get_element_ptr(lds_ptr_base, wave_offset)

            for i in range_constexpr(num_a_async_loads_v):
                row_a_local, col_a_local_i32 = a_tile_chunk_coord_i32_async_fn(i)
                col_a_local_sw = swizzle_xor16(
                    row_a_local, col_a_local_i32 * c4_v, k_blocks16_v
                )
                row_a_global = bx_m_v + row_a_local
                global_byte_idx = row_a_global * k_bytes_factor_v + (
                    base_k_div4 * c4_v + col_a_local_sw
                )
                global_offset = fx.Int32(global_byte_idx)

                if const_expr(i > 0):
                    lds_ptr = buffer_ops.get_element_ptr(
                        lds_ptr,
                        static_byte_offset=total_threads_v * dma_bytes_v,
                    )

                size_i32 = fx.Int32(dma_bytes_v)
                soffset = fx.Int32(0)
                offset_imm = fx.Int32(0)
                aux = fx.Int32(1)

                rocdl.raw_ptr_buffer_load_lds(
                    a_rsrc_v,
                    lds_ptr,
                    size_i32,
                    global_offset,
                    soffset,
                    offset_imm,
                    aux,
                )

        def prefetch_a_to_lds(
            base_k, lds_buffer, *, a_elem_vec_pack_v, dma_a_tile_to_lds_fn
        ):
            base_k_div4 = base_k // 4 // a_elem_vec_pack_v
            dma_a_tile_to_lds_fn(
                base_k_div4,
                lds_buffer,
                wave_id_v=wave_id,
                wave_size_v=wave_size,
                dma_bytes_v=a_async_load_bytes,
                num_a_async_loads_v=num_a_async_loads,
                a_tile_chunk_coord_i32_async_fn=a_tile_chunk_coord_i32_async,
                c4_v=c4,
                k_blocks16_v=k_blocks16,
                bx_m_v=bx_m,
                k_bytes_factor_v=k_bytes_factor,
                total_threads_v=total_threads,
                a_rsrc_v=a_rsrc,
            )

        def prefetch_a_tile(base_k):
            base_k_bytes = base_k * elem_bytes // a_elem_vec_pack
            base_k_div4 = base_k_bytes // 4
            return load_a_tile(base_k_div4)

        def prefetch_b_tile(base_k):
            base_k_packed = base_k // b_elem_vec_pack if b_elem_vec_pack > 1 else base_k
            return load_b_tile(base_k_packed)

        def prefetch_ab_tile(base_k):
            a_regs = prefetch_a_tile(base_k)
            b_regs = prefetch_b_tile(base_k)
            return a_regs, b_regs

        # ── FP4 scale pre-fetch (outside compute_tile for latency hiding) ──
        _fp4_tilek128 = False

        def load_fp4_scale_chunk(_base_k):
            raise RuntimeError("load_fp4_scale_chunk called when is_fp4=False")

        if const_expr(is_fp4):
            _fp4_pack_M_outer = 2
            _fp4_pack_N_outer = 2
            _fp4_pack_K_outer = 2
            _fp4_tilek128 = int(tile_k) == 128
            _fp4_scale_chunk_k = 32 * 4 * _fp4_pack_K_outer
            _K1_outer = K // (32 * 4 * _fp4_pack_K_outer)
            _k_unroll_packed_outer = (
                1 if _fp4_tilek128 else (k_unroll // _fp4_pack_K_outer)
            )
            _m_repeat_packed_outer = m_repeat // _fp4_pack_M_outer
            _num_acc_n_packed_outer = num_acc_n // _fp4_pack_N_outer
            _fp4_scale_k_stride = tile_k // (32 * 4 * _fp4_pack_K_outer)
            _fp4_use_scheduler = tile_m >= 64

            _scale_lane_elem_off = lane_div_16 * fx.Index(16) + lane_mod_16
            _scale_row_stride_elems = _K1_outer * 64

            _scale_a_base_elems = []
            for mi in range_constexpr(_m_repeat_packed_outer):
                mni_a = fx.Index(mi) + bx_m // fx.Index(_fp4_pack_M_outer * 16)
                _scale_a_base_elems.append(
                    mni_a * fx.Index(_scale_row_stride_elems) + _scale_lane_elem_off
                )

            _scale_b_base_elems = []
            for ni in range_constexpr(_num_acc_n_packed_outer):
                mni_b = fx.Index(ni) + (by_n + n_tile_base) // fx.Index(
                    _fp4_pack_N_outer * 16
                )
                _scale_b_base_elems.append(
                    mni_b * fx.Index(_scale_row_stride_elems) + _scale_lane_elem_off
                )

            _stride_k0_elems = 64

            def load_fp4_scales(base_k_scale_idx):
                a_scales, b_scales = [], []
                base_k_elem_off = base_k_scale_idx * fx.Index(_stride_k0_elems)
                for ku in range_constexpr(_k_unroll_packed_outer):
                    ku_elem_off = base_k_elem_off + fx.Index(ku * _stride_k0_elems)
                    for ni in range_constexpr(_num_acc_n_packed_outer):
                        b_scales.append(
                            buffer_ops.buffer_load(
                                scale_b_rsrc,
                                _scale_b_base_elems[ni] + ku_elem_off,
                                vec_width=1,
                                dtype=fx.Int32,
                            )
                        )
                    for mi in range_constexpr(_m_repeat_packed_outer):
                        a_scales.append(
                            buffer_ops.buffer_load(
                                scale_a_rsrc,
                                _scale_a_base_elems[mi] + ku_elem_off,
                                vec_width=1,
                                dtype=fx.Int32,
                            )
                        )
                return a_scales, b_scales

            def load_fp4_scale_chunk(base_k):
                return load_fp4_scales(base_k // fx.Index(_fp4_scale_chunk_k))

        # ── Compute tile (MFMA) ───────────────────────────────────────────
        def compute_tile(
            accs_in,
            b_tile_in,
            lds_buffer,
            *,
            is_last_tile=False,
            a0_prefetch=None,
            fp4_scales=None,
            fp4_scale_half=0,
        ):
            scales_pf = {}
            if const_expr(is_last_tile and (not is_f16_or_bf16)):
                s_b_vals = []
                for ni in range_constexpr(num_acc_n):
                    col_g = by_n + n_tile_base + (ni * 16) + lane_mod_16
                    s_b_vals.append(
                        buffer_ops.buffer_load(
                            scale_b_rsrc, col_g, vec_width=1, dtype=fx.Float32
                        )
                    )
                scales_pf["s_b_vals"] = s_b_vals
                scales_pf["s_a_vecs"] = []
                row_off_base = lane_div_16 * 4
                for mi in range_constexpr(m_repeat):
                    row_base_m = bx_m + (mi * 16)
                    row_g_base = row_base_m + row_off_base
                    s_a_vec = buffer_ops.buffer_load(
                        scale_a_rsrc, row_g_base, vec_width=4, dtype=fx.Float32
                    )
                    scales_pf["s_a_vecs"].append(Vec(s_a_vec))

            current_accs_list = list(accs_in)

            use_mfma_scale_128 = (
                str(gpu_arch).startswith("gfx95")
                and (not is_int8)
                and (not is_int4)
                and (not is_f16_or_bf16)
            )
            if const_expr(use_mfma_scale_128):
                if const_expr((int(tile_k) % 128) != 0):
                    raise ValueError(
                        f"tile_k must be divisible by 128 for mfma_scale_x128, got tile_k={tile_k}"
                    )
                mfma_res_ty = Vec.make_type(4, fx.Float32)
                c0_i64 = fx.Int64(0)

                _fp4_cbsz = 4 if is_fp4 else 0
                _fp4_blgp = 4 if is_fp4 else 0
                _fp4_pack_M = 2 if is_fp4 else 1
                _fp4_pack_N = 2 if is_fp4 else 1
                _fp4_pack_K = 2 if is_fp4 else 1
                _quant_block_size = 32
                _K1 = K // (_quant_block_size * 4 * _fp4_pack_K) if is_fp4 else 1
                _k_unroll_packed = k_unroll // _fp4_pack_K
                _m_repeat_packed = m_repeat // _fp4_pack_M
                _num_acc_n_packed = num_acc_n // _fp4_pack_N

                def pack_i64x4_to_i32x8(x0, x1, x2, x3):
                    return Vec.from_elements([x0, x1, x2, x3], fx.Int64).bitcast(
                        fx.Int32
                    )

                if const_expr(is_fp4):
                    _fp4_a_sc, _fp4_b_sc = fp4_scales if fp4_scales else ([], [])
                    ku128_iters = 1 if _fp4_tilek128 else _k_unroll_packed
                    ikxdl_iters = 1 if _fp4_tilek128 else _fp4_pack_K
                    for ku128 in range_constexpr(ku128_iters):
                        a_scale_base = 0 if _fp4_tilek128 else ku128 * _m_repeat_packed
                        b_scale_base = 0 if _fp4_tilek128 else ku128 * _num_acc_n_packed
                        for mi_p in range_constexpr(_m_repeat_packed):
                            a_scale_val = _fp4_a_sc[a_scale_base + mi_p]
                            for ni_p in range_constexpr(_num_acc_n_packed):
                                b_scale_val = _fp4_b_sc[b_scale_base + ni_p]
                                for ikxdl in range_constexpr(ikxdl_iters):
                                    k_idx = (
                                        0
                                        if _fp4_tilek128
                                        else ku128 * _fp4_pack_K + ikxdl
                                    )
                                    b_packs0, b_packs1 = b_tile_in[k_idx]
                                    col_base = (
                                        col_offset_base_bytes
                                        if _fp4_tilek128
                                        else (
                                            col_offset_base_bytes
                                            + fx.Index((k_idx * 128) // a_elem_vec_pack)
                                        )
                                    )
                                    scale_k_sel = (
                                        fp4_scale_half if _fp4_tilek128 else ikxdl
                                    )
                                    for imxdl in range_constexpr(_fp4_pack_M):
                                        mi_idx = mi_p * _fp4_pack_M + imxdl
                                        curr_row_a_lds = row_a_lds + (mi_idx * 16)
                                        a0 = fx.Int64(0).ir_value()
                                        a1 = fx.Int64(0).ir_value()
                                        if const_expr(
                                            (a0_prefetch is not None)
                                            and (k_idx == 0)
                                            and (mi_idx == 0)
                                        ):
                                            a0, a1 = a0_prefetch
                                        else:
                                            a0, a1 = lds_load_packs_k64(
                                                curr_row_a_lds, col_base, lds_buffer
                                            )
                                        a128 = pack_i64x4_to_i32x8(
                                            a0, a1, c0_i64, c0_i64
                                        )
                                        for inxdl in range_constexpr(_fp4_pack_N):
                                            ni_idx = ni_p * _fp4_pack_N + inxdl
                                            b0 = b_packs0[ni_idx]
                                            b1 = b_packs1[ni_idx]
                                            b128 = pack_i64x4_to_i32x8(
                                                b0, b1, c0_i64, c0_i64
                                            )
                                            acc_idx = mi_idx * num_acc_n + ni_idx
                                            if const_expr(not _fp4_use_scheduler):
                                                rocdl.sched_barrier(0)
                                            current_accs_list[acc_idx] = (
                                                rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                                    mfma_res_ty,
                                                    [
                                                        a128,
                                                        b128,
                                                        current_accs_list[acc_idx],
                                                        _fp4_cbsz,
                                                        _fp4_blgp,
                                                        scale_k_sel * _fp4_pack_M
                                                        + imxdl,
                                                        a_scale_val,
                                                        scale_k_sel * _fp4_pack_N
                                                        + inxdl,
                                                        b_scale_val,
                                                    ],
                                                )
                                            )
                else:
                    for ku128 in range_constexpr(k_unroll // 2):
                        ku0 = ku128 * 2
                        ku1 = ku0 + 1
                        b0_packs0, b0_packs1 = b_tile_in[ku0]
                        b1_packs0, b1_packs1 = b_tile_in[ku1]
                        col_base0 = col_offset_base_bytes + (ku0 * 64)
                        col_base1 = col_offset_base_bytes + (ku1 * 64)

                        for mi in range_constexpr(m_repeat):
                            curr_row_a_lds = row_a_lds + (mi * 16)
                            a0 = fx.Int64(0).ir_value()
                            a1 = fx.Int64(0).ir_value()
                            if const_expr(
                                (a0_prefetch is not None) and (ku0 == 0) and (mi == 0)
                            ):
                                a0, a1 = a0_prefetch
                            else:
                                a0, a1 = lds_load_packs_k64(
                                    curr_row_a_lds, col_base0, lds_buffer
                                )
                            a2, a3 = lds_load_packs_k64(
                                curr_row_a_lds, col_base1, lds_buffer
                            )
                            a128 = pack_i64x4_to_i32x8(a0, a1, a2, a3)

                            for ni in range_constexpr(num_acc_n):
                                b128 = pack_i64x4_to_i32x8(
                                    b0_packs0[ni],
                                    b0_packs1[ni],
                                    b1_packs0[ni],
                                    b1_packs1[ni],
                                )
                                acc_idx = mi * num_acc_n + ni
                                current_accs_list[acc_idx] = (
                                    rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                        mfma_res_ty,
                                        [
                                            a128,
                                            b128,
                                            current_accs_list[acc_idx],
                                            0,
                                            0,
                                            0,
                                            0x7F7F7F7F,
                                            0,
                                            0x7F7F7F7F,
                                        ],
                                    )
                                )
                return current_accs_list, scales_pf

            mfma_res_ty = Vec.make_type(4, fx.Int32 if is_int8 else fx.Float32)
            if const_expr(use_mfma_k32):
                mfma_fn_k32 = (
                    rocdl.mfma_f32_16x16x32_f16
                    if is_f16
                    else rocdl.mfma_f32_16x16x32_bf16
                )

                def i64x2_to_v8(lo, hi):
                    return Vec.from_elements([lo, hi], fx.Int64).bitcast(
                        fx.Float16 if is_f16 else fx.BFloat16
                    )

                def mfma_k64_bytes(acc_in, a0, a1, b0, b1):
                    av = i64x2_to_v8(a0, a1)
                    bv = i64x2_to_v8(b0, b1)
                    return mfma_fn_k32(mfma_res_ty, [av, bv, acc_in, 0, 0, 0])

            else:
                if const_expr(is_int8):
                    mfma_fn = mfma_i32_k32
                elif const_expr(is_f16):
                    mfma_fn = rocdl.mfma_f32_16x16x16f16
                elif const_expr(is_bf16):
                    mfma_fn = rocdl.mfma_f32_16x16x16bf16_1k
                else:
                    mfma_fn = rocdl.mfma_f32_16x16x32_fp8_fp8

                def mfma_step(acc_in, a, b):
                    return mfma_fn(mfma_res_ty, [a, b, acc_in, 0, 0, 0])

                def mfma_k64_bytes(acc_in, a0, a1, b0, b1):
                    acc_mid = mfma_step(acc_in, a0, b0)
                    return mfma_step(acc_mid, a1, b1)

            for ku in range_constexpr(k_unroll):
                b_packs0, b_packs1 = b_tile_in[ku]
                ki64 = ku * 64
                col_base = col_offset_base_bytes + ki64
                for mi in range_constexpr(m_repeat):
                    curr_row_a_lds = row_a_lds + (mi * 16)
                    a0 = fx.Int64(0).ir_value()
                    a1 = fx.Int64(0).ir_value()
                    if const_expr(
                        (a0_prefetch is not None) and (ku == 0) and (mi == 0)
                    ):
                        a0, a1 = a0_prefetch
                    else:
                        a0, a1 = lds_load_packs_k64(
                            curr_row_a_lds, col_base, lds_buffer
                        )
                    for ni in range_constexpr(num_acc_n):
                        acc_idx = mi * num_acc_n + ni
                        current_accs_list[acc_idx] = mfma_k64_bytes(
                            current_accs_list[acc_idx],
                            a0,
                            a1,
                            b_packs0[ni],
                            b_packs1[ni],
                        )
            return current_accs_list, scales_pf

        # ── Epilogue (store output) ───────────────────────────────────────
        def store_output(final_accs, scales):
            s_b_vals = []
            s_a_vecs = []
            if const_expr(not (is_f16_or_bf16 or is_fp4)):
                s_b_vals = scales["s_b_vals"]
                s_a_vecs = scales["s_a_vecs"]

            if const_expr(use_cshuffle_epilog):
                if const_expr(lds_out is None):
                    raise RuntimeError(
                        "use_cshuffle_epilog=True but lds_out is not allocated."
                    )
                gpu.barrier()

                def write_row_to_lds(
                    *,
                    mi,
                    ii,
                    row_in_tile,
                    row,
                    row_base_lds,
                    col_base_local,
                    num_acc_n,
                    lds_out,
                ):
                    s_a = fx.Float32(1.0)
                    if const_expr(_needs_per_token_scale):
                        s_a_vec4 = s_a_vecs[mi]
                        s_a = Vec(s_a_vec4)[ii]
                    for ni in range_constexpr(num_acc_n):
                        col_local = col_base_local + (ni * 16)
                        acc_idx = mi * num_acc_n + ni
                        acc = final_accs[acc_idx]
                        val = Vec(acc)[ii]
                        if const_expr(is_int8):
                            val = fx.Float32(val)
                        if const_expr(is_f16_or_bf16 or is_fp4):
                            val_s = val
                        elif const_expr(_needs_per_token_scale):
                            val_s = (val * s_a) * s_b_vals[ni]
                        else:
                            val_s = val
                        v16 = _out_dtype()(val_s)

                        lds_idx = row_base_lds + col_local
                        v1 = Vec.from_elements([v16], _out_dtype())
                        v1.store(lds_out, [lds_idx], alignment=2)

                def store_pair(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                    idx_out = row * c_n + col_g0
                    byte_off = idx_out * 2
                    e_vec = 4 if (int(tile_n) % (32 * 4)) == 0 else 2
                    if const_expr(e_vec == 4):
                        frag_i32x2 = Vec(frag).bitcast(fx.Int32)
                        buffer_ops.buffer_store(
                            frag_i32x2, c_rsrc, byte_off, offset_is_bytes=True
                        )
                    else:
                        frag_i32x1 = Vec(frag).bitcast(fx.Int32)
                        frag_i32 = frag_i32x1[0]
                        buffer_ops.buffer_store(
                            frag_i32, c_rsrc, byte_off, offset_is_bytes=True
                        )

                e_vec = 4 if (int(tile_n) % (32 * 4)) == 0 else 2
                mfma_epilog(
                    use_cshuffle=True,
                    arith=fx.arith,
                    vector=vector,
                    gpu=gpu,
                    range_constexpr=range_constexpr,
                    tile_m=tile_m,
                    tile_n=tile_n,
                    e_vec=e_vec,
                    m_repeat=m_repeat,
                    num_acc_n=num_acc_n,
                    tx=tx,
                    lane_div_16=lane_div_16,
                    lane_mod_16=lane_mod_16,
                    bx_m=bx_m,
                    by_n=by_n,
                    n_tile_base=n_tile_base,
                    lds_out=lds_out,
                    write_row_to_lds=write_row_to_lds,
                    store_pair=store_pair,
                    frag_elem_type=_out_elem(),
                )
                return

            def body_row(*, mi, ii, row_in_tile, row):
                s_a = fx.Float32(1.0)
                if const_expr(_needs_per_token_scale):
                    s_a_vec4 = s_a_vecs[mi]
                    s_a = Vec(s_a_vec4)[ii]
                col_base = by_n + n_tile_base + lane_mod_16
                idx_base = row * c_n + col_base
                for ni in range_constexpr(num_acc_n):
                    acc_idx = mi * num_acc_n + ni
                    acc = final_accs[acc_idx]
                    val = Vec(acc)[ii]
                    if const_expr(is_int8):
                        val = fx.Float32(val)
                    if const_expr(is_f16_or_bf16 or is_fp4):
                        val_s = val
                    elif const_expr(_needs_per_token_scale):
                        val_s = (val * s_a) * s_b_vals[ni]
                    else:
                        val_s = val

                    # ── Fused epilogue: bias + activation ──
                    if const_expr(_has_bias and bias_rsrc is not None):
                        col_idx = col_base + (ni * 16)
                        bias_val_f16 = buffer_ops.buffer_load(
                            bias_rsrc, col_idx, vec_width=1, dtype=_out_dtype()
                        )
                        bias_val_f32 = fx.Float32(bias_val_f16)
                        val_s = val_s + bias_val_f32

                    if const_expr(_has_relu):
                        # ReLU(x) = max(x, 0). Use maximumf rather than
                        # cmpf+select: the lower-level cmpf wrapper requires
                        # an integer CmpFPredicate enum value, not the string
                        # "ogt", so the previous form failed at compile time
                        # the moment the bias_relu epilogue was actually
                        # exercised (test coverage gap).
                        zero_f32 = fx.Float32(0.0)
                        val_s = fx.Float32(val_s).maximumf(zero_f32)
                    elif const_expr(_has_silu):
                        # SiLU(x) = x * sigmoid(x). Compute as
                        #   sigmoid_x = 1 / (1 + exp(-x))    # one rcp instead of fdiv
                        #   val_s    = val_s * sigmoid_x
                        # to lower to v_rcp_f32 + v_mul_f32 instead of v_div_*
                        # (~4x faster than fdiv on AMD GPUs).
                        neg_one = fx.Float32(-1.0)
                        neg_val = val_s * neg_one
                        exp_neg = math.exp(neg_val)
                        one_f32 = fx.Float32(1.0)
                        denom = one_f32 + exp_neg
                        sigmoid_x = one_f32 / denom
                        val_s = val_s * sigmoid_x
                    elif const_expr(_has_gelu):
                        # GeLU approx: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                        # math.tanh has no AMD libcall, so expand it via exp.
                        # Numerically stable form using only non-positive
                        # exponent (avoids fp32 overflow for large |x|):
                        #   a = -2 * |y|              (a <= 0, exp(a) in [0,1])
                        #   tanh(y) = sign(y) * (1 - exp(a)) / (1 + exp(a))
                        #   1 + tanh(y) = 1 + sign(y) * (1 - exp(a))/(1+exp(a))
                        # We compute (1 + tanh(y)) directly from y because we
                        # need the GeLU output, which is half * x * (1 + tanh).
                        half_f32 = fx.Float32(0.5)
                        coeff_f32 = fx.Float32(0.044715)
                        sqrt2pi_f32 = fx.Float32(0.7978845608)
                        neg_two_f32 = fx.Float32(-2.0)
                        one_f32 = fx.Float32(1.0)
                        zero_f32 = fx.Float32(0.0)
                        x3 = val_s * val_s * val_s
                        y = sqrt2pi_f32 * (val_s + coeff_f32 * x3)
                        # |y| via max(y, -y) — avoids math.absf dependency
                        neg_y = zero_f32 - y
                        abs_y = fx.Float32(y).maximumf(neg_y)
                        # exp(-2|y|) is in [0, 1], no overflow.
                        e_neg2abs = math.exp(neg_two_f32 * abs_y)
                        denom = one_f32 + e_neg2abs
                        # tanh(|y|) = (1 - e_neg2abs) / denom
                        # tanh(y)   = sign(y) * tanh(|y|)
                        # 1 + tanh(y):
                        #   y >= 0: 1 + tanh(|y|) = (denom + (1 - e)) / denom
                        #                         = (2)             / denom
                        #                          (because denom = 1 + e and
                        #                           denom + 1 - e = 2)
                        #   y <  0: 1 - tanh(|y|) = (denom - (1 - e)) / denom
                        #                         = (2 * e)          / denom
                        two_f32 = fx.Float32(2.0)
                        # numerator = 2          when y >= 0
                        #           = 2 * e_neg2abs  when y <  0
                        sign_pred = y > zero_f32
                        num_pos = two_f32
                        num_neg = two_f32 * e_neg2abs
                        numerator = sign_pred.select(num_pos, num_neg)
                        recip = one_f32 / denom
                        one_plus_tanh = numerator * recip
                        val_s = half_f32 * val_s * one_plus_tanh

                    val_f16 = _out_dtype()(val_s)
                    idx_out = idx_base + (ni * 16)
                    buffer_ops.buffer_store(val_f16, c_rsrc, idx_out)

            mfma_epilog(
                use_cshuffle=False,
                arith=fx.arith,
                range_constexpr=range_constexpr,
                m_repeat=m_repeat,
                lane_div_16=lane_div_16,
                bx_m=bx_m,
                body_row=body_row,
            )

        # ── Scheduling hints ──────────────────────────────────────────────
        rocdl.sched_barrier(0)

        def hot_loop_scheduler():
            def _build_scheduler(numer: int, denom: int):
                if const_expr(denom <= 0):
                    return []
                if const_expr(numer <= 0):
                    return [0] * denom
                out = []
                prev = 0
                for i in range_constexpr(denom):
                    cur = ((i + 1) * numer + (denom - 1)) // denom
                    out.append(cur - prev)
                    prev = cur
                return out

            if const_expr(_is_gfx942):
                mfma_group = num_acc_n
                mfma_total = (k_unroll * 2) * m_repeat * mfma_group
                mfma_per_iter = 2 * mfma_group
                sche_iters = 0 if mfma_per_iter == 0 else (mfma_total // mfma_per_iter)
                rocdl.sched_dsrd(2)
                rocdl.sched_mfma(1)
                if const_expr(tile_m == 16):
                    rocdl.sched_vmem(1)
                rocdl.sched_mfma(1)
                if const_expr(tile_m == 16):
                    rocdl.sched_vmem(1)
                if const_expr(num_acc_n < 4):
                    rocdl.sched_dsrd(1)
                    rocdl.sched_mfma(1)
                    if const_expr(tile_m == 16):
                        rocdl.sched_vmem(1)
                    rocdl.sched_dsrd(1)
                    rocdl.sched_mfma(1)
                    if const_expr(tile_m == 16):
                        rocdl.sched_vmem(1)
                    rocdl.sched_mfma(1)
                dswr_tail = num_a_loads
                dstr_advance = 2
                if const_expr(dswr_tail > sche_iters):
                    dswr_tail = sche_iters
                dswr_start = max(sche_iters - dswr_tail - dstr_advance, 0)
                for sche_i in range_constexpr(sche_iters):
                    rocdl.sched_vmem(1)
                    rocdl.sched_mfma(mfma_group)
                    rocdl.sched_dsrd(1)
                    rocdl.sched_mfma(mfma_group)
                    if const_expr(sche_i >= dswr_start - 1):
                        rocdl.sched_dswr(1)
            else:
                mfma_group = num_acc_n
                if const_expr(use_mfma_k32):
                    element_k_per_mfma = 32
                elif const_expr(_is_gfx950):
                    element_k_per_mfma = 128
                else:
                    element_k_per_mfma = 32
                num_mfma_per_tile_k = tile_k // element_k_per_mfma
                mfma_total = num_mfma_per_tile_k * m_repeat * mfma_group
                num_ds_load = num_a_lds_load
                dswr_tail = num_a_loads
                dstr_advance = 2
                if const_expr(dswr_tail > mfma_total):
                    dswr_tail = mfma_total
                num_gmem_loads = num_b_loads + num_a_async_loads
                if const_expr(is_fp4 and tile_k != 128):
                    num_fp4_scale_k_groups = (
                        1 if int(tile_k) == 128 else (k_unroll // 2)
                    )
                    num_a_scale_loads = num_fp4_scale_k_groups * (m_repeat // 2)
                    num_b_scale_loads = num_fp4_scale_k_groups * (num_acc_n // 2)
                    num_gmem_loads += num_a_scale_loads + num_b_scale_loads
                dsrd_preload_eff = min(int(dsrd_preload), num_ds_load)
                dvmem_preload_eff = min(int(dvmem_preload), num_gmem_loads)
                vmem_remaining = num_gmem_loads - dvmem_preload_eff
                dsrd_remaining = num_ds_load - dsrd_preload_eff
                vmem_schedule = []
                if const_expr(vmem_remaining > 0 and vmem_remaining < mfma_total):
                    vmem_schedule = _build_scheduler(vmem_remaining, vmem_remaining) + [
                        0
                    ] * (mfma_total - vmem_remaining)
                else:
                    vmem_schedule = _build_scheduler(vmem_remaining, mfma_total)
                dsrd_schedule = _build_scheduler(dsrd_remaining, mfma_total)
                dswr_start = max(mfma_total - dswr_tail - dstr_advance, 0)
                last_dsrd_mfma_idx = -1
                for sched_idx in range_constexpr(mfma_total):
                    if const_expr(dsrd_schedule[sched_idx]):
                        last_dsrd_mfma_idx = sched_idx
                dswr_start = max(dswr_start, last_dsrd_mfma_idx + 1)
                idx_ds_read = dsrd_preload_eff
                idx_gmem_load = dvmem_preload_eff
                idx_ds_write = 0
                if const_expr(dvmem_preload_eff):
                    rocdl.sched_vmem(dvmem_preload_eff)
                if const_expr(dsrd_preload_eff):
                    rocdl.sched_dsrd(dsrd_preload_eff)
                for mfma_idx in range_constexpr(mfma_total):
                    rocdl.sched_mfma(1)
                    n_dsrd = dsrd_schedule[mfma_idx]
                    if const_expr(n_dsrd and (idx_ds_read < num_ds_load)):
                        if const_expr(idx_ds_read + n_dsrd > num_ds_load):
                            n_dsrd = num_ds_load - idx_ds_read
                        if const_expr(n_dsrd):
                            rocdl.sched_dsrd(n_dsrd)
                            idx_ds_read += n_dsrd

                    n_vmem = vmem_schedule[mfma_idx]
                    if const_expr(n_vmem and (idx_gmem_load < num_gmem_loads)):
                        if const_expr(idx_gmem_load + n_vmem > num_gmem_loads):
                            n_vmem = num_gmem_loads - idx_gmem_load
                        if const_expr(n_vmem):
                            rocdl.sched_vmem(n_vmem)
                            idx_gmem_load += n_vmem
                    if const_expr(
                        (not use_async_copy)
                        and (idx_ds_write < dswr_tail)
                        and (mfma_idx >= dswr_start)
                    ):
                        rocdl.sched_dswr(1)
                        idx_ds_write += 1
                # if any other ds_write is not issued, issue here.
                if const_expr((not use_async_copy) and (idx_ds_write < num_a_loads)):
                    rocdl.sched_dswr(num_a_loads - idx_ds_write)
                # for ds_write_idx in range_constexpr(num_a_loads):
                #     rocdl.sched_dswr(1)

            rocdl.sched_barrier(0)

        # ── Main pipeline ─────────────────────────────────────────────────
        def _flatten_b_tile(bt):
            flat = []
            for packs0, packs1 in bt:
                flat.extend(packs0)
                flat.extend(packs1)
            return flat

        def _unflatten_b_tile(flat):
            bt = []
            idx = 0
            for _ in range_constexpr(k_unroll):
                p0 = [flat[idx + ni] for ni in range_constexpr(num_acc_n)]
                idx += num_acc_n
                p1 = [flat[idx + ni] for ni in range_constexpr(num_acc_n)]
                idx += num_acc_n
                bt.append((p0, p1))
            return bt

        n_accs = num_acc_n * m_repeat
        n_btile = k_unroll * 2 * num_acc_n
        n_a0pf = 2
        n_fp4_asc = 0
        n_fp4_bsc = 0

        if const_expr(is_fp4):
            n_fp4_asc = _k_unroll_packed_outer * _m_repeat_packed_outer
            n_fp4_bsc = _k_unroll_packed_outer * _num_acc_n_packed_outer

        def _pack_state(accs_l, bt_flat, a0pf, fp4_scales=None, *, is_fp4_v):
            state = list(accs_l) + list(bt_flat) + [a0pf[0], a0pf[1]]
            if const_expr(is_fp4_v):
                a_scales, b_scales = fp4_scales
                state.extend(a_scales)
                state.extend(b_scales)
            return state

        def _unpack_state(
            vals, *, n_accs_v, n_btile_v, n_a0pf_v, is_fp4_v, n_fp4_asc_v, n_fp4_bsc_v
        ):
            accs_l = list(vals[:n_accs_v])
            bt_flat = list(vals[n_accs_v : n_accs_v + n_btile_v])
            a0pf = (vals[n_accs_v + n_btile_v], vals[n_accs_v + n_btile_v + 1])
            if const_expr(not is_fp4_v):
                return accs_l, bt_flat, a0pf, None
            sc_base = n_accs_v + n_btile_v + n_a0pf_v
            a_scales = list(vals[sc_base : sc_base + n_fp4_asc_v])
            b_scales = list(
                vals[sc_base + n_fp4_asc_v : sc_base + n_fp4_asc_v + n_fp4_bsc_v]
            )
            return accs_l, bt_flat, a0pf, (a_scales, b_scales)

        def _build_pingpong_body(
            k_iv,
            inner_state,
            *,
            _unpack_state,
            _unflatten_b_tile,
            _fp4_tilek128,
            tile_k,
            use_async_copy,
            prefetch_a_to_lds,
            a_elem_vec_pack,
            dma_a_tile_to_lds,
            prefetch_a_tile,
            prefetch_b_tile,
            compute_tile,
            lds_a_pong,
            lds_a_ping,
            store_a_tile_to_lds,
            hot_loop_scheduler,
            num_b_loads,
            gpu,
            prefetch_a0_pack,
            load_fp4_scale_chunk,
            is_fp4,
            rocdl,
            _pack_state,
            _flatten_b_tile,
            lds_load_packs_k64,
            row_a_lds,
            col_offset_base_bytes,
            n_accs,
            n_btile,
            n_a0pf,
            n_fp4_asc,
            n_fp4_bsc,
        ):
            accs_in, bt_flat_in, a0pf_in, fp4_scales_pong_in = _unpack_state(
                inner_state,
                n_accs_v=n_accs,
                n_btile_v=n_btile,
                n_a0pf_v=n_a0pf,
                is_fp4_v=is_fp4,
                n_fp4_asc_v=n_fp4_asc,
                n_fp4_bsc_v=n_fp4_bsc,
            )
            b_tile_pong_in = _unflatten_b_tile(bt_flat_in)

            if const_expr(_fp4_tilek128):
                next_k1 = k_iv + tile_k
                if const_expr(use_async_copy):
                    prefetch_a_to_lds(
                        next_k1,
                        lds_a_ping,
                        a_elem_vec_pack_v=a_elem_vec_pack,
                        dma_a_tile_to_lds_fn=dma_a_tile_to_lds,
                    )
                else:
                    a_tile_ping = prefetch_a_tile(next_k1)
                b_tile_ping = prefetch_b_tile(next_k1)
                accs_in, _ = compute_tile(
                    accs_in,
                    b_tile_pong_in,
                    lds_a_pong,
                    a0_prefetch=a0pf_in,
                    fp4_scales=fp4_scales_pong_in,
                    fp4_scale_half=0,
                )
                if const_expr(not use_async_copy):
                    store_a_tile_to_lds(a_tile_ping, lds_a_ping)
                hot_loop_scheduler()
                rocdl.s_waitcnt(num_b_loads)
                gpu.barrier()
                a0_prefetch_ping = prefetch_a0_pack(
                    lds_a_ping,
                    lds_load_packs_k64_fn=lds_load_packs_k64,
                    row_a_lds_v=row_a_lds,
                    col_offset_base_bytes_v=col_offset_base_bytes,
                )

                next_k2 = k_iv + (tile_k * 2)
                _sc_ping = load_fp4_scale_chunk(next_k2) if is_fp4 else None
                rocdl.sched_barrier(0)
                if const_expr(use_async_copy):
                    prefetch_a_to_lds(
                        next_k2,
                        lds_a_pong,
                        a_elem_vec_pack_v=a_elem_vec_pack,
                        dma_a_tile_to_lds_fn=dma_a_tile_to_lds,
                    )
                else:
                    a_tile_pong = prefetch_a_tile(next_k2)
                b_tile_pong_new = prefetch_b_tile(next_k2)
                accs_in, _ = compute_tile(
                    accs_in,
                    b_tile_ping,
                    lds_a_ping,
                    a0_prefetch=a0_prefetch_ping,
                    fp4_scales=fp4_scales_pong_in,
                    fp4_scale_half=1,
                )
                if const_expr(not use_async_copy):
                    store_a_tile_to_lds(a_tile_pong, lds_a_pong)
                hot_loop_scheduler()
                rocdl.s_waitcnt(num_b_loads)
                gpu.barrier()
                a0_prefetch_pong_new = prefetch_a0_pack(
                    lds_a_pong,
                    lds_load_packs_k64_fn=lds_load_packs_k64,
                    row_a_lds_v=row_a_lds,
                    col_offset_base_bytes_v=col_offset_base_bytes,
                )

                return _pack_state(
                    accs_in,
                    _flatten_b_tile(b_tile_pong_new),
                    a0_prefetch_pong_new,
                    _sc_ping,
                    is_fp4_v=is_fp4,
                )

            next_k1 = k_iv + tile_k
            if const_expr(use_async_copy):
                prefetch_a_to_lds(
                    next_k1,
                    lds_a_ping,
                    a_elem_vec_pack_v=a_elem_vec_pack,
                    dma_a_tile_to_lds_fn=dma_a_tile_to_lds,
                )
            else:
                a_tile = prefetch_a_tile(next_k1)
            _sc_ping = load_fp4_scale_chunk(k_iv + fx.Index(tile_k)) if is_fp4 else None
            b_tile_ping = prefetch_b_tile(next_k1)
            accs_in, _ = compute_tile(
                accs_in,
                b_tile_pong_in,
                lds_a_pong,
                a0_prefetch=a0pf_in,
                fp4_scales=fp4_scales_pong_in,
            )
            if const_expr(not use_async_copy):
                store_a_tile_to_lds(a_tile, lds_a_ping)
            hot_loop_scheduler()
            rocdl.s_waitcnt(num_b_loads)
            gpu.barrier()
            a0_prefetch_ping = prefetch_a0_pack(
                lds_a_ping,
                lds_load_packs_k64_fn=lds_load_packs_k64,
                row_a_lds_v=row_a_lds,
                col_offset_base_bytes_v=col_offset_base_bytes,
            )

            next_k2 = k_iv + (tile_k * 2)
            if const_expr(use_async_copy):
                prefetch_a_to_lds(
                    next_k2,
                    lds_a_pong,
                    a_elem_vec_pack_v=a_elem_vec_pack,
                    dma_a_tile_to_lds_fn=dma_a_tile_to_lds,
                )
            else:
                a_tile = prefetch_a_tile(next_k2)
            _sc_pong = load_fp4_scale_chunk(k_iv + (tile_k * 2)) if is_fp4 else None
            b_tile_pong_new = prefetch_b_tile(next_k2)
            accs_in, _ = compute_tile(
                accs_in,
                b_tile_ping,
                lds_a_ping,
                a0_prefetch=a0_prefetch_ping,
                fp4_scales=_sc_ping,
            )
            if const_expr(not use_async_copy):
                store_a_tile_to_lds(a_tile, lds_a_pong)
            hot_loop_scheduler()
            rocdl.s_waitcnt(num_b_loads)
            gpu.barrier()
            a0_prefetch_pong_new = prefetch_a0_pack(
                lds_a_pong,
                lds_load_packs_k64_fn=lds_load_packs_k64,
                row_a_lds_v=row_a_lds,
                col_offset_base_bytes_v=col_offset_base_bytes,
            )

            return _pack_state(
                accs_in,
                _flatten_b_tile(b_tile_pong_new),
                a0_prefetch_pong_new,
                _sc_pong,
                is_fp4_v=is_fp4,
            )

        if const_expr(lds_stage == 2):

            def prefetch_a0_pack(
                lds_buffer,
                *,
                lds_load_packs_k64_fn,
                row_a_lds_v,
                col_offset_base_bytes_v,
            ):
                return lds_load_packs_k64_fn(
                    row_a_lds_v, col_offset_base_bytes_v, lds_buffer
                )

            k0 = fx.Index(0)
            b_tile0 = prefetch_b_tile(k0)
            if const_expr(use_async_copy):
                prefetch_a_to_lds(
                    k0,
                    lds_a_pong,
                    a_elem_vec_pack_v=a_elem_vec_pack,
                    dma_a_tile_to_lds_fn=dma_a_tile_to_lds,
                )
            else:
                store_a_tile_to_lds(prefetch_a_tile(k0), lds_a_pong)
            gpu.barrier()
            accs = [acc_init] * n_accs
            a0_prefetch_pong = prefetch_a0_pack(
                lds_a_pong,
                lds_load_packs_k64_fn=lds_load_packs_k64,
                row_a_lds_v=row_a_lds,
                col_offset_base_bytes_v=col_offset_base_bytes,
            )
            fp4_scales0 = load_fp4_scale_chunk(fx.Index(0)) if is_fp4 else None

            final_accs = 1
            scales = 1
            num_tiles = K // tile_k
            if const_expr(_fp4_tilek128):
                if const_expr((num_tiles % 2) == 1):
                    c_k_main = K - tile_k
                    init_state = _pack_state(
                        accs,
                        _flatten_b_tile(b_tile0),
                        a0_prefetch_pong,
                        fp4_scales0,
                        is_fp4_v=is_fp4,
                    )
                    results = init_state
                    for iv, inner in range(0, c_k_main, tile_k * 2, init=init_state):
                        results = yield _build_pingpong_body(
                            iv,
                            inner,
                            _unpack_state=_unpack_state,
                            _unflatten_b_tile=_unflatten_b_tile,
                            _fp4_tilek128=_fp4_tilek128,
                            tile_k=tile_k,
                            use_async_copy=use_async_copy,
                            prefetch_a_to_lds=prefetch_a_to_lds,
                            a_elem_vec_pack=a_elem_vec_pack,
                            dma_a_tile_to_lds=dma_a_tile_to_lds,
                            prefetch_a_tile=prefetch_a_tile,
                            prefetch_b_tile=prefetch_b_tile,
                            compute_tile=compute_tile,
                            lds_a_pong=lds_a_pong,
                            lds_a_ping=lds_a_ping,
                            store_a_tile_to_lds=store_a_tile_to_lds,
                            hot_loop_scheduler=hot_loop_scheduler,
                            num_b_loads=num_b_loads,
                            gpu=gpu,
                            prefetch_a0_pack=prefetch_a0_pack,
                            load_fp4_scale_chunk=load_fp4_scale_chunk,
                            is_fp4=is_fp4,
                            rocdl=rocdl,
                            _pack_state=_pack_state,
                            _flatten_b_tile=_flatten_b_tile,
                            lds_load_packs_k64=lds_load_packs_k64,
                            row_a_lds=row_a_lds,
                            col_offset_base_bytes=col_offset_base_bytes,
                            n_accs=n_accs,
                            n_btile=n_btile,
                            n_a0pf=n_a0pf,
                            n_fp4_asc=n_fp4_asc,
                            n_fp4_bsc=n_fp4_bsc,
                        )
                    accs, bt_flat, a0pf, fp4_scales_final = _unpack_state(
                        results,
                        n_accs_v=n_accs,
                        n_btile_v=n_btile,
                        n_a0pf_v=n_a0pf,
                        is_fp4_v=is_fp4,
                        n_fp4_asc_v=n_fp4_asc,
                        n_fp4_bsc_v=n_fp4_bsc,
                    )
                    b_tile_pong_final = _unflatten_b_tile(bt_flat)
                    final_accs, scales = compute_tile(
                        accs,
                        b_tile_pong_final,
                        lds_a_pong,
                        is_last_tile=not is_fp4,
                        a0_prefetch=a0pf,
                        fp4_scales=fp4_scales_final,
                        fp4_scale_half=0,
                    )
                else:
                    c_k_stop = K - (tile_k * 3)
                    init_state = _pack_state(
                        accs,
                        _flatten_b_tile(b_tile0),
                        a0_prefetch_pong,
                        fp4_scales0,
                        is_fp4_v=is_fp4,
                    )
                    results = init_state
                    for iv, inner in range(0, c_k_stop, tile_k * 2, init=init_state):
                        results = yield _build_pingpong_body(
                            iv,
                            inner,
                            _unpack_state=_unpack_state,
                            _unflatten_b_tile=_unflatten_b_tile,
                            _fp4_tilek128=_fp4_tilek128,
                            tile_k=tile_k,
                            use_async_copy=use_async_copy,
                            prefetch_a_to_lds=prefetch_a_to_lds,
                            a_elem_vec_pack=a_elem_vec_pack,
                            dma_a_tile_to_lds=dma_a_tile_to_lds,
                            prefetch_a_tile=prefetch_a_tile,
                            prefetch_b_tile=prefetch_b_tile,
                            compute_tile=compute_tile,
                            lds_a_pong=lds_a_pong,
                            lds_a_ping=lds_a_ping,
                            store_a_tile_to_lds=store_a_tile_to_lds,
                            hot_loop_scheduler=hot_loop_scheduler,
                            num_b_loads=num_b_loads,
                            gpu=gpu,
                            prefetch_a0_pack=prefetch_a0_pack,
                            load_fp4_scale_chunk=load_fp4_scale_chunk,
                            is_fp4=is_fp4,
                            rocdl=rocdl,
                            _pack_state=_pack_state,
                            _flatten_b_tile=_flatten_b_tile,
                            lds_load_packs_k64=lds_load_packs_k64,
                            row_a_lds=row_a_lds,
                            col_offset_base_bytes=col_offset_base_bytes,
                            n_accs=n_accs,
                            n_btile=n_btile,
                            n_a0pf=n_a0pf,
                            n_fp4_asc=n_fp4_asc,
                            n_fp4_bsc=n_fp4_bsc,
                        )
                    accs, bt_flat, a0pf, fp4_scales_ep = _unpack_state(
                        results,
                        n_accs_v=n_accs,
                        n_btile_v=n_btile,
                        n_a0pf_v=n_a0pf,
                        is_fp4_v=is_fp4,
                        n_fp4_asc_v=n_fp4_asc,
                        n_fp4_bsc_v=n_fp4_bsc,
                    )
                    b_tile_pong_ep = _unflatten_b_tile(bt_flat)

                    last_k = fx.Index(K - tile_k)
                    b_tile_ping = prefetch_b_tile(last_k)
                    if const_expr(use_async_copy):
                        prefetch_a_to_lds(
                            last_k,
                            lds_a_ping,
                            a_elem_vec_pack_v=a_elem_vec_pack,
                            dma_a_tile_to_lds_fn=dma_a_tile_to_lds,
                        )
                    else:
                        a_regs_ping = prefetch_a_tile(last_k)
                    accs, _ = compute_tile(
                        accs,
                        b_tile_pong_ep,
                        lds_a_pong,
                        a0_prefetch=a0pf,
                        fp4_scales=fp4_scales_ep,
                        fp4_scale_half=0,
                    )
                    if const_expr(not use_async_copy):
                        store_a_tile_to_lds(a_regs_ping, lds_a_ping)
                    rocdl.s_waitcnt(num_b_loads)
                    gpu.barrier()
                    a0_prefetch_ping = prefetch_a0_pack(
                        lds_a_ping,
                        lds_load_packs_k64_fn=lds_load_packs_k64,
                        row_a_lds_v=row_a_lds,
                        col_offset_base_bytes_v=col_offset_base_bytes,
                    )
                    final_accs, scales = compute_tile(
                        accs,
                        b_tile_ping,
                        lds_a_ping,
                        is_last_tile=not is_fp4,
                        a0_prefetch=a0_prefetch_ping,
                        fp4_scales=fp4_scales_ep,
                        fp4_scale_half=1,
                    )
            elif const_expr((num_tiles % 2) == 1):
                c_k_main = K - tile_k
                init_state = _pack_state(
                    accs,
                    _flatten_b_tile(b_tile0),
                    a0_prefetch_pong,
                    fp4_scales0,
                    is_fp4_v=is_fp4,
                )
                results = init_state
                for iv, inner in range(0, c_k_main, tile_k * 2, init=init_state):
                    results = yield _build_pingpong_body(
                        iv,
                        inner,
                        _unpack_state=_unpack_state,
                        _unflatten_b_tile=_unflatten_b_tile,
                        _fp4_tilek128=_fp4_tilek128,
                        tile_k=tile_k,
                        use_async_copy=use_async_copy,
                        prefetch_a_to_lds=prefetch_a_to_lds,
                        a_elem_vec_pack=a_elem_vec_pack,
                        dma_a_tile_to_lds=dma_a_tile_to_lds,
                        prefetch_a_tile=prefetch_a_tile,
                        prefetch_b_tile=prefetch_b_tile,
                        compute_tile=compute_tile,
                        lds_a_pong=lds_a_pong,
                        lds_a_ping=lds_a_ping,
                        store_a_tile_to_lds=store_a_tile_to_lds,
                        hot_loop_scheduler=hot_loop_scheduler,
                        num_b_loads=num_b_loads,
                        gpu=gpu,
                        prefetch_a0_pack=prefetch_a0_pack,
                        load_fp4_scale_chunk=load_fp4_scale_chunk,
                        is_fp4=is_fp4,
                        rocdl=rocdl,
                        _pack_state=_pack_state,
                        _flatten_b_tile=_flatten_b_tile,
                        lds_load_packs_k64=lds_load_packs_k64,
                        row_a_lds=row_a_lds,
                        col_offset_base_bytes=col_offset_base_bytes,
                        n_accs=n_accs,
                        n_btile=n_btile,
                        n_a0pf=n_a0pf,
                        n_fp4_asc=n_fp4_asc,
                        n_fp4_bsc=n_fp4_bsc,
                    )
                accs, bt_flat, a0pf, fp4_scales_final = _unpack_state(
                    results,
                    n_accs_v=n_accs,
                    n_btile_v=n_btile,
                    n_a0pf_v=n_a0pf,
                    is_fp4_v=is_fp4,
                    n_fp4_asc_v=n_fp4_asc,
                    n_fp4_bsc_v=n_fp4_bsc,
                )
                b_tile_pong_final = _unflatten_b_tile(bt_flat)
                final_accs, scales = compute_tile(
                    accs,
                    b_tile_pong_final,
                    lds_a_pong,
                    is_last_tile=not is_fp4,
                    a0_prefetch=a0pf,
                    fp4_scales=fp4_scales_final,
                )
            else:
                c_k_stop = K - (tile_k * 3)
                init_state = _pack_state(
                    accs,
                    _flatten_b_tile(b_tile0),
                    a0_prefetch_pong,
                    fp4_scales0,
                    is_fp4_v=is_fp4,
                )
                results = init_state
                for iv, inner in range(0, c_k_stop, tile_k * 2, init=init_state):
                    results = yield _build_pingpong_body(
                        iv,
                        inner,
                        _unpack_state=_unpack_state,
                        _unflatten_b_tile=_unflatten_b_tile,
                        _fp4_tilek128=_fp4_tilek128,
                        tile_k=tile_k,
                        use_async_copy=use_async_copy,
                        prefetch_a_to_lds=prefetch_a_to_lds,
                        a_elem_vec_pack=a_elem_vec_pack,
                        dma_a_tile_to_lds=dma_a_tile_to_lds,
                        prefetch_a_tile=prefetch_a_tile,
                        prefetch_b_tile=prefetch_b_tile,
                        compute_tile=compute_tile,
                        lds_a_pong=lds_a_pong,
                        lds_a_ping=lds_a_ping,
                        store_a_tile_to_lds=store_a_tile_to_lds,
                        hot_loop_scheduler=hot_loop_scheduler,
                        num_b_loads=num_b_loads,
                        gpu=gpu,
                        prefetch_a0_pack=prefetch_a0_pack,
                        load_fp4_scale_chunk=load_fp4_scale_chunk,
                        is_fp4=is_fp4,
                        rocdl=rocdl,
                        _pack_state=_pack_state,
                        _flatten_b_tile=_flatten_b_tile,
                        lds_load_packs_k64=lds_load_packs_k64,
                        row_a_lds=row_a_lds,
                        col_offset_base_bytes=col_offset_base_bytes,
                        n_accs=n_accs,
                        n_btile=n_btile,
                        n_a0pf=n_a0pf,
                        n_fp4_asc=n_fp4_asc,
                        n_fp4_bsc=n_fp4_bsc,
                    )
                accs, bt_flat, a0pf, fp4_scales_ep = _unpack_state(
                    results,
                    n_accs_v=n_accs,
                    n_btile_v=n_btile,
                    n_a0pf_v=n_a0pf,
                    is_fp4_v=is_fp4,
                    n_fp4_asc_v=n_fp4_asc,
                    n_fp4_bsc_v=n_fp4_bsc,
                )
                b_tile_pong_ep = _unflatten_b_tile(bt_flat)

                last_k = fx.Index(K - tile_k)
                b_tile_ping = prefetch_b_tile(last_k)
                if const_expr(use_async_copy):
                    prefetch_a_to_lds(
                        last_k,
                        lds_a_ping,
                        a_elem_vec_pack_v=a_elem_vec_pack,
                        dma_a_tile_to_lds_fn=dma_a_tile_to_lds,
                    )
                else:
                    a_regs_ping = prefetch_a_tile(last_k)
                _sc_last = load_fp4_scale_chunk(last_k) if is_fp4 else None
                accs, _ = compute_tile(
                    accs,
                    b_tile_pong_ep,
                    lds_a_pong,
                    a0_prefetch=a0pf,
                    fp4_scales=fp4_scales_ep,
                )
                if const_expr(not use_async_copy):
                    store_a_tile_to_lds(a_regs_ping, lds_a_ping)
                hot_loop_scheduler()
                rocdl.s_waitcnt(num_b_loads)
                gpu.barrier()
                a0_prefetch_ping = prefetch_a0_pack(
                    lds_a_ping,
                    lds_load_packs_k64_fn=lds_load_packs_k64,
                    row_a_lds_v=row_a_lds,
                    col_offset_base_bytes_v=col_offset_base_bytes,
                )
                final_accs, scales = compute_tile(
                    accs,
                    b_tile_ping,
                    lds_a_ping,
                    is_last_tile=not is_fp4,
                    a0_prefetch=a0_prefetch_ping,
                    fp4_scales=_sc_last,
                )
            store_output(final_accs, scales)
        else:
            a_regs0, b_tile0 = prefetch_ab_tile(fx.Index(0))
            store_a_tile_to_lds(a_regs0, lds_a_pong)
            gpu.barrier()
            accs = [acc_init] * n_accs
            bt_flat0 = _flatten_b_tile(b_tile0)

            init_state = list(accs) + list(bt_flat0)
            for iv, state in range(0, K - tile_k, tile_k, init=init_state):
                accs_in = list(state[:n_accs])
                bt_flat_in = list(state[n_accs:])
                b_tile_in = _unflatten_b_tile(bt_flat_in)

                next_k = iv + tile_k
                a_next, b_next = prefetch_ab_tile(next_k)
                _fp4_sc = (
                    load_fp4_scales(
                        iv // fx.Index(tile_k) * fx.Index(_fp4_scale_k_stride)
                    )
                    if is_fp4
                    else None
                )
                accs_in, _ = compute_tile(
                    accs_in, b_tile_in, lds_a_pong, fp4_scales=_fp4_sc
                )
                gpu.barrier()
                store_a_tile_to_lds(a_next, lds_a_pong)
                hot_loop_scheduler()
                rocdl.s_waitcnt(num_b_loads)
                gpu.barrier()
                results = yield list(accs_in) + _flatten_b_tile(b_next)

            accs_final = list(results[:n_accs])
            bt_final = _unflatten_b_tile(list(results[n_accs:]))
            _last_fp4_sc = (
                load_fp4_scales(fx.Index((K - tile_k) // tile_k * _fp4_scale_k_stride))
                if is_fp4
                else None
            )
            final_accs, scales = compute_tile(
                accs_final,
                bt_final,
                lds_a_pong,
                is_last_tile=not is_fp4,
                fp4_scales=_last_fp4_sc,
            )
            store_output(final_accs, scales)

    # ── Host launcher ──────────────────────────────────────────────────────
    @flyc.jit
    def launch_gemm(
        arg_c: fx.Pointer,
        arg_a: fx.Pointer,
        arg_b: fx.Pointer,
        arg_scale_a: fx.Pointer,
        arg_scale_b: fx.Pointer,
        arg_bias: fx.Pointer,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        stream: fx.Stream,
    ):
        allocator_pong.finalized = False
        allocator_ping.finalized = False
        ctx = CompilationContext.get_current()
        from flydsl._mlir import ir

        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator_pong.finalize()
            allocator_ping.finalize()

        gx = (i32_m + (tile_m - 1)) // tile_m
        gy = i32_n // tile_n

        kernel_gemm._func.__name__ = KERNEL_NAME
        launcher = kernel_gemm(
            arg_c, arg_a, arg_b, arg_scale_a, arg_scale_b, arg_bias, i32_m, i32_n
        )
        if const_expr(waves_per_eu is not None):
            _wpe = int(waves_per_eu)
            if const_expr(_wpe >= 1):
                for op in ctx.gpu_module_body.operations:
                    if const_expr(
                        hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func"
                    ):
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                            fx.Int32.ir_type, _wpe
                        )
        launcher.launch(
            grid=(gx, gy, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch_gemm


# ===========================================================================
# Host launcher
# ===========================================================================
__all__ = [
    "flydsl_gemm_a8w8_bpreshuffle",
    "build_gemm_a8w8_bpreshuffle_module",
    "compile_preshuffle_gemm_a8",
    "preshuffle_weight_a8",
]


def _ptr_view_safe(t: torch.Tensor):
    type_name = type(t).__name__
    module_name = type(t).__module__
    if type_name == "FakeTensor" or "fake_tensor" in module_name:
        return flyc.from_c_void_p(fx.Uint8, 0)
    return flyc.from_c_void_p(fx.Uint8, t.data_ptr())


def _run_compiled(exe, *args):
    cf = getattr(exe, "_cf", None)
    if cf is None:
        cf = flyc.compile(exe, *args)
        exe._cf = cf
    else:
        cf(*args)


def preshuffle_weight_a8(wq: torch.Tensor, layout=(16, 16)) -> torch.Tensor:
    """Pre-shuffle a quantized weight ``[N, K]`` into the CK/aiter B layout.

    Mirrors ``aiter.ops.shuffle.shuffle_weight(w, layout=(16, 16))`` for the
    fp8/int8 (1-byte) element case: the only host-side data prep this kernel
    needs. The result is a pure permutation of the K elements, so it does not
    change the GEMM result, only the memory order the kernel expects.
    """
    x_type = wq.dtype
    x = wq
    IN, IK = layout
    BK = IK * 2
    K = 16 // x.element_size()
    BN = IN
    if x.shape[-2] % BN != 0:
        raise ValueError(f"N={x.shape[-2]} must be divisible by {BN}")
    if x.shape[-1] % BK != 0:
        raise ValueError(f"K={x.shape[-1]} must be divisible by {BK}")
    x_ = x.view(-1, x.shape[-2] // BN, BN, x.shape[-1] // BK, BK // K, K)
    x_ = x_.permute(0, 1, 3, 4, 2, 5)
    x_ = x_.contiguous()
    x_ = x_.view(*x.shape)
    return x_.view(x_type)


def _in_dtype_str(dtype: torch.dtype) -> str:
    if dtype in (torch.float8_e4m3fn, getattr(torch, "float8_e4m3fnuz", dtype)):
        return "fp8"
    if dtype == torch.int8:
        return "int8"
    raise ValueError(f"unsupported input dtype {dtype!r}; expected fp8 or int8")


def _out_dtype_str(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "fp16"
    raise ValueError(f"unsupported output dtype {dtype!r}; expected bf16 or fp16")


def build_gemm_a8w8_bpreshuffle_module(
    n: int,
    k: int,
    *,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    in_dtype: str = "fp8",
    out_dtype: str = "bf16",
    lds_stage: int = 2,
    use_cshuffle_epilog: bool = False,
    use_async_copy: bool = False,
    waves_per_eu: Optional[int] = None,
    xcd_swizzle: int = 0,
):
    """Build (and cache) one inline FlyDSL preshuffle-GEMM launcher."""
    return compile_preshuffle_gemm_a8(
        N=n,
        K=k,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        lds_stage=lds_stage,
        use_cshuffle_epilog=use_cshuffle_epilog,
        use_async_copy=use_async_copy,
        waves_per_eu=waves_per_eu,
        xcd_swizzle=xcd_swizzle,
    )


def _as_i8(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int8) if "float8" in str(t.dtype) else t


def flydsl_gemm_a8w8_bpreshuffle(
    xq: torch.Tensor,
    wq: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    *,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    lds_stage: int = 2,
    use_cshuffle_epilog: int = 0,
    use_async_copy: int = 0,
    waves_per_eu: int = 0,
    xcd_swizzle: int = 0,
    stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """Run the inline FlyDSL a8w8 b-preshuffle GEMM.

    ``xq`` is fp8/int8 ``[M, K]``; ``wq`` is the (16, 16) pre-shuffled fp8/int8
    weight (shape ``[N, K]``); ``x_scale`` is ``[M, 1]`` and ``w_scale`` is
    ``[N, 1]`` fp32. Returns ``out = (xq @ wq.T) * x_scale * w_scale`` in
    ``out_dtype`` (bf16/f16), accumulated in fp32.
    """
    if xq.device.type != "cuda" or wq.device.type != "cuda":
        raise ValueError("flydsl_gemm_a8w8_bpreshuffle only supports CUDA/ROCm tensors")
    m, k = int(xq.shape[0]), int(xq.shape[-1])
    n = int(wq.shape[0])
    if n % tile_n != 0:
        raise ValueError(f"N ({n}) must be a multiple of tile_n ({tile_n})")
    if k % tile_k != 0:
        raise ValueError(f"K ({k}) must be a multiple of tile_k ({tile_k})")

    in_dtype = _in_dtype_str(xq.dtype)
    out_dtype_str = _out_dtype_str(out_dtype)
    wpe = None if waves_per_eu <= 0 else waves_per_eu

    if out is None:
        out = torch.empty((m, n), dtype=out_dtype, device=xq.device)

    exe = compile_preshuffle_gemm_a8(
        N=n,
        K=k,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        in_dtype=in_dtype,
        out_dtype=out_dtype_str,
        lds_stage=lds_stage,
        use_cshuffle_epilog=bool(use_cshuffle_epilog),
        use_async_copy=bool(use_async_copy),
        waves_per_eu=wpe,
        xcd_swizzle=int(xcd_swizzle),
    )

    launch_stream = (
        torch.cuda.current_stream(device=xq.device) if stream is None else stream
    )
    out_contig = out.contiguous()
    # The preshuffle kernel reserves an arg_bias slot used only when
    # epilogue != "none"; pass an empty placeholder for the default path.
    dummy_bias = torch.empty(0, dtype=out.dtype, device=out.device)
    _run_compiled(
        exe,
        _ptr_view_safe(out_contig.view(-1)),
        _ptr_view_safe(_as_i8(xq.contiguous()).view(-1)),
        _ptr_view_safe(_as_i8(wq.contiguous()).view(-1)),
        _ptr_view_safe(x_scale.contiguous().view(-1)),
        _ptr_view_safe(w_scale.contiguous().view(-1)),
        _ptr_view_safe(dummy_bias),
        m,
        n,
        fx.Stream(launch_stream),
    )
    if out_contig is not out:
        out.copy_(out_contig)
    return out
