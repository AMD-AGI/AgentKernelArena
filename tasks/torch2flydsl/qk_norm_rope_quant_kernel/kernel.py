# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL fused QK-RMSNorm + GPT-J RoPE with optional FP8 quant.

Per-row RMSNorm (scope = head_dim D, eps = 1e-6, per-channel gamma on KV and
optionally Q) + GPT-J pair-interleaved RoPE on the last ``rope_head_dim`` (RD)
elements, for Q ``[T, H, D]`` and KV ``[T, D]`` in one kernel launch. Host entry
points:

  * ``build_qk_norm_rope_quant_module(...)`` -> compiled FlyDSL launcher
  * ``flydsl_qk_norm_rope_quant(q, kv, ...)`` -> runs the kernel, returns
    ``(q_out, kv_out, q_scale_or_None, kv_scale_or_None)``

Grid: X = num_q_heads + 1 (``bid_x < H`` handle Q heads, ``bid_x == H`` handles
KV), Y = num_tokens (chunked at the 65535 HW grid-Y limit). One wave64 per
block: thread ``t`` owns ``VEC = D // 64`` elements, so RMSNorm/amax reductions
are wave-local (``shuffle_xor`` butterfly, no LDS, no barrier).

The ``quant=False`` path writes bf16; the ``quant=True`` path produces FP8
(e4m3) output with fp32 or e8m0 group block-scales.

# NOTE: do NOT add `from __future__ import annotations` to this file. PEP 563
# stringifies annotations, defeating flydsl's JitFunction._make_cache_key
# runtime detection (is_runtime = hasattr(ann, "__get_c_pointers__")). With
# string annotations the Int32 params (kv_in_row_stride, num_tokens) would be
# treated as compile-time constants and embedded in the cache key, forcing a
# fresh JIT compile per distinct batch size / KV stride.
"""

import math
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, ptrtoint, range_constexpr
from flydsl_compat import buffer_ops, vector
from flydsl.expr import math as fmath
from flydsl.expr.arith import ArithValue, CmpFPredicate
from flydsl.expr.typing import Int32, Stream, T
ReductionOp = vector.ReductionOp
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly, llvm, rocdl
from flydsl.compiler.protocol import extract_to_ir_values


# ===========================================================================
# On-device tensor shim
# ===========================================================================
def _to_raw(v):
    """Convert ArithValue / Numeric (Int32, Boolean, ...) to raw ir.Value."""
    if isinstance(v, ir.Value):
        return v
    if hasattr(v, "ir_value"):
        return _to_raw(v.ir_value())
    return ir.Value._CAPICreate(v._CAPIPtr)


class GTensor:
    """Minimal global-memory tensor (buffer-resource backed) used by the kernel.

    Inlined subset of ``tensor_shim.GTensor`` (only ``load``/``store``/``rsrc``
    /``get_llvm_ptr`` and the ``static_bytes_offset_i64`` per-token base shift
    are exercised by this kernel).
    """

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
        self.dtype = dtype
        self.shape = shape
        if stride is None:
            self.stride = tuple((np.cumprod(shape[::-1])[::-1].tolist() + [1])[1:])
        else:
            self.stride = stride
        self.base_offset = base_offset
        raw = extract_to_ir_values(memref)[0]
        if static_bytes_offset_i64 is None:
            if str(raw.type).startswith("!fly.ptr"):
                base_i64 = arith.index_cast(T.i64, ptrtoint(memref))
                self.rsrc = buffer_ops.create_buffer_resource_from_addr(base_i64)
            else:
                self.rsrc = buffer_ops.create_buffer_resource(memref, max_size=True)
        else:
            array_base_i64 = self.get_llvm_ptr(memref, static_bytes_offset_i64)
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
        bytes_offset_i64 = arith.index_cast(T.i64, bytes_offset_i64)
        _ptr_type = ir.Type.parse(ptr_type)
        raw = extract_to_ir_values(ptr)[0]
        if str(raw.type).startswith("!fly.ptr"):
            base_ptr = arith.index_cast(T.i64, ptrtoint(ptr))
        else:
            base_ptr = fly.extract_aligned_pointer_as_index(_ptr_type, raw)
            base_ptr = llvm.PtrToIntOp(T.i64, base_ptr).result
        llvm_ptr = llvm.AddOp(
            base_ptr, bytes_offset_i64, llvm.IntegerOverflowFlags(0)
        ).result
        return llvm_ptr


# ===========================================================================
# Shape / quant constants
# ===========================================================================
BLOCK_THREADS = 64  # 1 wave64

_SQRT2 = math.sqrt(2.0)

GROUP_SIZE_OPTIONS = (32, 64, 128)

SCALE_DTYPE_FP32 = "fp32"
SCALE_DTYPE_E8M0 = "e8m0"
SCALE_DTYPE_OPTIONS = (SCALE_DTYPE_FP32, SCALE_DTYPE_E8M0)

# E8M0 encoding (matches silu_and_mul_fq / mixed_moe_gemm). For e4m3fnuz
# (FP8_MAX = 240 ~= 2^7.9): headroom = 7 keeps factor * amax_safe <= 2^7 = 128.
_E8M0_HEADROOM = 7

_TORCH_DTYPE_FOR_SCALE = {
    SCALE_DTYPE_FP32: torch.float32,
    SCALE_DTYPE_E8M0: torch.uint8,
}


@lru_cache(maxsize=1)
def _fp8_const():
    """Lazy-resolve per-GFX native fp8 algebra coefficients (quant path only).

    ``aiter.utility.dtypes.fp8`` selects e4m3fnuz on gfx942 and e4m3fn on
    gfx950 / gfx1250; ``cvt_pk_fp8_f32`` emits the per-gfx native format, so
    FP8_MAX must track that. Imported lazily (only when ``quant=True``) so the
    bf16 path stays free of any aiter dependency.
    """
    from aiter.utility import dtypes as aiter_dtypes

    fp8_dtype = aiter_dtypes.fp8
    fp8_max = float(torch.finfo(fp8_dtype).max)
    return {
        "dtype": fp8_dtype,
        "max": fp8_max,
        "max_over_sqrt2": fp8_max / _SQRT2,
        "inv_max_sqrt2": _SQRT2 / fp8_max,
    }


# ===========================================================================
# Store helpers
# ===========================================================================
def _store_bf16_vec_g(vals_list, g_out, row_off_elems, idx, vec):
    """Convert VEC fp32 values to a bf16 vector and store via a GTensor whose
    base is already shifted per-token. ``row_off_elems`` is this head's row
    offset within the token (i32 elements); ``idx`` is the lane id."""
    vec_t = T.vec(vec, T.f32)
    raw = [v.ir_value() if hasattr(v, "ir_value") else v for v in vals_list]
    f32v = vector.from_elements(vec_t, raw)
    bf16v = f32v.truncf(T.vec(vec, T.bf16))
    my_off = ArithValue(row_off_elems) + ArithValue(idx) * arith.constant(
        vec, type=T.i32
    )
    g_out.store(my_off, bf16v, vec_size=vec)


def _store_fp8_packed(vals_list, out_rsrc, row_base_bytes, idx, vec):
    """Pack VEC fp32 -> VEC fp8 via cvt_pk_fp8_f32 and store (1 dwordx2/thread).

    Workaround for the e4m3fnuz NaN encoding 0x80: cvt_pk_fp8_f32 returns 0x80
    (NaN) for inputs that round to negative zero. Clamp v in (-2^-8, 0) to +0.
    """
    f32 = T.f32
    i32 = T.i32
    c0 = arith.constant(0.0, type=f32)
    c_neg_uf = arith.constant(-(2.0**-8), type=f32)
    c8 = arith.constant(8, type=i32)

    safe = []
    for v in vals_list:
        vv = v.ir_value() if hasattr(v, "ir_value") else v
        is_tn = arith.andi(
            arith.cmpf(CmpFPredicate.OLT, vv, c0),
            arith.cmpf(CmpFPredicate.OGT, vv, c_neg_uf),
        )
        safe.append(arith.select(is_tn, c0, vv))

    assert vec == 8, "fp8 store helper hardcoded for VEC=8"
    p0 = arith.constant(0, type=i32)
    p0 = rocdl.cvt_pk_fp8_f32(i32, safe[0], safe[1], p0, 0)
    p0 = rocdl.cvt_pk_fp8_f32(i32, safe[2], safe[3], p0, 1)
    p1 = arith.constant(0, type=i32)
    p1 = rocdl.cvt_pk_fp8_f32(i32, safe[4], safe[5], p1, 0)
    p1 = rocdl.cvt_pk_fp8_f32(i32, safe[6], safe[7], p1, 1)

    off_bytes = row_base_bytes + ArithValue(idx) * c8
    vec2_i32 = T.vec(2, i32)
    store_vec = vector.from_elements(vec2_i32, [p0, p1])
    buffer_ops.buffer_store(store_vec, out_rsrc, off_bytes, offset_is_bytes=True)


# ===========================================================================
# Kernel builder
# ===========================================================================
def _build_kernel(
    *,
    num_q_heads: int,
    head_dim: int,
    rope_head_dim: int,
    quant: bool,
    group_size: int,
    scale_dtype: str,
    q_weighted: bool,
):
    """Build the @flyc.kernel + @flyc.jit launcher for a given config.

    All shape constants are captured via closure (NOT module globals), so two
    launchers with different configs coexist safely. Returns the launcher.
    """
    H = num_q_heads
    D = head_dim
    RD = rope_head_dim
    NOPE = D - RD
    VEC = D // BLOCK_THREADS
    ROPE_THREAD_LO = NOPE // VEC
    PAIRS_PER_THREAD = VEC // 2

    assert (
        D % BLOCK_THREADS == 0
    ), f"D={D} must be divisible by BLOCK_THREADS={BLOCK_THREADS}"
    assert NOPE % VEC == 0, f"NOPE={NOPE} must be divisible by VEC={VEC}"
    assert RD % 2 == 0, "rope_head_dim must be even (GPT-J pair layout)"
    assert RD % VEC == 0, f"RD={RD} must be divisible by VEC={VEC}"
    # Hard-wired to VEC=8 (= D=512 with BLOCK_THREADS=64).
    assert VEC == 8, (
        f"VEC={VEC} unsupported (D={D}); only D=512 / VEC=8 is implemented. "
        "Atom widths and fp8 packing assume VEC=8 - generalising requires "
        "a wider refactor."
    )

    # --- quant-group layout ---
    assert (
        group_size > 0 and D % group_size == 0
    ), f"group_size {group_size} must divide head_dim {D}"
    assert (
        group_size % VEC == 0
    ), f"group_size {group_size} must be a multiple of VEC {VEC}"
    TPG = group_size // VEC  # threads per group
    NG = D // group_size  # number of groups per row
    assert (
        TPG > 0 and (TPG & (TPG - 1)) == 0
    ), f"TPG {TPG} must be a power of 2 (for butterfly reduce)"
    assert (
        scale_dtype in SCALE_DTYPE_OPTIONS
    ), f"scale_dtype {scale_dtype!r} must be one of {SCALE_DTYPE_OPTIONS}"

    log2_block = int(math.log2(BLOCK_THREADS))
    log2_tpg = int(math.log2(TPG))
    amax_start_step = log2_block - log2_tpg

    elem_dtype = fx.BFloat16
    is_e8m0 = scale_dtype == SCALE_DTYPE_E8M0

    _name_parts = ["qk_norm_rope", f"H{H}", f"D{D}", f"RD{RD}"]
    if q_weighted:
        _name_parts.append("qw")
    if quant:
        _name_parts.append(f"g{group_size}")
        _name_parts.append(scale_dtype)
    _name_parts.append("flydsl")
    _kname = "_".join(_name_parts)

    @flyc.kernel(name=_kname)
    def kernel(
        q_in: fx.Pointer,  # [T, H, D]         bf16, contig (H, D)
        kv_in: fx.Pointer,  # [T, D]            bf16, may be strided
        q_weight: fx.Tensor,  # [D]             bf16 (dummy when not q_weighted)
        kv_weight: fx.Tensor,  # [D]            bf16
        cos_cache: fx.Tensor,  # [max_pos, RD/2]  bf16
        sin_cache: fx.Tensor,  # [max_pos, RD/2]  bf16
        positions: fx.Pointer,  # [T]            i64
        q_out: fx.Pointer,  # [T, H, D]         bf16 or fp8
        kv_out: fx.Pointer,  # [T, D]           bf16 or fp8
        q_scale: fx.Pointer,  # [T, H, NG]       f32 or uint8 (e8m0)
        kv_scale: fx.Pointer,  # [T, NG]         f32 or uint8 (e8m0)
        kv_in_row_stride: Int32,  # KV row stride in bf16 elements
    ):
        f32 = T.f32
        i32 = T.i32
        fm_fast = arith.FastMathFlags.fast

        full_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), 16)
        rope_atom = fx.make_copy_atom(fx.rocdl.BufferCopy(64), 16)
        full_lay = fx.make_layout(VEC, 1)
        rope_lay = fx.make_layout(PAIRS_PER_THREAD, 1)

        def load_vec(
            div_tensor, idx, *, layout=full_lay, atom=full_atom, dt=elem_dtype
        ):
            r = fx.make_rmem_tensor(layout, dt)
            fx.copy_atom_call(atom, fx.slice(div_tensor, (None, idx)), r)
            return fx.memref_load_vec(r)

        bid_x = fx.block_idx.x  # 0..H-1 (Q head) or H (KV)
        bid_t = fx.block_idx.y  # token id (chunked at MAX_GRID_Y per launch)
        tid = fx.thread_idx.x
        bid_t_idx = arith.index_cast(T.index, _to_raw(bid_t))

        def _ptr_buffer_resource(ptr, num_records_bytes=None):
            addr = fx.ptrtoint(ptr)
            addr_i64 = arith.index_cast(T.i64, addr)
            if num_records_bytes is None:
                return buffer_ops.create_buffer_resource_from_addr(addr_i64)
            return buffer_ops.create_buffer_resource_from_addr(
                addr_i64, num_records_bytes=num_records_bytes
            )

        # --- shared: load position (i64 -> i32) ---
        pos_rsrc = _ptr_buffer_resource(positions)
        pos_val_i64 = buffer_ops.buffer_load(pos_rsrc, bid_t, vec_width=1, dtype=T.i64)
        pos_i32 = arith.trunci(i32, pos_val_i64)

        # --- shared: cos/sin buffer tensors (used by rope-threads only) ---
        cos_buf = fx.rocdl.make_buffer_tensor(cos_cache)
        sin_buf = fx.rocdl.make_buffer_tensor(sin_cache)
        cos_row = fx.slice(cos_buf, (pos_i32, None))
        sin_row = fx.slice(sin_buf, (pos_i32, None))
        cos_div = fx.logical_divide(cos_row, rope_lay)
        sin_div = fx.logical_divide(sin_row, rope_lay)

        def wave_reduce_add(x):
            w = _to_raw(x)
            for sh_exp in range_constexpr(int(math.log2(BLOCK_THREADS))):
                off = BLOCK_THREADS // (2 << sh_exp)
                peer = _to_raw(ArithValue(w).shuffle_xor(off, BLOCK_THREADS))
                w = arith.AddFOp(w, peer, fastmath=fm_fast).result
            return w

        def emit_body(
            *,
            weighted: bool,
            x_f32_vec,
            w_f32_vec,  # None for Q
            bf16_out_g,  # GTensor with per-token shifted base (when not quant)
            bf16_out_row_off,  # i32 element offset of this head's row in token
            fp8_out_rsrc,  # (rsrc_token_shifted, row_base_bytes) when quant
            scale_rsrc,
            scale_base_off,  # base elem-offset; per-lane adds (tid // TPG)
        ):
            """RMSNorm + GPT-J RoPE (+ optional FP8 quant) for this block's row."""
            x2 = x_f32_vec * x_f32_vec
            sq_local = x2.reduce(ReductionOp.ADD, fastmath=fm_fast)

            if const_expr(quant):
                if const_expr(weighted):
                    xw = x_f32_vec * w_f32_vec
                    am_local = fmath.absf(xw).reduce(ReductionOp.MAX)
                else:
                    am_local = fmath.absf(x_f32_vec).reduce(ReductionOp.MAX)

                # Fused wave reduce: interleave sumsq-ADD (scope = full row D)
                # and amax-MAX (scope = one quant group, TPG threads) so the
                # LLVM scheduler can overlap the two shuffle chains. amax only
                # shuffles in the tail steps where offset < TPG.
                w_sq = _to_raw(sq_local)
                w_am = _to_raw(am_local)
                for sh_exp in range_constexpr(log2_block):
                    off = BLOCK_THREADS // (2 << sh_exp)
                    peer_sq = _to_raw(ArithValue(w_sq).shuffle_xor(off, BLOCK_THREADS))
                    w_sq = arith.AddFOp(w_sq, peer_sq, fastmath=fm_fast).result
                    if const_expr(sh_exp >= amax_start_step):
                        peer_am = _to_raw(
                            ArithValue(w_am).shuffle_xor(off, BLOCK_THREADS)
                        )
                        w_am = arith.maximumf(w_am, peer_am)
                sq_block = w_sq
                am_group = w_am
            else:
                sq_block = wave_reduce_add(sq_local)

            rstd = fmath.rsqrt(sq_block * (1.0 / D) + 1e-6, fastmath=fm_fast)

            if const_expr(quant):
                am_safe = arith.maximumf(am_group, arith.constant(1e-12, type=f32))

                if const_expr(is_e8m0):
                    c_sqrt2 = arith.constant(_SQRT2, type=f32)
                    amax_post = am_safe * rstd * c_sqrt2

                    amax_i32 = amax_post.bitcast(T.i32)
                    bits_up = (
                        amax_i32 + arith.constant(0x400000, type=T.i32)
                    ) & arith.constant(0xFF800000, type=T.i32)
                    exp_field = bits_up >> arith.constant(23, type=T.i32)
                    e8m0_biased_signed = exp_field - arith.constant(
                        _E8M0_HEADROOM, type=T.i32
                    )
                    e8m0_biased = arith.maxsi(
                        e8m0_biased_signed, arith.constant(0, type=T.i32)
                    )
                    e8m0_biased = arith.minsi(
                        e8m0_biased, arith.constant(255, type=T.i32)
                    )
                    quant_exp = arith.constant(254, type=T.i32) - e8m0_biased
                    quant_scale = (quant_exp << arith.constant(23, type=T.i32)).bitcast(
                        T.f32
                    )
                    factor = rstd * quant_scale
                else:
                    rcp_am = llvm.call_intrinsic(
                        f32, "llvm.amdgcn.rcp.f32", [am_safe], [], []
                    )
                    _fc = _fp8_const()
                    factor = arith.constant(_fc["max_over_sqrt2"], type=f32) * rcp_am
                    scale_val = (
                        am_safe * rstd * arith.constant(_fc["inv_max_sqrt2"], type=f32)
                    )

                group_idx = tid >> fx.Int32(log2_tpg)
                lane_in_group = tid & fx.Int32(TPG - 1)
                if lane_in_group == 0:
                    my_scale_off = scale_base_off + ArithValue(group_idx)
                    if const_expr(is_e8m0):
                        e8m0_i8 = arith.TruncIOp(T.i8, e8m0_biased).result
                        buffer_ops.buffer_store(e8m0_i8, scale_rsrc, my_scale_off)
                    else:
                        buffer_ops.buffer_store(scale_val, scale_rsrc, my_scale_off)

            is_rope = tid >= fx.Int32(ROPE_THREAD_LO)
            if is_rope:
                # ---- ROPE path: 8 elements = 4 GPT-J pairs ----
                rope_rel = tid - fx.Int32(ROPE_THREAD_LO)
                cos_vec = load_vec(cos_div, rope_rel, layout=rope_lay, atom=rope_atom)
                sin_vec = load_vec(sin_div, rope_rel, layout=rope_lay, atom=rope_atom)
                cos_f32 = cos_vec.to(fx.Float32)
                sin_f32 = sin_vec.to(fx.Float32)

                pe = []
                for vi in range_constexpr(VEC):
                    xi = x_f32_vec[vi]
                    if const_expr(weighted):
                        xi = xi * w_f32_vec[vi]
                    if const_expr(quant):
                        pe.append(xi * factor)
                    else:
                        pe.append(xi * rstd)

                # GPT-J pair rotate: new_2k = e*c - o*s; new_2k+1 = e*s + o*c
                rope_out = []
                for k in range_constexpr(PAIRS_PER_THREAD):
                    e = pe[2 * k]
                    o = pe[2 * k + 1]
                    c = cos_f32[k]
                    s = sin_f32[k]
                    rope_out.append(e * c - o * s)
                    rope_out.append(e * s + o * c)

                if const_expr(quant):
                    rsrc, row_base = fp8_out_rsrc
                    _store_fp8_packed(rope_out, rsrc, row_base, tid, VEC)
                else:
                    _store_bf16_vec_g(rope_out, bf16_out_g, bf16_out_row_off, tid, VEC)
            else:
                # ---- NOPE path: direct scaled store ----
                scaled = []
                for vi in range_constexpr(VEC):
                    xi = x_f32_vec[vi]
                    if const_expr(weighted):
                        xi = xi * w_f32_vec[vi]
                    if const_expr(quant):
                        scaled.append(xi * factor)
                    else:
                        scaled.append(xi * rstd)
                if const_expr(quant):
                    rsrc, row_base = fp8_out_rsrc
                    _store_fp8_packed(scaled, rsrc, row_base, tid, VEC)
                else:
                    _store_bf16_vec_g(scaled, bf16_out_g, bf16_out_row_off, tid, VEC)

        # ============ runtime dispatch on bid_x < H ============
        q_tok_off_bytes = arith.MulIOp(
            bid_t_idx, arith.constant(H * D * 2, type=T.index)
        ).result

        if bid_x < fx.Int32(H):
            # ---------- Q path ----------
            head_idx = bid_x
            q_in_tok = GTensor(
                q_in,
                dtype=T.bf16,
                shape=(H, D),
                static_bytes_offset_i64=q_tok_off_bytes,
            )
            q_my_off = ArithValue(head_idx) * arith.constant(D, type=i32) + ArithValue(
                tid
            ) * arith.constant(VEC, type=i32)
            raw_x_vec = q_in_tok.load(q_my_off, vec_size=VEC)
            q_rmem = fx.make_rmem_tensor(full_lay, elem_dtype)
            fx.memref_store_vec(raw_x_vec, q_rmem)
            x_vec = fx.memref_load_vec(q_rmem)
            x_f32 = x_vec.to(fx.Float32)

            if const_expr(q_weighted):
                qw_buf = fx.rocdl.make_buffer_tensor(q_weight)
                qw_div = fx.logical_divide(qw_buf, full_lay)
                qw_vec = load_vec(qw_div, tid)
                qw_f32 = qw_vec.to(fx.Float32)
            else:
                qw_f32 = None

            row_off_q_elems = ArithValue(head_idx) * arith.constant(D, type=i32)
            if const_expr(quant):
                q_tok_off_fp8 = arith.MulIOp(
                    bid_t_idx, arith.constant(H * D, type=T.index)
                ).result
                qo_g_tmp = GTensor(
                    q_out,
                    dtype=T.i8,
                    shape=(H, D),
                    static_bytes_offset_i64=q_tok_off_fp8,
                )
                qo_rsrc = qo_g_tmp.rsrc
                row_base_bytes = ArithValue(head_idx) * arith.constant(D, type=i32)
                qs_rsrc = _ptr_buffer_resource(q_scale)
                scale_base_off_q = ArithValue(bid_t) * arith.constant(
                    H * NG, type=i32
                ) + ArithValue(head_idx) * arith.constant(NG, type=i32)
                emit_body(
                    weighted=q_weighted,
                    x_f32_vec=x_f32,
                    w_f32_vec=qw_f32,
                    bf16_out_g=None,
                    bf16_out_row_off=None,
                    fp8_out_rsrc=(qo_rsrc, row_base_bytes),
                    scale_rsrc=qs_rsrc,
                    scale_base_off=scale_base_off_q,
                )
            else:
                qo_g = GTensor(
                    q_out,
                    dtype=T.bf16,
                    shape=(H, D),
                    static_bytes_offset_i64=q_tok_off_bytes,
                )
                emit_body(
                    weighted=q_weighted,
                    x_f32_vec=x_f32,
                    w_f32_vec=qw_f32,
                    bf16_out_g=qo_g,
                    bf16_out_row_off=row_off_q_elems,
                    fp8_out_rsrc=None,
                    scale_rsrc=None,
                    scale_base_off=None,
                )
        else:
            # ---------- KV path ----------
            # KV is often a strided slice of a wider tensor (V4: kv = split of
            # qkv_a). Use raw buffer_ops with the explicit kv_in_row_stride.
            kv_rsrc = _ptr_buffer_resource(kv_in)
            kv_off_elems = ArithValue(bid_t) * ArithValue(
                kv_in_row_stride
            ) + ArithValue(tid) * arith.constant(VEC, type=i32)
            kv_off_dw = kv_off_elems >> arith.constant(1, type=i32)
            vec_bf16xV = T.vec(VEC, T.bf16)
            x_raw = buffer_ops.buffer_load(
                kv_rsrc, kv_off_dw, vec_width=VEC // 2, dtype=i32
            )
            x_vec_bf16_raw = vector.bitcast(vec_bf16xV, x_raw)
            kv_rmem = fx.make_rmem_tensor(full_lay, elem_dtype)
            fx.memref_store_vec(x_vec_bf16_raw, kv_rmem)
            x_vec = fx.memref_load_vec(kv_rmem)

            kvw_buf = fx.rocdl.make_buffer_tensor(kv_weight)
            w_div = fx.logical_divide(kvw_buf, full_lay)
            w_vec = load_vec(w_div, tid)
            x_f32 = x_vec.to(fx.Float32)
            w_f32 = w_vec.to(fx.Float32)

            if const_expr(quant):
                kv_tok_off_fp8 = arith.MulIOp(
                    bid_t_idx, arith.constant(D, type=T.index)
                ).result
                kvo_g_tmp = GTensor(
                    kv_out,
                    dtype=T.i8,
                    shape=(D,),
                    static_bytes_offset_i64=kv_tok_off_fp8,
                )
                kvo_rsrc = kvo_g_tmp.rsrc
                row_base_bytes = arith.constant(0, type=i32)
                kvs_rsrc = _ptr_buffer_resource(kv_scale)
                scale_base_off_kv = ArithValue(bid_t) * arith.constant(NG, type=i32)
                emit_body(
                    weighted=True,
                    x_f32_vec=x_f32,
                    w_f32_vec=w_f32,
                    bf16_out_g=None,
                    bf16_out_row_off=None,
                    fp8_out_rsrc=(kvo_rsrc, row_base_bytes),
                    scale_rsrc=kvs_rsrc,
                    scale_base_off=scale_base_off_kv,
                )
            else:
                kv_tok_off_bf16 = arith.MulIOp(
                    bid_t_idx, arith.constant(D * 2, type=T.index)
                ).result
                kvo_g = GTensor(
                    kv_out,
                    dtype=T.bf16,
                    shape=(D,),
                    static_bytes_offset_i64=kv_tok_off_bf16,
                )
                emit_body(
                    weighted=True,
                    x_f32_vec=x_f32,
                    w_f32_vec=w_f32,
                    bf16_out_g=kvo_g,
                    bf16_out_row_off=arith.constant(0, type=i32),
                    fp8_out_rsrc=None,
                    scale_rsrc=None,
                    scale_base_off=None,
                )

    @flyc.jit
    def launch_qk_norm_rope_quant(
        q_in: fx.Pointer,
        kv_in: fx.Pointer,
        q_weight: fx.Tensor,
        kv_weight: fx.Tensor,
        cos_cache: fx.Tensor,
        sin_cache: fx.Tensor,
        positions: fx.Pointer,
        q_out: fx.Pointer,
        kv_out: fx.Pointer,
        q_scale: fx.Pointer,
        kv_scale: fx.Pointer,
        kv_in_row_stride: fx.Int32,
        num_tokens: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        idx_tokens = arith.index_cast(T.index, _to_raw(num_tokens))
        k = kernel(
            q_in,
            kv_in,
            q_weight,
            kv_weight,
            cos_cache,
            sin_cache,
            positions,
            q_out,
            kv_out,
            q_scale,
            kv_scale,
            kv_in_row_stride,
        )
        k.launch(
            grid=(H + 1, idx_tokens, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_qk_norm_rope_quant


# ===========================================================================
# Cached compile + build entrypoint (host-side)
# ===========================================================================
# waves_per_eu=8 + fast/unsafe fp math: best occupancy at small/mid T with no
# measurable regression at large T (matches the AITER op default).
_DEFAULT_COMPILE_HINTS = {
    "waves_per_eu": 8,
    "fast_fp_math": True,
    "unsafe_fp_math": True,
}


@lru_cache(maxsize=32)
def compile_flydsl_qk_norm_rope_quant(
    *,
    num_q_heads: int,
    head_dim: int,
    rope_head_dim: int,
    quant: bool,
    group_size: int,
    scale_dtype: str,
    q_weighted: bool,
):
    """Compile (and cache) the inline launcher for a given config."""
    launcher = _build_kernel(
        num_q_heads=num_q_heads,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        quant=quant,
        group_size=group_size,
        scale_dtype=scale_dtype,
        q_weighted=q_weighted,
    )
    launcher.compile_hints = dict(_DEFAULT_COMPILE_HINTS)
    return launcher


def build_qk_norm_rope_quant_module(
    *,
    num_q_heads: int,
    head_dim: int,
    rope_head_dim: int,
    quant: bool = False,
    group_size: int = 64,
    scale_dtype: str = SCALE_DTYPE_FP32,
    q_weighted: bool = False,
):
    """Build (and cache) one inline FlyDSL QK-norm+RoPE(+quant) launcher.

    This is the build entrypoint invoked by ``config.yaml``'s
    ``compile_command``. It constructs the @flyc.kernel / @flyc.jit launcher
    (validating the config via the build-time asserts in ``_build_kernel``);
    the device binary is JIT-compiled on first launch.
    """
    return compile_flydsl_qk_norm_rope_quant(
        num_q_heads=num_q_heads,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        quant=quant,
        group_size=group_size,
        scale_dtype=scale_dtype,
        q_weighted=q_weighted,
    )


# ===========================================================================
# Host launcher
# ===========================================================================
def flydsl_qk_norm_rope_quant(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_weight: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    num_q_heads: int,
    head_dim: int,
    rope_head_dim: int,
    q_weight: Optional[torch.Tensor] = None,
    quant: bool = False,
    quant_group_size: Optional[int] = None,
    scale_dtype: str = SCALE_DTYPE_FP32,
    q_out: Optional[torch.Tensor] = None,
    kv_out: Optional[torch.Tensor] = None,
    q_scale: Optional[torch.Tensor] = None,
    kv_scale: Optional[torch.Tensor] = None,
    stream: Optional[torch.cuda.Stream] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Fused RMSNorm + GPT-J RoPE + optional FP8 quant for Q and KV in one launch.

    ``quant=False`` (default) writes bf16 and returns
    ``(q_out, kv_out, None, None)`` — this is the path the ``Model`` reference
    and the tight correctness gate use. ``quant=True`` writes fp8 (per-GFX
    native, e4m3fn on gfx950) with one ``quant_group_size``-wide block scale
    (fp32 or e8m0) per group and returns the scales — a separate capability,
    not part of the bf16 gate.

    See ``model.py`` for the I/O layout. ``q`` is ``[T, H*D]`` (or ``[T,H,D]``),
    ``kv`` is ``[T, D]`` (may be a strided slice; row stride read from
    ``kv.stride(0)``), cos/sin last dim is ``RD/2``, positions int64 ``[T]``.
    """
    if q.dtype != torch.bfloat16:
        raise TypeError(f"q must be bf16, got {q.dtype}")
    if kv.dtype != torch.bfloat16:
        raise TypeError(f"kv must be bf16, got {kv.dtype}")
    if kv_weight.dtype != torch.bfloat16:
        raise TypeError(f"kv_weight must be bf16, got {kv_weight.dtype}")
    if kv.stride(-1) != 1:
        raise ValueError(f"kv must be dense in the last dim, stride={kv.stride()}")
    if kv.stride(0) % 2 != 0:
        raise ValueError(
            "kv row stride (in bf16 elements) must be even for dword-cast "
            f"buffer loads, got kv.stride(0)={kv.stride(0)}"
        )
    if positions.dtype != torch.int64:
        raise TypeError(f"positions must be int64, got {positions.dtype}")
    if scale_dtype not in SCALE_DTYPE_OPTIONS:
        raise ValueError(f"scale_dtype {scale_dtype!r} not in {SCALE_DTYPE_OPTIONS}")
    if q_weight is not None and q_weight.dtype != torch.bfloat16:
        raise TypeError(f"q_weight must be bf16, got {q_weight.dtype}")

    H, D, RD = num_q_heads, head_dim, rope_head_dim
    T_tok = q.shape[0]
    G = quant_group_size if quant_group_size is not None else D
    NG = D // G
    if D % G != 0:
        raise ValueError(f"head_dim {D} must be divisible by quant_group_size {G}")
    q_weighted = q_weight is not None
    q_weight_arg = q_weight if q_weighted else kv_weight

    # Normalize Q to [T, H, D].
    if q.dim() == 2:
        if q.shape[1] != H * D:
            raise ValueError(f"q shape {tuple(q.shape)} != [T, H*D={H * D}]")
        if not q.is_contiguous():
            raise ValueError("2D q must be contiguous to .view as [T,H,D]")
        q_view = q.view(T_tok, H, D)
    else:
        if q.dim() != 3 or q.shape != (T_tok, H, D):
            raise ValueError(
                f"q shape {tuple(q.shape)} != (T, H, D)=({T_tok}, {H}, {D})"
            )
        q_view = q
        if q_view.stride(-1) != 1 or q_view.stride(-2) != D:
            raise ValueError(
                "3D q must be contiguous in the (H, D) inner block "
                f"(stride(-1)==1 and stride(-2)==D={D}), got stride={q_view.stride()}"
            )

    # Normalize cos/sin to 2D [max_pos, RD/2].
    if cos_cache.shape[-1] != RD // 2:
        raise ValueError(f"cos_cache last dim {cos_cache.shape[-1]} != RD/2 ({RD // 2})")
    if sin_cache.shape != cos_cache.shape:
        raise ValueError("cos/sin shape mismatch")
    if not (cos_cache.is_contiguous() and sin_cache.is_contiguous()):
        raise ValueError("cos/sin must be contiguous")
    cos_2d = cos_cache.view(cos_cache.shape[0], RD // 2)
    sin_2d = sin_cache.view(sin_cache.shape[0], RD // 2)

    out_dtype = _fp8_const()["dtype"] if quant else torch.bfloat16
    if q_out is None:
        q_out = torch.empty((T_tok, H, D), dtype=out_dtype, device=q.device)
    if kv_out is None:
        kv_out = torch.empty((T_tok, D), dtype=out_dtype, device=kv.device)

    scale_torch_dtype = _TORCH_DTYPE_FOR_SCALE[scale_dtype]
    if quant:
        if q_scale is None:
            q_scale = torch.empty((T_tok, H, NG), dtype=scale_torch_dtype, device=q.device)
        if kv_scale is None:
            kv_scale = torch.empty((T_tok, NG), dtype=scale_torch_dtype, device=kv.device)
        q_scale_arg, kv_scale_arg = q_scale, kv_scale
    else:
        q_scale_arg = q.new_empty(1, dtype=scale_torch_dtype)
        kv_scale_arg = q.new_empty(1, dtype=scale_torch_dtype)

    launcher = compile_flydsl_qk_norm_rope_quant(
        num_q_heads=H,
        head_dim=D,
        rope_head_dim=RD,
        quant=quant,
        group_size=G,
        scale_dtype=scale_dtype,
        q_weighted=q_weighted,
    )

    if stream is None:
        stream = torch.cuda.current_stream()
    fx_stream = Stream(stream)

    def _ptr_arg(t):
        return flyc.from_c_void_p(fx.Uint8, t.data_ptr())

    q_weight_static = flyc.from_dlpack(q_weight_arg)
    kv_weight_static = flyc.from_dlpack(kv_weight)
    cos_static = flyc.from_dlpack(cos_2d)
    sin_static = flyc.from_dlpack(sin_2d)

    # HW grid Y is a 16-bit field on AMD HIP -> cap 65535 blocks/launch.
    MAX_GRID_Y = 65535
    for start in range(0, T_tok, MAX_GRID_Y):
        n = min(MAX_GRID_Y, T_tok - start)
        end = start + n
        launcher(
            _ptr_arg(q_view[start:end]),
            _ptr_arg(kv[start:end]),
            q_weight_static,
            kv_weight_static,
            cos_static,
            sin_static,
            _ptr_arg(positions[start:end]),
            _ptr_arg(q_out[start:end]),
            _ptr_arg(kv_out[start:end]),
            _ptr_arg(q_scale_arg[start:end] if quant else q_scale_arg),
            _ptr_arg(kv_scale_arg[start:end] if quant else kv_scale_arg),
            kv.stride(0),
            n,
            stream=fx_stream,
        )

    return q_out, kv_out, (q_scale if quant else None), (kv_scale if quant else None)
