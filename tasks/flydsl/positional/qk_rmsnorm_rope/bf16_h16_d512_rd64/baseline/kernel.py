# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL fused QK-RMSNorm + GPT-J RoPE (bfloat16), after ROCm/aiter
aiter/ops/flydsl/kernels/qk_norm_rope_quant.py (quant=False path only).

Per-row RMSNorm (scope = head_dim D, eps = 1e-6, per-channel gamma on KV only) + GPT-J
pair-interleaved RoPE on the last ``rope_head_dim`` (RD) elements, for Q ``[T, H, D]`` and
KV ``[T, D]`` in one kernel launch.

Grid: X = num_q_heads + 1 (``bid_x < H`` handle Q heads, ``bid_x == H`` handles KV), Y = tokens
(chunked at the 65535 HW grid-Y limit). One wave64 per block: thread ``t`` owns ``VEC = D // 64``
elements, so the RMSNorm reduction is wave-local (``shuffle_xor`` butterfly, no LDS, no barrier).

# NOTE: do NOT add `from __future__ import annotations` to this file. PEP 563
# stringifies annotations, defeating flydsl's JitFunction._make_cache_key
# runtime detection (is_runtime = hasattr(ann, "__get_c_pointers__")).
"""

from functools import lru_cache
import math

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, ptrtoint, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import Int32, Stream, T
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly, llvm
from flydsl.compiler.protocol import extract_to_ir_values

from . import buffer_ops, vector

ReductionOp = vector.ReductionOp




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
            strides = []
            running = 1
            for extent in reversed(tuple(shape)):
                strides.append(running)
                running *= extent
            self.stride = tuple(reversed(strides))
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



BLOCK_THREADS = 64  # 1 wave64


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



# ===========================================================================
# Kernel builder
# ===========================================================================


# ===========================================================================
# Kernel builder
# ===========================================================================
def _build_kernel(*, num_q_heads: int, head_dim: int, rope_head_dim: int):
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
    assert VEC == 8, f"VEC={VEC} unsupported (D={D}); only D=512 / VEC=8 is implemented."

    elem_dtype = fx.BFloat16

    _kname = "_".join(["qk_norm_rope", f"H{H}", f"D{D}", f"RD{RD}", "flydsl"])

    @flyc.kernel(name=_kname)
    def kernel(
        q_in: fx.Pointer,  # [T, H, D]         bf16, contig (H, D)
        kv_in: fx.Pointer,  # [T, D]            bf16, may be strided
        kv_weight: fx.Tensor,  # [D]            bf16
        cos_cache: fx.Tensor,  # [max_pos, RD/2]  bf16
        sin_cache: fx.Tensor,  # [max_pos, RD/2]  bf16
        positions: fx.Pointer,  # [T]            i64
        q_out: fx.Pointer,  # [T, H, D]         bf16
        kv_out: fx.Pointer,  # [T, D]           bf16
        kv_in_row_stride: Int32,  # KV row stride in bf16 elements
    ):
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

        def _ptr_buffer_resource(ptr):
            addr = fx.ptrtoint(ptr)
            addr_i64 = arith.index_cast(T.i64, addr)
            return buffer_ops.create_buffer_resource_from_addr(addr_i64)

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
            bf16_out_g,  # GTensor with per-token shifted base
            bf16_out_row_off,  # i32 element offset of this head's row in token
        ):
            """RMSNorm + GPT-J RoPE for this block's row."""
            x2 = x_f32_vec * x_f32_vec
            sq_local = x2.reduce(ReductionOp.ADD, fastmath=fm_fast)
            sq_block = wave_reduce_add(sq_local)

            rstd = fmath.rsqrt(sq_block * (1.0 / D) + 1e-6, fastmath=fm_fast)

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

                _store_bf16_vec_g(rope_out, bf16_out_g, bf16_out_row_off, tid, VEC)
            else:
                # ---- NOPE path: direct scaled store ----
                scaled = []
                for vi in range_constexpr(VEC):
                    xi = x_f32_vec[vi]
                    if const_expr(weighted):
                        xi = xi * w_f32_vec[vi]
                    scaled.append(xi * rstd)
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

            row_off_q_elems = ArithValue(head_idx) * arith.constant(D, type=i32)
            qo_g = GTensor(
                q_out,
                dtype=T.bf16,
                shape=(H, D),
                static_bytes_offset_i64=q_tok_off_bytes,
            )
            emit_body(
                weighted=False,
                x_f32_vec=x_f32,
                w_f32_vec=None,
                bf16_out_g=qo_g,
                bf16_out_row_off=row_off_q_elems,
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
            )

    @flyc.jit
    def launch_qk_norm_rope(
        q_in: fx.Pointer,
        kv_in: fx.Pointer,
        kv_weight: fx.Tensor,
        cos_cache: fx.Tensor,
        sin_cache: fx.Tensor,
        positions: fx.Pointer,
        q_out: fx.Pointer,
        kv_out: fx.Pointer,
        kv_in_row_stride: fx.Int32,
        num_tokens: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        idx_tokens = arith.index_cast(T.index, _to_raw(num_tokens))
        k = kernel(
            q_in,
            kv_in,
            kv_weight,
            cos_cache,
            sin_cache,
            positions,
            q_out,
            kv_out,
            kv_in_row_stride,
        )
        k.launch(
            grid=(H + 1, idx_tokens, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_qk_norm_rope


# waves_per_eu=8 + fast/unsafe fp math: best occupancy at small/mid T with no
# measurable regression at large T (matches the AITER op default).
_DEFAULT_COMPILE_HINTS = {
    "waves_per_eu": 8,
    "fast_fp_math": True,
    "unsafe_fp_math": True,
}

# HW grid Y is a 16-bit field on AMD HIP -> cap 65535 blocks/launch.
_MAX_GRID_Y = 65535


@lru_cache(maxsize=32)
def _launcher(num_q_heads: int, head_dim: int, rope_head_dim: int):
    """Build (and cache) the launcher for a given config."""
    launcher = _build_kernel(
        num_q_heads=num_q_heads, head_dim=head_dim, rope_head_dim=rope_head_dim
    )
    launcher.compile_hints = dict(_DEFAULT_COMPILE_HINTS)
    return launcher


def run(*, q, kv, kv_weight, cos_cache, sin_cache, positions, q_out, kv_out):
    """Fused per-head RMSNorm + GPT-J RoPE of Q [T, H, D] and KV [T, D] into q_out / kv_out."""
    num_tokens, num_q_heads, head_dim = q.shape
    rope_head_dim = 2 * cos_cache.shape[-1]
    launcher = _launcher(num_q_heads, head_dim, rope_head_dim)
    fx_stream = Stream(torch.cuda.current_stream())

    def _ptr_arg(t):
        return flyc.from_c_void_p(fx.Uint8, t.data_ptr())

    kv_weight_static = flyc.from_dlpack(kv_weight)
    cos_static = flyc.from_dlpack(cos_cache)
    sin_static = flyc.from_dlpack(sin_cache)

    for start in range(0, num_tokens, _MAX_GRID_Y):
        n = min(_MAX_GRID_Y, num_tokens - start)
        end = start + n
        launcher(
            _ptr_arg(q[start:end]),
            _ptr_arg(kv[start:end]),
            kv_weight_static,
            cos_static,
            sin_static,
            _ptr_arg(positions[start:end]),
            _ptr_arg(q_out[start:end]),
            _ptr_arg(kv_out[start:end]),
            kv.stride(0),
            n,
            stream=fx_stream,
        )
