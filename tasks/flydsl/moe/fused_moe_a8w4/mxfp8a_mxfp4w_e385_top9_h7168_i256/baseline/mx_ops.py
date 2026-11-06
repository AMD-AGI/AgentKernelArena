# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# The kernel bodies follow ROCm/FlyDSL kernels/silu_and_mul_fq.py (MIT, Copyright (C) 2024-2026 Advanced
# Micro Devices, Inc.): the same buffer, MXFP8 quantization, fp8 packing and sorted-scale code.
"""FlyDSL helper kernels of the a8w4 MoE: MXFP8 quantization, sorted scales and the top-k sum.

``mxfp8_quant_rows``
    Per row and 32-element block: e8m0 = max(exponent(round_half_up(max|x|)) - 8, 0), the codes
    are float8_e4m3fn(x * 2^(127 - e8m0)) (round to nearest even), the block scale 2^(e8m0 - 127).
    This is the MXFP8 rule of the stage-1 GEMM's fused quantization (round the block maximum's
    exponent up at 1.5x, then divide by 2^8 so the largest code is at most 384).
``mx_scale_sort``
    Writes the e8m0 scales of the rows a sorted (token, slot) list refers to into the tiled layout
    the MoE GEMMs read (``moe_mxfp4_sort`` in AITER): for sorted row r and scale column c the byte
    is at (r >> 5) * cols * 32 + (c >> 3) * 256 + (c & 3) * 64 + (r & 15) * 4 + ((c >> 2) & 1) * 2
    + ((r >> 4) & 1); padding rows get scale 0.
``topk_sum``
    out[t] = sum over the top-k slots of the per-slot stage-2 rows, accumulated in float32 in slot
    order and rounded to bfloat16.
"""
import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr.arith import ArithValue, CmpIPredicate
from flydsl.expr.typing import Int32, T

try:
    from flydsl.expr import buffer_ops
except ImportError:  # FlyDSL releases that no longer ship the buffer helpers
    from . import buffer_ops
try:
    from flydsl.expr import vector
except ImportError:  # FlyDSL releases that no longer ship the vector helpers
    from . import vector

WARP_SIZE = 64
QUANT_BLOCK = 32
# 2^8: the e8m0 headroom of e4m3 (max 448) used by the MXFP8 rule above.
FP8_HEADROOM = 8


def _shuffle_dists(vec):
    dists, d = [], 1
    while d < QUANT_BLOCK // vec:
        dists.append(d)
        d *= 2
    return dists


@functools.lru_cache(maxsize=None)
def build_mxfp8_quant_rows(cols: int, block_threads: int, vec: int):
    """Launcher quantizing bf16 rows [rows, cols] to fp8 codes [rows, cols] and int32 e8m0 [rows, cols/32]."""
    assert vec % 4 == 0 and QUANT_BLOCK % vec == 0, vec
    assert block_threads % WARP_SIZE == 0
    assert cols % (block_threads * vec) == 0, (cols, block_threads, vec)
    n_iter = cols // (block_threads * vec)
    scale_cols = cols // QUANT_BLOCK
    vec_dw = vec * 2 // 4
    shuffle_dists = _shuffle_dists(vec)

    @flyc.kernel
    def mxfp8_quant_rows_kernel(
        x: fx.Tensor,
        out_q: fx.Tensor,
        out_scale: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        f32 = T.f32
        i32 = T.i32

        c0_i32 = arith.constant(0, type=i32)
        c1_i32 = arith.constant(1, type=i32)
        c5_i32 = arith.constant(5, type=i32)
        c23_i32 = arith.constant(23, type=i32)
        c31_i32 = arith.constant(31, type=i32)
        c64_i32 = arith.constant(64, type=i32)
        c254_i32 = arith.constant(254, type=i32)
        c0xFF800000_i32 = arith.constant(0xFF800000, type=i32)
        c0x400000_i32 = arith.constant(0x400000, type=i32)
        c0_f32 = arith.constant(0.0, type=f32)
        c_headroom_i32 = arith.constant(FP8_HEADROOM, type=i32)

        in_rsrc = buffer_ops.create_buffer_resource(x, max_size=True)
        q_rsrc = buffer_ops.create_buffer_resource(out_q, max_size=True)
        s_rsrc = buffer_ops.create_buffer_resource(out_scale, max_size=True)

        row = ArithValue(bid)
        thread_id = ArithValue(tid)
        row_elem_base = row * arith.constant(cols, type=i32)
        row_scale_base = row * arith.constant(scale_cols, type=i32)

        vec_bf16_ty = T.vec(vec, T.bf16)
        vec_f32_ty = T.vec(vec, f32)

        for iter_idx in range_constexpr(n_iter):
            col0 = thread_id * arith.constant(vec, type=i32) + arith.constant(
                iter_idx * block_threads * vec, type=i32
            )
            elem = row_elem_base + col0
            # bf16 elements -> dword offset.
            in_dw = elem >> c1_i32
            raw = buffer_ops.buffer_load(in_rsrc, in_dw, vec_width=vec_dw, dtype=i32)
            vals_bf16 = vector.bitcast(vec_bf16_ty, raw)
            vals_f32 = vals_bf16.extf(vec_f32_ty)
            vals = []
            for vi in range_constexpr(vec):
                vals.append(vector.extract(vals_f32, static_position=[vi], dynamic_position=[]))

            local_max = c0_f32
            for vi in range_constexpr(vec):
                abs_v = llvm.call_intrinsic(f32, "llvm.fabs.f32", [vals[vi]], [], [])
                local_max = arith.maximumf(local_max, abs_v)
            for sh_dist in shuffle_dists:
                off = arith.constant(sh_dist, type=i32)
                peer = local_max.shuffle_xor(off, c64_i32)
                local_max = arith.maximumf(local_max, peer)

            max_i32_v = local_max.bitcast(i32)
            # Round the block maximum's exponent up at the 1.5x threshold, then drop the mantissa.
            max_rounded = (max_i32_v + c0x400000_i32) & c0xFF800000_i32
            exp_field = max_rounded >> c23_i32
            e8m0_biased = arith.maxsi(exp_field - c_headroom_i32, c0_i32)
            quant_exp = c254_i32 - e8m0_biased
            quant_scale = (quant_exp << c23_i32).bitcast(f32)

            scaled_vals = []
            for vi in range_constexpr(vec):
                scaled_vals.append(vals[vi] * quant_scale)
            for _wg in range_constexpr(vec // 4):
                _b = _wg * 4
                lo = rocdl.cvt_pk_fp8_f32(i32, scaled_vals[_b], scaled_vals[_b + 1], c0_i32, False)
                packed_w = rocdl.cvt_pk_fp8_f32(i32, scaled_vals[_b + 2], scaled_vals[_b + 3], lo, True)
                word_off = elem + arith.constant(_wg * 4, type=i32)
                buffer_ops.buffer_store(packed_w, q_rsrc, word_off, offset_is_bytes=True)

            # One lane per 32-element block writes its e8m0 byte (as int32).
            lane_in_blk = col0 & c31_i32
            _if_sw = scf.IfOp(arith.cmpi(CmpIPredicate.eq, lane_in_blk, c0_i32))
            with ir.InsertionPoint(_if_sw.then_block):
                buffer_ops.buffer_store(e8m0_biased, s_rsrc, row_scale_base + (col0 >> c5_i32))
                scf.YieldOp([])

    @flyc.jit
    def launch_mxfp8_quant_rows(
        x: fx.Tensor,
        out_q: fx.Tensor,
        out_scale: fx.Tensor,
        num_rows: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            pass

        idx_rows = arith.index_cast(T.index, num_rows)
        launcher = mxfp8_quant_rows_kernel(x, out_q, out_scale)
        launcher.launch(
            grid=(idx_rows, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_mxfp8_quant_rows


@functools.lru_cache(maxsize=None)
def build_mx_scale_sort(scale_cols: int, src_topk: int, topk: int, block_threads: int):
    """Launcher gathering int32 e8m0 rows [src_rows, scale_cols] into the sorted tiled byte layout."""
    assert scale_cols % 8 == 0, scale_cols
    assert block_threads % WARP_SIZE == 0
    n_iter = (scale_cols + block_threads - 1) // block_threads

    @flyc.kernel
    def mx_scale_sort_kernel(
        scale_plain: fx.Tensor,
        sorted_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        out_scale_sorted: fx.Tensor,
        token_num: Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        i32 = T.i32

        c0_i32 = arith.constant(0, type=i32)
        c1_i32 = arith.constant(1, type=i32)
        c2_i32 = arith.constant(2, type=i32)
        c3_i32 = arith.constant(3, type=i32)
        c4_i32 = arith.constant(4, type=i32)
        c5_i32 = arith.constant(5, type=i32)
        c15_i32 = arith.constant(15, type=i32)
        c64_i32 = arith.constant(64, type=i32)
        c256_i32 = arith.constant(256, type=i32)

        scale_cols_i32 = arith.constant(scale_cols, type=i32)
        topk_i32 = arith.constant(topk, type=i32)
        n32_sort = arith.constant(scale_cols * 32, type=i32)

        src_rsrc = buffer_ops.create_buffer_resource(scale_plain, max_size=True)
        tid_rsrc = buffer_ops.create_buffer_resource(sorted_ids, max_size=True)
        nv_rsrc = buffer_ops.create_buffer_resource(num_valid_ids, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(out_scale_sorted, max_size=True)

        num_valid = buffer_ops.buffer_load(nv_rsrc, c0_i32, vec_width=1, dtype=i32)
        token_num_i32 = ArithValue(token_num)
        bid_i32 = ArithValue(bid)

        row_in_range = arith.cmpi(CmpIPredicate.ult, bid_i32, num_valid)
        # The grid covers whole 32-row scale tiles, which can extend past the sorted id list.
        safe_bid = arith.select(row_in_range, bid_i32, c0_i32)
        fused_tid_val = buffer_ops.buffer_load(tid_rsrc, safe_bid, vec_width=1, dtype=i32)
        mask24 = arith.constant(0xFFFFFF, type=i32)
        token_id = fused_tid_val & mask24
        slot_id = ArithValue(fused_tid_val) >> arith.constant(24, type=i32)
        t_ok = arith.cmpi(CmpIPredicate.ult, token_id, token_num_i32)
        s_ok = arith.cmpi(CmpIPredicate.ult, slot_id, topk_i32)
        is_valid = arith.andi(row_in_range, arith.andi(t_ok, s_ok))

        if const_expr(src_topk == 1):
            src_row = token_id
        else:
            src_row = token_id * arith.constant(src_topk, type=i32) + slot_id
        safe_src_row = arith.select(is_valid, src_row, c0_i32)
        src_base = safe_src_row * scale_cols_i32

        row_s = bid_i32
        d0 = row_s >> c5_i32
        d1 = (row_s >> c4_i32) & c1_i32
        d2 = row_s & c15_i32
        row_byte_base = d0 * n32_sort + d2 * c4_i32 + d1

        thread_id = ArithValue(tid)
        for iter_idx in range_constexpr(n_iter):
            col_s = thread_id + arith.constant(iter_idx * block_threads, type=i32)
            col_valid = arith.cmpi(CmpIPredicate.ult, col_s, scale_cols_i32)
            _if_col = scf.IfOp(col_valid)
            with ir.InsertionPoint(_if_col.then_block):
                e8m0 = buffer_ops.buffer_load(src_rsrc, src_base + col_s, vec_width=1, dtype=i32)
                e8m0 = arith.select(is_valid, e8m0, c0_i32)
                d3 = col_s >> c3_i32
                d4 = (col_s >> c2_i32) & c1_i32
                d5 = col_s & c3_i32
                s_byte_off = row_byte_base + d3 * c256_i32 + d5 * c64_i32 + d4 * c2_i32
                e8m0_i8 = arith.TruncIOp(T.i8, e8m0)
                buffer_ops.buffer_store(
                    e8m0_i8,
                    out_rsrc,
                    s_byte_off,
                    offset_is_bytes=True,
                )
                scf.YieldOp([])

    @flyc.jit
    def launch_mx_scale_sort(
        scale_plain: fx.Tensor,
        sorted_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        out_scale_sorted: fx.Tensor,
        token_num: fx.Int32,
        num_sorted_rows: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            pass

        idx_rows = arith.index_cast(T.index, num_sorted_rows)
        launcher = mx_scale_sort_kernel(
            scale_plain,
            sorted_ids,
            num_valid_ids,
            out_scale_sorted,
            token_num,
        )
        launcher.launch(
            grid=(idx_rows, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_mx_scale_sort


@functools.lru_cache(maxsize=None)
def build_topk_sum(cols: int, topk: int, block_threads: int, vec: int):
    """Launcher summing bf16 rows [tokens * topk, cols] over the top-k slots into bf16 [tokens, cols]."""
    assert vec % 2 == 0 and vec >= 4, vec
    assert cols % (block_threads * vec) == 0, (cols, block_threads, vec)
    n_iter = cols // (block_threads * vec)
    vec_dw = vec * 2 // 4

    @flyc.kernel
    def topk_sum_kernel(
        partial: fx.Tensor,
        out: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        f32 = T.f32
        i32 = T.i32
        c1_i32 = arith.constant(1, type=i32)

        in_rsrc = buffer_ops.create_buffer_resource(partial, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(out, max_size=True)

        token = ArithValue(bid)
        thread_id = ArithValue(tid)
        cols_i32 = arith.constant(cols, type=i32)
        in_row_base = token * arith.constant(topk * cols, type=i32)
        out_row_base = token * cols_i32

        vec_bf16_ty = T.vec(vec, T.bf16)
        vec_f32_ty = T.vec(vec, f32)

        for iter_idx in range_constexpr(n_iter):
            col0 = thread_id * arith.constant(vec, type=i32) + arith.constant(
                iter_idx * block_threads * vec, type=i32
            )
            acc = []
            for slot in range_constexpr(topk):
                elem = in_row_base + arith.constant(slot * cols, type=i32) + col0
                raw = buffer_ops.buffer_load(in_rsrc, elem >> c1_i32, vec_width=vec_dw, dtype=i32)
                vals_f32 = vector.bitcast(vec_bf16_ty, raw).extf(vec_f32_ty)
                for vi in range_constexpr(vec):
                    v = vector.extract(vals_f32, static_position=[vi], dynamic_position=[])
                    if const_expr(slot == 0):
                        acc.append(v)
                    else:
                        acc[vi] = acc[vi] + v
            out_vals = []
            for vi in range_constexpr(vec):
                out_vals.append(arith.trunc_f(T.bf16, acc[vi]))
            out_bf16 = vector.from_elements(vec_bf16_ty, out_vals)
            out_i32 = vector.bitcast(T.vec(vec_dw, i32), out_bf16)
            buffer_ops.buffer_store(out_i32, out_rsrc, (out_row_base + col0) >> c1_i32)

    @flyc.jit
    def launch_topk_sum(
        partial: fx.Tensor,
        out: fx.Tensor,
        num_tokens: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            pass

        idx_rows = arith.index_cast(T.index, num_tokens)
        launcher = topk_sum_kernel(partial, out)
        launcher.launch(
            grid=(idx_rows, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_topk_sum


def mxfp8_quant_rows(x, out_q, out_scale, *, block_threads, vec):
    """x: bf16 [rows, cols]; out_q: uint8 [rows, cols]; out_scale: int32 [rows, cols / 32]."""
    rows, cols = x.shape
    launcher = build_mxfp8_quant_rows(int(cols), int(block_threads), int(vec))
    launcher(x, out_q, out_scale, int(rows), torch.cuda.current_stream())


def mx_scale_sort(scale_plain, sorted_ids, num_valid_ids, out_scale_sorted, token_num, num_sorted_rows, *,
                  src_topk, topk, block_threads):
    """scale_plain: int32 [src_rows, cols]; out_scale_sorted: uint8 [>= num_sorted_rows rounded to 32, cols]."""
    scale_cols = scale_plain.shape[-1]
    launcher = build_mx_scale_sort(int(scale_cols), int(src_topk), int(topk), int(block_threads))
    launcher(scale_plain, sorted_ids, num_valid_ids, out_scale_sorted, int(token_num), int(num_sorted_rows),
             torch.cuda.current_stream())


def topk_sum(partial, out, topk, *, block_threads, vec):
    """partial: bf16 [tokens * topk, cols]; out: bf16 [tokens, cols]."""
    tokens, cols = out.shape
    launcher = build_topk_sum(int(cols), int(topk), int(block_threads), int(vec))
    launcher(partial, out, int(tokens), torch.cuda.current_stream())
