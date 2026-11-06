# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""RMSNorm kernel builder using the @flyc.kernel API.

RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma

Two paths:
  - Fast path (N % tile_cols == 0, tile_cols = BLOCK_THREADS * VEC_WIDTH): buffer_load/store vectorised access.
  - Generic path (arbitrary N): scalar copy_atom_call.
"""

import functools
import math

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import Vector as Vec
# FlyDSL 0.3 moved these vector helpers into typing. Preserve the older API
# for the original gfx942 runtime without hiding unrelated import failures.
try:
    from flydsl.expr.vector import ReductionOp, full
except ModuleNotFoundError as exc:
    if exc.name != "flydsl.expr.vector":
        raise
    from flydsl.expr.typing import ReductionOp, full
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from .kernels_common import dtype_to_elem_type, get_warp_size

KERNEL_NAME = "rmsnorm"

EPS = 1e-5

# 128 threads x 8 elements x 7 tiles = 7168 = hidden size, so the vectorised fast path applies.
BLOCK_THREADS = 128
WARP_SIZE = get_warp_size()
VEC_WIDTH = 8


def _make_reduction_storage(red_slots: int):
    @fx.struct
    class SharedStorage:
        s_red: fx.Array[fx.Float32, red_slots, 16]
        s_red2: fx.Array[fx.Float32, red_slots, 16]

    return SharedStorage


def _load_scalar(copy_atom, elem_dtype, divided_tensor, index):
    view = fx.slice(divided_tensor, (None, index))
    r = fx.make_rmem_tensor(1, elem_dtype)
    fx.copy_atom_call(copy_atom, view, r)
    return fx.memref_load_vec(r)[0]


def _store_scalar(copy_atom, elem_dtype, store_dtype, divided_tensor, index, val):
    r = fx.make_rmem_tensor(1, elem_dtype)
    ts = full(1, store_dtype(val), store_dtype)
    fx.memref_store_vec(ts, r)
    view = fx.slice(divided_tensor, (None, index))
    fx.copy_atom_call(copy_atom, r, view)


def _load_vec(copy_atom, vec_width, elem_dtype, div_tensor, idx):
    r = fx.make_rmem_tensor(vec_width, elem_dtype)
    fx.copy_atom_call(copy_atom, fx.slice(div_tensor, (None, idx)), r)
    return fx.memref_load_vec(r)


def _store_vec(copy_atom, vec_width, elem_dtype, val, div_tensor, idx):
    r = fx.make_rmem_tensor(vec_width, elem_dtype)
    fx.memref_store_vec(val, r)
    fx.copy_atom_call(copy_atom, r, fx.slice(div_tensor, (None, idx)))


def _to_elem_scalar(dtype_str: str, elem_dtype, y):
    if const_expr(dtype_str == "f32"):
        return y
    return y.to(elem_dtype)


def _to_elem_vec(dtype_str: str, elem_dtype, use_hw_cvt_bf16: bool, y):
    if const_expr(dtype_str == "bf16"):
        if const_expr(use_hw_cvt_bf16):
            return y.to(elem_dtype)
        u = y.bitcast(fx.Uint32)
        upper = u >> 16
        lsb = upper & 1
        bias = lsb + 0x7FFF
        u_round = y.bitcast(fx.Uint32) + bias
        bf16_bits = u_round >> 16
        even = bf16_bits.shuffle(bf16_bits, [0, 2, 4, 6])
        odd = bf16_bits.shuffle(bf16_bits, [1, 3, 5, 7])
        odd_sh = odd << 16
        packed = even | odd_sh
        return packed.bitcast(elem_dtype)
    if const_expr(dtype_str == "f32"):
        return y
    return y.to(elem_dtype)


def build_rmsnorm_module(M: int, N: int, dtype_str: str):
    arch = get_hip_arch()
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or str(arch).startswith("gfx95")

    tile_cols = BLOCK_THREADS * VEC_WIDTH
    RED_SLOTS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)
    elem_bits = 32 if dtype_str == "f32" else 16

    SharedStorage = _make_reduction_storage(RED_SLOTS)

    @flyc.kernel
    def rmsnorm_kernel(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        _Unused: fx.Tensor,
        Output: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_dtype = dtype_to_elem_type(dtype_str)
        fm_fast = arith.FastMathFlags.fast
        eps_c = EPS
        n_float = float(N)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_red = lds.s_red.view(fx.make_layout(RED_SLOTS, 1))
        s_red2 = lds.s_red2.view(fx.make_layout(RED_SLOTS, 1))

        def wave_reduce_add(x):
            w = x
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << _sh_exp)
                peer = w.shuffle_xor(off, WARP_SIZE)
                w = w.addf(peer, fastmath=fm_fast)
            return w

        def block_reduce_add(val):
            dummy = fx.Float32(0.0)
            r0, _ = block_reduce_add2(val, dummy)
            return r0

        def block_reduce_add2(val0, val1):
            if const_expr(RED_SLOTS == 1):
                return wave_reduce_add(val0), wave_reduce_add(val1)

            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE

            w0 = wave_reduce_add(val0)
            w1 = wave_reduce_add(val1)

            if lane == 0:
                fx.memref_store(w0, s_red, wave)
                fx.memref_store(w1, s_red2, wave)
            gpu.barrier()

            if wave == 0:
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, 0)
                v0 = fx.memref_load(s_red, lane_safe)
                v1 = fx.memref_load(s_red2, lane_safe)
                ww0 = in_range.select(v0, 0.0)
                ww1 = in_range.select(v1, 0.0)
                ww0 = wave_reduce_add(ww0)
                ww1 = wave_reduce_add(ww1)

                if lane == 0:
                    fx.memref_store(ww0, s_red, 0)
                    fx.memref_store(ww1, s_red2, 0)
            gpu.barrier()

            return fx.memref_load(s_red, 0), fx.memref_load(s_red2, 0)

        # ==================================================================
        # Fast path: N is a multiple of tile_cols
        # ==================================================================
        if const_expr(N >= tile_cols and N % tile_cols == 0 and elem_bits <= 16):
            num_tiles = N // tile_cols
            # ── Layout API: buffer-backed tensors + tiled access ─────
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)

            row_in = fx.slice(Input_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))

            in_div = fx.logical_divide(row_in, fx.make_layout(VEC_WIDTH, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(VEC_WIDTH, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(VEC_WIDTH, 1))

            copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)

            c_zero_f = fx.Float32(0.0)
            thread_sumsq = c_zero_f
            thread_dummy = c_zero_f
            in_local = []

            # Pass 1: load + cache + sumsq
            for tile_i in range_constexpr(num_tiles):
                idx = tid + tile_i * BLOCK_THREADS
                vec = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, in_div, idx)
                in_local.append(vec)
                x = vec.to(fx.Float32)

                x2 = x * x
                red2 = x2.reduce(ReductionOp.ADD, fastmath=fm_fast)
                thread_sumsq = thread_sumsq + red2

            _, sum_sq = block_reduce_add2(thread_dummy, thread_sumsq)
            mean_sq = sum_sq / n_float
            ms_eps = mean_sq + eps_c
            rrms = fmath.rsqrt(ms_eps, fastmath=fm_fast)

            # Pass 2: normalize + gamma + store (reuse cached input)
            for tile_i in range_constexpr(num_tiles):
                idx = tid + tile_i * BLOCK_THREADS

                g = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, gamma_div, idx).to(fx.Float32)
                x = in_local[tile_i].to(fx.Float32)

                y = (x * rrms) * g
                out_e = _to_elem_vec(dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, y)

                out_idx = tid + tile_i * BLOCK_THREADS
                _store_vec(copy_atom, VEC_WIDTH, elem_dtype, out_e, out_div, out_idx)

        else:
            # ==============================================================
            # Generic path: scalar 2-pass for arbitrary N
            # ==============================================================
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)

            row_in = fx.slice(Input_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))

            copy_atom_s = fx.make_copy_atom(
                fx.rocdl.BufferCopy16b() if elem_bits <= 16 else fx.rocdl.BufferCopy32b(),
                elem_bits,
            )

            row_div = fx.logical_divide(row_in, fx.make_layout(1, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(1, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(1, 1))

            c_zero_f = fx.Float32(0.0)
            thread_sumsq = c_zero_f

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = _load_scalar(copy_atom_s, elem_dtype, row_div, idx_safe)
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                x2 = x * x
                x2_safe = is_valid.select(x2, c_zero_f)
                thread_sumsq = thread_sumsq + x2_safe

            sum_sq = block_reduce_add(thread_sumsq)
            mean_sq = sum_sq / n_float
            ms_eps = mean_sq + eps_c
            rrms = fmath.rsqrt(ms_eps, fastmath=fm_fast)

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                if idx < N:
                    x_e = _load_scalar(copy_atom_s, elem_dtype, row_div, idx)
                    g_e = _load_scalar(copy_atom_s, elem_dtype, gamma_div, idx)
                    x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                    g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                    norm = x * rrms
                    y = norm * g
                    y_e = _to_elem_scalar(dtype_str, elem_dtype, y)
                    _store_scalar(copy_atom_s, elem_dtype, elem_dtype, out_div, idx, y_e)

    @flyc.jit
    def launch_rmsnorm(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        Output: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = rmsnorm_kernel(Input, Gamma, Gamma, Output)
        launcher.launch(
            grid=(m_in, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_rmsnorm


@functools.lru_cache(maxsize=None)
def _launcher(m: int, n: int, dtype_str: str):
    return build_rmsnorm_module(m, n, dtype_str)


def run(*, x, weight, out) -> None:
    m, n = x.shape
    launch = _launcher(m, n, "bf16")
    launch(x, weight, out, m, stream=torch.cuda.current_stream())
