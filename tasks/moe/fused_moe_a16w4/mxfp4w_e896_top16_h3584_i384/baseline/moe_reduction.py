# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Top-k reduction for the per-slot (reduce-mode) stage-2 output, vendored from ROCm/aiter
# (commit 652f2d4b768239aefd515c929e84817681ff598d), aiter/ops/flydsl/kernels/moe_gemm_2stage.py
# (compile_moe_reduction). Changes: imports made local; the fp8 route-output variant removed.

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch


@functools.lru_cache(maxsize=1024)
def compile_moe_reduction(
    *,
    topk: int,
    model_dim: int,
    dtype_str: str = "f16",
    use_mask: bool = False,
    num_experts: int = 0,
    out_dtype_str: str | None = None,
):
    """Compile a reduction kernel that sums over the topk dimension.

    Input:  X [tokens, topk, model_dim]
            expert_mask [num_experts] i32 (optional, if use_mask=True)
            topk_ids   [tokens, topk] i32 (optional, if use_mask=True)
    Output: Y [tokens, model_dim]

    This kernel performs: Y[t, d] = sum_k(X[t, k, d]) for all t, d.
    When use_mask=True, the kernel fuses the EP validity gather:
        valid[t, k] = expert_mask[topk_ids[t, k]] != 0
    and only accumulates X[t, k, :] when valid[t, k] is true.
    Used in conjunction with compile_moe_gemm2(accumulate=False) to avoid atomic contention.
    """
    if dtype_str == "fp8":
        raise NotImplementedError("the fp8 route-output reduction is not vendored")
    get_hip_arch()
    ir.ShapedType.get_dynamic_size()

    # Kernel Config
    BLOCK_SIZE = 256
    VEC_WIDTH = 8

    if dtype_str == "f32":
        elem_type_tag = "f32"
    elif dtype_str == "f16":
        elem_type_tag = "f16"
    elif dtype_str == "bf16":
        elem_type_tag = "bf16"
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

    # fx Numeric dtypes for typed copy fragments and accumulation.
    compute_numeric = fx.Float32
    elem_numeric = (
        fx.Float32
        if elem_type_tag == "f32"
        else (fx.Float16 if elem_type_tag == "f16" else fx.BFloat16)
    )

    module_name = (
        f"moe_reduction_kernel_{'masked' if use_mask else 'plain'}"
        f"_{dtype_str}_topk{topk}_md{model_dim}"
    )

    elem_bytes_c = (32 if dtype_str == "f32" else 16) // 8

    @flyc.kernel(name=module_name)
    def moe_reduction_kernel(
        X: fx.Pointer,
        Y: fx.Pointer,
        expert_mask: fx.Pointer,
        topk_ids: fx.Pointer,
        i32_m_tokens: fx.Int32,
    ):
        m_tokens = fx.Int64(i32_m_tokens)
        c_topk = fx.Int64(topk)
        c_model_dim = fx.Int64(model_dim)
        elem_bits = 32 if dtype_str == "f32" else 16
        copy_vec_width = 128 // elem_bits  # 8 for f16/bf16, 4 for f32
        n_sub = VEC_WIDTH // copy_vec_width  # 1 for f16/bf16, 2 for f32

        def _buffer_tensor(ptr, numeric, num_elems, num_bytes, byte_off_i64):
            addr_i64 = fx.Int64(fx.ptrtoint(ptr)) + byte_off_i64
            ptr_ty = fx.PointerType.get(
                numeric.ir_type,
                address_space=fx.AddressSpace.Global,
                alignment=numeric.width // 8,
            )
            base = fx.inttoptr(ptr_ty, addr_i64)
            view = fx.make_view(base, fx.make_layout(num_elems, 1))
            return fx.rocdl.make_buffer_tensor(view, num_records_bytes=num_bytes)

        def _tile(tensor, elem_offset, width):
            ptr = fx.add_offset(fx.get_iter(tensor), elem_offset)
            return fx.make_view(ptr, fx.make_layout(width, 1))

        token_idx = fx.Int64(gpu.block_id("x"))
        tile_idx = fx.Int64(gpu.block_id("y"))
        tid = fx.Int64(gpu.thread_id("x"))

        # ── 64-bit base-offset folding ─────────────────────────────────
        # X is [m_tokens, topk, model_dim]; total bytes can exceed 4 GiB
        # for large batches (e.g. 131072 * 6 * 4096 * 2 = 6 GiB), which
        # overflows the i32 voffset used by buffer_load. To stay i32-safe,
        # fold the per-WG token byte offset into the descriptor's 48-bit
        # base address (computed in i64). The in-kernel voffsets then only
        # need to address one token's slab.
        slab_elems_x = c_topk * c_model_dim
        x_slab_nbytes = slab_elems_x * fx.Int64(elem_bytes_c)
        y_slab_nbytes = c_model_dim * fx.Int64(elem_bytes_c)
        x_base_off_i64 = token_idx * x_slab_nbytes
        y_base_off_i64 = token_idx * c_model_dim * fx.Int64(elem_bytes_c)

        x_buf = _buffer_tensor(
            X, elem_numeric, slab_elems_x, x_slab_nbytes, x_base_off_i64
        )
        y_buf = _buffer_tensor(
            Y, elem_numeric, c_model_dim, y_slab_nbytes, y_base_off_i64
        )

        copy_vec = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_numeric)
        copy_scalar = fx.make_copy_atom(
            (fx.rocdl.BufferCopy32b() if elem_bits == 32 else fx.rocdl.BufferCopy16b()),
            elem_numeric,
        )
        copy_i32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)

        if const_expr(use_mask):
            tk_slab_nbytes = c_topk * fx.Int64(4)
            tk_base_off_i64 = token_idx * tk_slab_nbytes
            topk_ids_buf = _buffer_tensor(
                topk_ids,
                fx.Int32,
                c_topk,
                tk_slab_nbytes,
                tk_base_off_i64,
            )
            # expert_mask: [num_experts] i32. Caller supplies num_experts
            # at compile time so we can size the descriptor exactly.
            em_nbytes = fx.Int64(num_experts * 4)
            expert_mask_buf = _buffer_tensor(
                expert_mask,
                fx.Int32,
                fx.Int64(num_experts),
                em_nbytes,
                fx.Int64(0),
            )
        else:
            topk_ids_buf = None
            expert_mask_buf = None

        def load_valid_mask(k):
            """Fused EP gather: valid = expert_mask[topk_ids[token, k]] != 0."""
            eid_frag = fx.make_rmem_tensor(1, fx.Int32)
            fx.copy(copy_i32, _tile(topk_ids_buf, fx.Int32(k), 1), eid_frag)
            eid_i32 = fx.Vector(eid_frag.load())[0]
            valid_frag = fx.make_rmem_tensor(1, fx.Int32)
            fx.copy(
                copy_i32,
                _tile(expert_mask_buf, eid_i32, 1),
                valid_frag,
            )
            valid_i32 = fx.Vector(valid_frag.load())[0]
            return valid_i32 != fx.Int32(0)

        # Guard: token in range (unsigned compare on non-negative i64 ids).
        if token_idx < m_tokens:
            c_tile_cols = fx.Int64(BLOCK_SIZE * VEC_WIDTH)
            c_vecw = fx.Int64(VEC_WIDTH)
            col_base = tile_idx * c_tile_cols + tid * c_vecw

            # Guard: this thread's column window overlaps the row at all.
            if col_base < c_model_dim:
                # Fast path: the full VEC_WIDTH window is in-bounds (<= -> ule).
                if col_base + c_vecw <= c_model_dim:
                    # n_sub buffer_loads of copy_vec_width elems (128b each) cover
                    # the full VEC_WIDTH stride; accumulate in f32.
                    acc_vecs = [
                        fx.Vector.filled(copy_vec_width, 0.0, compute_numeric)
                        for _ in range(n_sub)
                    ]
                    for k in range_constexpr(topk):
                        # Within one token's slab, k strides the topk dim by model_dim.
                        k_off_elems = fx.Int64(k) * c_model_dim + col_base
                        mv_ok = load_valid_mask(k) if const_expr(use_mask) else None
                        for si in range_constexpr(n_sub):
                            elem_offset = k_off_elems + fx.Int64(si * copy_vec_width)
                            in_frag = fx.make_rmem_tensor(copy_vec_width, elem_numeric)
                            fx.copy(
                                copy_vec,
                                _tile(x_buf, elem_offset, copy_vec_width),
                                in_frag,
                            )
                            vec_e = fx.Vector(in_frag.load())
                            if const_expr(use_mask):
                                zero_e = fx.Vector.filled(
                                    copy_vec_width, 0.0, elem_numeric
                                )
                                vec_e = mv_ok.select(vec_e, zero_e)
                            vec_c = (
                                vec_e.to(fx.Float32)
                                if const_expr(elem_bits < 32)
                                else vec_e
                            )
                            acc_vecs[si] = acc_vecs[si] + vec_c

                    for si in range_constexpr(n_sub):
                        out_vec = acc_vecs[si]
                        if const_expr(elem_bits < 32):
                            out_vec = out_vec.to(elem_numeric)
                        elem_offset = col_base + fx.Int64(si * copy_vec_width)
                        out_frag = fx.make_rmem_tensor(copy_vec_width, elem_numeric)
                        out_frag.store(out_vec)
                        fx.copy(
                            copy_vec,
                            out_frag,
                            _tile(y_buf, elem_offset, copy_vec_width),
                        )
                else:
                    # Tail path: per-lane scalar accumulate for the partial window.
                    for lane in range_constexpr(VEC_WIDTH):
                        col = col_base + fx.Int64(lane)
                        if col < c_model_dim:
                            a = fx.Float32(0.0)
                            for k in range_constexpr(topk):
                                elem_offset = fx.Int64(k) * c_model_dim + col
                                in_frag = fx.make_rmem_tensor(1, elem_numeric)
                                fx.copy(
                                    copy_scalar,
                                    _tile(x_buf, elem_offset, 1),
                                    in_frag,
                                )
                                x_val = fx.Vector(in_frag.load())[0]
                                if const_expr(use_mask):
                                    v = load_valid_mask(k).select(
                                        x_val, elem_numeric(0.0)
                                    )
                                else:
                                    v = x_val
                                if const_expr(elem_bits < 32):
                                    v = v.to(fx.Float32)
                                a = a + v
                            out = (
                                a.to(elem_numeric) if const_expr(elem_bits < 32) else a
                            )
                            out_frag = fx.make_rmem_tensor(1, elem_numeric)
                            out_frag.store(fx.Vector.from_elements([out], elem_numeric))
                            fx.copy(
                                copy_scalar,
                                out_frag,
                                _tile(y_buf, col, 1),
                            )

    # ── Host launcher (flyc.jit + .launch) ────────────────────────────────
    tile_size = BLOCK_SIZE * VEC_WIDTH
    gy_static = (model_dim + tile_size - 1) // tile_size

    @flyc.jit
    def launch_moe_reduction(
        X: fx.Pointer,
        Y: fx.Pointer,
        expert_mask: fx.Pointer,
        topk_ids: fx.Pointer,
        i32_m_tokens: fx.Int32,
        stream: fx.Stream,
    ):
        gx = fx.Int64(i32_m_tokens)
        moe_reduction_kernel(X, Y, expert_mask, topk_ids, i32_m_tokens).launch(
            grid=(gx, gy_static, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    return launch_moe_reduction
