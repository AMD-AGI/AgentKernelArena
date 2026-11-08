# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Deterministic split-KV reduce for the persistent MLA decode kernel.

Vendored from ROCm/FlyDSL kernels/attention/pa_metadata.py (tag v0.3.2): the compile_pa_metadata_reduce kernel
that upstream uses as a drop-in replacement for aiter.pa_reduce_v1 / mla_reduce_v1, renamed compile_mla_reduce and
fixed to one query token per sequence (its query_length argument, 1 here, is folded into the launch grid). The kernel
body is unchanged.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import gpu

from .kernels_common import dtype_to_elem_type
from .utils import copy_load, exp2_f32_fast, rcp_f32

_FLAT_BUFFER_ELEMENTS = 1 << 30
LOG2E = 1.4426950408889634


# One thread per output element serially combines splits with online softmax.
@functools.lru_cache(maxsize=64)
def compile_mla_reduce(
    *,
    num_query_heads: int,
    head_dim: int,
    output_dtype_str: str,
):
    vec_width = 2 if head_dim % 2 == 0 else 1
    block_threads = head_dim // vec_width
    assert 0 < block_threads <= 1024, "head_dim must fit in one workgroup"
    output_dtype = dtype_to_elem_type(output_dtype_str)

    @flyc.kernel(known_block_size=(block_threads, 1, 1))
    def pa_metadata_reduce_kernel(
        final_output_ptr: fx.Tensor,
        partial_output_ptr: fx.Tensor,
        partial_lse_ptr: fx.Tensor,
        reduce_indptr_ptr: fx.Tensor,
        reduce_final_map_ptr: fx.Tensor,
        reduce_partial_map_ptr: fx.Tensor,
        stride_out_seq: fx.Int32,
        stride_out_head: fx.Int32,
        stride_po_row: fx.Int32,  # num_query_heads * head_dim
        stride_pl_row: fx.Int32,  # num_query_heads
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        g = fx.Int32(gpu.block_id("x"))
        qhead = fx.Int32(gpu.block_id("y"))
        qi = fx.Int32(gpu.block_id("z"))

        scalar_layout = fx.make_layout(1, 1)
        flat_layout = fx.make_layout(_FLAT_BUFFER_ELEMENTS, 1)

        def _flat_divide(tensor):
            buffer_tensor = fx.rocdl.make_buffer_tensor(tensor)
            flat = fx.make_view(fx.get_iter(buffer_tensor), flat_layout)
            return fx.logical_divide(flat, scalar_layout)

        final_output = _flat_divide(final_output_ptr)
        partial_output = _flat_divide(partial_output_ptr)
        partial_lse = _flat_divide(partial_lse_ptr)
        reduce_indptr = _flat_divide(reduce_indptr_ptr)
        reduce_final_map = _flat_divide(reduce_final_map_ptr)
        reduce_partial_map = _flat_divide(reduce_partial_map_ptr)

        copy_i32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        copy_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        copy_f32_vec = fx.make_copy_atom(
            fx.rocdl.BufferCopy64b() if vec_width == 2 else fx.rocdl.BufferCopy32b(),
            fx.Float32,
        )
        copy_output = fx.make_copy_atom(
            (
                fx.rocdl.BufferCopy64b()
                if output_dtype.width * vec_width == 64
                else fx.rocdl.BufferCopy32b() if output_dtype.width * vec_width == 32 else fx.rocdl.BufferCopy16b()
            ),
            output_dtype,
        )
        reg_i32 = fx.make_rmem_tensor(1, fx.Int32)
        reg_f32 = fx.make_rmem_tensor(1, fx.Float32)
        reg_f32_vec = fx.make_rmem_tensor(vec_width, fx.Float32)
        reg_output = fx.make_rmem_tensor(vec_width, output_dtype)

        lo = copy_load(reduce_indptr, g, copy_i32, reg_i32)[0]
        hi = copy_load(reduce_indptr, g + 1, copy_i32, reg_i32)[0]

        if hi > lo:
            qo_start = copy_load(reduce_final_map, g * 2, copy_i32, reg_i32)[0]
            out_row = qo_start + qi

            for slot, st in range(
                lo,
                hi,
                fx.Int32(1),
                init=[
                    fx.Float32(float("-inf")),
                    fx.Float32(0.0),
                    fx.Vector.filled(vec_width, 0.0, fx.Float32),
                ],
            ):
                m = fx.Float32(st[0])
                denom = fx.Float32(st[1])
                acc = fx.Vector(st[2])
                prow = copy_load(reduce_partial_map, slot, copy_i32, reg_i32)[0] + qi
                lse = copy_load(partial_lse, prow * stride_pl_row + qhead, copy_f32, reg_f32)[0]
                v = copy_load(
                    partial_output,
                    prow * stride_po_row + qhead * head_dim + tid * vec_width,
                    copy_f32_vec,
                    reg_f32_vec,
                )
                m_new = fx.maxnumf(m, lse)
                scale_old = exp2_f32_fast((m - m_new) * LOG2E)
                w = exp2_f32_fast((lse - m_new) * LOG2E)
                denom_new = denom * scale_old + w
                acc_new = acc * fx.Float32(scale_old) + fx.Vector(v) * fx.Float32(w)
                results = yield [m_new, denom_new, acc_new]

            denom_f = fx.Float32(results[1])
            acc_f = fx.Vector(results[2])
            safe_denom = (denom_f > 0.0).select(denom_f, 1.0)
            out_acc = acc_f * rcp_f32(safe_denom)

            out_off = out_row * stride_out_seq + qhead * stride_out_head + tid * vec_width
            output_val = fx.Vector(out_acc).to(output_dtype)
            fx.memref_store_vec(output_val, reg_output)
            fx.copy(copy_output, reg_output, fx.slice(final_output, (None, out_off)))

    @flyc.jit
    def launch_pa_metadata_reduce(
        final_output,
        partial_output,
        partial_lse,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        stride_out_seq,
        stride_out_head,
        stride_po_row,
        stride_pl_row,
        num_groups,
        stream: fx.Stream = fx.Stream(None),
    ):
        with CompilationContext.compile_hints({"fastmath": "contract"}):
            pa_metadata_reduce_kernel(
                final_output,
                partial_output,
                partial_lse,
                reduce_indptr,
                reduce_final_map,
                reduce_partial_map,
                stride_out_seq,
                stride_out_head,
                stride_po_row,
                stride_pl_row,
            ).launch(
                grid=(num_groups, num_query_heads, 1),
                block=(block_threads, 1, 1),
                stream=stream,
            )

    return {"launch": launch_pa_metadata_reduce, "kernel": pa_metadata_reduce_kernel}
