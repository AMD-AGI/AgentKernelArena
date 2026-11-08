# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# Vendored from ROCm/aiter @ 652f2d4b768239aefd515c929e84817681ff598d
# (aiter/ops/triton/gluon/pa_decode_gluon.py), MIT; see LICENSE.
# Task modifications: only the path used by this task is kept: the persistent-scheduling decode kernel
# for one KV head (paged_attention_decode_sliding_window_head_1) in its ONE_SHOT mode, where a single
# program per sequence writes the final output, so no reduce kernel is launched (the upstream reduce
# kernels are Triton or FlyDSL, and a Gluon baseline launches only Gluon kernels). The kernel and its
# helpers are unchanged. The flydsl / C++ reduce imports, the other kernels, the aiter imports and the
# Gluon-unavailable fallback (which printed a warning) are removed, the arch query is inlined, and the
# pa_decode_gluon wrapper is replaced by run() with the same dispatch for this configuration.

from functools import lru_cache

import torch
import triton
import triton.language as tl
from triton.language.extra.hip import libdevice as hip_libdevice
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

try:
    from triton.experimental.gluon.language.amd.cdna3 import (
        sched_barrier as _amd_iglp_sched_barrier,
    )
    from triton.experimental.gluon.language.amd.cdna3 import (
        sched_group_barrier as _amd_iglp_sched_group_barrier,
    )
    from triton.experimental.gluon.language.amd.cdna3 import (
        set_prio as _amd_set_prio,
    )
except ImportError:
    # ignore iglp hint
    @gluon.jit
    def _amd_iglp_sched_barrier(tensor, inst_mask):
        return tensor

    @gluon.jit
    def _amd_iglp_sched_group_barrier(tensor, inst_mask, cnt, _):
        return tensor

    @gluon.jit
    def _amd_set_prio(value, prio):
        pass


@gluon.jit
def _fused_max_combine(a1, a2, b1, b2):
    return tl.maximum(a1, b1), tl.maximum(a2, b2)



@lru_cache(maxsize=1)
def get_cdna_version():
    """Get CDNA version lazily to avoid CUDA initialization during import."""
    # --- inlined arch_info.get_arch() ---
    arch = triton.runtime.driver.active.get_current_target().arch
    if arch in ["gfx950"]:
        return 4
    elif arch in ["gfx942"]:
        return 3
    else:
        return -1


DS_WRITE = gl.constexpr(0x200)
DS_READ = gl.constexpr(0x100)
VMEM_LOAD = gl.constexpr(0x020)
MFMA = gl.constexpr(0x008)
COMPUTE = gl.constexpr(0x001)


def define_layout(
    QUERY_GROUP_SIZE_POW2: gl.constexpr,
    CONTEXT_PARTITION_SIZE: gl.constexpr,
    QUERY_SEQ_LEN_POW2: gl.constexpr,
) -> gl.constexpr:
    # Register allocation configuration based on group size and compute block size
    if QUERY_GROUP_SIZE_POW2 >= 16:
        if QUERY_GROUP_SIZE_POW2 == 16:
            if CONTEXT_PARTITION_SIZE == 128:
                register_bases: gl.constexpr = [[0, 1], [0, 2], [0, 64]]
            elif CONTEXT_PARTITION_SIZE == 256:
                register_bases: gl.constexpr = [[0, 1], [0, 2], [0, 64], [0, 128]]
        elif QUERY_GROUP_SIZE_POW2 == 32:
            if CONTEXT_PARTITION_SIZE == 128:
                register_bases: gl.constexpr = [[0, 1], [0, 2], [0, 64], [16, 0]]
            elif CONTEXT_PARTITION_SIZE == 256:
                register_bases: gl.constexpr = [
                    [0, 1],
                    [0, 2],
                    [0, 64],
                    [0, 128],
                    [16, 0],
                ]
        elif QUERY_GROUP_SIZE_POW2 == 64:
            if CONTEXT_PARTITION_SIZE == 128:
                register_bases: gl.constexpr = [
                    [0, 1],
                    [0, 2],
                    [0, 64],
                    [16, 0],
                    [32, 0],
                ]
            elif CONTEXT_PARTITION_SIZE == 256:
                register_bases: gl.constexpr = [
                    [0, 1],
                    [0, 2],
                    [0, 64],
                    [0, 128],
                    [16, 0],
                    [32, 0],
                ]

        # Distributed layout for QK linear operations
        qk_linear_layout: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=register_bases,
            lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]],
            warp_bases=[[0, 16], [0, 32]],
            block_bases=[],
            shape=[QUERY_GROUP_SIZE_POW2, CONTEXT_PARTITION_SIZE],
        )
    else:
        VGPRS0: gl.constexpr = QUERY_SEQ_LEN_POW2
        THREADS0: gl.constexpr = triton.cdiv(QUERY_GROUP_SIZE_POW2, (4 * VGPRS0))
        THREADS1: gl.constexpr = 64 // THREADS0

        qk_linear_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[VGPRS0, CONTEXT_PARTITION_SIZE // THREADS1],
            threads_per_warp=[THREADS0, THREADS1],
            warps_per_cta=[4, 1],
            order=[1, 0],
        )
    return qk_linear_layout


define_layout.__triton_builtin__ = True


def store_temporary_result(
    max_logits,
    exp_sums,
    attention_accumulator,
    max_logits_ptr,
    exp_sums_ptr,
    output_ptr,
    max_logits_offsets,
    output_offsets,
    qk_row_mask,
    output_mask,
    _semantic=None,
) -> None:
    gl.amd.cdna3.buffer_store(
        stored_value=max_logits,
        ptr=max_logits_ptr,
        offsets=max_logits_offsets,
        mask=qk_row_mask,
        _semantic=_semantic,
    )
    gl.amd.cdna3.buffer_store(
        stored_value=exp_sums,
        ptr=exp_sums_ptr,
        offsets=max_logits_offsets,
        mask=qk_row_mask,
        _semantic=_semantic,
    )
    gl.amd.cdna3.buffer_store(
        stored_value=attention_accumulator,
        ptr=output_ptr,
        offsets=output_offsets,
        mask=output_mask,
        _semantic=_semantic,
    )


store_temporary_result.__triton_builtin__ = True



# @triton.autotune(
#     configs=[
#         triton.Config(
#             {"waves_per_eu": wa}, maxnreg=512 // wa if wa > 0 else None, num_stages=1
#         )
#         for wa in range(5)
#     ],
#     key=[
#         "KV_BLOCK_SIZE",
#         "SLIDING_WINDOW",
#         "KV_QUANT_MODE",
#         "QUERY_QUANT_MODE",
#         "ONE_QUERY_GROUP_SIZE_POW2",
#         "HEAD_SIZE_POW2",
#         "COMPUTE_TYPE",
#         "ONE_SHOT",
#     ],
#     cache_results=True,
# )
@gluon.jit
def paged_attention_decode_sliding_window_head_1(
    exp_sums_ptr,  # [num_seqs, num_kv_heads, max_parts, q_group_size]
    max_logits_ptr,  # [num_seqs, num_kv_heads, max_parts, q_group_size]
    output_ptr,  # [batch_size, query_length, num_kv_heads, query_group_size, head_size]
    query_ptr,  # [batch_size, query_length, num_kv_heads, query_group_size, head_size]
    key_cache_ptr,  # [num_blocks, num_kv_heads, head_size // x, kv_block_size, x]
    value_cache_ptr,  # [num_blocks, num_kv_heads, head_size, kv_block_size]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    context_lengths_ptr,  # [num_seqs]
    softmax_scale: float,
    query_scale,  # [num_seqs, query_length, num_kv_heads, query_group_size, 1](per-token) or [1](per-tensor) or None
    key_scale,  # [num_blocks, num_kv_heads, kv_block_size, 1](per-token) or [1](per-tensor) or None
    value_scale,  # [num_blocks, num_kv_heads, kv_block_size, 1](per-token) or [1](per-tensor) or None
    sinks_ptr,  # [num_query_heads]
    stride_max_logits_seq: int,
    stride_max_logits_head: int,
    stride_max_logits_part: int,
    # 5D output strides for [batch_size, query_length, num_kv_heads, query_group_size, head_size]
    stride_output_bs: int,
    stride_output_len: int,
    stride_output_kv_head: int,
    stride_output_group_size: int,
    # 5D query strides for [batch_size, query_length, num_kv_heads, query_group_size, head_size]
    stride_query_bs: int,
    stride_query_qlen: int,
    stride_query_kv_head: int,
    stride_query_group_size: int,
    stride_key_block: int,
    stride_key_head: int,
    stride_key_head_split: int,
    stride_key_block_elem: int,
    stride_value_block: int,
    stride_value_head: int,
    stride_value_head_size: int,
    stride_block_table_seq: int,
    stride_query_scale_bs: int,
    stride_query_scale_qlen: int,
    stride_query_scale_kv_head: int,
    kv_scale_stride_0: int,
    kv_scale_stride_1: int,
    query_seq_len: int,
    query_group_size: int,
    head_size: int,
    COMPUTE_TYPE: gl.constexpr,
    QUERY_SEQ_LEN_POW2: gl.constexpr,
    ONE_QUERY_GROUP_SIZE_POW2: gl.constexpr,
    HEAD_SIZE_POW2: gl.constexpr,
    KV_BLOCK_SIZE: gl.constexpr,
    CONTEXT_PARTITION_SIZE: gl.constexpr,
    QUERY_QUANT_MODE: gl.constexpr,
    KV_QUANT_MODE: gl.constexpr,
    VALUE_TRANSPOSED: gl.constexpr,  # [num_blocks, num_kv_heads, kv_block_size // x, head_size, x]
    IS_CAUSAL: gl.constexpr,
    FP8_MAX_VALUE: gl.constexpr,
    SLIDING_WINDOW: gl.constexpr = 0,
    CDNA_VERSION: gl.constexpr = 3,
    ONE_SHOT: gl.constexpr = False,
):
    """
    Paged Attention Decode Kernel with FP8/BF16 support for AMD GPUs.

    This kernel implements the attention mechanism for decoding in transformer models
    with support for paged KV caches and FP8 quantization. It handles causal masking,
    ALiBi biases, and various quantization schemes.

    Args:
        exp_sums_ptr: Pointer to exponential sums output tensor
        max_logits_ptr: Pointer to maximum logits output tensor
        output_ptr: Pointer to attention output tensor
        query_ptr: Pointer to query tensor
        key_cache_ptr: Pointer to key cache in block layout
        value_cache_ptr: Pointer to value cache in block layout
        block_tables_ptr: Pointer to block tables mapping sequences to physical blocks
        context_lengths_ptr: Pointer to sequence lengths for each sequence
        softmax_scale: Scaling factor for softmax
        query_scale: Query quantization scales
        key_scale: Key quantization scales
        value_scale: Value quantization scales
        Various stride parameters for tensor access
        Compile-time constants for kernel configuration

    Note:
        This kernel uses AMD CDNA3 MFMA instructions for efficient matrix operations
        and supports both FP8 and BF16 data types with various quantization modes.
    """
    # ==================== VALIDATION CHECKS ====================
    # Data type validation
    gl.static_assert(
        query_ptr.dtype.is_fp8()
        or query_ptr.dtype.element_ty == gl.bfloat16
        or query_ptr.dtype.element_ty == gl.float16
    )
    gl.static_assert(
        key_cache_ptr.dtype.is_fp8()
        or key_cache_ptr.dtype.element_ty == gl.bfloat16
        or key_cache_ptr.dtype.element_ty == gl.float16
    )
    gl.static_assert(
        value_cache_ptr.dtype.is_fp8()
        or value_cache_ptr.dtype.element_ty == gl.bfloat16
        or value_cache_ptr.dtype.element_ty == gl.float16
    )

    if QUERY_QUANT_MODE >= 0:
        gl.static_assert(query_scale.dtype.element_ty == gl.float32)
    if KV_QUANT_MODE >= 0:
        gl.static_assert(key_scale.dtype.element_ty == gl.float32)
        gl.static_assert(value_scale.dtype.element_ty == gl.float32)

    sequence_idx = gl.program_id(0)
    mtp_idx = gl.program_id(1)
    split_idx = gl.program_id(2)

    # ==================== CONSTANTS AND CONFIGURATION ====================
    if COMPUTE_TYPE.is_fp8():
        MFMA_INSTR_K: gl.constexpr = 32
    else:
        MFMA_INSTR_K: gl.constexpr = 16

    QK_PV_MFMA_INSTR_SHAPE: gl.constexpr = [16, 16, MFMA_INSTR_K]

    if KV_QUANT_MODE >= 0:
        KV_16B_ELEMENT_COUNT: gl.constexpr = 16
    else:
        KV_16B_ELEMENT_COUNT: gl.constexpr = 8

    if COMPUTE_TYPE.is_fp8():
        OUTPUT_DTYPE: gl.constexpr = tl.bfloat16
    else:
        OUTPUT_DTYPE: gl.constexpr = COMPUTE_TYPE

    LOG2_E: gl.constexpr = 1.4426950408889634  # log2(e) for exponential conversion

    K_HEAD_SIZE_SPLITS: gl.constexpr = HEAD_SIZE_POW2 // KV_16B_ELEMENT_COUNT
    MAX_NUM_KV_BLOCKS_PER_COMPUTE: gl.constexpr = triton.cdiv(
        CONTEXT_PARTITION_SIZE, KV_BLOCK_SIZE
    )

    CONTEXT_PARTITION_SIZE_PER_BLOCK: gl.constexpr = triton.cdiv(
        KV_BLOCK_SIZE, CONTEXT_PARTITION_SIZE
    )
    sequence_split_idx = split_idx // CONTEXT_PARTITION_SIZE_PER_BLOCK
    block_split_idx = split_idx % CONTEXT_PARTITION_SIZE_PER_BLOCK
    # ==================== MEMORY LAYOUT DEFINITIONS ====================
    # Query tensor layout - optimized for sequential access (2D)
    if COMPUTE_TYPE.is_fp8():
        SHARED_LAYOUT_WIDTH: gl.constexpr = 8
    else:
        SHARED_LAYOUT_WIDTH: gl.constexpr = 4
    shared_query_layout: gl.constexpr = gl.SwizzledSharedLayout(
        SHARED_LAYOUT_WIDTH, 1, 16, order=[1, 0]
    )
    shared_probs_layout: gl.constexpr = gl.SwizzledSharedLayout(
        SHARED_LAYOUT_WIDTH, 1, 8, order=[1, 0]
    )
    shared_value_scale_layout: gl.constexpr = gl.SwizzledSharedLayout(
        1, 1, 8, order=[0]
    )
    shared_key_scale_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 8, order=[0])
    QUERY_GROUP_SIZE_POW2: gl.constexpr = QUERY_SEQ_LEN_POW2 * ONE_QUERY_GROUP_SIZE_POW2
    if ONE_QUERY_GROUP_SIZE_POW2 <= 16:
        Q_WARPS_PER_CTA_DIM1: gl.constexpr = triton.cdiv(ONE_QUERY_GROUP_SIZE_POW2, 4)
        Q_WARPS_PER_CTA_DIM0: gl.constexpr = 4 // Q_WARPS_PER_CTA_DIM1
    else:
        Q_WARPS_PER_CTA_DIM0: gl.constexpr = 1
        Q_WARPS_PER_CTA_DIM1: gl.constexpr = 4
    if QUERY_SEQ_LEN_POW2 == 1:
        blocked_query_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 8],
            threads_per_warp=[4, 16],
            warps_per_cta=[Q_WARPS_PER_CTA_DIM0, Q_WARPS_PER_CTA_DIM1],
            order=[1, 0],
        )
    else:
        mtp_blocked_query_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 1, 8],
            threads_per_warp=[1, 4, 16],
            warps_per_cta=[Q_WARPS_PER_CTA_DIM0, Q_WARPS_PER_CTA_DIM1, 1],
            order=[2, 1, 0],
        )
    # Key cache layout - optimized for block-wise access patterns
    if KV_BLOCK_SIZE > CONTEXT_PARTITION_SIZE:
        KV_COMPUTE_BLOCK_SIZE: gl.constexpr = CONTEXT_PARTITION_SIZE
    else:
        KV_COMPUTE_BLOCK_SIZE: gl.constexpr = KV_BLOCK_SIZE

    key_warps_per_cta: gl.constexpr = (
        [4, 1, 1, 1] if KV_COMPUTE_BLOCK_SIZE == 16 else [1, 1, 4, 1]
    )
    blocked_key_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1, 1, KV_16B_ELEMENT_COUNT],
        threads_per_warp=[1, 4, 16, 1],
        warps_per_cta=key_warps_per_cta,
        order=[3, 2, 1, 0],
    )

    # QK Matrix multiplication layout using AMD MFMA instructions
    qk_mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=CDNA_VERSION,
        instr_shape=QK_PV_MFMA_INSTR_SHAPE,
        transposed=True,
        warps_per_cta=[1, 4],
    )
    qk_lhs_operand_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=qk_mfma_layout, k_width=KV_16B_ELEMENT_COUNT
    )
    qk_rhs_operand_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=qk_mfma_layout, k_width=KV_16B_ELEMENT_COUNT
    )

    qk_linear_layout: gl.constexpr = define_layout(
        QUERY_GROUP_SIZE_POW2, CONTEXT_PARTITION_SIZE, QUERY_SEQ_LEN_POW2
    )
    if MAX_NUM_KV_BLOCKS_PER_COMPUTE == 1:
        kv_scale_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1],
            threads_per_warp=[64],
            warps_per_cta=[4],
            order=[0],
        )
        kv_scale_indices = gl.arange(0, CONTEXT_PARTITION_SIZE, layout=kv_scale_layout)

    context_length = gl.load(context_lengths_ptr + sequence_idx)
    # Value cache layout configuration based on transpose flag
    if VALUE_TRANSPOSED:
        # Transposed value layout for better memory access patterns
        value_threads_per_warp: gl.constexpr = (
            [4, 1, 16, 1] if KV_COMPUTE_BLOCK_SIZE == 16 else [1, 4, 16, 1]
        )

        blocked_value_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 1, 1, KV_16B_ELEMENT_COUNT],
            threads_per_warp=value_threads_per_warp,
            warps_per_cta=[1, 1, 4, 1],
            order=[3, 2, 1, 0],
        )

        value_dim1_offsets = gl.arange(
            0,
            KV_COMPUTE_BLOCK_SIZE // KV_16B_ELEMENT_COUNT,
            layout=gl.SliceLayout(
                0, gl.SliceLayout(2, gl.SliceLayout(3, blocked_value_layout))
            ),
        )
        value_dim2_offsets = gl.arange(
            0,
            HEAD_SIZE_POW2,
            layout=gl.SliceLayout(
                0, gl.SliceLayout(1, gl.SliceLayout(3, blocked_value_layout))
            ),
        )
        value_dim3_offsets = gl.arange(
            0,
            KV_16B_ELEMENT_COUNT,
            layout=gl.SliceLayout(
                0, gl.SliceLayout(1, gl.SliceLayout(2, blocked_value_layout))
            ),
        )
    else:
        # Standard value layout
        value_threads_per_warp: gl.constexpr = (
            [4, 16, 1] if KV_COMPUTE_BLOCK_SIZE == 16 else [1, 16, 4]
        )
        blocked_value_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 1, KV_16B_ELEMENT_COUNT],
            threads_per_warp=value_threads_per_warp,
            warps_per_cta=[1, 4, 1],
            order=[2, 1, 0],
        )

        value_dim1_offsets = gl.arange(
            0,
            HEAD_SIZE_POW2,
            layout=gl.SliceLayout(0, gl.SliceLayout(2, blocked_value_layout)),
        )
        value_dim2_offsets = gl.arange(
            0,
            KV_COMPUTE_BLOCK_SIZE,
            layout=gl.SliceLayout(0, gl.SliceLayout(1, blocked_value_layout)),
        )

    # PV Matrix multiplication layout using AMD MFMA instructions
    pv_mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=CDNA_VERSION,
        instr_shape=QK_PV_MFMA_INSTR_SHAPE,
        transposed=True,
        warps_per_cta=[1, 4],
    )
    pv_lhs_operand_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=pv_mfma_layout, k_width=KV_16B_ELEMENT_COUNT
    )
    pv_rhs_operand_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=pv_mfma_layout, k_width=KV_16B_ELEMENT_COUNT
    )

    # ==================== LAYOUT SLICE DEFINITIONS ====================

    if QUERY_SEQ_LEN_POW2 == 1:
        query_group_size_layout: gl.constexpr = gl.SliceLayout(1, blocked_query_layout)
        head_size_layout: gl.constexpr = gl.SliceLayout(0, blocked_query_layout)
    else:
        mtp_query_len_layout: gl.constexpr = gl.SliceLayout(
            1, gl.SliceLayout(2, mtp_blocked_query_layout)
        )
        mtp_query_group_size_layout: gl.constexpr = gl.SliceLayout(
            0, gl.SliceLayout(2, mtp_blocked_query_layout)
        )
        mtp_head_size_layout: gl.constexpr = gl.SliceLayout(
            0, gl.SliceLayout(1, mtp_blocked_query_layout)
        )

    # Key layout slices
    block_id_layout: gl.constexpr = gl.SliceLayout(
        1, gl.SliceLayout(2, gl.SliceLayout(3, blocked_key_layout))
    )
    head_size_split_layout: gl.constexpr = gl.SliceLayout(
        0, gl.SliceLayout(2, gl.SliceLayout(3, blocked_key_layout))
    )
    block_element_layout: gl.constexpr = gl.SliceLayout(
        0, gl.SliceLayout(1, gl.SliceLayout(3, blocked_key_layout))
    )
    contiguous_kv_elements_layout: gl.constexpr = gl.SliceLayout(
        0, gl.SliceLayout(1, gl.SliceLayout(2, blocked_key_layout))
    )

    # Coordinate offsets for various dimensions
    if QUERY_SEQ_LEN_POW2 == 1:
        query_group_size_offsets = gl.arange(
            0, ONE_QUERY_GROUP_SIZE_POW2, layout=query_group_size_layout
        )
        head_size_offsets = gl.arange(0, HEAD_SIZE_POW2, layout=head_size_layout)
    else:
        mtp_query_len_offsets = gl.arange(
            0, QUERY_SEQ_LEN_POW2, layout=mtp_query_len_layout
        )
        mtp_query_group_size_offsets = gl.arange(
            0, ONE_QUERY_GROUP_SIZE_POW2, layout=mtp_query_group_size_layout
        )
        mtp_head_size_offsets = gl.arange(
            0, HEAD_SIZE_POW2, layout=mtp_head_size_layout
        )

    head_size_split_offsets = gl.arange(
        0, K_HEAD_SIZE_SPLITS, layout=head_size_split_layout
    )
    block_element_offsets = gl.arange(
        0, KV_COMPUTE_BLOCK_SIZE, layout=block_element_layout
    )
    contiguous_kv_element_offsets = gl.arange(
        0, KV_16B_ELEMENT_COUNT, layout=contiguous_kv_elements_layout
    )
    max_logits_base_offsets = gl.arange(
        0, QUERY_GROUP_SIZE_POW2, layout=gl.SliceLayout(1, qk_linear_layout)
    )
    if QUERY_SEQ_LEN_POW2 == 1:
        query_row_mask_1d = query_group_size_offsets < query_group_size
    else:
        query_row_mask_3d = (
            mtp_idx * QUERY_SEQ_LEN_POW2 + mtp_query_len_offsets[:, None, None]
            < query_seq_len
        ) & (mtp_query_group_size_offsets[None, :, None] < query_group_size)
        query_row_mask_1d = gl.reshape(query_row_mask_3d, [QUERY_GROUP_SIZE_POW2])
    output_group_offsets = gl.arange(
        0, QUERY_GROUP_SIZE_POW2, layout=gl.SliceLayout(1, pv_mfma_layout)
    )
    if QUERY_SEQ_LEN_POW2 == 1:
        output_group_offsets = mtp_idx * query_group_size + output_group_offsets
    else:
        output_query_len_idx = output_group_offsets // ONE_QUERY_GROUP_SIZE_POW2
        output_group_idx_in_len = output_group_offsets % ONE_QUERY_GROUP_SIZE_POW2
        output_group_offsets = (
            mtp_idx * QUERY_SEQ_LEN_POW2 + output_query_len_idx
        ) * query_group_size + output_group_idx_in_len
    output_head_size_offsets = gl.arange(
        0, HEAD_SIZE_POW2, layout=gl.SliceLayout(0, pv_mfma_layout)
    )

    # ==================== PROGRAM ID AND INITIALIZATION ====================

    max_logits = gl.full(
        (QUERY_GROUP_SIZE_POW2,),
        float("-inf"),
        dtype=gl.float32,
        layout=gl.SliceLayout(1, qk_linear_layout),
    )
    exp_sums = gl.full(
        (QUERY_GROUP_SIZE_POW2,),
        0.0,
        dtype=gl.float32,
        layout=gl.SliceLayout(1, qk_linear_layout),
    )
    attention_accumulator = gl.zeros(
        (QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2), dtype=gl.float32, layout=pv_mfma_layout
    )
    if QUERY_SEQ_LEN_POW2 == 1:
        query_offsets = (
            sequence_idx * stride_query_bs
            + mtp_idx * stride_query_qlen
            + query_group_size_offsets[:, None] * stride_query_group_size
            + head_size_offsets[None, :]
        )
        query_mask = (query_group_size_offsets[:, None] < query_group_size) & (
            head_size_offsets[None, :] < head_size
        )
        query_tensor = gl.amd.cdna3.buffer_load(
            ptr=query_ptr, offsets=query_offsets, mask=query_mask
        )
    else:
        mtp_query_offsets = (
            sequence_idx * stride_query_bs
            + (mtp_idx * QUERY_SEQ_LEN_POW2 + mtp_query_len_offsets[:, None, None])
            * stride_query_qlen
            + mtp_query_group_size_offsets[None, :, None] * stride_query_group_size
            + mtp_head_size_offsets[None, None, :]
        )
        mtp_query_mask = (
            (
                mtp_idx * QUERY_SEQ_LEN_POW2 + mtp_query_len_offsets[:, None, None]
                < query_seq_len
            )
            & (mtp_query_group_size_offsets[None, :, None] < query_group_size)
            & (mtp_head_size_offsets[None, None, :] < head_size)
        )
        mtp_query_tensor = gl.amd.cdna3.buffer_load(
            ptr=query_ptr, offsets=mtp_query_offsets, mask=mtp_query_mask
        )
    if SLIDING_WINDOW > 0:
        sequence_end_idx = context_length
        sequence_start_idx = context_length - SLIDING_WINDOW
        window_partition_start_idx = gl.maximum(
            0, sequence_start_idx // CONTEXT_PARTITION_SIZE
        )
        window_partition_end_idx = gl.cdiv(sequence_end_idx, CONTEXT_PARTITION_SIZE)
        window_partition_count = gl.maximum(
            0, window_partition_end_idx - window_partition_start_idx
        )
        sequence_split_count = gl.maximum(
            1, gl.num_programs(2) // CONTEXT_PARTITION_SIZE_PER_BLOCK
        )
        partitions_per_sequence_split = gl.cdiv(
            window_partition_count, sequence_split_count
        )
        local_partition_start_idx = (
            window_partition_start_idx
            + partitions_per_sequence_split * sequence_split_idx
        )
        local_partition_end_idx = gl.minimum(
            window_partition_end_idx,
            window_partition_start_idx
            + partitions_per_sequence_split * (sequence_split_idx + 1),
        )
        sequence_partition_start_idx = local_partition_start_idx + block_split_idx
        sequence_partition_end_idx = local_partition_end_idx + block_split_idx
    else:
        page_size = gl.cdiv(
            context_length, gl.num_programs(2) // CONTEXT_PARTITION_SIZE_PER_BLOCK
        )
        sequence_start_idx = page_size * sequence_split_idx

        sequence_end_idx = gl.minimum(context_length, page_size + sequence_start_idx)
        sequence_partition_start_idx = (
            sequence_start_idx // CONTEXT_PARTITION_SIZE + block_split_idx
        )
        sequence_partition_end_idx = (
            gl.cdiv(sequence_end_idx, CONTEXT_PARTITION_SIZE) + block_split_idx
        )
    block_indices = gl.arange(0, MAX_NUM_KV_BLOCKS_PER_COMPUTE, layout=block_id_layout)
    max_num_kv_blocks = gl.cdiv(context_length, KV_COMPUTE_BLOCK_SIZE)
    kv_block_start_idx = sequence_partition_start_idx * MAX_NUM_KV_BLOCKS_PER_COMPUTE
    if MAX_NUM_KV_BLOCKS_PER_COMPUTE == 1:
        if kv_block_start_idx < max_num_kv_blocks:
            kv_block_numbers = gl.load(
                block_tables_ptr
                + sequence_idx * stride_block_table_seq
                + kv_block_start_idx // CONTEXT_PARTITION_SIZE_PER_BLOCK
            )
        else:
            kv_block_numbers = 0
    else:
        kv_block_numbers = gl.amd.cdna3.buffer_load(
            ptr=block_tables_ptr
            + sequence_idx * stride_block_table_seq
            + kv_block_start_idx // CONTEXT_PARTITION_SIZE_PER_BLOCK,
            offsets=block_indices,
            mask=block_indices < (max_num_kv_blocks - kv_block_start_idx),
        )
    if QUERY_SEQ_LEN_POW2 == 1:
        max_logits_group_idx_in_len = max_logits_base_offsets
        query_token_idx = mtp_idx
        max_logits_base_offsets = mtp_idx * query_group_size + max_logits_base_offsets
    else:
        max_logits_query_len_idx = max_logits_base_offsets // ONE_QUERY_GROUP_SIZE_POW2
        max_logits_group_idx_in_len = (
            max_logits_base_offsets % ONE_QUERY_GROUP_SIZE_POW2
        )
        # Keep the per-row query token index in the original query sequence space.
        # `max_logits_base_offsets` below already includes the mtp split offset, so
        # recomputing from it later would double-count `mtp_idx` for the second split.
        query_token_idx = mtp_idx * QUERY_SEQ_LEN_POW2 + max_logits_query_len_idx
        max_logits_base_offsets = (
            query_token_idx
        ) * query_group_size + max_logits_group_idx_in_len
    if ONE_SHOT:
        if QUERY_SEQ_LEN_POW2 == 1:
            output_offsets = (
                sequence_idx * stride_output_bs
                + mtp_idx * stride_output_len
                + query_group_size_offsets[:, None] * stride_output_group_size
                + head_size_offsets[None, :]
            )
        else:
            output_offsets = (
                sequence_idx * stride_output_bs
                + (mtp_idx * QUERY_SEQ_LEN_POW2 + mtp_query_len_offsets[:, None, None])
                * stride_output_len
                + mtp_query_group_size_offsets[None, :, None] * stride_output_group_size
                + mtp_head_size_offsets[None, None, :]
            )
    else:
        output_offsets = sequence_idx * stride_output_bs
        output_offsets += (
            split_idx * stride_output_kv_head
            + output_group_offsets[:, None] * stride_output_group_size
            + output_head_size_offsets[None, :]
        )

        max_logits_offsets = (
            sequence_idx * stride_max_logits_seq
            + split_idx * stride_max_logits_part
            + max_logits_base_offsets
        )
    qk_row_mask = gl.convert_layout(
        query_row_mask_1d, layout=gl.SliceLayout(1, qk_linear_layout)
    )
    pv_row_mask = gl.convert_layout(
        qk_row_mask, layout=gl.SliceLayout(1, pv_mfma_layout)
    )
    if ONE_SHOT:
        if QUERY_SEQ_LEN_POW2 == 1:
            output_mask = query_mask
        else:
            output_mask = mtp_query_mask
    else:
        output_mask = pv_row_mask[:, None] & (
            output_head_size_offsets[None, :] < head_size
        )

    if (sequence_start_idx >= context_length) | (
        sequence_partition_start_idx >= sequence_partition_end_idx
    ):
        if not ONE_SHOT:
            store_temporary_result(
                max_logits,
                exp_sums,
                attention_accumulator.to(OUTPUT_DTYPE),
                max_logits_ptr,
                exp_sums_ptr,
                output_ptr,
                max_logits_offsets,
                output_offsets,
                qk_row_mask,
                output_mask,
            )
        return  # No computation needed for this partition

    if QUERY_SEQ_LEN_POW2 > 1:
        mtp_query_tensor = gl.reshape(
            mtp_query_tensor, [QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2]
        )

    query_shared = gl.allocate_shared_memory(
        COMPUTE_TYPE, [QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2], shared_query_layout
    )
    probs_shared = gl.allocate_shared_memory(
        COMPUTE_TYPE,
        [QUERY_GROUP_SIZE_POW2, CONTEXT_PARTITION_SIZE],
        shared_probs_layout,
    )
    qk_column_indices = gl.arange(
        0, CONTEXT_PARTITION_SIZE, layout=gl.SliceLayout(0, qk_linear_layout)
    )
    value_scale_shared = gl.allocate_shared_memory(
        gl.float32,
        [CONTEXT_PARTITION_SIZE],
        shared_value_scale_layout,
    )
    key_scale_shared = gl.allocate_shared_memory(
        gl.float32,
        [CONTEXT_PARTITION_SIZE],
        shared_key_scale_layout,
    )
    if QUERY_QUANT_MODE == 0:
        query_scale_value = gl.load(query_scale)
    elif QUERY_QUANT_MODE == 1:
        if QUERY_SEQ_LEN_POW2 == 1:
            query_scale_offsets = (
                sequence_idx * stride_query_scale_bs
                + mtp_idx * stride_query_scale_qlen
                + query_group_size_offsets[:, None]
            )
            query_scale_mask = query_group_size_offsets[:, None] < query_group_size
            query_scale_value = gl.amd.cdna3.buffer_load(
                ptr=query_scale,
                offsets=query_scale_offsets,
                mask=query_scale_mask,
            )
        else:
            query_scale_offsets = (
                sequence_idx * stride_query_scale_bs
                + (mtp_idx * QUERY_SEQ_LEN_POW2 + mtp_query_len_offsets[:, None, None])
                * stride_query_scale_qlen
                + mtp_query_group_size_offsets[None, :, None]
            )
            query_scale_mask = (
                mtp_idx * QUERY_SEQ_LEN_POW2 + mtp_query_len_offsets[:, None, None]
                < query_seq_len
            ) & (mtp_query_group_size_offsets[None, :, None] < query_group_size)
            query_scale_value = gl.amd.cdna3.buffer_load(
                ptr=query_scale,
                offsets=query_scale_offsets,
                mask=query_scale_mask,
            )
            query_scale_value = gl.reshape(
                query_scale_value, [QUERY_GROUP_SIZE_POW2, 1]
            )
        query_scale_value = gl.convert_layout(
            query_scale_value, layout=qk_linear_layout
        )

    # ==================== SEQUENCE PROCESSING ====================

    if ONE_SHOT and sinks_ptr is not None:
        # sinks_ptr is per-query-head: [num_query_heads] where
        # num_query_heads = num_kv_heads * query_group_size.
        # It is shared across query positions (query_seq_len).
        sinks_values = gl.load(
            sinks_ptr + max_logits_group_idx_in_len,
            mask=qk_row_mask,
            other=float("-inf"),
        )

    if KV_QUANT_MODE == 0:
        # Per-tensor quantization
        key_scale_value = gl.load(key_scale)
        value_scale_value = gl.load(value_scale)

    if QUERY_SEQ_LEN_POW2 == 1:
        mtp_query_tensor = query_tensor.to(COMPUTE_TYPE)
    else:
        mtp_query_tensor = mtp_query_tensor.to(COMPUTE_TYPE)
    query_shared.store(mtp_query_tensor)
    page_offset = (
        kv_block_start_idx % CONTEXT_PARTITION_SIZE_PER_BLOCK
    ) * CONTEXT_PARTITION_SIZE
    kv_block_numbers = kv_block_numbers.to(gl.int64)
    if MAX_NUM_KV_BLOCKS_PER_COMPUTE == 1:
        key_block_offsets = (
            kv_block_numbers * stride_key_block
            + head_size_split_offsets[None, :, None, None] * stride_key_head_split
            # Use runtime stride for KV block element (may be padded for large blocks).
            + (page_offset + block_element_offsets)[None, None, :, None]
            * stride_key_block_elem
            + contiguous_kv_element_offsets[None, None, None, :]
        )
    else:
        key_block_offsets = (
            kv_block_numbers[:, None, None, None] * stride_key_block
            + head_size_split_offsets[None, :, None, None] * stride_key_head_split
            # Use runtime stride for KV block element (may be padded for large blocks).
            + (page_offset + block_element_offsets)[None, None, :, None]
            * stride_key_block_elem
            + contiguous_kv_element_offsets[None, None, None, :]
        )

    key_tensor = gl.load(key_cache_ptr + key_block_offsets)
    query_converted = query_shared.load(qk_lhs_operand_layout)
    for sequence_partition_idx in range(
        sequence_partition_start_idx,
        sequence_partition_end_idx,
        CONTEXT_PARTITION_SIZE_PER_BLOCK,
    ):
        kv_block_start_idx2 = (
            kv_block_start_idx
            + MAX_NUM_KV_BLOCKS_PER_COMPUTE * CONTEXT_PARTITION_SIZE_PER_BLOCK
        )
        # Prepare QK MFMA while key loads (these don't depend on key data)
        qk_accumulator = gl.zeros(
            (QUERY_GROUP_SIZE_POW2, CONTEXT_PARTITION_SIZE),
            dtype=gl.float32,
            layout=qk_mfma_layout,
        )
        # Load key quantization scales if needed (overlaps with key tensor load)
        key_tensor = _amd_iglp_sched_barrier(key_tensor, 0x0)
        if KV_QUANT_MODE == 1:
            # Per-token quantization - prepare offsets while key loads
            if MAX_NUM_KV_BLOCKS_PER_COMPUTE == 1:
                key_scale_offsets = kv_block_numbers * kv_scale_stride_0 + (
                    page_offset + kv_scale_indices
                )
            else:
                key_scale_offsets = (
                    kv_block_numbers[:, None, None, None] * kv_scale_stride_0
                    + (page_offset + block_element_offsets)[None, None, :, None]
                )
            # Optimize: Load both scales with VMEM scheduling, overlap with key reshape
            key_scale_value_blocked = gl.load(key_scale + key_scale_offsets)
            # Shared memory is used as a 1D staging buffer for layout conversion.
            key_scale_value_blocked = gl.reshape(
                key_scale_value_blocked, [CONTEXT_PARTITION_SIZE]
            )
            key_scale_shared.store(key_scale_value_blocked)
            value_scale_value_blocked = gl.load(value_scale + key_scale_offsets)
            value_scale_value_blocked = gl.reshape(
                value_scale_value_blocked, [CONTEXT_PARTITION_SIZE]
            )
            value_scale_shared.store(value_scale_value_blocked)

        # Reshape key tensor for matrix multiplication
        key_converted = gl.permute(key_tensor, [1, 3, 0, 2])
        key_converted = gl.reshape(
            key_converted, [HEAD_SIZE_POW2, CONTEXT_PARTITION_SIZE]
        )

        # ==================== VALUE LOADING WITH QK MFMA OVERLAP ====================
        # Convert key layout for MFMA (query_converted and qk_accumulator already prepared above)
        key_converted = gl.convert_layout(key_converted, layout=qk_rhs_operand_layout)
        key_converted = key_converted.to(COMPUTE_TYPE)
        if VALUE_TRANSPOSED:
            # Load values from transposed cache layout
            if MAX_NUM_KV_BLOCKS_PER_COMPUTE == 1:
                kv_block_numbers_reshaped = kv_block_numbers
                value_block_offsets = (
                    kv_block_numbers_reshaped * stride_value_block
                    + (page_offset // KV_16B_ELEMENT_COUNT + value_dim1_offsets)[
                        None, :, None, None
                    ]
                    * stride_value_head_size
                    + value_dim2_offsets[None, None, :, None] * KV_16B_ELEMENT_COUNT
                    + value_dim3_offsets[None, None, None, :]
                )

            else:
                kv_block_numbers_reshaped = gl.convert_layout(
                    kv_block_numbers,
                    layout=gl.SliceLayout(
                        1, gl.SliceLayout(2, gl.SliceLayout(3, blocked_value_layout))
                    ),
                )

                value_block_offsets = (
                    kv_block_numbers_reshaped[:, None, None, None] * stride_value_block
                    + (page_offset // KV_16B_ELEMENT_COUNT + value_dim1_offsets)[
                        None, :, None, None
                    ]
                    * stride_value_head_size
                    + value_dim2_offsets[None, None, :, None] * KV_16B_ELEMENT_COUNT
                    + value_dim3_offsets[None, None, None, :]
                )
            value_tensor = gl.load(value_cache_ptr + value_block_offsets)

            # Permute and reshape for matrix multiplication
            value_tensor = gl.permute(value_tensor, [0, 1, 3, 2])
        else:
            # Load values from standard cache layout
            if MAX_NUM_KV_BLOCKS_PER_COMPUTE == 1:
                kv_block_numbers_reshaped = kv_block_numbers
                value_block_offsets = (
                    kv_block_numbers_reshaped * stride_value_block
                    + value_dim1_offsets[None, :, None] * stride_value_head_size
                    + (page_offset + value_dim2_offsets)[None, None, :]
                )
            else:
                kv_block_numbers_reshaped = gl.convert_layout(
                    kv_block_numbers,
                    layout=gl.SliceLayout(1, gl.SliceLayout(2, blocked_value_layout)),
                )
                value_block_offsets = (
                    kv_block_numbers_reshaped[:, None, None] * stride_value_block
                    + value_dim1_offsets[None, :, None] * stride_value_head_size
                    + (page_offset + value_dim2_offsets)[None, None, :]
                )

            # Schedule: Start value VMEM load, then QK MFMA
            value_tensor = gl.load(value_cache_ptr + value_block_offsets)

            # Permute and resape for matrix multiplication
            value_tensor = gl.permute(value_tensor, [0, 2, 1])
        value_tensor = gl.reshape(
            value_tensor, [CONTEXT_PARTITION_SIZE, HEAD_SIZE_POW2]
        )
        # Compute QK attention scores using MFMA (overlaps with value load)
        attention_scores = gl.amd.cdna3.mfma(
            query_converted, key_converted, qk_accumulator
        )
        attention_scores = gl.reshape(
            attention_scores, [QUERY_GROUP_SIZE_POW2, CONTEXT_PARTITION_SIZE]
        )
        if KV_QUANT_MODE == 1:
            value_tensor = _amd_iglp_sched_group_barrier(value_tensor, VMEM_LOAD, 1, 0)
            value_tensor = _amd_iglp_sched_group_barrier(value_tensor, MFMA, 2, 0)
            value_tensor = _amd_iglp_sched_group_barrier(value_tensor, VMEM_LOAD, 1, 0)
            value_tensor = _amd_iglp_sched_group_barrier(value_tensor, MFMA, 2, 0)
        value_tensor = _amd_iglp_sched_group_barrier(value_tensor, VMEM_LOAD, 2, 0)
        value_tensor = _amd_iglp_sched_group_barrier(value_tensor, MFMA, 2, 0)
        value_tensor = _amd_iglp_sched_group_barrier(value_tensor, VMEM_LOAD, 2, 0)
        value_tensor = _amd_iglp_sched_group_barrier(value_tensor, MFMA, 2, 0)
        value_tensor = _amd_iglp_sched_group_barrier(value_tensor, VMEM_LOAD, 2, 0)
        value_tensor = _amd_iglp_sched_group_barrier(value_tensor, MFMA, 2, 0)
        value_tensor = _amd_iglp_sched_group_barrier(value_tensor, VMEM_LOAD, 2, 0)
        value_tensor = _amd_iglp_sched_group_barrier(value_tensor, MFMA, 2, 0)

        value_tensor = _amd_iglp_sched_barrier(value_tensor, 0x0)
        qk_column_offsets = (
            kv_block_start_idx * KV_COMPUTE_BLOCK_SIZE + qk_column_indices
        )

        # Apply quantization scaling to attention scores
        if KV_QUANT_MODE >= 0:
            if KV_QUANT_MODE == 1:
                key_scale_value = key_scale_shared.load(
                    gl.SliceLayout(0, qk_linear_layout)
                )
                # key_scale_value = gl.convert_layout(key_scale_value_blocked, layout=gl.SliceLayout(0, qk_linear_layout))
                key_scale_value = key_scale_value[None, :]
            if QUERY_QUANT_MODE >= 0:
                qk_scale_value = softmax_scale * query_scale_value * key_scale_value
            else:
                qk_scale_value = softmax_scale * key_scale_value
        else:
            if QUERY_QUANT_MODE >= 0:
                qk_scale_value = softmax_scale * query_scale_value
            else:
                qk_scale_value = softmax_scale

        # ==================== ATTENTION MASKING ====================

        if QUERY_SEQ_LEN_POW2 == 1:
            if IS_CAUSAL:
                sequence_position_extension = query_seq_len - 1 - mtp_idx
                causal_mask = (
                    sequence_position_extension + qk_column_offsets[None, :]
                    < sequence_end_idx
                )
                if SLIDING_WINDOW > 0:
                    causal_mask = causal_mask & (
                        sequence_position_extension + qk_column_offsets[None, :]
                        >= sequence_start_idx + query_seq_len
                    )
                else:
                    causal_mask = causal_mask & (
                        sequence_position_extension + qk_column_offsets[None, :]
                        >= sequence_start_idx
                    )
            else:
                causal_mask = qk_column_offsets[None, :] < sequence_end_idx
                if SLIDING_WINDOW > 0:
                    causal_mask = causal_mask & (
                        qk_column_offsets[None, :] >= sequence_start_idx + mtp_idx + 1
                    )
                else:
                    causal_mask = causal_mask & (
                        qk_column_offsets[None, :] >= sequence_start_idx
                    )
        else:

            if IS_CAUSAL:
                sequence_position_extension = query_seq_len - 1 - query_token_idx
                causal_mask = (
                    sequence_position_extension[:, None] + qk_column_offsets[None, :]
                    < sequence_end_idx
                )
                if SLIDING_WINDOW > 0:
                    causal_mask = causal_mask & (
                        sequence_position_extension[:, None]
                        + qk_column_offsets[None, :]
                        >= sequence_start_idx + query_seq_len
                    )
                else:
                    causal_mask = causal_mask & (
                        sequence_position_extension[:, None]
                        + qk_column_offsets[None, :]
                        >= sequence_start_idx
                    )
            else:
                causal_mask = qk_column_offsets[None, :] < sequence_end_idx
                if SLIDING_WINDOW > 0:
                    causal_mask = causal_mask & (
                        qk_column_offsets[None, :]
                        >= sequence_start_idx + query_token_idx[:, None] + 1
                    )
                else:
                    causal_mask = causal_mask & (
                        qk_column_offsets[None, :] >= sequence_start_idx
                    )
        if MAX_NUM_KV_BLOCKS_PER_COMPUTE == 1:
            if kv_block_start_idx2 < max_num_kv_blocks:
                kv_block_numbers2 = gl.load(
                    block_tables_ptr
                    + sequence_idx * stride_block_table_seq
                    + kv_block_start_idx2 // CONTEXT_PARTITION_SIZE_PER_BLOCK
                )
            else:
                kv_block_numbers2 = 0
        else:
            kv_block_numbers2 = gl.amd.cdna3.buffer_load(
                ptr=block_tables_ptr
                + sequence_idx * stride_block_table_seq
                + kv_block_start_idx2 // CONTEXT_PARTITION_SIZE_PER_BLOCK,
                offsets=block_indices,
                mask=block_indices < (max_num_kv_blocks - kv_block_start_idx2),
            )

        kv_block_numbers2 = _amd_iglp_sched_barrier(kv_block_numbers2, 0x0)
        boundary_mask = qk_row_mask[:, None] & causal_mask

        attention_scores = gl.convert_layout(attention_scores, layout=qk_linear_layout)
        attention_scores = qk_scale_value * attention_scores
        # Apply masking to attention scores (if [0, CONTEXT_PARTITION_SIZE) are all -inf, the result will be NaN, so we use -3.4e38 other than -inf)
        attention_scores = gl.where(boundary_mask, attention_scores, (-3.4e38))

        # ==================== SOFTMAX COMPUTATION ====================
        # Optimization: For per-token quant mode, load value_scale early and fuse its reduction
        # with softmax max reduction to share the same barrier synchronization
        if KV_QUANT_MODE == 1:
            # Load value_scale for fused reduction (reduces one barrier)
            value_scale_value = value_scale_shared.load(
                gl.SliceLayout(0, qk_linear_layout)
            )
            valid_token_mask = qk_column_offsets < context_length
            # Mask out value_scale of invalid tokens
            value_scale_value = gl.where(valid_token_mask, value_scale_value, 0.0)
            value_scale_broadcast = tl.broadcast_to(
                value_scale_value[None, :], attention_scores.shape
            )
            current_max_logits, value_scale_max_2d = gl.reduce(
                (attention_scores, value_scale_broadcast),
                axis=1,
                combine_fn=_fused_max_combine,
            )
            # All rows are identical (broadcast from 1D); collapse to scalar.
            # This is a cheap intra-wave reduction, no extra cross-wave barrier.
            value_scale_max = gl.max(value_scale_max_2d, axis=0)
        else:
            # Update running maximum for numerical stability
            current_max_logits = gl.max(attention_scores, axis=1)

        new_max_logits = gl.maximum(max_logits, current_max_logits)

        accumulator_scale = tl.math.exp2((max_logits - new_max_logits) * LOG2_E)
        # Compute attention probabilities
        attention_probs = tl.math.exp2(
            (attention_scores - new_max_logits[:, None]) * LOG2_E
        )

        exp_sums = accumulator_scale * exp_sums + gl.sum(attention_probs, axis=1)
        kv_block_numbers2 = kv_block_numbers2.to(gl.int64)
        if MAX_NUM_KV_BLOCKS_PER_COMPUTE == 1:
            key_block_offsets2 = (
                kv_block_numbers2 * stride_key_block
                + head_size_split_offsets[None, :, None, None] * stride_key_head_split
                # Use runtime stride for KV block element (may be padded for large blocks).
                + (page_offset + block_element_offsets)[None, None, :, None]
                * stride_key_block_elem
                + contiguous_kv_element_offsets[None, None, None, :]
            )
        else:
            key_block_offsets2 = (
                kv_block_numbers2[:, None, None, None] * stride_key_block
                + head_size_split_offsets[None, :, None, None] * stride_key_head_split
                # Use runtime stride for KV block element (may be padded for large blocks).
                + (page_offset + block_element_offsets)[None, None, :, None]
                * stride_key_block_elem
                + contiguous_kv_element_offsets[None, None, None, :]
            )

        # ==================== VALUE ACCUMULATION ====================
        # Handle value quantization scaling for FP8
        if KV_QUANT_MODE >= 0:
            if KV_QUANT_MODE == 1:
                # Per-token quantization scaling
                # value_scale_value and value_scale_max already computed above (fused with softmax max)
                # Scale the maximum value of value_scale to FP8_MAX_VALUE to improve the precision of P * V
                # Optimization: use fast reciprocal (v_rcp_f32) instead of full IEEE-754 division
                # Use HIP libdevice fast_dividef which generates v_rcp_f32 + one Newton-Raphson step
                # This reduces ~12 instructions to ~3-4 instructions with acceptable precision loss
                # The relative error is < 2^-22 which is sufficient for FP8 quantization scaling
                inv_value_scale_max = hip_libdevice.fast_dividef(
                    1.0, value_scale_max + 1e-8
                )
                fp8_inv_scale = float(FP8_MAX_VALUE) * inv_value_scale_max
                value_scale_value = value_scale_value * fp8_inv_scale
                attention_probs = value_scale_value[None, :] * attention_probs
                # Use multiply by reciprocal instead of divide (precomputed 1/FP8_MAX_VALUE)
                probability_scale = value_scale_max * (1.0 / float(FP8_MAX_VALUE))
            elif KV_QUANT_MODE == 0:
                # Per-tensor quantization scaling
                probability_scale = value_scale_value
            else:
                raise ValueError(f"Invalid KV_QUANT_MODE: {KV_QUANT_MODE}")

        # Convert attention probabilities to compute type for MFMA operation
        # Convert layouts for PV MFMA operation
        attention_probs = attention_probs.to(COMPUTE_TYPE)
        probs_shared.store(attention_probs)
        values_converted = value_tensor.to(COMPUTE_TYPE)
        values_converted = gl.convert_layout(
            values_converted, layout=pv_rhs_operand_layout
        )

        accumulator_scale_expanded = gl.convert_layout(
            accumulator_scale[:, None], layout=pv_mfma_layout
        )
        attention_accumulator *= accumulator_scale_expanded

        pv_accumulator = gl.zeros(
            (QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2),
            dtype=gl.float32,
            layout=pv_mfma_layout,
        )
        probs_converted = probs_shared.load(pv_lhs_operand_layout)
        key_block_offsets2 = _amd_iglp_sched_barrier(key_block_offsets2, 0x0)
        key_tensor2 = gl.load(key_cache_ptr + key_block_offsets2)

        attention_output = gl.amd.cdna3.mfma(
            probs_converted, values_converted, pv_accumulator
        )
        key_block_offsets2 = _amd_iglp_sched_group_barrier(
            key_block_offsets2, VMEM_LOAD, 2, 1
        )
        key_block_offsets2 = _amd_iglp_sched_group_barrier(
            key_block_offsets2, MFMA, 4, 1
        )
        key_block_offsets2 = _amd_iglp_sched_group_barrier(
            key_block_offsets2, VMEM_LOAD, 2, 1
        )
        key_block_offsets2 = _amd_iglp_sched_group_barrier(
            key_block_offsets2, MFMA, 4, 1
        )
        key_block_offsets2 = _amd_iglp_sched_group_barrier(
            key_block_offsets2, VMEM_LOAD, 2, 1
        )
        key_block_offsets2 = _amd_iglp_sched_group_barrier(
            key_block_offsets2, MFMA, 4, 1
        )
        key_block_offsets2 = _amd_iglp_sched_group_barrier(
            key_block_offsets2, VMEM_LOAD, 2, 1
        )
        key_block_offsets2 = _amd_iglp_sched_group_barrier(
            key_block_offsets2, MFMA, 4, 1
        )
        key_block_offsets2 = _amd_iglp_sched_barrier(key_block_offsets2, 0x0)
        if KV_QUANT_MODE >= 0:
            attention_accumulator += probability_scale * attention_output
        else:
            attention_accumulator += attention_output
        max_logits = new_max_logits
        if (
            sequence_partition_idx + CONTEXT_PARTITION_SIZE_PER_BLOCK
            < sequence_partition_end_idx
        ):
            kv_block_numbers = kv_block_numbers2
            key_tensor = key_tensor2
            kv_block_start_idx = kv_block_start_idx2

    # ==================== SINKS HANDLING ====================
    # Add sinks contribution to exp_sums (does not contribute to attention output)
    if ONE_SHOT and sinks_ptr is not None:
        exp_sums += tl.math.exp2((sinks_values.to(gl.float32) - max_logits) * LOG2_E)

    # ==================== OUTPUT NORMALIZATION AND STORING ====================
    # Normalize attention output by softmax denominator
    # Guard against division by zero when all tokens are masked for a query position
    exp_sums_safe = tl.where(exp_sums > 0, exp_sums, 1.0)
    exp_sums_reciprocal = 1.0 / exp_sums_safe
    exp_sums_reciprocal_cvt = gl.convert_layout(
        exp_sums_reciprocal[:, None], layout=pv_mfma_layout
    )
    attention_accumulator = attention_accumulator * exp_sums_reciprocal_cvt

    attention_accumulator = attention_accumulator.to(OUTPUT_DTYPE)

    if not ONE_SHOT:
        # Store results to global memory
        store_temporary_result(
            max_logits,
            exp_sums,
            attention_accumulator,
            max_logits_ptr,
            exp_sums_ptr,
            output_ptr,
            max_logits_offsets,
            output_offsets,
            qk_row_mask,
            output_mask,
        )
    else:
        if QUERY_SEQ_LEN_POW2 == 1:
            attention_accumulator = gl.reshape(
                attention_accumulator,
                [ONE_QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2],
            )
            attention_accumulator = gl.convert_layout(
                attention_accumulator, layout=blocked_query_layout
            )
        else:
            attention_accumulator = gl.reshape(
                attention_accumulator,
                [QUERY_SEQ_LEN_POW2, ONE_QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2],
            )
            attention_accumulator = gl.convert_layout(
                attention_accumulator, layout=mtp_blocked_query_layout
            )

        gl.amd.cdna3.buffer_store(
            stored_value=attention_accumulator,
            ptr=output_ptr,
            offsets=output_offsets,
            mask=output_mask,
        )



_KV_BLOCK_SIZE = 16
_CONTEXT_PARTITION_SIZE = 256


def run(*, query, key_cache, value_cache, block_table, seq_lens, scale, out):
    """pa_decode_gluon for bf16 query/KV, one KV head, query length 1, 16-token pages.

    query [num_seqs, num_query_heads, head_size]; key_cache [num_blocks, 1, head_size // 8, 16, 8];
    value_cache [num_blocks, 1, head_size, 16]; block_table [num_seqs, max_blocks]; seq_lens [num_seqs].
    max_context_partition_num is 1 (the wrapper's one-shot mode).
    """
    batch_size, num_query_heads, head_size = query.shape
    num_kv_heads = key_cache.shape[1]
    query_length = 1
    query_group_size = num_query_heads // num_kv_heads
    kv_block_size = key_cache.shape[-2]
    assert num_kv_heads == 1, "the head_1 kernel serves one KV head"
    assert kv_block_size == _KV_BLOCK_SIZE
    assert query_group_size * query_length <= 64
    cdna_version = get_cdna_version()
    max_context_partition_num = 1
    grid = (batch_size, num_kv_heads, max_context_partition_num)

    # Buffers pa_decode_gluon allocates for the partition results; unused in one-shot mode.
    exp_sums = torch.empty(batch_size, num_kv_heads, max_context_partition_num,
                           query_length * query_group_size, device=query.device, dtype=torch.float32)
    max_logits = torch.empty_like(exp_sums)

    query_5d = query.reshape(batch_size, query_length, num_kv_heads, query_group_size, head_size)
    output_5d = out.reshape(batch_size, query_length, num_kv_heads, query_group_size, head_size)

    # _paged_attention_decode_v2_with_dot_kernel_reshape_wrapper, PS branch, num_kv_heads == 1.
    head_size_pow2 = triton.next_power_of_2(head_size)
    query_seq_len_pow2 = triton.next_power_of_2(query_length)
    one_query_group_size_pow2 = triton.next_power_of_2(query_group_size)
    assert one_query_group_size_pow2 >= 16
    grid = (batch_size, query_length * num_kv_heads, max_context_partition_num)
    query_seq_len_pow2 = 1

    paged_attention_decode_sliding_window_head_1[grid](
        exp_sums,
        max_logits,
        output_5d,
        query_5d,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        scale,
        None,
        None,
        None,
        None,
        exp_sums.stride(0),
        exp_sums.stride(1),
        exp_sums.stride(2),
        output_5d.stride(0),
        output_5d.stride(1),
        output_5d.stride(2),
        output_5d.stride(3),
        query_5d.stride(0),
        query_5d.stride(1),
        query_5d.stride(2),
        query_5d.stride(3),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        block_table.stride(0),
        0,
        0,
        0,
        0,
        0,
        query_seq_len=query_length,
        query_group_size=query_group_size,
        head_size=head_size,
        COMPUTE_TYPE=tl.bfloat16,
        QUERY_SEQ_LEN_POW2=query_seq_len_pow2,
        ONE_QUERY_GROUP_SIZE_POW2=one_query_group_size_pow2,
        HEAD_SIZE_POW2=head_size_pow2,
        KV_BLOCK_SIZE=kv_block_size,
        CONTEXT_PARTITION_SIZE=_CONTEXT_PARTITION_SIZE,
        QUERY_QUANT_MODE=-1,
        KV_QUANT_MODE=-1,
        VALUE_TRANSPOSED=False,
        IS_CAUSAL=False,
        FP8_MAX_VALUE=1.0,
        SLIDING_WINDOW=0,
        CDNA_VERSION=cdna_version,
        ONE_SHOT=True,
    )
