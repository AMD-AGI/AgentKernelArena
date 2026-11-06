# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang (https://github.com/sgl-project/sglang), Apache License 2.0:
# python/sglang/kernels/ops/attention/nsa_triton_decode/triton_mla_kernels_decode_fused.py
"""
Fused Gather+Dequant+Attention Kernel for DSV4 (d_qk=512)

This module implements a fused kernel that combines:
1. Gather: Load KV from sparse indices
2. Dequant: FP8 to BF16 dequantization
3. Attention: Compute attention scores and output

Supports:
- DSV4 (d_qk=512): 7 tiles of 64, uint8 scales
- All configs: with/without topk_length, with/without attn_sink

Vendored from SGLang python/sglang/kernels/ops/attention/nsa_triton_decode/
triton_mla_kernels_decode_fused.py (the single-scope path of
triton_mla_kernels_decode_optimized.py, selected on ROCm by
SGLANG_HACK_FLASHMLA_BACKEND=triton). _process_kv_block_aggressive and
_fused_gather_attn_dsv4_kernel are unchanged. Changes: the gfx942 probe is
inlined for gfx950 (OCP e4m3fn), @triton.autotune is replaced by one of its
configurations, the split-K (topk >= 8192) and >2 GB buffer-ops branches that
this scenario never takes are dropped, and the outputs are written into the
caller's tensors.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

# gfx942/MI300/MI325 stores e4m3fnuz (bias 8);
# gfx950/MI350 and CUDA store OCP e4m3fn (bias 7).
# This task targets gfx950: is_gfx942_supported() is False.
_KV_FP8_TY = tl.float8e4nv


def _bucket_total_tokens(total_tokens: int) -> int:
    """Round total_tokens up to the nearest power of 2 for autotune key stability."""
    if total_tokens <= 0:
        return 1
    n = 1
    while n < total_tokens:
        n <<= 1
    return n


# ============================================================================
# Constants for DSV4 layout
# ============================================================================
DSV4_D_QK = 512
DSV4_D_NOPE = 448
DSV4_D_ROPE = 64
DSV4_D_V = 512
DSV4_TILE_SIZE = 64

DSV4_NUM_TILES = 7
DSV4_BYTES_PER_TOKEN_DATA = 576  # 448 nope + 128 rope
DSV4_BYTES_PER_TOKEN_SCALE = 8  # 7 scales + 1 padding

# ============================================================================
# Helper: Process KV block and compute QK scores + accumulator update
# This is the core computation shared by both single and dual scope kernels
# ============================================================================


@triton.jit
def _process_kv_block_aggressive(
    # KV cache parameters
    kv_block_base,
    nope_rope_offset,
    scale_base_offset,
    valid,
    valid_2d,
    # Query tiles
    q_0,
    q_1,
    q_2,
    q_3,
    q_4,
    q_5,
    q_6,
    q_7,
    # Accumulators (passed by reference via return)
    acc_0,
    acc_1,
    acc_2,
    acc_3,
    acc_4,
    acc_5,
    acc_6,
    acc_7,
    # Softmax state
    m_i,
    l_i,
    # Other parameters
    offs_tile,
    sm_scale,
    # Constants
    TILE_SIZE: tl.constexpr,
    D_NOPE: tl.constexpr,
    LOG2E: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Process one block of KV tokens with batch loading.
    Key optimization: Load all KV tiles first, then process them.
    """
    NEG_INF = float("-inf")

    scale_ptrs = kv_block_base + scale_base_offset
    scale_uint8_0 = tl.load(scale_ptrs, mask=valid, other=127).to(tl.uint8)
    scale_uint8_1 = tl.load(scale_ptrs + 1, mask=valid, other=127).to(tl.uint8)
    scale_uint8_2 = tl.load(scale_ptrs + 2, mask=valid, other=127).to(tl.uint8)
    scale_uint8_3 = tl.load(scale_ptrs + 3, mask=valid, other=127).to(tl.uint8)
    scale_uint8_4 = tl.load(scale_ptrs + 4, mask=valid, other=127).to(tl.uint8)
    scale_uint8_5 = tl.load(scale_ptrs + 5, mask=valid, other=127).to(tl.uint8)
    scale_uint8_6 = tl.load(scale_ptrs + 6, mask=valid, other=127).to(tl.uint8)

    tile_base = kv_block_base[:, None] + nope_rope_offset[:, None]

    # Batch load all tiles
    nope_uint8_0 = tl.load(tile_base + offs_tile[None, :], mask=valid_2d, other=0)
    nope_uint8_1 = tl.load(
        tile_base + TILE_SIZE + offs_tile[None, :], mask=valid_2d, other=0
    )
    nope_uint8_2 = tl.load(
        tile_base + 2 * TILE_SIZE + offs_tile[None, :], mask=valid_2d, other=0
    )
    nope_uint8_3 = tl.load(
        tile_base + 3 * TILE_SIZE + offs_tile[None, :], mask=valid_2d, other=0
    )
    nope_uint8_4 = tl.load(
        tile_base + 4 * TILE_SIZE + offs_tile[None, :], mask=valid_2d, other=0
    )
    nope_uint8_5 = tl.load(
        tile_base + 5 * TILE_SIZE + offs_tile[None, :], mask=valid_2d, other=0
    )
    nope_uint8_6 = tl.load(
        tile_base + 6 * TILE_SIZE + offs_tile[None, :], mask=valid_2d, other=0
    )
    rope_ptrs = tile_base + D_NOPE + offs_tile[None, :] * 2
    rope_lo = tl.load(rope_ptrs, mask=valid_2d, other=0).to(tl.uint16)
    rope_hi = tl.load(rope_ptrs + 1, mask=valid_2d, other=0).to(tl.uint16)

    scale_bf16_0 = tl.math.exp2(scale_uint8_0.to(tl.float32) - 127.0).to(tl.bfloat16)
    scale_bf16_1 = tl.math.exp2(scale_uint8_1.to(tl.float32) - 127.0).to(tl.bfloat16)
    scale_bf16_2 = tl.math.exp2(scale_uint8_2.to(tl.float32) - 127.0).to(tl.bfloat16)
    scale_bf16_3 = tl.math.exp2(scale_uint8_3.to(tl.float32) - 127.0).to(tl.bfloat16)
    scale_bf16_4 = tl.math.exp2(scale_uint8_4.to(tl.float32) - 127.0).to(tl.bfloat16)
    scale_bf16_5 = tl.math.exp2(scale_uint8_5.to(tl.float32) - 127.0).to(tl.bfloat16)
    scale_bf16_6 = tl.math.exp2(scale_uint8_6.to(tl.float32) - 127.0).to(tl.bfloat16)

    qk = tl.zeros([BLOCK_H, BLOCK_N], dtype=tl.float32)

    nope_fp8_0 = nope_uint8_0.to(_KV_FP8_TY, bitcast=True)
    kv_0 = (nope_fp8_0.to(tl.bfloat16) * scale_bf16_0[:, None]).to(tl.bfloat16)
    kv_0 = tl.where(valid_2d, kv_0, 0.0)
    qk += tl.dot(q_0, tl.trans(kv_0)).to(tl.float32)

    nope_fp8_1 = nope_uint8_1.to(_KV_FP8_TY, bitcast=True)
    kv_1 = (nope_fp8_1.to(tl.bfloat16) * scale_bf16_1[:, None]).to(tl.bfloat16)
    kv_1 = tl.where(valid_2d, kv_1, 0.0)
    qk += tl.dot(q_1, tl.trans(kv_1)).to(tl.float32)

    nope_fp8_2 = nope_uint8_2.to(_KV_FP8_TY, bitcast=True)
    kv_2 = (nope_fp8_2.to(tl.bfloat16) * scale_bf16_2[:, None]).to(tl.bfloat16)
    kv_2 = tl.where(valid_2d, kv_2, 0.0)
    qk += tl.dot(q_2, tl.trans(kv_2)).to(tl.float32)

    nope_fp8_3 = nope_uint8_3.to(_KV_FP8_TY, bitcast=True)
    kv_3 = (nope_fp8_3.to(tl.bfloat16) * scale_bf16_3[:, None]).to(tl.bfloat16)
    kv_3 = tl.where(valid_2d, kv_3, 0.0)
    qk += tl.dot(q_3, tl.trans(kv_3)).to(tl.float32)

    nope_fp8_4 = nope_uint8_4.to(_KV_FP8_TY, bitcast=True)
    kv_4 = (nope_fp8_4.to(tl.bfloat16) * scale_bf16_4[:, None]).to(tl.bfloat16)
    kv_4 = tl.where(valid_2d, kv_4, 0.0)
    qk += tl.dot(q_4, tl.trans(kv_4)).to(tl.float32)

    nope_fp8_5 = nope_uint8_5.to(_KV_FP8_TY, bitcast=True)
    kv_5 = (nope_fp8_5.to(tl.bfloat16) * scale_bf16_5[:, None]).to(tl.bfloat16)
    kv_5 = tl.where(valid_2d, kv_5, 0.0)
    qk += tl.dot(q_5, tl.trans(kv_5)).to(tl.float32)

    nope_fp8_6 = nope_uint8_6.to(_KV_FP8_TY, bitcast=True)
    kv_6 = (nope_fp8_6.to(tl.bfloat16) * scale_bf16_6[:, None]).to(tl.bfloat16)
    kv_6 = tl.where(valid_2d, kv_6, 0.0)
    qk += tl.dot(q_6, tl.trans(kv_6)).to(tl.float32)

    kv_7 = (rope_lo | (rope_hi << 8)).to(tl.bfloat16, bitcast=True)
    kv_7 = tl.where(valid_2d, kv_7, 0.0)
    qk += tl.dot(q_7, tl.trans(kv_7)).to(tl.float32)

    qk = qk * sm_scale
    qk = tl.where(valid[None, :], qk, NEG_INF)

    m_ij = tl.max(qk, axis=1)
    m_new = tl.maximum(m_i, m_ij)
    alpha = tl.where(m_i == NEG_INF, 0.0, tl.math.exp2((m_i - m_new) * LOG2E))
    p = tl.where(qk == NEG_INF, 0.0, tl.math.exp2((qk - m_new[:, None]) * LOG2E))
    l_new = alpha * l_i + tl.sum(p, axis=1)
    p_bf16 = p.to(tl.bfloat16)

    acc_0 = acc_0 * alpha[:, None] + tl.dot(p_bf16, kv_0).to(tl.float32)
    acc_1 = acc_1 * alpha[:, None] + tl.dot(p_bf16, kv_1).to(tl.float32)
    acc_2 = acc_2 * alpha[:, None] + tl.dot(p_bf16, kv_2).to(tl.float32)
    acc_3 = acc_3 * alpha[:, None] + tl.dot(p_bf16, kv_3).to(tl.float32)
    acc_4 = acc_4 * alpha[:, None] + tl.dot(p_bf16, kv_4).to(tl.float32)
    acc_5 = acc_5 * alpha[:, None] + tl.dot(p_bf16, kv_5).to(tl.float32)
    acc_6 = acc_6 * alpha[:, None] + tl.dot(p_bf16, kv_6).to(tl.float32)
    acc_7 = acc_7 * alpha[:, None] + tl.dot(p_bf16, kv_7).to(tl.float32)

    return acc_0, acc_1, acc_2, acc_3, acc_4, acc_5, acc_6, acc_7, m_new, l_new


# ============================================================================
# DSV4 Fused Gather+Dequant+Attention Kernel (Single Scope)
# ============================================================================

# The production kernel is @triton.autotune'd over BLOCK_H in {16, 64, 128},
# BLOCK_N in {32, 64, 128} and 4 or 8 warps, keyed on (total_tokens_bucket,
# h_q, topk). Autotuning benchmarks inside the first call, so this baseline
# launches one of those configurations instead.
_LAUNCH_CONFIG = {"BLOCK_H": 64, "BLOCK_N": 64, "num_warps": 8, "num_stages": 1}


@triton.jit
def _fused_gather_attn_dsv4_kernel(
    Q,
    KV_Cache,
    Indices,
    TopkLength,
    AttnSink,
    Output,
    LSE,
    sm_scale,
    total_tokens,
    total_tokens_bucket,
    h_q,
    topk,
    num_blocks,
    block_size,
    s_q,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_kv_block,
    stride_idx_t,
    stride_idx_k,
    stride_o_t,
    stride_o_h,
    stride_o_d,
    stride_lse_t,
    stride_lse_h,
    HAS_TOPK_LENGTH: tl.constexpr,
    HAS_ATTN_SINK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused gather+dequant+attention kernel for DSV4."""
    LOG2E: tl.constexpr = 1.4426950408889634
    D_NOPE: tl.constexpr = 448
    D_ROPE: tl.constexpr = 64
    TILE_SIZE: tl.constexpr = 64
    BYTES_PER_TOKEN_DATA: tl.constexpr = 576
    BYTES_PER_TOKEN_SCALE: tl.constexpr = 8

    # OPTIMIZED: Swapped grid - pid_h first for better cache locality
    pid_h = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_t_64 = pid_t.to(tl.int64)

    NEG_INF = float("-inf")

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < h_q

    m_i = tl.full([BLOCK_H], NEG_INF, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)

    acc_0 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_1 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_2 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_3 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_4 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_5 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_6 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)
    acc_7 = tl.zeros([BLOCK_H, TILE_SIZE], dtype=tl.float32)

    stride_q_t_64 = tl.cast(stride_q_t, tl.int64)
    q_base = Q + pid_t_64 * stride_q_t_64

    batch_idx = pid_t // s_q
    offs_tile = tl.arange(0, TILE_SIZE)

    q_0 = tl.load(
        q_base + offs_h[:, None] * stride_q_h + offs_tile[None, :] * stride_q_d,
        mask=mask_h[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    q_1 = tl.load(
        q_base
        + offs_h[:, None] * stride_q_h
        + (TILE_SIZE + offs_tile[None, :]) * stride_q_d,
        mask=mask_h[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    q_2 = tl.load(
        q_base
        + offs_h[:, None] * stride_q_h
        + (2 * TILE_SIZE + offs_tile[None, :]) * stride_q_d,
        mask=mask_h[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    q_3 = tl.load(
        q_base
        + offs_h[:, None] * stride_q_h
        + (3 * TILE_SIZE + offs_tile[None, :]) * stride_q_d,
        mask=mask_h[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    q_4 = tl.load(
        q_base
        + offs_h[:, None] * stride_q_h
        + (4 * TILE_SIZE + offs_tile[None, :]) * stride_q_d,
        mask=mask_h[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    q_5 = tl.load(
        q_base
        + offs_h[:, None] * stride_q_h
        + (5 * TILE_SIZE + offs_tile[None, :]) * stride_q_d,
        mask=mask_h[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    q_6 = tl.load(
        q_base
        + offs_h[:, None] * stride_q_h
        + (6 * TILE_SIZE + offs_tile[None, :]) * stride_q_d,
        mask=mask_h[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    q_7 = tl.load(
        q_base
        + offs_h[:, None] * stride_q_h
        + (7 * TILE_SIZE + offs_tile[None, :]) * stride_q_d,
        mask=mask_h[:, None],
        other=0.0,
    ).to(tl.bfloat16)

    # Early-exit: pre-load topk_len and skip invalid blocks
    if HAS_TOPK_LENGTH:
        topk_len = tl.load(TopkLength + batch_idx)

    for n_start in range(0, topk, BLOCK_N):
        # Skip entire block if beyond valid topk range
        should_compute = not HAS_TOPK_LENGTH or n_start < topk_len
        if should_compute:
            offs_n = n_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < topk

            idx_ptrs = Indices + pid_t * stride_idx_t + offs_n * stride_idx_k
            indices = tl.load(idx_ptrs, mask=mask_n, other=-1)

            is_invalid = indices == -1
            if HAS_TOPK_LENGTH:
                is_invalid = is_invalid | (offs_n >= topk_len)

            valid = mask_n & ~is_invalid
            indices_clamped = tl.maximum(indices, 0)

            block_idx = indices_clamped // block_size
            offset_in_block = indices_clamped % block_size

            block_idx_64 = block_idx.to(tl.int64)
            offset_in_block_64 = offset_in_block.to(tl.int64)

            stride_kv_block_64 = tl.cast(stride_kv_block, tl.int64)
            kv_block_base = KV_Cache + block_idx_64 * stride_kv_block_64
            nope_rope_offset = offset_in_block_64 * BYTES_PER_TOKEN_DATA
            scale_base_offset = (
                block_size * BYTES_PER_TOKEN_DATA
                + offset_in_block_64 * BYTES_PER_TOKEN_SCALE
            )

            valid_2d = valid[:, None]

            # Use helper function for KV processing
            acc_0, acc_1, acc_2, acc_3, acc_4, acc_5, acc_6, acc_7, m_i, l_i = (
                _process_kv_block_aggressive(
                    kv_block_base,
                    nope_rope_offset,
                    scale_base_offset,
                    valid,
                    valid_2d,
                    q_0,
                    q_1,
                    q_2,
                    q_3,
                    q_4,
                    q_5,
                    q_6,
                    q_7,
                    acc_0,
                    acc_1,
                    acc_2,
                    acc_3,
                    acc_4,
                    acc_5,
                    acc_6,
                    acc_7,
                    m_i,
                    l_i,
                    offs_tile,
                    sm_scale,
                    TILE_SIZE,
                    D_NOPE,
                    LOG2E,
                    BLOCK_H,
                    BLOCK_N,
                )
            )

    # Finalize
    lse = m_i + tl.math.log2(tl.where(l_i == 0.0, 1.0, l_i)) / LOG2E
    is_lonely_q = l_i == 0.0

    if HAS_ATTN_SINK:
        attn_sink_vals = tl.load(AttnSink + offs_h, mask=mask_h, other=0.0)
        exp_attn_sink_minus_m = tl.math.exp2((attn_sink_vals - m_i) * LOG2E)
        denominator = l_i + exp_attn_sink_minus_m
        denominator = tl.where(denominator == 0.0, 1.0, denominator)
        output_scale = 1.0 / denominator
    else:
        output_scale = tl.where(l_i == 0.0, 0.0, 1.0 / l_i)

    acc_0 = tl.where(is_lonely_q[:, None], 0.0, acc_0 * output_scale[:, None])
    acc_1 = tl.where(is_lonely_q[:, None], 0.0, acc_1 * output_scale[:, None])
    acc_2 = tl.where(is_lonely_q[:, None], 0.0, acc_2 * output_scale[:, None])
    acc_3 = tl.where(is_lonely_q[:, None], 0.0, acc_3 * output_scale[:, None])
    acc_4 = tl.where(is_lonely_q[:, None], 0.0, acc_4 * output_scale[:, None])
    acc_5 = tl.where(is_lonely_q[:, None], 0.0, acc_5 * output_scale[:, None])
    acc_6 = tl.where(is_lonely_q[:, None], 0.0, acc_6 * output_scale[:, None])
    acc_7 = tl.where(is_lonely_q[:, None], 0.0, acc_7 * output_scale[:, None])
    lse = tl.where(is_lonely_q, float("+inf"), lse)

    stride_o_t_64 = tl.cast(stride_o_t, tl.int64)
    o_base = Output + pid_t_64 * stride_o_t_64

    # Optimized output stores with pre-computed row base pointers
    # Convert to bfloat16 first (batch conversion)
    o_0 = acc_0.to(tl.bfloat16)
    o_1 = acc_1.to(tl.bfloat16)
    o_2 = acc_2.to(tl.bfloat16)
    o_3 = acc_3.to(tl.bfloat16)
    o_4 = acc_4.to(tl.bfloat16)
    o_5 = acc_5.to(tl.bfloat16)
    o_6 = acc_6.to(tl.bfloat16)
    o_7 = acc_7.to(tl.bfloat16)

    # Pre-compute row base pointers (shared across all 8 stores)
    row_ptrs = o_base + offs_h[:, None] * stride_o_h

    # Store all 8 tiles with optimized pointer arithmetic
    tl.store(row_ptrs + offs_tile[None, :] * stride_o_d, o_0, mask=mask_h[:, None])
    tl.store(
        row_ptrs + (TILE_SIZE + offs_tile[None, :]) * stride_o_d,
        o_1,
        mask=mask_h[:, None],
    )
    tl.store(
        row_ptrs + (2 * TILE_SIZE + offs_tile[None, :]) * stride_o_d,
        o_2,
        mask=mask_h[:, None],
    )
    tl.store(
        row_ptrs + (3 * TILE_SIZE + offs_tile[None, :]) * stride_o_d,
        o_3,
        mask=mask_h[:, None],
    )
    tl.store(
        row_ptrs + (4 * TILE_SIZE + offs_tile[None, :]) * stride_o_d,
        o_4,
        mask=mask_h[:, None],
    )
    tl.store(
        row_ptrs + (5 * TILE_SIZE + offs_tile[None, :]) * stride_o_d,
        o_5,
        mask=mask_h[:, None],
    )
    tl.store(
        row_ptrs + (6 * TILE_SIZE + offs_tile[None, :]) * stride_o_d,
        o_6,
        mask=mask_h[:, None],
    )
    tl.store(
        row_ptrs + (7 * TILE_SIZE + offs_tile[None, :]) * stride_o_d,
        o_7,
        mask=mask_h[:, None],
    )

    lse_ptrs = LSE + pid_t * stride_lse_t + offs_h * stride_lse_h
    tl.store(lse_ptrs, lse, mask=mask_h)


def fused_gather_attn_decode_dsv4(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    block_size: int,
    sm_scale: float,
    output: torch.Tensor,
    lse: torch.Tensor,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    s_q: int = 1,
) -> None:
    """
    Fused gather+dequant+attention for DSV4, writing ``output`` and ``lse``.

    Args:
        q: Query tensor [total_tokens, h_q, d_qk]
        kv_cache: Quantized KV cache
        indices: KV indices [total_tokens, topk]
        block_size: Block size for KV cache
        sm_scale: Softmax scale
        output: Attention output [total_tokens, h_q, d_v]
        lse: Log-sum-exp values [total_tokens, h_q]
        topk_length: Optional per-batch topk length [b]
        attn_sink: Optional attention sink values [h_q]
        s_q: Sequence length per batch
    """
    total_tokens, h_q, d_qk = q.shape
    topk = indices.shape[1]

    kv_uint8 = kv_cache.view(torch.uint8)
    num_blocks = kv_cache.shape[0]
    stride_kv_block = kv_uint8.stride(0)
    kv_flat = kv_uint8.reshape(num_blocks, -1)

    if q.dtype != torch.bfloat16 or not q.is_contiguous():
        q = q.to(torch.bfloat16).contiguous()

    if not indices.is_contiguous():
        indices = indices.contiguous()

    # Production switches to a split-K kernel for topk >= 8192 and disables
    # AMD buffer ops for caches above 2 GB; neither applies here (topk 128,
    # a 302 MB pool).
    topk_length_tensor = topk_length if topk_length is not None else lse[:1, 0]
    attn_sink_tensor = attn_sink if attn_sink is not None else lse[0, :]

    grid = (triton.cdiv(h_q, _LAUNCH_CONFIG["BLOCK_H"]), total_tokens)

    _fused_gather_attn_dsv4_kernel[grid](
        q,
        kv_flat,
        indices,
        topk_length_tensor,
        attn_sink_tensor,
        output,
        lse,
        sm_scale,
        total_tokens,
        _bucket_total_tokens(total_tokens),
        h_q,
        topk,
        num_blocks,
        block_size,
        s_q,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        stride_kv_block,
        indices.stride(0),
        indices.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        lse.stride(0),
        lse.stride(1),
        HAS_TOPK_LENGTH=topk_length is not None,
        HAS_ATTN_SINK=attn_sink is not None,
        **_LAUNCH_CONFIG,
    )


def run(*, q, kv_cache, sparse_indices, sparse_lens, sm_scale, sinks, output, lse):
    # triton_sparse_attn_decode, single-scope branch: flatten (batch, s_q).
    b, s_q, h_q, d_qk = q.shape
    total_tokens = b * s_q
    topk_main = sparse_indices.shape[-1]
    block_size_main = kv_cache.shape[1]

    q_reshaped = q.reshape(total_tokens, h_q, d_qk).contiguous()
    indices_main = sparse_indices.reshape(total_tokens, topk_main).contiguous()

    fused_gather_attn_decode_dsv4(
        q_reshaped,
        kv_cache,
        indices_main,
        block_size_main,
        sm_scale,
        output.view(total_tokens, h_q, DSV4_D_V),
        lse.view(total_tokens, h_q),
        topk_length=sparse_lens,
        attn_sink=sinks,
        s_q=s_q,
    )
