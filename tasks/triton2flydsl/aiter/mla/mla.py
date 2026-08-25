# The kernels in this file are adapted from vLLM:
# https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_unified_attention.py
"""Standalone Multi-head Latent Attention (MLA) Triton kernels.

Source: aiter/ops/triton/attention/mla.py (+ _triton_kernels).
The gfx1250-only Gluon kernel dispatch is dropped; only the @triton.jit path is kept.
"""

import math

import torch
import triton
import triton.language as tl


# --- inlined device CU count (utils.device_info.get_num_sms) ---
def get_num_sms():
    # Returns the Compute Unit count of the current device
    current_device_index = torch.cuda.current_device()
    current_device = torch.cuda.get_device_properties(current_device_index)
    num_sms = current_device.multi_processor_count
    return num_sms


# --- inlined arch detection (utils._triton.arch_info) ---
try:
    _CACHED_ARCH = triton.runtime.driver.active.get_current_target().arch
except RuntimeError:
    _CACHED_ARCH = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).gcnArchName.split(":")[0]


def get_arch():
    return _CACHED_ARCH


# --- inlined e4m3 dtype selection (utils.types.get_fp8_dtypes) ---
if get_arch() in ("gfx950", "gfx1250", "gfx1200", "gfx1201"):
    e4m3_dtype = torch.float8_e4m3fn
else:
    e4m3_dtype = torch.float8_e4m3fnuz

float8_info = torch.finfo(e4m3_dtype)


# --- inlined constexpr-aware kernel naming (utils._triton.kernel_repr) ---
def _sanitize_constexpr_value(value):
    if value is None:
        return "NONE"
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value)

    # for lists, tuples, sets - recursively join each
    if isinstance(value, (list, tuple, set)):
        items = sorted(value, key=str) if isinstance(value, set) else value
        sanitized_items = [_sanitize_constexpr_value(item) for item in items]
        joined = "_".join(sanitized_items)
        return joined if joined else "NONE"

    if isinstance(value, str):
        cleaned_value = "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")
        return cleaned_value.upper() if cleaned_value else "NONE"

    cleaned_value = "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")
    return cleaned_value.upper() if cleaned_value else "NONE"


def make_kernel_repr(base_name, config_keys, name_key=None):
    def _repr(specialization):
        constants = specialization.constants

        name = base_name
        if name_key is not None:
            override = constants.get(name_key, None)
            if override:
                cleaned = "".join(
                    ch if ch.isalnum() or ch == "_" else "_" for ch in str(override)
                )
                if cleaned:
                    name = cleaned

        name_parts = []
        for key in config_keys:
            value = constants.get(key, None)
            symbol = _sanitize_constexpr_value(value)
            name_parts.append(f"{key}_{symbol}")

        if not name_parts:
            return name

        suffix = "_".join(name_parts)
        return f"{name}_{suffix}"

    return _repr


DEVICE_ARCH = get_arch()
IS_DEVICE_ARCH_GFX12 = DEVICE_ARCH in ("gfx1250",)
WARP_SIZE = 32 if IS_DEVICE_ARCH_GFX12 else 64


@triton.jit
def fast_exp(x):
    RCP_LN2: tl.constexpr = 1.4426950408889634
    return tl.math.exp2(x * RCP_LN2)


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def apply_softcap(S, x):
    Sdiv = S / x
    p1 = tl.math.exp2(Sdiv)
    p2 = tl.math.exp2(-Sdiv)
    return x * (p1 - p2) / (p1 + p2)


@triton.jit
def _find_seq_idx(
    query_start_len_ptr,
    target_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
    use_q_block_mode: tl.constexpr,
):
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val

        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid

    return left - 1


_mla_prefill_fwd_kernel_repr = make_kernel_repr(
    "_mla_prefill_fwd_kernel",
    [
        "num_query_heads",
        "num_kv_heads",
        "TILE_SIZE",
        "KV_LORA_RANK",
        "QK_ROPE_HEAD_DIM",
        "BLOCK_Q",
        "BLOCK_M",
        "NUM_HEAD_BLOCKS",
        "NUM_SEGMENTS_PER_SEQ",
        "num_warps",
        "num_stages",
    ],
)


@triton.jit(repr=_mla_prefill_fwd_kernel_repr)
def _mla_prefill_fwd_kernel(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    kv_buffer_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    scale: tl.constexpr,  # float32
    q_scale_ptr,  # float32
    kv_scale_ptr,  # float32
    out_scale_ptr,  # float32
    num_query_heads: tl.constexpr,  # int
    num_kv_heads: tl.constexpr,  # int
    block_tables_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    KV_LORA_RANK: tl.constexpr,  # int
    QK_ROPE_HEAD_DIM: tl.constexpr,  # int
    stride_kv_buffer_0: tl.int64,  # int
    stride_kv_buffer_1: tl.int64,  # int
    stride_kv_buffer_2: tl.int64,  # int
    stride_kv_buffer_3: tl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    num_seqs: tl.int32,
    TILE_SIZE: tl.constexpr,  # int
    BLOCK_Q: tl.constexpr,  # int
    BLOCK_M: tl.constexpr,  # int
    num_warps: tl.constexpr,  # int
    num_stages: tl.constexpr,  # int
    NUM_HEAD_BLOCKS: tl.constexpr = 1,  # int
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    kv_head_idx = tl.program_id(0)
    q_block_global_idx = tl.program_id(1)

    # needed to use exp2 (exp2 -> exp conversion)
    RCP_LN2 = 1.4426950408889634
    qk_scale = scale * RCP_LN2

    # split the flat block index into a token-block part and a head-block part
    token_q_block_global_idx = q_block_global_idx // NUM_HEAD_BLOCKS
    head_block_idx = q_block_global_idx % NUM_HEAD_BLOCKS
    head_offset = head_block_idx * BLOCK_M

    seq_idx = _find_seq_idx(
        query_start_len_ptr, token_q_block_global_idx, num_seqs, BLOCK_Q, True
    )

    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx

    q_block_local_idx = token_q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    qk_factor: tl.float32 = qk_scale
    if q_scale_ptr is not None:
        q_scale = tl.load(q_scale_ptr)
        qk_factor = qk_factor * q_scale
    else:
        q_scale = None

    if kv_scale_ptr is not None:
        kv_scale = tl.load(kv_scale_ptr)
        qk_factor = qk_factor * kv_scale
    else:
        kv_scale = None
    out_scale = None
    if out_scale_ptr is not None:
        out_scale = 1 / tl.load(out_scale_ptr)

    offs_m = tl.arange(0, BLOCK_M)
    offs_lora_rank = tl.arange(0, KV_LORA_RANK)
    offs_rope_head_dim = tl.arange(0, QK_ROPE_HEAD_DIM)
    offs_t = tl.arange(0, TILE_SIZE)

    num_queries_per_kv: tl.constexpr = num_query_heads // num_kv_heads
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = (
        kv_head_idx * num_queries_per_kv + head_offset + offs_m % num_queries_per_kv
    )
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
    )

    query_mask_0 = query_pos < cur_batch_query_len
    query_mask_1 = query_offset_1 < num_query_heads

    # Q_lora : (BLOCK_M, KV_LORA_RANK)
    # Q_rope : (BLOCK_M, QK_ROPE_HEAD_DIM)
    Q_lora = tl.load(
        query_ptr + query_offset + offs_lora_rank[None, :],
        mask=query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )
    Q_rope = tl.load(
        query_ptr + query_offset + (KV_LORA_RANK + offs_rope_head_dim)[None, :],
        mask=query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    block_tables_ptr_shifted = block_tables_ptr + seq_idx * block_tables_stride

    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, KV_LORA_RANK], dtype=tl.float32)

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # context length for this particular sequences
    context_len = seq_len - cur_batch_query_len

    # compute the length of the longest sequence prefix spanned by any
    # query token in the current q_block (q_block_local_idx)
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )

    # adjust for potential padding in the last q_block by considering the
    # actual sequence length
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)

    # calculate the number of tiles that need to be processed to
    # cover the longest sequence prefix (due to causal masking, tiles beyond
    # this prefix can be skipped)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    # ---- Sliding-window tile pruning --------------------
    # Default: keep previous global behavior
    tile_start = 0
    tile_end = num_tiles
    seq_offset = offs_t

    # iterate through tiles (now limited to the sliding window range)
    for j in range(tile_start, tile_end):
        physical_block_idx = tl.load(block_tables_ptr_shifted + j).to(tl.int64)

        kv_offset = (
            physical_block_idx * stride_kv_buffer_0 + kv_head_idx * stride_kv_buffer_2
        )

        kv_lora_offset = (
            kv_offset
            + offs_t[:, None] * stride_kv_buffer_1
            + offs_lora_rank[None, :] * stride_kv_buffer_3
        )
        # KV_lora : (BLOCK_M, KV_LORA_RANK)
        KV_lora = tl.load(
            kv_buffer_ptr + kv_lora_offset,
        )

        k_rope_offset = (
            kv_offset
            + offs_t[None, :] * stride_kv_buffer_1
            + (KV_LORA_RANK + offs_rope_head_dim)[:, None] * stride_kv_buffer_3
        )
        # K_rope : (BLOCK_M, QK_ROPE_HEAD_DIM)
        K_rope = tl.load(
            kv_buffer_ptr + k_rope_offset,
        )

        seq_mask = seq_offset[None, :] < context_len + query_pos[:, None] + 1

        S_lora = tl.dot(Q_lora, KV_lora.trans(1, 0).to(Q_lora.dtype))
        S_rope = tl.dot(Q_rope, K_rope.to(Q_lora.dtype))
        S = qk_factor * (S_lora + S_rope)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        # compute running maximum
        # m_j : (BLOCK_M,)
        m_j = tl.maximum(M, tl.max(S, axis=1))

        # For sliding window there's a chance the max is -inf due to masking of
        # the entire row. In this case we need to set m_j 0 to avoid NaN
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

        # P : (BLOCK_M, TILE_SIZE,)
        P = tl.math.exp2(S - m_j[:, None])

        # l_j : (BLOCK_M,)
        l_j = tl.sum(P, axis=1)

        # alpha : (BLOCK_M, )
        alpha = tl.math.exp2(M - m_j)

        # acc : (BLOCK_M, KV_LORA_RANK)
        acc = acc * alpha[:, None]

        # update constants
        L = L * alpha + l_j
        M = m_j

        # acc : (BLOCK_M, KV_LORA_RANK)
        acc += tl.dot(P.to(KV_lora.dtype), KV_lora)
        seq_offset += TILE_SIZE

    # epilogue
    # This helps the compiler do Newton Raphson on l_i vs on acc which is much larger.
    if kv_scale_ptr is not None:
        one_over_L = kv_scale / L[:, None]
    else:
        one_over_L = 1.0 / L[:, None]
    acc = acc * one_over_L

    if out_scale_ptr is not None:
        acc = acc * out_scale
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    output_offset = (
        query_offset_0[:, None] * output_stride_0
        + query_offset_1[:, None] * output_stride_1
        + offs_lora_rank[None, :]
    )

    tl.store(
        output_ptr + output_offset,
        acc,
        mask=query_mask_0[:, None] & query_mask_1[:, None],
    )


_mla_decode_fwd_kernel_repr = make_kernel_repr(
    "_mla_decode_fwd_kernel",
    [
        "num_query_heads",
        "num_kv_heads",
        "num_tokens_per_seq",
        "TILE_SIZE",
        "KV_LORA_RANK",
        "QK_ROPE_HEAD_DIM",
        "BLOCK_Q",
        "BLOCK_M",
        "NUM_HEAD_BLOCKS",
        "NUM_SEGMENTS_PER_SEQ",
        "num_warps",
        "waves_per_eu",
        "num_stages",
        "SHUFFLED_KV_CACHE",
        "IS_Q_FP8",
        "IS_KV_FP8",
    ],
)


@triton.jit(repr=_mla_decode_fwd_kernel_repr)
def _mla_decode_fwd_kernel(
    segm_output_ptr,  # [total_num_tokens, num_query_heads, KV_LORA_RANK + qk_rope_head_dim]
    segm_max_ptr,  # [total_num_tokens, num_query_heads, num_segments]
    segm_expsum_ptr,  # [total_num_tokens, num_query_heads, num_segments]
    query_ptr,  # [total_num_tokens, num_query_heads, head_size]
    query_scales_ptr,  # nvfp4 query scales (unused for non-shuffled bf16/fp8)
    kv_buffer_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    scale,  # float32
    q_scale_ptr,  # float32
    kv_scale_ptr,  # float32
    num_query_heads: tl.constexpr,  # int
    num_kv_heads: tl.constexpr,  # int
    block_tables_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    query_scales_stride_0: tl.int64,  # int
    query_scales_stride_1: tl.int64,  # int
    KV_LORA_RANK: tl.constexpr,  # int
    QK_ROPE_HEAD_DIM: tl.constexpr,  # int
    stride_kv_buffer_0: tl.int64,  # int
    stride_kv_buffer_1: tl.int64,  # int
    stride_kv_buffer_2: tl.int64,  # int
    stride_kv_buffer_3: tl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    num_tokens_per_seq: tl.int32,
    TILE_SIZE: tl.constexpr,  # int
    BLOCK_Q: tl.constexpr,  # int
    BLOCK_M: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    num_warps: tl.constexpr,  # int
    waves_per_eu: tl.constexpr,  # int
    num_stages: tl.constexpr,  # int
    NUM_HEAD_BLOCKS: tl.constexpr = 1,  # int
    ALL_DECODE: tl.constexpr = False,  # bool
    SHUFFLED_KV_CACHE: tl.constexpr = False,  # bool
    IS_Q_FP8: tl.constexpr = False,  # bool
    IS_KV_FP8: tl.constexpr = False,  # bool
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2)

    # needed to use exp2 (exp2 -> exp conversion)
    RCP_LN2 = 1.4426950408889634
    qk_scale = scale * RCP_LN2
    num_token_blocks_per_seq = cdiv_fn(num_tokens_per_seq, BLOCK_Q)
    num_q_blocks_per_seq = num_token_blocks_per_seq * NUM_HEAD_BLOCKS

    if ALL_DECODE:
        seq_idx = q_block_global_idx // NUM_HEAD_BLOCKS
    else:
        seq_idx = q_block_global_idx // num_q_blocks_per_seq

    q_start_idx = tl.load(query_start_len_ptr + seq_idx)
    q_block_local_idx = q_block_global_idx - seq_idx * num_q_blocks_per_seq

    token_q_block_local_idx = q_block_local_idx // NUM_HEAD_BLOCKS
    head_block_idx = q_block_local_idx % NUM_HEAD_BLOCKS
    head_offset = head_block_idx * BLOCK_M

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
        return

    qk_factor: tl.float32 = qk_scale
    if q_scale_ptr is not None:
        q_scale = tl.load(q_scale_ptr)
        qk_factor = qk_factor * q_scale
    else:
        q_scale = None

    if kv_scale_ptr is not None:
        kv_scale = tl.load(kv_scale_ptr)
        qk_factor = qk_factor * kv_scale
    else:
        kv_scale = None

    offs_m = tl.arange(0, BLOCK_M)
    offs_lora_rank = tl.arange(0, KV_LORA_RANK)
    offs_rope_head_dim = tl.arange(0, QK_ROPE_HEAD_DIM)
    offs_t = tl.arange(0, TILE_SIZE)

    offs_lora_rank_shfl = None
    offs_rope_head_dim_shfl = None
    offs_t_shfl = None
    if SHUFFLED_KV_CACHE:
        offs_lora_rank_shfl = tl.arange(0, KV_LORA_RANK * 16)
        offs_rope_head_dim_shfl = tl.arange(0, QK_ROPE_HEAD_DIM * 16)
        offs_t_shfl = tl.arange(0, TILE_SIZE // 16)

    if IS_KV_FP8:
        K_WIDTH: tl.constexpr = 16
    else:
        K_WIDTH: tl.constexpr = 8

    num_queries_per_kv: tl.constexpr = num_query_heads // num_kv_heads
    query_pos = token_q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = q_start_idx + query_pos
    query_offset_1 = (
        kv_head_idx * num_queries_per_kv + head_offset + offs_m % num_queries_per_kv
    )
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
    )

    query_mask_0 = query_pos < num_tokens_per_seq
    query_mask_1 = query_offset_1 < num_query_heads

    # Q_lora : (BLOCK_M, KV_LORA_RANK)
    # Q_rope : (BLOCK_M, QK_ROPE_HEAD_DIM)
    Q_lora = tl.load(
        query_ptr + query_offset + offs_lora_rank[None, :],
        mask=query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )
    Q_rope = tl.load(
        query_ptr + query_offset + (KV_LORA_RANK + offs_rope_head_dim)[None, :],
        mask=query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    block_tables_ptr_shifted = block_tables_ptr + seq_idx * block_tables_stride

    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, KV_LORA_RANK], dtype=tl.float32)

    # context length for this particular sequences
    context_len = seq_len - num_tokens_per_seq

    # compute the length of the longest sequence prefix spanned by any
    # query token in the current q_block (token_q_block_local_idx)
    max_seq_prefix_len = (
        context_len
        + token_q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )

    # adjust for potential padding in the last q_block by considering the
    # actual sequence length
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)

    # calculate the number of tiles that need to be processed to
    # cover the longest sequence prefix (due to causal masking, tiles beyond
    # this prefix can be skipped)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    KV_cache_modifier: tl.constexpr = ".cg" if ALL_DECODE else ""
    seq_offset = segm_idx * tiles_per_segment * TILE_SIZE + offs_t

    # iterate through tiles within current segment
    for j in range(
        segm_idx * tiles_per_segment,
        min((segm_idx + 1) * tiles_per_segment, num_tiles),
    ):
        physical_block_idx = tl.load(block_tables_ptr_shifted + j).to(tl.int64)

        if SHUFFLED_KV_CACHE:
            kv_offset = (
                physical_block_idx * stride_kv_buffer_0
                + kv_head_idx * stride_kv_buffer_1
            )
            kv_lora_offset = (
                kv_offset
                + offs_t_shfl[:, None] * (KV_LORA_RANK * 16) * stride_kv_buffer_3
                + offs_lora_rank_shfl[None, :] * stride_kv_buffer_3
            )

            k_rope_offset = (
                kv_offset
                + (TILE_SIZE * KV_LORA_RANK) * stride_kv_buffer_3
                + offs_t_shfl[:, None] * (QK_ROPE_HEAD_DIM * 16) * stride_kv_buffer_3
                + offs_rope_head_dim_shfl[None, :] * stride_kv_buffer_3
            )
        else:
            kv_offset = (
                physical_block_idx * stride_kv_buffer_0
                + kv_head_idx * stride_kv_buffer_2
            )
            kv_lora_offset = (
                kv_offset
                + offs_t[:, None] * stride_kv_buffer_1
                + offs_lora_rank[None, :] * stride_kv_buffer_3
            )

            k_rope_offset = (
                kv_offset
                + offs_t[None, :] * stride_kv_buffer_1
                + (KV_LORA_RANK + offs_rope_head_dim)[:, None] * stride_kv_buffer_3
            )

        # KV_lora : (BLOCK_M, KV_LORA_RANK)
        KV_lora = tl.load(
            kv_buffer_ptr + kv_lora_offset,
            cache_modifier=KV_cache_modifier,
        )
        KV_lora = KV_lora.to(Q_lora.dtype)

        # K_rope : (BLOCK_M, QK_ROPE_HEAD_DIM)
        K_rope = tl.load(
            kv_buffer_ptr + k_rope_offset,
            cache_modifier=KV_cache_modifier,
        )
        K_rope = K_rope.to(Q_rope.dtype)

        if SHUFFLED_KV_CACHE:
            KV_lora = (
                KV_lora.reshape(
                    (
                        1,
                        TILE_SIZE // 16,
                        KV_LORA_RANK // (2 * K_WIDTH),
                        2,
                        16,
                        K_WIDTH,
                    )
                )
                .permute((0, 1, 4, 2, 3, 5))
                .reshape((TILE_SIZE, KV_LORA_RANK))
                .permute((1, 0))
            )
            K_rope = (
                K_rope.reshape(
                    (
                        1,
                        TILE_SIZE // 16,
                        QK_ROPE_HEAD_DIM // (2 * K_WIDTH),
                        2,
                        16,
                        K_WIDTH,
                    )
                )
                .permute((0, 1, 4, 2, 3, 5))
                .reshape((TILE_SIZE, QK_ROPE_HEAD_DIM))
                .permute((1, 0))
            )

        seq_mask = seq_offset[None, :] < context_len + query_pos[:, None] + 1

        if SHUFFLED_KV_CACHE:
            S_lora = tl.dot(Q_lora, KV_lora)
        else:
            S_lora = tl.dot(Q_lora, KV_lora.permute((1, 0)))
        S_rope = tl.dot(Q_rope, K_rope)
        S = qk_factor * (S_lora + S_rope)

        S = tl.where(seq_mask, S, float("-inf"))

        # compute running maximum
        # m_j : (BLOCK_M,)
        m_j = tl.maximum(M, tl.max(S, axis=1))

        # For sliding window there's a chance the max is -inf due to masking of
        # the entire row. In this case we need to set m_j 0 to avoid NaN
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

        # P : (BLOCK_M, TILE_SIZE,)
        P = tl.math.exp2(S - m_j[:, None])

        # l_j : (BLOCK_M,)
        l_j = tl.sum(P, axis=1)

        # alpha : (BLOCK_M, )
        alpha = tl.math.exp2(M - m_j)

        # acc : (BLOCK_M, KV_LORA_RANK)
        acc = acc * alpha[:, None]

        # update constants
        L = L * alpha + l_j
        M = m_j

        # acc : (BLOCK_M, KV_LORA_RANK)
        if SHUFFLED_KV_CACHE:
            acc += tl.dot(P.to(KV_lora.dtype), KV_lora.permute((1, 0)))
        else:
            acc += tl.dot(P.to(KV_lora.dtype), KV_lora)
        seq_offset += TILE_SIZE

    if kv_scale_ptr is not None:
        acc = acc * kv_scale

    segm_output_offset = (
        query_offset_0[:, None].to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * KV_LORA_RANK)
        + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * KV_LORA_RANK)
        + segm_idx * KV_LORA_RANK
        + tl.arange(0, KV_LORA_RANK)[None, :]
    )
    tl.store(
        segm_output_ptr + segm_output_offset,
        acc,
        mask=query_mask_0[:, None] & query_mask_1[:, None],
    )
    segm_offset = (
        query_offset_0.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_offset_1 * NUM_SEGMENTS_PER_SEQ
        + segm_idx
    )
    tl.store(segm_max_ptr + segm_offset, M, mask=query_mask_0 & query_mask_1)
    tl.store(segm_expsum_ptr + segm_offset, L, mask=query_mask_0 & query_mask_1)


_mla_decode_fwd_reduce_kernel_repr = make_kernel_repr(
    "_mla_decode_fwd_reduce_kernel",
    [
        "num_query_heads",
        "TILE_SIZE",
        "KV_LORA_RANK",
        "NUM_SEGMENTS_PER_SEQ",
        "ALL_DECODE",
    ],
)


@triton.jit(repr=_mla_decode_fwd_reduce_kernel_repr)
def _mla_decode_fwd_reduce_kernel(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    segm_output_ptr,
    # [num_tokens, num_query_heads, max_num_segments, head_size]
    segm_max_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    seq_lens_ptr,  # [num_seqs]
    out_scale_ptr,  # float32
    num_seqs,  # int
    num_query_heads: tl.constexpr,  # int
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    block_tables_stride: tl.int64,  # int
    num_tokens_per_seq: tl.int32,
    total_num_tokens: tl.int32,
    TILE_SIZE: tl.constexpr,  # int
    KV_LORA_RANK: tl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    ALL_DECODE: tl.constexpr = False,  # int
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    if ALL_DECODE:
        seq_idx = query_token_idx
    else:
        seq_idx = query_token_idx // num_tokens_per_seq

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    out_scale = None
    if out_scale_ptr is not None:
        out_scale = 1 / tl.load(out_scale_ptr)

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    # create masks for subsequent loads
    act_num_segments = cdiv_fn(seq_len, tiles_per_segment * TILE_SIZE)
    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full(
        [NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32
    )

    # load segment maxima
    segm_offset = (
        query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_head_idx * NUM_SEGMENTS_PER_SEQ
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)
    )
    segm_max = tl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf"))
    overall_max = tl.max(segm_max)

    # load and rescale segment exp sums
    segm_expsum = tl.load(segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0)
    segm_expsum = segm_expsum * tl.math.exp2(segm_max - overall_max)
    overall_expsum = tl.sum(segm_expsum)

    # load, rescale, and add segment attention outputs
    segm_output_offset = (
        query_token_idx.to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * KV_LORA_RANK)
        + query_head_idx * (NUM_SEGMENTS_PER_SEQ * KV_LORA_RANK)
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * KV_LORA_RANK
        + tl.arange(0, KV_LORA_RANK)[None, :]
    )
    segm_output = tl.load(
        segm_output_ptr + segm_output_offset,
        mask=segm_mask[:, None],
        other=0.0,
    )
    segm_output *= tl.math.exp2(segm_max - overall_max)[:, None]
    acc_sum = tl.sum(segm_output, axis=0)
    # safely divide by overall_expsum, returning 0.0 if overall_expsum is 0
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)

    if out_scale_ptr is not None:
        acc = acc * out_scale

    if output_ptr.type.element_ty.is_fp8():
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    # write result
    output_offset = (
        query_token_idx * output_stride_0
        + query_head_idx * output_stride_1
        + tl.arange(0, KV_LORA_RANK)
    )
    tl.store(output_ptr + output_offset, acc.to(output_ptr.type.element_ty))


# Host-side launch logic. The gfx1250-only Gluon kernel dispatch is dropped (those
# kernels cannot be detached from the _gluon_kernels package).
def select_2d_config(
    block_size,
    head_size,
    max_seqlen_k,
    num_queries_per_kv,
    num_2d_prgms,
):
    TILE_SIZE = block_size
    num_stages_2d = 1
    num_warps = 8

    return {
        "TILE_SIZE": TILE_SIZE,
        "num_warps": num_warps,
        "num_stages": num_stages_2d,
        "waves_per_eu": 1,
    }


def select_3d_config(
    block_size,
    max_seqlen_k,
    target_num_prgms,
    num_2d_prgms,
    q_dtype,
    kv_dtype,
    shuffled_kv_cache,
):
    attn_num_warps = 2
    reduce_num_warps = 2
    attn_waves_per_eu = 1
    reduce_waves_per_eu = 2
    num_segments = 0
    TILE_SIZE = block_size
    if IS_DEVICE_ARCH_GFX12:
        # If we cannot infer max_seqlen_k during graph capture
        maybe_guess_max_seqlen_k = 128000 if max_seqlen_k == 0 else max_seqlen_k
        attn_num_warps = 2
        reduce_num_warps = 4
        attn_waves_per_eu = 1
        reduce_waves_per_eu = 1
        if shuffled_kv_cache:
            if kv_dtype == torch.uint8:
                assert (
                    block_size == 128
                ), "Only block_size == 128 is supported for FP4 KV cache"

        occ = attn_waves_per_eu * 4 // attn_num_warps
        MAX_SEGMENTS = max(1, math.ceil(maybe_guess_max_seqlen_k / TILE_SIZE))
        num_segments = max(1, target_num_prgms // 4 * occ // max(1, num_2d_prgms))
        num_segments = min(MAX_SEGMENTS, num_segments)
        num_segments = triton.next_power_of_2(num_segments)

    MAX_SEGMENTS = min(128, math.ceil(max_seqlen_k / TILE_SIZE))
    if num_segments == 0:
        num_segments = math.ceil(target_num_prgms / num_2d_prgms) * 2
        num_segments = min(num_segments, MAX_SEGMENTS)
        num_segments = triton.next_power_of_2(num_segments)
        num_segments = min(num_segments, 128)
        MIN_SEGMENTS = max(8, num_segments)
        num_segments = max(num_segments, MIN_SEGMENTS)

        if num_segments == MIN_SEGMENTS:
            reduce_num_warps = 1

    attn_config = {
        "TILE_SIZE": TILE_SIZE,
        "NUM_SEGMENTS_PER_SEQ": num_segments,
        "num_warps": attn_num_warps,
        "waves_per_eu": attn_waves_per_eu,
        "num_stages": 2 if DEVICE_ARCH in ("gfx1250", "gfx950") else 1,
    }
    reduce_config = {
        "TILE_SIZE": TILE_SIZE,
        "NUM_SEGMENTS_PER_SEQ": num_segments,
        "num_warps": reduce_num_warps,
        "waves_per_eu": reduce_waves_per_eu,
        "num_stages": 1,
    }
    return attn_config, reduce_config


def mla_prefill_fwd(
    q,  # [num_tokens_per_seq * num_seqs, num_query_heads, qk_lora_rank + qk_rope_head_dim]
    kv_buffer,  # [num_blocks, block_size, num_kv_heads, qk_lora_rank + qk_rope_head_dim]
    out,
    cu_seqlens_q,  # [num_seqs + 1]
    seqused_k,  # [num_seqs]
    max_seqlen_kv: int,
    block_tables,  # [batch_size, max_num_blocks_per_seq]
    softmax_scale: float,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    causal: bool,
    q_descale,
    kv_descale,
    out_scale=None,
    shuffled_kv_cache: bool = False,
):
    assert causal, "Only causal attention is supported"
    assert (
        not shuffled_kv_cache
    ), "Shuffled kv cache is not supported in mla_prefill_fwd"

    total_num_tokens, num_query_heads, qk_head_dim = q.shape
    num_blocks, block_size, num_kv_heads, _ = kv_buffer.shape
    num_seqs = len(seqused_k)
    num_queries_per_kv = num_query_heads // num_kv_heads
    q_dtype = q.dtype
    kv_buffer_dtype = kv_buffer.dtype
    K_WIDTH = 16 if kv_buffer_dtype == e4m3_dtype else 8
    QUERY_DTYPE = "fp8" if q_dtype == e4m3_dtype else "bf16"
    KV_CACHE_DTYPE = "fp8" if kv_buffer_dtype == e4m3_dtype else "bf16"

    assert (
        kv_lora_rank + qk_rope_head_dim == qk_head_dim
    ), "qk_head_dim must be equal to kv_lora_rank + qk_rope_head_dim"

    # BLOCK_M = 128
    BLOCK_M = 16
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    assert BLOCK_Q >= 1 or (num_queries_per_kv > BLOCK_M)
    BLOCK_Q = max(BLOCK_Q, 1)
    # When num_queries_per_kv > BLOCK_M the query heads of a single KV head do
    # not fit into one BLOCK_M tile, so we split them across NUM_HEAD_BLOCKS
    # blocks along the head dimension.
    NUM_HEAD_BLOCKS = (num_queries_per_kv + BLOCK_M - 1) // BLOCK_M
    # Ideally we would launch with kernel with:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)] blocks.
    # However, it is slow to realize the query_lens on cpu.
    # Instead we use upper-bound:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)]
    #   <= \sum_i[floor(query_len[i] / BLOCK_Q) + 1]
    #    = \sum_i[floor(query_len[i] / BLOCK_Q)] + num_seqs
    #   <= floor(\sum_i(query_len[i]) / BLOCK_Q) + num_seqs
    #    = floor(q.shape[0] / BLOCK_Q) + num_seqs
    # cu_count = get_num_sms()
    total_num_q_blocks = (q.shape[0] // BLOCK_Q + num_seqs) * NUM_HEAD_BLOCKS
    num_2d_prgms = total_num_q_blocks * num_kv_heads
    # if batch contains a prefill
    attn_config = select_2d_config(
        block_size,
        kv_lora_rank,
        max_seqlen_kv,
        num_queries_per_kv,
        num_2d_prgms,
    )

    _mla_prefill_fwd_kernel[(num_kv_heads, total_num_q_blocks)](
        output_ptr=out,
        query_ptr=q,
        kv_buffer_ptr=kv_buffer,
        block_tables_ptr=block_tables,
        seq_lens_ptr=seqused_k,
        scale=softmax_scale,
        q_scale_ptr=q_descale,
        kv_scale_ptr=kv_descale,
        out_scale_ptr=out_scale,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        block_tables_stride=block_tables.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        KV_LORA_RANK=kv_lora_rank,
        QK_ROPE_HEAD_DIM=qk_rope_head_dim,
        stride_kv_buffer_0=kv_buffer.stride(0),
        stride_kv_buffer_1=kv_buffer.stride(1),
        stride_kv_buffer_2=kv_buffer.stride(2),
        stride_kv_buffer_3=kv_buffer.stride(3),
        query_start_len_ptr=cu_seqlens_q,
        num_seqs=num_seqs,
        BLOCK_Q=BLOCK_Q,
        BLOCK_M=BLOCK_M,
        NUM_HEAD_BLOCKS=NUM_HEAD_BLOCKS,
        **attn_config,
    )
    return out


def mla_decode_fwd(
    q,  # [num_tokens_per_seq * num_seqs, num_query_heads, qk_lora_rank + qk_rope_head_dim]
    kv_buffer,  # [num_blocks, block_size, num_kv_heads, qk_lora_rank + qk_rope_head_dim]
    out,
    cu_seqlens_q,  # [num_seqs + 1]
    seqused_k,  # [num_seqs]
    max_seqlen_kv: int,
    block_tables,  # [batch_size, max_num_blocks_per_seq]
    softmax_scale: float,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    causal: bool,
    q_descale,
    kv_descale,
    q_scales=None,
    out_scale=None,
    shuffled_kv_cache: bool = False,
    skip_reduce: bool = False,
):
    assert causal, "Only causal attention is supported"
    q_dtype = q.dtype
    kv_buffer_dtype = kv_buffer.dtype
    total_num_tokens, num_query_heads, qk_head_dim = q.shape

    BLOCK_SCALES_SIZE = 16
    if q_dtype == torch.uint8:
        # A4W4
        assert q_scales is not None and q_scales.dtype == e4m3_dtype
        qk_head_dim = qk_head_dim * 2
        QUERY_DTYPE = "nvfp4"
    elif q_dtype == e4m3_dtype:
        QUERY_DTYPE = "fp8"
    else:
        QUERY_DTYPE = "bf16"

    if kv_buffer_dtype == torch.uint8:
        # A8W4 A4W4
        assert IS_DEVICE_ARCH_GFX12, "FP4 KV cache is only supported on GFX12"
        KV_CACHE_DTYPE = "nvfp4"
    elif kv_buffer_dtype == e4m3_dtype:
        KV_CACHE_DTYPE = "fp8"
    else:
        KV_CACHE_DTYPE = "bf16"

    SCALE_K_WIDTH_LORA = 0
    SCALE_K_WIDTH_ROPE = 0
    if shuffled_kv_cache:
        SCALE_K_WIDTH_LORA = 4
        SCALE_K_WIDTH_ROPE = 4
        if kv_buffer_dtype == torch.uint8:
            num_blocks, num_kv_heads, block_size, _ = kv_buffer.shape
            K_WIDTH = 16
            SCALE_K_LORA = kv_lora_rank // 16
            SCALE_K_ROPE = qk_rope_head_dim // 16
            SCALE_K_WIDTH_LORA = (
                min(16, triton.next_power_of_2(SCALE_K_LORA))
                if SCALE_K_LORA >= 4
                else SCALE_K_LORA
            )
            SCALE_K_WIDTH_ROPE = (
                min(16, triton.next_power_of_2(SCALE_K_ROPE))
                if SCALE_K_ROPE >= 4
                else SCALE_K_ROPE
            )
        else:
            num_blocks, num_kv_heads, block_size, _ = kv_buffer.shape
            K_WIDTH = 16 if kv_buffer_dtype == e4m3_dtype else 8
    else:
        num_blocks, block_size, num_kv_heads, _ = kv_buffer.shape
        K_WIDTH = 16 if kv_buffer_dtype == e4m3_dtype else 8

    num_seqs = len(seqused_k)
    num_tokens_per_seq = total_num_tokens // num_seqs
    num_queries_per_kv = num_query_heads // num_kv_heads

    assert (
        kv_lora_rank + qk_rope_head_dim == qk_head_dim
    ), "qk_head_dim must be equal to kv_lora_rank + qk_rope_head_dim"

    MAX_BLOCK_M = 16
    if num_queries_per_kv <= 16:
        BLOCK_M = 16
    else:
        BLOCK_M = min(triton.next_power_of_2(num_queries_per_kv), MAX_BLOCK_M)
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    assert BLOCK_Q >= 1 or (num_queries_per_kv > BLOCK_M)
    BLOCK_Q = max(BLOCK_Q, 1)
    NUM_HEAD_BLOCKS = (num_queries_per_kv + BLOCK_M - 1) // BLOCK_M
    cu_count = get_num_sms()
    target_num_prgms = cu_count * 4
    ALL_DECODE = num_tokens_per_seq == 1
    if ALL_DECODE:
        total_num_q_blocks = num_seqs * NUM_HEAD_BLOCKS
    else:
        total_num_q_blocks = (
            ((num_tokens_per_seq + BLOCK_Q - 1) // BLOCK_Q) * num_seqs * NUM_HEAD_BLOCKS
        )
    num_2d_prgms = total_num_q_blocks * num_kv_heads
    # if batch contains a prefill

    attn_config, reduce_config = select_3d_config(
        block_size,
        max_seqlen_kv,
        target_num_prgms,
        num_2d_prgms,
        q_dtype,
        kv_buffer_dtype,
        shuffled_kv_cache,
    )

    NUM_SEGMENTS = attn_config["NUM_SEGMENTS_PER_SEQ"]
    if NUM_SEGMENTS > 1:
        segm_output = torch.empty(
            total_num_tokens,
            num_query_heads,
            NUM_SEGMENTS,
            triton.next_power_of_2(kv_lora_rank),
            dtype=torch.float32,
            device=q.device,
        )
        segm_max = torch.empty(
            total_num_tokens,
            num_query_heads,
            NUM_SEGMENTS,
            dtype=torch.float32,
            device=q.device,
        )
        segm_expsum = torch.empty(
            total_num_tokens,
            num_query_heads,
            NUM_SEGMENTS,
            dtype=torch.float32,
            device=q.device,
        )
    else:
        segm_output = out
        segm_max = out  # dummy ptr
        segm_expsum = out  # dummy ptr

    _mla_decode_fwd_kernel[(total_num_q_blocks, num_kv_heads, NUM_SEGMENTS)](
        segm_output_ptr=segm_output,
        segm_max_ptr=segm_max,
        segm_expsum_ptr=segm_expsum,
        query_ptr=q,
        query_scales_ptr=q_scales,
        kv_buffer_ptr=kv_buffer,
        block_tables_ptr=block_tables,
        seq_lens_ptr=seqused_k,
        scale=softmax_scale,
        q_scale_ptr=q_descale,
        kv_scale_ptr=kv_descale,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        block_tables_stride=block_tables.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        query_scales_stride_0=q_scales.stride(0) if q_scales is not None else 0,
        query_scales_stride_1=q_scales.stride(1) if q_scales is not None else 0,
        KV_LORA_RANK=kv_lora_rank,
        QK_ROPE_HEAD_DIM=qk_rope_head_dim,
        stride_kv_buffer_0=kv_buffer.stride(0),
        stride_kv_buffer_1=kv_buffer.stride(1),
        stride_kv_buffer_2=kv_buffer.stride(2),
        stride_kv_buffer_3=kv_buffer.stride(3),
        query_start_len_ptr=cu_seqlens_q,
        num_tokens_per_seq=num_tokens_per_seq,
        BLOCK_Q=BLOCK_Q,
        BLOCK_M=BLOCK_M,
        NUM_HEAD_BLOCKS=NUM_HEAD_BLOCKS,
        ALL_DECODE=ALL_DECODE,
        SHUFFLED_KV_CACHE=shuffled_kv_cache,
        IS_Q_FP8=(q_dtype == e4m3_dtype),
        IS_KV_FP8=(kv_buffer_dtype == e4m3_dtype),
        **attn_config,
    )

    if NUM_SEGMENTS == 1:
        return segm_output
    elif skip_reduce:
        return segm_output, segm_max, segm_expsum

    _reduce_kernel = _mla_decode_fwd_reduce_kernel

    _reduce_kernel[(total_num_tokens, num_query_heads)](
        output_ptr=out,
        segm_output_ptr=segm_output,
        segm_max_ptr=segm_max,
        segm_expsum_ptr=segm_expsum,
        seq_lens_ptr=seqused_k,
        out_scale_ptr=out_scale,
        num_seqs=num_seqs,
        num_query_heads=num_query_heads,
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        block_tables_stride=block_tables.stride(0),
        num_tokens_per_seq=num_tokens_per_seq,
        total_num_tokens=total_num_tokens,
        KV_LORA_RANK=kv_lora_rank,
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=BLOCK_Q,
        ALL_DECODE=ALL_DECODE,
        **reduce_config,
    )
    return out
