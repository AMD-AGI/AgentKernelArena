# The kernels in this file are adapted from vLLM:
# https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_unified_attention.py
"""Standalone unified (paged) attention Triton kernels.

Source: aiter/ops/triton/attention/unified_attention.py (+ _triton_kernels).
The gfx1250-only Gluon kernel dispatch and gfx1250 config branches are dropped;
only the @triton.jit path is kept.
"""

import math

import torch
import triton
import triton.language as tl


# The 3D path needs temporary segment reductions.  Keep one workspace per
# device stream and shape so warmup pays the allocation cost while graph replay
# measures only the attention and reduction kernels.
_segment_workspace_cache = {}


# --- inlined arch detection (utils._triton.arch_info) ---
def _detect_arch_str() -> str:
    try:
        return triton.runtime.driver.active.get_current_target().arch
    except Exception:  # noqa: BLE001 - import-only / no GPU
        return "unknown"


# --- inlined device CU count (utils.device_info.get_num_sms) ---
def get_num_sms():
    # Returns the Compute Unit count of the current device
    current_device_index = torch.cuda.current_device()
    current_device = torch.cuda.get_device_properties(current_device_index)
    num_sms = current_device.multi_processor_count
    return num_sms


# --- inlined e4m3 dtype selection (utils.types.get_fp8_dtypes) ---
if _detect_arch_str() in ("gfx950", "gfx1250", "gfx1200", "gfx1201"):
    e4m3_dtype = torch.float8_e4m3fn
else:
    e4m3_dtype = torch.float8_e4m3fnuz

float8_info = torch.finfo(e4m3_dtype)


# --- inlined GpuArch + get_arch (flash_attn_triton_amd.utils) ---
_CDNA_ARCHS = frozenset({"gfx908", "gfx90a", "gfx940", "gfx941", "gfx942", "gfx950"})
_RDNA_ARCHS = frozenset(
    {
        "gfx1030",
        "gfx1100",
        "gfx1101",
        "gfx1102",
        "gfx1150",
        "gfx1151",
        "gfx1200",
        "gfx1201",
    }
)


class GpuArch:
    """GPU architecture information."""

    def __init__(self, name: str, family=None):
        self.name = name
        self.family = family

    @property
    def is_cdna(self) -> bool:
        return self.family == "cdna"

    @property
    def is_rdna(self) -> bool:
        return self.family == "rdna"


def get_arch() -> GpuArch:
    name = _detect_arch_str()
    if name in _CDNA_ARCHS:
        return GpuArch(name=name, family="cdna")
    if name in _RDNA_ARCHS:
        return GpuArch(name=name, family="rdna")
    return GpuArch(name=name)


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


DEVICE_ARCH = _detect_arch_str()
IS_DEVICE_ARCH_GFX12 = DEVICE_ARCH in ("gfx1250",)
WARP_SIZE = 32 if IS_DEVICE_ARCH_GFX12 else 64
WAPR_SIZE_LOG2 = int(math.log2(WARP_SIZE))


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
def find_seq_idx(
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


@triton.jit
def kernel_unified_attention_2d(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    value_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    sink_ptr,  # [num_query_heads]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    alibi_slopes_ptr,  # [num_query_heads]
    qq_bias_ptr,  # [num_query_tokens, num_query_tokens]
    scale: tl.constexpr,  # float32
    q_descale_ptr,  # float32
    k_descale_ptr,  # float32
    v_descale_ptr,  # float32
    out_scale_ptr,  # float32
    softcap,  # float32
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    qq_bias_stride_0: tl.int64,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    TILE_SIZE: tl.constexpr,  # int must be power of 2
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    USE_QQ_BIAS: tl.constexpr,  # bool
    USE_SOFTCAP: tl.constexpr,  # bool
    USE_SINKS: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.constexpr,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,  # int
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
    ALL_DECODE: tl.constexpr = False,  # bool
    SHUFFLED_KV_CACHE: tl.constexpr = False,  # bool
    K_WIDTH: tl.constexpr = 0,  # int
):
    kv_head_idx = tl.program_id(0)
    q_block_global_idx = tl.program_id(1)

    # needed to use exp2 (exp2 -> exp conversion)
    RCP_LN2 = 1.4426950408889634
    qk_scale = scale * RCP_LN2

    if ALL_DECODE:
        seq_idx = q_block_global_idx
        q_block_local_idx: tl.int32 = 0
        cur_batch_query_len: tl.int32 = 1
        cur_batch_in_all_start_index: tl.int32 = q_block_global_idx
    else:
        seq_idx = find_seq_idx(
            query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
        )

        q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx

        q_block_local_idx = q_block_global_idx - q_block_start_idx

        cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
        cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

        cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

        if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
            return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    offs_shfl = None
    if SHUFFLED_KV_CACHE:
        offs_shfl = tl.arange(0, TILE_SIZE * HEAD_SIZE_PADDED)

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    if HEAD_SIZE_PADDED != HEAD_SIZE:
        dim_mask = offs_d < HEAD_SIZE
    else:
        dim_mask = tl.full((1,), 1, dtype=tl.int1)
    query_mask_0 = query_pos < cur_batch_query_len
    query_mask_1 = query_offset_1 < num_query_heads

    if ALL_DECODE or BLOCK_M >= num_query_heads:
        Q_cache_modifier: tl.constexpr = ".cg"
    else:
        Q_cache_modifier: tl.constexpr = ""
    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
        cache_modifier=Q_cache_modifier,
    )

    block_table_offset = seq_idx * block_table_stride

    if not USE_SINKS:
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    else:
        # Prescale with RCP_LN2, needed for exp2
        M = (
            tl.load(
                sink_ptr + query_offset_1,
                mask=query_mask_1,
                other=float("-inf"),
            ).to(dtype=tl.float32)
            * RCP_LN2
        )

    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # context length for this particular sequences
    context_len = seq_len - cur_batch_query_len

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    # query-query attention bias
    if USE_QQ_BIAS:
        qq_bias_row_ptrs = (
            qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0
        )  # shape: [BLOCK_M]

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
    if SLIDING_WINDOW > 0:
        # Query rows covered by this Q-block
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = tl.minimum(
            qpos_lo + (BLOCK_M - 1) // num_queries_per_kv,
            cur_batch_query_len - 1,
        )
        # For sliding window, each query position q can only attend to
        # keys in the range [q_abs - SLIDING_WINDOW + 1, q_abs]
        # where q_abs = context_len + q
        # The union of allowed key positions for this Q-block is:
        # [context_len + qpos_lo - SLIDING_WINDOW + 1, context_len + qpos_hi]
        first_allowed_key = context_len + qpos_lo - SLIDING_WINDOW + 1
        last_allowed_key = context_len + qpos_hi
        # Convert to tile indices and clamp
        tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
        tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)
    if q_descale_ptr is not None:
        q_descale = tl.load(q_descale_ptr)
        qk_scale = qk_scale * q_descale
    else:
        q_descale = None
    if k_descale_ptr is not None and v_descale_ptr is not None:
        k_descale = tl.load(k_descale_ptr)
        v_descale = tl.load(v_descale_ptr)
        qk_scale = qk_scale * k_descale
    else:
        k_descale = None
        v_descale = None
    KV_cache_modifier: tl.constexpr = ".cg" if ALL_DECODE else ""
    # iterate through tiles (now limited to the sliding window range)
    for j in range(tile_start, tile_end):
        seq_offset = j * TILE_SIZE + offs_t
        # to reduce the masking effect when not needed
        if TILE_SIZE == BLOCK_SIZE:
            tile_mask = tl.full((1,), 1, dtype=tl.int1)
        else:
            tile_mask = seq_offset < max_seq_prefix_len

        k_mask = None
        v_mask = None
        other = None
        if SHUFFLED_KV_CACHE:
            physical_block_idx_shfl = tl.load(
                block_tables_ptr + block_table_offset + j
            ).to(tl.int64)
            k_offset = (
                physical_block_idx_shfl * stride_k_cache_0
                + kv_head_idx * stride_k_cache_1
                + offs_shfl
            )

            v_offset = (
                physical_block_idx_shfl * stride_v_cache_0
                + kv_head_idx * stride_v_cache_1
                + offs_shfl
            )
        else:
            physical_block_idx = tl.load(
                block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
            ).to(tl.int64)

            v_offset = (
                physical_block_idx[:, None] * stride_v_cache_0
                + kv_head_idx * stride_v_cache_2
                + offs_d[None, :] * stride_v_cache_3
                + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
            )
            v_mask = dim_mask[None, :] & tile_mask[:, None]

            k_offset = (
                physical_block_idx[None, :] * stride_k_cache_0
                + kv_head_idx * stride_k_cache_2
                + offs_d[:, None] * stride_k_cache_3
                + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
            )
            k_mask = dim_mask[:, None] & tile_mask[None, :]
            other = 0.0

        # K : (HEAD_SIZE, TILE_SIZE)
        K_load = tl.load(
            key_cache_ptr + k_offset,
            mask=k_mask,
            other=other,
            cache_modifier=KV_cache_modifier,
        )

        K = K_load.to(Q.dtype)
        if SHUFFLED_KV_CACHE:
            K = (
                K.reshape(
                    HEAD_SIZE_PADDED // K_WIDTH,
                    TILE_SIZE,
                    K_WIDTH,
                )
                .permute(1, 0, 2)
                .reshape(TILE_SIZE, HEAD_SIZE_PADDED)
                .trans(1, 0)
            )

        # V : (TILE_SIZE, HEAD_SIZE)
        V_load = tl.load(
            value_cache_ptr + v_offset,
            mask=v_mask,
            other=other,
            cache_modifier=KV_cache_modifier,
        )

        V = V_load.to(Q.dtype)
        if SHUFFLED_KV_CACHE:
            V = (
                V.reshape(
                    TILE_SIZE // K_WIDTH,
                    HEAD_SIZE_PADDED,
                    K_WIDTH,
                )
                .permute(0, 2, 1)
                .reshape(TILE_SIZE, HEAD_SIZE_PADDED)
            )

        # S : (BLOCK_M, TILE_SIZE)
        # qk_scale = scale * RCP_LN2 (log_2 e) so that we can use exp2 later
        S = qk_scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            # softcap here uses exp2 and consumes RCP_LN2 conversion.
            # multiply by RCP_LN2 again to be used in later exp2
            S = apply_softcap(S, softcap) * RCP_LN2
        seq_mask = seq_offset[None, :] < context_len + query_pos[:, None] + 1

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        if SLIDING_WINDOW > 0:
            S = tl.where(
                (context_len + query_pos[:, None] - seq_offset) < SLIDING_WINDOW,
                S,
                float("-inf"),
            )

        if USE_ALIBI_SLOPES:
            # prescale w. RCP_LN2 for later exp2
            S += alibi_slope[:, None] * (seq_offset - context_len) * RCP_LN2

        if USE_QQ_BIAS:
            # compute key positions relative to query section
            key_rel_pos = seq_offset - context_len  # shape: [BLOCK_SIZE]
            # load bias only for keys that correspond to queries
            is_query_key = key_rel_pos >= 0 and key_rel_pos < qq_bias_stride_0
            qq_bias = tl.load(
                qq_bias_row_ptrs + key_rel_pos[None, :],
                mask=is_query_key[None, :],  # avoid OOB for context keys
                other=0.0,
            )
            # prescale w. RCP_LN2 for later exp2
            S += qq_bias * RCP_LN2

        # compute running maximum
        # m_j : (BLOCK_M,)
        m_j = tl.maximum(M, tl.max(S, axis=1))

        # For sliding window there's a chance the max is -inf due to masking of
        # the entire row. In this case we need to set m_j 0 to avoid NaN
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

        # P : (BLOCK_M, TILE_SIZE)
        P = tl.math.exp2(S - m_j[:, None])

        # l_j : (BLOCK_M,)
        l_j = tl.sum(P, axis=1)

        # alpha : (BLOCK_M, )
        alpha = tl.math.exp2(M - m_j)

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc = acc * alpha[:, None]

        # update constants
        L = L * alpha + l_j
        M = m_j

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc = tl.dot(P.to(V.dtype), V, acc=acc)

    # epilogue
    # This helps the compiler do Newton Raphson on l_i vs on acc which is much larger.
    if v_descale is not None:
        one_over_L = v_descale / L[:, None]
    else:
        one_over_L = 1.0 / L[:, None]
    acc = acc * one_over_L
    if out_scale_ptr is not None:
        acc = acc / tl.load(out_scale_ptr)

    if output_ptr.type.element_ty.is_fp8():
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    output_offset = (
        query_offset_0[:, None] * output_stride_0
        + query_offset_1[:, None] * output_stride_1
        + offs_d[None, :]
    )

    tl.store(
        output_ptr + output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )


kernel_unified_attention_3d_repr = make_kernel_repr(
    "kernel_unified_attention_3d",
    [
        "num_query_heads",
        "num_queries_per_kv",
        "BLOCK_SIZE",
        "TILE_SIZE",
        "HEAD_SIZE",
        "NUM_SEGMENTS_PER_SEQ",
        "num_warps",
        "waves_per_eu",
        "num_stages",
        "ALL_DECODE",
        "SHUFFLED_KV_CACHE",
        "IS_Q_FP8",
        "IS_KV_FP8",
    ],
)


@triton.jit(repr=kernel_unified_attention_3d_repr)
def kernel_unified_attention_3d(
    segm_output_ptr,
    # [num_tokens, num_query_heads, num_segments, head_size]
    segm_max_ptr,  # [num_tokens, num_query_heads, num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, num_segments]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    value_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    sink_ptr,  # [num_query_heads]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    alibi_slopes_ptr,  # [num_query_heads]
    qq_bias_ptr,  # [num_query_tokens, num_query_tokens]
    scale,  # float32
    q_descale_ptr,  # float32
    k_descale_ptr,  # float32
    v_descale_ptr,  # float32
    out_scale_ptr,  # float32
    softcap,  # float32
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    qq_bias_stride_0: tl.int64,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    TILE_SIZE: tl.constexpr,  # int, must be power of 2
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    USE_QQ_BIAS: tl.constexpr,  # bool
    USE_SOFTCAP: tl.constexpr,  # bool
    USE_SINKS: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.constexpr,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,  # int
    num_warps: tl.constexpr,  # int
    waves_per_eu: tl.constexpr,  # int
    num_stages: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    ALL_DECODE: tl.constexpr = False,  # bool
    SHUFFLED_KV_CACHE: tl.constexpr = False,  # bool
    K_WIDTH: tl.constexpr = 0,  # int
    IS_Q_FP8: tl.constexpr = False,  # bool
    IS_KV_FP8: tl.constexpr = False,  # bool
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2)

    # needed to use exp2 (exp2 -> exp conversion)
    RCP_LN2 = 1.4426950408889634
    qk_scale = scale * RCP_LN2

    if ALL_DECODE:
        seq_idx = q_block_global_idx
        q_block_local_idx: tl.int32 = 0
        cur_batch_query_len: tl.int32 = 1
        cur_batch_in_all_start_index: tl.int32 = q_block_global_idx
    else:
        seq_idx = find_seq_idx(
            query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
        )

        q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx

        q_block_local_idx = q_block_global_idx - q_block_start_idx

        cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
        cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

        cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

        if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
            return

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)

    offs_shfl = None
    if SHUFFLED_KV_CACHE:
        offs_shfl = tl.arange(0, TILE_SIZE * HEAD_SIZE_PADDED)

    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    if HEAD_SIZE_PADDED != HEAD_SIZE:
        dim_mask = offs_d < HEAD_SIZE
    else:
        dim_mask = tl.full((1,), 1, dtype=tl.int1)
    query_mask_0 = query_pos < cur_batch_query_len
    query_mask_1 = query_offset_1 < num_query_heads

    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    if USE_SINKS:
        if segm_idx == 0:
            # Prescale with RCP_LN2, needed for exp2
            M = (
                tl.load(
                    sink_ptr + query_offset_1,
                    mask=query_mask_1,
                    other=float("-inf"),
                ).to(dtype=tl.float32)
                * RCP_LN2
            )
        else:
            M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    else:
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)

    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    # context length for this particular sequences
    context_len = seq_len - cur_batch_query_len

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    # query-query attention bias
    if USE_QQ_BIAS:
        qq_bias_row_ptrs = (
            qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0
        )  # shape: [BLOCK_M]

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

    KV_cache_modifier: tl.constexpr = ".cg" if ALL_DECODE else ""
    if q_descale_ptr is not None:
        q_descale = tl.load(q_descale_ptr)
        qk_scale = qk_scale * q_descale
    else:
        q_descale = None

    if k_descale_ptr is not None:
        k_scale = tl.load(k_descale_ptr)
        qk_scale = qk_scale * k_scale
    else:
        k_scale = None

    out_factor: tl.float32 = 1.0
    if v_descale_ptr is not None:
        out_factor = tl.load(v_descale_ptr)

    if out_scale_ptr is not None:
        out_factor = out_factor / tl.load(out_scale_ptr)

    # iterate through tiles within current segment
    for j in range(
        segm_idx * tiles_per_segment,
        min((segm_idx + 1) * tiles_per_segment, num_tiles),
    ):
        seq_offset = j * TILE_SIZE + offs_t
        if TILE_SIZE == BLOCK_SIZE:
            tile_mask = tl.full((1,), 1, dtype=tl.int1)
        else:
            tile_mask = seq_offset < max_seq_prefix_len

        k_mask = None
        v_mask = None
        other = None
        if SHUFFLED_KV_CACHE:
            physical_block_idx_shfl = tl.load(
                block_tables_ptr + block_table_offset + j
            ).to(tl.int64)
            k_offset = (
                physical_block_idx_shfl * stride_k_cache_0
                + kv_head_idx * stride_k_cache_1
                + offs_shfl
            )

            v_offset = (
                physical_block_idx_shfl * stride_v_cache_0
                + kv_head_idx * stride_v_cache_1
                + offs_shfl
            )
        else:
            physical_block_idx = tl.load(
                block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
            ).to(tl.int64)

            v_offset = (
                physical_block_idx[:, None] * stride_v_cache_0
                + kv_head_idx * stride_v_cache_2
                + offs_d[None, :] * stride_v_cache_3
                + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
            )
            v_mask = dim_mask[None, :] & tile_mask[:, None]

            k_offset = (
                physical_block_idx[None, :] * stride_k_cache_0
                + kv_head_idx * stride_k_cache_2
                + offs_d[:, None] * stride_k_cache_3
                + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
            )
            k_mask = dim_mask[:, None] & tile_mask[None, :]
            other = 0.0

        # K : (HEAD_SIZE, TILE_SIZE)
        K_load = tl.load(
            key_cache_ptr + k_offset,
            mask=k_mask,
            other=other,
            cache_modifier=KV_cache_modifier,
        )

        K = K_load.to(Q.dtype)
        if SHUFFLED_KV_CACHE:
            K = (
                K.reshape(
                    HEAD_SIZE_PADDED // K_WIDTH,
                    TILE_SIZE,
                    K_WIDTH,
                )
                .permute(1, 0, 2)
                .reshape(TILE_SIZE, HEAD_SIZE_PADDED)
                .trans(1, 0)
            )

        # V : (TILE_SIZE, HEAD_SIZE)
        V_load = tl.load(
            value_cache_ptr + v_offset,
            mask=v_mask,
            other=other,
            cache_modifier=KV_cache_modifier,
        )

        V = V_load.to(Q.dtype)
        if SHUFFLED_KV_CACHE:
            V = (
                V.reshape(
                    TILE_SIZE // K_WIDTH,
                    HEAD_SIZE_PADDED,
                    K_WIDTH,
                )
                .permute(0, 2, 1)
                .reshape(TILE_SIZE, HEAD_SIZE_PADDED)
            )

        seq_mask = seq_offset[None, :] < context_len + query_pos[:, None] + 1

        # S : (BLOCK_M, TILE_SIZE)
        # qk_scale = scale * RCP_LN2 (log_2 e) so that we can use exp2 later
        S = qk_scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            # softcap here uses exp2 and consumes RCP_LN2 conversion.
            # multiply by RCP_LN2 again to be used in later exp2
            S = apply_softcap(S, softcap) * RCP_LN2

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        if SLIDING_WINDOW > 0:
            S = tl.where(
                (context_len + query_pos[:, None] - seq_offset) < SLIDING_WINDOW,
                S,
                float("-inf"),
            )

        if USE_ALIBI_SLOPES:
            # prescale w. RCP_LN2 for later exp2
            S += alibi_slope[:, None] * (seq_offset - context_len) * RCP_LN2

        if USE_QQ_BIAS:
            # compute key positions relative to query section
            key_rel_pos = seq_offset - context_len  # shape: [BLOCK_SIZE]
            # load bias only for keys that correspond to queries
            is_query_key = key_rel_pos >= 0 and key_rel_pos < qq_bias_stride_0
            qq_bias = tl.load(
                qq_bias_row_ptrs + key_rel_pos[None, :],
                mask=is_query_key[None, :],  # avoid OOB for context keys
                other=0.0,
            )
            # prescale w. RCP_LN2 for later exp2
            S += qq_bias * RCP_LN2

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

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc = acc * alpha[:, None]

        # update constants
        L = L * alpha + l_j
        M = m_j

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc = tl.dot(P.to(V.dtype), V, acc=acc)

    acc = acc * out_factor
    if NUM_SEGMENTS_PER_SEQ == 1:
        one_over_L = 1.0 / L[:, None]
        acc = acc * one_over_L

    segm_output_offset = (
        query_offset_0[:, None].to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + segm_idx * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    tl.store(
        segm_output_ptr + segm_output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )
    if NUM_SEGMENTS_PER_SEQ > 1:
        segm_offset = (
            query_offset_0.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
            + query_offset_1 * NUM_SEGMENTS_PER_SEQ
            + segm_idx
        )
        tl.store(segm_max_ptr + segm_offset, M, mask=query_mask_0 & query_mask_1)
        tl.store(segm_expsum_ptr + segm_offset, L, mask=query_mask_0 & query_mask_1)


_reduce_segments_repr = make_kernel_repr(
    "reduce_segments",
    [
        "num_query_heads",
        "TILE_SIZE",
        "HEAD_SIZE",
        "NUM_SEGMENTS_PER_SEQ",
    ],
)


@triton.jit(repr=_reduce_segments_repr)
def reduce_segments(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    segm_output_ptr,
    # [num_tokens, num_query_heads, max_num_segments, head_size]
    segm_max_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    seq_lens_ptr,  # [num_seqs]
    num_seqs,  # int
    num_query_heads: tl.constexpr,  # int
    out_scale_ptr,  # float32
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    block_table_stride: tl.int64,  # int
    TILE_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int, must be power of 2
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    out_scale = None
    if out_scale_ptr is not None:
        out_scale = 1 / tl.load(out_scale_ptr)

    seq_idx = find_seq_idx(
        query_start_len_ptr, query_token_idx, num_seqs, BLOCK_Q, False
    )

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    # create masks for subsequent loads
    act_num_segments = cdiv_fn(seq_len, tiles_per_segment * TILE_SIZE)
    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full(
        [NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32
    )

    if HEAD_SIZE_PADDED != HEAD_SIZE:
        offs_d = tl.arange(0, HEAD_SIZE_PADDED)
        dim_mask = offs_d < HEAD_SIZE
    else:
        dim_mask = tl.full((1,), 1, dtype=tl.int1)

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
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_head_idx * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    segm_output = tl.load(
        segm_output_ptr + segm_output_offset,
        mask=segm_mask[:, None] & dim_mask[None, :],
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
        + tl.arange(0, HEAD_SIZE_PADDED)
    )
    tl.store(
        output_ptr + output_offset, acc.to(output_ptr.type.element_ty), mask=dim_mask
    )


# Host-side launch logic. The gfx1250-only Gluon kernel dispatch is dropped (those
# kernels cannot be detached from the _gluon_kernels package).
def is_2d_gluon_available(
    q_dtype, kv_cache_dtype, softcap, use_qq_bias, use_alibi_slopes
):
    return False


def select_2d_config(
    block_size,
    head_size,
    sliding_window,
    all_decode,
    max_seqlen_q,
    max_seqlen_k,
    num_queries_per_kv,
    num_2d_prgms,
    q_dtype,
    kv_cache_dtype,
    shuffled_kv_cache,
):
    arch = get_arch()

    BLOCK_M = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )

    TILE_SIZE = 32 if arch.name == "gfx1201" else 16 if arch.is_rdna else 64
    waves_per_eu = 8 if arch.name == "gfx1151" else 6 if arch.is_rdna else 2

    max_num_stages_2d = 2 if head_size > 128 else 4

    # base prefill, for short cases
    if not all_decode:
        num_stages_2d, num_warps = 1, 2
    # pure decode config
    else:
        # to not have masking when loading KV
        TILE_SIZE = min(64, triton.next_power_of_2(block_size))
        if arch.is_rdna:
            num_stages_2d, num_warps = 1, 4
        else:
            num_stages_2d, num_warps = 3, 2

    # large prefill config
    if max_seqlen_q >= 256:
        BLOCK_M = 64 if arch.is_rdna else 128
        num_stages_2d, num_warps = 1, 4

    BLOCK_Q = BLOCK_M // num_queries_per_kv
    num_stages_2d = min(max_num_stages_2d, num_stages_2d)

    # fix TILE_SIZE to block_size if shuffled_kv_cache is True
    if shuffled_kv_cache:
        if q_dtype == e4m3_dtype and kv_cache_dtype == e4m3_dtype:
            assert (
                block_size >= 32
            ), "For A8W8 Unified Attention with pre-shuffled KV cache, only block_size >= 32 is supported"
        TILE_SIZE = block_size
    elif q_dtype == e4m3_dtype and kv_cache_dtype == e4m3_dtype:
        TILE_SIZE = max(32, TILE_SIZE)

    return {
        "BLOCK_M": BLOCK_M,
        "BLOCK_Q": BLOCK_Q,
        "TILE_SIZE": TILE_SIZE,
        "num_warps": num_warps,
        "num_stages": num_stages_2d,
        "waves_per_eu": waves_per_eu,
    }


def select_3d_config(
    head_size,
    block_size,
    max_seqlen_k,
    target_num_prgms,
    num_2d_prgms,
    q_dtype: torch.dtype,
    kv_cache_dtype: torch.dtype,
    shuffled_kv_cache: bool = False,
    NUM_BLOCKS_GATHER_PER_TILE: int = 1,
):
    # TODO: wait for Triton compiler to support ds_load_tr4 before we can include torch.uint8 kv_cache_dtype
    # assert kv_cache_dtype in (torch.bfloat16, e4m3_dtype, torch.uint8, ), f"kv_cache_dtype only supports BF16 ({torch.bfloat16}), FP8 ({e4m3_dtype}), FP4 ({torch.uint8})"
    assert kv_cache_dtype in (
        torch.bfloat16,
        e4m3_dtype,
    ), f"kv_cache_dtype only supports BF16 ({torch.bfloat16}), FP8 ({e4m3_dtype})"
    reduce_num_warps = 2
    attn_warps = 2
    waves_per_eu = 2
    num_segments = 0
    attn_stages = 2

    occ = waves_per_eu * 4 // attn_warps
    target_num_prgms = target_num_prgms * occ

    TILE_SIZE = min(64, triton.next_power_of_2(block_size))

    MAX_SEGMENTS = min(128, math.ceil(max_seqlen_k / TILE_SIZE))
    MIN_SEGMENTS = min(8, MAX_SEGMENTS)
    if num_segments == 0:
        num_segments = math.ceil(target_num_prgms / num_2d_prgms)
        num_segments = min(num_segments, MAX_SEGMENTS)
        num_segments = triton.next_power_of_2(num_segments)
        num_segments = min(num_segments, 128)
        num_segments = max(num_segments, MIN_SEGMENTS)

    if num_segments == MIN_SEGMENTS:
        reduce_num_warps = 1

    if shuffled_kv_cache:
        if q_dtype == e4m3_dtype and kv_cache_dtype == e4m3_dtype:
            assert (
                block_size >= 32
            ), "For A8W8 Unified Attention with pre-shuffled KV cache, only block_size >= 32 is supported"
        TILE_SIZE = block_size
    elif q_dtype == e4m3_dtype and kv_cache_dtype == e4m3_dtype:
        TILE_SIZE = max(32, TILE_SIZE)

    if NUM_BLOCKS_GATHER_PER_TILE > 1:
        # force gather mode
        assert NUM_BLOCKS_GATHER_PER_TILE in [
            4,
            8,
        ], "Only NUM_BLOCKS_GATHER_PER_TILE = 4 or 8 is supported"
        attn_warps = 2
        waves_per_eu = 1
        num_segments = max(1, num_segments // NUM_BLOCKS_GATHER_PER_TILE)
        TILE_SIZE = block_size * NUM_BLOCKS_GATHER_PER_TILE
    elif TILE_SIZE > block_size:
        assert (
            TILE_SIZE % block_size == 0
        ), "TILE_SIZE needs to be divisible by block_size"
        NUM_BLOCKS_GATHER_PER_TILE = TILE_SIZE // block_size

    attn_config = {
        "TILE_SIZE": TILE_SIZE,
        "NUM_SEGMENTS_PER_SEQ": num_segments,
        "num_warps": attn_warps,
        "waves_per_eu": waves_per_eu,
        "num_stages": attn_stages,
    }

    reduce_config = {
        "TILE_SIZE": TILE_SIZE,
        "NUM_SEGMENTS_PER_SEQ": num_segments,
        "num_warps": reduce_num_warps,
        "waves_per_eu": 2,
        "num_stages": 1,
    }

    return attn_config, reduce_config


def use_2d_kernel(
    head_size,
    sliding_window,
    all_decode,
    max_seqlen_q,
    max_seqlen_k,
    target_num_prgms,
    num_2d_prgms,
):
    return (
        (sliding_window > 0)
        or (max_seqlen_k <= 512)
        or (num_2d_prgms > target_num_prgms)
    )


def unified_attention(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    q_scales=None,
    alibi_slopes=None,
    output_scale=None,
    qq_bias=None,
    # Optional tensor for sinks
    sinks=None,
    shuffled_kv_cache: bool = False,
    skip_reduce: bool = False,
):
    assert causal, "Only causal attention is supported"

    use_alibi_slopes = alibi_slopes is not None
    use_qq_bias = qq_bias is not None
    SLIDING_WINDOW = 1 + window_size[0]

    q_dtype = q.dtype
    kv_cache_dtype = k.dtype
    num_tokens, num_query_heads, head_size = q.shape

    if sinks is not None:
        assert sinks.shape[0] == num_query_heads, "Sinks must be num_query_heads size"

    BLOCK_SCALES_SIZE = 16
    if q_dtype == torch.uint8:
        # A4W4
        assert q_scales is not None and q_scales.dtype == e4m3_dtype
        head_size = head_size * 2
        QUERY_DTYPE = "nvfp4"
    elif q_dtype == e4m3_dtype:
        QUERY_DTYPE = "fp8"
    else:
        QUERY_DTYPE = "bf16"

    if kv_cache_dtype == torch.uint8:
        KV_CACHE_DTYPE = "nvfp4"
    elif kv_cache_dtype == e4m3_dtype:
        KV_CACHE_DTYPE = "fp8"
    else:
        KV_CACHE_DTYPE = "bf16"

    if shuffled_kv_cache:
        SCALE_K_WIDTH = 4
        if kv_cache_dtype == torch.uint8:
            num_blocks, num_kv_heads, block_size, _ = k.shape
            K_WIDTH = 16
            SCALE_K = head_size // 16
            SCALE_K_WIDTH = (
                min(16, triton.next_power_of_2(SCALE_K)) if SCALE_K >= 4 else SCALE_K
            )
        else:
            # key_cache: num_blocks, num_kv_heads, head_size // x, block_size, x
            # value_cache: num_blocks, num_kv_heads, block_size // x, head_size, x
            num_blocks, num_kv_heads, _, block_size, K_WIDTH = k.shape
    else:
        # key_cache and value_cache: num_blocks, block_size, num_kv_heads, head_size
        num_blocks, block_size, num_kv_heads, _ = k.shape
        K_WIDTH = 16 if kv_cache_dtype == e4m3_dtype else 8
        SCALE_K_WIDTH = 4

    num_seqs = len(seqused_k)
    num_queries_per_kv = num_query_heads // num_kv_heads

    BLOCK_M = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    assert BLOCK_Q >= 1
    cu_count = get_num_sms()
    target_num_prgms = cu_count * 4
    ALL_DECODE = max_seqlen_q == 1
    if ALL_DECODE:
        total_num_q_blocks = num_seqs
    else:
        total_num_q_blocks = num_tokens // BLOCK_Q + num_seqs
    num_2d_prgms = total_num_q_blocks * num_kv_heads
    ALL_DECODE = int(max_seqlen_q) == 1
    # if batch contains a prefill
    if use_2d_kernel(
        head_size,
        SLIDING_WINDOW,
        ALL_DECODE,
        max_seqlen_q,
        max_seqlen_k,
        target_num_prgms,
        num_2d_prgms,
    ):
        config = select_2d_config(
            block_size,
            head_size,
            SLIDING_WINDOW,
            ALL_DECODE,
            max_seqlen_q,
            max_seqlen_k,
            num_queries_per_kv,
            num_2d_prgms,
            q_dtype,
            kv_cache_dtype,
            shuffled_kv_cache,
        )
        assert config["BLOCK_Q"] >= 1
        if ALL_DECODE:
            total_num_q_blocks = num_seqs
        else:
            total_num_q_blocks = q.shape[0] // config["BLOCK_Q"] + num_seqs

        kernel_unified_attention_2d[
            (
                num_kv_heads,
                total_num_q_blocks,
            )
        ](
            output_ptr=out,
            query_ptr=q,
            key_cache_ptr=k,
            value_cache_ptr=v,
            sink_ptr=sinks,
            block_tables_ptr=block_table,
            seq_lens_ptr=seqused_k,
            alibi_slopes_ptr=alibi_slopes,
            qq_bias_ptr=qq_bias,
            scale=softmax_scale,
            q_descale_ptr=q_descale,
            k_descale_ptr=k_descale,
            v_descale_ptr=v_descale,
            out_scale_ptr=output_scale,
            softcap=softcap,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
            block_table_stride=block_table.stride(0),
            query_stride_0=q.stride(0),
            query_stride_1=q.stride(1),
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
            BLOCK_SIZE=block_size,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            USE_ALIBI_SLOPES=use_alibi_slopes,
            USE_QQ_BIAS=use_qq_bias,
            USE_SOFTCAP=(softcap > 0),
            USE_SINKS=(sinks is not None),
            SLIDING_WINDOW=SLIDING_WINDOW,
            stride_k_cache_0=k.stride(0),
            stride_k_cache_1=k.stride(1),
            stride_k_cache_2=k.stride(2),
            stride_k_cache_3=k.stride(3),
            stride_v_cache_0=v.stride(0),
            stride_v_cache_1=v.stride(1),
            stride_v_cache_2=v.stride(2),
            stride_v_cache_3=v.stride(3),
            query_start_len_ptr=cu_seqlens_q,
            num_seqs=num_seqs,
            ALL_DECODE=ALL_DECODE,
            SHUFFLED_KV_CACHE=shuffled_kv_cache,
            K_WIDTH=K_WIDTH,
            **config,
        )
        return out

    else:
        NUM_BLOCKS_GATHER_PER_TILE = 1
        attn_config, reduce_config = select_3d_config(
            head_size,
            block_size,
            max_seqlen_k,
            target_num_prgms,
            num_2d_prgms,
            q_dtype,
            kv_cache_dtype,
            shuffled_kv_cache,
            NUM_BLOCKS_GATHER_PER_TILE,
        )
        NUM_SEGMENTS = attn_config["NUM_SEGMENTS_PER_SEQ"]
        if NUM_SEGMENTS > 1:
            stream = torch.cuda.current_stream(q.device)
            workspace_key = (
                q.device,
                stream.cuda_stream,
                q.shape[0],
                num_query_heads,
                NUM_SEGMENTS,
                triton.next_power_of_2(head_size),
            )
            workspace = _segment_workspace_cache.get(workspace_key)
            if workspace is None:
                workspace = (
                    torch.empty(
                        q.shape[0],
                        num_query_heads,
                        NUM_SEGMENTS,
                        triton.next_power_of_2(head_size),
                        dtype=torch.float32,
                        device=q.device,
                    ),
                    torch.empty(
                        q.shape[0],
                        num_query_heads,
                        NUM_SEGMENTS,
                        dtype=torch.float32,
                        device=q.device,
                    ),
                    torch.empty(
                        q.shape[0],
                        num_query_heads,
                        NUM_SEGMENTS,
                        dtype=torch.float32,
                        device=q.device,
                    ),
                )
                _segment_workspace_cache[workspace_key] = workspace
            segm_output, segm_max, segm_expsum = workspace
        else:
            segm_output = out
            segm_max = out  # dummy ptr
            segm_expsum = out  # dummy ptr

        kernel_unified_attention_3d[
            (total_num_q_blocks, num_kv_heads, NUM_SEGMENTS)
        ](
            segm_output_ptr=segm_output,
            segm_max_ptr=segm_max,
            segm_expsum_ptr=segm_expsum,
            query_ptr=q,
            key_cache_ptr=k,
            value_cache_ptr=v,
            sink_ptr=sinks,
            block_tables_ptr=block_table,
            seq_lens_ptr=seqused_k,
            alibi_slopes_ptr=alibi_slopes,
            qq_bias_ptr=qq_bias,
            scale=softmax_scale,
            q_descale_ptr=q_descale,
            k_descale_ptr=k_descale,
            v_descale_ptr=v_descale,
            out_scale_ptr=(
                output_scale
                if (output_scale is not None and NUM_SEGMENTS == 1)
                else None
            ),
            softcap=softcap,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
            block_table_stride=block_table.stride(0),
            query_stride_0=q.stride(0),
            query_stride_1=q.stride(1),
            qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
            BLOCK_SIZE=block_size,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            USE_ALIBI_SLOPES=use_alibi_slopes,
            USE_QQ_BIAS=use_qq_bias,
            USE_SOFTCAP=(softcap > 0),
            USE_SINKS=(sinks is not None),
            SLIDING_WINDOW=SLIDING_WINDOW,
            stride_k_cache_0=k.stride(0),
            stride_k_cache_1=k.stride(1),
            stride_k_cache_2=k.stride(2),
            stride_k_cache_3=k.stride(3),
            stride_v_cache_0=v.stride(0),
            stride_v_cache_1=v.stride(1),
            stride_v_cache_2=v.stride(2),
            stride_v_cache_3=v.stride(3),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            num_seqs=num_seqs,
            BLOCK_M=BLOCK_M,
            ALL_DECODE=ALL_DECODE,
            SHUFFLED_KV_CACHE=shuffled_kv_cache,
            K_WIDTH=K_WIDTH,
            IS_Q_FP8=(q_dtype == e4m3_dtype),
            IS_KV_FP8=(kv_cache_dtype == e4m3_dtype),
            **attn_config,
        )

        if NUM_SEGMENTS == 1:
            return segm_output
        elif skip_reduce:
            return segm_output, segm_max, segm_expsum

        reduce_segments[(q.shape[0], num_query_heads)](
            output_ptr=out,
            segm_output_ptr=segm_output,
            segm_max_ptr=segm_max,
            segm_expsum_ptr=segm_expsum,
            seq_lens_ptr=seqused_k,
            num_seqs=num_seqs,
            num_query_heads=num_query_heads,
            out_scale_ptr=output_scale,
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            block_table_stride=block_table.stride(0),
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            **reduce_config,
        )

        return out
