# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused split-K reduce + per-head weighted RMSNorm + RoPE (tail) on Q,
per-row weighted RMSNorm + RoPE (tail) on KV (+ optional SWA KV write).

Grid: ``(cdiv(M, BLOCK_SIZE_M), num_local_heads + 1)``. Each program tile
handles ``BLOCK_SIZE_M`` tokens. Programs with ``pid_h < num_local_heads``
load a query head tile ``[NUM_SPLITK, BLOCK_SIZE_M, HEAD_DIM]`` from
``q_in``, reduce over the split-K axis, apply per-head weighted batched
RMSNorm, store the (pre-RoPE) head into ``q_out``, then call the batched
RoPE on the last ``rope_head_dim`` elements. Programs with
``pid_h == num_local_heads`` load the full ``[BLOCK_SIZE_M, HEAD_DIM]``
kv tile, apply weighted batched RMSNorm over ``head_dim``, store the
normed nope part back into ``kv``, then extract the tail with the same
reshape+sum trick used for q, apply RoPE, write the result to the kv
tail, and optionally scatter both parts into ``swa_kv``.

``q_in`` layout (driven by API helper):
- 2D: ``[M, N]`` — ``q_in_splitk_stride`` = 0, ``NUM_SPLITK`` = 1.
- 3D: ``[num_splitk, M, N]`` — ``q_in_splitk_stride`` = ``q_in.stride(0)``,
  ``NUM_SPLITK`` = ``num_splitk``.
"""

# Vendored from ROCm/aiter (aiter/ops/triton/_triton_kernels/fusions/fused_reduce_qk_norm_rope_swa_write.py, header above).
# Helpers: _get_neox_rotated_x and _get_gptj_rotated_x of aiter/ops/triton/_triton_kernels/rope/rope.py (SPDX MIT,
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc.) and make_kernel_repr of
# aiter/ops/triton/utils/_triton/kernel_repr.py (no header upstream; the repository MIT license applies).
# Launch code: _pick_block_size_m and fused_reduce_qk_norm_rope_swa_write of
# aiter/ops/triton/fusions/fused_reduce_qk_norm_rope_swa_write.py (same header), reduced to run().

import triton
import triton.language as tl


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
    # When name_key is set, the base name is taken from the matching constexpr
    # kwarg at call time (falling back to base_name if missing/empty). Lets a
    # single shared kernel produce caller-specific names in compiled artifacts.
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


@triton.jit
def _get_neox_rotated_x(
    x,
    x_rotated_mask,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
    IS_BWD: tl.constexpr = False,
):
    if IS_BWD:
        x_rotated = tl.where(x_rotated_mask, -x, x)
    else:
        x_rotated = tl.where(x_rotated_mask, x, -x)

    x_rotated = tl.reshape(x_rotated, (BLOCK_T, 2, BLOCK_D_HALF))
    x_rotated = tl.flip(x_rotated, 2)
    x_rotated = tl.reshape(
        x_rotated,
        (
            BLOCK_T,
            BLOCK_D,
        ),
    )
    x_rotated = tl.flip(x_rotated, 1)
    return x_rotated


@triton.jit
def _get_gptj_rotated_x(
    x,
    x_rotated_mask,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
    IS_BWD: tl.constexpr = False,
):
    if IS_BWD:
        x_rotated = tl.where(x_rotated_mask, -x, x)
    else:
        x_rotated = tl.where(x_rotated_mask, x, -x)

    x_rotated = tl.reshape(x_rotated, (BLOCK_T, BLOCK_D_HALF, 2))
    x_rotated = tl.flip(x_rotated, 2)
    x_rotated = tl.reshape(
        x_rotated,
        (
            BLOCK_T,
            BLOCK_D,
        ),
    )
    return x_rotated


@triton.jit
def _batched_rmsnorm_op(row, weight, n_cols, epsilon):
    """Per-row RMSNorm over the last axis of a [BLOCK_M, N] tile (row in fp32)."""
    row_norm = row * row
    row_norm = tl.sum(row_norm, axis=-1)
    norm_factor = tl.math.rsqrt((row_norm / n_cols) + epsilon)
    if weight is not None:
        rms_norm = row * norm_factor[:, None] * weight[None, :]
    else:
        rms_norm = row * norm_factor[:, None]
    return rms_norm


@triton.jit
def _batched_unit_rope(
    x_pe,
    cos,
    sin,
    d_pe_offs,
    IS_NEOX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D_pe: tl.constexpr,
    BLOCK_D_HALF_pe: tl.constexpr,
):
    """RoPE on a [BLOCK_M, BLOCK_D_pe] tile; cos/sin are [BLOCK_M, BLOCK_D_pe]."""
    if IS_NEOX:
        x_rotated_mask = (d_pe_offs < BLOCK_D_HALF_pe)[None, :]
        x_pe_rotated = _get_neox_rotated_x(
            x_pe, x_rotated_mask, BLOCK_M, BLOCK_D_pe, BLOCK_D_HALF_pe
        )
    else:
        x_rotated_mask = (d_pe_offs % 2 == 0)[None, :]
        x_pe_rotated = _get_gptj_rotated_x(
            x_pe, x_rotated_mask, BLOCK_M, BLOCK_D_pe, BLOCK_D_HALF_pe
        )

    return x_pe * cos + x_pe_rotated * sin


_fused_reduce_qk_norm_rope_swa_write_repr = make_kernel_repr(
    "_fused_reduce_qk_norm_rope_swa_write_kernel",
    [
        "BLOCK_SIZE_M",
        "HEAD_DIM",
        "ROPE_DIM",
        "NUM_LOCAL_HEADS",
        "NUM_SPLITK",
        "HAS_SWA",
        "IS_NEOX",
        "REUSE_FREQS_FRONT_PART",
    ],
)


@triton.jit(repr=_fused_reduce_qk_norm_rope_swa_write_repr)
def _fused_reduce_qk_norm_rope_swa_write_kernel(
    q_in_ptr,
    q_out_ptr,
    kv_ptr,
    q_norm_weight_ptr,
    kv_norm_weight_ptr,
    positions_ptr,
    cos_ptr,
    sin_ptr,
    swa_write_active_ptr,
    batch_id_per_token_ptr,
    state_slot_per_seq_ptr,
    swa_kv_ptr,
    M,
    q_in_splitk_stride,
    q_in_m_stride,
    q_in_d_stride,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kv_m,
    stride_kv_d,
    cos_stride_t,
    cos_stride_d,
    swa_kv_slot_stride,
    swa_kv_pos_stride,
    win,
    q_eps,
    kv_eps,
    BLOCK_SIZE_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    NUM_LOCAL_HEADS: tl.constexpr,
    NUM_SPLITK: tl.constexpr,
    HAS_SWA: tl.constexpr,
    IS_NEOX: tl.constexpr,
    REUSE_FREQS_FRONT_PART: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_h = tl.program_id(1).to(tl.int64)
    NOPE_DIM: tl.constexpr = HEAD_DIM - ROPE_DIM
    NUM_PE_CHUNKS: tl.constexpr = HEAD_DIM // ROPE_DIM

    m_offs = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    m_mask = m_offs < M

    offs_d_full = tl.arange(0, HEAD_DIM)
    nope_d_mask = offs_d_full < NOPE_DIM

    d_pe_offs = tl.arange(0, ROPE_DIM).to(tl.int64)
    if REUSE_FREQS_FRONT_PART:
        if IS_NEOX:
            d_cos_offs = d_pe_offs
            d_cos_offs = tl.where(
                (d_cos_offs >= (ROPE_DIM // 2)) & (d_cos_offs < ROPE_DIM),
                d_cos_offs - (ROPE_DIM // 2),
                d_cos_offs,
            ).to(d_cos_offs.dtype)
        else:
            d_cos_offs = d_pe_offs // 2
    else:
        d_cos_offs = d_pe_offs

    if pid_h < NUM_LOCAL_HEADS:
        head_id = pid_h.to(tl.int32)
        offs_n = head_id * HEAD_DIM + offs_d_full

        splitk_offs = tl.arange(0, NUM_SPLITK).to(tl.int64)
        q_ptrs = (
            q_in_ptr
            + splitk_offs[:, None, None] * q_in_splitk_stride
            + m_offs[None, :, None] * q_in_m_stride
            + offs_n[None, None, :] * q_in_d_stride
        )
        q_tile = tl.load(
            q_ptrs,
            mask=m_mask[None, :, None],
            other=0.0,
        ).to(
            tl.float32
        )  # [NUM_SPLITK, BLOCK_SIZE_M, HEAD_DIM]
        q_acc = tl.sum(q_tile, axis=0)  # [BLOCK_SIZE_M, HEAD_DIM]

        if q_norm_weight_ptr is not None:
            w_q = tl.load(q_norm_weight_ptr + offs_d_full).to(tl.float32)
        else:
            w_q = None
        q_out_normed = _batched_rmsnorm_op(q_acc, w_q, HEAD_DIM, q_eps)

        q_base_ptrs = q_out_ptr + m_offs[:, None] * stride_qm + pid_h * stride_qh
        tl.store(
            q_base_ptrs + offs_d_full[None, :] * stride_qd,
            q_out_normed.to(q_out_ptr.dtype.element_ty),
            mask=m_mask[:, None] & nope_d_mask[None, :],
        )

        # Slice the trailing ROPE_DIM elements: only the last chunk is nonzero.
        q_pe = tl.where(
            (offs_d_full >= NOPE_DIM)[None, :], q_out_normed, 0.0
        )  # [BLOCK_SIZE_M, HEAD_DIM]
        q_pe = q_pe.reshape(BLOCK_SIZE_M, NUM_PE_CHUNKS, ROPE_DIM)
        q_pe = tl.sum(q_pe, axis=1)  # [BLOCK_SIZE_M, ROPE_DIM]

        pos = tl.load(positions_ptr + m_offs, mask=m_mask, other=0)  # [BLOCK_SIZE_M]
        cos_offs = pos[:, None] * cos_stride_t + d_cos_offs[None, :] * cos_stride_d
        cos = tl.load(cos_ptr + cos_offs, mask=m_mask[:, None], other=0)
        sin = tl.load(sin_ptr + cos_offs, mask=m_mask[:, None], other=0)

        q_pe_ptrs = q_base_ptrs + (NOPE_DIM + d_pe_offs[None, :]) * stride_qd
        q_pe = _batched_unit_rope(
            q_pe,
            cos,
            sin,
            d_pe_offs,
            IS_NEOX,
            BLOCK_SIZE_M,
            ROPE_DIM,
            ROPE_DIM // 2,
        )
        tl.store(
            q_pe_ptrs,
            q_pe.to(q_out_ptr.dtype.element_ty),
            mask=m_mask[:, None],
        )
        return

    if HAS_SWA:
        src_id = tl.load(
            swa_write_active_ptr + m_offs, mask=m_mask, other=-1
        )  # [BLOCK_SIZE_M]
    else:
        src_id = m_offs.to(tl.int32)
    src_mask = m_mask & (src_id >= 0)

    pos = tl.load(positions_ptr + src_id, mask=src_mask, other=0)
    cos_offs = pos[:, None] * cos_stride_t + d_cos_offs[None, :] * cos_stride_d
    cos = tl.load(cos_ptr + cos_offs, mask=src_mask[:, None], other=0)
    sin = tl.load(sin_ptr + cos_offs, mask=src_mask[:, None], other=0)

    kv_base_ptrs = kv_ptr + src_id[:, None].to(tl.int64) * stride_kv_m
    kv_full_ptrs = kv_base_ptrs + offs_d_full[None, :] * stride_kv_d
    kv_pe_ptrs = kv_base_ptrs + (NOPE_DIM + d_pe_offs[None, :]) * stride_kv_d

    # Load the entire kv row (nope + pe) so we can RMSNorm over head_dim.
    kv_full = tl.load(kv_full_ptrs, mask=src_mask[:, None], other=0.0).to(tl.float32)

    if kv_norm_weight_ptr is not None:
        w_kv = tl.load(kv_norm_weight_ptr + offs_d_full).to(tl.float32)
    else:
        w_kv = None
    kv_normed = _batched_rmsnorm_op(
        kv_full, w_kv, HEAD_DIM, kv_eps
    )  # [BLOCK_SIZE_M, HEAD_DIM]

    # Store the normed nope portion back into kv.
    tl.store(
        kv_full_ptrs,
        kv_normed.to(kv_ptr.dtype.element_ty),
        mask=src_mask[:, None] & nope_d_mask[None, :],
    )

    # Extract pe via the same reshape+sum trick used for q.
    kv_pe = tl.where(
        (offs_d_full >= NOPE_DIM)[None, :], kv_normed, 0.0
    )  # [BLOCK_SIZE_M, HEAD_DIM]
    kv_pe = kv_pe.reshape(BLOCK_SIZE_M, NUM_PE_CHUNKS, ROPE_DIM)
    kv_pe = tl.sum(kv_pe, axis=1)  # [BLOCK_SIZE_M, ROPE_DIM]

    kv_pe = _batched_unit_rope(
        kv_pe,
        cos,
        sin,
        d_pe_offs,
        IS_NEOX,
        BLOCK_SIZE_M,
        ROPE_DIM,
        ROPE_DIM // 2,
    )
    tl.store(
        kv_pe_ptrs,
        kv_pe.to(kv_ptr.dtype.element_ty),
        mask=src_mask[:, None],
    )

    if HAS_SWA:
        bid = tl.load(batch_id_per_token_ptr + src_id, mask=src_mask, other=0)
        slot = tl.load(state_slot_per_seq_ptr + bid, mask=src_mask, other=0)
        ring_idx = pos % win
        swa_kv_ptrs = (
            swa_kv_ptr
            + slot[:, None].to(tl.int64) * swa_kv_slot_stride
            + ring_idx[:, None].to(tl.int64) * swa_kv_pos_stride
        )
        tl.store(
            swa_kv_ptrs + offs_d_full[None, :],
            kv_normed.to(swa_kv_ptr.dtype.element_ty),
            mask=src_mask[:, None] & nope_d_mask[None, :],
        )
        tl.store(
            swa_kv_ptrs + NOPE_DIM + d_pe_offs[None, :],
            kv_pe.to(swa_kv_ptr.dtype.element_ty),
            mask=src_mask[:, None],
        )


def _pick_block_size_m(M: int, num_local_heads: int, num_splitk: int) -> int:
    """Pick BLOCK_SIZE_M for the fused reduce/norm/rope kernel.

    Each program loads ``[NUM_SPLITK, BLOCK_SIZE_M, HEAD_DIM]`` of fp32 q-tile,
    so register pressure scales with ``num_splitk * BLOCK_SIZE_M``. Wide head
    grids (large ``num_local_heads``) amortize per-program constant work
    (cos/sin offset prep, etc.) better with larger BM; small head grids need
    more M-tiles for occupancy, which favors smaller BM.
    """
    # Register-pressure cap from the splitk fp32 q-tile.
    if num_splitk >= 4:
        cap = 4
    elif num_splitk >= 2:
        cap = 8
    else:
        cap = 16

    if num_local_heads >= 64:
        target = 16
    elif num_local_heads >= 16:
        target = 8
    else:
        target = 4

    bm = min(target, cap)

    # Shrink to a power-of-two not exceeding M so tiny inputs don't pay tail
    # masking overhead for the whole tile.
    while bm > 1 and bm > M:
        bm //= 2

    num_warps = 4
    waves_per_eu = 1

    return bm, num_warps, waves_per_eu


def run(*, q, kv, kv_norm_weight, positions, cos, sin, write_indices, batch_id_per_token, state_slot_mapping,
        swa_kv, q_out, q_eps, kv_eps) -> None:
    """Split-K reduce + weight-free per-head Q RMSNorm + RoPE tail on q; weighted RMSNorm + RoPE tail on kv (in
    place) and the sliding-window write of the normalized, rotated rows into swa_kv[slot, position % win].

    q [num_splitk, M, heads * head_dim]; kv [M, head_dim]; cos/sin [max_position, rope_dim // 2];
    swa_kv [num_slots, win, head_dim]; interleaved (GPT-J) RoPE on the last rope_dim = 2 * cos.shape[-1] values.
    """
    num_splitk, M, N = q.shape
    head_dim = kv.shape[1]
    rope_head_dim = 2 * cos.shape[-1]
    num_local_heads = N // head_dim
    win = swa_kv.shape[1]

    BLOCK_SIZE_M, num_warps, waves_per_eu = _pick_block_size_m(M, num_local_heads, num_splitk)
    grid = (triton.cdiv(M, BLOCK_SIZE_M), num_local_heads + 1)
    _fused_reduce_qk_norm_rope_swa_write_kernel[grid](
        q,
        q_out,
        kv,
        None,
        kv_norm_weight,
        positions,
        cos,
        sin,
        write_indices,
        batch_id_per_token,
        state_slot_mapping,
        swa_kv,
        M,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q_out.stride(0),
        q_out.stride(1),
        q_out.stride(2),
        kv.stride(0),
        kv.stride(1),
        cos.stride(0),
        cos.stride(-1),
        swa_kv.stride(0),
        swa_kv.stride(1),
        win,
        q_eps,
        kv_eps,
        HEAD_DIM=head_dim,
        ROPE_DIM=rope_head_dim,
        NUM_LOCAL_HEADS=num_local_heads,
        NUM_SPLITK=num_splitk,
        HAS_SWA=True,
        IS_NEOX=False,
        REUSE_FREQS_FRONT_PART=True,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )
