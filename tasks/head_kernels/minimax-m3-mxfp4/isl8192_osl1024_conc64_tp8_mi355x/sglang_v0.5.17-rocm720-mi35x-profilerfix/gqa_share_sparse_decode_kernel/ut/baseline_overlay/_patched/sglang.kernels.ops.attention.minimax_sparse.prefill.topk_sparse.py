# Copyright 2025 XunhaoLai. All rights reserved.

from typing import Optional

import torch
import triton
import triton.language as tl

from ..common.utils import (
    check_sparse_kv_fp8,
    get_cu_seqblocks,
    robust_allocator,
    sparse_out_dtype,
    unit_scale,
)


@triton.heuristics(
    {
        "BLOCK_SIZE_KD": lambda args: triton.next_power_of_2(args["qk_head_dim"]),
        "BLOCK_SIZE_VD": lambda args: triton.next_power_of_2(args["v_head_dim"]),
        "BLOCK_SIZE_H": lambda args: triton.next_power_of_2(
            max(
                16 // args["BLOCK_SIZE_Q"],
                triton.next_power_of_2(args["gqa_group_size"]),
            )
        ),
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["max_topk"]),
        "BLOCK_SIZE_QH": lambda args: args["BLOCK_SIZE_Q"] * args["BLOCK_SIZE_H"],
        "HAS_SINK": lambda args: args["sink_ptr"] is not None,
    }
)
@triton.autotune(
    # Configs that fail to compile on the target arch are skipped, so widening
    # the num_warps x num_stages grid only adds candidates, never a bad kernel.
    configs=[
        triton.Config({}, num_warps=nw, num_stages=ns)
        for nw in (2, 4, 8)
        for ns in (2, 3, 4)
    ],
    key=[
        "BLOCK_SIZE_Q",
        "BLOCK_SIZE_K",
        "qk_head_dim",
        "v_head_dim",
        "gqa_group_size",
    ],
)
@triton.jit
def _gqa_share_sparse_fwd_kernel(
    q_ptr,  # Q: n x h x d
    k_cache_ptr,  # K paged: max_slots x kh x d
    v_cache_ptr,  # V paged: max_slots x kh x d
    sink_ptr,  # Sink: h x d
    t_ptr,  # topk_idx: kh x n x k
    o_ptr,  # O: n x h x d
    req_to_token_ptr,  # req_to_token: max_reqs x max_kv_len
    # seqlens
    cu_seqlens_q,
    cu_seqblocks_q,
    seq_lens,
    prefix_lens,
    slot_ids,
    # shape
    max_slots,
    num_kv_heads,
    gqa_group_size,
    qk_head_dim,
    v_head_dim,
    max_topk,
    # q loop num
    num_q_loop,
    # sm_scale
    sm_scale,
    # per-tensor KV dequant scales (1.0 when the cache is unit-scaled)
    k_scale,
    v_scale,
    # stride
    stride_qn,
    stride_qh,
    stride_qd,
    stride_ks,
    stride_kh,
    stride_kd,
    stride_vs,
    stride_vh,
    stride_vd,
    stride_sh,
    stride_sd,
    stride_th,
    stride_tn,
    stride_tk,
    stride_on,
    stride_oh,
    stride_od,
    stride_r2t_b,
    # META parameters
    BLOCK_SIZE_Q: tl.constexpr,  # q block size
    BLOCK_SIZE_K: tl.constexpr,  # k block size
    BLOCK_SIZE_KD: tl.constexpr,
    BLOCK_SIZE_VD: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_QH: tl.constexpr,
    # has sink
    HAS_SINK: tl.constexpr,
    USE_TMA: tl.constexpr,
    IS_FP8: tl.constexpr,
):
    sm_scale_log2e = sm_scale * 1.4426950409
    # get batch id and head id
    pid_q = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_b = tl.program_id(2)
    pid_h = pid_kh * gqa_group_size
    # get q k start and len after rmpad
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    q_block_start = tl.load(cu_seqblocks_q + pid_b)
    q_block_len = tl.load(cu_seqblocks_q + pid_b + 1) - q_block_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    sid = (
        tl.load(slot_ids + pid_b).to(tl.int64) + max_slots
    ) % max_slots  # safety against negative
    if pid_q * num_q_loop >= q_block_len:
        return
    real_q_loop = min(num_q_loop, q_block_len - pid_q * num_q_loop)
    if HAS_SINK:
        sink_ptrs = tl.make_block_ptr(
            base=sink_ptr + pid_h * stride_sh,
            shape=(gqa_group_size, qk_head_dim),
            strides=(stride_sh, stride_sd),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_KD),
            order=(1, 0),
        )
        sink = tl.load(sink_ptrs, boundary_check=(0, 1), padding_option="zero").to(
            tl.float32
        )
    # offsets for paged K/V load
    off_n = tl.arange(0, BLOCK_SIZE_K)
    off_kd = tl.arange(0, BLOCK_SIZE_KD)
    off_vd = tl.arange(0, BLOCK_SIZE_VD)
    kd_mask = off_kd < qk_head_dim
    vd_mask = off_vd < v_head_dim
    for j in range(real_q_loop):
        pid_q_j = pid_q * num_q_loop + j
        # init topk idx pointer
        t_ptr_j = t_ptr + (q_block_start + pid_q_j) * stride_tn + pid_kh * stride_th
        # we assume that the topk_idx is right padded with -1
        off_t = tl.arange(0, BLOCK_SIZE_T)
        topk_idx = tl.load(t_ptr_j + off_t * stride_tk, mask=off_t < max_topk, other=-1)
        valid_idx = tl.where(topk_idx >= 0, off_t, -1)
        real_topk = tl.sum(valid_idx != -1, axis=0)
        # init qkv pointer
        q_ptrs = tl.make_block_ptr(
            base=q_ptr + q_start * stride_qn + pid_h * stride_qh,
            shape=(q_len, gqa_group_size, qk_head_dim),
            strides=(stride_qn, stride_qh, stride_qd),
            offsets=(pid_q_j * BLOCK_SIZE_Q, 0, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_KD),
            order=(2, 1, 0),
        )
        # load q, shape: [BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D] -> [BLOCK_SIZE_QH, BLOCK_SIZE_D]
        q = tl.load(q_ptrs, boundary_check=(0, 1, 2), padding_option="zero")
        # init statistics
        off_q_k = (
            tl.arange(0, BLOCK_SIZE_Q)[:, None]
            + pid_q_j * BLOCK_SIZE_Q
            + prefix_len
            - tl.arange(0, BLOCK_SIZE_K)[None, :]
        )
        if HAS_SINK:
            m_i = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_H), dtype=tl.float32)
            lse_i = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_H), dtype=tl.float32)
            qsink = (
                tl.sum(q.to(tl.float32) * sink[None, :, :], axis=2) * sm_scale_log2e
            )  # (BLOCK_SIZE_Q, BLOCK_SIZE_H)
            m_i += qsink
            lse_i += qsink
            m_i = tl.reshape(m_i, BLOCK_SIZE_QH)
            lse_i = tl.reshape(lse_i, BLOCK_SIZE_QH)
        else:
            m_i = tl.full((BLOCK_SIZE_QH,), float("-inf"), dtype=tl.float32)
            lse_i = tl.full((BLOCK_SIZE_QH,), float("-inf"), dtype=tl.float32)
        acc_o = tl.full((BLOCK_SIZE_QH, BLOCK_SIZE_VD), 0, dtype=tl.float32)
        q = tl.reshape(q, BLOCK_SIZE_QH, BLOCK_SIZE_KD)
        # sparse attention
        for i in range(real_topk):
            # get current block start index (absolute K position)
            c = tl.load(t_ptr_j).to(tl.int32) * BLOCK_SIZE_K
            t_ptr_j = t_ptr_j + stride_tk
            # paged load K via req_to_token: pos -> slot -> k_cache
            pos = c + off_n
            pos_mask = pos < seq_len
            slots = tl.load(
                req_to_token_ptr + sid * stride_r2t_b + pos,
                mask=pos_mask,
                other=0,
            ).to(tl.int64)
            slots = (slots + max_slots) % max_slots  # safety against negative
            # k shape: [BLOCK_SIZE_KD, BLOCK_SIZE_K] (transposed for tl.dot)
            k = tl.load(
                k_cache_ptr
                + slots[None, :] * stride_ks
                + pid_kh * stride_kh
                + off_kd[:, None] * stride_kd,
                mask=kd_mask[:, None] & pos_mask[None, :],
                other=0.0,
            )
            if IS_FP8:
                # fp8 main K cache: widening cast with bf16/fp16 Q (unit-scaled
                # cache -> exact inverse dequant; k_scale covers calibrated
                # caches), no-op with fp8 Q (fp8 attn-GEMM mode) so tl.dot runs
                # fp8x8. Compiled out when the cache is bf16.
                k = k.to(q.dtype)
            # compute qk
            qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32)
            # causal mask
            qk += tl.where(off_q_k[:, None, :] >= c, 0, float("-inf"))
            qk = tl.reshape(qk, BLOCK_SIZE_QH, BLOCK_SIZE_K)
            # [BLOCK_SIZE_QH, qk_head_dim] @ [qk_head_dim, BLOCK_SIZE_K]
            #   -> [BLOCK_SIZE_QH, BLOCK_SIZE_K]
            qk += tl.dot(q, k) * (sm_scale_log2e * k_scale)
            # K boundary mask: positions beyond seq_len contribute -inf
            qk += tl.where(pos_mask[None, :], 0, float("-inf"))
            # compute m_ij and l_ij
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)
            # scale acc_o
            acc_o_scale = tl.exp2(m_i - m_ij)
            acc_o = acc_o * acc_o_scale[:, None]
            # paged load V
            v = tl.load(
                v_cache_ptr
                + slots[:, None] * stride_vs
                + pid_kh * stride_vh
                + off_vd[None, :] * stride_vd,
                mask=pos_mask[:, None] & vd_mask[None, :],
                other=0.0,
            )
            if IS_FP8:
                # Cast V to the compute dtype: widening with bf16/fp16 Q (so
                # `p.to(v.dtype)` keeps P in the compute dtype), no-op with fp8
                # Q where P is quantized to e4m3 for the fp8 PV MMA — the same
                # accuracy contract as fmha_sm100's fp8 kernel.
                v = v.to(q.dtype)
            p = p.to(v.dtype)
            acc_o += tl.dot(p, v) * v_scale
            # update statistics
            m_i = m_ij
            lse_i = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)
        # final scale
        acc_o = acc_o * tl.exp2(m_i - lse_i)[:, None]
        # save output
        acc_o = tl.reshape(acc_o, BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_VD)
        o_ptrs = tl.make_block_ptr(
            base=o_ptr + q_start * stride_on + pid_h * stride_oh,
            shape=(q_len, gqa_group_size, v_head_dim),
            strides=(stride_on, stride_oh, stride_od),
            offsets=(pid_q_j * BLOCK_SIZE_Q, 0, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_VD),
            order=(2, 1, 0),
        )
        tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1, 2))


# ---------------------------------------------------------------------------
# Q-GROUP UNION KERNEL (fast path for block_size_q == 1, no sink)
#
# One program owns GROUP_Q consecutive query tokens of ONE sequence and walks
# the UNION of their selected KV blocks in ascending order, loading each K/V
# block exactly ONCE for the whole group and masking per row.  Consecutive
# query tokens pick nearly the same blocks, so the union is only marginally
# bigger than a single row's top-k -> the (token, block) KV load count drops by
# ~7x at GROUP_Q=8 / ~13x at GROUP_Q=16 while the extra (masked-out) MFMA work
# grows only ~1.2x.  It also fills the MFMA M tile with real rows instead of
# padding (GROUP_Q * next_pow2(gqa_group_size) rows).
#
# The union is enumerated without any sort/compaction/prepass: the whole
# [GROUP_Q, BLOCK_SIZE_T] top-k tile is resident in registers and each step
# takes the smallest still-unvisited block id (a 256-element int reduction,
# free next to a pair of 128x128x128 dots).  The launch grid stays a pure
# function of host ints, so the CUDA-graph replay gate is unaffected.
# ---------------------------------------------------------------------------
_UNION_BIG = 0x3FFFFFFF


@triton.jit
def _union_slots(
    r2t_row,
    blk,
    seq_len,
    max_slots,
    off_n,
    BLOCK_SIZE_K: tl.constexpr,
):
    BIG: tl.constexpr = 0x3FFFFFFF
    blk_s = tl.where(blk < BIG, blk, 0)
    pos = blk_s * BLOCK_SIZE_K + off_n
    slots = tl.load(r2t_row + pos, mask=(pos < seq_len) & (blk < BIG), other=0)
    slots = tl.where(slots < 0, slots + max_slots, slots)
    return pos, slots


@triton.jit
def _union_kv(
    slots,
    k_head_base,
    v_head_base,
    kd_off,
    vd_off,
    kd_mask,
    vd_mask,
    stride_ks,
    stride_vs,
    KD_FULL: tl.constexpr,
    VD_FULL: tl.constexpr,
    SLOT_I32: tl.constexpr,
):
    # Scale the slot index BEFORE the broadcast: the row-base multiply then runs
    # on BLOCK_N values instead of BLOCK_N*BLOCK_SIZE_D.
    if SLOT_I32:
        ks_row = slots * stride_ks
        vs_row = slots * stride_vs
    else:
        s64 = slots.to(tl.int64)
        ks_row = s64 * stride_ks
        vs_row = s64 * stride_vs
    k_off = ks_row[None, :] + kd_off
    v_off = vs_row[:, None] + vd_off
    if KD_FULL:
        k = tl.load(k_head_base + k_off)
    else:
        k = tl.load(k_head_base + k_off, mask=kd_mask[:, None], other=0.0)
    if VD_FULL:
        v = tl.load(v_head_base + v_off)
    else:
        v = tl.load(v_head_base + v_off, mask=vd_mask[None, :], other=0.0)
    return k, v


@triton.jit
def _union_step(
    q,
    k_cur,
    v_cur,
    sel_m,
    q_abs,
    cur_pos,
    m_i,
    l_i,
    acc_o,
    qk_scale,
    IS_FP8: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    if IS_FP8:
        k_cur = k_cur.to(q.dtype)
    # RAW logits: the softmax scale is folded into the exp2 argument as a single
    # FMA (qk * qk_scale - m) instead of a separate [M, N] multiply.
    qk = tl.dot(q, k_cur)
    if CAUSAL:
        # `q_abs >= cur_pos` already implies `cur_pos < seq_len` (q_abs is a
        # valid position of this sequence), so the K boundary mask is redundant.
        qk = tl.where(q_abs[:, None] >= cur_pos[None, :], qk, float("-inf"))
    # The row-selection predicate is NOT ANDed into a [M, N] mask (that AND cost
    # 64 VALU/lane/step); it becomes a per-ROW +inf bias on the exp2 pivot, which
    # is exactly equivalent: an unselected row gets qk - inf == -inf -> p == 0.
    # Scaling is monotone (qk_scale > 0) so max-then-scale == scale-then-max.
    row_max = tl.where(sel_m, tl.max(qk, axis=1) * qk_scale, float("-inf"))
    m_ij = tl.maximum(m_i, row_max)
    # a row that has contributed nothing yet AND skips this block keeps
    # m == -inf; substitute a finite pivot so exp2 stays 0 instead of NaN.
    m_pv = tl.where(m_ij == float("-inf"), 0.0, m_ij)
    m_sub = tl.where(sel_m, m_pv, float("inf"))
    p = tl.exp2(qk * qk_scale - m_sub[:, None])
    l_ij = tl.sum(p, axis=1)
    acc_o_scale = tl.exp2(m_i - m_pv)
    acc_o = acc_o * acc_o_scale[:, None]
    if IS_FP8:
        v_cur = v_cur.to(q.dtype)
    # v_scale is a loop-invariant scalar and the accumulator recurrence is linear
    # in it, so it is hoisted out to the single final rescale.
    # 3-operand dot: the MFMA accumulates straight into acc_o, which removes a
    # whole [M, VD] fp32 add (64 VALU/lane/step) and the accumulator copy.
    acc_o = tl.dot(p.to(v_cur.dtype), v_cur, acc_o)
    # running LINEAR denominator: one FMA instead of exp2 + log2 + add
    l_i = l_i * acc_o_scale + l_ij
    return m_ij, l_i, acc_o


@triton.heuristics(
    {
        "BLOCK_SIZE_KD": lambda args: triton.next_power_of_2(args["qk_head_dim"]),
        "BLOCK_SIZE_VD": lambda args: triton.next_power_of_2(args["v_head_dim"]),
        "BLOCK_SIZE_H": lambda args: triton.next_power_of_2(args["gqa_group_size"]),
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["max_topk"]),
        "BLOCK_SIZE_M": lambda args: args["GROUP_Q"]
        * triton.next_power_of_2(args["gqa_group_size"]),
        # head dims are exact powers of two -> the head-dim masks are all-true
        # and the per-gather 2-D mask AND compiles away entirely.
        # Column-split the paged K block so the live [M, N] logit tile, p tile and
        # K/V tiles halve -- the whole point is to get the VGPR budget under the
        # 2-waves-per-SIMD cliff (512 regs / wave-pair) instead of 1.
        "NSPLIT": lambda args: max(1, args["BLOCK_SIZE_K"] // 64),
        "KD_FULL": lambda args: triton.next_power_of_2(args["qk_head_dim"])
        == args["qk_head_dim"],
        "VD_FULL": lambda args: triton.next_power_of_2(args["v_head_dim"])
        == args["v_head_dim"],
    }
)
@triton.autotune(
    configs=[
        triton.Config({"waves_per_eu": wpe}, num_warps=nw, num_stages=ns)
        for nw in (4, 8)
        for ns in (1, 2, 3, 4)
        for wpe in (0, 1, 2)
    ],
    key=[
        "GROUP_Q",
        "BLOCK_SIZE_K",
        "qk_head_dim",
        "v_head_dim",
        "gqa_group_size",
        "max_topk",
    ],
)
@triton.jit
def _gqa_share_sparse_fwd_group_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    t_ptr,
    o_ptr,
    req_to_token_ptr,
    # seqlens
    cu_seqlens_q,
    cu_seqblocks_q,
    seq_lens,
    prefix_lens,
    slot_ids,
    # shape
    max_slots,
    num_kv_heads,
    gqa_group_size,
    qk_head_dim,
    v_head_dim,
    max_topk,
    # sm_scale
    sm_scale,
    k_scale,
    v_scale,
    # stride
    stride_qn,
    stride_qh,
    stride_qd,
    stride_ks,
    stride_kh,
    stride_kd,
    stride_vs,
    stride_vh,
    stride_vd,
    stride_th,
    stride_tn,
    stride_tk,
    stride_on,
    stride_oh,
    stride_od,
    stride_r2t_b,
    # META
    GROUP_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_KD: tl.constexpr,
    BLOCK_SIZE_VD: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    KD_FULL: tl.constexpr,
    VD_FULL: tl.constexpr,
    SLOT_I32: tl.constexpr,
    IS_FP8: tl.constexpr,
    NSPLIT: tl.constexpr,
):
    BIG: tl.constexpr = 0x3FFFFFFF
    BLOCK_N: tl.constexpr = BLOCK_SIZE_K // NSPLIT
    sm_scale_log2e = sm_scale * 1.4426950409
    # Longest-processing-time-first: with a causal prefix the union grows with
    # the token index, so the LAST groups are the heaviest.  Dispatch order on
    # AMD follows pid, so walk the groups backwards and the heavy work lands in
    # the first dispatch wave, leaving the cheap tail to fill the ragged end.
    pid_g = tl.num_programs(0) - 1 - tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_b = tl.program_id(2)
    pid_h = pid_kh * gqa_group_size

    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    q_block_start = tl.load(cu_seqblocks_q + pid_b)
    q_block_len = tl.load(cu_seqblocks_q + pid_b + 1) - q_block_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    # branchless equivalent of ((sid + max_slots) % max_slots) for sid in
    # [-max_slots, max_slots): no 64-bit remainder, and it is a scalar.
    sid = tl.load(slot_ids + pid_b)
    sid = tl.where(sid < 0, sid + max_slots, sid)
    r2t_row = req_to_token_ptr + sid.to(tl.int64) * stride_r2t_b

    tok0 = pid_g * GROUP_Q
    if tok0 >= q_block_len:
        return

    off_g = tl.arange(0, GROUP_Q)
    off_t = tl.arange(0, BLOCK_SIZE_T)
    off_n = tl.arange(0, BLOCK_N)
    off_kd = tl.arange(0, BLOCK_SIZE_KD)
    off_vd = tl.arange(0, BLOCK_SIZE_VD)
    kd_mask = off_kd < qk_head_dim
    vd_mask = off_vd < v_head_dim
    # loop-invariant address pieces, computed ONCE per program: the head-dim
    # term factors out of the paged gather, so only the slot term varies.
    k_head_base = k_cache_ptr + pid_kh * stride_kh
    v_head_base = v_cache_ptr + pid_kh * stride_vh
    kd_off = off_kd[:, None] * stride_kd
    vd_off = off_vd[None, :] * stride_vd
    qk_scale = sm_scale_log2e * k_scale

    tok = tok0 + off_g
    tok_ok = tok < q_block_len

    # resident top-k tile for the whole group: [GROUP_Q, BLOCK_SIZE_T]
    tt = tl.load(
        t_ptr
        + pid_kh * stride_th
        + (q_block_start + tok)[:, None] * stride_tn
        + off_t[None, :] * stride_tk,
        mask=tok_ok[:, None] & (off_t < max_topk)[None, :],
        other=-1,
    ).to(tl.int32)
    tt = tl.where(tt >= 0, tt, BIG)

    # Q for the whole group: [GROUP_Q, BLOCK_SIZE_H, D] -> [BLOCK_SIZE_M, D]
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + q_start * stride_qn + pid_h * stride_qh,
        shape=(q_len, gqa_group_size, qk_head_dim),
        strides=(stride_qn, stride_qh, stride_qd),
        offsets=(tok0, 0, 0),
        block_shape=(GROUP_Q, BLOCK_SIZE_H, BLOCK_SIZE_KD),
        order=(2, 1, 0),
    )
    q = tl.load(q_ptrs, boundary_check=(0, 1, 2), padding_option="zero")
    q = tl.reshape(q, BLOCK_SIZE_M, BLOCK_SIZE_KD)

    # per-row absolute query position (for the causal mask)
    q_abs = tl.reshape(
        tl.broadcast_to((tok + prefix_len)[:, None], (GROUP_Q, BLOCK_SIZE_H)),
        (BLOCK_SIZE_M,),
    )

    m_i = tl.full((BLOCK_SIZE_M,), float("-inf"), dtype=tl.float32)
    # lse_i == -inf initially  =>  linear denominator l_i == 0
    l_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_VD), dtype=tl.float32)

    NT: tl.constexpr = GROUP_Q * BLOCK_SIZE_T
    ttf = tl.reshape(tt, NT)

    # Second, ROW-REPLICATED copy of the top-k tile, [BLOCK_SIZE_M, BLOCK_SIZE_T].
    # `sel_m` has to live in the same [BLOCK_SIZE_M] row layout as m_i / l_i.
    # Deriving it from the [GROUP_Q, BLOCK_SIZE_T] tile forces a convert_layout
    # (LDS round trip + barrier) on EVERY union step; loading the replicated tile
    # once costs 8 VGPRs and makes the per-step reduction a pure in-register
    # axis-1 reduce in the right layout.
    tokm = tl.reshape(
        tl.broadcast_to(tok[:, None], (GROUP_Q, BLOCK_SIZE_H)), (BLOCK_SIZE_M,)
    )
    tokm_ok = tl.reshape(
        tl.broadcast_to(tok_ok[:, None], (GROUP_Q, BLOCK_SIZE_H)), (BLOCK_SIZE_M,)
    )
    ttm = tl.load(
        t_ptr
        + pid_kh * stride_th
        + (q_block_start + tokm)[:, None] * stride_tn
        + off_t[None, :] * stride_tk,
        mask=tokm_ok[:, None] & (off_t < max_topk)[None, :],
        other=-1,
    ).to(tl.int32)
    ttm = tl.where(ttm >= 0, ttm, BIG)

    # ---- manual 2-stage software pipeline over the union ------------------
    # The serial chain per union step is  min-reduce -> blk -> req_to_token
    # gather -> slot -> K/V gather -> tl.dot.  Issuing the NEXT step's index
    # chain and K/V loads before the CURRENT step's MFMA lets the gathers
    # overlap the dots instead of stalling behind them.
    # Number of union blocks that lie STRICTLY below the causal diagonal for
    # EVERY row of this group: block b is fully visible iff
    #   b * BLOCK_SIZE_K + (BLOCK_SIZE_K - 1) <= min_row(q_abs)   <=>   b < safe_cnt.
    # Because the union is walked ascending, all such blocks come first, so the
    # loop can be SPLIT: a mask-free prologue loop and a short causal epilogue.
    # This is a loop BOUND (scalar, amortised over the whole phase), not a
    # per-iteration branch -- the [M, N] compare + select simply does not exist
    # in the hot phase.
    safe_cnt = (tok0 + prefix_len + 1) // BLOCK_SIZE_K

    blk = tl.min(ttf)

    while blk < safe_cnt:
        cur = blk
        sel_m = tl.max(tl.where(ttm == cur, 1, 0), axis=1) > 0
        blk = tl.min(tl.where(ttf > cur, ttf, BIG))
        for h in tl.static_range(NSPLIT):
            pos_h, slots_h = _union_slots(
                r2t_row, cur, seq_len, max_slots, h * BLOCK_N + off_n, BLOCK_SIZE_K
            )
            k_h, v_h = _union_kv(
                slots_h, k_head_base, v_head_base, kd_off, vd_off, kd_mask,
                vd_mask, stride_ks, stride_vs, KD_FULL, VD_FULL, SLOT_I32,
            )
            m_i, l_i, acc_o = _union_step(
                q, k_h, v_h, sel_m, q_abs, pos_h, m_i, l_i, acc_o, qk_scale,
                IS_FP8, False,
            )

    while blk < BIG:
        cur = blk
        sel_m = tl.max(tl.where(ttm == cur, 1, 0), axis=1) > 0
        blk = tl.min(tl.where(ttf > cur, ttf, BIG))
        for h in tl.static_range(NSPLIT):
            pos_h, slots_h = _union_slots(
                r2t_row, cur, seq_len, max_slots, h * BLOCK_N + off_n, BLOCK_SIZE_K
            )
            k_h, v_h = _union_kv(
                slots_h, k_head_base, v_head_base, kd_off, vd_off, kd_mask,
                vd_mask, stride_ks, stride_vs, KD_FULL, VD_FULL, SLOT_I32,
            )
            m_i, l_i, acc_o = _union_step(
                q, k_h, v_h, sel_m, q_abs, pos_h, m_i, l_i, acc_o, qk_scale,
                IS_FP8, True,
            )

    # final scale: exp2(m_i - lse_i) == 1 / l_i  (l_i > 0 for every stored row)
    acc_o = acc_o * (v_scale / tl.where(l_i > 0.0, l_i, 1.0))[:, None]
    acc_o = tl.reshape(acc_o, GROUP_Q, BLOCK_SIZE_H, BLOCK_SIZE_VD)
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + q_start * stride_on + pid_h * stride_oh,
        shape=(q_len, gqa_group_size, v_head_dim),
        strides=(stride_on, stride_oh, stride_od),
        offsets=(tok0, 0, 0),
        block_shape=(GROUP_Q, BLOCK_SIZE_H, BLOCK_SIZE_VD),
        order=(2, 1, 0),
    )
    tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1, 2))


def _pick_group_q(gqa_group_size: int, topk: int) -> int:
    """Heuristic GROUP_Q: fill a 128-row MFMA M tile with real query rows.

    Measured on the served geometry (gqa_group_size=8, topk=16, block_size_k=128):
    M=64 -> 2.39x, M=128 -> 4.27x, M=256 -> 2.68x.  M=128 is the sweet spot
    between union-sharing (fewer K/V loads as GROUP_Q grows) and register
    pressure / occupancy (the [M, VD] fp32 accumulator plus the prefetched K/V
    tiles).  A pure host-side function of host ints, so the launch grid stays
    CUDA-graph-capture safe.
    """
    h = triton.next_power_of_2(gqa_group_size)
    return max(1, 128 // h)


@torch.no_grad()
def flash_prefill_with_gqa_share_sparse(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    sink: Optional[torch.Tensor],
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    topk_idx: torch.Tensor,
    block_size_q: int,
    block_size_k: int,
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_seqlen_q: int,
    sm_scale: Optional[float] = None,
    use_tma: bool = True,
    cu_seqblocks_q: Optional[torch.Tensor] = None,
    max_seqblock_q: Optional[int] = None,
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
) -> torch.Tensor:
    triton.set_allocator(robust_allocator)
    is_fp8 = check_sparse_kv_fp8(q, k_cache, v_cache, label="prefill")
    k_scale = unit_scale(k_scale)
    v_scale = unit_scale(v_scale)
    assert block_size_q in {1, 2, 4, 8, 16, 32, 64}
    assert block_size_k in {16, 32, 64, 128}
    # shape
    total_q, num_q_heads, qk_head_dim = q.shape
    max_slots, num_k_heads, _ = k_cache.shape
    _, num_v_heads, v_head_dim = v_cache.shape
    batch_size = cu_seqlens.shape[0] - 1
    topk = topk_idx.shape[-1]
    assert topk_idx.shape[0] == num_k_heads
    # gqa
    assert num_k_heads == num_v_heads
    assert num_q_heads % num_k_heads == 0
    gqa_group_size = num_q_heads // num_k_heads
    assert gqa_group_size * block_size_q <= 128
    if sm_scale is None:
        sm_scale = qk_head_dim**-0.5
    # q_scale multiplies every Q-side logit (QK dot and sink), so it folds into
    # sm_scale; k_scale must not touch the sink term and stays a kernel arg.
    sm_scale = sm_scale * unit_scale(q_scale)
    if cu_seqblocks_q is None or max_seqblock_q is None:
        cu_seqblocks_q, max_seqblock_q, _, _, _, _ = get_cu_seqblocks(
            cu_seqlens, max_seqlen_q, block_size_q, block_size_k
        )
    # output tensor
    o = torch.empty(
        total_q, num_q_heads, v_head_dim, device=q.device, dtype=sparse_out_dtype(q)
    )
    # ---- Q-group union fast path -----------------------------------------
    # Requires the per-token top-k layout (block_size_q == 1) and no sink.
    # Everything else (fp8 KV, ragged multi-seq batches, num_kv_heads > 1,
    # block_size_k in {16,32,64,128}) is handled.  Grid is a pure function of
    # host ints, so the CUDA-graph replay gate is unaffected.
    if block_size_q == 1 and sink is None and topk >= 1:
        GROUP_Q = _pick_group_q(gqa_group_size, topk)
        # int32 paged-gather addressing is safe iff the largest element offset
        # reachable through a slot index fits in a signed 32-bit int; otherwise
        # the kernel takes the widening (int64) path.  A pure host-int
        # predicate, so the launch grid / arg identity stays graph-safe.
        slot_i32 = bool(
            max_slots * max(int(k_cache.stride(0)), int(v_cache.stride(0)))
            < (2**31 - 2**20)
        )
        grid_g = (
            triton.cdiv(max_seqlen_q, GROUP_Q),
            num_k_heads,
            batch_size,
        )
        _gqa_share_sparse_fwd_group_kernel[grid_g](
            q,
            k_cache,
            v_cache,
            topk_idx,
            o,
            req_to_token,
            cu_seqlens,
            cu_seqblocks_q,
            seq_lens,
            prefix_lens,
            slot_ids,
            max_slots,
            num_k_heads,
            gqa_group_size,
            qk_head_dim,
            v_head_dim,
            topk,
            sm_scale,
            k_scale,
            v_scale,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            topk_idx.stride(0),
            topk_idx.stride(1),
            topk_idx.stride(2),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            req_to_token.stride(0),
            GROUP_Q=GROUP_Q,
            BLOCK_SIZE_K=triton.next_power_of_2(block_size_k),
            SLOT_I32=slot_i32,
            IS_FP8=is_fp8,
        )
        return o

    # launch kernel
    num_q_loop = (
        max_seqblock_q // 131072 + 1
    )  # calculate multiple queries in one kernel if seqlence length is too long
    BLOCK_SIZE_Q = triton.next_power_of_2(block_size_q)
    BLOCK_SIZE_K = triton.next_power_of_2(block_size_k)
    grid = (
        triton.cdiv(triton.cdiv(max_seqlen_q, block_size_q), num_q_loop),
        num_k_heads,
        batch_size,
    )
    _gqa_share_sparse_fwd_kernel[grid](
        q,
        k_cache,
        v_cache,
        sink,
        topk_idx,
        o,
        req_to_token,
        cu_seqlens,
        cu_seqblocks_q,
        seq_lens,
        prefix_lens,
        slot_ids,
        max_slots,
        num_k_heads,
        gqa_group_size,
        qk_head_dim,
        v_head_dim,
        topk,
        num_q_loop,
        sm_scale,
        k_scale,
        v_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        sink.stride(0) if sink is not None else 0,
        sink.stride(1) if sink is not None else 0,
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        req_to_token.stride(0),
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        USE_TMA=use_tma,
        IS_FP8=is_fp8,
    )
    return o
