# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Lean Attention + Paged Attention decode kernel from ROCm/aiter
(aiter/ops/triton/_triton_kernels/attention/lean_atten_paged.py and
aiter/ops/triton/attention/lean_atten_paged.py), wrapped in the run() interface.

Persistent stream-K scheduling: a fixed grid of programs walks the (head, KV block) tiles; a program
that ends inside a tile publishes its partial (m, l, acc) and the host program of that tile merges them
through a lock array.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

BLOCK_M = 16
BLOCK_N = 16
WAVES_PER_EU = 2
NUM_WARPS = 4


@triton.jit
def find_group(x):
    group_id = 0
    total_blocks = 0
    while total_blocks + (group_id + 1) <= x:
        total_blocks += group_id + 1
        group_id += 1
    group_size = group_id + 1
    return group_id, group_size, total_blocks


@triton.jit
def la_persistent_paged(
    Q,
    K,
    V,
    qk_scale,
    Mp,
    Lp,
    Op,
    Out,
    kv_block_tables,
    kv_shape,
    batch_num_block_n,
    locks,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oh,
    stride_om,
    stride_on,
    stride_oph,
    stride_opm,
    stride_opn,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    batch_size: tl.constexpr,
    num_m_blocks: tl.constexpr,
    high_load_wgs: tl.constexpr,
    max_tiles_per_wg: tl.constexpr,
    tiles_per_head: tl.constexpr,
    num_splits: tl.constexpr,
):
    current_pid = tl.program_id(0)

    if current_pid < high_load_wgs:
        iter = max_tiles_per_wg * current_pid
        cta_end_tile_gid = iter + max_tiles_per_wg
    else:
        iter = (max_tiles_per_wg - 1) * (
            current_pid - high_load_wgs
        ) + high_load_wgs * max_tiles_per_wg
        cta_end_tile_gid = iter + (max_tiles_per_wg - 1)

    while iter < cta_end_tile_gid:
        tile_head_idx = iter // tiles_per_head
        tile_idx = tile_head_idx * batch_size
        tile_iter = tile_head_idx * tiles_per_head
        if batch_size == 1:
            req_size = tiles_per_head
        else:
            req_size = tl.load(batch_num_block_n)
        tile_iter_end = tile_iter + req_size
        for b in range(1, batch_size):
            next_req_size = tl.load(batch_num_block_n + b)
            local_head_iter = iter % tiles_per_head
            if (local_head_iter < next_req_size) and (local_head_iter >= req_size):
                tile_iter = tile_iter + req_size
                tile_idx = tile_idx + b
                tile_iter_end = tile_iter + (next_req_size - req_size)
            req_size = next_req_size

        local_iter = iter - tile_iter
        local_iter_end = tl.minimum(tile_iter_end, cta_end_tile_gid) - tile_iter

        host_block = iter == tile_iter
        finishing_block = cta_end_tile_gid >= tile_iter_end

        KV_block_tables_ptr = kv_block_tables + iter
        kv_offset = tile_head_idx * stride_kh

        K_base = K + kv_offset
        V_base = V + kv_offset
        Q_base = Q + tile_idx * (stride_qh // batch_size)

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        acc, l_i, m_i = _attn_lean_tile(
            acc,
            l_i,
            m_i,
            Q_base,
            stride_qm,
            stride_qk,
            kv_shape,
            K_base,
            V_base,
            KV_block_tables_ptr,
            stride_kn,
            stride_kk,
            stride_vn,
            stride_vk,
            qk_scale,
            BLOCK_M,
            BLOCK_N,
            HEAD_DIM,
            tile_idx,
            local_iter,
            local_iter_end,
        )

        m_cta = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_cta = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc_cta = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        offs_m = tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, HEAD_DIM)

        if not host_block:
            mp_ptrs = Mp + current_pid * BLOCK_M + offs_m
            lp_ptrs = Lp + current_pid * BLOCK_M + offs_m
            op_ptrs = (
                Op
                + current_pid * stride_oph
                + offs_m[:, None] * stride_opm
                + offs_k[None, :] * stride_opn
            )

            tl.store(mp_ptrs, m_i, cache_modifier=".wt")
            tl.store(lp_ptrs, l_i, cache_modifier=".wt")
            tl.store(op_ptrs, acc, cache_modifier=".wt")
            tl.debug_barrier()
            tl.atomic_xchg(locks + current_pid, 1)

        if host_block and finishing_block:
            o_h_offs = Out + tile_idx * (stride_oh // batch_size)
            o_ptrs = (
                o_h_offs + offs_m[:, None] * stride_om + offs_k[None, :] * stride_on
            )
            acc = acc / l_i[:, None]
            tl.store(o_ptrs, acc.to(Out.type.element_ty))

        if host_block and not finishing_block:
            o_h_offs = Out + tile_idx * (stride_oh // batch_size)
            o_ptrs = (
                o_h_offs + offs_m[:, None] * stride_om + offs_k[None, :] * stride_on
            )

            last_cta = current_pid + 1
            temp_end_gid = cta_end_tile_gid
            split = 1
            while (split < num_splits) and (temp_end_gid < tile_iter_end):
                if last_cta < high_load_wgs:
                    if (tile_iter_end - temp_end_gid) < max_tiles_per_wg:
                        temp_end_gid += tile_iter_end - temp_end_gid
                    else:
                        temp_end_gid += max_tiles_per_wg
                else:
                    if (tile_iter_end - temp_end_gid) < (max_tiles_per_wg - 1):
                        temp_end_gid += tile_iter_end - temp_end_gid
                    else:
                        temp_end_gid += max_tiles_per_wg - 1

                last_cta += 1
                split += 1

            for cta in range((current_pid + 1), last_cta):
                while tl.atomic_cas(locks + cta, 1, 1) != 1:
                    pass

                offs_mplp = cta * BLOCK_M + tl.arange(0, BLOCK_M)
                mp_ptrs = Mp + offs_mplp
                lp_ptrs = Lp + offs_mplp
                op_h_offs = Op + cta * stride_oph
                op_ptrs = (
                    op_h_offs
                    + offs_m[:, None] * stride_opm
                    + offs_k[None, :] * stride_opn
                )
                m_cta = tl.load(mp_ptrs)
                l_cta = tl.load(lp_ptrs)
                acc_cta = tl.load(op_ptrs)

                m_new = tl.maximum(m_cta, m_i)
                alpha = tl.math.exp2(m_cta - m_new)
                alpha1 = tl.math.exp2(m_i - m_new)
                l_new = alpha * l_cta + alpha1 * l_i
                acc = acc_cta * alpha[:, None] + acc * alpha1[:, None]
                m_i = m_new
                l_i = l_new

            acc = acc / l_i[:, None]
            tl.store(o_ptrs, acc.to(Out.type.element_ty))

        iter = iter + (local_iter_end - local_iter)


@triton.jit
def _attn_lean_tile(
    acc,
    l_i,
    m_i,
    Q_base,
    stride_qm,
    stride_qk,
    kv_shape,
    K_base,
    V_base,
    KV_block_tables_ptr,
    stride_kn,
    stride_kk,
    stride_vn,
    stride_vk,
    qk_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    tile_idx,
    local_iter,
    local_iter_end,
):
    Q_block_ptr = tl.make_block_ptr(
        base=Q_base,
        shape=(BLOCK_M, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    q = tl.load(Q_block_ptr)

    K_block_ptr = tl.make_block_ptr(
        base=K_base,
        shape=(HEAD_DIM, kv_shape),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V_base,
        shape=(kv_shape, HEAD_DIM),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )

    for iter in range(local_iter, local_iter_end):
        kv_block_id = tl.load(KV_block_tables_ptr, cache_modifier=".cg")
        V_bptr = tl.advance(V_block_ptr, (kv_block_id * BLOCK_N, 0))
        K_bptr = tl.advance(K_block_ptr, (0, kv_block_id * BLOCK_N))

        k = tl.load(K_bptr, cache_modifier=".cg")
        qk = tl.dot(q, k)
        qk = qk * qk_scale

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)

        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        v = tl.load(V_bptr, cache_modifier=".cg")
        acc += tl.dot(p.to(v.dtype), v)

        l_ij = tl.sum(p, 1)
        l_i = l_i * alpha + l_ij
        m_i = m_ij.to(m_i.dtype)
        KV_block_tables_ptr += 1

    return acc, l_i, m_i


def get_num_splits_and_buffer_sizes(
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_heads: int,
    num_heads_k: int,
    head_size: int,
    BLOCK_M: int,
    BLOCK_N: int,
    num_SMs: int,
):
    _ = head_size
    num_m_blocks = (max_seqlen_q + BLOCK_M - 1) // BLOCK_M
    num_n_blocks = (max_seqlen_k + BLOCK_N - 1) // BLOCK_N
    max_seqlen_q = max_seqlen_q * num_heads // num_heads_k

    tiles_per_head = num_m_blocks * num_n_blocks
    total_tiles = tiles_per_head * num_heads_k
    lean_griddimz = num_SMs
    max_tiles_per_tb = (total_tiles + lean_griddimz - 1) // lean_griddimz

    if total_tiles % lean_griddimz == 0:
        even_split = True
        num_splits = 1 + ((num_n_blocks + max_tiles_per_tb - 2) // max_tiles_per_tb)
    else:
        even_split = False
        num_splits = 1 + (
            (num_n_blocks + max_tiles_per_tb - 3) // (max_tiles_per_tb - 1)
        )

    high_load_tbs = total_tiles - ((max_tiles_per_tb - 1) * lean_griddimz)

    return (
        num_m_blocks,
        high_load_tbs,
        max_tiles_per_tb,
        tiles_per_head,
        lean_griddimz,
        num_splits,
        even_split,
    )


def run(*, q, k, v, kv_block_tables, batch_num_block_n, sm_scale, out):
    h, rows, head_dim = q.shape
    batch_size = batch_num_block_n.shape[0]
    n_ctx_q = rows // batch_size
    n_ctx_k = k.shape[1]
    assert n_ctx_q == BLOCK_M, "decode assumes the query rows per request equal BLOCK_M"

    # One program per CU, but never more programs than tiles: the stream-K split
    # arithmetic needs at least one tile per program.
    num_cus = torch.cuda.get_device_properties(q.device).multi_processor_count
    total_programs = min(num_cus, h * ((n_ctx_k + BLOCK_N - 1) // BLOCK_N))

    # Scratch for the cross-program reduction; the lock array must start at zero on every launch.
    Mp = torch.empty((total_programs, BLOCK_M), device=q.device, dtype=torch.float32)
    Lp = torch.empty((total_programs, BLOCK_M), device=q.device, dtype=torch.float32)
    Op = torch.empty((total_programs, BLOCK_M, head_dim), device=q.device, dtype=torch.float32)
    locks = torch.zeros((total_programs,), device=q.device, dtype=torch.int32)

    qk_scale = float(sm_scale) * 1.44269504

    (
        num_m_blocks,
        high_load_wgs,
        max_tiles_per_wg,
        tiles_per_head,
        total_programs,
        num_splits,
        even_split,
    ) = get_num_splits_and_buffer_sizes(
        n_ctx_q, n_ctx_k, h, h, head_dim, BLOCK_M, BLOCK_N, total_programs
    )
    _ = even_split

    kv_shape = (n_ctx_k + BLOCK_N - 1) // BLOCK_N
    grid = (total_programs, 1, 1)

    la_persistent_paged[grid](
        q,
        k,
        v,
        qk_scale,
        Mp,
        Lp,
        Op,
        out,
        kv_block_tables,
        kv_shape,
        batch_num_block_n,
        locks,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        Op.stride(0),
        Op.stride(1),
        Op.stride(2),
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        batch_size=batch_size,
        num_m_blocks=num_m_blocks,
        high_load_wgs=high_load_wgs,
        max_tiles_per_wg=max_tiles_per_wg,
        tiles_per_head=tiles_per_head,
        num_splits=num_splits,
        waves_per_eu=WAVES_PER_EU,
        num_warps=NUM_WARPS,
    )
