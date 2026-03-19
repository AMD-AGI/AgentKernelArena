# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Lean Attention + Paged Attention Kernel Implementation

Based on aiter's lean_atten_paged implementation (ROCm/aiter):
- Uses persistent Stream-K style scheduling for decode attention
- Supports paged KV access through per-head block tables
- Inlines both the Triton kernel and the minimal Python launch wrapper

All Triton kernel code and the wrapper logic are inlined in this file
for self-contained execution without an aiter dependency.
"""

from __future__ import annotations

import argparse
import math
import random
from typing import Sequence

import torch
import triton
import triton.language as tl


# ============================================================================
# INLINED: aiter/ops/triton/_triton_kernels/lean_atten_paged.py
# ============================================================================


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


# ============================================================================
# INLINED: aiter/ops/triton/lean_atten_paged.py
# ============================================================================


def persistent_lean_attention_paged(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_block_tables: torch.Tensor,
    Mp: torch.Tensor,
    Lp: torch.Tensor,
    Op: torch.Tensor,
    locks: torch.Tensor,
    batch_num_block_n: torch.Tensor,
    total_programs: int,
    BLOCK_M: int,
    BLOCK_N: int,
    batch_size: int,
    sm_scale: float,
    num_warps: int,
    waves_per_eu: int,
):
    head_dim_q, head_dim_k, head_dim_v = q.shape[-1], k.shape[-1], v.shape[-1]
    assert (
        head_dim_q == head_dim_k and head_dim_k == head_dim_v
    ), "Incompatible Q/K/V hidden dimensions"
    assert head_dim_k in {16, 32, 64, 128, 256}

    n_ctx_q = q.shape[1] // batch_size
    n_ctx_k = k.shape[1]
    h = q.shape[0]
    assert n_ctx_q == BLOCK_M, "Current decode harness assumes N_CTX_Q == BLOCK_M"

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
        n_ctx_q, n_ctx_k, h, h, head_dim_q, BLOCK_M, BLOCK_N, total_programs
    )
    _ = even_split

    kv_shape = (k.shape[1] + BLOCK_N - 1) // BLOCK_N
    grid = (total_programs, 1, 1)
    o = torch.empty_like(q, dtype=v.dtype)

    la_persistent_paged[grid](
        q,
        k,
        v,
        qk_scale,
        Mp,
        Lp,
        Op,
        o,
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
        o.stride(0),
        o.stride(1),
        o.stride(2),
        Op.stride(0),
        Op.stride(1),
        Op.stride(2),
        HEAD_DIM=head_dim_k,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        batch_size=batch_size,
        num_m_blocks=num_m_blocks,
        high_load_wgs=high_load_wgs,
        max_tiles_per_wg=max_tiles_per_wg,
        tiles_per_head=tiles_per_head,
        num_splits=num_splits,
        waves_per_eu=waves_per_eu,
        num_warps=num_warps,
    )
    return o


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


##################################################################################################################################################
# HARNESS / REFERENCE / BENCHMARK / PROFILE


RTOL, ATOL = 3e-3, 1e-2
_DTYPE = torch.float16


def _config_tag(
    batch: int,
    h: int,
    n_ctx_q: int,
    n_ctx: Sequence[int],
    d: int,
    total_programs: int,
    block_m: int,
    block_n: int,
    waves_per_eu: int,
    num_warps: int,
) -> str:
    n_ctx_str = "[" + ",".join(str(x) for x in n_ctx) + "]"
    return (
        f"B={batch} H={h} NQ={n_ctx_q} N_CTX={n_ctx_str} D={d} "
        f"TP={total_programs} BM={block_m} BN={block_n} "
        f"WPE={waves_per_eu} NW={num_warps}"
    )


def _build_batch_num_block_n(
    n_ctx: Sequence[int], block_n: int, device: torch.device
) -> torch.Tensor:
    running = 0
    cumulative = []
    for seq_len in n_ctx:
        assert (
            seq_len % block_n == 0
        ), "Current harness assumes each sequence length is divisible by BLOCK_N"
        running += seq_len // block_n
        cumulative.append(running)
    return torch.tensor(cumulative, device=device, dtype=torch.int32)


def _build_kv_block_tables(
    h: int,
    n_ctx: Sequence[int],
    block_n: int,
    device: torch.device,
    seed: int,
):
    num_blocks_per_req = [seq_len // block_n for seq_len in n_ctx]
    num_kv_blocks = sum(num_blocks_per_req)

    block_tables = []
    ref_indices = []
    for head_idx in range(h):
        rng = random.Random(seed + head_idx)
        perm = rng.sample(range(num_kv_blocks), num_kv_blocks)
        block_tables.append(perm)

        head_indices = []
        cursor = 0
        for num_req_blocks in num_blocks_per_req:
            req_blocks = perm[cursor : cursor + num_req_blocks]
            cursor += num_req_blocks
            idxs = [
                block_id * block_n + offset
                for block_id in req_blocks
                for offset in range(block_n)
            ]
            head_indices.append(torch.tensor(idxs, dtype=torch.int32, device=device))
        ref_indices.append(head_indices)

    kv_block_tables = torch.tensor(block_tables, dtype=torch.int32, device=device)
    return kv_block_tables, ref_indices


def _make_test_case(
    batch: int,
    h: int,
    n_ctx_q: int,
    n_ctx: Sequence[int],
    d: int,
    total_programs: int,
    dtype: torch.dtype,
    block_m: int,
    block_n: int,
    waves_per_eu: int,
    num_warps: int,
):
    assert batch == len(n_ctx), "batch must equal len(n_ctx)"
    device = torch.device("cuda")
    seed = 20 + batch * 17 + h * 13 + sum(n_ctx) + d * 7 + block_n
    torch.manual_seed(seed)

    sum_n_ctx = sum(int(n) for n in n_ctx)
    batch_num_block_n = _build_batch_num_block_n(n_ctx, block_n, device)

    q = torch.empty((h, n_ctx_q * batch, d), dtype=dtype, device=device).normal_(
        mean=0.0, std=0.5
    )
    k = torch.empty((h, sum_n_ctx, d), dtype=dtype, device=device).normal_(
        mean=0.0, std=0.5
    )
    v = torch.empty((h, sum_n_ctx, d), dtype=dtype, device=device).normal_(
        mean=0.0, std=0.5
    )

    kv_block_tables, ref_indices = _build_kv_block_tables(
        h, n_ctx, block_n, device, seed
    )

    Mp = torch.empty((total_programs, block_m), device=device, dtype=torch.float32)
    Lp = torch.empty((total_programs, block_m), device=device, dtype=torch.float32)
    Op = torch.empty((total_programs, block_m, d), device=device, dtype=torch.float32)
    locks = torch.zeros((total_programs,), device=device, dtype=torch.int32)

    return {
        "q": q,
        "k": k,
        "v": v,
        "kv_block_tables": kv_block_tables,
        "ref_indices": ref_indices,
        "Mp": Mp,
        "Lp": Lp,
        "Op": Op,
        "locks": locks,
        "batch_num_block_n": batch_num_block_n,
        "sm_scale": 0.5,
        "waves_per_eu": waves_per_eu,
        "num_warps": num_warps,
    }


def torch_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ref_indices,
    n_ctx_q: int,
    sm_scale: float,
):
    ref_out = torch.empty_like(q, dtype=v.dtype)
    for head_idx in range(q.shape[0]):
        start_q = 0
        for batch_idx in range(len(ref_indices[head_idx])):
            qb = q[head_idx, start_q : start_q + n_ctx_q, :]
            idxs = ref_indices[head_idx][batch_idx]
            kb = torch.index_select(k[head_idx], dim=0, index=idxs)
            vb = torch.index_select(v[head_idx], dim=0, index=idxs)
            p = torch.matmul(qb, kb.transpose(0, 1)) * sm_scale
            p = torch.softmax(p.float(), dim=-1).to(q.dtype)
            ref_out[head_idx, start_q : start_q + n_ctx_q, :] = torch.matmul(p, vb)
            start_q += n_ctx_q
    return ref_out


# ============================================================================
# CONFIGS
# ============================================================================


# Correctness-focused configs adapted from op_tests/triton_tests/test_la_paged.py.
CORRECTNESS_CONFIGS = [
    (1, 64, 16, (4096,), 64, 304, _DTYPE, 16, 64, 2, 4),
    (1, 96, 16, (32768,), 64, 304, _DTYPE, 16, 64, 2, 4),
    (1, 128, 16, (65536,), 64, 304, _DTYPE, 16, 64, 2, 4),
    (3, 64, 16, (4096, 32768, 65536), 64, 304, _DTYPE, 16, 64, 2, 4),
]

# Benchmark-focused decode configs adapted from op_benchmarks/triton/bench_la_paged_decode.py.
ALL_CONFIGS = [
    (1, 32, 16, (512,), 128, 304, _DTYPE, 16, 16, 2, 4),
    (1, 32, 16, (1024,), 128, 304, _DTYPE, 16, 16, 2, 4),
    (1, 32, 16, (2048,), 128, 304, _DTYPE, 16, 16, 2, 4),
    (1, 32, 16, (4096,), 128, 304, _DTYPE, 16, 16, 2, 4),
    (1, 32, 16, (8192,), 128, 304, _DTYPE, 16, 16, 2, 4),
    (1, 32, 16, (16384,), 128, 304, _DTYPE, 16, 16, 2, 4),
    (1, 32, 16, (32768,), 128, 304, _DTYPE, 16, 16, 2, 4),
]

_n_all = len(ALL_CONFIGS)
if _n_all <= 25:
    HARNESS_CONFIGS = ALL_CONFIGS
else:
    _harness_indices = [int(round(i * (_n_all - 1) / 24)) for i in range(25)]
    HARNESS_CONFIGS = [ALL_CONFIGS[i] for i in _harness_indices]

_profile_indices = [int(round(i * (_n_all - 1) / 4)) for i in range(5)]
PROFILE_CONFIGS = [ALL_CONFIGS[i] for i in _profile_indices]

# Backward compatibility with other harness conventions.
EVAL_CONFIGS = HARNESS_CONFIGS
PROFILE_SHAPES = PROFILE_CONFIGS


# ============================================================================
# TEST HARNESS
# ============================================================================


def _run_single_correctness(
    batch: int,
    h: int,
    n_ctx_q: int,
    n_ctx: Sequence[int],
    d: int,
    total_programs: int,
    dtype: torch.dtype,
    block_m: int,
    block_n: int,
    waves_per_eu: int,
    num_warps: int,
):
    case = _make_test_case(
        batch,
        h,
        n_ctx_q,
        n_ctx,
        d,
        total_programs,
        dtype,
        block_m,
        block_n,
        waves_per_eu,
        num_warps,
    )

    out_triton = persistent_lean_attention_paged(
        q=case["q"],
        k=case["k"],
        v=case["v"],
        kv_block_tables=case["kv_block_tables"],
        Mp=case["Mp"],
        Lp=case["Lp"],
        Op=case["Op"],
        locks=case["locks"],
        batch_num_block_n=case["batch_num_block_n"],
        total_programs=total_programs,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        batch_size=batch,
        sm_scale=case["sm_scale"],
        num_warps=case["num_warps"],
        waves_per_eu=case["waves_per_eu"],
    )
    out_torch = torch_op(
        case["q"],
        case["k"],
        case["v"],
        case["ref_indices"],
        n_ctx_q,
        case["sm_scale"],
    )
    torch.testing.assert_close(out_torch, out_triton, atol=ATOL, rtol=RTOL)


def run_correctness(configs=None, verbose=True):
    if configs is None:
        configs = CORRECTNESS_CONFIGS
    print(f"Running correctness on {len(configs)} configs...")
    results = []
    failures = []

    for cfg in configs:
        batch, h, n_ctx_q, n_ctx, d, total_programs, dtype, block_m, block_n, waves_per_eu, num_warps = cfg
        tag = _config_tag(
            batch, h, n_ctx_q, n_ctx, d, total_programs, block_m, block_n, waves_per_eu, num_warps
        )
        try:
            _run_single_correctness(*cfg)
            results.append(tag)
            if verbose:
                print(f"  PASS: {tag}")
        except Exception as exc:
            failures.append({"config": tag, "error": str(exc)})
            if verbose:
                print(f"  FAIL: {tag} - {str(exc)[:120]}")
        torch.cuda.empty_cache()

    if verbose:
        print("-" * 70)
        status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{len(configs)})"
        print(f"{'Status:':<22} {status}")

    return {
        "correct": len(failures) == 0,
        "num_correct": len(results),
        "num_failed": len(failures),
        "failures": failures,
    }


def run_profile(configs=None, warmup=50, iters=200, verbose=True):
    if configs is None:
        configs = PROFILE_CONFIGS
    if verbose:
        print(f"Profile: {len(configs)} config(s), {warmup} warmup, {iters} iter(s)")

    for cfg in configs:
        batch, h, n_ctx_q, n_ctx, d, total_programs, dtype, block_m, block_n, waves_per_eu, num_warps = cfg
        case = _make_test_case(
            batch,
            h,
            n_ctx_q,
            n_ctx,
            d,
            total_programs,
            dtype,
            block_m,
            block_n,
            waves_per_eu,
            num_warps,
        )
        for _ in range(warmup):
            persistent_lean_attention_paged(
                q=case["q"],
                k=case["k"],
                v=case["v"],
                kv_block_tables=case["kv_block_tables"],
                Mp=case["Mp"],
                Lp=case["Lp"],
                Op=case["Op"],
                locks=case["locks"],
                batch_num_block_n=case["batch_num_block_n"],
                total_programs=total_programs,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                batch_size=batch,
                sm_scale=case["sm_scale"],
                num_warps=case["num_warps"],
                waves_per_eu=case["waves_per_eu"],
            )
        torch.cuda.synchronize()
        for _ in range(iters):
            persistent_lean_attention_paged(
                q=case["q"],
                k=case["k"],
                v=case["v"],
                kv_block_tables=case["kv_block_tables"],
                Mp=case["Mp"],
                Lp=case["Lp"],
                Op=case["Op"],
                locks=case["locks"],
                batch_num_block_n=case["batch_num_block_n"],
                total_programs=total_programs,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                batch_size=batch,
                sm_scale=case["sm_scale"],
                num_warps=case["num_warps"],
                waves_per_eu=case["waves_per_eu"],
            )
        torch.cuda.synchronize()
        if verbose:
            print(f"  {_config_tag(batch, h, n_ctx_q, n_ctx, d, total_programs, block_m, block_n, waves_per_eu, num_warps)} done")
        torch.cuda.empty_cache()


def run_benchmark(configs=None, warmup=50, iters=200, verbose=True, baseline_fn=None):
    """Benchmark kernel vs reference. Uses baseline_fn (Triton) when provided; else torch_op (PyTorch)."""
    if configs is None:
        configs = HARNESS_CONFIGS

    latencies = []
    speedups = []
    results = []
    ref_label = "baseline_triton" if baseline_fn is not None else "PyTorch"

    print(
        f"Running benchmark on {len(configs)} configs, {warmup} warmup, {iters} iterations each..."
    )
    print(f"  Comparing kernel vs {ref_label}")
    if verbose:
        print(f"{'Config':<72} {'Ref':>10} {'Triton':>10} {'Speedup':>10}")
        print("-" * 108)

    for cfg in configs:
        batch, h, n_ctx_q, n_ctx, d, total_programs, dtype, block_m, block_n, waves_per_eu, num_warps = cfg
        case = _make_test_case(
            batch,
            h,
            n_ctx_q,
            n_ctx,
            d,
            total_programs,
            dtype,
            block_m,
            block_n,
            waves_per_eu,
            num_warps,
        )
        tag = _config_tag(
            batch, h, n_ctx_q, n_ctx, d, total_programs, block_m, block_n, waves_per_eu, num_warps
        )

        for _ in range(warmup):
            persistent_lean_attention_paged(
                q=case["q"],
                k=case["k"],
                v=case["v"],
                kv_block_tables=case["kv_block_tables"],
                Mp=case["Mp"],
                Lp=case["Lp"],
                Op=case["Op"],
                locks=case["locks"],
                batch_num_block_n=case["batch_num_block_n"],
                total_programs=total_programs,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                batch_size=batch,
                sm_scale=case["sm_scale"],
                num_warps=case["num_warps"],
                waves_per_eu=case["waves_per_eu"],
            )
        torch.cuda.synchronize()

        triton_times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            persistent_lean_attention_paged(
                q=case["q"],
                k=case["k"],
                v=case["v"],
                kv_block_tables=case["kv_block_tables"],
                Mp=case["Mp"],
                Lp=case["Lp"],
                Op=case["Op"],
                locks=case["locks"],
                batch_num_block_n=case["batch_num_block_n"],
                total_programs=total_programs,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                batch_size=batch,
                sm_scale=case["sm_scale"],
                num_warps=case["num_warps"],
                waves_per_eu=case["waves_per_eu"],
            )
            end.record()
            torch.cuda.synchronize()
            triton_times.append(start.elapsed_time(end))
        triton_ms = sorted(triton_times)[len(triton_times) // 2]

        ref_times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            if baseline_fn is not None:
                baseline_fn(
                    q=case["q"],
                    k=case["k"],
                    v=case["v"],
                    kv_block_tables=case["kv_block_tables"],
                    Mp=case["Mp"],
                    Lp=case["Lp"],
                    Op=case["Op"],
                    locks=case["locks"],
                    batch_num_block_n=case["batch_num_block_n"],
                    total_programs=total_programs,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    batch_size=batch,
                    sm_scale=case["sm_scale"],
                    num_warps=case["num_warps"],
                    waves_per_eu=case["waves_per_eu"],
                )
            else:
                torch_op(
                    case["q"],
                    case["k"],
                    case["v"],
                    case["ref_indices"],
                    n_ctx_q,
                    case["sm_scale"],
                )
            end.record()
            torch.cuda.synchronize()
            ref_times.append(start.elapsed_time(end))
        ref_ms = sorted(ref_times)[len(ref_times) // 2]

        speedup = ref_ms / triton_ms if triton_ms > 0 else 1.0
        latencies.append(triton_ms)
        speedups.append(speedup)
        results.append(
            {
                "config": tag,
                "ref_ms": ref_ms,
                "triton_ms": triton_ms,
                "speedup": speedup,
            }
        )

        if verbose:
            marker = " *" if speedup > 1.0 else ""
            print(f"{tag:<72} {ref_ms:>8.4f}ms {triton_ms:>8.4f}ms {speedup:>8.2f}x{marker}")

        torch.cuda.empty_cache()

    geomean_latency = math.exp(sum(math.log(t) for t in latencies) / len(latencies))
    geomean_speedup = math.exp(sum(math.log(s) for s in speedups) / len(speedups))

    if verbose:
        print("-" * 108)
        print(f"{'Geometric mean latency:':<72} {geomean_latency:.4f} ms")
        print(f"{'Geometric mean speedup:':<72} {geomean_speedup:.2f}x")
        print(f"GEAK_RESULT_LATENCY_MS={geomean_latency:.4f}")
        print(f"GEAK_RESULT_GEOMEAN_SPEEDUP={geomean_speedup:.4f}")

    return {
        "geomean_latency_ms": geomean_latency,
        "geomean_speedup": geomean_speedup,
        "results": results,
    }


# ============================================================================
# MAIN
# ============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lean Attention + Paged Attention Kernel Test Harness"
    )
    parser.add_argument(
        "--correctness",
        action="store_true",
        help="Run correctness tests on correctness configs",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run minimal profiling workload",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark on HARNESS_CONFIGS",
    )
    parser.add_argument(
        "--full-benchmark",
        action="store_true",
        help="Run benchmark on ALL_CONFIGS",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Number of warmup iterations (default: 50)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Number of benchmark iterations (default: 200)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Lean Attention + Paged Attention Kernel Test Harness")
    print("=" * 70)

    if args.correctness:
        print("\n[Correctness Mode]")
        run_correctness(CORRECTNESS_CONFIGS)
    elif args.profile:
        print("\n[Profile Mode]")
        run_profile(PROFILE_CONFIGS, warmup=args.warmup, iters=args.iterations)
    elif args.full_benchmark:
        print("\n[Full Benchmark Mode]")
        run_benchmark(ALL_CONFIGS, warmup=args.warmup, iters=args.iterations)
    else:
        print("\n[Benchmark Mode]")
        run_benchmark(HARNESS_CONFIGS, warmup=args.warmup, iters=args.iterations)

    print("=" * 70)
