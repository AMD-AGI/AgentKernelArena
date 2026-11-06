# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_bmm.py
# and vllm-project/vllm, vllm/model_executor/layers/mamba/ops/ssd_bmm.py

# ruff: noqa: E501,SIM102
"""Mamba2 SSD intra-chunk C B^T Gram matrices (_bmm_chunk_fwd) in Triton.

For chunk c (tokens cu_chunk_seqlens[c]:cu_chunk_seqlens[c + 1]) and group g:

    out[c, g] = C[chunk, g, :] @ B[chunk, g, :]^T      # [chunk_size, chunk_size], float32

Rows and columns past the end of a short chunk are zero. Non-causal, as in vLLM's SSD prefill;
the autotune is replaced by one fixed launch configuration.
"""
import triton
import triton.language as tl

BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 128
BLOCK_SIZE_K = 32
NUM_WARPS = 4
NUM_STAGES = 2


@triton.jit
def _bmm_chunk_fwd_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    out_ptr,
    cu_chunk_seqlens_ptr,
    # Matrix dimensions
    chunk_size: tl.constexpr,
    K: tl.constexpr,
    ngroups: tl.constexpr,
    stride_a_seqlen: tl.int64,
    stride_a_head: tl.int64,
    stride_ak: tl.constexpr,
    stride_b_seqlen: tl.int64,
    stride_b_head: tl.int64,
    stride_bk: tl.constexpr,
    stride_out_chunk: tl.int64,
    stride_out_head: tl.int64,
    stride_outm: tl.int64,
    stride_outn: tl.constexpr,
    # Meta-parameters
    IS_CAUSAL: tl.constexpr,
    dot_dtype: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_ch = tl.program_id(axis=1).to(tl.int64)
    pid_c = pid_ch // ngroups
    pid_h = pid_ch - pid_c * ngroups
    num_pid_n = tl.cdiv(chunk_size, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    if IS_CAUSAL:
        if pid_n * BLOCK_SIZE_N >= (pid_m + 1) * BLOCK_SIZE_M:
            return

    chunk_seqlen_start = tl.load(cu_chunk_seqlens_ptr + pid_c)
    chunk_seqlen_end = tl.load(cu_chunk_seqlens_ptr + pid_c + 1)

    a_ptr += chunk_seqlen_start * stride_a_seqlen + pid_h * stride_a_head
    b_ptr += chunk_seqlen_start * stride_b_seqlen + pid_h * stride_b_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_a_seqlen + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_b_seqlen)
    chunk_size_limit = chunk_seqlen_end - chunk_seqlen_start

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # compute a * b.T
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < chunk_size_limit)
            & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        ).to(dot_dtype)
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K)
            & (offs_n[None, :] < chunk_size_limit),
            other=0.0,
        ).to(dot_dtype)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    out = acc.to(out_ptr.dtype.element_ty)
    out_ptr += pid_c * stride_out_chunk + pid_h * stride_out_head
    out_ptrs = out_ptr + (stride_outm * offs_m[:, None] + offs_n[None, :] * stride_outn)
    tl.store(
        out_ptrs,
        out,
        mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < chunk_size),
    )


def run(*, C, B, cu_chunk_seqlens, out) -> None:
    seqlen, ngroups, k = C.shape
    nchunks, _, chunk_size, _ = out.shape
    grid = (
        triton.cdiv(chunk_size, BLOCK_SIZE_M) * triton.cdiv(chunk_size, BLOCK_SIZE_N),
        nchunks * ngroups,
    )
    _bmm_chunk_fwd_kernel[grid](
        a_ptr=C,
        b_ptr=B,
        out_ptr=out,
        cu_chunk_seqlens_ptr=cu_chunk_seqlens,
        chunk_size=chunk_size,
        K=k,
        ngroups=ngroups,
        stride_a_seqlen=C.stride(0),
        stride_a_head=C.stride(1),
        stride_ak=C.stride(2),
        stride_b_seqlen=B.stride(0),
        stride_b_head=B.stride(1),
        stride_bk=B.stride(2),
        stride_out_chunk=out.stride(0),
        stride_out_head=out.stride(1),
        stride_outm=out.stride(-2),
        stride_outn=out.stride(-1),
        IS_CAUSAL=False,
        dot_dtype=tl.bfloat16,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
