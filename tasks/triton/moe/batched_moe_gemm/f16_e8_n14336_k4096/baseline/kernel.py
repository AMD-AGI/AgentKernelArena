# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batched MoE GEMM in Triton, adapted from vLLM fused_batched_moe.py.

All experts run in one launch on a 2D grid (expert, M_blocks * N_blocks). Experts and tiles beyond the
expert's token count write zeros, so every row of the padded output is defined.
"""
import triton
import triton.language as tl


@triton.jit
def batched_triton_kernel(
    a_ptr,  # [E, max_num_tokens, K]
    b_ptr,  # [E, K, N] (strided view of [E, N, K])
    c_ptr,  # [E, max_num_tokens, N]
    expert_num_tokens,  # [E]
    compute_type: tl.constexpr,
    max_num_tokens,
    K,
    N,
    stride_ae: tl.int64,
    stride_am: tl.int64,
    stride_ak: tl.int64,
    stride_be: tl.int64,
    stride_bk: tl.int64,
    stride_bn: tl.int64,
    stride_ce: tl.int64,
    stride_cm: tl.int64,
    stride_cn: tl.int64,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    expert_id = tl.program_id(axis=0)
    e_num_tokens = tl.load(expert_num_tokens + expert_id)

    pid_mn = tl.program_id(axis=1)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid_mn // num_pid_n
    pid_n = pid_mn % num_pid_n

    cta_m_start = pid_m * BLOCK_M
    cta_n_start = pid_n * BLOCK_N

    cta_m_size = tl.minimum(BLOCK_M, e_num_tokens - cta_m_start)
    cta_n_size = min(BLOCK_N, N - cta_n_start)

    a_base = a_ptr + expert_id * stride_ae + cta_m_start * stride_am
    b_base = b_ptr + expert_id * stride_be + cta_n_start * stride_bn
    c_base = c_ptr + expert_id * stride_ce + cta_m_start * stride_cm + cta_n_start * stride_cn

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N) % N
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < cta_m_size

    a_ptrs = a_base + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_base + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Tiles without valid tokens skip the K loop and store zeros.
    num_k = tl.where(cta_m_size > 0, tl.cdiv(K, BLOCK_K), 0)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, num_k):
        a = tl.load(a_ptrs, mask=mask_m[:, None] & (offs_k[None, :] < K - k * BLOCK_K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_K) & (offs_n[None, :] < cta_n_size),
                    other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    accumulator = accumulator.to(compute_type)

    offs_cn = tl.arange(0, BLOCK_N)
    c_ptrs = c_base + offs_m[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = ((cta_m_start + offs_m)[:, None] < max_num_tokens) & (offs_cn[None, :] < cta_n_size)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def run(*, A, B, expert_num_tokens, C):
    n_experts, max_num_tokens, K = A.shape
    N = B.shape[1]
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (n_experts, triton.cdiv(max_num_tokens, BLOCK_M) * triton.cdiv(N, BLOCK_N))
    # B is stored as [E, N, K]; the kernel reads it as [E, K, N].
    batched_triton_kernel[grid](
        A, B, C, expert_num_tokens, tl.float16,
        max_num_tokens, K, N,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(2), B.stride(1),
        C.stride(0), C.stride(1), C.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
