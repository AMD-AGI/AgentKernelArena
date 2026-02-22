"""MoE inner matrix multiply kernel (moe_mmk) using Triton, adapted from vLLM fused_batched_moe.py.

This is the inner GEMM helper used by expert_triton_kernel. Made standalone for testing.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def moe_mmk(
    a_ptrs,
    b_ptrs,
    K,
    a_scale_ptr,
    b_scale_ptr,
    # Strides
    stride_ak: tl.int64,
    stride_bk: tl.int64,
    # Offsets and masks
    offs_m,
    offs_n,
    mask_m,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    compute_type: tl.constexpr,
):
    """
    Inner matrix multiply for MoE: compute one [BLOCK_M, BLOCK_N] tile of C = A @ B.
    Iterates over K dimension in BLOCK_K steps with masking.
    """
    offs_k = tl.arange(0, BLOCK_K)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(
            a_ptrs,
            mask=mask_m[:, None] & (offs_k[None, :] < K - k * BLOCK_K),
            other=0.0,
        )
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    accumulator = accumulator.to(compute_type)
    return accumulator


@triton.jit
def moe_mmk_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am: tl.int64,
    stride_ak: tl.int64,
    stride_bk: tl.int64,
    stride_bn: tl.int64,
    stride_cm: tl.int64,
    stride_cn: tl.int64,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    compute_type: tl.constexpr,
):
    """Standalone wrapper kernel that calls moe_mmk for one tile and stores result."""
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N) % N
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Dummy scale pointers (not used in non-quantized path)
    accumulator = moe_mmk(
        a_ptrs, b_ptrs, K,
        a_ptr, b_ptr,  # dummy scale ptrs
        stride_ak, stride_bk,
        offs_m, offs_n, mask_m,
        BLOCK_M, BLOCK_N, BLOCK_K, compute_type,
    )

    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = mask_m[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def moe_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiply using the moe_mmk kernel.

    Args:
        A: [M, K] input matrix
        B: [K, N] weight matrix
    Returns:
        C: [M, N] output matrix
    """
    assert A.dim() == 2 and B.dim() == 2
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    C = torch.empty(M, N, device=A.device, dtype=A.dtype)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    moe_mmk_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        compute_type=tl.float16,
    )
    return C
