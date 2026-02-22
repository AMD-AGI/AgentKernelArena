"""Per-expert GEMM kernel using Triton, adapted from vLLM fused_batched_moe.py.

Single-expert matrix multiply: C = A @ B for one expert's tokens.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def expert_triton_kernel(
    a_ptr,  # [max_tokens, K]
    b_ptr,  # [K, N]
    c_ptr,  # [max_tokens, N]
    compute_type: tl.constexpr,
    # Dimensions
    M,
    N,
    K,
    # strides
    stride_am: tl.int64,
    stride_ak: tl.int64,
    stride_bk: tl.int64,
    stride_bn: tl.int64,
    stride_cm: tl.int64,
    stride_cn: tl.int64,
    # Kernel config
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Per-expert GEMM kernel. Computes C[0:M, block_n] = A[0:M, :] @ B[:, block_n]
    for a single expert's tokens. Each program handles one (block_m, block_n) tile.
    """
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

    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = mask_m[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def expert_gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Per-expert GEMM: C = A @ B.

    Args:
        A: [M, K] input tokens for one expert
        B: [K, N] expert weight matrix
    Returns:
        C: [M, N] output
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
    expert_triton_kernel[grid](
        A, B, C,
        tl.float16,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return C
