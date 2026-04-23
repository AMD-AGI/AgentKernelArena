#!/usr/bin/env python3
"""
GEMM (General Matrix Multiplication) Kernel Implementation

Based on aiter's gemm_a16w16 implementation:
- Computes Y = X @ W^T + bias
- Supports optional activation functions (GELU, SiLU, ReLU)
- Optimized for AMD MI325X GPUs
"""

import torch
import triton
import triton.language as tl

# ============================================================================
# TRITON KERNELS
# ============================================================================


@triton.jit
def _tanh(x):
    """Tanh approximation using sigmoid (from aiter)."""
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _gemm_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_ym,
    stride_yn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ADD_BIAS: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """Matrix multiplication kernel: Y = X @ W^T + bias."""
    pid = tl.program_id(0)

    # Compute block indices with grouping for better L2 cache utilization
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Compute block offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Initialize pointers to first block
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + (offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Main loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load X and W tiles
        k_offs = k * BLOCK_SIZE_K + offs_k
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        w_mask = (offs_n[:, None] < N) & (k_offs[None, :] < K)

        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Compute matmul for this block
        acc += tl.dot(x_tile, tl.trans(w_tile))

        # Advance pointers
        x_ptrs += BLOCK_SIZE_K * stride_xk
        w_ptrs += BLOCK_SIZE_K * stride_wk

    # Add bias if present
    if ADD_BIAS:
        bias_ptrs = bias_ptr + offs_n
        bias_mask = offs_n < N
        bias_vals = tl.load(bias_ptrs, mask=bias_mask, other=0.0)
        acc += bias_vals[None, :]

    # Apply activation function
    if ACTIVATION == "gelu":
        # GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        acc = (
            acc * 0.5 * (1.0 + _tanh(0.7978845608 * (acc + 0.044715 * acc * acc * acc)))
        )
    elif ACTIVATION == "silu":
        # SiLU: x * sigmoid(x)
        acc = acc * tl.sigmoid(acc)
    elif ACTIVATION == "relu":
        acc = tl.where(acc > 0, acc, 0.0)

    # Store output
    y_ptrs = y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=y_mask)


# ============================================================================
# PYTHON WRAPPERS
# ============================================================================


def get_config(M, N, K):
    """Get kernel configuration based on matrix dimensions."""
    # Default configuration
    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    }

    # Adjust for small matrices
    if M <= 32:
        config["BLOCK_SIZE_M"] = 32
    if N <= 32:
        config["BLOCK_SIZE_N"] = 32
    if K <= 32:
        config["BLOCK_SIZE_K"] = 16

    # Adjust for large matrices
    if M >= 2048 and N >= 2048:
        config["BLOCK_SIZE_M"] = 128
        config["BLOCK_SIZE_N"] = 128
        config["BLOCK_SIZE_K"] = 64
        config["GROUP_SIZE_M"] = 8

    return config


def gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor = None,
    activation: str = None,
) -> torch.Tensor:
    """
    Compute matrix multiplication Y = X @ W^T + bias with optional activation.

    Args:
        x: Input matrix with shape (M, K)
        w: Weight matrix with shape (N, K) - will be transposed internally
        bias: Optional bias vector with shape (N,)
        activation: Optional activation function ('gelu', 'silu', 'relu', None)

    Returns:
        Output matrix with shape (M, N)
    """
    assert x.shape[1] == w.shape[1], f"Incompatible shapes: x={x.shape}, w={w.shape}"

    M, K = x.shape
    N, _ = w.shape

    # Transpose W for computation
    w_t = w.T.contiguous()

    y = torch.empty((M, N), dtype=x.dtype, device=x.device)

    config = get_config(M, N, K)

    grid = (
        triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(N, config["BLOCK_SIZE_N"]),
    )

    _gemm_kernel[grid](
        x,
        w,
        bias if bias is not None else x,  # Dummy if no bias
        y,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
        GROUP_SIZE_M=config["GROUP_SIZE_M"],
        ADD_BIAS=(bias is not None),
        ACTIVATION=activation if activation else "",
        num_warps=4,
        num_stages=2,
    )

    return y


def triton_op(x, w, bias=None, activation=None):
    """Main GEMM entry point."""
    return gemm(x, w, bias, activation)


gemm_a16w16 = gemm
