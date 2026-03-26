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


def torch_op(x, w, bias=None, activation=None):
    """PyTorch reference implementation."""
    import torch.nn.functional as F

    out = F.linear(x, w, bias)

    if activation == "gelu":
        out = F.gelu(out)
    elif activation == "silu":
        out = F.silu(out)
    elif activation == "relu":
        out = F.relu(out)

    return out


# ============================================================================
# TEST CONFIGURATIONS (importable)
# ============================================================================

import math

# Evaluation configs: (M, N, K)
EVAL_CONFIGS = [
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (256, 1280, 8192),
    (512, 1280, 8192),
    (2048, 8192, 1024),
]

PROFILE_CONFIGS = [
    (2048, 2048, 2048),
]

RTOL, ATOL = 0.1, 0.1


# ============================================================================
# TEST HARNESS (importable functions)
# ============================================================================


def get_inputs(
    M: int, N: int, K: int, dtype=torch.bfloat16, device="cuda", use_bias=True
):
    """Generate inputs for the GEMM kernel."""
    x = torch.randn(M, K, dtype=dtype, device=device)
    w = torch.randn(N, K, dtype=dtype, device=device)
    bias = torch.randn(N, dtype=dtype, device=device) if use_bias else None
    return x, w, bias


def check_correctness(M: int, N: int, K: int, activation=None) -> dict:
    """Check correctness for a single config."""
    try:
        x, w, bias = get_inputs(M, N, K)
        out_tri = triton_op(x, w, bias, activation)
        out_ref = torch_op(x, w, bias, activation)
        correct = torch.allclose(out_tri, out_ref, rtol=RTOL, atol=ATOL)
        max_diff = (
            torch.max(torch.abs(out_tri - out_ref)).item() if not correct else 0.0
        )
        return {"correct": correct, "max_diff": max_diff, "error": None}
    except Exception as e:
        return {"correct": False, "error": str(e)}


def benchmark_config(
    M: int, N: int, K: int, activation=None, warmup: int = 500, iters: int = 2000
) -> dict:
    """Benchmark a single config."""
    import time

    x, w, bias = get_inputs(M, N, K)

    for _ in range(warmup):
        torch_op(x, w, bias, activation)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        torch_op(x, w, bias, activation)
    torch.cuda.synchronize()
    torch_ms = (time.perf_counter() - start) * 1000 / iters

    for _ in range(warmup):
        triton_op(x, w, bias, activation)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        triton_op(x, w, bias, activation)
    torch.cuda.synchronize()
    triton_ms = (time.perf_counter() - start) * 1000 / iters

    return {
        "torch_ms": torch_ms,
        "triton_ms": triton_ms,
        "speedup": torch_ms / triton_ms,
    }


def evaluate(
    configs=None, warmup: int = 500, iters: int = 2000, verbose: bool = True
) -> dict:
    """Run full evaluation: correctness + benchmark."""
    configs = configs or EVAL_CONFIGS
    results, failures = [], []

    if verbose:
        print(
            f"{'Config (M,N,K)':<20} {'Correct':>8} {'Torch':>10} {'Triton':>10} {'Speedup':>10}"
        )
        print("-" * 60)

    for M, N, K in configs:
        corr = check_correctness(M, N, K)
        if not corr["correct"]:
            failures.append({"config": (M, N, K), **corr})
            if verbose:
                err = corr["error"] or f"max_diff={corr['max_diff']:.2e}"
                print(f"({M},{N},{K}){'':<6} {'FAIL':>8}   {err[:30]}")
            continue

        bench = benchmark_config(M, N, K, None, warmup, iters)
        results.append({"config": (M, N, K), "correct": True, **bench})
        if verbose:
            marker = " *" if bench["speedup"] > 1.0 else ""
            print(
                f"({M},{N},{K}){'':<6} {'PASS':>8} {bench['torch_ms']:>8.4f}ms {bench['triton_ms']:>8.4f}ms {bench['speedup']:>8.2f}x{marker}"
            )

    speedups = [r["speedup"] for r in results]
    geomean = math.prod(speedups) ** (1 / len(speedups)) if speedups else 0.0

    if verbose:
        print("-" * 60)
        print(
            f"{'Status:':<20} {'ALL PASS' if not failures else f'FAILED ({len(failures)}/{len(configs)})'}"
        )
        if speedups:
            print(f"{'Speedup (geomean):':<20} {geomean:.2f}x")

    return {
        "correct": len(failures) == 0,
        "num_correct": len(results),
        "num_failed": len(failures),
        "failures": failures,
        "results": results,
        "speedup_geomean": geomean,
    }


def run_profile(configs=None, warmup: int = 3, iters: int = 1, verbose: bool = True):
    """Run minimal workload for hardware profiling."""
    configs = configs or PROFILE_CONFIGS
    if verbose:
        print(f"Profile: {len(configs)} config(s), {warmup} warmup, {iters} iter(s)")

    for M, N, K in configs:
        x, w, bias = get_inputs(M, N, K)
        for _ in range(warmup):
            triton_op(x, w, bias, None)
        torch.cuda.synchronize()
        for _ in range(iters):
            triton_op(x, w, bias, None)
        torch.cuda.synchronize()
        if verbose:
            print(f"  ({M},{N},{K}) done")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GEMM Kernel Test Harness")
    parser.add_argument(
        "--profile", action="store_true", help="Run minimal profiling workload"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("GEMM Kernel")
    print("=" * 60)

    if args.profile:
        print("\n[Profile Mode]")
        run_profile()
    else:
        print("\n[Evaluation]")
        evaluate()

    print("=" * 60)
