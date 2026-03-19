#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Self-sufficient test harness for moe_routing_sigmoid_top1_fused kernel
# Inlined from ROCm/aiter — no aiter imports required.

import argparse
import os
import math
import sys
import time
from functools import lru_cache, partial
from typing import Optional

import torch
import triton
import triton.language as tl

# ============================================================================
# Triton JIT kernel (inlined from aiter/ops/triton/_triton_kernels/moe/
#   moe_routing_sigmoid_top1_fused.py)
# ============================================================================

@triton.jit
def _routing_sigmoid_top1_kernel(
    X_ptr,
    W_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_topk_ids_m,
    stride_topk_ids_n,
    stride_topk_weights_m,
    stride_topk_weights_n,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TOPK: tl.constexpr,
    FUSED_SHARED_EXPERTS: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    _TOPK: tl.constexpr = TOPK + 1 if FUSED_SHARED_EXPERTS else TOPK

    offs_topk = tl.arange(0, _TOPK)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k_iter = k + offs_k
        mask_k = offs_k_iter < K

        X_ptrs = X_ptr + (
            offs_m[:, None] * stride_xm
            + offs_k_iter[None, :] * stride_xk
        )
        W_ptrs = W_ptr + (
            offs_k_iter[:, None] * stride_wk + offs_n[None, :] * stride_wn
        )

        x = tl.load(X_ptrs, mask=(mask_m[:, None] & mask_k[None, :]), other=0.0)
        w = tl.load(W_ptrs, mask=(mask_k[:, None] & mask_n[None, :]), other=0.0)

        acc = tl.dot(x, w, acc=acc)

    acc = tl.sigmoid(acc)
    topk_ids = tl.argmax(acc, axis=1, tie_break_left=True)
    topk_weights = tl.max(acc, axis=1)

    topk_ids_buffer = tl.zeros((BLOCK_M, _TOPK), dtype=tl.int32)
    topk_weights_buffer = tl.zeros((BLOCK_M, _TOPK), dtype=tl.float32)

    if FUSED_SHARED_EXPERTS:
        topk_ids_buffer = tl.where(
            (offs_topk[None, :] < _TOPK - 1), topk_ids[:, None], N
        )
        topk_weights_buffer = tl.where(
            (offs_topk[None, :] < _TOPK - 1), topk_weights[:, None], 1.0
        )
    else:
        topk_ids_buffer = topk_ids[:, None]
        topk_weights_buffer = topk_weights[:, None]

    topk_ids_ptrs = (
        topk_ids_ptr
        + offs_m[:, None] * stride_topk_ids_m
        + offs_topk[None, :] * stride_topk_ids_n
    )

    topk_weights_ptrs = (
        topk_weights_ptr
        + offs_m[:, None] * stride_topk_weights_m
        + offs_topk[None, :] * stride_topk_weights_n
    )

    tl.store(topk_ids_ptrs, topk_ids_buffer)
    tl.store(topk_weights_ptrs, topk_weights_buffer)


# ============================================================================
# Tuning configs (inlined from aiter/ops/triton/configs/moe/
#   gfx942-MOE_ROUTING_SIGMOID_TOPK1.json and gfx950 variant)
# ============================================================================

_CONFIG_DICT = {
    "gfx942": {
        "N16": {
            "small":  {"BLOCK_M": 16, "BLOCK_K": 256, "num_warps": 4, "num_stages": 2, "waves_per_eu": 3, "kpack": 1},
            "medium": {"BLOCK_M": 16, "BLOCK_K": 256, "num_warps": 4, "num_stages": 2, "waves_per_eu": 3, "kpack": 1},
            "large":  {"BLOCK_M": 16, "BLOCK_K": 256, "num_warps": 4, "num_stages": 2, "waves_per_eu": 3, "kpack": 2},
            "xlarge": {"BLOCK_M": 32, "BLOCK_K": 256, "num_warps": 4, "num_stages": 2, "waves_per_eu": 0, "kpack": 2},
        },
        "N128": {
            "small":  {"BLOCK_M": 16, "BLOCK_K": 256, "num_warps": 8, "num_stages": 2, "waves_per_eu": 0, "kpack": 2},
            "medium": {"BLOCK_M": 16, "BLOCK_K": 256, "num_warps": 8, "num_stages": 2, "waves_per_eu": 0, "kpack": 2},
            "large":  {"BLOCK_M": 16, "BLOCK_K": 256, "num_warps": 8, "num_stages": 2, "waves_per_eu": 0, "kpack": 2},
            "xlarge": {"BLOCK_M": 32, "BLOCK_K": 256, "num_warps": 4, "num_stages": 2, "waves_per_eu": 0, "kpack": 2},
        },
    },
    "gfx950": {
        "N16": {
            "small":  {"BLOCK_M": 16, "BLOCK_K": 256, "num_warps": 4, "num_stages": 2, "waves_per_eu": 0, "kpack": 1},
            "medium": {"BLOCK_M": 16, "BLOCK_K": 256, "num_warps": 4, "num_stages": 2, "waves_per_eu": 0, "kpack": 1},
            "large":  {"BLOCK_M": 16, "BLOCK_K": 256, "num_warps": 4, "num_stages": 2, "waves_per_eu": 0, "kpack": 1},
            "xlarge": {"BLOCK_M": 32, "BLOCK_K": 256, "num_warps": 4, "num_stages": 2, "waves_per_eu": 0, "kpack": 1},
        },
        "N128": {
            "small":  {"BLOCK_M": 16, "BLOCK_K": 256, "num_warps": 8, "num_stages": 2, "waves_per_eu": 0, "kpack": 1},
            "medium": {"BLOCK_M": 16, "BLOCK_K": 256, "num_warps": 8, "num_stages": 2, "waves_per_eu": 0, "kpack": 1},
            "large":  {"BLOCK_M": 16, "BLOCK_K": 256, "num_warps": 8, "num_stages": 2, "waves_per_eu": 0, "kpack": 1},
            "xlarge": {"BLOCK_M": 32, "BLOCK_K": 256, "num_warps": 4, "num_stages": 2, "waves_per_eu": 0, "kpack": 1},
        },
    },
}


@lru_cache(maxsize=1)
def _get_arch():
    try:
        return triton.runtime.driver.active.get_current_target().arch
    except RuntimeError:
        from jax._src.lib import gpu_triton as triton_kernel_call_lib
        return triton_kernel_call_lib.get_arch_details("0").split(":")[0]


@lru_cache(maxsize=1024)
def _get_config(M, N, K):
    arch = _get_arch()
    configs = _CONFIG_DICT.get(arch, _CONFIG_DICT["gfx942"])
    n_key = "N16" if N <= 16 else "N128"
    m_key = (
        "xlarge"
        if M >= 4096
        else "large" if M >= 2048 else "medium" if M >= 1024 else "small"
    )
    return configs[n_key][m_key]


# ============================================================================
# Operator-level wrapper (inlined from aiter/ops/triton/moe/
#   moe_routing_sigmoid_top1_fused.py)
# ============================================================================

def routing_sigmoid_top1(
    x, w, topk, fused_shared_experts=False, config: Optional[dict] = None
):
    """
    Computes top-1 MoE routing with sigmoid activation for expert selection.

    Args:
        x (torch.Tensor): Input activations with shape (batch_size, seq_len, hidden_dim) or (M, K).
        w (torch.Tensor): Routing weights with shape (hidden_dim, num_experts).
        topk (int): Number of experts to select. Must be 1.
        fused_shared_experts (bool): Include shared expert (always selected) alongside top-1.
        config (Optional[dict]): Kernel tuning parameters (BLOCK_M, BLOCK_K).

    Returns:
        tuple: (topk_ids, topk_weights)
            - topk_ids (torch.Tensor): Selected expert IDs with shape (M, topk) or (M, topk+1) if fused_shared_experts.
            - topk_weights (torch.Tensor): Routing weights (sigmoid scores) with shape (M, topk) or (M, topk+1).
    """
    x = x.view(-1, x.shape[-1])

    assert topk == 1

    M, K = x.shape
    Kb, N = w.shape
    assert K == Kb

    _topk = topk
    if fused_shared_experts:
        _topk += 1

    topk_ids = torch.empty((M, _topk), device=x.device, dtype=torch.int32)
    topk_weights = torch.empty((M, _topk), device=x.device, dtype=torch.float32)

    config = _get_config(M, N, K)

    def grid(META):
        return (triton.cdiv(M, META["BLOCK_M"]),)

    _routing_sigmoid_top1_kernel[grid](
        x,
        w,
        topk_ids,
        topk_weights,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        BLOCK_N=N,
        TOPK=topk,
        FUSED_SHARED_EXPERTS=fused_shared_experts,
        **config,
    )

    return topk_ids, topk_weights


##################################################################################################################################################

# ============================================================================
# Shape definitions extracted from aiter test/bench files
# ============================================================================
# test_moe_routing_sigmoid_top1_fused.py:
#   M: [128, 1024, 2048, 4096, 8192]  N: [16, 128]  K: [16, 128]
# bench_moe_routing_sigmoid_top1_fused.py:
#   Prefill: M=[1024, 2048, 4096, 8192], K=5120, N=[16, 128]
#   Decode:  M=[64, 128, 256],           K=5120, N=[16, 128]

ALL_SHAPES = [
    (128, 16, 16),
    (128, 16, 128),
    (128, 128, 16),
    (128, 128, 128),
    (64, 16, 5120),
    (64, 128, 5120),
    (128, 16, 5120),
    (128, 128, 5120),
    (256, 16, 5120),
    (256, 128, 5120),
    (1024, 16, 16),
    (1024, 16, 128),
    (1024, 128, 16),
    (1024, 128, 128),
    (1024, 16, 5120),
    (1024, 128, 5120),
    (2048, 16, 16),
    (2048, 16, 128),
    (2048, 128, 16),
    (2048, 128, 128),
    (2048, 16, 5120),
    (2048, 128, 5120),
    (4096, 16, 16),
    (4096, 16, 128),
    (4096, 128, 16),
    (4096, 128, 128),
    (4096, 16, 5120),
    (4096, 128, 5120),
    (8192, 16, 16),
    (8192, 16, 128),
    (8192, 128, 16),
    (8192, 128, 128),
    (8192, 16, 5120),
    (8192, 128, 5120),
]

_n_all = len(ALL_SHAPES)
_bench_indices = [int(i * (_n_all - 1) / 19) for i in range(20)]
HARNESS_SHAPES = [ALL_SHAPES[i] for i in _bench_indices]

_profile_indices = [int(i * (_n_all - 1) / 4) for i in range(5)]
PROFILE_SHAPES = [ALL_SHAPES[i] for i in _profile_indices]


# ============================================================================
# Reference implementation
# ============================================================================

def _torch_routing_sigmoid_top1(
    x, w, topk, fused_shared_experts=False, dummy_ids=None, dummy_weights=None
):
    """Reference implementation using PyTorch."""
    scores = torch.sigmoid(torch.matmul(x, w).to(torch.float32))
    assert topk == 1
    topk_weights, topk_ids = torch.topk(scores, topk, dim=1)
    topk_ids = topk_ids.to(torch.int32)
    topk_weights = topk_weights.to(torch.float32)
    if fused_shared_experts:
        topk_ids = torch.cat([topk_ids, dummy_ids], dim=1)
        topk_weights = torch.cat([topk_weights, dummy_weights], dim=1)
    return topk_ids, topk_weights


def _gpu_median_time(fn, warmup, iterations):
    """Time *fn* using CUDA events and return the median elapsed time in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        fn()
        end_evt.record()
        torch.cuda.synchronize()
        times.append(start_evt.elapsed_time(end_evt))

    times.sort()
    return times[len(times) // 2]


# ============================================================================
# Harness modes
# ============================================================================

def run_correctness(shapes, atol, rtol):
    """Correctness: kernel outputs vs PyTorch reference on *shapes*."""
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16
    TOPK = 1

    print(f"Running correctness tests on {len(shapes)} shapes "
          f"(atol={atol}, rtol={rtol})...")

    all_passed = True
    for i, (M, N, K) in enumerate(shapes):
        x = torch.randint(-2, 3, (M, K), dtype=dtype, device=device)
        w = torch.randint(-2, 3, (K, N), dtype=dtype, device=device)

        dummy_ids = torch.ones((M, 1), dtype=torch.int32, device=device) * N
        dummy_weights = torch.ones((M, 1), dtype=torch.float32, device=device)

        topk_ids, topk_weights = routing_sigmoid_top1(
            x, w, TOPK, fused_shared_experts=True
        )

        ref_fn = partial(
            _torch_routing_sigmoid_top1,
            dummy_ids=dummy_ids, dummy_weights=dummy_weights,
        )
        ref_ids, ref_weights = ref_fn(x, w, TOPK, fused_shared_experts=True)

        try:
            torch.testing.assert_close(ref_ids, topk_ids, atol=atol, rtol=rtol)
            torch.testing.assert_close(ref_weights, topk_weights, atol=atol, rtol=rtol)
            print(f"  [{i+1}/{len(shapes)}] M={M}, N={N}, K={K}: PASS")
        except AssertionError as e:
            print(f"  [{i+1}/{len(shapes)}] M={M}, N={N}, K={K}: FAIL")
            print(f"    {e}")
            all_passed = False

    if all_passed:
        print("ALL PASS")
    else:
        print("Some correctness tests FAILED.")
    return all_passed


def run_profile(shapes, warmup):
    """Profile: run every shape in *shapes* with warmup for external profiler."""
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16
    TOPK = 1

    print(f"Running profiling on {len(shapes)} shapes (warmup={warmup})...")

    for i, (M, N, K) in enumerate(shapes):
        x = torch.randn((M, K), dtype=dtype, device=device)
        w = torch.randn((K, N), dtype=dtype, device=device) * 0.1

        for _ in range(warmup):
            routing_sigmoid_top1(x, w, TOPK, fused_shared_experts=True)
        torch.cuda.synchronize()

        routing_sigmoid_top1(x, w, TOPK, fused_shared_experts=True)
        torch.cuda.synchronize()

        print(f"  [{i+1}/{len(shapes)}] M={M}, N={N}, K={K}: done")

    print("Profile run complete.")


def run_benchmark(shapes, warmup, iterations):
    """Benchmark kernel vs reference; report per-shape speedups and geomean."""
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16
    TOPK = 1

    print(f"Running benchmark on {len(shapes)} shapes "
          f"(warmup={warmup}, iterations={iterations})...")
    print(f"{'#':>4s}  {'Shape':>24s}  {'Ref (ms)':>10s}  "
          f"{'Kernel (ms)':>12s}  {'Speedup':>8s}")
    print("-" * 68)

    speedups = []
    latencies = []

    for i, (M, N, K) in enumerate(shapes):
        x = torch.randn((M, K), dtype=dtype, device=device)
        w = torch.randn((K, N), dtype=dtype, device=device) * 0.1

        dummy_ids = torch.ones((M, 1), dtype=torch.int32, device=device) * N
        dummy_weights = torch.ones((M, 1), dtype=torch.float32, device=device)

        ref_fn = partial(
            _torch_routing_sigmoid_top1,
            dummy_ids=dummy_ids, dummy_weights=dummy_weights,
        )

        def _run_ref(ref_fn=ref_fn, x=x, w=w):
            ref_fn(x, w, TOPK, fused_shared_experts=True)

        def _run_kernel(x=x, w=w):
            routing_sigmoid_top1(x, w, TOPK, fused_shared_experts=True)

        ref_time = _gpu_median_time(_run_ref, warmup, iterations)
        kernel_time = _gpu_median_time(_run_kernel, warmup, iterations)

        speedup = ref_time / kernel_time if kernel_time > 0 else float("inf")
        speedups.append(speedup)
        latencies.append(kernel_time)

        shape_str = f"M={M}, N={N}, K={K}"
        print(f"  {i+1:>3d}   {shape_str:>24s}  {ref_time:>10.4f}  "
              f"{kernel_time:>12.4f}  {speedup:>7.2f}x")

    print("-" * 68)
    geomean_speedup = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
    geomean_latency = math.exp(sum(math.log(t) for t in latencies) / len(latencies))
    print(f"{'Geometric mean latency:':<22} {geomean_latency:.4f} ms")
    print(f"{'Geometric mean speedup:':<22} {geomean_speedup:.2f}x")
    print(f"GEAK_RESULT_LATENCY_MS={geomean_latency:.4f}")
    print(f"GEAK_RESULT_SPEEDUP={geomean_speedup:.2f}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test harness for moe_routing_sigmoid_top1_fused kernel",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--correctness", action="store_true",
                      help="Run correctness tests on HARNESS_SHAPES")
    mode.add_argument("--profile", action="store_true",
                      help="Run profiling on PROFILE_SHAPES")
    mode.add_argument("--benchmark", action="store_true",
                      help="Run benchmark on HARNESS_SHAPES")
    mode.add_argument("--full-benchmark", action="store_true",
                      help="Run benchmark on ALL_SHAPES")

    parser.add_argument("--warmup", type=int, default=50,
                        help="Warmup iterations (default: 50)")
    parser.add_argument("--iterations", type=int, default=200,
                        help="Benchmark iterations (default: 200)")
    parser.add_argument("--atol", type=float, default=1e-4,
                        help="Absolute tolerance for correctness (default: 1e-4)")
    parser.add_argument("--rtol", type=float, default=1e-4,
                        help="Relative tolerance for correctness (default: 1e-4)")

    args = parser.parse_args()

    if args.correctness:
        success = run_correctness(HARNESS_SHAPES, atol=args.atol, rtol=args.rtol)
        sys.exit(0 if success else 1)
    elif args.profile:
        run_profile(PROFILE_SHAPES, warmup=args.warmup)
    elif args.benchmark:
        run_benchmark(HARNESS_SHAPES, warmup=args.warmup,
                      iterations=args.iterations)
    elif args.full_benchmark:
        run_benchmark(ALL_SHAPES, warmup=args.warmup,
                      iterations=args.iterations)
    else:        run_benchmark(HARNESS_SHAPES, warmup=args.warmup,                      iterations=args.iterations)


if __name__ == "__main__":
    main()
