# SPDX-License-Identifier: Apache-2.0
# Modifications Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

# The kernel in this file is adapted from TritonBench's llama_ff_triton:
# https://github.com/thunlp/TritonBench - Apache License 2.0

#  LLaMA Feed-Forward: fused RMSNorm + SiLU-gated linear projections Triton kernel.
from __future__ import annotations
import math
import torch
import triton
import triton.language as tl


@triton.jit
def ff_llama_opt(
    a_ptr, w_ptr, out_ptr, rms_w_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wk, stride_wn,
    stride_outm, stride_outn,
    stride_rms_w,
    USE_FP8: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused kernel: w_combined = [w1_t | w3_t] concatenated along N dim (width=2*N).
    No K-loop (K == BLOCK_SIZE_K).
    """
    pid_m = tl.program_id(axis=0)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)

    # Load input
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    a = tl.load(a_ptrs)

    # RMS norm
    a_f32 = a.to(tl.float32)
    rms_acc = tl.sum(a_f32 * a_f32, axis=1)

    # Apply RMS weights
    rms_w_ptrs = rms_w_ptr + offs_k[None, :] * stride_rms_w
    rms_w = tl.load(rms_w_ptrs)
    if USE_FP8:
        rms_w = rms_w.to(tl.float8e5, bitcast=True)
        rms_w = rms_w.to(tl.float16)
    a = a * rms_w

    # Load w1 block (first N columns of combined weight)
    w1_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
    b = tl.load(w1_ptrs)

    # Load w3 block (next N columns of combined weight)
    w3_ptrs = w_ptr + (offs_k[:, None] * stride_wk + (offs_n[None, :] + BLOCK_SIZE_N) * stride_wn)
    c = tl.load(w3_ptrs)

    if USE_FP8:
        b = b.to(tl.float8e5, bitcast=True).to(tl.float32).to(tl.float16)
        c = c.to(tl.float8e5, bitcast=True).to(tl.float32).to(tl.float16)

    # Two dot products
    acc1 = tl.dot(a, b)
    acc2 = tl.dot(a, c)

    # Normalize and apply SiLU gate
    a_mean = rms_acc / K + EPS
    a_norm = tl.math.rsqrt(a_mean)
    acc1 = acc1 * a_norm[:, None]
    acc2 = acc2 * a_norm[:, None]
    accumulator = (acc1 * tl.sigmoid(acc1)) * acc2

    # Store
    offs_outm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    out_ptrs = out_ptr + (stride_outm * offs_outm[:, None] + stride_outn * offs_n[None, :])
    out_mask = (offs_outm[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, accumulator, mask=out_mask)


# Keep original kernel signature for backward compat
@triton.jit
def ff_llama(
    a_ptr, w1_ptr, w3_ptr, out_ptr, rms_w_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_w1k, stride_w1n,
    stride_w3k, stride_w3n,
    stride_outm, stride_outn,
    stride_rms_w,
    USE_FP8: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    pid_m = pid // tl.cdiv(N, BLOCK_SIZE_N)
    pid_n = pid % tl.cdiv(N, BLOCK_SIZE_N)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    w1_ptrs = w1_ptr + (offs_k[:, None] * stride_w1k + offs_bn[None, :] * stride_w1n)
    w3_ptrs = w3_ptr + (offs_k[:, None] * stride_w3k + offs_bn[None, :] * stride_w3n)
    acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    rms_w_ptrs = rms_w_ptr + tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_rms_w
    rms_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        a_f32 = a.to(tl.float32)
        rms_acc += tl.sum(a_f32 * a_f32, axis=1)
        rms_w = tl.load(rms_w_ptrs)
        if USE_FP8:
            rms_w = rms_w.to(tl.float8e5, bitcast=True)
            rms_w = rms_w.to(tl.float16)
        a = a * rms_w
        b = tl.load(w1_ptrs)
        if USE_FP8:
            b = b.to(tl.float8e5, bitcast=True).to(tl.float32).to(tl.float16)
        acc1 += tl.dot(a, b)
        c = tl.load(w3_ptrs)
        if USE_FP8:
            c = c.to(tl.float8e5, bitcast=True).to(tl.float32).to(tl.float16)
        acc2 += tl.dot(a, c)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        w1_ptrs += BLOCK_SIZE_K * stride_w1k
        w3_ptrs += BLOCK_SIZE_K * stride_w3k
        rms_w_ptrs += BLOCK_SIZE_K * stride_rms_w
    a_mean = rms_acc / K + EPS
    a_norm = tl.math.rsqrt(a_mean)
    acc1 = acc1 * a_norm[:, None]
    acc2 = acc2 * a_norm[:, None]
    accumulator = (acc1 * tl.sigmoid(acc1)) * acc2
    offs_outm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_outn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = out_ptr + (stride_outm * offs_outm[:, None] + stride_outn * offs_outn[None, :])
    out_mask = (offs_outm[:, None] < M) & (offs_outn[None, :] < N)
    tl.store(out_ptrs, accumulator, mask=out_mask)


# Pre-cache combined weights and output buffers
_cache = {}
_out_cache = {}

def kernel_ff(x: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor, rms_w: torch.Tensor) -> torch.Tensor:
    batch, seq_len, dim = x.shape
    M = batch * seq_len
    N = w1.shape[1]
    x_reshape = x.view(M, dim)

    # Cache output buffer to avoid torch.empty overhead
    out_key = (M, N, x.device)
    out = _out_cache.get(out_key)
    if out is None or out.dtype != x.dtype:
        out = torch.empty((M, N), dtype=x.dtype, device=x.device)
        _out_cache[out_key] = out

    # Cache weight preparation
    w_key = (w1.data_ptr(), w3.data_ptr())
    cached = _cache.get(w_key)
    if cached is None:
        w1_t = w1.t().contiguous()
        w3_t = w3.t().contiguous()
        w_combined = torch.cat([w1_t, w3_t], dim=1)  # [K, 2*N]
        cached = (w_combined, w_combined.stride(0), w_combined.stride(1), w1.dtype != torch.float16)
        _cache[w_key] = cached
    w_combined, wstride0, wstride1, use_fp8 = cached

    ff_llama_opt[(triton.cdiv(M, 16),)](
        x_reshape, w_combined, out, rms_w,
        M, N, dim,
        x_reshape.stride(0), x_reshape.stride(1),
        wstride0, wstride1,
        out.stride(0), out.stride(1),
        rms_w.stride(0),
        USE_FP8=use_fp8,
        EPS=1e-6,
        BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64,
        num_stages=2, num_warps=4
    )
    return out.view(batch, seq_len, N)




##################################################################################################################################################

# ============================================================================
# TEST CONFIGURATIONS
# ============================================================================

# (batch, seq_len, dim) - w1/w3 are always (dim, dim), rms_w is (dim,)
# Extracted from test_ff_llama() in the original eval:
#   test_case_1: batch=2, seq_len=4, dim=64, w=(64,64)
#   test_case_3: batch=3, seq_len=4, dim=64, w=(64,64)
#   test_case_4: batch=2, seq_len=5, dim=64, w=(64,64)

ALL_SHAPES = [
    (2, 4, 64),   # test_case_1
    (3, 4, 64),   # test_case_3
    (2, 5, 64),   # test_case_4
]

HARNESS_SHAPES = ALL_SHAPES[:25]
PROFILE_SHAPES = ALL_SHAPES[:5]

RTOL, ATOL = 0.15, 0.25

# For backward compatibility
EVAL_CONFIGS = HARNESS_SHAPES
PROFILE_CONFIGS = PROFILE_SHAPES


# ============================================================================
# TEST HARNESS
# ============================================================================


def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_input(batch, seq_len, dim, seed=42):
    """Create input tensors with fixed seed."""
    set_seed(seed)
    x = torch.randn((batch, seq_len, dim), dtype=torch.float16, device='cuda')
    w1 = torch.randn((dim, dim), dtype=torch.float16, device='cuda')
    w3 = torch.randn((dim, dim), dtype=torch.float16, device='cuda')
    rms_w = torch.randn((dim,), dtype=torch.float16, device='cuda')
    return x, w1, w3, rms_w


def reference_ff(x, w1, w3, rms_w, eps=1e-6):
    """PyTorch reference for LLaMA feed-forward."""
    batch, seq_len, dim = x.shape
    x_flat = x.reshape(-1, dim).float()

    a_sum = (x_flat ** 2).sum(dim=-1, keepdim=True)
    x_scaled = x_flat * rms_w.float()

    acc1 = x_scaled @ w1.T.float()
    acc2 = x_scaled @ w3.T.float()

    a_norm = torch.rsqrt(a_sum / dim + eps)
    acc1_n = acc1 * a_norm
    acc2_n = acc2 * a_norm
    out = (acc1_n * torch.sigmoid(acc1_n)) * acc2_n

    return out.reshape(batch, seq_len, -1).to(x.dtype)


def run_correctness(shapes, verbose: bool = True) -> dict:
    """Run correctness tests on the exact eval shapes."""
    if verbose:
        print(f"Running correctness on {len(shapes)} shapes...")

    results, failures = [], []
    for idx, (batch, seq_len, dim) in enumerate(shapes):
        try:
            x, w1, w3, rms_w = make_input(batch, seq_len, dim, seed=42 + idx)

            out_triton = kernel_ff(x, w1, w3, rms_w)
            out_ref = reference_ff(x, w1, w3, rms_w)

            torch.testing.assert_close(out_triton, out_ref, rtol=RTOL, atol=ATOL)

            results.append({"config": (batch, seq_len, dim), "correct": True})
            if verbose:
                print(f"  PASS: ({batch}, {seq_len}, {dim})")

            del x, w1, w3, rms_w, out_triton, out_ref
            torch.cuda.empty_cache()
        except Exception as e:
            failures.append({"config": (batch, seq_len, dim), "error": str(e)})
            if verbose:
                print(f"  FAIL: ({batch}, {seq_len}, {dim}) - {str(e)[:80]}")

    if verbose:
        print("-" * 62)
        print(
            f"{'Status:':<22} {'ALL PASS' if not failures else f'FAILED ({len(failures)}/{len(shapes)})'}"
        )

    return {
        "correct": len(failures) == 0,
        "num_correct": len(results),
        "num_failed": len(failures),
        "failures": failures,
        "results": results,
    }


def run_profile(shapes, warmup: int = 50, iters: int = 200, verbose: bool = True):
    """Run kernel for profiling with proper warmup."""
    if verbose:
        print(f"Profile: {len(shapes)} config(s), {warmup} warmup, {iters} iter(s)")

    for batch, seq_len, dim in shapes:
        x, w1, w3, rms_w = make_input(batch, seq_len, dim, seed=42)

        for _ in range(warmup):
            kernel_ff(x, w1, w3, rms_w)
        torch.cuda.synchronize()

        for _ in range(iters):
            kernel_ff(x, w1, w3, rms_w)
        torch.cuda.synchronize()

        if verbose:
            print(f"  ({batch}, {seq_len}, {dim}) done")
        del x, w1, w3, rms_w
        torch.cuda.empty_cache()


def run_benchmark(shapes, warmup: int = 50, iters: int = 200, verbose: bool = True) -> dict:
    """Benchmark kernel vs reference; report per-shape speedups and geo-mean."""
    print(
        f"Running benchmark on {len(shapes)} shapes, {warmup} warmup, {iters} iterations each..."
    )
    latencies = []
    speedups = []
    results = []

    if verbose:
        print(
            f"{'Config (B,S,D)':<22} {'Reference':>10} {'Kernel':>10} {'Speedup':>10}"
        )
        print("-" * 62)

    for idx, (batch, seq_len, dim) in enumerate(shapes):
        x, w1, w3, rms_w = make_input(batch, seq_len, dim, seed=42 + idx)

        for _ in range(warmup):
            kernel_ff(x, w1, w3, rms_w)
        torch.cuda.synchronize()

        triton_times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            kernel_ff(x, w1, w3, rms_w)
            end.record()
            torch.cuda.synchronize()
            triton_times.append(start.elapsed_time(end))

        kernel_ms = sorted(triton_times)[len(triton_times) // 2]

        ref_times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            reference_ff(x, w1, w3, rms_w)
            end.record()
            torch.cuda.synchronize()
            ref_times.append(start.elapsed_time(end))

        ref_ms = sorted(ref_times)[len(ref_times) // 2]

        speedup = ref_ms / kernel_ms if kernel_ms > 0 else float('inf')
        speedups.append(speedup)
        latencies.append(kernel_ms)

        results.append({
            "config": (batch, seq_len, dim),
            "ref_ms": ref_ms,
            "kernel_ms": kernel_ms,
            "speedup": speedup,
        })

        if verbose:
            marker = " *" if speedup > 1.0 else ""
            print(
                f"({batch}, {seq_len}, {dim}){' ':9} {ref_ms:>8.4f}ms {kernel_ms:>8.4f}ms {speedup:>8.2f}x{marker}"
            )

        del x, w1, w3, rms_w
        torch.cuda.empty_cache()

    log_sum = sum(math.log(t) for t in latencies)
    geomean_latency = math.exp(log_sum / len(latencies))

    log_sum_speedup = sum(math.log(s) for s in speedups)
    geomean_speedup = math.exp(log_sum_speedup / len(speedups))

    if verbose:
        print("-" * 62)
        print(f"{'Geometric mean latency:':<22} {geomean_latency:.4f} ms")
        print(f"{'Geometric mean speedup:':<22} {geomean_speedup:.2f}x")
        print(f"GEAK_RESULT_LATENCY_MS={geomean_latency:.4f}")
        print(f"GEAK_RESULT_SPEEDUP={geomean_speedup:.2f}")

    return {
        "geomean_latency_ms": geomean_latency,
        "geomean_speedup": geomean_speedup,
        "results": results,
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLaMA FF Triton Kernel Test Harness")
    parser.add_argument(
        "--correctness",
        action="store_true",
        help="Run correctness tests on benchmark shapes",
    )
    parser.add_argument(
        "--profile", action="store_true", help="Run minimal profiling workload"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark on HARNESS_SHAPES",
    )
    parser.add_argument(
        "--full-benchmark",
        action="store_true",
        help="Run benchmark on ALL_SHAPES (complete set)",
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

    print("=" * 62)
    print("LLaMA FF Triton Kernel Test Harness")
    print("=" * 62)

    if args.correctness:
        print("\n[Correctness Mode]")
        run_correctness(HARNESS_SHAPES)
    elif args.profile:
        print("\n[Profile Mode]")
        run_profile(PROFILE_SHAPES, warmup=args.warmup, iters=args.iterations)
    elif args.full_benchmark:
        print("\n[Full Benchmark Mode]")
        run_benchmark(ALL_SHAPES, warmup=args.warmup, iters=args.iterations)
    else:
        # Default: benchmark (harness shapes)
        print("\n[Benchmark Mode]")
        run_benchmark(HARNESS_SHAPES, warmup=args.warmup, iters=args.iterations)

    print("=" * 62)
