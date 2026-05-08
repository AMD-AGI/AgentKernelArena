#!/usr/bin/env python3
"""
Test harness for the FP8 block-scale GEMM Triton kernel.

Modes: --correctness, --profile, --benchmark, --full-benchmark
"""
import argparse
import math
import os
import sys
import torch

from kernel import (
    fp8_blockwise_mm_triton, get_inputs,
    EVAL_CONFIGS, PROFILE_CONFIGS,
    BLOCK_SHAPE_N, BLOCK_SHAPE_K,
    RTOL, ATOL,
)


# ============================================================================
# SHAPE SUBSETS
# ============================================================================

ALL_SHAPES = EVAL_CONFIGS

_n_all = len(ALL_SHAPES)
if _n_all <= 25:
    HARNESS_SHAPES = ALL_SHAPES
else:
    _harness_indices = [int(round(i * (_n_all - 1) / 24)) for i in range(25)]
    HARNESS_SHAPES = [ALL_SHAPES[i] for i in _harness_indices]

# PROFILE_CONFIGS already provides the profile subset (3 entries) per kernel.py.
PROFILE_SHAPES = PROFILE_CONFIGS


# ============================================================================
# PYTORCH REFERENCE (moved from kernel.py; correctness-only)
# ============================================================================

def fp8_blockwise_mm_pytorch(a, b, a_scale, b_scale, c):
    a_c = a.contiguous()
    a_s = a_scale.contiguous()
    b_s = b_scale.contiguous()

    m, k = a_c.shape
    n = b.shape[0]
    block_n, block_k = BLOCK_SHAPE_N, BLOCK_SHAPE_K
    sn = b_s.shape[0]
    sk = b_s.shape[1]

    a_sc = a_s.unsqueeze(-1).repeat(1, 1, block_k).reshape(m, sk * block_k)[:, :k]
    a_deq = a_c.to(a_sc.dtype) * a_sc

    b_sc = (b_s.view(-1, 1).repeat(1, block_n * block_k)
            .view(sn, sk, block_n, block_k)
            .permute(0, 2, 1, 3)
            .reshape(sn * block_n, sk * block_k))[:n, :k]
    b_deq = b.to(b_sc.dtype) * b_sc

    c[...] = (a_deq @ b_deq.T).to(torch.bfloat16)
    return c


# ============================================================================
# TEST HARNESS
# ============================================================================

def _label(cfg):
    return f"M={cfg['m']:>5}, N={cfg['n']:>5}, K={cfg['k']:>5}"


def run_correctness(shapes=None, verbose=True):
    if shapes is None:
        shapes = HARNESS_SHAPES
    if verbose:
        print(f"Running correctness on {len(shapes)} shapes...")

    results, failures = [], []

    for cfg in shapes:
        try:
            a, b, a_scale, b_scale, c_triton = get_inputs(**cfg)
            c_ref = c_triton.clone()

            fp8_blockwise_mm_triton(a, b, a_scale, b_scale, c_triton)
            fp8_blockwise_mm_pytorch(a, b, a_scale, b_scale, c_ref)
            torch.cuda.synchronize()

            torch.testing.assert_close(c_triton.float(), c_ref.float(), atol=ATOL, rtol=RTOL)
            results.append({"config": cfg, "correct": True})
            if verbose:
                print(f"  PASS: {_label(cfg)}")
            del a, b, a_scale, b_scale, c_triton, c_ref
            torch.cuda.empty_cache()
        except Exception as e:
            failures.append({"config": cfg, "error": str(e)})
            if verbose:
                print(f"  FAIL: {_label(cfg)} - {str(e)[:80]}")

    if verbose:
        print("-" * 62)
        status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{len(shapes)})"
        print(f"{'Status:':<22} {status}")

    return {
        "correct": len(failures) == 0,
        "num_correct": len(results),
        "num_failed": len(failures),
        "failures": failures,
        "results": results,
    }


def run_profile(shapes=None, warmup=50, iters=200, verbose=True):
    if shapes is None:
        shapes = PROFILE_SHAPES
    if verbose:
        print(f"Profile: {len(shapes)} config(s), {warmup} warmup, {iters} iter(s)")

    for cfg in shapes:
        a, b, a_scale, b_scale, c = get_inputs(**cfg)
        for _ in range(warmup):
            fp8_blockwise_mm_triton(a, b, a_scale, b_scale, c)
        torch.cuda.synchronize()
        for _ in range(iters):
            fp8_blockwise_mm_triton(a, b, a_scale, b_scale, c)
        torch.cuda.synchronize()
        if verbose:
            print(f"  {_label(cfg)} done")
        del a, b, a_scale, b_scale, c
        torch.cuda.empty_cache()


def run_benchmark(shapes=None, warmup=50, iters=200, verbose=True):
    if shapes is None:
        shapes = HARNESS_SHAPES

    latencies = []

    print(f"Running benchmark on {len(shapes)} shapes, {warmup} warmup, {iters} iterations each...")
    if verbose:
        print(f"{'Config':<32} {'Triton':>10}")
        print("-" * 44)

    for cfg in shapes:
        # NOTE: c is mutated in place by the kernel. Allocate inputs ONCE per cfg
        # OUTSIDE the timed loop and reuse them; the OLD benchmark_config did
        # c.clone() per iteration, which incorrectly timed allocation.
        a, b, a_scale, b_scale, c = get_inputs(**cfg)

        for _ in range(warmup):
            fp8_blockwise_mm_triton(a, b, a_scale, b_scale, c)
        torch.cuda.synchronize()

        triton_times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fp8_blockwise_mm_triton(a, b, a_scale, b_scale, c)
            end.record()
            torch.cuda.synchronize()
            triton_times.append(start.elapsed_time(end))

        triton_ms = sorted(triton_times)[len(triton_times) // 2]
        latencies.append(triton_ms)

        if verbose:
            print(f"{_label(cfg):<32} {triton_ms:>8.4f}ms", flush=True)

        del a, b, a_scale, b_scale, c
        torch.cuda.empty_cache()

    geomean_latency = math.exp(sum(math.log(l) for l in latencies) / len(latencies))

    print("-" * 44)
    print(f"{'Geometric mean latency:':<22} {geomean_latency:.4f} ms")
    print(f"GEAK_SHAPES_USED={list(range(len(shapes)))}")
    print(f"GEAK_RESULT_LATENCY_MS={geomean_latency:.4f}", flush=True)

    return {"geomean_latency_ms": geomean_latency, "latencies": latencies}


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FP8 Block-Scale GEMM Test Harness")
    parser.add_argument("--correctness", action="store_true",
                        help="Run correctness tests on HARNESS_SHAPES")
    parser.add_argument("--profile", action="store_true",
                        help="Run minimal profiling workload")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark on HARNESS_SHAPES (25 uniformly sampled)")
    parser.add_argument("--full-benchmark", action="store_true",
                        help="Run benchmark on ALL_SHAPES (complete set)")
    parser.add_argument("--warmup", type=int, default=50,
                        help="Number of warmup iterations (default: 50)")
    parser.add_argument("--iterations", type=int,
                        default=int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200")),
                        help="Number of benchmark iterations (default: GEAK_BENCHMARK_ITERATIONS or 200)")
    args = parser.parse_args()

    print("=" * 62)
    print("FP8 Block-Scale GEMM Test Harness")
    print("=" * 62)

    if args.correctness:
        print("\n[Correctness Mode]")
        result = run_correctness(HARNESS_SHAPES)
        sys.exit(0 if result["correct"] else 1)
    elif args.profile:
        print("\n[Profile Mode]")
        run_profile(PROFILE_SHAPES, warmup=args.warmup, iters=args.iterations)
    elif args.full_benchmark:
        print("\n[Full Benchmark Mode]")
        run_benchmark(ALL_SHAPES, warmup=args.warmup, iters=args.iterations)
    else:
        print("\n[Benchmark Mode]")
        run_benchmark(HARNESS_SHAPES, warmup=args.warmup, iters=args.iterations)

    print("=" * 62)
