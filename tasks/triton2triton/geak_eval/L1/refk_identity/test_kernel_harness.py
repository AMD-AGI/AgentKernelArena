#!/usr/bin/env python3
"""
Test harness for the identity copy Triton kernel.

Modes: --correctness, --profile, --benchmark, --full-benchmark
"""
import argparse
import math
import os
import sys
import torch

from kernel import identity_triton, get_inputs, EVAL_CONFIGS, PROFILE_CONFIGS, RTOL, ATOL


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

def identity_pytorch(input_tensor, output_tensor):
    output_tensor[...] = input_tensor
    return output_tensor


# ============================================================================
# TEST HARNESS
# ============================================================================

def run_correctness(shapes=None, verbose=True):
    if shapes is None:
        shapes = HARNESS_SHAPES
    if verbose:
        print(f"Running correctness on {len(shapes)} shapes...")

    results, failures = [], []

    for cfg in shapes:
        try:
            data, out_triton = get_inputs(**cfg)
            out_ref = torch.empty_like(data)
            identity_triton(data, out_triton)
            identity_pytorch(data, out_ref)
            torch.cuda.synchronize()

            torch.testing.assert_close(out_triton, out_ref, atol=ATOL, rtol=RTOL)
            results.append({"config": cfg, "correct": True})
            if verbose:
                print(f"  PASS: size={cfg['size']}")
            del data, out_triton, out_ref
            torch.cuda.empty_cache()
        except Exception as e:
            failures.append({"config": cfg, "error": str(e)})
            if verbose:
                print(f"  FAIL: size={cfg['size']} - {str(e)[:50]}")

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
        data, output = get_inputs(**cfg)
        for _ in range(warmup):
            identity_triton(data, output)
        torch.cuda.synchronize()
        for _ in range(iters):
            identity_triton(data, output)
        torch.cuda.synchronize()
        if verbose:
            print(f"  size={cfg['size']} done")
        del data, output
        torch.cuda.empty_cache()


def run_benchmark(shapes=None, warmup=50, iters=200, verbose=True):
    if shapes is None:
        shapes = HARNESS_SHAPES

    latencies = []

    print(f"Running benchmark on {len(shapes)} shapes, {warmup} warmup, {iters} iterations each...")
    if verbose:
        print(f"{'Config':<22} {'Triton':>10}")
        print("-" * 34)

    for cfg in shapes:
        data, output = get_inputs(**cfg)

        for _ in range(warmup):
            identity_triton(data, output)
        torch.cuda.synchronize()

        triton_times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            identity_triton(data, output)
            end.record()
            torch.cuda.synchronize()
            triton_times.append(start.elapsed_time(end))

        triton_ms = sorted(triton_times)[len(triton_times) // 2]
        latencies.append(triton_ms)

        if verbose:
            print(f"size={cfg['size']:<14} {triton_ms:>8.4f}ms", flush=True)

        del data, output
        torch.cuda.empty_cache()

    geomean_latency = math.exp(sum(math.log(l) for l in latencies) / len(latencies))

    print("-" * 34)
    print(f"{'Geometric mean latency:':<22} {geomean_latency:.4f} ms")
    print(f"GEAK_SHAPES_USED={list(range(len(shapes)))}")
    print(f"GEAK_RESULT_LATENCY_MS={geomean_latency:.4f}", flush=True)

    return {"geomean_latency_ms": geomean_latency, "latencies": latencies}


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identity Kernel Test Harness")
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
    print("Identity Kernel Test Harness")
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
