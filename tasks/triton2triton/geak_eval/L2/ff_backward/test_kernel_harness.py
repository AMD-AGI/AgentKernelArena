#!/usr/bin/env python3
"""
Test harness for ff_backward (SwiGLU fused gated backward) kernel.

Modes:
  --correctness      : validate Triton backward against PyTorch reference
  --benchmark        : benchmark on HARNESS_CONFIGS, report GEAK_RESULT_LATENCY_MS
  --full-benchmark   : benchmark on ALL configs, report GEAK_RESULT_LATENCY_MS
  --profile          : run 3 configs for profiler capture
  --iterations N     : override iteration count (default from GEAK_BENCHMARK_ITERATIONS or 200)
"""
import argparse
import math
import os
import sys
import time

import torch

# Ensure kernel.py is importable
_harness_dir = os.path.dirname(os.path.abspath(__file__))
if _harness_dir not in sys.path:
    sys.path.insert(0, _harness_dir)

from kernel import (
    EVAL_CONFIGS,
    check_correctness,
    benchmark_config,
    triton_op,
    get_inputs,
)

# ── Config space ────────────────────────────────────────────────────────────
ALL_CONFIGS = EVAL_CONFIGS
HARNESS_CONFIGS = ALL_CONFIGS  # use all configs so benchmark matches full-benchmark
PROFILE_CONFIGS = ALL_CONFIGS[:3]


def _pick(configs, count):
    if len(configs) <= count:
        return list(range(len(configs)))
    n = len(configs)
    return [round(i * (n - 1) / (count - 1)) for i in range(count)]


# ── Correctness ────────────────────────────────────────────────────────────
def run_correctness(configs, indices):
    print(f"Running correctness on {len(indices)} configs...")
    all_passed = True
    for idx in indices:
        M, N, K = configs[idx]
        result = check_correctness(M, K, N)
        if result["correct"]:
            print(f"  PASS config[{idx}] M={M} N={N} K={K}")
        else:
            err = result.get("error", f"rel_dx={result.get('rel_dx', '?')}")
            print(f"  FAIL config[{idx}] M={M} N={N} K={K}: {err}")
            all_passed = False
    print(f"GEAK_SHAPES_USED={indices}")
    if all_passed:
        print("ALL CORRECTNESS CHECKS PASSED")
        return 0
    print("CORRECTNESS FAILED")
    return 1


# ── Benchmark ──────────────────────────────────────────────────────────────
def run_benchmark(configs, indices, warmup=50, iters=200):
    print(f"Running benchmark on {len(indices)} configs...")
    latencies = []
    for idx in indices:
        M, N, K = configs[idx]
        result = benchmark_config(M, K, N, warmup=warmup, iters=iters)
        lat = result.get("triton_ms", 0)
        latencies.append(lat)
        print(f"  M={M} N={N} K={K}  {lat:.4f}ms")

    valid = [l for l in latencies if l > 0]
    if valid:
        geo_mean = math.exp(sum(math.log(l) for l in valid) / len(valid))
    else:
        geo_mean = 0.0
    print(f"GEAK_SHAPES_USED={indices}")
    print(f"GEAK_RESULT_LATENCY_MS={geo_mean:.4f}")
    return 0


# ── Profile ────────────────────────────────────────────────────────────────
def run_profile(configs, indices):
    print(f"Running profile on {len(indices)} configs...")
    for idx in indices:
        M, N, K = configs[idx]
        x, w_up, w_down, dy = get_inputs(M, K, N)
        # Warmup
        for _ in range(3):
            triton_op(M, N, K, x, w_up, w_down, dy)
        torch.cuda.synchronize()
        # One profiled run
        triton_op(M, N, K, x, w_up, w_down, dy)
        torch.cuda.synchronize()
        print(f"  M={M} N={N} K={K} done")
    return 0


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    default_iters = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200"))

    parser = argparse.ArgumentParser(description="ff_backward test harness")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--correctness", action="store_true")
    group.add_argument("--benchmark", action="store_true")
    group.add_argument("--full-benchmark", action="store_true")
    group.add_argument("--profile", action="store_true")
    parser.add_argument("--iterations", type=int, default=default_iters)
    parser.add_argument("--warmup", type=int, default=50)
    args = parser.parse_args()

    if args.correctness:
        indices = _pick(ALL_CONFIGS, 25)
        sys.exit(run_correctness(ALL_CONFIGS, indices))
    elif args.benchmark:
        indices = _pick(HARNESS_CONFIGS, 25)
        sys.exit(run_benchmark(HARNESS_CONFIGS, indices, args.warmup, args.iterations))
    elif args.full_benchmark:
        indices = list(range(len(ALL_CONFIGS)))
        sys.exit(run_benchmark(ALL_CONFIGS, indices, args.warmup, args.iterations))
    elif args.profile:
        indices = list(range(len(PROFILE_CONFIGS)))
        sys.exit(run_profile(PROFILE_CONFIGS, indices))


if __name__ == "__main__":
    main()
