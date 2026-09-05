#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Test harness for gemm_a16w16 kernel

import os
import sys
from _aka_benchmark import benchmark_cuda_graph_or_events_samples


def benchmark_cuda_graph_or_events(*args, **kwargs):
    samples, metadata = benchmark_cuda_graph_or_events_samples(*args, **kwargs)
    values = sorted(samples)
    midpoint = len(values) // 2
    median_ms = (
        values[midpoint]
        if len(values) % 2
        else (values[midpoint - 1] + values[midpoint]) / 2.0
    )
    return median_ms, metadata

# Only set GPU visibility if explicitly requested via GEAK_GPU_DEVICE.
# Don't override HIP_VISIBLE_DEVICES if it's already set by the caller
# (e.g., GEAK's parallel GPU scheduler).
_gpu = os.environ.get("GEAK_GPU_DEVICE")
if _gpu is not None:
    os.environ["HIP_VISIBLE_DEVICES"] = _gpu
    os.environ["ROCR_VISIBLE_DEVICES"] = _gpu

import argparse
import math

# kernel.py lives next to this harness; Python puts the script dir on sys.path[0].
_HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
if _HARNESS_DIR not in sys.path:
    sys.path.insert(0, _HARNESS_DIR)

import torch
import torch.nn.functional as F
import triton

from kernel import gemm_a16w16

# ---------------------------------------------------------------------------
# Config list: from bench_gemm_a16w16.py -> benchmark_utils.get_x_vals(dims=3)
# This is the authoritative ordered full case stream.
# ---------------------------------------------------------------------------
ALL_CONFIGS = [
    (1, 1280, 8192),
    (32, 1280, 8192),
    (64, 1280, 8192),
    (128, 1280, 8192),
    (192, 1280, 8192),
    (256, 1280, 8192),
    (320, 1280, 8192),
    (512, 1280, 8192),
    (1024, 1280, 8192),
    (2048, 1280, 8192),
    (4096, 1280, 8192),
    (8192, 1280, 8192),
    (16384, 1280, 8192),
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WARMUP = 50
ITERATIONS = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200"))
DTYPE = torch.bfloat16


def _pick(configs, count):
    """Deterministic uniform subsetting."""
    if len(configs) <= count:
        return list(range(len(configs)))
    n = len(configs)
    return [round(i * (n - 1) / (count - 1)) for i in range(count)]


def _format_config(cfg):
    M, N, K = cfg
    return "M={} N={} K={}".format(M, N, K)


# ---------------------------------------------------------------------------
# Input generation (inlined from aiter's test_gemm_a16w16.generate_gemm_a16w16_inputs)
# ---------------------------------------------------------------------------
def _generate_inputs(M, N, K, dtype):
    x = torch.randn((M, K), dtype=dtype, device="cuda")
    w = torch.randn((K, N), dtype=dtype, device="cuda").T
    # Initialize bias with real values: aiter's generator uses torch.empty but
    # defaults bias=False, so its uninitialized path is never exercised. This
    # harness always adds bias, so uninitialized memory could carry a NaN bit
    # pattern that makes assert_close flag NaN positions (NaN != NaN) even
    # though kernel and reference propagate it identically.
    bias = torch.randn((N,), dtype=dtype, device="cuda")
    return x, w, bias


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------
def run_correctness(indices):
    torch.manual_seed(42)
    print("Running correctness checks...")
    all_pass = True
    for idx in indices:
        M, N, K = ALL_CONFIGS[idx]
        x, w, bias = _generate_inputs(M, N, K, DTYPE)
        torch_out = F.linear(x, w, bias=bias)
        triton_out = gemm_a16w16(x, w, bias)
        try:
            torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)
            print("  [{}] {}  PASS".format(idx, _format_config(ALL_CONFIGS[idx])))
        except AssertionError as e:
            print("  [{}] {}  FAIL: {}".format(idx, _format_config(ALL_CONFIGS[idx]), e))
            all_pass = False
        del x, w, bias, torch_out, triton_out
        torch.cuda.empty_cache()

    print("GEAK_SHAPES_USED={}".format(indices))
    if not all_pass:
        print("CORRECTNESS FAILED")
        sys.exit(1)
    print("ALL CORRECTNESS CHECKS PASSED")


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def run_benchmark(indices):
    torch.manual_seed(42)
    print("Running benchmark...")
    latencies = []
    methods = []
    for idx in indices:
        M, N, K = ALL_CONFIGS[idx]
        x, w, bias = _generate_inputs(M, N, K, DTYPE)
        ms, metadata = benchmark_cuda_graph_or_events(
            lambda: gemm_a16w16(x, w, bias),
            warmup=WARMUP,
            repetition=ITERATIONS,
        )
        latencies.append(ms)
        methods.append(metadata["benchmark_method"])
        print("  [{}] {}  {:.4f}ms".format(idx, _format_config(ALL_CONFIGS[idx]), ms))
        del x, w, bias
        torch.cuda.empty_cache()

    # Geometric mean
    log_sum = sum(math.log(lat) for lat in latencies)
    geo_mean = math.exp(log_sum / len(latencies))

    print("GEAK_SHAPES_USED={}".format(indices))
    print("GEAK_RESULT_LATENCY_MS={:.4f}".format(geo_mean))
    print("GEAK_BENCHMARK_METHOD={}".format(
        methods[0] if len(set(methods)) == 1 else "mixed:" + ",".join(sorted(set(methods)))
    ))


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------
def run_profile(indices):
    torch.manual_seed(42)
    print("Running profile mode...")
    for idx in indices:
        M, N, K = ALL_CONFIGS[idx]
        x, w, bias = _generate_inputs(M, N, K, DTYPE)
        gemm_a16w16(x, w, bias)
        torch.cuda.synchronize()
        print("  [{}] {}  profiled".format(idx, _format_config(ALL_CONFIGS[idx])))
        del x, w, bias
        torch.cuda.empty_cache()

    print("GEAK_SHAPES_USED={}".format(indices))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Test harness for gemm_a16w16")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--correctness", action="store_true")
    group.add_argument("--benchmark", action="store_true")
    group.add_argument("--full-benchmark", action="store_true")
    group.add_argument("--profile", action="store_true")
    parser.add_argument("--iterations", type=int, default=None, help="Number of benchmark iterations (overrides GEAK_BENCHMARK_ITERATIONS env var)")
    args = parser.parse_args()
    if args.iterations is not None:
        global ITERATIONS
        ITERATIONS = args.iterations

    if args.correctness:
        indices = list(range(len(ALL_CONFIGS)))
        run_correctness(indices)
    elif args.benchmark:
        indices = list(range(len(ALL_CONFIGS)))  # use all configs so benchmark matches full-benchmark
        run_benchmark(indices)
    elif args.full_benchmark:
        indices = list(range(len(ALL_CONFIGS)))
        run_benchmark(indices)
    elif args.profile:
        indices = _pick(ALL_CONFIGS, 5)
        run_profile(indices)


if __name__ == "__main__":
    main()
