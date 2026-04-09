#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Test harness for gemm_a16w16_atomic kernel

import argparse
import os
import sys
import math

import torch
import torch.nn.functional as F
import triton

# ---------------------------------------------------------------------------
# Resolve imports: prefer GEAK_WORK_DIR, then GEAK_REPO_ROOT + kernel subdir,
# then the original kernel directory.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.environ.get(
    "GEAK_WORK_DIR",
    os.environ.get(
        "GEAK_REPO_ROOT",
        os.path.dirname(os.path.abspath(__file__)),
    ),
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ── Dynamic kernel.py loader (matches old kernel pattern) ──────────────────
import importlib.util
import types

def _resolve_geak_kernel_dir():
    candidates = []
    work_dir = os.environ.get("GEAK_WORK_DIR", "").strip()
    if work_dir:
        candidates.append(work_dir)
    repo_root = os.environ.get("GEAK_REPO_ROOT", "").strip()
    if repo_root:
        candidates.append(os.path.join(repo_root, '.'))
    original_kernel_dir = os.path.dirname(os.path.abspath(__file__))
    if original_kernel_dir:
        candidates.append(original_kernel_dir)
    for candidate in candidates:
        if candidate and os.path.isfile(os.path.join(candidate, "kernel.py")):
            return candidate
    return original_kernel_dir or os.getcwd()

def _ensure_geak_package(module_name):
    parts = module_name.split(".")
    for idx in range(1, len(parts)):
        prefix = ".".join(parts[:idx])
        if prefix in sys.modules:
            continue
        pkg = types.ModuleType(prefix)
        pkg.__path__ = []
        sys.modules[prefix] = pkg

def _register_geak_aliases(kernel_dir):
    aliases = ['gemm_a16w16_atomic', 'aiter.ops.triton.gemm_a16w16_atomic']
    entry_file = os.path.join(kernel_dir, "kernel.py")
    if not os.path.isfile(entry_file):
        return
    for alias in aliases:
        if alias in sys.modules:
            continue
        _ensure_geak_package(alias)
        spec = importlib.util.spec_from_file_location(alias, entry_file)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            pass

_KERNEL_DIR = _resolve_geak_kernel_dir()
if _KERNEL_DIR and _KERNEL_DIR not in sys.path:
    sys.path.insert(0, _KERNEL_DIR)
_register_geak_aliases(_KERNEL_DIR)
# ── End dynamic loader ─────────────────────────────────────────────────────

from aiter.ops.triton.gemm_a16w16_atomic import gemm_a16w16_atomic

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WARMUP = 50
ITERATIONS = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200"))

# ---------------------------------------------------------------------------
# Full ordered config list - from bench_gemm_a16w16.py -> benchmark_utils.get_x_vals()
# Each entry is (M, N, K).
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
# Deterministic subset picker
# ---------------------------------------------------------------------------
def _pick(configs, count):
    if len(configs) <= count:
        return list(range(len(configs)))
    n = len(configs)
    return [round(i * (n - 1) / (count - 1)) for i in range(count)]


# ---------------------------------------------------------------------------
# Input generation (mirrors op_tests/triton_tests/gemm/basic/test_gemm_a16w16.py)
# ---------------------------------------------------------------------------
def generate_inputs(M, N, K, dtype=torch.bfloat16):
    """Generate inputs for gemm_a16w16_atomic: x (M,K), w (N,K), y (M,N) fp32 zeroed."""
    x = torch.randn((M, K), dtype=dtype, device="cuda")
    w = torch.randn((N, K), dtype=dtype, device="cuda")
    y = torch.zeros((M, N), dtype=torch.float32, device="cuda")
    return x, w, y


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------
def reference_impl(x, w):
    """torch.nn.functional.linear: Y = X @ W^T"""
    return F.linear(x, w, bias=None)


# ---------------------------------------------------------------------------
# Correctness check for one config
# ---------------------------------------------------------------------------
def check_correctness(M, N, K, dtype=torch.bfloat16):
    x, w, y = generate_inputs(M, N, K, dtype)
    torch_out = reference_impl(x, w)
    triton_out = gemm_a16w16_atomic(x, w, torch.float32, y).to(dtype)
    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)


# ---------------------------------------------------------------------------
# Benchmark one config - returns median latency in ms
# ---------------------------------------------------------------------------
def bench_one(M, N, K, dtype=torch.bfloat16):
    x, w, y = generate_inputs(M, N, K, dtype)

    def _fn():
        y.zero_()
        return gemm_a16w16_atomic(x, w, torch.float32, y)

    ms = triton.testing.do_bench(
        _fn,
        warmup=WARMUP,
        rep=ITERATIONS,
    )
    return ms


# ---------------------------------------------------------------------------
# CLI modes
# ---------------------------------------------------------------------------
def run_correctness(indices):
    torch.manual_seed(42)
    print("Running correctness on {} configs ...".format(len(indices)))
    for idx in indices:
        M, N, K = ALL_CONFIGS[idx]
        try:
            check_correctness(M, N, K)
            print("  [{}] M={} N={} K={}  PASS".format(idx, M, N, K))
        except Exception as e:
            print("  [{}] M={} N={} K={}  FAIL: {}".format(idx, M, N, K, e))
            print("GEAK_SHAPES_USED={}".format(indices))
            sys.exit(1)
    print("GEAK_SHAPES_USED={}".format(indices))
    print("All correctness checks passed.")


def run_benchmark(indices):
    torch.manual_seed(42)
    latencies = []
    print("Running benchmark on {} configs ...".format(len(indices)))
    for idx in indices:
        M, N, K = ALL_CONFIGS[idx]
        ms = bench_one(M, N, K)
        latencies.append(ms)
        print("  M={} N={} K={}  {:.4f}ms".format(M, N, K, ms))
    # Geometric mean
    log_sum = sum(math.log(l) for l in latencies)
    geo_mean = math.exp(log_sum / len(latencies))
    print("GEAK_SHAPES_USED={}".format(indices))
    print("GEAK_RESULT_LATENCY_MS={:.4f}".format(geo_mean))


def run_profile(indices):
    torch.manual_seed(42)
    print("Running profile on {} configs ...".format(len(indices)))
    for idx in indices:
        M, N, K = ALL_CONFIGS[idx]
        ms = bench_one(M, N, K)
        print("  M={} N={} K={}  {:.4f}ms".format(M, N, K, ms))
    print("GEAK_SHAPES_USED={}".format(indices))


def main():
    parser = argparse.ArgumentParser(description="Test harness for gemm_a16w16_atomic")
    parser.add_argument("--correctness", action="store_true", help="Run correctness checks")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark (up to 25 configs)")
    parser.add_argument("--full-benchmark", action="store_true", help="Run full benchmark (all configs)")
    parser.add_argument("--profile", action="store_true", help="Run profile (5 configs)")
    parser.add_argument("--iterations", type=int, default=None, help="Number of benchmark iterations (overrides GEAK_BENCHMARK_ITERATIONS env var)")
    args = parser.parse_args()
    if args.iterations is not None:
        global ITERATIONS
        ITERATIONS = args.iterations

    if not any([args.correctness, args.benchmark, args.full_benchmark, args.profile]):
        parser.print_help()
        sys.exit(1)

    if args.correctness:
        indices = list(range(len(ALL_CONFIGS)))
        run_correctness(indices)

    if args.profile:
        indices = _pick(ALL_CONFIGS, 5)
        run_profile(indices)

    if args.benchmark:
        indices = list(range(len(ALL_CONFIGS)))  # use all configs so benchmark matches full-benchmark
        run_benchmark(indices)

    if args.full_benchmark:
        indices = list(range(len(ALL_CONFIGS)))
        run_benchmark(indices)


if __name__ == "__main__":
    main()
