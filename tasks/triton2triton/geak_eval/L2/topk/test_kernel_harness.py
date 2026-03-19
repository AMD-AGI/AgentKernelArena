#!/usr/bin/env python3
"""
TopK kernel test harness for Triton/ROCm.

This script validates and measures a custom `topk` implementation across predefined
input shapes `(batch_size, hidden_size, k)`.

It supports four modes:
- correctness: compares kernel outputs against `torch.topk` reference results.
- profile: runs a small, representative shape subset once for profiler capture.
- benchmark: times a sampled harness subset and reports per-shape median latency.
- full-benchmark: times all shapes and reports geometric-mean latency.

Shape groups:
- `ALL_SHAPES`: full test/benchmark matrix.
- `HARNESS_SHAPES`: 25 uniformly sampled shapes from `ALL_SHAPES`.
- `PROFILE_SHAPES`: 5 evenly spaced shapes from `ALL_SHAPES`.

Benchmark iteration count is taken from `--iterations`, or defaults to
`GEAK_BENCHMARK_ITERATIONS` (fallback: 20). Final benchmark summary is emitted as:
`GEAK_RESULT_LATENCY_MS=<value>`.
"""
from __future__ import annotations

# GEAK materialized harness bootstrap
import importlib.util
import os
import sys
import types
from pathlib import Path

def _find_baseline_kernel_dir():
    """Find preprocess dir (has benchmark_baseline.txt) by walking up from GEAK_WORK_DIR."""
    work = os.environ.get("GEAK_WORK_DIR", "").strip()
    if not work:
        return None
    d = Path(work).resolve()
    for _ in range(10):
        if d is None or not d.exists():
            break
        bb = d / "benchmark_baseline.txt"
        if bb.is_file():
            return str(d)
        d = d.parent
    return None

def _load_baseline_triton(baseline_dir, module_alias, entry_name):
    """Load kernel from baseline_dir. Returns callable or None."""
    entry_file = Path(baseline_dir) / "kernel.py"
    if not entry_file.is_file():
        return None
    if baseline_dir not in sys.path:
        sys.path.insert(0, baseline_dir)
    spec = importlib.util.spec_from_file_location(module_alias, entry_file)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_alias] = module
    try:
        spec.loader.exec_module(module)
        return getattr(module, entry_name, None)
    except Exception:
        return None

def _resolve_geak_kernel_dir():
    candidates = []
    work_dir = os.environ.get("GEAK_WORK_DIR", "").strip()
    if work_dir:
        candidates.append(work_dir)
    repo_root = os.environ.get("GEAK_REPO_ROOT", "").strip()
    rel_kernel_dir = '.'
    if repo_root and rel_kernel_dir:
        candidates.append(os.path.join(repo_root, rel_kernel_dir))
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

def _ensure_geak_aiter_fp8_dtype(module):
    fp8_value = getattr(module, "fp8_dtype", None)
    if fp8_value is None:
        return
    aiter_mod = sys.modules.get("aiter")
    if aiter_mod is None:
        try:
            import aiter as aiter_mod
        except Exception:
            _ensure_geak_package("aiter")
            aiter_mod = sys.modules.get("aiter")
    if aiter_mod is None:
        return
    dtypes_obj = getattr(aiter_mod, "dtypes", None)
    if dtypes_obj is None:
        dtypes_obj = types.SimpleNamespace()
        setattr(aiter_mod, "dtypes", dtypes_obj)
    if getattr(dtypes_obj, "fp8", None) is None:
        setattr(dtypes_obj, "fp8", fp8_value)

def _register_geak_aliases(kernel_dir):
    aliases = ['topk', 'aiter.ops.triton.topk']
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
        spec.loader.exec_module(module)
        _ensure_geak_aiter_fp8_dtype(module)

_KERNEL_DIR = _resolve_geak_kernel_dir()
if _KERNEL_DIR and _KERNEL_DIR not in sys.path:
    sys.path.insert(0, _KERNEL_DIR)
_register_geak_aliases(_KERNEL_DIR)

import argparse
import math
import os
import sys
import torch

# ── Shape lists ──────────────────────────────────────────────────────────────
# Extracted from:
#   op_tests/triton_tests/test_topk.py:
#     BATCH_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 16, 1335]
#     DIM2 = [16, 128256]
#     K = [2, 8]
#   op_tests/op_benchmarks/triton/bench_topk.py:
#     BATCH_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 16, 1335]
#     DIM2S = (16, 128, 256, 128256)
#     KS = (2, 8)
#
# Each shape is (batch_size, hidden_size, topk).
# Sorted by total element count (batch * hidden).

ALL_SHAPES = [
    (1, 16, 2),
    (1, 16, 8),
    (2, 16, 2),
    (2, 16, 8),
    (3, 16, 2),
    (3, 16, 8),
    (4, 16, 2),
    (4, 16, 8),
    (5, 16, 2),
    (5, 16, 8),
    (6, 16, 2),
    (6, 16, 8),
    (7, 16, 2),
    (7, 16, 8),
    (1, 128, 2),
    (1, 128, 8),
    (8, 16, 2),
    (8, 16, 8),
    (1, 256, 2),
    (1, 256, 8),
    (2, 128, 2),
    (2, 128, 8),
    (16, 16, 2),
    (16, 16, 8),
    (3, 128, 2),
    (3, 128, 8),
    (2, 256, 2),
    (2, 256, 8),
    (4, 128, 2),
    (4, 128, 8),
    (5, 128, 2),
    (5, 128, 8),
    (3, 256, 2),
    (3, 256, 8),
    (6, 128, 2),
    (6, 128, 8),
    (7, 128, 2),
    (7, 128, 8),
    (4, 256, 2),
    (4, 256, 8),
    (8, 128, 2),
    (8, 128, 8),
    (5, 256, 2),
    (5, 256, 8),
    (6, 256, 2),
    (6, 256, 8),
    (7, 256, 2),
    (7, 256, 8),
    (8, 256, 2),
    (8, 256, 8),
    (16, 128, 2),
    (16, 128, 8),
    (16, 256, 2),
    (16, 256, 8),
    (1335, 16, 2),
    (1335, 16, 8),
    (1, 128256, 2),
    (1, 128256, 8),
    (1335, 128, 2),
    (1335, 128, 8),
    (2, 128256, 2),
    (2, 128256, 8),
    (1335, 256, 2),
    (1335, 256, 8),
    (3, 128256, 2),
    (3, 128256, 8),
    (4, 128256, 2),
    (4, 128256, 8),
    (5, 128256, 2),
    (5, 128256, 8),
    (6, 128256, 2),
    (6, 128256, 8),
    (7, 128256, 2),
    (7, 128256, 8),
    (8, 128256, 2),
    (8, 128256, 8),
    (16, 128256, 2),
    (16, 128256, 8),
    (1335, 128256, 2),
    (1335, 128256, 8),
]

# HARNESS_SHAPES: 25 uniformly sampled from ALL_SHAPES
_n_all = len(ALL_SHAPES)
_harness_indices = [int(round(i * (_n_all - 1) / 24)) for i in range(25)]
HARNESS_SHAPES = [ALL_SHAPES[i] for i in _harness_indices]

# PROFILE_SHAPES: 5 evenly-spaced from ALL_SHAPES
_profile_indices = [int(round(i * (_n_all - 1) / 4)) for i in range(5)]
PROFILE_SHAPES = [ALL_SHAPES[i] for i in _profile_indices]


# ── Helpers ──────────────────────────────────────────────────────────────────
def make_input(batch, hidden, seed=42):
    """Create input tensor on CPU with fixed seed, then move to GPU."""
    torch.manual_seed(seed)
    x_cpu = torch.randn(batch, hidden, dtype=torch.float32)
    return x_cpu.to("cuda")


def reference_topk(x, k, largest=True):
    """Torch reference on CPU."""
    return torch.topk(x.cpu(), k, dim=-1, largest=largest)


def triton_op(x, k):
    """Triton TopK implementation."""
    from aiter.ops.triton.topk import topk as triton_topk
    return triton_topk(x, k, largest=True)


def torch_op(x, k):
    """PyTorch reference implementation."""
    return torch.topk(x, k, dim=-1, largest=True, sorted=True)


# ── Modes ────────────────────────────────────────────────────────────────────
def run_correctness(shapes, verbose: bool = True) -> dict:
    from aiter.ops.triton.topk import topk as triton_topk

    if verbose:
        print(f"Running correctness on {len(shapes)} shapes...")
    
    results, failures = [], []
    for idx, (batch, hidden, k) in enumerate(shapes):
        try:
            x = make_input(batch, hidden, seed=42 + idx)
            ref_val, ref_idx = reference_topk(x, k, largest=True)
            res_val, res_idx = triton_topk(x, k, largest=True)

            res_val_cpu = res_val.cpu()
            res_idx_cpu = res_idx.cpu()

            # Check values match
            torch.testing.assert_close(
                res_val_cpu,
                ref_val.to(torch.float32),
                atol=1e-4 * hidden,
                rtol=1.3e-6,
            )
            # Check indices: gather from input using result indices and compare values
            gathered_res = torch.gather(x.cpu(), 1, res_idx_cpu)
            gathered_ref = torch.gather(x.cpu(), 1, ref_idx)
            torch.testing.assert_close(
                gathered_res,
                gathered_ref.to(torch.float32),
                atol=1e-4 * hidden,
                rtol=1.3e-6,
            )

            results.append({"config": (batch, hidden, k), "correct": True})
            if verbose:
                print(f"  PASS: ({batch}, {hidden}), k={k}")

            del x, res_val, res_idx
            torch.cuda.empty_cache()
        except Exception as e:
            failures.append({"config": (batch, hidden, k), "error": str(e)})
            if verbose:
                print(f"  FAIL: ({batch}, {hidden}), k={k} - {str(e)[:50]}")

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
    from aiter.ops.triton.topk import topk as triton_topk

    if verbose:
        print(f"Profile: {len(shapes)} config(s), {warmup} warmup, {iters} iter(s)")

    for batch, hidden, k in shapes:
        x = torch.randn(batch, hidden, dtype=torch.float32, device="cpu").to("cuda")
        for _ in range(warmup):
            triton_topk(x, k, largest=True)
        torch.cuda.synchronize()
        for _ in range(iters):
            triton_topk(x, k, largest=True)
        torch.cuda.synchronize()
        if verbose:
            print(f"  ({batch}, {hidden}), k={k} done")
        del x
        torch.cuda.empty_cache()


def run_benchmark(shapes, warmup: int = 50, iters: int = 200, verbose: bool = True) -> dict:
    from aiter.ops.triton.topk import topk as triton_topk

    baseline_dir = _find_baseline_kernel_dir()
    kernel_dir = _resolve_geak_kernel_dir()
    baseline_topk = None
    if baseline_dir and baseline_dir != kernel_dir:
        baseline_topk = _load_baseline_triton(baseline_dir, "baseline_topk", "topk")
    ref_label = "baseline_triton" if baseline_topk else "PyTorch"

    print(f"Running benchmark on {len(shapes)} shapes, {warmup} warmup, {iters} iterations each...")
    print(f"  Comparing kernel vs {ref_label}")
    latencies = []
    speedups = []
    results = []

    if verbose:
        print(
            f"{'Config (B,M,K)':<22} {'Ref':>10} {'Triton':>10} {'Speedup':>10}"
        )
        print("-" * 62)

    for idx, (batch, hidden, k) in enumerate(shapes):
        x = make_input(batch, hidden, seed=42 + idx)

        # Warmup
        for _ in range(warmup):
            triton_op(x, k)
        torch.cuda.synchronize()

        # Benchmark Triton (kernel under test)
        triton_times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            triton_op(x, k)
            end.record()
            torch.cuda.synchronize()
            triton_times.append(start.elapsed_time(end))

        triton_ms = sorted(triton_times)[len(triton_times) // 2]  # median

        # Benchmark reference (baseline Triton or PyTorch)
        ref_times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            if baseline_topk is not None:
                baseline_topk(x, k, largest=True)
            else:
                torch_op(x, k)
            end.record()
            torch.cuda.synchronize()
            ref_times.append(start.elapsed_time(end))

        ref_ms = sorted(ref_times)[len(ref_times) // 2]  # median

        speedup = ref_ms / triton_ms if triton_ms > 0 else 1.0
        speedups.append(speedup)
        latencies.append(triton_ms)

        results.append({
            "config": (batch, hidden, k),
            "ref_ms": ref_ms,
            "triton_ms": triton_ms,
            "speedup": speedup,
        })

        if verbose:
            marker = " *" if speedup > 1.0 else ""
            print(
                f"({batch}, {hidden}), k={k}{' ':4} {ref_ms:>8.4f}ms {triton_ms:>8.4f}ms {speedup:>8.2f}x{marker}"
            )

        del x
        torch.cuda.empty_cache()

    # Compute geometric means
    log_sum = sum(math.log(t) for t in latencies)
    geomean_latency = math.exp(log_sum / len(latencies))
    
    log_sum_speedup = sum(math.log(s) for s in speedups)
    geomean_speedup = math.exp(log_sum_speedup / len(speedups))

    if verbose:
        print("-" * 62)
        print(f"{'Geometric mean latency:':<22} {geomean_latency:.4f} ms")
        print(f"{'Geometric mean speedup:':<22} {geomean_speedup:.2f}x")
        print(f"GEAK_RESULT_LATENCY_MS={geomean_latency:.4f}")
        print(f"GEAK_RESULT_GEOMEAN_SPEEDUP={geomean_speedup:.4f}")

    return {
        "geomean_latency_ms": geomean_latency,
        "geomean_speedup": geomean_speedup,
        "results": results,
    }


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="TopK kernel test harness")
    parser.add_argument(
        "--correctness",
        action="store_true",
        help="Run correctness tests on benchmark shapes",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run minimal profiling workload",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark on HARNESS_SHAPES (25 uniformly sampled)",
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
        default=int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200")),
        help="Number of benchmark iterations (default: GEAK_BENCHMARK_ITERATIONS or 200)",
    )
    args = parser.parse_args()

    print("=" * 62)
    print("TopK Kernel Test Harness")
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


if __name__ == "__main__":
    main()
