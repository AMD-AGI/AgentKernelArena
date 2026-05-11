#!/usr/bin/env python3
"""Test harness for FlyDSL softmax_kernel (flydsl2flydsl)."""
import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

# ============================================================================
# GEAK bootstrap
# ============================================================================

KERNEL_FILE = "kernel.py"


def _find_baseline_kernel_dir():
    work = os.environ.get("GEAK_WORK_DIR", "").strip()
    if not work:
        return None
    d = Path(work).resolve()
    for _ in range(10):
        if d is None or not d.exists():
            break
        if (d / "benchmark_baseline.txt").is_file():
            return str(d)
        d = d.parent
    return None


def _resolve_kernel_dir():
    candidates = []
    work_dir = os.environ.get("GEAK_WORK_DIR", "").strip()
    if work_dir:
        candidates.append(work_dir)
    original = os.path.dirname(os.path.abspath(__file__))
    candidates.append(original)
    for c in candidates:
        if c and os.path.isfile(os.path.join(c, KERNEL_FILE)):
            return c
    return original


def _load_kernel(kernel_dir, alias="flydsl_kernel"):
    entry = os.path.join(kernel_dir, KERNEL_FILE)
    if not os.path.isfile(entry):
        return None
    if kernel_dir not in sys.path:
        sys.path.insert(0, kernel_dir)
    spec = importlib.util.spec_from_file_location(alias, entry)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


_KERNEL_DIR = _resolve_kernel_dir()

# ============================================================================
# Test shapes
# ============================================================================

ALL_SHAPES = [
    (32, 1024, "f32"),
    (64, 1024, "f32"),
    (32, 2048, "f32"),
    (64, 2048, "f32"),
    (128, 2048, "f32"),
    (128, 4096, "f32"),
    (256, 4096, "f32"),
    (512, 4096, "f32"),
    (256, 8192, "f32"),
    (512, 8192, "f32"),
]

_n_all = len(ALL_SHAPES)
if _n_all <= 25:
    HARNESS_SHAPES = ALL_SHAPES
else:
    _idx = [int(round(i * (_n_all - 1) / 24)) for i in range(25)]
    HARNESS_SHAPES = [ALL_SHAPES[i] for i in _idx]

_pidx = [int(round(i * (_n_all - 1) / 4)) for i in range(5)]
PROFILE_SHAPES = [ALL_SHAPES[i] for i in _pidx]

RTOL, ATOL = 1e-3, 1e-3
DTYPE_MAP = {"f16": "float16", "bf16": "bfloat16", "f32": "float32"}

# ============================================================================
# Reference
# ============================================================================


def reference_softmax(x):
    import torch

    return torch.softmax(x.float(), dim=-1).to(x.dtype)


# ============================================================================
# Modes
# ============================================================================


def run_correctness(shapes=None, verbose=True):
    import torch

    if shapes is None:
        shapes = HARNESS_SHAPES
    if verbose:
        print(f"Running correctness on {len(shapes)} shapes...")

    mod = _load_kernel(_KERNEL_DIR)
    if mod is None:
        print("FAIL: cannot load kernel.py")
        return {"correct": False, "num_correct": 0, "num_failed": len(shapes), "failures": []}

    results, failures = [], []
    for i, (M, N, dtype_str) in enumerate(shapes):
        try:
            torch_dtype = getattr(torch, DTYPE_MAP[dtype_str])
            torch.manual_seed(42 + i)
            x = torch.randn(M, N, device="cuda", dtype=torch_dtype)
            output = torch.empty_like(x)

            launch_fn = mod.build_softmax_module(M, N, dtype_str)
            launch_fn(x, output, M)
            torch.cuda.synchronize()

            ref = reference_softmax(x)
            torch.testing.assert_close(output, ref, atol=ATOL, rtol=RTOL)
            results.append({"config": (M, N, dtype_str), "correct": True})
            if verbose:
                print(f"  PASS: (M={M}, N={N}, {dtype_str})")
        except Exception as e:
            failures.append({"config": (M, N, dtype_str), "error": str(e)})
            if verbose:
                print(f"  FAIL: (M={M}, N={N}, {dtype_str}) - {str(e)[:80]}")

    if verbose:
        print("-" * 62)
        status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{len(shapes)})"
        print(f"{'Status:':<22} {status}")

    return {
        "correct": len(failures) == 0,
        "num_correct": len(results),
        "num_failed": len(failures),
        "failures": failures,
    }


def run_profile(shapes=None, warmup=50, iters=200, verbose=True):
    import torch

    if shapes is None:
        shapes = PROFILE_SHAPES
    if verbose:
        print(f"Profile: {len(shapes)} config(s), {warmup} warmup, {iters} iter(s)")

    mod = _load_kernel(_KERNEL_DIR)
    if mod is None:
        return

    for M, N, dtype_str in shapes:
        torch_dtype = getattr(torch, DTYPE_MAP[dtype_str])
        x = torch.randn(M, N, device="cuda", dtype=torch_dtype)
        output = torch.empty_like(x)
        launch_fn = mod.build_softmax_module(M, N, dtype_str)

        for _ in range(warmup):
            launch_fn(x, output, M)
        torch.cuda.synchronize()
        for _ in range(iters):
            launch_fn(x, output, M)
        torch.cuda.synchronize()
        if verbose:
            print(f"  (M={M}, N={N}, {dtype_str}) done")


def run_benchmark(shapes=None, warmup=50, iters=200, verbose=True):
    import torch

    if shapes is None:
        shapes = HARNESS_SHAPES

    mod = _load_kernel(_KERNEL_DIR)
    if mod is None:
        print("FAIL: cannot load kernel.py")
        return {"geomean_latency_ms": -1, "geomean_speedup": -1}

    latencies, speedups, report_cases = [], [], []

    print(f"Running benchmark on {len(shapes)} shapes, {warmup} warmup, {iters} iterations...")
    print(f"  Comparing kernel vs PyTorch")
    print(f"{'Config (M,N,dtype)':<26} {'Ref':>10} {'FlyDSL':>10} {'Speedup':>10}")
    print("-" * 62)

    for idx, (M, N, dtype_str) in enumerate(shapes):
        torch_dtype = getattr(torch, DTYPE_MAP[dtype_str])
        torch.manual_seed(42)
        x = torch.randn(M, N, device="cuda", dtype=torch_dtype)
        output = torch.empty_like(x)

        launch_fn = mod.build_softmax_module(M, N, dtype_str)

        for _ in range(warmup):
            launch_fn(x, output, M)
        torch.cuda.synchronize()

        kernel_times = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            launch_fn(x, output, M)
            e.record()
            torch.cuda.synchronize()
            kernel_times.append(s.elapsed_time(e))
        kernel_ms = sorted(kernel_times)[len(kernel_times) // 2]

        ref_times = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            _ = reference_softmax(x)
            e.record()
            torch.cuda.synchronize()
            ref_times.append(s.elapsed_time(e))
        ref_ms = sorted(ref_times)[len(ref_times) // 2]

        speedup = ref_ms / kernel_ms if kernel_ms > 0 else 1.0
        latencies.append(kernel_ms)
        speedups.append(speedup)
        report_cases.append({
            "test_case_id": f"test_case_{idx}",
            "execution_time_ms": kernel_ms,
            "shape": [M, N],
            "params": {"M": M, "N": N, "dtype": dtype_str},
        })

        marker = " *" if speedup > 1.0 else ""
        if verbose:
            print(
                f"(M={M:>4}, N={N:>5}, {dtype_str}){' ':2} "
                f"{ref_ms:>8.4f}ms {kernel_ms:>8.4f}ms {speedup:>8.2f}x{marker}",
                flush=True,
            )

        del x, output
        torch.cuda.empty_cache()

    geomean_latency = math.exp(sum(math.log(l) for l in latencies) / len(latencies))
    geomean_speedup = math.exp(sum(math.log(s) for s in speedups) / len(speedups))

    build_dir = Path(_KERNEL_DIR) / "build"
    build_dir.mkdir(exist_ok=True)
    with open(build_dir / "performance_report.json", "w") as f:
        json.dump(report_cases, f, indent=2)

    print("-" * 62)
    print(f"{'Geometric mean latency:':<26} {geomean_latency:.4f} ms")
    print(f"{'Geometric mean speedup:':<26} {geomean_speedup:.2f}x")
    print(f"GEAK_RESULT_LATENCY_MS={geomean_latency:.4f}", flush=True)
    print(f"GEAK_RESULT_GEOMEAN_SPEEDUP={geomean_speedup:.4f}", flush=True)

    return {"geomean_latency_ms": geomean_latency, "geomean_speedup": geomean_speedup}


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlyDSL Softmax Kernel Test Harness")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument(
        "--iterations",
        type=int,
        default=int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200")),
    )
    args = parser.parse_args()

    print("=" * 62)
    print("FlyDSL Softmax Kernel")
    print("=" * 62)

    if args.correctness:
        print("\n[Correctness Mode]")
        result = run_correctness(HARNESS_SHAPES)
        sys.exit(0 if result.get("correct", False) else 1)
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
