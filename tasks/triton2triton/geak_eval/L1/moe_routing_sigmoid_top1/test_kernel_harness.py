#!/usr/bin/env python3
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
    aliases = ['moe_routing_sigmoid_top1', 'aiter.ops.triton.moe.moe_routing_sigmoid_top1_fused']
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

# SPDX-License-Identifier: MIT
# Test harness for moe_routing_sigmoid_top1_fused kernel

import argparse
import math
import os
import sys
from functools import partial

import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from aiter.ops.triton.moe.moe_routing_sigmoid_top1_fused import routing_sigmoid_top1

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


# ---- modes ----------------------------------------------------------------

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
        print("All correctness tests passed!")
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
    """Benchmark kernel vs reference; report per-shape speedups and geomean.
    Uses baseline Triton when benchmark_baseline.txt exists (patch eval); else PyTorch (preprocess)."""
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16
    TOPK = 1

    baseline_dir = _find_baseline_kernel_dir()
    kernel_dir = _resolve_geak_kernel_dir()
    baseline_fn = None
    if baseline_dir and baseline_dir != kernel_dir:
        baseline_fn = _load_baseline_triton(baseline_dir, "baseline_moe", "routing_sigmoid_top1")
    ref_label = "baseline_triton" if baseline_fn else "ref"

    print(f"Running benchmark on {len(shapes)} shapes "
          f"(warmup={warmup}, iterations={iterations})...")
    print(f"  Comparing kernel vs {ref_label}")
    print(f"{'#':>4s}  {'Shape':>24s}  {'Ref (ms)':>10s}  "
          f"{'Kernel (ms)':>12s}  {'Speedup':>8s}")
    print("-" * 68)

    speedups = []
    kernel_times = []

    for i, (M, N, K) in enumerate(shapes):
        x = torch.randn((M, K), dtype=dtype, device=device)
        w = torch.randn((K, N), dtype=dtype, device=device) * 0.1

        dummy_ids = torch.ones((M, 1), dtype=torch.int32, device=device) * N
        dummy_weights = torch.ones((M, 1), dtype=torch.float32, device=device)

        if baseline_fn is not None:
            def _run_ref(x=x, w=w, bf=baseline_fn):
                bf(x, w, TOPK, fused_shared_experts=True)
        else:
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
        kernel_times.append(kernel_time)

        shape_str = f"M={M}, N={N}, K={K}"
        print(f"  {i+1:>3d}   {shape_str:>24s}  {ref_time:>10.4f}  "
              f"{kernel_time:>12.4f}  {speedup:>7.2f}x")

    print("-" * 68)
    geomean_speedup = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
    geomean_latency_ms = math.exp(sum(math.log(t) for t in kernel_times) / len(kernel_times))
    print(f"Geometric mean latency: {geomean_latency_ms:.4f} ms")
    print(f"Geometric mean speedup: {geomean_speedup:.4f}x")
    print(f"GEAK_RESULT_LATENCY_MS={geomean_latency_ms:.4f}")
    print(f"GEAK_RESULT_GEOMEAN_SPEEDUP={geomean_speedup:.4f}")


# ---- CLI ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test harness for moe_routing_sigmoid_top1_fused kernel",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
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
    parser.add_argument("--iterations", type=int,
                        default=int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200")),
                        help="Benchmark iterations (default: GEAK_BENCHMARK_ITERATIONS or 200)")
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


if __name__ == "__main__":
    main()
