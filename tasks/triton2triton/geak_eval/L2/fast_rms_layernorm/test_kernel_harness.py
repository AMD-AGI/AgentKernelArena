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
    aliases = ['fast_rms_layernorm']
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

"""
Test harness for fast_rms_layernorm kernel.
Modes: --correctness, --profile, --benchmark, --full-benchmark

Shapes taken from the GEAK-eval ground-truth test function:
  test_fast_rms_layernorm_with_backward() in fast_rms_layernorm.py
    test_case_1: X=(2,4,8), gemma=False  (forward + backward)
    test_case_2: X=(2,4,8), gemma=True   (forward + backward)
"""

import argparse
import math
import os
import sys
import torch
import random
import numpy as np
import statistics

KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))
if KERNEL_DIR not in sys.path:
    sys.path.insert(0, KERNEL_DIR)

from fast_rms_layernorm import fast_rms_layernorm, SimpleLayerNorm

# ============================================================================
# Shapes from the GEAK-eval ground-truth test:
#   X = torch.randn(2, 4, 8, device='cuda', dtype=torch.float32, requires_grad=True)
# ============================================================================

ALL_SHAPES = [
    (2, 4, 8),
]


HARNESS_SHAPES = ALL_SHAPES[:25]
PROFILE_SHAPES = ALL_SHAPES[:5]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rms_layernorm_reference(x, weight, eps=1e-5):
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


def gemma_rms_layernorm_reference(x, weight, eps=1e-5):
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * (weight + 1.0)


def benchmark_fn(fn, warmup=50, iterations=200):
    """Time a callable using CUDA events. Returns median latency in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

    for i in range(iterations):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return statistics.median(times)


def run_correctness(shapes, atol=1e-2, rtol=1e-2):
    """Run correctness tests matching the eval test cases exactly.

    Mirrors test_fast_rms_layernorm_with_backward():
      test_case_1: backward grad for gemma=False
      test_case_2: backward grad for gemma=True
    """
    set_seed(42)
    print(f"Running correctness tests on {len(shapes)} shapes (atol={atol}, rtol={rtol})...")

    all_passed = True
    for shape in shapes:
        hidden_dim = shape[-1]
        x = torch.randn(*shape, dtype=torch.float32, device='cpu', requires_grad=True).to('cuda')
        layernorm = SimpleLayerNorm(hidden_dim, eps=1e-5).to('cuda')

        output = fast_rms_layernorm(layernorm, x, gemma=False)
        output.mean().backward()
        grad1 = x.grad.clone()
        x.grad.zero_()

        x_ref = x.detach().clone().requires_grad_(True)
        rms_layernorm_reference(x_ref, layernorm.weight, eps=1e-5).mean().backward()
        try:
            torch.testing.assert_close(grad1, x_ref.grad, rtol=rtol, atol=atol)
            print(f"  PASS: {shape} gemma=False backward")
        except AssertionError as e:
            print(f"  FAIL: {shape} gemma=False backward: {e}")
            all_passed = False

        output_g = fast_rms_layernorm(layernorm, x, gemma=True)
        output_g.mean().backward()
        grad2 = x.grad.clone()

        x_ref2 = x.detach().clone().requires_grad_(True)
        gemma_rms_layernorm_reference(x_ref2, layernorm.weight, eps=1e-5).mean().backward()
        try:
            torch.testing.assert_close(grad2, x_ref2.grad, rtol=rtol, atol=atol)
            print(f"  PASS: {shape} gemma=True backward")
        except AssertionError as e:
            print(f"  FAIL: {shape} gemma=True backward: {e}")
            all_passed = False

    if all_passed:
        print("\nAll correctness tests PASSED!")
        return 0
    else:
        print("\nSome correctness tests FAILED!")
        return 1


def run_profile(shapes, warmup=50):
    """Run kernel once per shape for profiling with proper warmup."""
    set_seed(42)
    print(f"Running profile mode on {len(shapes)} shapes (warmup={warmup})...")
    for shape in shapes:
        hidden_dim = shape[-1]
        x = torch.randn(*shape, dtype=torch.float32, device='cpu').to('cuda')
        layernorm = SimpleLayerNorm(hidden_dim, eps=1e-5).to('cuda')

        for _ in range(warmup):
            fast_rms_layernorm(layernorm, x, gemma=False)
        torch.cuda.synchronize()

        fast_rms_layernorm(layernorm, x, gemma=False)
        torch.cuda.synchronize()
        print(f"  Profiled: {shape}")
    return 0


def run_benchmark(shapes, warmup=50, iterations=200):
    """Benchmark kernel vs reference; report per-shape speedups and geo-mean.
    Uses baseline Triton when benchmark_baseline.txt exists (patch eval); else PyTorch (preprocess)."""
    set_seed(42)
    baseline_dir = _find_baseline_kernel_dir()
    kernel_dir = _resolve_geak_kernel_dir()
    baseline_fn = None
    if baseline_dir and baseline_dir != kernel_dir:
        baseline_fn = _load_baseline_triton(baseline_dir, "baseline_fast_rms", "fast_rms_layernorm")
    ref_label = "baseline_triton" if baseline_fn else "ref"

    print(f"Benchmarking {len(shapes)} shapes (warmup={warmup}, iterations={iterations})...")
    print(f"  Comparing kernel vs {ref_label}")
    print()

    speedups = []
    kernel_latencies = []

    for shape in shapes:
        hidden_dim = shape[-1]
        x = torch.randn(*shape, dtype=torch.float32, device='cpu').to('cuda')
        layernorm = SimpleLayerNorm(hidden_dim, eps=1e-5).to('cuda')

        kernel_ms = benchmark_fn(
            lambda: fast_rms_layernorm(layernorm, x, gemma=False),
            warmup=warmup, iterations=iterations,
        )
        if baseline_fn is not None:
            ref_ms = benchmark_fn(
                lambda: baseline_fn(layernorm, x, gemma=False),
                warmup=warmup, iterations=iterations,
            )
        else:
            ref_ms = benchmark_fn(
                lambda: rms_layernorm_reference(x, layernorm.weight, eps=1e-5),
                warmup=warmup, iterations=iterations,
            )

        speedup = ref_ms / kernel_ms if kernel_ms > 0 else float('inf')
        speedups.append(speedup)
        kernel_latencies.append(kernel_ms)
        print(f"  Shape {shape}: kernel={kernel_ms:.4f} ms | ref={ref_ms:.4f} ms | speedup={speedup:.3f}x")

    geo_mean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
    median_latency = statistics.median(kernel_latencies)

    print()
    print(f"Geometric mean speedup: {geo_mean:.3f}x")
    print(f"Median kernel latency: {median_latency:.4f} ms")
    print(f"GEAK_RESULT_LATENCY_MS={median_latency:.6f}")
    print(f"GEAK_RESULT_GEOMEAN_SPEEDUP={geo_mean:.4f}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Test harness for fast_rms_layernorm")
    parser.add_argument("--correctness", action="store_true", help="Run correctness tests")
    parser.add_argument("--profile", action="store_true", help="Run kernel once for profiling")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark on HARNESS_SHAPES")
    parser.add_argument("--full-benchmark", action="store_true", help="Run benchmark on ALL_SHAPES")
    parser.add_argument("--warmup", type=int, default=50,
                        help="Number of warmup iterations (default: 50)")
    parser.add_argument("--iterations", type=int,
                        default=int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200")),
                        help="Number of timed iterations (default: GEAK_BENCHMARK_ITERATIONS or 200)")
    parser.add_argument("--atol", type=float, default=1e-2,
                        help="Absolute tolerance for correctness (default: 1e-2)")
    parser.add_argument("--rtol", type=float, default=1e-2,
                        help="Relative tolerance for correctness (default: 1e-2)")

    args = parser.parse_args()

    if args.correctness:
        sys.exit(run_correctness(HARNESS_SHAPES, atol=args.atol, rtol=args.rtol))
    elif args.profile:
        sys.exit(run_profile(PROFILE_SHAPES, warmup=args.warmup))
    elif args.benchmark:
        sys.exit(run_benchmark(HARNESS_SHAPES, warmup=args.warmup, iterations=args.iterations))
    elif args.full_benchmark:
        sys.exit(run_benchmark(ALL_SHAPES, warmup=args.warmup, iterations=args.iterations))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
