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
    aliases = ['llama_ff_triton']
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
Test harness for llama_ff_triton kernel.
Modes: --correctness, --profile, --benchmark, --full-benchmark

Shapes taken from the GEAK-eval ground-truth test function:
  test_ff_llama() in llama_ff_triton.py
    test_case_1: batch=2, seq_len=4, dim=64, w=(64,64)
    test_case_3: batch=3, seq_len=4, dim=64, w=(64,64)
    test_case_4: batch=2, seq_len=5, dim=64, w=(64,64)
"""

import argparse
import math
import os
import sys
import statistics

KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))
if KERNEL_DIR not in sys.path:
    sys.path.insert(0, KERNEL_DIR)

import torch

from llama_ff_triton import kernel_ff

# ============================================================================
# Shapes from the GEAK-eval ground-truth test: test_ff_llama()
# (batch, seq_len, dim) — w1/w3 are always (dim, dim), rms_w is (dim,)
# ============================================================================

ALL_SHAPES = [
    (2, 4, 64),   # test_case_1
    (3, 4, 64),   # test_case_3
    (2, 5, 64),   # test_case_4
]


HARNESS_SHAPES = ALL_SHAPES[:25]
PROFILE_SHAPES = ALL_SHAPES[:5]


def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reference_ff(x, w1, w3, rms_w, eps=1e-6):
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


def create_inputs(batch, seq_len, dim, device='cuda'):
    x = torch.randn((batch, seq_len, dim), dtype=torch.float16, device=device)
    w1 = torch.randn((dim, dim), dtype=torch.float16, device=device)
    w3 = torch.randn((dim, dim), dtype=torch.float16, device=device)
    rms_w = torch.randn((dim,), dtype=torch.float16, device=device)
    return x, w1, w3, rms_w


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


def run_correctness(shapes, atol=0.25, rtol=0.15):
    """Run correctness tests on the exact eval shapes."""
    print(f"Running correctness tests on {len(shapes)} shapes (atol={atol}, rtol={rtol})...")
    all_passed = True

    for batch, seq_len, dim in shapes:
        set_seed(42)
        x, w1, w3, rms_w = create_inputs(batch, seq_len, dim)

        out_triton = kernel_ff(x, w1, w3, rms_w)
        out_ref = reference_ff(x, w1, w3, rms_w)

        try:
            torch.testing.assert_close(out_triton, out_ref, rtol=rtol, atol=atol)
            print(f"  PASS: ({batch}, {seq_len}, {dim})")
        except AssertionError as e:
            print(f"  FAIL: ({batch}, {seq_len}, {dim}): {e}")
            all_passed = False

    if all_passed:
        print("\nAll correctness tests PASSED!")
    else:
        print("\nSome correctness tests FAILED!")
    return 0 if all_passed else 1


def run_profile(shapes, warmup=50):
    """Run kernel once per shape for profiling with proper warmup."""
    print(f"Running profile mode on {len(shapes)} shapes (warmup={warmup})...")
    for batch, seq_len, dim in shapes:
        set_seed(42)
        x, w1, w3, rms_w = create_inputs(batch, seq_len, dim)

        for _ in range(warmup):
            kernel_ff(x, w1, w3, rms_w)
        torch.cuda.synchronize()

        kernel_ff(x, w1, w3, rms_w)
        torch.cuda.synchronize()
        print(f"  Profiled: ({batch}, {seq_len}, {dim})")
    return 0


def run_benchmark(shapes, warmup=50, iterations=200):
    """Benchmark kernel vs reference; report per-shape speedups and geo-mean.
    Uses baseline Triton when benchmark_baseline.txt exists (patch eval); else PyTorch (preprocess)."""
    baseline_dir = _find_baseline_kernel_dir()
    kernel_dir = _resolve_geak_kernel_dir()
    if baseline_dir and baseline_dir != kernel_dir:
        ref_fn = _load_baseline_triton(baseline_dir, "baseline_llama_ff", "kernel_ff")
        ref_label = "baseline_triton"
    else:
        ref_fn = reference_ff
        ref_label = "ref"

    if ref_fn is None:
        ref_fn = reference_ff
        ref_label = "ref"

    print(f"Benchmarking {len(shapes)} shapes (warmup={warmup}, iterations={iterations})...")
    print(f"  Comparing kernel vs {ref_label}")
    print()

    speedups = []
    kernel_latencies = []

    for batch, seq_len, dim in shapes:
        set_seed(42)
        x, w1, w3, rms_w = create_inputs(batch, seq_len, dim)

        kernel_ms = benchmark_fn(
            lambda: kernel_ff(x, w1, w3, rms_w),
            warmup=warmup, iterations=iterations,
        )
        ref_ms = benchmark_fn(
            lambda: ref_fn(x, w1, w3, rms_w),
            warmup=warmup, iterations=iterations,
        )

        speedup = ref_ms / kernel_ms if kernel_ms > 0 else float('inf')
        speedups.append(speedup)
        kernel_latencies.append(kernel_ms)
        print(f"  ({batch}, {seq_len}, {dim}): kernel={kernel_ms:.4f} ms | ref={ref_ms:.4f} ms | speedup={speedup:.3f}x")

    geo_mean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
    median_latency = statistics.median(kernel_latencies)

    print()
    print(f"Geometric mean speedup: {geo_mean:.3f}x")
    print(f"Median kernel latency: {median_latency:.4f} ms")
    print(f"GEAK_RESULT_LATENCY_MS={median_latency:.6f}")
    print(f"GEAK_RESULT_GEOMEAN_SPEEDUP={geo_mean:.4f}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Test harness for llama_ff_triton kernel")
    parser.add_argument('--correctness', action='store_true', help='Run correctness tests')
    parser.add_argument('--profile', action='store_true', help='Run kernel once for profiling')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark on HARNESS_SHAPES')
    parser.add_argument('--full-benchmark', action='store_true', help='Run benchmark on ALL_SHAPES')
    parser.add_argument('--warmup', type=int, default=50,
                        help='Number of warmup iterations (default: 50)')
    parser.add_argument('--iterations', type=int,
                        default=int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200")),
                        help='Number of timed iterations (default: GEAK_BENCHMARK_ITERATIONS or 200)')
    parser.add_argument('--atol', type=float, default=0.25,
                        help='Absolute tolerance for correctness (default: 0.25)')
    parser.add_argument('--rtol', type=float, default=0.15,
                        help='Relative tolerance for correctness (default: 0.15)')

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


if __name__ == '__main__':
    main()
