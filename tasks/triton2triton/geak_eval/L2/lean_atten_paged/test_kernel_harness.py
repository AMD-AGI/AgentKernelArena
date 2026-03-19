#!/usr/bin/env python3
"""
Lean Attention + Paged Attention kernel test harness.

Wraps the built-in harness in kernel.py to ensure:
- --correctness exits non-zero on failure
- --iterations reads GEAK_BENCHMARK_ITERATIONS env var
- --benchmark uses HARNESS_CONFIGS
- --full-benchmark uses ALL_CONFIGS
- --profile uses PROFILE_CONFIGS
- GEAK_RESULT_LATENCY_MS is always the LAST line of benchmark output

Modes:
  --correctness    : validate kernel against torch reference
  --profile        : run kernel once per PROFILE_SHAPES for profiler capture
  --benchmark      : benchmark on HARNESS_CONFIGS, print GEAK_RESULT_LATENCY_MS
  --full-benchmark : benchmark on ALL_CONFIGS, print GEAK_RESULT_LATENCY_MS
  --iterations N   : override iteration count (default from GEAK_BENCHMARK_ITERATIONS or 20)
"""
from __future__ import annotations

import argparse
import os
import sys

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
    aliases = ['lean_atten_paged']
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

from kernel import (
    run_correctness,
    run_profile,
    run_benchmark,
    CORRECTNESS_CONFIGS,
    HARNESS_CONFIGS,
    ALL_CONFIGS,
    PROFILE_CONFIGS,
)


def _get_baseline_fn():
    """Resolve baseline Triton kernel when in patch-eval mode."""
    baseline_dir = _find_baseline_kernel_dir()
    kernel_dir = _resolve_geak_kernel_dir()
    if baseline_dir and baseline_dir != kernel_dir:
        return _load_baseline_triton(baseline_dir, "baseline_lean_atten", "persistent_lean_attention_paged")
    return None


def main():
    default_iters = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200"))

    parser = argparse.ArgumentParser(
        description="Lean Attention + Paged Attention Kernel Test Harness"
    )
    parser.add_argument("--correctness", action="store_true",
                        help="Run correctness tests")
    parser.add_argument("--profile", action="store_true",
                        help="Run minimal profiling workload")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark on HARNESS_CONFIGS")
    parser.add_argument("--full-benchmark", action="store_true",
                        help="Run benchmark on ALL_CONFIGS")
    parser.add_argument("--iterations", type=int, default=default_iters,
                        help=f"Number of benchmark iterations (default: {default_iters})")
    parser.add_argument("--warmup", type=int, default=50,
                        help="Number of warmup iterations (default: 50)")
    args = parser.parse_args()

    if args.correctness:
        print("=" * 70)
        print("[Correctness Mode]")
        print("=" * 70)
        result = run_correctness(CORRECTNESS_CONFIGS, verbose=True)
        if not result["correct"]:
            print(f"\nFAILED: {result['num_failed']} correctness test(s) failed")
            sys.exit(1)
        print("\nAll correctness tests PASSED")
        sys.exit(0)

    elif args.profile:
        print("=" * 70)
        print("[Profile Mode]")
        print("=" * 70)
        run_profile(PROFILE_CONFIGS, warmup=args.warmup, iters=args.iterations,
                    verbose=True)
        sys.exit(0)

    elif args.full_benchmark:
        print("=" * 70)
        print("[Full Benchmark Mode]")
        print("=" * 70)
        baseline_fn = _get_baseline_fn()
        result = run_benchmark(ALL_CONFIGS, warmup=args.warmup,
                               iters=args.iterations, verbose=True, baseline_fn=baseline_fn)
        # Ensure GEAK_RESULT_LATENCY_MS is the LAST line of output
        print(f"GEAK_RESULT_LATENCY_MS={result['geomean_latency_ms']:.4f}")
        sys.exit(0)

    elif args.benchmark:
        print("=" * 70)
        print("[Benchmark Mode]")
        print("=" * 70)
        baseline_fn = _get_baseline_fn()
        result = run_benchmark(HARNESS_CONFIGS, warmup=args.warmup,
                               iters=args.iterations, verbose=True, baseline_fn=baseline_fn)
        # Ensure GEAK_RESULT_LATENCY_MS is the LAST line of output
        print(f"GEAK_RESULT_LATENCY_MS={result['geomean_latency_ms']:.4f}")
        sys.exit(0)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
