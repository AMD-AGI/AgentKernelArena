#!/usr/bin/env python3
"""
Test harness for fused_qkv_split_qk_rope kernel (aiter reference).

Modes: --correctness, --profile, --benchmark, --full-benchmark

This file is structurally identical to the test harness embedded in
kernel.py, except it imports the kernel from the aiter package rather
than using the inlined implementation.
"""
from __future__ import annotations
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
    aliases = ['fused_qkv_rope', 'aiter.ops.triton.fused_qkv_split_qk_rope', 'op_tests.triton_tests.test_fused_qk_concat', 'op_tests.test_rope']
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
from enum import IntEnum

import sys
import os

import torch

sys.path.insert(0, os.environ.get("AITER_ROOT", "/sgl-workspace/aiter"))

from aiter.ops.triton.fused_qkv_split_qk_rope import fused_qkv_split_qk_rope
from op_tests.triton_tests.test_fused_qk_concat import generate_rope_cached_freqs
from op_tests.test_rope import ref_rope_sbhd_fwd, RotateStyle


def triton_op(qkv, cos, sin, positions, qh, kvh, head_dim, is_neox,
              reuse_freqs_front_part, nope_first):
    return fused_qkv_split_qk_rope(
        qkv, cos, sin, positions, qh, kvh, head_dim,
        is_neox=is_neox, offsets=None,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=nope_first,
    )


# ============================================================================
# REFERENCE IMPLEMENTATIONS
# ============================================================================


def generate_qkv_inputs(
    B, QH_PER_KH, KH, D, nope, nope_first, dtype
):
    qkv = torch.randn(
        (B, (QH_PER_KH * KH + 2 * KH) * (D * (2 if nope else 1))),
        dtype=dtype,
        device="cuda",
    )
    return qkv


def torch_op(
    qkv,
    QH_PER_KH,
    KH,
    D,
    ref_freqs,
    reuse_freqs_front_part,
    nope,
    nope_first,
    rotate_style,
):
    q_size = QH_PER_KH * KH * D
    kv_size = KH * D
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    q = q.view(-1, QH_PER_KH * KH, D).contiguous()
    k = k.view(-1, KH, D).contiguous()
    v = v.view(-1, KH, D).contiguous()

    q = ref_rope_sbhd_fwd(
        q,
        ref_freqs,
        rotate_style=rotate_style,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=nope_first,
    )
    k = ref_rope_sbhd_fwd(
        k,
        ref_freqs,
        rotate_style=rotate_style,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=nope_first,
    )

    return q, k, v


# ============================================================================
# TEST CONFIGURATIONS
# ============================================================================

_B_VALUES = [1, 4, 8, 16, 32]
_QH_PER_KH_VALUES = [1, 2, 4, 8, 16]
_KH_VALUES = [1, 4]
_D_VALUES = [64, 128]
_ROTATE_STYLES = [RotateStyle.GPTJ, RotateStyle.NEOX]
_MAX_EMBED_POSITIONS = 131072
_NOPE_CONFIGS = [(False, False), (True, False), (True, True)]
_REUSE_FREQS = [False, True]
_DTYPE = torch.bfloat16

ALL_CONFIGS = []
for B in _B_VALUES:
    for QH_PER_KH in _QH_PER_KH_VALUES:
        for KH in _KH_VALUES:
            for D in _D_VALUES:
                for rotate_style in _ROTATE_STYLES:
                    for nope, nope_first in _NOPE_CONFIGS:
                        for reuse in _REUSE_FREQS:
                            ALL_CONFIGS.append(
                                (B, QH_PER_KH, KH, D, rotate_style, nope, nope_first, reuse)
                            )

# HARNESS_CONFIGS: use ALL configs so task-local and verified benchmarks match
HARNESS_CONFIGS = ALL_CONFIGS

_n_all = len(ALL_CONFIGS)
_profile_indices = [int(round(i * (_n_all - 1) / 4)) for i in range(5)]
PROFILE_CONFIGS = [ALL_CONFIGS[i] for i in _profile_indices]

# For backward compatibility
EVAL_CONFIGS = HARNESS_CONFIGS
PROFILE_SHAPES = PROFILE_CONFIGS

RTOL, ATOL = 1e-2, 1e-2


# ============================================================================
# TEST HARNESS
# ============================================================================


def _run_single_correctness(B, QH_PER_KH, KH, D, rotate_style, nope, nope_first,
                            reuse_freqs_front_part, dtype=_DTYPE):
    """Run a single correctness check. Returns (passed, error_msg)."""
    head_dim = D * (2 if nope else 1)
    qkv = generate_qkv_inputs(B, QH_PER_KH, KH, D, nope, nope_first, dtype)

    pos, freqs, cos, sin = generate_rope_cached_freqs(
        B, _MAX_EMBED_POSITIONS,
        (D // 2) if reuse_freqs_front_part else D,
        dtype,
    )
    ref_freqs = freqs[pos].squeeze(-2)

    q_triton, k_triton, v_triton = fused_qkv_split_qk_rope(
        qkv, cos, sin, pos,
        QH_PER_KH * KH, KH, head_dim,
        is_neox=(rotate_style == RotateStyle.NEOX),
        offsets=None,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=nope_first,
    )
    q_torch, k_torch, v_torch = torch_op(
        qkv, QH_PER_KH, KH, head_dim,
        ref_freqs, reuse_freqs_front_part, nope, nope_first, rotate_style,
    )

    torch.testing.assert_close(q_torch, q_triton, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(k_torch, k_triton, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(v_torch, v_triton, atol=ATOL, rtol=RTOL)


def run_correctness(configs=None, verbose=True):
    if configs is None:
        configs = HARNESS_CONFIGS
    print(f"Running correctness on {len(configs)} configs...")
    results, failures = [], []
    for idx, (B, QH_PER_KH, KH, D, rs, nope, nope_first, reuse) in enumerate(configs):
        tag = f"B={B} QH_PER_KH={QH_PER_KH} KH={KH} D={D} rs={rs.name} nope={nope} nope_first={nope_first} reuse={reuse}"
        try:
            _run_single_correctness(B, QH_PER_KH, KH, D, rs, nope, nope_first, reuse)
            results.append(tag)
            if verbose:
                print(f"  PASS: {tag}")
        except Exception as e:
            failures.append({"config": tag, "error": str(e)})
            if verbose:
                print(f"  FAIL: {tag} - {str(e)[:60]}")
        torch.cuda.empty_cache()

    if verbose:
        print("-" * 62)
        status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{len(configs)})"
        print(f"{'Status:':<22} {status}")

    return {
        "correct": len(failures) == 0,
        "num_correct": len(results),
        "num_failed": len(failures),
        "failures": failures,
    }


def run_profile(configs=None, warmup=50, iters=200, verbose=True):
    if configs is None:
        configs = PROFILE_CONFIGS
    if verbose:
        print(f"Profile: {len(configs)} config(s), {warmup} warmup, {iters} iter(s)")

    dtype = _DTYPE
    for B, QH_PER_KH, KH, D, rs, nope, nope_first, reuse in configs:
        head_dim = D * (2 if nope else 1)
        qkv = generate_qkv_inputs(B, QH_PER_KH, KH, D, nope, nope_first, dtype)
        pos, freqs, cos, sin = generate_rope_cached_freqs(
            B, _MAX_EMBED_POSITIONS, (D // 2) if reuse else D, dtype,
        )
        for _ in range(warmup):
            fused_qkv_split_qk_rope(
                qkv, cos, sin, pos, QH_PER_KH * KH, KH, head_dim,
                is_neox=(rs == RotateStyle.NEOX), reuse_freqs_front_part=reuse,
                nope_first=nope_first,
            )
        torch.cuda.synchronize()
        for _ in range(iters):
            fused_qkv_split_qk_rope(
                qkv, cos, sin, pos, QH_PER_KH * KH, KH, head_dim,
                is_neox=(rs == RotateStyle.NEOX), reuse_freqs_front_part=reuse,
                nope_first=nope_first,
            )
        torch.cuda.synchronize()
        if verbose:
            print(f"  B={B} QH_PER_KH={QH_PER_KH} KH={KH} D={D} rs={rs.name} done")
        del qkv
        torch.cuda.empty_cache()


def run_benchmark(configs=None, warmup=50, iters=200, verbose=True):
    """Benchmark kernel vs reference. Uses baseline Triton when available; else PyTorch."""
    if configs is None:
        configs = HARNESS_CONFIGS
    dtype = _DTYPE
    baseline_dir = _find_baseline_kernel_dir()
    kernel_dir = _resolve_geak_kernel_dir()
    baseline_fn = None
    if baseline_dir and baseline_dir != kernel_dir:
        baseline_fn = _load_baseline_triton(baseline_dir, "baseline_fused_qkv", "fused_qkv_split_qk_rope")
    ref_label = "baseline_triton" if baseline_fn else "PyTorch"

    latencies = []
    speedups = []
    results = []
    benchmark_methods = []

    print(f"Running benchmark on {len(configs)} configs, {warmup} warmup, {iters} iterations each...")
    print(f"  Comparing kernel vs {ref_label}")
    if verbose:
        print(f"{'Config':<50} {'Ref':>10} {'Triton':>10} {'Speedup':>10}")
        print("-" * 90)

    for B, QH_PER_KH, KH, D, rs, nope, nope_first, reuse in configs:
        head_dim = D * (2 if nope else 1)
        qkv = generate_qkv_inputs(B, QH_PER_KH, KH, D, nope, nope_first, dtype)
        pos, freqs, cos, sin = generate_rope_cached_freqs(
            B, _MAX_EMBED_POSITIONS, (D // 2) if reuse else D, dtype,
        )
        ref_freqs = freqs[pos].squeeze(-2)

        def run_kernel():
            return fused_qkv_split_qk_rope(
                qkv, cos, sin, pos, QH_PER_KH * KH, KH, head_dim,
                is_neox=(rs == RotateStyle.NEOX), reuse_freqs_front_part=reuse,
                nope_first=nope_first,
            )

        triton_ms, triton_meta = benchmark_cuda_graph_or_events(
            run_kernel, warmup=warmup, repetition=iters,
        )

        def run_reference():
            if baseline_fn is not None:
                return baseline_fn(
                    qkv, cos, sin, pos, QH_PER_KH * KH, KH, head_dim,
                    is_neox=(rs == RotateStyle.NEOX), reuse_freqs_front_part=reuse,
                    nope_first=nope_first,
                )
            return torch_op(
                qkv, QH_PER_KH, KH, head_dim, ref_freqs,
                reuse, nope, nope_first, rs,
            )

        ref_ms, ref_meta = benchmark_cuda_graph_or_events(
            run_reference, warmup=warmup, repetition=iters,
        )
        methods_match = triton_meta["benchmark_method"] == ref_meta["benchmark_method"]
        speedup = ref_ms / triton_ms if methods_match and triton_ms > 0 else None
        latencies.append(triton_ms)
        if speedup is not None:
            speedups.append(speedup)
        benchmark_methods.append(triton_meta["benchmark_method"])

        tag = f"B={B} QH={QH_PER_KH} KH={KH} D={D} {rs.name} nope={nope}"
        results.append({
            "config": tag,
            "ref_ms": ref_ms,
            "triton_ms": triton_ms,
            "speedup": speedup,
            **triton_meta,
            "reference_benchmark_method": ref_meta["benchmark_method"],
            "benchmark_method_consistent": methods_match,
        })

        if verbose:
            marker = " *" if speedup is not None and speedup > 1.0 else ""
            speedup_text = f"{speedup:.2f}x" if speedup is not None else "N/A"
            print(f"{tag:<50} {ref_ms:>8.4f}ms {triton_ms:>8.4f}ms {speedup_text:>9s}{marker}")

        del qkv
        torch.cuda.empty_cache()

    log_sum = sum(math.log(t) for t in latencies)
    geomean_latency = math.exp(log_sum / len(latencies))

    methods_consistent = len(speedups) == len(latencies)
    geomean_speedup = (
        math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        if methods_consistent else None
    )

    if verbose:
        print("-" * 90)
        print(f"{'Geometric mean latency:':<50} {geomean_latency:.4f} ms")
        print(
            f"{'Geometric mean speedup:':<50} {geomean_speedup:.2f}x"
            if geomean_speedup is not None else
            f"{'Geometric mean speedup:':<50} N/A (timing methods differ)"
        )
        print(f"GEAK_RESULT_LATENCY_MS={geomean_latency:.4f}")
        if geomean_speedup is not None:
            print(f"GEAK_RESULT_GEOMEAN_SPEEDUP={geomean_speedup:.4f}")

    print(f"GEAK_BENCHMARK_METHOD_CONSISTENT={int(methods_consistent)}")

    print("GEAK_BENCHMARK_METHOD={}".format(
        benchmark_methods[0] if len(set(benchmark_methods)) == 1
        else "mixed:" + ",".join(sorted(set(benchmark_methods)))
    ))

    return {
        "geomean_latency_ms": geomean_latency,
        "geomean_speedup": geomean_speedup,
        "results": results,
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fused QKV Split + QK RoPE Kernel Test Harness")
    parser.add_argument(
        "--correctness",
        action="store_true",
        help="Run correctness tests on benchmark configs",
    )
    parser.add_argument(
        "--profile", action="store_true", help="Run minimal profiling workload"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark on HARNESS_CONFIGS (25 uniformly sampled)",
    )
    parser.add_argument(
        "--full-benchmark",
        action="store_true",
        help="Run benchmark on ALL_CONFIGS (complete set)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of benchmark iterations",
    )
    args = parser.parse_args()

    print("=" * 62)
    print("Fused QKV Split + QK RoPE Kernel Test Harness")
    print("=" * 62)

    if args.correctness:
        print("\n[Correctness Mode]")
        run_correctness(HARNESS_CONFIGS)
    elif args.profile:
        print("\n[Profile Mode]")
        warmup = args.warmup if args.warmup is not None else 50
        iters = args.iterations if args.iterations is not None else 200
        run_profile(PROFILE_CONFIGS, warmup=warmup, iters=iters)
    elif args.full_benchmark:
        print("\n[Full Benchmark Mode]")
        warmup = args.warmup if args.warmup is not None else 50
        iters = args.iterations if args.iterations is not None else int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200"))
        run_benchmark(ALL_CONFIGS, warmup=warmup, iters=iters)
    else:
        # Default: benchmark (harness configs = all configs, reduced iters for 600 shapes)
        print("\n[Benchmark Mode]")
        warmup = args.warmup if args.warmup is not None else 5
        iters = args.iterations if args.iterations is not None else int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "10"))
        run_benchmark(HARNESS_CONFIGS, warmup=warmup, iters=iters)

    print("=" * 62)
