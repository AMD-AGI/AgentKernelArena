#!/usr/bin/env python3
"""Test harness for FlyDSL hgemm_splitk_kernel (flydsl2flydsl)."""
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
# Test shapes: (M, N, K, dtype_str, kwargs)
#
# Shapes + per-shape tuning kwargs come from FlyDSL v0.2.0
# tests/kernels/test_hgemm_splitk.py (the gfx942 parameter set). The kernel's
# get_default_kwargs() does NOT cover arbitrary square shapes on gfx942, so the
# explicit per-shape kwargs are required for the larger GEMMs to compile/run.
# kwargs order: TILE_M, TILE_N, TILE_K, STAGES, SPLIT_K, BLOCK_M_WARPS,
#               BLOCK_N_WARPS, BLOCK_K_WARPS
# ============================================================================


def _kw(TILE_M, TILE_N, TILE_K, STAGES, SPLIT_K, BM, BN, BK):
    return {
        "TILE_M": TILE_M, "TILE_N": TILE_N, "TILE_K": TILE_K,
        "STAGES": STAGES, "SPLIT_K": SPLIT_K,
        "BLOCK_M_WARPS": BM, "BLOCK_N_WARPS": BN, "BLOCK_K_WARPS": BK,
    }


# gfx942 (m, n, k, TILE_M, TILE_N, TILE_K, STAGES, SPLIT_K, BM, BN, BK)
_GFX942_CONFIGS = [
    (32, 384, 7168, 16, 64, 128, 2, 14, 1, 2, 1),
    (4, 384, 7168, 16, 64, 128, 2, 14, 1, 2, 1),
    (65, 1024, 8192, 48, 64, 128, 2, 8, 1, 2, 1),
    (8, 5120, 2880, 32, 128, 64, 2, 9, 2, 2, 1),
    (4096, 4096, 4096, 128, 128, 64, 2, 1, 2, 2, 1),
    (8192, 8192, 8192, 128, 128, 64, 2, 1, 2, 2, 1),
    (32, 2880, 2048, 32, 64, 128, 2, 4, 1, 2, 1),
]

ALL_SHAPES = []
for _dt in ("f16", "bf16"):
    for _m, _n, _k, *_p in _GFX942_CONFIGS:
        ALL_SHAPES.append((_m, _n, _k, _dt, _kw(*_p)))

_n_all = len(ALL_SHAPES)
if _n_all <= 25:
    HARNESS_SHAPES = ALL_SHAPES
else:
    _idx = [int(round(i * (_n_all - 1) / 24)) for i in range(25)]
    HARNESS_SHAPES = [ALL_SHAPES[i] for i in _idx]

_pidx = [int(round(i * (_n_all - 1) / 4)) for i in range(5)]
PROFILE_SHAPES = [ALL_SHAPES[i] for i in _pidx]

RTOL, ATOL = 1e-1, 1e-1

# ============================================================================
# Reference
# ============================================================================


def reference_gemm(a, b_t, dtype=None):
    """C = A @ B^T computed in float32 for accuracy."""
    import torch

    c = torch.mm(a.float(), b_t.float().T)
    if dtype is not None:
        c = c.to(dtype)
    return c


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

    dtype_map = {"f16": torch.float16, "bf16": torch.bfloat16}
    results, failures = [], []
    for i, (M, N, K, dtype_str, kw) in enumerate(shapes):
        try:
            torch_dtype = dtype_map[dtype_str]
            torch.manual_seed(42 + i)

            a = torch.randn(M, K, dtype=torch_dtype, device="cuda").uniform_(-1, 1)
            b = torch.randn(N, K, dtype=torch_dtype, device="cuda").uniform_(-1, 1)
            c = torch.zeros(M, N, dtype=torch_dtype, device="cuda")

            mod.hgemm_splitk_(c, a, b, None, kw, torch.cuda.current_stream())
            torch.cuda.synchronize()

            ref = reference_gemm(a, b, dtype=torch.float32)
            max_err = (c.float() - ref).abs().max().item()
            rel_err = max_err / (ref.abs().max().item() + 1e-6)

            if rel_err > RTOL:
                raise AssertionError(f"rel_err={rel_err:.4e} > {RTOL}")

            results.append({"config": (M, N, K, dtype_str), "correct": True})
            if verbose:
                print(f"  PASS: (M={M}, N={N}, K={K}, {dtype_str}) rel_err={rel_err:.4e}")
        except Exception as e:
            failures.append({"config": (M, N, K, dtype_str), "error": str(e)})
            if verbose:
                print(f"  FAIL: (M={M}, N={N}, K={K}, {dtype_str}) - {str(e)[:80]}")

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


def run_profile(shapes=None, warmup=10, iters=50, verbose=True):
    import torch

    if shapes is None:
        shapes = PROFILE_SHAPES
    if verbose:
        print(f"Profile: {len(shapes)} config(s), {warmup} warmup, {iters} iter(s)")

    mod = _load_kernel(_KERNEL_DIR)
    if mod is None:
        return

    dtype_map = {"f16": torch.float16, "bf16": torch.bfloat16}
    for M, N, K, dtype_str, kw in shapes:
        torch_dtype = dtype_map[dtype_str]
        a = torch.randn(M, K, dtype=torch_dtype, device="cuda")
        b = torch.randn(N, K, dtype=torch_dtype, device="cuda")
        c = torch.zeros(M, N, dtype=torch_dtype, device="cuda")

        for _ in range(warmup):
            mod.hgemm_splitk_(c, a, b, None, kw, torch.cuda.current_stream())
        torch.cuda.synchronize()
        for _ in range(iters):
            mod.hgemm_splitk_(c, a, b, None, kw, torch.cuda.current_stream())
        torch.cuda.synchronize()
        if verbose:
            print(f"  (M={M}, N={N}, K={K}, {dtype_str}) done")


def run_benchmark(shapes=None, warmup=10, iters=50, verbose=True):
    import torch

    if shapes is None:
        shapes = HARNESS_SHAPES

    mod = _load_kernel(_KERNEL_DIR)
    if mod is None:
        print("FAIL: cannot load kernel.py")
        return {"geomean_latency_ms": -1, "geomean_speedup": -1}

    dtype_map = {"f16": torch.float16, "bf16": torch.bfloat16}
    latencies, speedups, report_cases = [], [], []

    print(f"Running benchmark on {len(shapes)} shapes, {warmup} warmup, {iters} iterations...")
    print(f"{'Config (M,N,K,dtype)':<32} {'Ref':>10} {'FlyDSL':>10} {'Speedup':>10}")
    print("-" * 68)

    for idx, (M, N, K, dtype_str, kw) in enumerate(shapes):
        torch_dtype = dtype_map[dtype_str]
        torch.manual_seed(42)

        a = torch.randn(M, K, dtype=torch_dtype, device="cuda").uniform_(-1, 1)
        b = torch.randn(N, K, dtype=torch_dtype, device="cuda").uniform_(-1, 1)
        c = torch.zeros(M, N, dtype=torch_dtype, device="cuda")

        mod.hgemm_splitk_(c, a, b, None, kw, torch.cuda.current_stream())
        torch.cuda.synchronize()

        for _ in range(warmup):
            mod.hgemm_splitk_(c, a, b, None, kw, torch.cuda.current_stream())
        torch.cuda.synchronize()

        kernel_times = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            mod.hgemm_splitk_(c, a, b, None, kw, torch.cuda.current_stream())
            e.record()
            torch.cuda.synchronize()
            kernel_times.append(s.elapsed_time(e))
        kernel_ms = sorted(kernel_times)[len(kernel_times) // 2]

        ref_times = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            _ = torch.mm(a, b.T)
            e.record()
            torch.cuda.synchronize()
            ref_times.append(s.elapsed_time(e))
        ref_ms = sorted(ref_times)[len(ref_times) // 2]

        speedup = ref_ms / kernel_ms if kernel_ms > 0 else 1.0
        latencies.append(kernel_ms)
        speedups.append(speedup)

        flops = 2.0 * M * N * K
        tflops = flops / (kernel_ms * 1e-3) / 1e12

        report_cases.append({
            "test_case_id": f"test_case_{idx}",
            "execution_time_ms": kernel_ms,
            "shape": [M, N, K],
            "params": {"M": M, "N": N, "K": K, "dtype": dtype_str},
            "tflops": tflops,
        })

        marker = " *" if speedup > 1.0 else ""
        if verbose:
            print(
                f"(M={M:>5}, N={N:>5}, K={K:>5}, {dtype_str})"
                f" {ref_ms:>8.4f}ms {kernel_ms:>8.4f}ms {speedup:>8.2f}x{marker}",
                flush=True,
            )

        del a, b, c
        torch.cuda.empty_cache()

    geomean_latency = math.exp(sum(math.log(l) for l in latencies) / len(latencies))
    geomean_speedup = math.exp(sum(math.log(s) for s in speedups) / len(speedups))

    build_dir = Path(_KERNEL_DIR) / "build"
    build_dir.mkdir(exist_ok=True)
    with open(build_dir / "performance_report.json", "w") as f:
        json.dump(report_cases, f, indent=2)

    print("-" * 68)
    print(f"{'Geometric mean latency:':<26} {geomean_latency:.4f} ms")
    print(f"{'Geometric mean speedup:':<26} {geomean_speedup:.2f}x")
    print(f"GEAK_RESULT_LATENCY_MS={geomean_latency:.4f}", flush=True)
    print(f"GEAK_RESULT_GEOMEAN_SPEEDUP={geomean_speedup:.4f}", flush=True)

    return {"geomean_latency_ms": geomean_latency, "geomean_speedup": geomean_speedup}


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlyDSL HGEMM SplitK Kernel Test Harness")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--iterations",
        type=int,
        default=int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "50")),
    )
    args = parser.parse_args()

    print("=" * 62)
    print("FlyDSL HGEMM SplitK Kernel")
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
