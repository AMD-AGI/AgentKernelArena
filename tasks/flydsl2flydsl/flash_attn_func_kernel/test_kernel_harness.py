#!/usr/bin/env python3
"""Test harness for FlyDSL flash_attn_func_kernel (flydsl2flydsl)."""
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
# Test shapes: (batch, seq_len, num_heads, head_dim, dtype_str, causal)
# ============================================================================

ALL_SHAPES = [
    (1, 128, 8, 128, "f16", True),
    (1, 256, 8, 128, "f16", True),
    (1, 512, 8, 128, "f16", True),
    (1, 1024, 8, 128, "f16", True),
    (1, 2048, 8, 128, "f16", True),
    (4, 512, 16, 128, "f16", True),
    (8, 256, 32, 128, "f16", True),
    (8, 512, 64, 128, "f16", True),
    (1, 512, 8, 128, "bf16", True),
    (1, 1024, 8, 128, "bf16", True),
]

_n_all = len(ALL_SHAPES)
if _n_all <= 25:
    HARNESS_SHAPES = ALL_SHAPES
else:
    _idx = [int(round(i * (_n_all - 1) / 24)) for i in range(25)]
    HARNESS_SHAPES = [ALL_SHAPES[i] for i in _idx]

_pidx = [int(round(i * (_n_all - 1) / 4)) for i in range(5)]
PROFILE_SHAPES = [ALL_SHAPES[i] for i in _pidx]

RTOL, ATOL = 1e-2, 1e-2
# bf16 accumulation needs a looser bound than fp16 (matches upstream test_flash_attn_func.py).
ATOL_BY_DTYPE = {"f16": 1e-2, "bf16": 3e-2}

# ============================================================================
# Reference
# ============================================================================


def reference_flash_attn(q_4d, k_4d, v_4d, causal=True):
    import torch
    import torch.nn.functional as F

    q_t = q_4d.transpose(1, 2).float()
    k_t = k_4d.transpose(1, 2).float()
    v_t = v_4d.transpose(1, 2).float()
    out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=causal)
    return out.transpose(1, 2)


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
    for i, (B, S, H, D, dtype_str, causal) in enumerate(shapes):
        try:
            torch_dtype = dtype_map[dtype_str]
            torch.manual_seed(42 + i)

            exe = mod.build_flash_attn_func_module(
                num_heads=H, head_dim=D, causal=causal, dtype_str=dtype_str,
            )

            q_4d = torch.randn(B, S, H, D, dtype=torch_dtype, device="cuda")
            k_4d = torch.randn(B, S, H, D, dtype=torch_dtype, device="cuda")
            v_4d = torch.randn(B, S, H, D, dtype=torch_dtype, device="cuda")

            q_flat = q_4d.contiguous().view(-1)
            k_flat = k_4d.contiguous().view(-1)
            v_flat = v_4d.contiguous().view(-1)
            o_flat = torch.zeros_like(q_flat)

            exe(q_flat, k_flat, v_flat, o_flat, B, S)
            torch.cuda.synchronize()

            ref = reference_flash_attn(q_4d, k_4d, v_4d, causal=causal).to(torch_dtype)
            ref_flat = ref.contiguous().view(-1)

            tol = ATOL_BY_DTYPE.get(dtype_str, ATOL)
            max_err = (o_flat.float() - ref_flat.float()).abs().max().item()
            passed = max_err < tol
            if not passed:
                raise AssertionError(f"max_err={max_err:.4e} > {tol}")

            results.append({"config": (B, S, H, D, dtype_str), "correct": True})
            if verbose:
                causal_tag = "causal" if causal else "nocausal"
                print(f"  PASS: (B={B}, S={S}, H={H}, D={D}, {dtype_str}, {causal_tag}) max_err={max_err:.4e}")
        except Exception as e:
            failures.append({"config": (B, S, H, D, dtype_str), "error": str(e)})
            if verbose:
                print(f"  FAIL: (B={B}, S={S}, H={H}, D={D}, {dtype_str}) - {str(e)[:80]}")

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
    for B, S, H, D, dtype_str, causal in shapes:
        torch_dtype = dtype_map[dtype_str]
        exe = mod.build_flash_attn_func_module(
            num_heads=H, head_dim=D, causal=causal, dtype_str=dtype_str,
        )
        q = torch.randn(B * S * H * D, dtype=torch_dtype, device="cuda")
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        o = torch.zeros_like(q)

        for _ in range(warmup):
            exe(q, k, v, o, B, S)
        torch.cuda.synchronize()
        for _ in range(iters):
            exe(q, k, v, o, B, S)
        torch.cuda.synchronize()
        if verbose:
            causal_tag = "causal" if causal else "nocausal"
            print(f"  (B={B}, S={S}, H={H}, D={D}, {dtype_str}, {causal_tag}) done")


def run_benchmark(shapes=None, warmup=10, iters=100, verbose=True):
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
    print(f"{'Config':<42} {'Ref':>10} {'FlyDSL':>10} {'Speedup':>10}")
    print("-" * 76)

    for idx, (B, S, H, D, dtype_str, causal) in enumerate(shapes):
        torch_dtype = dtype_map[dtype_str]
        torch.manual_seed(42)

        exe = mod.build_flash_attn_func_module(
            num_heads=H, head_dim=D, causal=causal, dtype_str=dtype_str,
        )

        q_4d = torch.randn(B, S, H, D, dtype=torch_dtype, device="cuda")
        k_4d = torch.randn(B, S, H, D, dtype=torch_dtype, device="cuda")
        v_4d = torch.randn(B, S, H, D, dtype=torch_dtype, device="cuda")
        q_flat = q_4d.contiguous().view(-1)
        k_flat = k_4d.contiguous().view(-1)
        v_flat = v_4d.contiguous().view(-1)
        o_flat = torch.zeros_like(q_flat)

        for _ in range(warmup):
            exe(q_flat, k_flat, v_flat, o_flat, B, S)
        torch.cuda.synchronize()

        kernel_times = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            exe(q_flat, k_flat, v_flat, o_flat, B, S)
            e.record()
            torch.cuda.synchronize()
            kernel_times.append(s.elapsed_time(e))
        kernel_ms = sum(kernel_times) / len(kernel_times)

        ref_times = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            _ = reference_flash_attn(q_4d, k_4d, v_4d, causal=causal).to(torch_dtype)
            e.record()
            torch.cuda.synchronize()
            ref_times.append(s.elapsed_time(e))
        ref_ms = sum(ref_times) / len(ref_times)

        speedup = ref_ms / kernel_ms if kernel_ms > 0 else 1.0
        latencies.append(kernel_ms)
        speedups.append(speedup)

        s_eff = S / 2.0 if causal else float(S)
        flops = 4.0 * S * s_eff * D * H * B
        tflops = flops / (kernel_ms * 1e-3) / 1e12

        report_cases.append({
            "test_case_id": f"test_case_{idx}",
            "execution_time_ms": kernel_ms,
            "shape": [B, S, H, D],
            "params": {"B": B, "S": S, "H": H, "D": D, "dtype": dtype_str, "causal": causal},
            "tflops": tflops,
        })

        marker = " *" if speedup > 1.0 else ""
        causal_tag = "causal" if causal else "nocausal"
        if verbose:
            print(
                f"(B={B:>2},S={S:>5},H={H:>3},D={D},{dtype_str},{causal_tag})"
                f" {ref_ms:>8.4f}ms {kernel_ms:>8.4f}ms {speedup:>8.2f}x{marker}",
                flush=True,
            )

        del q_4d, k_4d, v_4d, q_flat, k_flat, v_flat, o_flat
        torch.cuda.empty_cache()

    geomean_latency = math.exp(sum(math.log(l) for l in latencies) / len(latencies))
    geomean_speedup = math.exp(sum(math.log(s) for s in speedups) / len(speedups))

    build_dir = Path(_KERNEL_DIR) / "build"
    build_dir.mkdir(exist_ok=True)
    with open(build_dir / "performance_report.json", "w") as f:
        json.dump(report_cases, f, indent=2)

    print("-" * 76)
    print(f"{'Geometric mean latency:':<26} {geomean_latency:.4f} ms")
    print(f"{'Geometric mean speedup:':<26} {geomean_speedup:.2f}x")
    print(f"GEAK_RESULT_LATENCY_MS={geomean_latency:.4f}", flush=True)
    print(f"GEAK_RESULT_GEOMEAN_SPEEDUP={geomean_speedup:.4f}", flush=True)

    return {"geomean_latency_ms": geomean_latency, "geomean_speedup": geomean_speedup}


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlyDSL Flash Attention Kernel Test Harness")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--iterations",
        type=int,
        default=int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "100")),
    )
    args = parser.parse_args()

    print("=" * 62)
    print("FlyDSL Flash Attention Kernel")
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
