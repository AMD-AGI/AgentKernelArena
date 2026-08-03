#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Test harness for the torch2flydsl batched_gemm_a8w8 task.

The op is a batched INT8 a8w8 GEMM (per-batch
``out[b] = (x[b] @ w[b].T) * (x_scale[b] @ w_scale[b])``) with a per-token
activation scale, a per-output-channel weight scale, fp32 accumulation and a bf16
output.

Correctness is the (b)-faithful PyTorch reference in model.py compared against the
real AMD runtime op (``aiter.batched_gemm_a8w8_CK``) over byte-identical INT8
operands produced by ``model.quantize_batched_a8w8``. The gate is the normalized
worst-element error (``max|ref-gt| / max|ref| <= 1e-2``). When the FlyDSL
kernel.py exists it is additionally validated against the reference.

Modes:
  --correctness     compare the model.py reference to the aiter ground truth
  --full-benchmark  time the FlyDSL kernel (or the aiter op when no kernel.py),
                    write build/performance_report.json
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from _aka_benchmark import benchmark_cuda_graph_or_events

KERNEL_FILE = "kernel.py"
MODEL_FILE = "model.py"
KERNEL_ENTRY = "flydsl_batched_gemm_a8w8"


def _resolve_kernel_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.isfile(os.path.join(here, KERNEL_FILE)):
        return here
    cwd = os.getcwd()
    if os.path.isfile(os.path.join(cwd, KERNEL_FILE)):
        return cwd
    return here


def _load_module(kernel_dir, filename, alias):
    entry = os.path.join(kernel_dir, filename)
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

# Real batched a8w8 GEMM shapes (B, M, N, K) from
# configs/a8w8_untuned_batched_gemm.csv (B=16 qkv/attn projections).
SHAPES = [
    {"name": "b16_m32_n1280_k8192", "b": 16, "m": 32, "n": 1280, "k": 8192},
    {"name": "b16_m128_n1280_k8192", "b": 16, "m": 128, "n": 1280, "k": 8192},
    {"name": "b16_m64_n8192_k1024", "b": 16, "m": 64, "n": 8192, "k": 1024},
    {"name": "b16_m256_n8192_k1024", "b": 16, "m": 256, "n": 8192, "k": 1024},
    {"name": "b16_m512_n1280_k8192", "b": 16, "m": 512, "n": 1280, "k": 8192},
]

# Quantized GEMM gate: normalized worst-element error vs the aiter ground truth.
TOL = 1e-2
SEED = 20260401


def _retry(fn, tries=5, what="kernel call"):
    """Retry on transient out-of-memory / HIP errors (shared-GPU friendly)."""
    import torch

    last = None
    for attempt in range(tries):
        try:
            return fn()
        except RuntimeError as exc:  # noqa: PERF203
            msg = str(exc).lower()
            if "out of memory" not in msg and "hip" not in msg:
                raise
            last = exc
            torch.cuda.empty_cache()
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{what} failed after {tries} retries: {last}")


def _make_inputs(b, m, n, k, device="cuda"):
    import torch

    gen = torch.Generator(device=device)
    gen.manual_seed(SEED)
    x = torch.randn((b, m, k), generator=gen, device=device, dtype=torch.bfloat16)
    w = torch.randn((b, n, k), generator=gen, device=device, dtype=torch.bfloat16)
    return x, w


def _norm_worst(ref, out):
    rf, of = ref.float(), out.float()
    worst = (rf - of).abs().max().item()
    denom = rf.abs().max().item()
    denom = denom if denom > 0 else 1.0
    return worst, worst / denom


def run_correctness(verbose=True):
    import torch
    import aiter

    mmod = _load_module(_KERNEL_DIR, MODEL_FILE, "torch_model")
    if mmod is None:
        print("FAIL: cannot load model.py")
        assert False, "cannot load model.py"
    kmod = _load_module(_KERNEL_DIR, KERNEL_FILE, "flydsl_kernel")

    model = mmod.Model().to("cuda").eval()
    has_kernel = kmod is not None and hasattr(kmod, KERNEL_ENTRY)

    failures = []
    for shape in SHAPES:
        b, m, n, k = shape["b"], shape["m"], shape["n"], shape["k"]
        try:
            x, w = _make_inputs(b, m, n, k)
            with torch.no_grad():
                ref = model(x, w)

            x_i8, x_scale, w_i8, w_scale = mmod.quantize_batched_a8w8(x, w)
            gt = _retry(
                lambda: aiter.batched_gemm_a8w8_CK(
                    x_i8, w_i8, x_scale, w_scale, None
                ),
                what="aiter batched_gemm_a8w8",
            )
            torch.cuda.synchronize()

            worst, norm = _norm_worst(ref, gt)
            ok = norm <= TOL
            note = ""
            if has_kernel:
                try:
                    out = _retry(
                        lambda: kmod.flydsl_batched_gemm_a8w8(x, w),
                        what="flydsl kernel",
                    )
                except NotImplementedError:
                    has_kernel = False
                    print(
                        "  SKIP: kernel.py FlyDSL target not implemented yet "
                        "(reference validated against the aiter op above)"
                    )
                else:
                    torch.cuda.synchronize()
                    _, knorm = _norm_worst(ref, out)
                    kok = knorm <= TOL
                    ok = ok and kok
                    note = f" | kernel norm={knorm:.4g} {'ok' if kok else 'BAD'}"

            if verbose:
                print(
                    f"  {'PASS' if ok else 'FAIL'}: {shape['name']} "
                    f"({b}x{m}x{n}x{k}) worst={worst:.4g} norm={norm:.4g} "
                    f"tol={TOL}{note}"
                )
            if not ok:
                failures.append(shape["name"])
            del x, w, x_i8, x_scale, w_i8, w_scale, gt
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            failures.append(shape["name"])
            if verbose:
                print(f"  FAIL: {shape['name']} - {str(e)[:160]}")

    status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{len(SHAPES)})"
    print(f"Status: {status}")
    print(f"correctness: {'pass' if not failures else 'fail'}")
    assert not failures, f"correctness FAILED for: {failures}"
    return True


def run_benchmark(warmup=10, iters=100, verbose=True):
    import torch
    import aiter

    mmod = _load_module(_KERNEL_DIR, MODEL_FILE, "torch_model")
    kmod = _load_module(_KERNEL_DIR, KERNEL_FILE, "flydsl_kernel")
    has_kernel = kmod is not None and hasattr(kmod, KERNEL_ENTRY)

    if has_kernel:
        s0 = SHAPES[0]
        x0, w0 = _make_inputs(s0["b"], s0["m"], s0["n"], s0["k"])
        try:
            kmod.flydsl_batched_gemm_a8w8(x0, w0)
        except NotImplementedError:
            has_kernel = False
            print(
                "SKIP: kernel.py FlyDSL target not implemented yet "
                "(benchmarking aiter op instead)"
            )
        del x0, w0
        torch.cuda.empty_cache()

    def device_op(x, w):
        if has_kernel:
            return kmod.flydsl_batched_gemm_a8w8(x, w)
        x_i8, x_scale, w_i8, w_scale = mmod.quantize_batched_a8w8(x, w)
        return aiter.batched_gemm_a8w8_CK(x_i8, w_i8, x_scale, w_scale, None)

    label = "FlyDSL" if has_kernel else "aiter"
    latencies, speedups, report = [], [], []
    print(f"{'Config (B,M,N,K)':<30} {'Ref':>10} {label:>10} {'Speedup':>10}")
    print("-" * 64)
    for idx, shape in enumerate(SHAPES):
        b, m, n, k = shape["b"], shape["m"], shape["n"], shape["k"]
        x, w = _make_inputs(b, m, n, k)

        _retry(lambda: device_op(x, w), what="benchmark warmup")
        torch.cuda.synchronize()
        for _ in range(warmup):
            device_op(x, w)
        torch.cuda.synchronize()

        kernel_ms, kernel_bench_meta = benchmark_cuda_graph_or_events(
            lambda: device_op(x, w), warmup=0, repetition=iters
        )

        ref_ms, ref_bench_meta = benchmark_cuda_graph_or_events(
            lambda: torch.bmm(x.float(), w.float().transpose(1, 2)), warmup=0, repetition=iters
        )

        methods_match = kernel_bench_meta["benchmark_method"] == ref_bench_meta["benchmark_method"]
        speedup = (
            ref_ms / kernel_ms if methods_match and kernel_ms > 0 else None
        )
        speedup_display = (
            format(speedup, ">8.2f") + "x"
            if speedup is not None
            else f"{'N/A':>9}"
        )
        latencies.append(kernel_ms)
        if speedup is not None:
            speedups.append(speedup)
        tflops = 2.0 * b * m * n * k / (kernel_ms * 1e-3) / 1e12
        report.append({
            "test_case_id": f"test_case_{idx}",
            "execution_time_ms": kernel_ms,
            **kernel_bench_meta,
            "reference_benchmark_method": ref_bench_meta["benchmark_method"],
            "benchmark_method_consistent": kernel_bench_meta["benchmark_method"] == ref_bench_meta["benchmark_method"],
            "shape": [b, m, n, k],
            "params": {"B": b, "M": m, "N": n, "K": k, "dtype": "int8"},
            "tflops": tflops,
        })
        if verbose:
            print(
                f"(B={b:>2}, M={m:>4}, N={n:>5}, K={k:>5}) {ref_ms:>8.4f}ms "
                f"{kernel_ms:>8.4f}ms {speedup_display}"
            )
        del x, w
        torch.cuda.empty_cache()

    geomean_latency = math.exp(sum(math.log(x) for x in latencies) / len(latencies))
    geomean_speedup = math.exp(sum(math.log(x) for x in speedups) / len(speedups)) if speedups else None
    geomean_speedup_display = (
        format(geomean_speedup, ".2f") + "x"
        if geomean_speedup is not None
        else "N/A"
    )

    build_dir = Path(_KERNEL_DIR) / "build"
    build_dir.mkdir(exist_ok=True)
    with open(build_dir / "performance_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("-" * 64)
    print(f"Geometric mean latency: {geomean_latency:.4f} ms")
    print(f"Geometric mean speedup: {geomean_speedup_display}")
    return {"geomean_latency_ms": geomean_latency, "geomean_speedup": geomean_speedup}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="torch2flydsl batched_gemm_a8w8 harness")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    print("=" * 64)
    print("torch2flydsl batched GEMM a8w8 (INT8)")
    print("=" * 64)

    if args.correctness:
        try:
            run_correctness()
        except AssertionError as exc:
            print(f"ASSERTION: {exc}")
            sys.exit(1)
        sys.exit(0)
    else:
        run_benchmark(warmup=args.warmup, iters=args.iterations)
