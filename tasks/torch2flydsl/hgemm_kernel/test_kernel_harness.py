#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Test harness for the torch2flydsl hgemm task.

Builds the PyTorch reference from model.py (`Model`/`get_inputs`/
`get_init_inputs`) and runs the FlyDSL kernel from kernel.py over a set of GEMM
shapes, comparing for correctness and benchmarking against a torch baseline.

Modes:
  --correctness     compare FlyDSL output to the PyTorch Model reference
  --full-benchmark  time FlyDSL vs torch baseline, write performance report
"""
import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from _aka_benchmark import benchmark_cuda_graph_or_events

KERNEL_FILE = "kernel.py"
MODEL_FILE = "model.py"


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

# bf16 GEMM shapes: (M, N, K) plus per-case FlyDSL tiling that satisfies the
# kernel's tile constraints.
SHAPES = [
    {"name": "untuned_m64_n256_k5120", "m": 64, "n": 256, "k": 5120},
    {"name": "untuned_m256_n256_k5120", "m": 256, "n": 256, "k": 5120},
    {"name": "untuned_m512_n256_k5120", "m": 512, "n": 256, "k": 5120},
    {"name": "dsv3_m128_n3072_k1536", "m": 128, "n": 3072, "k": 1536},
    {"name": "dsv3_m64_n2112_k7168_tn64", "m": 64, "n": 2112, "k": 7168, "tile_n": 64},
]

# bf16 GEMM element-wise tolerance.
ATOL, RTOL, PASS_PCT = 1e-2, 1e-2, 99.9
SEED = 20260401
TILING_KEYS = ("tile_m", "tile_n", "tile_k", "split_k", "block_m_warps", "block_n_warps")


def _make_inputs(m, n, k, device="cuda"):
    import torch

    gen = torch.Generator(device=device)
    gen.manual_seed(SEED)
    a = torch.rand((m, k), generator=gen, device=device, dtype=torch.bfloat16)
    b = torch.rand((n, k), generator=gen, device=device, dtype=torch.bfloat16)
    return a, b


def run_correctness(verbose=True):
    import torch

    kmod = _load_module(_KERNEL_DIR, KERNEL_FILE, "flydsl_kernel")
    mmod = _load_module(_KERNEL_DIR, MODEL_FILE, "torch_model")
    if kmod is None or mmod is None:
        print("FAIL: cannot load kernel.py / model.py")
        return {"correct": False}

    init = mmod.get_init_inputs()
    model = mmod.Model().to("cuda").eval() if not init else mmod.Model(*init).to("cuda").eval()

    failures = []
    for shape in SHAPES:
        tiling = {k: shape[k] for k in TILING_KEYS if k in shape}
        try:
            a, b = _make_inputs(shape["m"], shape["n"], shape["k"])
            with torch.no_grad():
                ref = model(a, b)
            out = kmod.flydsl_hgemm(a, b, **tiling)
            torch.cuda.synchronize()

            ref_f, out_f = ref.float(), out.float()
            close = torch.isclose(ref_f, out_f, atol=ATOL, rtol=RTOL)
            pct = close.float().mean().item() * 100.0
            max_delta = (ref_f - out_f).abs().max().item()
            ok = pct >= PASS_PCT
            if verbose:
                print(
                    f"  {'PASS' if ok else 'FAIL'}: {shape['name']} "
                    f"({shape['m']}x{shape['n']}x{shape['k']}) "
                    f"{pct:.4f}% close, max_delta={max_delta:.4f}"
                )
            if not ok:
                failures.append(shape["name"])
        except Exception as e:  # noqa: BLE001
            failures.append(shape["name"])
            if verbose:
                print(f"  FAIL: {shape['name']} - {str(e)[:100]}")

    status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{len(SHAPES)})"
    print(f"Status: {status}")
    print(f"correctness: {'pass' if not failures else 'fail'}")
    assert not failures, f"correctness FAILED for: {failures}"
    return True


def run_benchmark(warmup=10, iters=100, verbose=True):
    import torch

    kmod = _load_module(_KERNEL_DIR, KERNEL_FILE, "flydsl_kernel")
    if kmod is None:
        print("FAIL: cannot load kernel.py")
        return {"geomean_latency_ms": -1, "geomean_speedup": -1}

    latencies, speedups, report = [], [], []
    print(f"{'Config (M,N,K)':<28} {'Ref':>10} {'FlyDSL':>10} {'Speedup':>10}")
    print("-" * 62)
    for idx, shape in enumerate(SHAPES):
        m, n, k = shape["m"], shape["n"], shape["k"]
        tiling = {kk: shape[kk] for kk in TILING_KEYS if kk in shape}
        a, b = _make_inputs(m, n, k)

        kmod.flydsl_hgemm(a, b, **tiling)
        torch.cuda.synchronize()
        for _ in range(warmup):
            kmod.flydsl_hgemm(a, b, **tiling)
        torch.cuda.synchronize()

        kernel_ms, kernel_bench_meta = benchmark_cuda_graph_or_events(
            lambda: kmod.flydsl_hgemm(a, b, **tiling), warmup=0, repetition=iters
        )

        ref_ms, ref_bench_meta = benchmark_cuda_graph_or_events(
            lambda: torch.mm(a, b.transpose(-1, -2)), warmup=0, repetition=iters
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
        tflops = 2.0 * m * n * k / (kernel_ms * 1e-3) / 1e12
        report.append({
            "test_case_id": f"test_case_{idx}",
            "execution_time_ms": kernel_ms,
            **kernel_bench_meta,
            "reference_benchmark_method": ref_bench_meta["benchmark_method"],
            "benchmark_method_consistent": kernel_bench_meta["benchmark_method"] == ref_bench_meta["benchmark_method"],
            "shape": [m, n, k],
            "params": {"M": m, "N": n, "K": k, "dtype": "bf16"},
            "tflops": tflops,
        })
        if verbose:
            print(f"(M={m:>4}, N={n:>5}, K={k:>5}) {ref_ms:>8.4f}ms {kernel_ms:>8.4f}ms {speedup_display}")
        del a, b
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

    print("-" * 62)
    print(f"Geometric mean latency: {geomean_latency:.4f} ms")
    print(f"Geometric mean speedup: {geomean_speedup_display}")
    return {"geomean_latency_ms": geomean_latency, "geomean_speedup": geomean_speedup}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="torch2flydsl hgemm harness")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    print("=" * 62)
    print("torch2flydsl HGEMM")
    print("=" * 62)

    if args.correctness:
        try:
            run_correctness()
        except AssertionError as exc:
            print(f"ASSERTION: {exc}")
            sys.exit(1)
        sys.exit(0)
    else:
        run_benchmark(warmup=args.warmup, iters=args.iterations)
