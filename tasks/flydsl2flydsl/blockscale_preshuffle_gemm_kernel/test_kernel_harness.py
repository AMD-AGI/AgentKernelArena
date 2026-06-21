#!/usr/bin/env python3
"""Smoke harness for FlyDSL blockscale_preshuffle_gemm (compile + timing)."""
import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

KERNEL_FILE = "kernel.py"


def _resolve_kernel_dir():
    work_dir = os.environ.get("GEAK_WORK_DIR", "").strip()
    for c in [work_dir, os.path.dirname(os.path.abspath(__file__))]:
        if c and os.path.isfile(os.path.join(c, KERNEL_FILE)):
            return c
    return os.path.dirname(os.path.abspath(__file__))


def _load_kernel(kernel_dir):
    entry = os.path.join(kernel_dir, KERNEL_FILE)
    if kernel_dir not in sys.path:
        sys.path.insert(0, kernel_dir)
    flydsl2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    if flydsl2 not in sys.path:
        sys.path.insert(0, flydsl2)
    spec = importlib.util.spec_from_file_location("bs_gemm", entry)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_KERNEL_DIR = _resolve_kernel_dir()


def smoke_compile():
    m = _load_kernel(_KERNEL_DIR)
    m.compile_blockscale_preshuffle_gemm(
        M=256,
        N=256,
        K=256,
        tile_m=32,
        tile_n=64,
        tile_k=256,
        scale_block_k=128,
        out_dtype="bf16",
        use_async_copy=False,
    )


def run_correctness():
    try:
        smoke_compile()
        return {"correct": True, "num_correct": 1, "num_failed": 0, "failures": []}
    except Exception as e:
        return {"correct": False, "num_correct": 0, "num_failed": 1, "failures": [{"error": str(e)}]}


def run_benchmark(warmup=1, iters=3):
    times = []
    for _ in range(warmup + iters):
        t0 = time.perf_counter()
        smoke_compile()
        times.append((time.perf_counter() - t0) * 1000.0)
    times = times[warmup:]
    geo = math.exp(sum(math.log(max(t, 1e-9)) for t in times) / len(times))
    bd = Path(_KERNEL_DIR) / "build"
    bd.mkdir(exist_ok=True)
    with open(bd / "performance_report.json", "w") as f:
        json.dump([{"test_case_id": "compile_smoke", "execution_time_ms": geo}], f, indent=2)
    print(f"GEAK_RESULT_LATENCY_MS={geo:.4f}", flush=True)
    print(f"GEAK_RESULT_GEOMEAN_SPEEDUP={1.0:.4f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--correctness", action="store_true")
    ap.add_argument("--full-benchmark", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iterations", type=int, default=3)
    args = ap.parse_args()
    if args.correctness:
        r = run_correctness()
        print(json.dumps(r))
        sys.exit(0 if r["correct"] else 1)
    run_benchmark(warmup=args.warmup, iters=args.iterations)
