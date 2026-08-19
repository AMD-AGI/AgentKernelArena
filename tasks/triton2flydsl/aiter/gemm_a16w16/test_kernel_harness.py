#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Test harness for triton2flydsl/aiter/gemm_a16w16 (FLAT layout).

The kernel under test is AITER's 16-bit GEMM Triton kernel (`gemm_a16w16` ->
`_gemm_a16_w16_kernel`): Y = X @ W^T with fp32 accumulation, an XCD-balanced +
grouped pid remap, and bf16/fp16 output. The standalone source keeps the
non-split-K (NUM_KSPLIT == 1) triton path with a static tile config.

Modes:
  --compile         ast-parse + import the standalone source, assert entry/kernel symbols
  --correctness     run the Triton kernel on TEST_SHAPES, assert finite output AND
                    match torch F.linear at the upstream bf16 tolerance
                    (atol=1e-1, rtol=1e-2)  [mirrors test_gemm_a16w16.py:test_gemm_a16_w16]
  --full-benchmark  graph-first GPU timing, write build/performance_report.json
"""
import argparse
import ast
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from _aka_benchmark import benchmark_cuda_graph_or_events

SOURCE_FILE = "gemm_a16w16.py"
ENTRY = "gemm_a16w16"
KERNEL = "_gemm_a16_w16_kernel"

# (M, N, K) from op_tests/triton_tests/gemm/basic/test_gemm_a16w16.py::get_x_vals
# (real model GEMMs: DSR1 router, GPT-OSS-120B QKV/out/router projections) plus
# minimal/irregular edge cases.
TEST_SHAPES = [
    {"name": "m1_n1_k1", "M": 1, "N": 1, "K": 1},
    {"name": "m3_n5_k2", "M": 3, "N": 5, "K": 2},
    {"name": "m1024_n1024_k1024", "M": 1024, "N": 1024, "K": 1024},
    {"name": "m2048_n2048_k2048", "M": 2048, "N": 2048, "K": 2048},
    {"name": "dsr1_router_m32_n256_k7168", "M": 32, "N": 256, "K": 7168},
    {"name": "gptoss_qkv_m128_n5120_k2880", "M": 128, "N": 5120, "K": 2880},
    {"name": "gptoss_out_m256_n2880_k4096", "M": 256, "N": 2880, "K": 4096},
    {"name": "gptoss_router_m128_n128_k2880", "M": 128, "N": 128, "K": 2880},
]

SEED = 20260601
WARMUP, ITERS = 10, 100

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_source():
    entry = os.path.join(_HERE, SOURCE_FILE)
    spec = importlib.util.spec_from_file_location("gemm_a16w16_src", entry)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_inputs(M, N, K, device="cuda", dtype=None):
    import torch

    if dtype is None:
        dtype = torch.bfloat16
    gen = torch.Generator(device=device)
    gen.manual_seed(SEED)
    x = torch.randn((M, K), generator=gen, device=device, dtype=dtype)
    w = torch.randn((N, K), generator=gen, device=device, dtype=dtype)
    return x, w


def run_compile():
    with open(os.path.join(_HERE, SOURCE_FILE)) as f:
        ast.parse(f.read())
    mod = _load_source()
    assert hasattr(mod, ENTRY), f"Missing entry {ENTRY}"
    assert hasattr(mod, KERNEL), f"Missing kernel {KERNEL}"
    print("Compilation: PASS")
    return True


def run_correctness(verbose=True):
    import torch
    import torch.nn.functional as F

    mod = _load_source()
    failures = []
    for shape in TEST_SHAPES:
        tag = shape["name"]
        try:
            x, w = _make_inputs(shape["M"], shape["N"], shape["K"])
            y = mod.gemm_a16w16(x, w)
            torch.cuda.synchronize()
            ref = F.linear(x, w, bias=None)
            finite = bool(torch.isfinite(y).all().item())
            close = torch.allclose(y, ref, atol=1e-1, rtol=1e-2)
            ok = finite and close
            if verbose:
                print(
                    f"  {'PASS' if ok else 'FAIL'}: {tag} "
                    f"(M={shape['M']},N={shape['N']},K={shape['K']}) "
                    f"out={tuple(y.shape)} finite={finite} close={close}"
                )
            if not ok:
                failures.append(tag)
        except Exception as e:  # noqa: BLE001
            failures.append(tag)
            if verbose:
                print(f"  FAIL: {tag} - {str(e)[:160]}")

    status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{len(TEST_SHAPES)})"
    print(f"Status: {status}")
    print(f"correctness: {'pass' if not failures else 'fail'}")
    return not failures


def run_benchmark(verbose=True):
    import torch

    mod = _load_source()
    report, latencies = [], []
    for idx, shape in enumerate(TEST_SHAPES):
        x, w = _make_inputs(shape["M"], shape["N"], shape["K"])
        fn = lambda: mod.gemm_a16w16(x, w)  # noqa: E731
        fn()
        torch.cuda.synchronize()
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        ms, bench_meta = benchmark_cuda_graph_or_events(
            fn, warmup=0, repetition=ITERS
        )
        latencies.append(ms)
        flops = 2.0 * shape["M"] * shape["N"] * shape["K"]
        report.append(
            {
                "test_case_id": f"perf{idx + 1}",
                "execution_time_ms": ms,
                **bench_meta,
                "params": {k: shape[k] for k in ("M", "N", "K")},
                "tflops": flops / (ms * 1e-3) / 1e12,
            }
        )
        if verbose:
            print(f"  {shape['name']}: {ms:.4f} ms")

    build_dir = Path(_HERE) / "build"
    build_dir.mkdir(exist_ok=True)
    with open(build_dir / "performance_report.json", "w") as f:
        json.dump(report, f, indent=2)
    geomean = math.exp(sum(math.log(x) for x in latencies) / len(latencies))
    print(f"Geometric mean latency: {geomean:.4f} ms")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gemm_a16w16 harness")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    args = parser.parse_args()

    print("=" * 62)
    print("triton2flydsl 16-bit (bf16) GEMM")
    print("=" * 62)

    if args.compile:
        try:
            run_compile()
            sys.exit(0)
        except Exception as e:  # noqa: BLE001
            print(f"Compilation: FAIL\nError: {e}")
            sys.exit(1)
    elif args.correctness:
        sys.exit(0 if run_correctness() else 1)
    else:
        run_benchmark()
        sys.exit(0)
