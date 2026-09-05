#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Test harness for triton2flydsl/aiter/batched_gemm_bf16 (FLAT layout).

The kernel under test is AITER's batched 16-bit GEMM Triton kernel
(`batched_gemm_bf16` -> `_batched_gemm_bf16_kernel`): Y[i] = X[i] @ W[i]^T (+ bias[i])
for every i in the batch, fp32 accumulation, bf16/fp16 output, 2D (batch, grouped MN)
grid. The standalone source replaces the on-disk tuned-config lookup with a static
tile config.

Modes:
  --compile         ast-parse + import the standalone source, assert entry/kernel symbols
  --correctness     run the Triton kernel on TEST_SHAPES, assert finite output AND
                    match the fp32 torch reference at the upstream tolerance
                    (atol=0.01, rtol=1e-2) [mirrors gemm/batched/test_batched_gemm_bf16.py:run_torch]
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

SOURCE_FILE = "batched_gemm_bf16.py"
ENTRY = "batched_gemm_bf16"
KERNEL = "_batched_gemm_bf16_kernel"

# (B, M, N, K): B=16 mirrors op_tests/triton_tests/gemm/batched/test_batched_gemm_bf16.py
# (b in [16]); shapes are real GEMM tiles from get_x_vals / minimal_x_vals kept to a
# bounded subset (small / decode / projection) plus minimal & irregular edge cases.
TEST_SHAPES = [
    {"name": "b16_m1_n1_k1", "B": 16, "M": 1, "N": 1, "K": 1},
    {"name": "b16_m3_n5_k2", "B": 16, "M": 3, "N": 5, "K": 2},
    {"name": "b16_m16_n16_k16", "B": 16, "M": 16, "N": 16, "K": 16},
    {"name": "b16_m128_n256_k512", "B": 16, "M": 128, "N": 256, "K": 512},
    {"name": "b16_m256_n512_k1024", "B": 16, "M": 256, "N": 512, "K": 1024},
    {"name": "b16_m1_n1280_k1024", "B": 16, "M": 1, "N": 1280, "K": 1024},
    {"name": "b16_m512_n1024_k1024", "B": 16, "M": 512, "N": 1024, "K": 1024},
]

SEED = 0  # match upstream generate_batched_gemm_a16w16_inputs (torch.manual_seed(0))
WARMUP, ITERS = 10, 100

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_source():
    entry = os.path.join(_HERE, SOURCE_FILE)
    spec = importlib.util.spec_from_file_location("batched_gemm_bf16_src", entry)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_inputs(B, M, N, K, dtype, with_bias, device="cuda"):
    # Mirrors generate_batched_gemm_a16w16_inputs (layout TN).
    import torch

    torch.manual_seed(SEED)
    x = torch.randint(-20, 20, (B, M, K), dtype=dtype, device=device)
    weight = torch.randint(-20, 20, (B, N, K), dtype=dtype, device=device)
    bias = torch.rand([B, 1, N], dtype=dtype, device=device) * 10 if with_bias else None
    return x, weight, bias


def _torch_ref(x, weight, bias, out_dtype):
    # fp32 reference (matches test_batched_gemm_bf16.py:run_torch).
    import torch
    import torch.nn.functional as F

    B, M, _ = x.shape
    N = weight.shape[1]
    out = torch.empty(B, M, N, dtype=torch.float32, device=x.device)
    for b in range(B):
        b_out = F.linear(x[b].to(torch.float32), weight[b].to(torch.float32))
        if bias is not None:
            b_out = b_out + bias[b].to(torch.float32)
        out[b] = b_out
    return out.to(out_dtype)


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

    mod = _load_source()
    dtype = torch.bfloat16
    failures = []
    for shape in TEST_SHAPES:
        for with_bias in (False, True):
            tag = f"{shape['name']}_{'bias' if with_bias else 'nobias'}"
            try:
                x, w, bias = _make_inputs(
                    shape["B"], shape["M"], shape["N"], shape["K"], dtype, with_bias
                )
                y = mod.batched_gemm_bf16(x, w, bias, dtype)
                torch.cuda.synchronize()
                ref = _torch_ref(x, w, bias, dtype)
                finite = bool(torch.isfinite(y).all().item())
                close = torch.allclose(y, ref, atol=0.01, rtol=1e-2)
                ok = finite and close
                if verbose:
                    print(
                        f"  {'PASS' if ok else 'FAIL'}: {tag} "
                        f"(B={shape['B']},M={shape['M']},N={shape['N']},K={shape['K']}) "
                        f"out={tuple(y.shape)} finite={finite} close={close}"
                    )
                if not ok:
                    failures.append(tag)
            except Exception as e:  # noqa: BLE001
                failures.append(tag)
                if verbose:
                    print(f"  FAIL: {tag} - {str(e)[:160]}")

    total = len(TEST_SHAPES) * 2
    status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{total})"
    print(f"Status: {status}")
    print(f"correctness: {'pass' if not failures else 'fail'}")
    return not failures


def run_benchmark(verbose=True):
    import torch

    mod = _load_source()
    dtype = torch.bfloat16
    report, latencies = [], []
    for idx, shape in enumerate(TEST_SHAPES):
        x, w, bias = _make_inputs(
            shape["B"], shape["M"], shape["N"], shape["K"], dtype, False
        )
        fn = lambda: mod.batched_gemm_bf16(x, w, None, dtype)  # noqa: E731
        fn()
        torch.cuda.synchronize()
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        ms, bench_meta = benchmark_cuda_graph_or_events(
            fn, warmup=0, repetition=ITERS
        )
        latencies.append(ms)
        flops = 2.0 * shape["B"] * shape["M"] * shape["N"] * shape["K"]
        report.append(
            {
                "test_case_id": f"perf{idx + 1}",
                "execution_time_ms": ms,
                **bench_meta,
                "params": {k: shape[k] for k in ("B", "M", "N", "K")},
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
    parser = argparse.ArgumentParser(description="batched_gemm_bf16 harness")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    args = parser.parse_args()

    print("=" * 62)
    print("triton2flydsl batched 16-bit (bf16) GEMM")
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
