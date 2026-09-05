#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Test harness for triton2flydsl/aiter/batched_gemm_a8w8 (FLAT layout).

The kernel under test is AITER's batched 8-bit scaled GEMM Triton kernel
(`batched_gemm_a8w8` -> `_batched_gemm_a8w8_kernel`):
Y[i] = (X[i] @ W[i]^T) * (x_scale[i] * w_scale[i]) (+ bias[i]) for every i in the
batch, with int32/fp32 accumulation, per-row x_scale [B,M,1] and per-column
w_scale [B,1,N], a 2D (batch, grouped MN) grid, and bf16 output. The standalone
source replaces the on-disk tuned-config lookup with a static tile config.

Upstream's op_test (gemm/batched/test_batched_gemm_a8w8.py) exercises int8 inputs
with fp32 per-row/per-column scales; the harness mirrors that exactly.

Modes:
  --compile         ast-parse + import the standalone source, assert entry/kernel symbols
  --correctness     run the Triton kernel on TEST_SHAPES, assert finite output AND
                    match the fp32 dequant torch reference at the upstream tolerance
                    (atol=0.01, rtol=1e-2) [mirrors test_batched_gemm_a8w8.py:run_torch]
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

SOURCE_FILE = "batched_gemm_a8w8.py"
ENTRY = "batched_gemm_a8w8"
KERNEL = "_batched_gemm_a8w8_kernel"

# (B, M, N, K): B=16 mirrors op_tests/triton_tests/gemm/batched/test_batched_gemm_a8w8.py
# (b in [16]); shapes are real GEMM tiles kept to a bounded subset plus edge cases.
TEST_SHAPES = [
    {"name": "b16_m1_n1_k1", "B": 16, "M": 1, "N": 1, "K": 1},
    {"name": "b16_m3_n5_k2", "B": 16, "M": 3, "N": 5, "K": 2},
    {"name": "b16_m16_n16_k16", "B": 16, "M": 16, "N": 16, "K": 16},
    {"name": "b16_m128_n256_k512", "B": 16, "M": 128, "N": 256, "K": 512},
    {"name": "b16_m256_n512_k1024", "B": 16, "M": 256, "N": 512, "K": 1024},
    {"name": "b16_m1_n1280_k1024", "B": 16, "M": 1, "N": 1280, "K": 1024},
    {"name": "b16_m512_n1024_k1024", "B": 16, "M": 512, "N": 1024, "K": 1024},
]

SEED = 0  # match upstream generate_batched_gemm_a8w8_inputs (torch.manual_seed(0))
WARMUP, ITERS = 10, 100

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_source():
    entry = os.path.join(_HERE, SOURCE_FILE)
    spec = importlib.util.spec_from_file_location("batched_gemm_a8w8_src", entry)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_inputs(B, M, N, K, out_dtype, with_bias, device="cuda"):
    # Mirrors generate_batched_gemm_a8w8_inputs (layout TN): int8 inputs, fp32
    # per-row x_scale and per-column w_scale.
    import torch

    torch.manual_seed(SEED)
    x = torch.randint(-20, 20, (B, M, K), dtype=torch.int8, device=device)
    weight = torch.randint(-20, 20, (B, N, K), dtype=torch.int8, device=device)
    x_scale = torch.rand([B, M, 1], dtype=torch.float32, device=device) + 1e-6
    w_scale = torch.rand([B, 1, N], dtype=torch.float32, device=device) + 1e-6
    bias = torch.rand([B, 1, N], dtype=out_dtype, device=device) * 10 if with_bias else None
    return x, weight, x_scale, w_scale, bias


def _torch_ref(x, weight, x_scale, w_scale, bias, out_dtype):
    # fp32 dequant reference (matches test_batched_gemm_a8w8.py:run_torch).
    import torch
    import torch.nn.functional as F

    B, M, _ = x.shape
    N = weight.shape[1]
    out = torch.empty(B, M, N, dtype=out_dtype, device=x.device)
    for b in range(B):
        b_x = F.linear(x[b].to(torch.float32), weight[b].to(torch.float32))
        b_scale = torch.matmul(x_scale[b], w_scale[b])
        b_out = torch.mul(b_x, b_scale)
        if bias is not None:
            # Mirror upstream run_torch: cast to bias dtype before the add, so the
            # bf16 rounding of the bias add matches the kernel epilogue.
            b_out = b_out.to(bias[b]) + bias[b]
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
    out_dtype = torch.bfloat16
    failures = []
    for shape in TEST_SHAPES:
        for with_bias in (False, True):
            tag = f"{shape['name']}_{'bias' if with_bias else 'nobias'}"
            try:
                x, w, x_scale, w_scale, bias = _make_inputs(
                    shape["B"], shape["M"], shape["N"], shape["K"], out_dtype, with_bias
                )
                y = mod.batched_gemm_a8w8(x, w, x_scale, w_scale, bias, out_dtype)
                torch.cuda.synchronize()
                ref = _torch_ref(x, w, x_scale, w_scale, bias, out_dtype)
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
    out_dtype = torch.bfloat16
    report, latencies = [], []
    for idx, shape in enumerate(TEST_SHAPES):
        x, w, x_scale, w_scale, bias = _make_inputs(
            shape["B"], shape["M"], shape["N"], shape["K"], out_dtype, False
        )
        fn = lambda: mod.batched_gemm_a8w8(x, w, x_scale, w_scale, None, out_dtype)  # noqa: E731
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
    parser = argparse.ArgumentParser(description="batched_gemm_a8w8 harness")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    args = parser.parse_args()

    print("=" * 62)
    print("triton2flydsl batched 8-bit (int8/fp8) scaled GEMM")
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
