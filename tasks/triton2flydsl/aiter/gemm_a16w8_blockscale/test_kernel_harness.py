#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Test harness for triton2flydsl/aiter/gemm_a16w8_blockscale (FLAT layout).

The kernel under test is AITER's a16w8 block-scaled GEMM Triton kernel
(`gemm_a16w8_blockscale` -> `_gemm_a16w8_blockscale_kernel`, non-prequant path):
Y = X @ W^T with 16-bit activations X [M, K] and 8-bit weights W [N, K], a 128x128
block W_scale applied inside the K loop (W upcast to the activation dtype, scaled,
fp32-accumulated), bf16 output. The standalone source keeps the non-split-K
(NUM_KSPLIT == 1), non-prequant triton path with a static tile config (BLOCK_SIZE_K
== GROUP_K == 128).

fp8 weight dtype is arch-specific: gfx942 = e4m3fnuz, gfx950 = e4m3fn. The harness
builds weights + reference with the arch-matched fp8 e4m3 dtype.

Modes:
  --compile         ast-parse + import the standalone source, assert entry/kernel symbols
  --correctness     run the Triton kernel on TEST_SHAPES, assert finite output AND
                    match the fp32 dequant torch reference at the upstream tolerance
                    (atol=0.1, rtol=0.1) [mirrors test_gemm_a16w8_blockscale.py:run_torch]
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

SOURCE_FILE = "gemm_a16w8_blockscale.py"
ENTRY = "gemm_a16w8_blockscale"
KERNEL = "_gemm_a16w8_blockscale_kernel"
BLOCK_N, BLOCK_K = 128, 128

# (M, N, K) with N % 128 == 0 and K % 128 == 0 so the 128x128 block-scale layout is
# exact (GROUP_K == BLOCK_SIZE_K == 128). Real GEMM tiles from
# op_tests/.../test_gemm_a16w8_blockscale.py:get_x_vals, bounded subset.
TEST_SHAPES = [
    {"name": "m256_n512_k1024", "M": 256, "N": 512, "K": 1024},
    {"name": "m1024_n1024_k1024", "M": 1024, "N": 1024, "K": 1024},
    {"name": "m2048_n2048_k2048", "M": 2048, "N": 2048, "K": 2048},
    {"name": "m64_n256_k7168", "M": 64, "N": 256, "K": 7168},
    {"name": "m128_n2048_k4096", "M": 128, "N": 2048, "K": 4096},
]

SEED = 0  # match upstream generate_gemm_a16w8_blockscale_inputs (torch.manual_seed(0))
WARMUP, ITERS = 10, 100

_HERE = os.path.dirname(os.path.abspath(__file__))


def _fp8_e4m3_dtype():
    import torch

    name = torch.cuda.get_device_properties(0).gcnArchName
    arch = name.split(":")[0]
    if arch in ("gfx950", "gfx1250", "gfx1200", "gfx1201"):
        return torch.float8_e4m3fn
    return torch.float8_e4m3fnuz


def _load_source():
    entry = os.path.join(_HERE, SOURCE_FILE)
    spec = importlib.util.spec_from_file_location("gemm_a16w8_blockscale_src", entry)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_inputs(M, N, K, w_dtype, device="cuda"):
    # Mirrors generate_gemm_a16w8_blockscale_inputs (layout TN, non-shuffle).
    import torch

    torch.manual_seed(SEED)
    scale_n = (N + BLOCK_N - 1) // BLOCK_N
    scale_k = (K + BLOCK_K - 1) // BLOCK_K
    x = torch.randn((M, K), dtype=torch.bfloat16, device=device) / 10
    weight = (torch.rand((N, K), dtype=torch.float16, device=device) / 10).to(w_dtype)
    w_scale = torch.rand([scale_n, scale_k], dtype=torch.float32, device=device)
    return x, weight, w_scale


def _torch_ref(x, weight, w_scale, out_dtype):
    # fp32 dequant reference (matches test_gemm_a16w8_blockscale.py:run_torch).
    import torch
    import torch.nn.functional as F

    m, k = x.shape
    n = weight.shape[0]
    w_s = w_scale.repeat_interleave(BLOCK_N, dim=0).repeat_interleave(BLOCK_K, dim=1)
    weight_dq = weight.to(torch.float32) * w_s[:n, :k]
    out = F.linear(x.to(torch.float32), weight_dq.to(torch.float32))
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
    w_dtype = _fp8_e4m3_dtype()
    out_dtype = torch.bfloat16
    if verbose:
        print(f"  fp8 weight dtype = {w_dtype}")
    failures = []
    for shape in TEST_SHAPES:
        tag = shape["name"]
        try:
            x, w, w_scale = _make_inputs(shape["M"], shape["N"], shape["K"], w_dtype)
            y = mod.gemm_a16w8_blockscale(x, w, w_scale, out_dtype)
            torch.cuda.synchronize()
            ref = _torch_ref(x, w, w_scale, out_dtype)
            finite = bool(torch.isfinite(y).all().item())
            close = torch.allclose(y, ref, atol=0.1, rtol=0.1)
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

    total = len(TEST_SHAPES)
    status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{total})"
    print(f"Status: {status}")
    print(f"correctness: {'pass' if not failures else 'fail'}")
    return not failures


def run_benchmark(verbose=True):
    import torch

    mod = _load_source()
    w_dtype = _fp8_e4m3_dtype()
    out_dtype = torch.bfloat16
    report, latencies = [], []
    for idx, shape in enumerate(TEST_SHAPES):
        x, w, w_scale = _make_inputs(shape["M"], shape["N"], shape["K"], w_dtype)
        fn = lambda: mod.gemm_a16w8_blockscale(x, w, w_scale, out_dtype)  # noqa: E731
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
    parser = argparse.ArgumentParser(description="gemm_a16w8_blockscale harness")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    args = parser.parse_args()

    print("=" * 62)
    print("triton2flydsl a16w8 block-scaled GEMM")
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
