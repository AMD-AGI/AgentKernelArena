#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Test harness for triton2flydsl/aiter/gemm_a8w8_blockscale (FLAT layout).

The kernel under test is AITER's fp8 block-scaled GEMM Triton kernel
(`gemm_a8w8_blockscale` -> `_gemm_a8w8_blockscale_kernel`): Y = X @ W^T with
128x128 block-wise dequant (A_scale [M, ceil(K/128)], W_scale [ceil(N/128),
ceil(K/128)]) applied inside the K loop, fp32 accumulation, bf16 output. This is
the DeepSeek-V3 / Qwen3 fp8 dense matmul. The standalone source keeps the
non-split-K (NUM_KSPLIT == 1) triton path with a static tile config
(BLOCK_SIZE_K == GROUP_K == 128).

fp8 dtype is arch-specific: gfx942 = e4m3fnuz (max ~240), gfx950 = e4m3fn
(max 448). The harness quantizes inputs (and builds its reference) with the
arch-matched fp8 e4m3 dtype, mirroring aiter.ops.triton.utils.types.get_fp8_dtypes
so the torch reference is apples-to-apples with the kernel at the tight gate.

Modes:
  --compile         ast-parse + import the standalone source, assert entry/kernel symbols
  --correctness     run the Triton kernel on TEST_SHAPES, assert finite output AND
                    match the block-dequant torch reference at the upstream
                    tolerance (atol=0.01, rtol=1e-2)
                    [mirrors gemm/basic/test_gemm_a8w8_blockscale.py:run_torch]
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

SOURCE_FILE = "gemm_a8w8_blockscale.py"
ENTRY = "gemm_a8w8_blockscale"
KERNEL = "_gemm_a8w8_blockscale_kernel"

BLOCK_SHAPE = (128, 128)  # (block_shape_n, block_shape_k)

# (M, N, K) from op_tests/triton_tests/gemm/basic/test_gemm_a8w8_blockscale.py
# (DeepSeek-V3 / Qwen3 fp8 dense projections; K >= 512 = the triton path) plus
# square decode/prefill tiles. All N, K multiples of 128 (block scale).
TEST_SHAPES = [
    {"name": "m1024_n1024_k1024", "M": 1024, "N": 1024, "K": 1024},
    {"name": "m2048_n2048_k2048", "M": 2048, "N": 2048, "K": 2048},
    {"name": "ds_m128_n9216_k7168", "M": 128, "N": 9216, "K": 7168},
    {"name": "ds_m192_n7168_k4608", "M": 192, "N": 7168, "K": 4608},
    {"name": "m128_n8192_k512", "M": 128, "N": 8192, "K": 512},
    {"name": "m256_n7168_k4608", "M": 256, "N": 7168, "K": 4608},
    {"name": "m4096_n4096_k4096", "M": 4096, "N": 4096, "K": 4096},
]

SEED = 0  # match upstream generate_gemm_a8w8_blockscale_inputs (torch.manual_seed(0))
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
    spec = importlib.util.spec_from_file_location("gemm_a8w8_blockscale_src", entry)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_inputs(M, N, K, in_dtype, device="cuda"):
    # Mirrors generate_gemm_a8w8_blockscale_inputs (layout TN, no shuffle).
    import torch

    torch.manual_seed(SEED)
    block_n, block_k = BLOCK_SHAPE
    scale_n = (N + block_n - 1) // block_n
    scale_k = (K + block_k - 1) // block_k

    x = (torch.rand((M, K), dtype=torch.float16, device=device) / 10).to(in_dtype)
    weight = (torch.rand((N, K), dtype=torch.float16, device=device) / 10).to(in_dtype)
    x_scale = torch.rand([M, scale_k], dtype=torch.float32, device=device)
    w_scale = torch.rand([scale_n, scale_k], dtype=torch.float32, device=device)
    return x, weight, x_scale, w_scale


def _torch_ref(x, weight, x_scale, w_scale, out_dtype):
    # block-dequant reference (matches test_gemm_a8w8_blockscale.py:run_torch).
    import torch
    import torch.nn.functional as F

    block_n, block_k = BLOCK_SHAPE
    m, k = x.shape
    n = weight.shape[0]
    xs = x_scale.repeat_interleave(block_k, dim=1)
    xq = x.to(xs.dtype) * xs[:m, :k]
    xq = xq.view(m, k)
    ws = w_scale.repeat_interleave(block_n, dim=0)
    ws = ws.repeat_interleave(block_k, dim=1)
    ws = ws[:n, :k]
    wq = weight.to(ws.dtype) * ws
    out = F.linear(xq.to(torch.float32), wq.to(torch.float32))
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
    in_dtype = _fp8_e4m3_dtype()
    out_dtype = torch.bfloat16
    if verbose:
        print(f"  fp8 in_dtype = {in_dtype}")
    failures = []
    for shape in TEST_SHAPES:
        tag = shape["name"]
        try:
            x, w, x_scale, w_scale = _make_inputs(
                shape["M"], shape["N"], shape["K"], in_dtype
            )
            y = mod.gemm_a8w8_blockscale(x, w, x_scale, w_scale, out_dtype)
            torch.cuda.synchronize()
            ref = _torch_ref(x, w, x_scale, w_scale, out_dtype)
            finite = bool(torch.isfinite(y).all().item())
            close = torch.allclose(y, ref, atol=0.01, rtol=1e-2)
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
    in_dtype = _fp8_e4m3_dtype()
    out_dtype = torch.bfloat16
    report, latencies = [], []
    for idx, shape in enumerate(TEST_SHAPES):
        x, w, x_scale, w_scale = _make_inputs(
            shape["M"], shape["N"], shape["K"], in_dtype
        )
        fn = lambda: mod.gemm_a8w8_blockscale(  # noqa: E731
            x, w, x_scale, w_scale, out_dtype
        )
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
    parser = argparse.ArgumentParser(description="gemm_a8w8_blockscale harness")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    args = parser.parse_args()

    print("=" * 62)
    print("triton2flydsl fp8 block-scaled GEMM (128x128)")
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
