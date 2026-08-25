#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Test harness for triton2flydsl/aiter/gemm_a8w8 (FLAT layout).

The kernel under test is AITER's 8-bit scaled GEMM Triton kernel (`gemm_a8w8` ->
`_gemm_a8w8_kernel`): Y = (X @ W^T) * (x_scale * w_scale) (+ bias) with int32/fp32
accumulation, per-row x_scale [M,1] and per-column w_scale [1,N], an XCD-balanced
+ grouped pid remap, and bf16 output. The standalone source keeps the non-split-K
(NUM_KSPLIT == 1) triton path with a static tile config.

fp8 dtype is arch-specific: gfx942 = e4m3fnuz (max ~240), gfx950 = e4m3fn
(max 448). The harness quantizes inputs (and builds its reference) with the
arch-matched fp8 e4m3 dtype, mirroring aiter.ops.triton.utils.types.get_fp8_dtypes
so the torch reference is apples-to-apples with the kernel at the tight gate.

Modes:
  --compile         ast-parse + import the standalone source, assert entry/kernel symbols
  --correctness     run the Triton kernel on TEST_SHAPES, assert finite output AND
                    match the fp32 dequant torch reference at the upstream fp8
                    tolerance (atol=0.02, rtol=1e-2)
                    [mirrors gemm/basic/test_gemm_a8w8.py:run_torch]
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

SOURCE_FILE = "gemm_a8w8.py"
ENTRY = "gemm_a8w8"
KERNEL = "_gemm_a8w8_kernel"

# (M, N, K) from op_tests/triton_tests/gemm/basic/test_gemm_a8w8.py::get_x_vals /
# get_fewer_x_vals (real fp8 GEMMs: Llama-3 70B QKV input projection, square
# decode/prefill tiles) plus minimal/irregular edge cases.
TEST_SHAPES = [
    {"name": "m1_n1_k1", "M": 1, "N": 1, "K": 1},
    {"name": "m3_n5_k2", "M": 3, "N": 5, "K": 2},
    {"name": "m16_n1024_k1024", "M": 16, "N": 1024, "K": 1024},
    {"name": "m128_n8192_k512", "M": 128, "N": 8192, "K": 512},
    {"name": "m256_n512_k8192", "M": 256, "N": 512, "K": 8192},
    {"name": "m1024_n1024_k1024", "M": 1024, "N": 1024, "K": 1024},
    {"name": "m2048_n2048_k2048", "M": 2048, "N": 2048, "K": 2048},
    {"name": "m4096_n4096_k4096", "M": 4096, "N": 4096, "K": 4096},
    {"name": "ll3_70b_qkv_m256_n10240_k8192", "M": 256, "N": 10240, "K": 8192},
]

SEED = 0  # match upstream generate_gemm_a8w8_inputs (torch.manual_seed(0))
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
    spec = importlib.util.spec_from_file_location("gemm_a8w8_src", entry)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_inputs(M, N, K, in_dtype, with_bias, device="cuda"):
    # Mirrors generate_gemm_a8w8_inputs (layout TN): per-row x_scale, per-row
    # w_scale (transposed to per-column), quantize to in_dtype.
    import torch

    torch.manual_seed(SEED)
    dtype_max = torch.finfo(in_dtype).max

    x = torch.randn((M, K), dtype=torch.float32, device=device)
    weight = torch.randn((N, K), dtype=torch.float32, device=device)

    max_x = x.abs().float().amax(dim=1, keepdim=True)
    x_scale = max_x / dtype_max
    x = (x / x_scale).to(in_dtype)

    max_weight = weight.abs().float().amax(dim=1, keepdim=True).T.contiguous()
    w_scale = max_weight / dtype_max
    weight = (weight / w_scale.T).to(in_dtype)

    bias = (
        torch.rand([1, N], dtype=torch.float32, device=device) * 10
        if with_bias
        else None
    )
    return x, weight, x_scale, w_scale, bias


def _torch_ref(x, weight, x_scale, w_scale, bias, out_dtype):
    # fp32 dequant reference (matches test_gemm_a8w8.py:run_torch).
    import torch
    import torch.nn.functional as F

    acc = F.linear(x.to(torch.float32), weight.to(torch.float32))
    scale = torch.matmul(x_scale, w_scale)
    out = torch.mul(acc, scale)
    if bias is not None:
        out = out.to(bias) + bias
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
        for with_bias in (False, True):
            tag = f"{shape['name']}_{'bias' if with_bias else 'nobias'}"
            try:
                x, w, x_scale, w_scale, bias = _make_inputs(
                    shape["M"], shape["N"], shape["K"], in_dtype, with_bias
                )
                y = mod.gemm_a8w8(x, w, x_scale, w_scale, bias, out_dtype)
                torch.cuda.synchronize()
                ref = _torch_ref(x, w, x_scale, w_scale, bias, out_dtype)
                finite = bool(torch.isfinite(y).all().item())
                close = torch.allclose(y, ref, atol=0.02, rtol=1e-2)
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

    total = len(TEST_SHAPES) * 2
    status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{total})"
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
        x, w, x_scale, w_scale, bias = _make_inputs(
            shape["M"], shape["N"], shape["K"], in_dtype, False
        )
        fn = lambda: mod.gemm_a8w8(x, w, x_scale, w_scale, None, out_dtype)  # noqa: E731
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
    parser = argparse.ArgumentParser(description="gemm_a8w8 harness")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    args = parser.parse_args()

    print("=" * 62)
    print("triton2flydsl 8-bit (fp8) scaled GEMM")
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
