#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Test harness for triton2flydsl/aiter/dynamic_mxfp8_quant (FLAT layout).

The kernel under test is AITER's per-1x32 MXFP8 quantization Triton kernel
(`dynamic_mxfp8_quant` -> `_dynamic_mxfp8_quant_kernel` + `_mxfp8_quant_op`):
derive a uint8 e8m0 block scale (1 per 32 K elements) and FP8 e4m3fn values. The
standalone source copies the device kernels verbatim (triton-only) with a thin
torch host wrapper.

Modes:
  --compile         ast-parse + import the standalone source, assert entry/kernel symbols
  --correctness     run the kernel on TEST_SHAPES, assert finite output AND match the
                    bit-faithful fp32 torch reference: e8m0 scales BIT-EXACT and FP8
                    values within 1 ULP on the uint8 view
                    [mirrors quant/test_quant_mxfp8.py:torch_mxfp8_quant_from_fp32]
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

SOURCE_FILE = "dynamic_mxfp8_quant.py"
ENTRY = "dynamic_mxfp8_quant"
KERNEL = "_dynamic_mxfp8_quant_kernel"
QUANT_BLOCK_SIZE = 32
_E8M0_MASK_INT32 = -8388608  # 0xFF800000 as signed int32

# (M, K), K % 32 == 0; from op_tests/triton_tests/quant/test_quant_mxfp8.py
# (incl. non-power-of-2 M). Last entry is a 3D shape to exercise dim folding.
TEST_SHAPES = [
    {"name": "m1_k32", "shape": (1, 32)},
    {"name": "m1_k128", "shape": (1, 128)},
    {"name": "m8_k64", "shape": (8, 64)},
    {"name": "m16_k128", "shape": (16, 128)},
    {"name": "m32_k256", "shape": (32, 256)},
    {"name": "m64_k512", "shape": (64, 512)},
    {"name": "m128_k1024", "shape": (128, 1024)},
    {"name": "m137_k64", "shape": (137, 64)},
    {"name": "m256_k32", "shape": (256, 32)},
    {"name": "b4_m8_k128", "shape": (4, 8, 128)},
]

SEED = 20  # match upstream test
WARMUP, ITERS = 10, 100

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_source():
    entry = os.path.join(_HERE, SOURCE_FILE)
    spec = importlib.util.spec_from_file_location("dynamic_mxfp8_quant_src", entry)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _torch_mxfp8_quant_from_fp32(x_fp32):
    # Bit-faithful port of _dynamic_mxfp8_quant_kernel (matches
    # test_quant_mxfp8.py:torch_mxfp8_quant_from_fp32).
    import torch

    M, K = x_fp32.shape
    Ng = K // QUANT_BLOCK_SIZE
    x_2d = x_fp32.reshape(M, Ng, QUANT_BLOCK_SIZE).to(torch.float32)
    amax = torch.amax(torch.abs(x_2d), dim=-1, keepdim=True)
    amax_i32 = amax.contiguous().view(torch.int32)
    amax_i32 = (amax_i32 + 0x200000) & _E8M0_MASK_INT32
    amax_p2 = amax_i32.view(torch.float32)
    scale_unbiased = torch.log2(amax_p2).floor() - 8
    scale_unbiased = torch.clamp(scale_unbiased, min=-127, max=127)
    scale_e8m0 = (scale_unbiased.to(torch.int32) + 127).to(torch.uint8)
    quant_scale = torch.exp2(-scale_unbiased)
    qx_2d = x_2d * quant_scale
    qx = qx_2d.reshape(M, K)
    y_fp8 = qx.to(torch.float8_e4m3fn)
    s = scale_e8m0.reshape(M, Ng)
    return y_fp8, s


def run_compile():
    with open(os.path.join(_HERE, SOURCE_FILE)) as f:
        ast.parse(f.read())
    mod = _load_source()
    assert hasattr(mod, ENTRY), f"Missing entry {ENTRY}"
    assert hasattr(mod, KERNEL), f"Missing kernel {KERNEL}"
    assert hasattr(mod, "_mxfp8_quant_op"), "Missing _mxfp8_quant_op"
    print("Compilation: PASS")
    return True


def run_correctness(verbose=True):
    import torch

    mod = _load_source()
    failures = []
    for shape in TEST_SHAPES:
        tag = shape["name"]
        sh = shape["shape"]
        try:
            torch.manual_seed(SEED)
            x = torch.randn(sh, dtype=torch.bfloat16, device="cuda") * 4.0
            K = sh[-1]
            x_flat = x.reshape(-1, K).to(torch.float32)
            y_ref, s_ref = _torch_mxfp8_quant_from_fp32(x_flat)
            y_kern, s_kern = mod.dynamic_mxfp8_quant(x)
            torch.cuda.synchronize()
            y_kern_flat = y_kern.reshape(-1, K)
            s_kern_flat = s_kern.reshape(-1, K // QUANT_BLOCK_SIZE)
            finite = bool(
                torch.isfinite(y_kern_flat.to(torch.float32)).all().item()
            )
            scales_exact = torch.equal(s_kern_flat, s_ref)
            vdiff = (
                y_kern_flat.view(torch.uint8).to(torch.int32)
                - y_ref.view(torch.uint8).to(torch.int32)
            ).abs().max().item()
            v_close = vdiff <= 1
            ok = finite and scales_exact and v_close
            if verbose:
                print(
                    f"  {'PASS' if ok else 'FAIL'}: {tag} shape={tuple(sh)} "
                    f"finite={finite} scales_exact={scales_exact} val_ulp<=1={v_close} "
                    f"(maxdiff={vdiff})"
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
    report, latencies = [], []
    for idx, shape in enumerate(TEST_SHAPES):
        sh = shape["shape"]
        torch.manual_seed(SEED)
        x = torch.randn(sh, dtype=torch.bfloat16, device="cuda")
        fn = lambda: mod.dynamic_mxfp8_quant(x)  # noqa: E731
        fn()
        torch.cuda.synchronize()
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        ms, bench_meta = benchmark_cuda_graph_or_events(
            fn, warmup=0, repetition=ITERS
        )
        latencies.append(ms)
        report.append(
            {
                "test_case_id": f"perf{idx + 1}",
                "execution_time_ms": ms,
                **bench_meta,
                "params": {"shape": list(sh)},
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
    parser = argparse.ArgumentParser(description="dynamic_mxfp8_quant harness")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    args = parser.parse_args()

    print("=" * 62)
    print("triton2flydsl per-1x32 MXFP8 quant")
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
