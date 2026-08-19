#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Test harness for triton2flydsl/aiter/dynamic_quant_fp8 (FLAT layout).

The kernels under test are AITER's fp8/int8 quantization Triton kernels:
  - `static_per_tensor_quant_fp8_i8`  (caller scale)
  - `dynamic_per_tensor_quant_fp8_i8` (amax/DTYPE_MAX over the whole tensor, atomic)
  - `dynamic_per_token_quant_fp8_i8`  (per-row amax/DTYPE_MAX)
All write fp8 (e4m3) or int8 outputs + an fp32 scale. The standalone source copies
the device kernels verbatim (triton-only) with thin torch host wrappers.

fp8 dtype is arch-specific: gfx942 = e4m3fnuz (max ~240), gfx950 = e4m3fn (max 448).
The harness selects the arch-matched fp8 e4m3 dtype (mirroring
aiter.ops.triton.utils.types.get_fp8_e4m3_dtype) so the torch reference is
apples-to-apples with the kernel at the upstream gate.

Modes:
  --compile         ast-parse + import the standalone source, assert entry/kernel symbols
  --correctness     run each quant kernel on TEST_SHAPES x {int8, fp8}, assert finite
                    output AND match the torch reference at the upstream tolerance
                    (static atol=1e-2; dynamic atol=1e-1) [mirrors quant/test_quant.py]
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

SOURCE_FILE = "dynamic_quant_fp8.py"
ENTRIES = (
    "static_per_tensor_quant_fp8_i8",
    "dynamic_per_tensor_quant_fp8_i8",
    "dynamic_per_token_quant_fp8_i8",
)
KERNELS = (
    "_static_per_tensor_quant_fp8_i8_kernel",
    "_dynamic_per_tensor_quant_fp8_i8_kernel",
    "_dynamic_per_token_quant_fp8_i8_kernel",
)

# (M, N) from op_tests/triton_tests/quant/test_quant.py parametrizations (union of
# the per-tensor / per-token shape lists, incl. non-power-of-2 rows/cols).
TEST_SHAPES = [
    {"name": "m1_n32", "M": 1, "N": 32},
    {"name": "m32_n32", "M": 32, "N": 32},
    {"name": "m2_n16", "M": 2, "N": 16},
    {"name": "m10_n128", "M": 10, "N": 128},
    {"name": "m193_n75", "M": 193, "N": 75},
    {"name": "m1024_n128", "M": 1024, "N": 128},
    {"name": "m32_n8192", "M": 32, "N": 8192},
    {"name": "m400_n400", "M": 400, "N": 400},
]

SEED = 20  # match upstream test (torch.manual_seed(20))
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
    spec = importlib.util.spec_from_file_location("dynamic_quant_fp8_src", entry)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _dtype_max(qdtype):
    import torch

    return (
        torch.iinfo(qdtype).max
        if qdtype == torch.int8
        else torch.finfo(qdtype).max
    )


def run_compile():
    with open(os.path.join(_HERE, SOURCE_FILE)) as f:
        ast.parse(f.read())
    mod = _load_source()
    for e in ENTRIES:
        assert hasattr(mod, e), f"Missing entry {e}"
    for k in KERNELS:
        assert hasattr(mod, k), f"Missing kernel {k}"
    print("Compilation: PASS")
    return True


def _check(mode, mod, M, N, qdtype, verbose):
    import torch

    torch.manual_seed(SEED)
    if mode == "static":
        x = torch.randn((M, N), dtype=torch.bfloat16, device="cuda")
        scale = torch.randn(1, dtype=torch.float32, device="cuda")
        ref = (x / scale).to(qdtype)
        qx = torch.empty_like(x, dtype=qdtype)
        out = mod.static_per_tensor_quant_fp8_i8(qx, x, scale)
        torch.cuda.synchronize()
        close = torch.allclose(
            out.to(torch.float32), ref.to(torch.float32), atol=1e-2, rtol=1e-2
        )
        finite = bool(torch.isfinite(out.to(torch.float32)).all().item())
        return finite and close, finite, close
    elif mode == "dyn_tensor":
        x = torch.randn((M, N), dtype=torch.bfloat16, device="cuda")
        x_f32 = x.to(torch.float32)
        x_max = torch.max(torch.abs(x_f32))
        scale_ref = x_max / _dtype_max(qdtype)
        ref = (x_f32 / scale_ref).to(qdtype)
        qx = torch.empty_like(x, dtype=qdtype)
        scale_out = torch.zeros(1, dtype=torch.float32, device="cuda")
        out, s = mod.dynamic_per_tensor_quant_fp8_i8(qx, x, scale_out)
        torch.cuda.synchronize()
        s_close = torch.allclose(
            s, torch.tensor([scale_ref], device="cuda"), atol=1e-1, rtol=1e-1
        )
        v_close = torch.allclose(
            out.to(torch.float32), ref.to(torch.float32), atol=1e-1, rtol=1e-1
        )
        finite = bool(torch.isfinite(out.to(torch.float32)).all().item())
        return finite and s_close and v_close, finite, (s_close and v_close)
    else:  # dyn_token
        x = torch.rand((M, N), dtype=torch.bfloat16, device="cuda")
        x_max, _ = torch.max(torch.abs(x), axis=-1)
        scale_ref = x_max.to(torch.float32) / _dtype_max(qdtype)
        ref = (x * (1 / scale_ref[:, None])).to(qdtype)
        qx = torch.empty_like(x, dtype=qdtype)
        scale_out = torch.zeros(M, dtype=torch.float32, device="cuda")
        out, s = mod.dynamic_per_token_quant_fp8_i8(qx, x, scale_out)
        torch.cuda.synchronize()
        s_close = torch.allclose(s, scale_ref, atol=1e-1, rtol=1e-1)
        v_close = torch.allclose(
            out.to(torch.float32), ref.to(torch.float32), atol=1e-1, rtol=1e-1
        )
        finite = bool(torch.isfinite(out.to(torch.float32)).all().item())
        return finite and s_close and v_close, finite, (s_close and v_close)


def run_correctness(verbose=True):
    import torch

    mod = _load_source()
    fp8 = _fp8_e4m3_dtype()
    if verbose:
        print(f"  fp8 dtype = {fp8}")
    qdtypes = [("int8", torch.int8), ("fp8", fp8)]
    modes = ["static", "dyn_tensor", "dyn_token"]
    failures = []
    for shape in TEST_SHAPES:
        for mode in modes:
            for qname, qdtype in qdtypes:
                tag = f"{mode}_{qname}_{shape['name']}"
                try:
                    ok, finite, close = _check(
                        mode, mod, shape["M"], shape["N"], qdtype, verbose
                    )
                    if verbose:
                        print(
                            f"  {'PASS' if ok else 'FAIL'}: {tag} "
                            f"(M={shape['M']},N={shape['N']}) finite={finite} close={close}"
                        )
                    if not ok:
                        failures.append(tag)
                except Exception as e:  # noqa: BLE001
                    failures.append(tag)
                    if verbose:
                        print(f"  FAIL: {tag} - {str(e)[:160]}")

    total = len(TEST_SHAPES) * len(modes) * len(qdtypes)
    status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{total})"
    print(f"Status: {status}")
    print(f"correctness: {'pass' if not failures else 'fail'}")
    return not failures


def run_benchmark(verbose=True):
    import torch

    mod = _load_source()
    fp8 = _fp8_e4m3_dtype()
    report, latencies = [], []
    for idx, shape in enumerate(TEST_SHAPES):
        M, N = shape["M"], shape["N"]
        x = torch.randn((M, N), dtype=torch.bfloat16, device="cuda")
        qx = torch.empty_like(x, dtype=fp8)
        scale_out = torch.zeros(M, dtype=torch.float32, device="cuda")
        fn = lambda: mod.dynamic_per_token_quant_fp8_i8(qx, x, scale_out)  # noqa: E731
        fn()
        torch.cuda.synchronize()
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        ms, bench_meta = benchmark_cuda_graph_or_events(
            fn, warmup=0, repetition=ITERS
        )
        latencies.append(ms)
        gb = (M * N) * (2 + 1) / 1e9  # bf16 in + fp8 out approx
        report.append(
            {
                "test_case_id": f"perf{idx + 1}",
                "execution_time_ms": ms,
                **bench_meta,
                "params": {k: shape[k] for k in ("M", "N")},
                "gbps": gb / (ms * 1e-3),
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
    parser = argparse.ArgumentParser(description="dynamic_quant_fp8 harness")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    args = parser.parse_args()

    print("=" * 62)
    print("triton2flydsl fp8/int8 static + dynamic per-tensor/per-token quant")
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
