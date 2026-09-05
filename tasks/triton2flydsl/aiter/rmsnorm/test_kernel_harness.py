#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Test harness for triton2flydsl/aiter/rmsnorm (FLAT layout).

The kernel under test is AITER's RMSNorm forward Triton kernel (`rms_norm` ->
`_rms_norm_kernel` / `_rmsnorm_kernel_large_m_small_n`): fp32 sum-of-squares
reduction, rsqrt scale, weight multiply, written in the input dtype.

Modes:
  --compile         ast-parse + import the standalone source, assert entry/kernel symbols
  --correctness     run the Triton kernel on TEST_SHAPES, assert finite output AND
                    match the fp32-reduce torch reference at the upstream bf16/fp16
                    tolerance (1e-2)  [mirrors op_tests/.../test_rmsnorm.py:torch_rmsnorm]
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

SOURCE_FILE = "rmsnorm.py"
ENTRY = "rms_norm"
KERNEL = "_rms_norm_kernel"

# (M, N) subset of op_tests/triton_tests/normalization/test_rmsnorm.py::get_vals.
# Includes wide rows (USE_BLOCKED) and the tall/narrow regime (M>8192,N<=2048 ->
# _rmsnorm_kernel_large_m_small_n).
TEST_SHAPES = [
    {"name": "m1_n4", "M": 1, "N": 4},
    {"name": "m256_n4096", "M": 256, "N": 4096},
    {"name": "m4096_n8192", "M": 4096, "N": 8192},
    {"name": "m873_n1245", "M": 873, "N": 1245},
    {"name": "m8192_n8192", "M": 8192, "N": 8192},
    {"name": "m2048_n4096", "M": 2048, "N": 4096},
    {"name": "m768_n2048", "M": 768, "N": 2048},
    {"name": "m64_n512", "M": 64, "N": 512},
    {"name": "m16380_n1536", "M": 16380, "N": 1536},
    # Wide row past the bf16/fp16 block_size cap (65536//2 = 32768) so N>32768
    # forces the USE_BLOCKED persistent path in _rms_norm_kernel.
    {"name": "m64_n40960", "M": 64, "N": 40960},
]

DTYPES = ["bf16", "fp16"]
EPS = 1e-5
SEED = 20260601
WARMUP, ITERS = 10, 100

_HERE = os.path.dirname(os.path.abspath(__file__))


def _torch_dtype(name):
    import torch

    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _load_source():
    entry = os.path.join(_HERE, SOURCE_FILE)
    spec = importlib.util.spec_from_file_location("rmsnorm_src", entry)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_inputs(M, N, dtype, device="cuda"):
    import torch

    gen = torch.Generator(device=device)
    gen.manual_seed(SEED)
    x = torch.randn((M, N), generator=gen, device=device, dtype=dtype)
    weight = torch.randn((N,), generator=gen, device=device, dtype=dtype)
    return x, weight


def _torch_rmsnorm(x, g, out_dtype):
    # fp32-reduce reference (matches test_rmsnorm.py:torch_rmsnorm).
    import torch

    N = x.shape[1]
    x_f32 = x.float()
    g_f32 = g.float()
    rms = torch.sqrt(torch.sum(x_f32 * x_f32, dim=-1) * (1.0 / N))
    rsigma = 1.0 / rms
    out = x_f32 * rsigma.unsqueeze(1) * g_f32
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
    failures = []
    for shape in TEST_SHAPES:
        for dt in DTYPES:
            tag = f"{shape['name']}_{dt}"
            try:
                x, weight = _make_inputs(shape["M"], shape["N"], _torch_dtype(dt))
                y = mod.rms_norm(x, weight, EPS)
                torch.cuda.synchronize()
                ref = _torch_rmsnorm(x, weight, y.dtype)
                finite = bool(torch.isfinite(y).all().item())
                close = torch.allclose(y, ref, atol=1e-2, rtol=1e-2)
                ok = finite and close
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

    total = len(TEST_SHAPES) * len(DTYPES)
    status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{total})"
    print(f"Status: {status}")
    print(f"correctness: {'pass' if not failures else 'fail'}")
    return not failures


def run_benchmark(verbose=True):
    import torch

    mod = _load_source()
    report, latencies = [], []
    for idx, shape in enumerate(TEST_SHAPES):
        x, weight = _make_inputs(shape["M"], shape["N"], _torch_dtype("bf16"))
        fn = lambda: mod.rms_norm(x, weight, EPS)  # noqa: E731
        fn()
        torch.cuda.synchronize()
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        ms, bench_meta = benchmark_cuda_graph_or_events(
            fn, warmup=0, repetition=ITERS
        )
        latencies.append(ms)
        nbytes = 2.0 * shape["M"] * shape["N"] * 2  # bf16 read+write
        report.append(
            {
                "test_case_id": f"perf{idx + 1}",
                "execution_time_ms": ms,
                **bench_meta,
                "params": {k: shape[k] for k in ("M", "N")},
                "gbps": nbytes / (ms * 1e-3) / 1e9,
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
    parser = argparse.ArgumentParser(description="rmsnorm harness")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    args = parser.parse_args()

    print("=" * 62)
    print("triton2flydsl RMSNorm (forward)")
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
