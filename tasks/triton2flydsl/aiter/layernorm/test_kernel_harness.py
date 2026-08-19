#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Test harness for triton2flydsl/aiter/layernorm (FLAT layout).

The kernel under test is AITER's LayerNorm forward Triton kernel (`layer_norm` ->
`_layernorm_kernel`): fp32 mean/variance reduction, rsqrt normalization, affine
weight + bias, written in the input dtype. One program normalizes a full row.

Modes:
  --compile         ast-parse + import the standalone source, assert entry/kernel symbols
  --correctness     run the Triton kernel on TEST_SHAPES, assert finite output AND
                    match torch.nn.functional.layer_norm at the upstream bf16/fp16
                    tolerance (1e-2)  [mirrors op_tests/.../test_layernorm.py:run_torch]
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

SOURCE_FILE = "layernorm.py"
ENTRY = "layer_norm"
KERNEL = "_layernorm_kernel"

# (M, N) from op_tests/triton_tests/normalization/test_layernorm.py::get_vals
# (the enabled shapes) plus a couple of common transformer hidden sizes.
TEST_SHAPES = [
    {"name": "m2_n128", "M": 2, "N": 128},
    {"name": "m1_n4", "M": 1, "N": 4},
    {"name": "m128_n2", "M": 128, "N": 2},
    {"name": "m1_n128", "M": 1, "N": 128},
    {"name": "m359_n1", "M": 359, "N": 1},
    {"name": "m1_n359", "M": 1, "N": 359},
    {"name": "m1_n131072", "M": 1, "N": 131072},
    {"name": "m1_n89999", "M": 1, "N": 89999},
    {"name": "m4096_n4096", "M": 4096, "N": 4096},
    {"name": "m2048_n8192", "M": 2048, "N": 8192},
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
    spec = importlib.util.spec_from_file_location("layernorm_src", entry)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_inputs(M, N, dtype, device="cuda"):
    import torch

    gen = torch.Generator(device=device)
    gen.manual_seed(SEED)
    x = torch.randn((M, N), generator=gen, device=device, dtype=dtype)
    weight = torch.rand((N,), generator=gen, device=device, dtype=dtype)
    bias = torch.rand((N,), generator=gen, device=device, dtype=dtype)
    return x, weight, bias


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
        for dt in DTYPES:
            tag = f"{shape['name']}_{dt}"
            try:
                x, weight, bias = _make_inputs(shape["M"], shape["N"], _torch_dtype(dt))
                y = mod.layer_norm(x, weight, bias, EPS)
                torch.cuda.synchronize()
                ref = F.layer_norm(
                    x, (shape["N"],), weight=weight, bias=bias, eps=EPS
                )
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
        x, weight, bias = _make_inputs(shape["M"], shape["N"], _torch_dtype("bf16"))
        fn = lambda: mod.layer_norm(x, weight, bias, EPS)  # noqa: E731
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
    parser = argparse.ArgumentParser(description="layernorm harness")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    args = parser.parse_args()

    print("=" * 62)
    print("triton2flydsl LayerNorm (forward)")
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
