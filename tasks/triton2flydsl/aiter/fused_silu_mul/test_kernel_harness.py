#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Test harness for triton2flydsl/aiter/fused_silu_mul (FLAT layout).

The kernel under test is AITER's fused SiLU-and-mul Triton kernel
(`fused_silu_mul` -> `fused_silu_mul_kernel` / `_silu_exp2`): for a last dim of
2*d, the first d lanes go through SiLU (exp2 form) in fp32 and are multiplied by
the second d lanes, written in the input dtype.

Modes:
  --compile         ast-parse + import the standalone source, assert entry/kernel symbols
  --correctness     run the Triton kernel on TEST_SHAPES, assert finite output AND
                    match the fp32 silu_exp2 torch reference at the upstream
                    tolerance (atol=1e-2, rtol=1e-2)
                    [mirrors fusions/test_fused_silu_mul.py:torch_silu_mul_last_dim_ref]
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

SOURCE_FILE = "fused_silu_mul.py"
ENTRY = "fused_silu_mul"
KERNEL = "fused_silu_mul_kernel"

LOG2_E = 1.44269504089

# (rows, last_dim) from op_tests/triton_tests/fusions/test_fused_silu_mul.py.
# GLM-4.7-FP8 MoE TP4: moe_intermediate_size=1536 -> local d=384, last=768, top_k=8.
# Kimi-K2.5 MoE TP4: moe_intermediate_size=2048 -> local d=512, last=1024, top_k=8.
# Plus the basic shapes and a tall prefill+decode mix.
_GLM47_TP4_LAST = 768
_KIMI_K25_TP4_LAST = 1024
TOP_K = 8
TEST_SHAPES = [
    {"name": "r4_l64", "rows": 4, "last": 64},
    {"name": "r128_l256", "rows": 128, "last": 256},
    {"name": "r31_l500", "rows": 31, "last": 500},
    {"name": "glm47_tp4_decode4", "rows": 4 * TOP_K, "last": _GLM47_TP4_LAST},
    {"name": "kimi_k25_tp4_decode4", "rows": 4 * TOP_K, "last": _KIMI_K25_TP4_LAST},
    {"name": "glm47_tp4_rows256x8", "rows": 256 * TOP_K, "last": _GLM47_TP4_LAST},
    {"name": "kimi_k25_tp4_rows256x8", "rows": 256 * TOP_K, "last": _KIMI_K25_TP4_LAST},
    {"name": "glm47_tp4_pref8190_dec3", "rows": (8190 + 3) * TOP_K, "last": _GLM47_TP4_LAST},
    {"name": "kimi_k25_tp4_pref7235_dec3", "rows": (7235 + 3) * TOP_K, "last": _KIMI_K25_TP4_LAST},
]

DTYPES = ["bf16", "fp16"]
SEED = 20260601
WARMUP, ITERS = 10, 100

_HERE = os.path.dirname(os.path.abspath(__file__))


def _torch_dtype(name):
    import torch

    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _load_source():
    entry = os.path.join(_HERE, SOURCE_FILE)
    spec = importlib.util.spec_from_file_location("fused_silu_mul_src", entry)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_inputs(rows, last, dtype, device="cuda"):
    import torch

    gen = torch.Generator(device=device)
    gen.manual_seed(SEED)
    return torch.randn((rows, last), generator=gen, device=device, dtype=dtype)


def _torch_silu_mul(x):
    # fp32 exp2-form silu reference (matches test_fused_silu_mul.py).
    import torch

    d = x.size(-1) // 2
    a, b = x[..., :d], x[..., d:]
    af = a.float()
    silu = af / (1.0 + torch.exp2(-(af * LOG2_E)))
    return (silu * b).to(x.dtype)


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
                x = _make_inputs(shape["rows"], shape["last"], _torch_dtype(dt))
                y = mod.fused_silu_mul(x)
                torch.cuda.synchronize()
                ref = _torch_silu_mul(x)
                finite = bool(torch.isfinite(y).all().item())
                close = torch.allclose(y, ref, atol=1e-2, rtol=1e-2)
                ok = finite and close
                if verbose:
                    print(
                        f"  {'PASS' if ok else 'FAIL'}: {tag} "
                        f"(rows={shape['rows']},last={shape['last']}) "
                        f"finite={finite} close={close}"
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
        x = _make_inputs(shape["rows"], shape["last"], _torch_dtype("bf16"))
        fn = lambda: mod.fused_silu_mul(x)  # noqa: E731
        fn()
        torch.cuda.synchronize()
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        ms, bench_meta = benchmark_cuda_graph_or_events(
            fn, warmup=0, repetition=ITERS
        )
        latencies.append(ms)
        # read 2*d + write d per row, bf16
        nbytes = shape["rows"] * (shape["last"] + shape["last"] // 2) * 2
        report.append(
            {
                "test_case_id": f"perf{idx + 1}",
                "execution_time_ms": ms,
                **bench_meta,
                "params": {k: shape[k] for k in ("rows", "last")},
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
    parser = argparse.ArgumentParser(description="fused_silu_mul harness")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    args = parser.parse_args()

    print("=" * 62)
    print("triton2flydsl fused SiLU-and-mul")
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
