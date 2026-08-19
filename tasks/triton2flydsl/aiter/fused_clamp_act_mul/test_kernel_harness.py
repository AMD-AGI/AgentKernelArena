#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Test harness for triton2flydsl/aiter/fused_clamp_act_mul (FLAT layout).

The kernel under test is AITER's fused clamped-SwiGLU Triton kernel
(`fused_clamp_act_mul` -> `_fused_clamp_silu_mul_kernel`), non-quant path used by
GPT-OSS / DeepSeek-V4-style FFNs: inp is [M, 2*N] (gate | up); when
swiglu_limit > 0 the gate is min-clamped and up is symmetric-clamped to the
limit, then out = act(gate) * up with optional row/element weights, fp32 compute,
output in inp.dtype.

Modes:
  --compile         ast-parse + import the standalone source, assert entry/kernel symbols
  --correctness     run the Triton kernel on TEST_SHAPES x configs, assert finite
                    output AND match the F.silu torch reference at the upstream
                    tolerance (atol=1e-2, rtol=1e-2)
                    [mirrors fusions/test_fused_clamp_act_mul.py:_torch_reference]
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

SOURCE_FILE = "fused_clamp_act_mul.py"
ENTRY = "fused_clamp_act_mul"
KERNEL = "_fused_clamp_silu_mul_kernel"

# (M, D) from op_tests/triton_tests/fusions/test_fused_clamp_act_mul.py
# (GPT-OSS clamped-swiglu FFN intermediate widths) x clamp/weight configs.
MS = [1, 2, 4, 8, 32]
DS = [2048, 3072]
LIMITS = [0.0, 7.0]
# weight modes: none, per-row broadcast [M,1], per-element [M,N]
WEIGHT_MODES = ["none", "broadcast", "elementwise"]

SEED = 20260601
WARMUP, ITERS = 10, 100

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_source():
    entry = os.path.join(_HERE, SOURCE_FILE)
    spec = importlib.util.spec_from_file_location("fused_clamp_act_mul_src", entry)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_inputs(M, D, weight_mode, device="cuda"):
    import torch

    gen = torch.Generator(device=device)
    gen.manual_seed(SEED)
    N = D // 2
    inp = torch.randn((M, D), generator=gen, device=device, dtype=torch.bfloat16)
    if weight_mode == "broadcast":
        w = torch.randn((M, 1), generator=gen, device=device, dtype=torch.float32) * 0.5
    elif weight_mode == "elementwise":
        w = torch.randn((M, N), generator=gen, device=device, dtype=torch.float32) * 0.1
    else:
        w = None
    return inp, w


def _torch_reference(inp, swiglu_limit, weights):
    # F.silu reference (matches test_fused_clamp_act_mul.py:_torch_reference, non-quant).
    import torch
    import torch.nn.functional as F

    gate, up = inp.chunk(2, dim=-1)
    if swiglu_limit > 0:
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
        gate = torch.clamp(gate, max=swiglu_limit)
    y = F.silu(gate) * up
    if weights is not None:
        y = weights * y
    return y.to(inp.dtype)


def _cases():
    cases = []
    for M in MS:
        for D in DS:
            for limit in LIMITS:
                for wm in WEIGHT_MODES:
                    name = f"m{M}_d{D}_lim{limit:g}_{wm}"
                    cases.append({"name": name, "M": M, "D": D, "limit": limit, "wm": wm})
    return cases


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
    cases = _cases()
    failures = []
    for c in cases:
        tag = c["name"]
        try:
            inp, w = _make_inputs(c["M"], c["D"], c["wm"])
            out = mod.fused_clamp_act_mul(
                inp, swiglu_limit=c["limit"], activation="silu", weights=w
            )
            torch.cuda.synchronize()
            ref = _torch_reference(inp, c["limit"], w)
            finite = bool(torch.isfinite(out).all().item())
            close = torch.allclose(out, ref, atol=1e-2, rtol=1e-2)
            ok = finite and close and (out.dtype == inp.dtype)
            if verbose:
                print(
                    f"  {'PASS' if ok else 'FAIL'}: {tag} "
                    f"out={tuple(out.shape)} finite={finite} close={close}"
                )
            if not ok:
                failures.append(tag)
        except Exception as e:  # noqa: BLE001
            failures.append(tag)
            if verbose:
                print(f"  FAIL: {tag} - {str(e)[:160]}")

    status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{len(cases)})"
    print(f"Status: {status}")
    print(f"correctness: {'pass' if not failures else 'fail'}")
    return not failures


def run_benchmark(verbose=True):
    import torch

    mod = _load_source()
    # bench a representative subset (swiglu clamp on, no weights) over (M, D).
    shapes = [{"name": f"m{M}_d{D}", "M": M, "D": D} for M in (8, 32) for D in DS]
    report, latencies = [], []
    for idx, shape in enumerate(shapes):
        inp, _ = _make_inputs(shape["M"], shape["D"], "none")
        fn = lambda: mod.fused_clamp_act_mul(  # noqa: E731
            inp, swiglu_limit=7.0, activation="silu", weights=None
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
        nbytes = shape["M"] * (shape["D"] + shape["D"] // 2) * 2
        report.append(
            {
                "test_case_id": f"perf{idx + 1}",
                "execution_time_ms": ms,
                **bench_meta,
                "params": {k: shape[k] for k in ("M", "D")},
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
    parser = argparse.ArgumentParser(description="fused_clamp_act_mul harness")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    args = parser.parse_args()

    print("=" * 62)
    print("triton2flydsl fused clamped-SwiGLU act-mul")
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
