#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Build / correctness / performance harness for the silu_and_mul task.

This is a model-only task: there is no shipped FlyDSL ``kernel.py`` (FlyDSL is
the agent's target). Correctness therefore validates the pure-torch reference in
``model.py`` against AMD's real runtime op (``aiter.silu_and_mul``) as ground
truth. ``model.py`` itself imports no ``aiter``/``flydsl``; only this harness may.

The gate is the normalized max error ``max|ref - truth| / max|truth|`` <=
``REL_TOL`` (bf16 floor); element-wise close% at 1e-2 is also reported. Once an
agent drops a ``kernel.py`` exposing ``flydsl_silu_and_mul``, the harness also
checks that kernel against the same reference.

Modes:
  --compile         import the reference, build the Model, run a CPU smoke pass
  --correctness     compare the reference (and kernel.py if present) vs the op
  --full-benchmark  time the op + reference (+ kernel.py if present), write report
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from _aka_benchmark import benchmark_cuda_graph_or_events

KERNEL_FILE = "kernel.py"
MODEL_FILE = "model.py"
KERNEL_ENTRY = "flydsl_silu_and_mul"


def _resolve_kernel_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.isfile(os.path.join(here, MODEL_FILE)):
        return here
    cwd = os.getcwd()
    if os.path.isfile(os.path.join(cwd, MODEL_FILE)):
        return cwd
    return here


def _load_module(kernel_dir, filename, alias):
    entry = os.path.join(kernel_dir, filename)
    if not os.path.isfile(entry):
        return None
    if kernel_dir not in sys.path:
        sys.path.insert(0, kernel_dir)
    spec = importlib.util.spec_from_file_location(alias, entry)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


_KERNEL_DIR = _resolve_kernel_dir()

# Real silu_and_mul shapes from aiter/op_tests/test_activation.py
# (m in {1, 32, ..., 8192}, n in {1024, 4096, 6400, 8192}); input is [m, n],
# output [m, n // 2]. Covers small-m, power-of-two and non-power-of-two d.
SHAPES = [
    {"name": "m1_n4096", "m": 1, "n": 4096},
    {"name": "m128_n8192", "m": 128, "n": 8192},
    {"name": "m1024_n6400", "m": 1024, "n": 6400},
    {"name": "m4096_n4096", "m": 4096, "n": 4096},
]

# GPT-OSS clamp threshold; 0.0 selects the unclamped silu_and_mul path.
LIMIT = 0.0
# Tight element-wise gate: normalized max error <= REL_TOL (bf16 floor).
REL_TOL = 1e-2
SEED = 20260401


def _make_inputs(shape, device="cuda"):
    import torch

    gen = torch.Generator(device=device).manual_seed(SEED)
    return torch.randn(
        shape["m"], shape["n"], dtype=torch.bfloat16, device=device, generator=gen
    )


def _aiter_op(inp):
    import torch
    import aiter

    m, n = inp.shape
    out = torch.empty((m, n // 2), dtype=torch.bfloat16, device=inp.device)
    aiter.silu_and_mul(out, inp, LIMIT)
    return out


def _retry(fn, tries=5, what="op"):
    import torch

    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - transient HIP/OOM on a shared GPU
            msg = str(exc).lower()
            if "out of memory" in msg or "hip" in msg:
                last = exc
                torch.cuda.empty_cache()
                time.sleep(2.0 * (i + 1))
                continue
            raise
    raise RuntimeError(f"{what} failed after {tries} retries: {last}")


def run_compile(verbose=True):
    mmod = _load_module(_KERNEL_DIR, MODEL_FILE, "torch_model")
    assert mmod is not None, "cannot load model.py"
    model = mmod.Model(*mmod.get_init_inputs())
    out = model(*mmod.get_inputs())
    assert out is not None and out.shape[-1] * 2 == mmod.get_inputs()[0].shape[-1]
    if verbose:
        print("compile ok")
    return True


def run_correctness(verbose=True):
    import torch

    mmod = _load_module(_KERNEL_DIR, MODEL_FILE, "torch_model")
    assert mmod is not None, "cannot load model.py"
    kmod = _load_module(_KERNEL_DIR, KERNEL_FILE, "flydsl_kernel")
    has_kernel = kmod is not None and hasattr(kmod, KERNEL_ENTRY)

    failures = []
    for shape in SHAPES:
        inp = _make_inputs(shape)
        model = mmod.Model(*mmod.get_init_inputs())
        with torch.no_grad():
            ref = model(inp).float()
            truth = _retry(lambda: _aiter_op(inp), what="aiter.silu_and_mul").float()
        torch.cuda.synchronize()

        max_abs = (ref - truth).abs().max().item()
        scale = truth.abs().max().item() + 1e-9
        rel_err = max_abs / scale
        pct1e2 = torch.isclose(ref, truth, atol=1e-2, rtol=1e-2).float().mean().item() * 100
        ok = rel_err <= REL_TOL
        if verbose:
            print(
                f"  {'PASS' if ok else 'FAIL'}: {shape['name']} "
                f"(m{shape['m']}/n{shape['n']}) ref-vs-aiter "
                f"norm_max_err={rel_err:.6f} (tol={REL_TOL}) "
                f"max_abs={max_abs:.5f} close%@1e-2={pct1e2:.2f}"
            )
        if not ok:
            failures.append(shape["name"])

        if has_kernel:
            try:
                kout = _retry(
                    lambda: kmod.flydsl_silu_and_mul(inp, LIMIT), what=KERNEL_ENTRY
                ).float()
            except NotImplementedError:
                has_kernel = False
                if verbose:
                    print(
                        "        SKIP: kernel.py FlyDSL target not implemented yet "
                        "(reference validated against the aiter op above)"
                    )
                kout = None
            if kout is not None:
                torch.cuda.synchronize()
                k_abs = (ref - kout).abs().max().item()
                k_rel = k_abs / (ref.abs().max().item() + 1e-9)
                k_ok = k_rel <= REL_TOL
                if verbose:
                    print(
                        f"        {'PASS' if k_ok else 'FAIL'}: {shape['name']} "
                        f"kernel-vs-ref norm_max_err={k_rel:.6f}"
                    )
                if not k_ok:
                    failures.append(f"{shape['name']}:kernel")

        del inp, model
        torch.cuda.empty_cache()

    status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{len(SHAPES)})"
    print(f"Status: {status}")
    print(f"correctness: {'pass' if not failures else 'fail'}")
    assert not failures, f"correctness FAILED for: {failures}"
    return True


def _mean_ms(fn, warmup, iters):
    mean_ms, bench_meta = benchmark_cuda_graph_or_events(
        fn, warmup=warmup, repetition=iters
    )
    _mean_ms.benchmark_metadata = bench_meta
    return mean_ms


def run_benchmark(warmup=10, iters=100, verbose=True):
    import torch

    mmod = _load_module(_KERNEL_DIR, MODEL_FILE, "torch_model")
    assert mmod is not None, "cannot load model.py"
    kmod = _load_module(_KERNEL_DIR, KERNEL_FILE, "flydsl_kernel")
    has_kernel = kmod is not None and hasattr(kmod, KERNEL_ENTRY)

    latencies, report = [], []
    print(f"{'Config':<20} {'aiter':>10} {'ref':>10} {'kernel':>10}")
    print("-" * 56)
    for idx, shape in enumerate(SHAPES):
        inp = _make_inputs(shape)
        model = mmod.Model(*mmod.get_init_inputs())
        with torch.no_grad():
            op_ms = _mean_ms(lambda: _aiter_op(inp), warmup, iters)
            ref_ms = _mean_ms(lambda: model(inp), warmup, iters)
            ref_bench_meta = _mean_ms.benchmark_metadata
            ker_ms = None
            if has_kernel:
                try:
                    ker_ms = _mean_ms(
                        lambda: kmod.flydsl_silu_and_mul(inp, LIMIT), warmup, iters
                    )
                except NotImplementedError:
                    has_kernel = False
                    if verbose:
                        print(
                            "        SKIP: kernel.py FlyDSL target not implemented "
                            "yet (benchmarking reference instead)"
                        )

        primary_ms = ker_ms if ker_ms is not None else ref_ms
        bench_meta = _mean_ms.benchmark_metadata
        latencies.append(primary_ms)
        report.append({
            "test_case_id": f"test_case_{idx}",
            "execution_time_ms": primary_ms,
            **bench_meta,
            "reference_benchmark_method": ref_bench_meta["benchmark_method"],
            "benchmark_method_consistent": bench_meta["benchmark_method"] == ref_bench_meta["benchmark_method"],
            "shape": [shape["m"], shape["n"]],
            "params": {"m": shape["m"], "n": shape["n"], "limit": LIMIT},
            "aiter_ms": op_ms,
            "reference_ms": ref_ms,
        })
        if verbose:
            ker_s = f"{ker_ms:>8.4f}ms" if ker_ms is not None else f"{'n/a':>10}"
            print(f"{shape['name']:<20} {op_ms:>8.4f}ms {ref_ms:>8.4f}ms {ker_s}")
        del inp, model
        torch.cuda.empty_cache()

    geomean = math.exp(sum(math.log(x) for x in latencies) / len(latencies))
    build_dir = Path(_KERNEL_DIR) / "build"
    build_dir.mkdir(exist_ok=True)
    with open(build_dir / "performance_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("-" * 56)
    print(f"Geometric mean latency: {geomean:.4f} ms")
    return {"geomean_latency_ms": geomean}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="torch2flydsl silu_and_mul harness")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    print("=" * 56)
    print("torch2flydsl silu_and_mul (model.py vs aiter ground truth)")
    print("=" * 56)

    if args.compile:
        try:
            run_compile()
        except AssertionError as exc:
            print(f"ASSERTION: {exc}")
            sys.exit(1)
        sys.exit(0)
    elif args.correctness:
        try:
            run_correctness()
        except AssertionError as exc:
            print(f"ASSERTION: {exc}")
            sys.exit(1)
        sys.exit(0)
    else:
        run_benchmark(warmup=args.warmup, iters=args.iterations)
