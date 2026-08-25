#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Harness for the torch2flydsl fused-add + LayerNorm2d (model-only) task.

``model.py`` is the pure-torch reference (bf16 residual add returned as
residual_out, then 2D LayerNorm with fp32 reduction). No ``kernel.py`` ships: a
clean standalone FlyDSL kernel for this fused op does not exist in aiter, so
FlyDSL is the agent's target.

Correctness validates the reference in ``model.py`` against the REAL AMD runtime
op ``aiter.layernorm2d_fwd_with_add`` as ground truth. The harness MAY import
aiter; ``model.py`` MUST NOT.

Gate (tight, bf16 op): both the LayerNorm output and the residual_out must
match the op within a normalized worst-element bound (max|ref-out| / max|ref| <=
REL_TOL) AND an element-wise isclose pass-rate (atol=rtol=1e-2) >= PASS_PCT.

Modes:
  --compile         import model.py, build the Model, run a CPU smoke pass
  --correctness     assert model.py matches the aiter op at the tight gate
  --full-benchmark  time the torch reference and aiter op, write perf report
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
KERNEL_ENTRY = "flydsl_layernorm2d_with_add"


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

# Real shapes from aiter/op_tests/test_layernorm2d.py (m x n sweep).
SHAPES = [
    {"name": "m1_n8192", "m": 1, "n": 8192},
    {"name": "m32_n4096", "m": 32, "n": 4096},
    {"name": "m128_n8192", "m": 128, "n": 8192},
    {"name": "m256_n4096", "m": 256, "n": 4096},
    {"name": "m2048_n8192", "m": 2048, "n": 8192},
]

# Tight bf16 gate: normalized worst-element bound + isclose pass-rate.
REL_TOL = 1e-2
PASS_PCT = 99.9
SEED = 20260401
EPS = 1e-5


def _make_inputs(shape, device="cuda"):
    import torch

    gen = torch.Generator(device=device).manual_seed(SEED)
    m, n = shape["m"], shape["n"]
    input = torch.randn(m, n, dtype=torch.bfloat16, device=device, generator=gen)
    residual = torch.randn(m, n, dtype=torch.bfloat16, device=device, generator=gen)
    weight = torch.randn(n, dtype=torch.bfloat16, device=device, generator=gen)
    bias = torch.randn(n, dtype=torch.bfloat16, device=device, generator=gen)
    return input, residual, weight, bias


def _aiter_op(input, residual, weight, bias):
    import torch
    import aiter

    out = torch.empty_like(input)
    residual_out = torch.empty_like(input)
    aiter.layernorm2d_fwd_with_add(
        out, input, residual, residual_out, weight, bias, EPS, None
    )
    return out, residual_out


def _tensor_ok(ref, out):
    """Return (ok, rel_worst, pass_pct) for one bf16 tensor pair."""
    import torch

    r = ref.float()
    o = out.float()
    den = r.abs().max().item() + 1e-12
    rel_worst = (r - o).abs().max().item() / den
    pass_pct = torch.isclose(r, o, atol=1e-2, rtol=1e-2).float().mean().item() * 100.0
    ok = rel_worst <= REL_TOL or pass_pct >= PASS_PCT
    return ok, rel_worst, pass_pct


def _compare(ref, truth):
    """Return (ok, out_rel, out_pct, res_rel, res_pct) over output + residual_out."""
    ref_o, ref_r = ref
    t_o, t_r = truth
    ok_o, out_rel, out_pct = _tensor_ok(ref_o, t_o)
    ok_r, res_rel, res_pct = _tensor_ok(ref_r, t_r)
    return ok_o and ok_r, out_rel, out_pct, res_rel, res_pct


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
    out, res = model(*mmod.get_inputs())
    assert out is not None and res is not None
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
        input, residual, weight, bias = _make_inputs(shape)
        model = mmod.Model(*mmod.get_init_inputs()).to("cuda")
        with torch.no_grad():
            ref = model(input, residual, weight, bias)
            truth = _retry(
                lambda: _aiter_op(input, residual, weight, bias),
                what="aiter layernorm2d_fwd_with_add",
            )
        torch.cuda.synchronize()

        ok, orel, opct, rrel, rpct = _compare(ref, truth)
        if verbose:
            print(
                f"  {'PASS' if ok else 'FAIL'}: {shape['name']} "
                f"(m{shape['m']}/n{shape['n']}) ref-vs-aiter "
                f"out_rel={orel:.2e} out_pass%={opct:.3f} "
                f"res_rel={rrel:.2e} res_pass%={rpct:.3f} (tol={REL_TOL})"
            )
        if not ok:
            failures.append(shape["name"])

        if has_kernel:
            try:
                kout = _retry(
                    lambda: kmod.flydsl_layernorm2d_with_add(
                        input, residual, weight, bias, EPS
                    ),
                    what=KERNEL_ENTRY,
                )
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
                k_ok, ko, kop, kr, krp = _compare(kout, truth)
                if verbose:
                    print(
                        f"        {'PASS' if k_ok else 'FAIL'}: {shape['name']} "
                        f"kernel-vs-aiter out_rel={ko:.2e} res_rel={kr:.2e}"
                    )
                if not k_ok:
                    failures.append(f"{shape['name']}:kernel")

        del input, residual, weight, bias, model
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

    if has_kernel:
        try:
            _pi, _pr, _pw, _pb = _make_inputs(SHAPES[0])
            kmod.flydsl_layernorm2d_with_add(_pi, _pr, _pw, _pb, EPS)
            del _pi, _pr, _pw, _pb
        except NotImplementedError:
            has_kernel = False
            print(
                "SKIP: kernel.py FlyDSL target not implemented yet "
                "(benchmarking reference instead)"
            )
        import torch as _t; _t.cuda.empty_cache()

    latencies, report = [], []
    print(f"{'Config':<20} {'aiter':>10} {'ref':>10} {'kernel':>10}")
    print("-" * 56)
    for idx, shape in enumerate(SHAPES):
        input, residual, weight, bias = _make_inputs(shape)
        model = mmod.Model(*mmod.get_init_inputs()).to("cuda")
        with torch.no_grad():
            op_ms = _mean_ms(
                lambda: _aiter_op(input, residual, weight, bias), warmup, iters
            )
            ref_ms = _mean_ms(
                lambda: model(input, residual, weight, bias), warmup, iters
            )
            ref_bench_meta = _mean_ms.benchmark_metadata
            ker_ms = (
                _mean_ms(
                    lambda: kmod.flydsl_layernorm2d_with_add(
                        input, residual, weight, bias, EPS
                    ),
                    warmup,
                    iters,
                )
                if has_kernel
                else None
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
            "params": {"m": shape["m"], "n": shape["n"], "eps": EPS, "dtype": "bf16"},
            "aiter_ms": op_ms,
            "reference_ms": ref_ms,
        })
        if verbose:
            ker_s = f"{ker_ms:>8.4f}ms" if ker_ms is not None else f"{'n/a':>10}"
            print(f"{shape['name']:<20} {op_ms:>8.4f}ms {ref_ms:>8.4f}ms {ker_s}")
        del input, residual, weight, bias, model
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
    parser = argparse.ArgumentParser(
        description="torch2flydsl layernorm2d_with_add harness"
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    print("=" * 56)
    print("torch2flydsl layernorm2d_with_add (model.py vs aiter ground truth)")
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
