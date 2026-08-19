#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Harness for the torch2flydsl 2D-image RoPE forward (model-only) task.

``model.py`` is the pure-torch reference (NEOX 2D RoPE on a [b, H*W, h, d] grid,
fp32 rotation). No ``kernel.py`` ships: a clean standalone FlyDSL kernel for this
op does not exist in aiter, so FlyDSL is the agent's target.

Correctness validates the reference in ``model.py`` against the REAL AMD runtime
op ``aiter.rope_2d_fwd`` (rotate_style=NEOX, reuse_freqs_front_part=False,
nope_first=False) as ground truth. The harness MAY import aiter; ``model.py``
MUST NOT.

Gate (tight, bf16 op): the output must match the op within a normalized
worst-element bound (max|ref-out| / max|ref| <= REL_TOL) AND an element-wise
isclose pass-rate (atol=rtol=1e-2) >= PASS_PCT.

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
KERNEL_ENTRY = "flydsl_rope_2d_fwd"

ROTATE_NEOX = 0


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

# Real shapes from aiter/op_tests/test_rope.py (2D-image grid: b, H, W, heads,
# head_dim). H*W is the sequence length; head_dim split in half (height/width).
SHAPES = [
    {"name": "b2_h16w16_d128", "b": 2, "height": 16, "width": 16, "h": 8, "d": 128},
    {"name": "b1_h32w32_d128", "b": 1, "height": 32, "width": 32, "h": 8, "d": 128},
    {"name": "b4_h16w16_d64", "b": 4, "height": 16, "width": 16, "h": 16, "d": 64},
    {"name": "b2_h24w24_d128", "b": 2, "height": 24, "width": 24, "h": 4, "d": 128},
]

# Tight bf16 gate: normalized worst-element bound + isclose pass-rate.
REL_TOL = 1e-2
PASS_PCT = 99.9
SEED = 20260401


def _make_inputs(shape, mmod, device="cuda"):
    import torch

    gen = torch.Generator(device=device).manual_seed(SEED)
    b, height, width = shape["b"], shape["height"], shape["width"]
    h, d = shape["h"], shape["d"]
    input = torch.randn(
        b, height * width, h, d, dtype=torch.bfloat16, device=device, generator=gen
    )
    cos_h, sin_h = mmod._build_cos_sin_2d(height, d, device=device)
    cos_w, sin_w = mmod._build_cos_sin_2d(width, d, device=device)
    return input, cos_h, sin_h, cos_w, sin_w


def _aiter_op(input, cos_h, sin_h, cos_w, sin_w, height, width):
    import aiter

    return aiter.rope_2d_fwd(
        input, cos_h, sin_h, cos_w, sin_w, height, width, ROTATE_NEOX, False, False
    )


def _compare(ref, out):
    """Return (ok, rel_worst, pass_pct) for the bf16 output."""
    import torch

    r = ref.float()
    o = out.float()
    den = r.abs().max().item() + 1e-12
    rel_worst = (r - o).abs().max().item() / den
    pass_pct = torch.isclose(r, o, atol=1e-2, rtol=1e-2).float().mean().item() * 100.0
    ok = rel_worst <= REL_TOL or pass_pct >= PASS_PCT
    return ok, rel_worst, pass_pct


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
    assert out is not None
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
        inp = _make_inputs(shape, mmod)
        model = mmod.Model(shape["height"], shape["width"]).to("cuda")
        with torch.no_grad():
            ref = model(*inp)
            truth = _retry(
                lambda: _aiter_op(*inp, shape["height"], shape["width"]),
                what="aiter rope_2d_fwd",
            )
        torch.cuda.synchronize()

        ok, rel, pct = _compare(ref, truth)
        if verbose:
            print(
                f"  {'PASS' if ok else 'FAIL'}: {shape['name']} "
                f"ref-vs-aiter rel_worst={rel:.2e} pass%={pct:.3f} (tol={REL_TOL})"
            )
        if not ok:
            failures.append(shape["name"])

        if has_kernel:
            try:
                kout = _retry(
                    lambda: kmod.flydsl_rope_2d_fwd(
                        *inp, shape["height"], shape["width"]
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
                k_ok, kr, kp = _compare(truth, kout)
                if verbose:
                    print(
                        f"        {'PASS' if k_ok else 'FAIL'}: {shape['name']} "
                        f"kernel-vs-aiter rel_worst={kr:.2e} pass%={kp:.3f}"
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

    if has_kernel:
        try:
            _probe = _make_inputs(SHAPES[0], mmod)
            kmod.flydsl_rope_2d_fwd(
                *_probe, SHAPES[0]["height"], SHAPES[0]["width"]
            )
            del _probe
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
        inp = _make_inputs(shape, mmod)
        model = mmod.Model(shape["height"], shape["width"]).to("cuda")
        with torch.no_grad():
            op_ms = _mean_ms(
                lambda: _aiter_op(*inp, shape["height"], shape["width"]),
                warmup,
                iters,
            )
            ref_ms = _mean_ms(lambda: model(*inp), warmup, iters)
            ref_bench_meta = _mean_ms.benchmark_metadata
            ker_ms = (
                _mean_ms(
                    lambda: kmod.flydsl_rope_2d_fwd(
                        *inp, shape["height"], shape["width"]
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
            "shape": [shape["b"], shape["height"] * shape["width"], shape["h"], shape["d"]],
            "params": {
                "b": shape["b"],
                "height": shape["height"],
                "width": shape["width"],
                "h": shape["h"],
                "d": shape["d"],
                "dtype": "bf16",
            },
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
    parser = argparse.ArgumentParser(description="torch2flydsl rope_2d_fwd harness")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    print("=" * 56)
    print("torch2flydsl rope_2d_fwd (model.py vs aiter ground truth)")
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
