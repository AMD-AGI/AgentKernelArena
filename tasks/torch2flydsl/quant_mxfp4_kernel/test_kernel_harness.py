#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Build / correctness / performance harness for the quant_mxfp4 task.

Model-only task: there is no shipped FlyDSL ``kernel.py`` (FlyDSL is the agent's
target). Correctness validates the pure-torch reference in ``model.py`` against
AMD's real runtime op (``aiter.quant_mxfp4_hip`` / ``per_1x32_f4_quant``,
project default round mode RoundUp) as ground truth. ``model.py`` imports no
``aiter``/``flydsl``; only this harness may.

Gate (EXACT, byte-for-byte): the packed FP4 (E2M1) codes and the E8M0 block
scales must match the device op bit-for-bit (this holds on gfx950, where the
E2M1 conversion uses the hardware round-to-nearest-even path). Once an agent
drops a ``kernel.py`` exposing ``flydsl_quant_mxfp4``, the harness also checks it
against the same op.

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
KERNEL_ENTRY = "flydsl_quant_mxfp4"
GROUP_SIZE = 32


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

# Real shapes from aiter/op_tests/test_quant_mxfp4.py (no_shuffle_shapes); n must
# be a multiple of 32. Includes small-m, odd-m, and non-power-of-two n.
SHAPES = [
    {"name": "m1_n32", "m": 1, "n": 32},
    {"name": "m3_n128", "m": 3, "n": 128},
    {"name": "m125_n64", "m": 125, "n": 64},
    {"name": "m4096_n128", "m": 4096, "n": 128},
    {"name": "m4096_n256", "m": 4096, "n": 256},
    {"name": "m4096_n1024", "m": 4096, "n": 1024},
    {"name": "m4097_n256", "m": 4097, "n": 256},
]

# EXACT gate: packed FP4 codes and E8M0 scales must match byte-for-byte.
SEED = 20260401


def _make_inputs(shape, device="cuda"):
    import torch

    gen = torch.Generator(device=device).manual_seed(SEED)
    return (
        torch.randn(
            shape["m"], shape["n"], dtype=torch.bfloat16, device=device, generator=gen
        ),
    )


def _aiter_op(inp):
    from aiter.ops.quant import quant_mxfp4_hip

    return quant_mxfp4_hip(inp, group_size=GROUP_SIZE)


def _compare(ref, truth):
    """Return (ok, packed_max_diff, packed_exact_pct, scale_max_diff,
    scale_exact_pct)."""
    import torch

    ref_p, ref_s = ref
    t_p, t_s = truth
    pa = ref_p.view(torch.uint8).to(torch.int32).cpu()
    pb = t_p.view(torch.uint8).to(torch.int32).cpu()
    pd = (pa - pb).abs()
    p_max = int(pd.max().item())
    p_exact = (pd == 0).float().mean().item() * 100.0
    sa = ref_s.view(torch.uint8).to(torch.int32).cpu()
    sb = t_s.view(torch.uint8).to(torch.int32).cpu()
    sd = (sa - sb).abs()
    s_max = int(sd.max().item())
    s_exact = (sd == 0).float().mean().item() * 100.0
    ok = p_max == 0 and s_max == 0
    return ok, p_max, p_exact, s_max, s_exact


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
    packed, scale = model(*mmod.get_inputs())
    assert packed is not None and scale is not None
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
        model = mmod.Model(*mmod.get_init_inputs()).to("cuda")
        with torch.no_grad():
            ref = model(*inp)
            truth = _retry(lambda: _aiter_op(*inp), what="aiter quant_mxfp4")
        torch.cuda.synchronize()

        ok, pmax, ppct, smax, spct = _compare(ref, truth)
        if verbose:
            print(
                f"  {'PASS' if ok else 'FAIL'}: {shape['name']} "
                f"(m{shape['m']}/n{shape['n']}) ref-vs-aiter "
                f"packed_max_diff={pmax} exact%={ppct:.3f} | "
                f"scale_max_diff={smax} exact%={spct:.3f}"
            )
        if not ok:
            failures.append(shape["name"])

        if has_kernel:
            try:
                kout = _retry(lambda: kmod.flydsl_quant_mxfp4(*inp, group_size=GROUP_SIZE),
                              what=KERNEL_ENTRY)
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
                k_ok, kp, kpp, ks, ksp = _compare(kout, truth)
                if verbose:
                    print(
                        f"        {'PASS' if k_ok else 'FAIL'}: {shape['name']} "
                        f"kernel-vs-aiter packed_max_diff={kp} exact%={kpp:.3f} | "
                        f"scale_max_diff={ks} exact%={ksp:.3f}"
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
            _probe = _make_inputs(SHAPES[0])
            kmod.flydsl_quant_mxfp4(*_probe, group_size=GROUP_SIZE)
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
        inp = _make_inputs(shape)
        model = mmod.Model(*mmod.get_init_inputs()).to("cuda")
        with torch.no_grad():
            op_ms = _mean_ms(lambda: _aiter_op(*inp), warmup, iters)
            ref_ms = _mean_ms(lambda: model(*inp), warmup, iters)
            ref_bench_meta = _mean_ms.benchmark_metadata
            ker_ms = (
                _mean_ms(
                    lambda: kmod.flydsl_quant_mxfp4(*inp, group_size=GROUP_SIZE),
                    warmup, iters,
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
            "params": {"m": shape["m"], "n": shape["n"], "group_size": GROUP_SIZE,
                       "dtype": "mxfp4_e2m1"},
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
    try:
        import torch as _t
        _arch = _t.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    except Exception:
        _arch = ""
    if _arch != "gfx950":
        print(f"SKIPPED: gfx950-only task on arch={_arch or 'unknown'} (FP4/MX scaled-MFMA requires CDNA4/gfx950)")
        print("correctness: skip")
        sys.exit(0)
    parser = argparse.ArgumentParser(description="torch2flydsl quant_mxfp4 harness")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    print("=" * 56)
    print("torch2flydsl quant_mxfp4 (model.py vs aiter ground truth)")
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
