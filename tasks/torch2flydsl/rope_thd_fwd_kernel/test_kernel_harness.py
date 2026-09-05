#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Test harness for the torch2flydsl rope_thd_fwd task.

``model.py`` is the pure-torch reference (variable-length thd RoPE, bf16 I/O,
fp32 rotation). ``kernel.py`` is the editable FlyDSL target; its starter entry
raises ``NotImplementedError`` until a solution is supplied.

The reference in ``model.py`` is validated against the real AMD runtime op
``aiter.rope_thd_fwd`` (the ground truth), and an implemented FlyDSL target is
validated against the same oracle. The harness MAY import aiter; ``model.py``
MUST NOT. The normalized worst-element error ``max|truth - ref| / max|truth|``
must be <= REL_TOL.

Modes:
  --compile         import model.py + aiter and report readiness
  --correctness     assert model.py matches aiter.rope_thd_fwd at the tight gate
  --full-benchmark  graph-first timing of FlyDSL (or starter fallback), reference,
                    and aiter; write the performance report
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
KERNEL_ENTRY = "flydsl_rope_thd_fwd"


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

# Real thd shapes from aiter/op_tests/test_rope.py: packed variable-length
# sequences (cu_seqlens), h heads, d head_dim, full rotary (rotary_dim == d),
# NEOX, reuse_freqs_front_part=True, nope_first=False.
SHAPES = [
    {"name": "seqs5_h8_d128", "cu": [0, 100, 228, 484, 712, 1024], "h": 8, "d": 128},
    {"name": "seqs4_h16_d128", "cu": [0, 233, 456, 711, 1024], "h": 16, "d": 128},
    {
        "name": "seqs8_h8_d64",
        "cu": [0, 100, 102, 128, 233, 456, 460, 711, 1024],
        "h": 8,
        "d": 64,
    },
    {"name": "seqs3_h32_d128", "cu": [0, 512, 1024, 2048], "h": 32, "d": 128},
]

REL_TOL = 1e-2
SEED = 0
ROTATE_STYLE = 0  # NEOX
REUSE_FREQS_FRONT_PART = True
NOPE_FIRST = False


def _retry(fn, *, tries=5, what="op"):
    import torch

    delay = 0.5
    for attempt in range(tries):
        try:
            return fn()
        except RuntimeError as e:  # noqa: PERF203
            msg = str(e).lower()
            transient = "out of memory" in msg or "hip" in msg or "ran out" in msg
            if not transient or attempt == tries - 1:
                raise
            print(
                f"  [retry] transient GPU error on {what} "
                f"(attempt {attempt + 1}/{tries}): {str(e)[:80]} — backing off {delay:.1f}s"
            )
            torch.cuda.empty_cache()
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def _make_inputs(mmod, shape, device="cuda"):
    import torch

    torch.manual_seed(SEED)
    cu = shape["cu"]
    h, d = shape["h"], shape["d"]
    t = cu[-1]
    cu_seqlens = torch.tensor(cu, dtype=torch.int32, device=device)
    input = torch.randn(t, h, d, dtype=torch.bfloat16, device=device)
    freqs = mmod._build_freqs(
        t, rotary_dim=d, reuse_freqs_front_part=REUSE_FREQS_FRONT_PART, device=device
    ).to(torch.bfloat16)
    return input, cu_seqlens, freqs


def _aiter_op(input, cu_seqlens, freqs):
    import aiter

    return aiter.rope_thd_fwd(
        input, cu_seqlens, freqs, ROTATE_STYLE, REUSE_FREQS_FRONT_PART, NOPE_FIRST
    )


def _make_reference_runner(model, input, freqs, cu_values):
    """Build the exact Model reference without a timed device-to-host tolist()."""
    import torch

    seqlens = tuple(b - a for a, b in zip(cu_values, cu_values[1:]))
    slices = []
    start = 0
    for length in seqlens:
        slices.append((input[start : start + length], length))
        start += length

    def run_ref():
        with torch.no_grad():
            f = freqs.float()
            outs = []
            for xi, length in slices:
                x = xi.float().unsqueeze(1)
                outs.append(model._rope_sbhd(x, f[:length]).squeeze(1))
            return torch.cat(outs).to(input.dtype)

    return run_ref


def _norm_max_err(ref, out):
    ref_f, out_f = ref.float(), out.float()
    max_abs = (ref_f - out_f).abs().max().item()
    denom = ref_f.abs().max().item() + 1e-9
    return max_abs / denom, max_abs, denom


def run_correctness(verbose=True):
    import torch
    import aiter  # noqa: F401

    mmod = _load_module(_KERNEL_DIR, MODEL_FILE, "torch_model")
    assert mmod is not None, "cannot load model.py"
    kmod = _load_module(_KERNEL_DIR, KERNEL_FILE, "flydsl_kernel")
    has_kernel = kmod is not None and hasattr(kmod, KERNEL_ENTRY)

    # End-to-end smoke: Model(*get_init_inputs()) + get_inputs() must run.
    init = mmod.get_init_inputs()
    smoke_model = mmod.Model(*init).eval()
    with torch.no_grad():
        smoke_args = [a.to("cuda") for a in mmod.get_inputs()]
        smoke_out = smoke_model(*smoke_args)
    assert smoke_out.shape == smoke_args[0].shape, "smoke Model forward shape mismatch"
    if verbose:
        print(
            f"  smoke: Model(*get_init_inputs())+get_inputs() OK "
            f"(init={init}, out={tuple(smoke_out.shape)})"
        )

    failures = []
    worst = 0.0
    for shape in SHAPES:
        try:
            model = mmod.Model(ROTATE_STYLE, REUSE_FREQS_FRONT_PART, NOPE_FIRST).eval()
            input, cu_seqlens, freqs = _make_inputs(mmod, shape)

            with torch.no_grad():
                ref = model(input, cu_seqlens, freqs)

            truth = _retry(
                lambda: _aiter_op(input, cu_seqlens, freqs), what=shape["name"]
            )
            torch.cuda.synchronize()

            err, max_abs, _ = _norm_max_err(truth, ref)
            worst = max(worst, err)
            pct = (
                torch.isclose(truth.float(), ref.float(), atol=1e-2, rtol=1e-2)
                .float()
                .mean()
                .item()
                * 100
            )
            ok = err <= REL_TOL
            if verbose:
                print(
                    f"  {'PASS' if ok else 'FAIL'}: {shape['name']} "
                    f"norm_max_err={err:.6f} (tol={REL_TOL}) max_abs={max_abs:.5f} "
                    f"close%@1e-2={pct:.2f}"
                )
            if not ok:
                failures.append(shape["name"])

            if has_kernel:
                try:
                    kout = _retry(
                        lambda: kmod.flydsl_rope_thd_fwd(
                            input,
                            cu_seqlens,
                            freqs,
                            ROTATE_STYLE,
                            REUSE_FREQS_FRONT_PART,
                            NOPE_FIRST,
                        ),
                        what=f"{shape['name']}:kernel",
                    )
                except NotImplementedError:
                    has_kernel = False
                    print(
                        "        SKIP: kernel.py FlyDSL target not implemented yet "
                        "(reference validated against the aiter op above)"
                    )
                else:
                    torch.cuda.synchronize()
                    k_err, _, _ = _norm_max_err(truth, kout)
                    k_ok = k_err <= REL_TOL
                    if verbose:
                        print(
                            f"        {'PASS' if k_ok else 'FAIL'}: {shape['name']} "
                            f"kernel-vs-aiter norm_max_err={k_err:.6f}"
                        )
                    if not k_ok:
                        failures.append(f"{shape['name']}:kernel")
        except Exception as e:  # noqa: BLE001
            failures.append(shape["name"])
            if verbose:
                print(f"  FAIL: {shape['name']} - {str(e)[:160]}")

    status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{len(SHAPES)})"
    print(f"Status: {status}")
    print(f"worst normalized max error across all shapes: {worst:.6f} (tol={REL_TOL})")
    print(f"correctness: {'pass' if not failures else 'fail'}")
    assert not failures, f"correctness FAILED for: {failures}"
    return True


def run_benchmark(warmup=10, iters=100, verbose=True):
    import torch
    import aiter  # noqa: F401

    mmod = _load_module(_KERNEL_DIR, MODEL_FILE, "torch_model")
    assert mmod is not None, "cannot load model.py"
    kmod = _load_module(_KERNEL_DIR, KERNEL_FILE, "flydsl_kernel")
    has_kernel = kmod is not None and hasattr(kmod, KERNEL_ENTRY)

    latencies, report = [], []
    label = "FlyDSL" if has_kernel else "TorchRef"
    print(f"{'Config':<20} {'TorchRef':>12} {label:>12} {'aiter':>12}")
    print("-" * 62)
    for idx, shape in enumerate(SHAPES):
        model = mmod.Model(ROTATE_STYLE, REUSE_FREQS_FRONT_PART, NOPE_FIRST).eval()
        input, cu_seqlens, freqs = _make_inputs(mmod, shape)
        run_ref = _make_reference_runner(model, input, freqs, shape["cu"])

        def run_truth():
            return _aiter_op(input, cu_seqlens, freqs)

        def run_kernel():
            return kmod.flydsl_rope_thd_fwd(
                input,
                cu_seqlens,
                freqs,
                ROTATE_STYLE,
                REUSE_FREQS_FRONT_PART,
                NOPE_FIRST,
            )

        _retry(run_truth, what=shape["name"])
        torch.cuda.synchronize()

        if has_kernel:
            try:
                _retry(run_kernel, what=f"{shape['name']}:kernel")
            except NotImplementedError:
                has_kernel = False
                label = "TorchRef"
                if verbose:
                    print(
                        "SKIP: kernel.py FlyDSL target not implemented yet "
                        "(benchmarking Model baseline instead)"
                    )
            torch.cuda.synchronize()

        def _mean(fn):
            return benchmark_cuda_graph_or_events(
                fn, warmup=warmup, repetition=iters
            )

        ref_ms, ref_bench_meta = _mean(run_ref)
        aiter_ms, _aiter_bench_meta = _mean(run_truth)
        if has_kernel:
            primary_ms, bench_meta = _mean(run_kernel)
        else:
            primary_ms, bench_meta = ref_ms, ref_bench_meta
        latencies.append(primary_ms)
        t, h, d = input.shape
        # bytes moved: input + output (bf16) + freqs (bf16).
        bytes_total = (t * h * d * 2 * 2) + freqs.numel() * 2
        gbps = bytes_total / (primary_ms * 1e-3) / 1e9
        report.append(
            {
                "test_case_id": f"test_case_{idx}",
                "execution_time_ms": primary_ms,
                **bench_meta,
                "reference_benchmark_method": ref_bench_meta["benchmark_method"],
                "benchmark_method_consistent": bench_meta["benchmark_method"] == ref_bench_meta["benchmark_method"],
                "shape": [t, h, d],
                "params": {
                    "t": t,
                    "h": h,
                    "d": d,
                    "num_seqs": len(shape["cu"]) - 1,
                    "dtype": "bf16",
                },
                "reference_execution_time_ms": ref_ms,
                "aiter_ms": aiter_ms,
                "gbps": gbps,
            }
        )
        if verbose:
            print(
                f"{shape['name']:<20} {ref_ms:>10.4f}ms "
                f"{primary_ms:>10.4f}ms {aiter_ms:>10.4f}ms"
            )
        del model, input, cu_seqlens, freqs
        torch.cuda.empty_cache()

    geomean_latency = math.exp(sum(math.log(x) for x in latencies) / len(latencies))
    build_dir = Path(_KERNEL_DIR) / "build"
    build_dir.mkdir(exist_ok=True)
    with open(build_dir / "performance_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("-" * 62)
    print(f"Geometric mean primary latency: {geomean_latency:.4f} ms")
    return {"geomean_latency_ms": geomean_latency}


def run_compile():
    import torch  # noqa: F401
    import aiter  # noqa: F401

    mmod = _load_module(_KERNEL_DIR, MODEL_FILE, "torch_model")
    assert mmod is not None, "cannot load model.py"
    assert hasattr(mmod, "Model") and hasattr(mmod, "get_inputs"), "model.py contract"
    print("compile ok")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="torch2flydsl rope_thd_fwd harness")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    print("=" * 60)
    print("torch2flydsl rope_thd_fwd (bf16)")
    print("=" * 60)

    if args.compile:
        run_compile()
        sys.exit(0)
    if args.correctness:
        try:
            run_correctness()
        except AssertionError as exc:
            print(f"ASSERTION: {exc}")
            sys.exit(1)
        sys.exit(0)
    else:
        run_benchmark(warmup=args.warmup, iters=args.iterations)
