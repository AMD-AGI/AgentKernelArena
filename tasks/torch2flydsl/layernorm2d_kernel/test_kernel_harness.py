#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Harness for the torch2flydsl layernorm2d starter task.

``model.py`` is the pure-torch specification and ``kernel.py`` is the FlyDSL
starter/target. Correctness always validates the reference against the independent
AMD runtime oracle ``aiter.layer_norm`` and also invokes ``flydsl_layernorm2d``.
Once implemented, the target is compared to the same oracle. Only the starter's
explicit ``NotImplementedError`` is a SKIP; missing entry points and all other
target errors fail validation.

The normalized worst-element gate is
``max|truth - result| / max|truth| <= REL_TOL``.

Modes:
  --compile         import model.py + kernel.py and run a CPU reference smoke pass
  --correctness     validate reference and implemented target against AITER truth
  --full-benchmark  time AITER/reference/target and report target latency when implemented
"""
import argparse
import ast
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
KERNEL_ENTRY = "flydsl_layernorm2d"


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


def _load_target():
    """Load the required target; absence is a broken task, not a starter SKIP."""
    kmod = _load_module(_KERNEL_DIR, KERNEL_FILE, "flydsl_kernel")
    assert kmod is not None, f"cannot load {KERNEL_FILE}"
    target = getattr(kmod, KERNEL_ENTRY)
    assert callable(target), f"{KERNEL_ENTRY} must be callable"
    return target


def _is_pure_starter_source(source):
    """Recognize only an unconditional top-level NotImplementedError stub."""
    tree = ast.parse(source)
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == KERNEL_ENTRY
    ]
    if len(matches) != 1:
        return False
    body = matches[0].body
    if body and isinstance(body[0], ast.Expr):
        value = body[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            body = body[1:]
    body = [node for node in body if not isinstance(node, ast.Pass)]
    if len(body) != 1 or not isinstance(body[0], ast.Raise):
        return False
    exc = body[0].exc
    if isinstance(exc, ast.Call):
        exc = exc.func
    return isinstance(exc, ast.Name) and exc.id == "NotImplementedError"


def _is_pure_starter():
    entry = Path(_KERNEL_DIR) / KERNEL_FILE
    return _is_pure_starter_source(entry.read_text(encoding="utf-8"))


def _probe_target(target, pure_starter, *args):
    """Catch NotImplementedError only for a statically proven pure starter."""
    try:
        return True, target(*args)
    except NotImplementedError:
        if not pure_starter:
            raise
        return False, None


_KERNEL_DIR = _resolve_kernel_dir()

# Real transformer hidden shapes (m rows, n hidden).
SHAPES = [
    {"name": "m1_n4096", "m": 1, "n": 4096},
    {"name": "m8_n4096", "m": 8, "n": 4096},
    {"name": "m32_n8192", "m": 32, "n": 8192},
    {"name": "m128_n8192", "m": 128, "n": 8192},
    {"name": "m256_n6144", "m": 256, "n": 6144},
    {"name": "m64_n4096", "m": 64, "n": 4096},
]

REL_TOL = 1e-2
SEED = 0
EPS = 1e-5


def _retry(fn, *, tries=5, what="op"):
    """Retry on transient OOM/contention (a 2nd worker may share the GPU)."""
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


def _make_inputs(shape, device="cuda"):
    import torch

    torch.manual_seed(SEED)
    m, n = shape["m"], shape["n"]
    input = torch.randn(m, n, dtype=torch.bfloat16, device=device)
    weight = torch.randn(n, dtype=torch.bfloat16, device=device)
    bias = torch.randn(n, dtype=torch.bfloat16, device=device)
    return input, weight, bias


def _norm_max_err(ref, out):
    ref_f, out_f = ref.float(), out.float()
    max_abs = (ref_f - out_f).abs().max().item()
    denom = ref_f.abs().max().item() + 1e-9
    return max_abs / denom, max_abs, denom


def run_correctness(verbose=True):
    import torch
    import aiter

    mmod = _load_module(_KERNEL_DIR, MODEL_FILE, "torch_model")
    assert mmod is not None, "cannot load model.py"
    target = _load_target()
    pure_starter = _is_pure_starter()

    init = mmod.get_init_inputs()
    smoke_model = mmod.Model(*init).to("cuda").eval()
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
    target_implemented = None
    for shape in SHAPES:
        m, n = shape["m"], shape["n"]
        model = mmod.Model(EPS).to("cuda").eval()
        input, weight, bias = _make_inputs(shape)

        with torch.no_grad():
            ref = model(input, weight, bias)

        truth = _retry(
            lambda: aiter.layer_norm(input, weight, bias, EPS), what=shape["name"]
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
                f"  {'PASS' if ok else 'FAIL'}: {shape['name']} (m{m}/n{n}) "
                f"ref-vs-aiter norm_max_err={err:.6f} (tol={REL_TOL}) "
                f"max_abs={max_abs:.5f} close%@1e-2={pct:.2f}"
            )
        if not ok:
            failures.append(shape["name"])

        target_args = (input, weight, bias, EPS)
        if target_implemented is None:
            target_implemented, kout = _probe_target(
                target, pure_starter, *target_args
            )
            if not target_implemented and verbose:
                print(
                    "        SKIP: kernel.py FlyDSL starter is not implemented "
                    "(reference was validated against AITER above)"
                )
        elif target_implemented:
            kout = target(*target_args)
        else:
            kout = None

        if target_implemented:
            assert kout is not None, f"{KERNEL_ENTRY} returned None"
            torch.cuda.synchronize()
            kerr, kmax_abs, _ = _norm_max_err(truth, kout)
            k_ok = kerr <= REL_TOL
            if verbose:
                print(
                    f"        {'PASS' if k_ok else 'FAIL'}: {shape['name']} "
                    f"kernel-vs-aiter norm_max_err={kerr:.6f} "
                    f"max_abs={kmax_abs:.5f}"
                )
            if not k_ok:
                failures.append(f"{shape['name']}:kernel")

        del model, input, weight, bias
        torch.cuda.empty_cache()

    status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{len(SHAPES)})"
    print(f"Status: {status}")
    print(f"worst normalized max error across all shapes: {worst:.6f} (tol={REL_TOL})")
    print(f"correctness: {'pass' if not failures else 'fail'}")
    assert not failures, f"correctness FAILED for: {failures}"
    return True


def run_benchmark(warmup=10, iters=100, verbose=True):
    import torch
    import aiter

    mmod = _load_module(_KERNEL_DIR, MODEL_FILE, "torch_model")
    assert mmod is not None, "cannot load model.py"
    target = _load_target()
    pure_starter = _is_pure_starter()

    probe_input, probe_weight, probe_bias = _make_inputs(SHAPES[0])
    target_implemented, probe_output = _probe_target(
        target, pure_starter, probe_input, probe_weight, probe_bias, EPS
    )
    if not target_implemented:
        print(
            "SKIP: kernel.py FlyDSL starter is not implemented "
            "(benchmarking the reference only; no target latency is claimed)"
        )
    else:
        assert probe_output is not None, f"{KERNEL_ENTRY} returned None"
    del probe_input, probe_weight, probe_bias, probe_output
    torch.cuda.empty_cache()

    latencies, report = [], []
    print(f"{'Config':<20} {'aiter':>12} {'TorchRef':>12} {'target':>12}")
    print("-" * 62)
    for idx, shape in enumerate(SHAPES):
        m, n = shape["m"], shape["n"]
        model = mmod.Model(EPS).to("cuda").eval()
        input, weight, bias = _make_inputs(shape)

        def run_ref():
            with torch.no_grad():
                return model(input, weight, bias)

        def run_truth():
            return aiter.layer_norm(input, weight, bias, EPS)

        def run_target():
            return target(input, weight, bias, EPS)

        _retry(run_truth, what=shape["name"])
        torch.cuda.synchronize()

        def _mean(fn):
            return benchmark_cuda_graph_or_events(
                fn, warmup=warmup, repetition=iters
            )

        ref_ms, ref_bench_meta = _mean(run_ref)
        aiter_ms, _aiter_bench_meta = _mean(run_truth)
        target_result = (
            _mean(run_target) if target_implemented else (None, None)
        )
        target_ms, target_bench_meta = target_result
        primary_ms = target_ms if target_ms is not None else ref_ms
        bench_meta = (
            target_bench_meta if target_ms is not None else ref_bench_meta
        )
        latencies.append(primary_ms)
        # bytes moved: input + output (bf16) + weight + bias (bf16).
        bytes_total = (m * n * 2 * 2) + (n * 2 * 2)
        gbps = bytes_total / (primary_ms * 1e-3) / 1e9
        report.append(
            {
                "test_case_id": f"test_case_{idx}",
                "execution_time_ms": primary_ms,
                **bench_meta,
                "reference_benchmark_method": ref_bench_meta["benchmark_method"],
                "benchmark_method_consistent": bench_meta["benchmark_method"] == ref_bench_meta["benchmark_method"],
                "shape": [m, n],
                "params": {"m": m, "n": n, "eps": EPS, "dtype": "bf16"},
                "aiter_ms": aiter_ms,
                "reference_ms": ref_ms,
                "target_ms": target_ms,
                "target_implemented": target_implemented,
                "gbps": gbps,
            }
        )
        if verbose:
            target_s = f"{target_ms:>10.4f}ms" if target_ms is not None else f"{'n/a':>12}"
            print(
                f"{shape['name']:<20} {aiter_ms:>10.4f}ms "
                f"{ref_ms:>10.4f}ms {target_s}"
            )
        del model, input, weight, bias
        torch.cuda.empty_cache()

    geomean_latency = math.exp(sum(math.log(x) for x in latencies) / len(latencies))
    build_dir = Path(_KERNEL_DIR) / "build"
    build_dir.mkdir(exist_ok=True)
    with open(build_dir / "performance_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("-" * 62)
    latency_kind = "target" if target_implemented else "reference fallback"
    print(f"Geometric mean {latency_kind} latency: {geomean_latency:.4f} ms")
    return {"geomean_latency_ms": geomean_latency}


def run_compile():
    mmod = _load_module(_KERNEL_DIR, MODEL_FILE, "torch_model")
    assert mmod is not None, "cannot load model.py"
    assert hasattr(mmod, "Model") and hasattr(mmod, "get_inputs"), "model.py contract"
    _load_target()
    inputs = mmod.get_inputs()
    out = mmod.Model(*mmod.get_init_inputs()).eval()(*inputs)
    assert out.shape == inputs[0].shape, "CPU reference smoke shape mismatch"
    print("compile ok")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="torch2flydsl layernorm2d harness")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    print("=" * 60)
    print("torch2flydsl layernorm2d (bf16, FlyDSL starter target)")
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
