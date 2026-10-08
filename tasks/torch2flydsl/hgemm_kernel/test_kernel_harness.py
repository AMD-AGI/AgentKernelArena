#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Test harness for the torch2flydsl hgemm task.

Builds the PyTorch reference from model.py (`Model`/`get_inputs`/
`get_init_inputs`) and runs the FlyDSL kernel from kernel.py over a set of GEMM
shapes, comparing for correctness and benchmarking against a torch baseline.

Modes:
  --correctness     compare FlyDSL output to the PyTorch Model reference
  --full-benchmark  time FlyDSL vs torch baseline, write performance report
"""
import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from _aka_benchmark import TimedRun, benchmark_cuda_graph_or_events
from scripts.replay_checks import require_tensor_contract, require_unchanged, verify_timed_run
from scripts.sample_controls import MeasuredInputStream

from task_runtime import candidate_relative_path
KERNEL_FILE = candidate_relative_path()
ARENA_PROVIDED_BASELINE = False
MODEL_FILE = "model.py"


def _resolve_kernel_dir():
    return str(__import__("pathlib").Path(__file__).resolve().parent)


def _load_module(kernel_dir, filename, alias):
    if filename == KERNEL_FILE and ARENA_PROVIDED_BASELINE:
        return None
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
    if filename == KERNEL_FILE:
        _require_candidate_outputs(mod)
    return mod


_KERNEL_DIR = _resolve_kernel_dir()

# bf16 GEMM shapes: (M, N, K) plus per-case FlyDSL tiling that satisfies the
# kernel's tile constraints.
SHAPES = [
    {"name": "untuned_m64_n256_k5120", "m": 64, "n": 256, "k": 5120},
    {"name": "untuned_m256_n256_k5120", "m": 256, "n": 256, "k": 5120},
    {"name": "untuned_m512_n256_k5120", "m": 512, "n": 256, "k": 5120},
    {"name": "dsv3_m128_n3072_k1536", "m": 128, "n": 3072, "k": 1536},
    {"name": "dsv3_m64_n2112_k7168_tn64", "m": 64, "n": 2112, "k": 7168, "tile_n": 64},
]

# bf16 GEMM element-wise tolerance.
ATOL, RTOL, PASS_PCT = 1e-2, 1e-2, 99.9
SEED = 20260401
TILING_KEYS = ("tile_m", "tile_n", "tile_k", "split_k", "block_m_warps", "block_n_warps")


def _make_inputs(m, n, k, device="cuda"):
    import torch

    gen = torch.Generator(device=device)
    gen.manual_seed(SEED)
    a = torch.rand((m, k), generator=gen, device=device, dtype=torch.bfloat16)
    b = torch.rand((n, k), generator=gen, device=device, dtype=torch.bfloat16)
    return a, b


def _checked_gemm_output(actual, expected):
    import torch
    require_tensor_contract(actual, expected)
    if not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
        raise AssertionError("Non-finite operator/reference output")
    return actual


def _gemm_reference(a, b):
    import torch
    return torch.matmul(a.float(), b.float().transpose(-1, -2)).to(a.dtype)


def _compare_gemm_output(actual, expected):
    import torch
    _checked_gemm_output(actual, expected)
    close = torch.isclose(expected.float(), actual.float(), atol=ATOL, rtol=RTOL)
    pct = close.float().mean().item() * 100.0
    if pct < PASS_PCT:
        raise AssertionError(f"Numerical mismatch: {pct}% close, required={PASS_PCT}%")


def _timed_gemm_case(candidate, a, b, *, case_index, warmup, iters):
    """Time the same full call for both roles and check all reported outputs."""
    import torch

    stream = MeasuredInputStream(a, b, seed=SEED, case_index=case_index, samples=iters)
    reason = "capture_unsafe_hipblaslt_reference"
    try:
        candidate()
        torch.cuda.synchronize()
        require_unchanged((a, b), (stream.original_a, stream.original_b))
        for _ in range(warmup):
            candidate()
            torch.cuda.synchronize()
            require_unchanged((a, b), (stream.original_a, stream.original_b))

        timed = TimedRun()
        timed.after_sample = stream.observe
        kernel_ms, kernel_meta = benchmark_cuda_graph_or_events(
            candidate, warmup=0, repetition=iters, use_cuda_graph=False,
            fallback_reason=reason, prepare_fn=stream.prepare, timed_run=timed,
        )
        if kernel_meta.get("benchmark_samples") != iters:
            raise AssertionError("Reported sample count differs from the declared count")
        last_expected = stream.validate(lambda: _gemm_reference(a, b), _compare_gemm_output)
        last_inputs = (a.clone(), b.clone())
        kernel_meta.update(verify_timed_run(
            timed, inputs=(a, b), originals=last_inputs, expected=last_expected,
            perturb=lambda: (a.mul_(0.5), b.mul_(0.5)),
            reference=lambda: _gemm_reference(a, b), compare=_compare_gemm_output,
        ))
        kernel_meta["validated_sample_count"] = len(stream.outputs)

        # The diagnostic PyTorch side receives the identical prepared stream.
        stream.restore()
        reference_stream = MeasuredInputStream(a, b, seed=SEED,
                                               case_index=case_index, samples=iters)
        try:
            ref_ms, ref_meta = benchmark_cuda_graph_or_events(
                lambda: torch.mm(a, b.transpose(-1, -2)),
                warmup=0, repetition=iters, use_cuda_graph=False,
                fallback_reason=reason, prepare_fn=reference_stream.prepare,
            )
            if reference_stream.prepared != iters or ref_meta.get("benchmark_samples") != iters:
                raise AssertionError("Reference stream differs from candidate sample count")
        finally:
            reference_stream.restore()
        return kernel_ms, kernel_meta, ref_ms, ref_meta
    finally:
        stream.restore()


def _signed_candidate_control(candidate, model, shape, *, device="cuda"):
    """Exercise the candidate on signed operands at an existing supported shape."""
    import torch

    m, n, k = shape["m"], shape["n"], shape["k"]
    row_sign = torch.ones(m, device=device, dtype=torch.bfloat16)
    col_sign = torch.ones(n, device=device, dtype=torch.bfloat16)
    row_sign[1::2] = -1
    col_sign[1::2] = -1
    a = row_sign[:, None].expand(m, k).clone()
    b = col_sign[:, None].expand(n, k).clone()
    originals = (a.clone(), b.clone())
    exact = (row_sign.float()[:, None] * col_sign.float()[None, :] * k).to(a.dtype)
    reference = _gemm_reference(a, b)
    if not torch.equal(reference, exact) or not torch.equal(model(a, b), exact):
        raise AssertionError("Signed FP32 reference disagrees with independent integer dot products")
    tiling = {key: shape[key] for key in TILING_KEYS if key in shape}
    output = candidate(a, b, **tiling)
    if a.is_cuda:
        torch.cuda.synchronize()
    require_unchanged((a, b), originals)
    _compare_gemm_output(output, exact)
    _compare_gemm_output(output, reference)


def _mixed_sign_candidate_control(candidate, model, shape, *, device="cuda"):
    """Exact mixed-sign dots with cancellation at an existing supported shape."""
    import torch

    m, n, k = shape["m"], shape["n"], shape["k"]
    period = 16
    if k % period:
        raise AssertionError("Mixed-sign exact control requires a full period")
    rows = torch.arange(m, device=device)[:, None]
    cols = torch.arange(n, device=device)[:, None]
    offsets = torch.arange(period, device=device)[None, :]
    a_period = torch.where((rows + 3 * offsets) % 7 < 3, 1, -1).to(torch.int32)
    b_period = torch.where((3 * cols + 5 * offsets) % 11 < 5, 1, -1).to(torch.int32)
    a = a_period.to(torch.bfloat16).repeat(1, k // period)
    b = b_period.to(torch.bfloat16).repeat(1, k // period)
    originals = (a.clone(), b.clone())
    # Sum 16 exact integer products explicitly. This control shares neither
    # the candidate's GEMM implementation nor the FP32 reference reduction.
    exact_period = torch.zeros((m, n), device=device, dtype=torch.int32)
    for offset in range(period):
        exact_period += a_period[:, offset, None] * b_period[None, :, offset]
    exact = (exact_period * (k // period)).to(torch.bfloat16)
    if not bool((exact < 0).any() and (exact > 0).any() and (exact == 0).any()):
        raise AssertionError("Mixed-sign control lacks cancellation or output sign diversity")
    reference = _gemm_reference(a, b)
    if not torch.equal(reference, exact) or not torch.equal(model(a, b), exact):
        raise AssertionError("Mixed-sign FP32 reference disagrees with exact integer dots")
    tiling = {key: shape[key] for key in TILING_KEYS if key in shape}
    output = candidate(a, b, **tiling)
    if a.is_cuda:
        torch.cuda.synchronize()
    require_unchanged((a, b), originals)
    _compare_gemm_output(output, exact)
    _compare_gemm_output(output, reference)


def run_correctness(verbose=True):
    import torch

    kmod = _load_module(_KERNEL_DIR, KERNEL_FILE, "flydsl_kernel")
    mmod = _load_module(_KERNEL_DIR, MODEL_FILE, "torch_model")
    if kmod is None or mmod is None:
        print("FAIL: cannot load kernel.py / model.py")
        return {"correct": False}

    init = mmod.get_init_inputs()
    model = mmod.Model().to("cuda").eval() if not init else mmod.Model(*init).to("cuda").eval()

    failures = []
    for shape in SHAPES:
        tiling = {k: shape[k] for k in TILING_KEYS if k in shape}
        try:
            a, b = _make_inputs(shape["m"], shape["n"], shape["k"])
            originals = (a.clone(), b.clone())
            with torch.no_grad():
                ref = model(a, b)
            out = kmod.flydsl_hgemm(a, b, **tiling)
            torch.cuda.synchronize()

            require_unchanged((a, b), originals)
            _checked_gemm_output(out, ref)
            ref_f, out_f = ref.float(), out.float()
            close = torch.isclose(ref_f, out_f, atol=ATOL, rtol=RTOL)
            pct = close.float().mean().item() * 100.0
            max_delta = (ref_f - out_f).abs().max().item()
            ok = pct >= PASS_PCT
            if verbose:
                print(
                    f"  {'PASS' if ok else 'FAIL'}: {shape['name']} "
                    f"({shape['m']}x{shape['n']}x{shape['k']}) "
                    f"{pct:.4f}% close, max_delta={max_delta:.4f}"
                )
            if not ok:
                failures.append(shape["name"])
        except Exception as e:  # noqa: BLE001
            failures.append(shape["name"])
            if verbose:
                print(f"  FAIL: {shape['name']} - {str(e)[:100]}")

    for shape in SHAPES:
        for name, control in (("signed_candidate_control", _signed_candidate_control),
                              ("mixed_sign_candidate_control", _mixed_sign_candidate_control)):
            label = f"{name}/{shape['name']}"
            try:
                with torch.no_grad():
                    control(kmod.flydsl_hgemm, model, shape)
                if verbose:
                    print(f"  PASS: {label}")
            except Exception as e:  # noqa: BLE001
                failures.append(label)
                if verbose:
                    print(f"  FAIL: {label} - {str(e)[:100]}")

    status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{len(SHAPES) * 3})"
    print(f"Status: {status}")
    print(f"correctness: {'pass' if not failures else 'fail'}")
    assert not failures, f"correctness FAILED for: {failures}"
    return True


def run_benchmark(warmup=10, iters=100, verbose=True):
    import torch

    kmod = _load_module(_KERNEL_DIR, KERNEL_FILE, "flydsl_kernel")
    if kmod is None:
        print("FAIL: cannot load kernel.py")
        return {"geomean_latency_ms": -1, "geomean_speedup": -1}

    latencies, speedups, report = [], [], []
    print(f"{'Config (M,N,K)':<28} {'Ref':>10} {'FlyDSL':>10} {'Speedup':>10}")
    print("-" * 62)
    for idx, shape in enumerate(SHAPES):
        m, n, k = shape["m"], shape["n"], shape["k"]
        tiling = {kk: shape[kk] for kk in TILING_KEYS if kk in shape}
        a, b = _make_inputs(m, n, k)

        kernel_ms, kernel_bench_meta, ref_ms, ref_bench_meta = _timed_gemm_case(
            lambda: kmod.flydsl_hgemm(a, b, **tiling), a, b,
            case_index=idx, warmup=warmup, iters=iters,
        )

        methods_match = kernel_bench_meta["benchmark_method"] == ref_bench_meta["benchmark_method"]
        speedup = (
            ref_ms / kernel_ms if methods_match and kernel_ms > 0 else None
        )
        speedup_display = (
            format(speedup, ">8.2f") + "x"
            if speedup is not None
            else f"{'N/A':>9}"
        )
        latencies.append(kernel_ms)
        if speedup is not None:
            speedups.append(speedup)
        tflops = 2.0 * m * n * k / (kernel_ms * 1e-3) / 1e12
        report.append({
            "test_case_id": f"test_case_{idx}",
            "execution_time_ms": kernel_ms,
            **kernel_bench_meta,
            "reference_benchmark_method": ref_bench_meta["benchmark_method"],
            "benchmark_method_consistent": kernel_bench_meta["benchmark_method"] == ref_bench_meta["benchmark_method"],
            "shape": [m, n, k],
            "params": {"M": m, "N": n, "K": k, "dtype": "bf16"},
            "tflops": tflops,
        })
        if verbose:
            print(f"(M={m:>4}, N={n:>5}, K={k:>5}) {ref_ms:>8.4f}ms {kernel_ms:>8.4f}ms {speedup_display}")
        del a, b
        torch.cuda.empty_cache()

    geomean_latency = math.exp(sum(math.log(x) for x in latencies) / len(latencies))
    geomean_speedup = math.exp(sum(math.log(x) for x in speedups) / len(speedups)) if speedups else None
    geomean_speedup_display = (
        format(geomean_speedup, ".2f") + "x"
        if geomean_speedup is not None
        else "N/A"
    )

    build_dir = Path(_KERNEL_DIR) / "build"
    build_dir.mkdir(exist_ok=True)
    with open(build_dir / "performance_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("-" * 62)
    print(f"Geometric mean latency: {geomean_latency:.4f} ms")
    print(f"Geometric mean speedup: {geomean_speedup_display}")
    return {"geomean_latency_ms": geomean_latency, "geomean_speedup": geomean_speedup}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="torch2flydsl hgemm harness")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    print("=" * 62)
    print("torch2flydsl HGEMM")
    print("=" * 62)

    if args.correctness:
        try:
            run_correctness()
        except AssertionError as exc:
            print(f"ASSERTION: {exc}")
            sys.exit(1)
        sys.exit(0)
    else:
        run_benchmark(warmup=args.warmup, iters=args.iterations)


def _require_candidate_outputs(mod):
    import functools
    for name in tuple(vars(mod)):
        target = getattr(mod, name)
        if name.startswith("flydsl_") and callable(target):
            @functools.wraps(target)
            def checked(*args, __target=target, **kwargs):
                try:
                    result = __target(*args, **kwargs)
                except NotImplementedError as exc:
                    raise RuntimeError("Executed candidate is unimplemented; no baseline fallback") from exc
                if result is None:
                    raise RuntimeError("Candidate operator returned None; output is required")
                return result
            setattr(mod, name, checked)


# V2 direct timing evidence from this invocation.
def arena_benchmark(warmup=10, iters=100, verbose=True):
    import torch

    kmod = _load_module(_KERNEL_DIR, KERNEL_FILE, "flydsl_kernel")
    if kmod is None:
        print("FAIL: cannot load kernel.py")
        return {"geomean_latency_ms": -1, "geomean_speedup": -1}

    latencies, speedups, report = [], [], []
    print(f"{'Config (M,N,K)':<28} {'Ref':>10} {'FlyDSL':>10} {'Speedup':>10}")
    print("-" * 62)
    for idx, shape in enumerate(SHAPES):
        m, n, k = shape["m"], shape["n"], shape["k"]
        tiling = {kk: shape[kk] for kk in TILING_KEYS if kk in shape}
        a, b = _make_inputs(m, n, k)

        kernel_ms, kernel_bench_meta, ref_ms, ref_bench_meta = _timed_gemm_case(
            lambda: kmod.flydsl_hgemm(a, b, **tiling), a, b,
            case_index=idx, warmup=warmup, iters=iters,
        )

        methods_match = kernel_bench_meta["benchmark_method"] == ref_bench_meta["benchmark_method"]
        speedup = (
            ref_ms / kernel_ms if methods_match and kernel_ms > 0 else None
        )
        speedup_display = (
            format(speedup, ">8.2f") + "x"
            if speedup is not None
            else f"{'N/A':>9}"
        )
        latencies.append(kernel_ms)
        if speedup is not None:
            speedups.append(speedup)
        tflops = 2.0 * m * n * k / (kernel_ms * 1e-3) / 1e12
        report.append({
            "test_case_id": f"test_case_{idx}",
            "execution_time_ms": kernel_ms,
            **kernel_bench_meta,
            "reference_benchmark_method": ref_bench_meta["benchmark_method"],
            "benchmark_method_consistent": kernel_bench_meta["benchmark_method"] == ref_bench_meta["benchmark_method"],
            "shape": [m, n, k],
            "params": {"M": m, "N": n, "K": k, "dtype": "bf16"},
            "tflops": tflops,
        })
        if verbose:
            print(f"(M={m:>4}, N={n:>5}, K={k:>5}) {ref_ms:>8.4f}ms {kernel_ms:>8.4f}ms {speedup_display}")
        del a, b
        torch.cuda.empty_cache()

    geomean_latency = math.exp(sum(math.log(x) for x in latencies) / len(latencies))
    geomean_speedup = math.exp(sum(math.log(x) for x in speedups) / len(speedups)) if speedups else None
    geomean_speedup_display = (
        format(geomean_speedup, ".2f") + "x"
        if geomean_speedup is not None
        else "N/A"
    )

    build_dir = Path(_KERNEL_DIR) / "build"
    build_dir.mkdir(exist_ok=True)
    with open(build_dir / "performance_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("-" * 62)
    print(f"Geometric mean latency: {geomean_latency:.4f} ms")
    print(f"Geometric mean speedup: {geomean_speedup_display}")
    return report
