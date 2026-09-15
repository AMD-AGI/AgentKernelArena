#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Test harness for the torch2flydsl a8w8 b-preshuffle GEMM task.

Builds the PyTorch reference from model.py (`Model`/`get_inputs`/
`get_init_inputs`) and runs the inline FlyDSL kernel from kernel.py over a set
of real (M, N, K) GEMM shapes, comparing for correctness and benchmarking
against a torch baseline.

The activation is per-token fp8-quantized and the weight is per-channel
fp8-quantized (via model.py's `pertoken_quant`); the weight is then pre-shuffled
into the kernel's (16, 16) layout with `preshuffle_weight_a8` before launch, so
the kernel is validated apples-to-apple against the dequant-matmul reference.

Modes:
  --correctness     compare FlyDSL output to the PyTorch Model reference
  --full-benchmark  time FlyDSL vs a torch baseline, write performance report
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from _aka_benchmark import TimedRun, benchmark_cuda_graph_or_events
from scripts.replay_checks import require_tensor_contract, require_unchanged, verify_timed_run

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

# Real a8w8 b-preshuffle (M, N, K) shapes from
# configs/a8w8_bpreshuffle_tuned_gemm.csv / model_configs (gfx950 fp8), each
# with a per-case FlyDSL tiling that the preshuffle kernel supports
# (tile_n | N, tile_k | K, tile_k % 64 == 0, tile_m*tile_k % 4096 == 0).
SHAPES = [
    {"name": "skinny_m16_n5120_k1280", "m": 16, "n": 5120, "k": 1280,
     "tile_m": 16, "tile_n": 64, "tile_k": 256},
    {"name": "m64_n5120_k1280", "m": 64, "n": 5120, "k": 1280,
     "tile_m": 32, "tile_n": 64, "tile_k": 256},
    {"name": "m512_n5120_k1280", "m": 512, "n": 5120, "k": 1280,
     "tile_m": 128, "tile_n": 128, "tile_k": 256},
    {"name": "m1024_n8192_k1024", "m": 1024, "n": 8192, "k": 1024,
     "tile_m": 128, "tile_n": 128, "tile_k": 128},
    {"name": "m2048_n5120_k1280", "m": 2048, "n": 5120, "k": 1280,
     "tile_m": 128, "tile_n": 128, "tile_k": 256},
]

TILING_KEYS = ("tile_m", "tile_n", "tile_k")

# Normalized worst-element gate for a quantized bf16 GEMM (NEVER loosen):
# max|ref - out| / max|ref| <= NORM_TOL. ATOL/RTOL/PASS_PCT report the
# element-wise close fraction for context only.
NORM_TOL = 1e-2
ATOL, RTOL, PASS_PCT = 1e-2, 1e-2, 99.9
SEED = 20260401


def _retry(fn, tries=5, what="kernel"):
    """Call `fn`, backing off on transient OOM / HIP errors (shared GPU)."""
    delay = 0.5
    last = None
    for attempt in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            transient = ("out of memory" in msg) or ("hip" in msg) or ("oom" in msg)
            last = exc
            if not transient or attempt == tries - 1:
                raise
            import torch

            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            time.sleep(delay)
            delay *= 2
    raise last


def _make_inputs(m, n, k, device="cuda"):
    import torch

    gen = torch.Generator(device=device)
    gen.manual_seed(SEED)
    x = torch.randn((m, k), generator=gen, device=device, dtype=torch.bfloat16)
    weight = torch.randn((n, k), generator=gen, device=device, dtype=torch.bfloat16)
    return x, weight


def _checked_preshuffle(kmod, wq):
    """Check the original (16,16) packed layout, outside operator timing."""
    from scripts.candidate_checks import candidate_preparation_only
    import torch

    original = wq.clone()
    n, k = wq.shape
    expected = wq.view(n // 16, 16, k // 32, 2, 16).permute(0, 2, 3, 1, 4).contiguous().view(n, k)
    with candidate_preparation_only():
        actual = kmod.preshuffle_weight_a8(wq)
    require_unchanged((wq,), (original,))
    require_tensor_contract(actual, expected)
    if not torch.equal(actual.contiguous().view(torch.uint8), expected.view(torch.uint8)):
        raise AssertionError("Weight preshuffle must preserve the exact (16,16) byte permutation")
    return actual


def _compare_preshuffle_output(actual, expected):
    import torch

    require_tensor_contract(actual, expected)
    if not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
        raise AssertionError("Non-finite GEMM output/reference")
    delta = (actual.float() - expected.float()).abs().max().item()
    denom = expected.float().abs().max().item()
    norm = delta / denom if denom > 0 else delta
    if norm > NORM_TOL:
        raise AssertionError(f"Numerical mismatch: normalized_max_error={norm}, tolerance={NORM_TOL}")


def _quantized_dense_reference(xq, wq, x_scale, w_scale):
    """Original model's quantized FP32 accumulation, before BF16 output cast."""
    import torch

    acc = torch.matmul(xq.float(), wq.float().transpose(-1, -2))
    return (acc * x_scale * w_scale.transpose(0, 1)).to(torch.bfloat16)


def _perturb_preshuffle_scales(x_scale, w_scale):
    # Positive finite scales stay in-domain; change the actual measured inputs
    # without changing the timed case, quantization or packed-weight preparation.
    x_scale.mul_(0.5)
    w_scale.mul_(0.5)


def run_correctness(verbose=True):
    import torch

    kmod = _load_module(_KERNEL_DIR, KERNEL_FILE, "flydsl_kernel")
    mmod = _load_module(_KERNEL_DIR, MODEL_FILE, "torch_model")
    if kmod is None or mmod is None:
        print("FAIL: cannot load kernel.py / model.py")
        print("Status: FAILED (load)")
        print("correctness: fail")
        raise AssertionError("cannot load kernel.py / model.py")

    init = mmod.get_init_inputs()
    model = (mmod.Model().to("cuda").eval() if not init
             else mmod.Model(*init).to("cuda").eval())

    failures = []
    for shape in SHAPES:
        tiling = {kk: shape[kk] for kk in TILING_KEYS if kk in shape}
        try:
            x, weight = _make_inputs(shape["m"], shape["n"], shape["k"])
            with torch.no_grad():
                ref = model(x, weight)

            xq, x_scale = mmod.pertoken_quant(x)
            wq, w_scale = mmod.pertoken_quant(weight)
            wq_shuf = _checked_preshuffle(kmod, wq)
            protected_inputs = (x, weight, xq, wq, wq_shuf, x_scale, w_scale)
            originals = tuple(v.clone() for v in protected_inputs)
            out = _retry(
                lambda: kmod.flydsl_gemm_a8w8_bpreshuffle(
                    xq, wq_shuf, x_scale, w_scale, **tiling
                ),
                what=shape["name"],
            )
            torch.cuda.synchronize()

            require_unchanged(protected_inputs, originals)
            _compare_preshuffle_output(out, ref)
            ref_f, out_f = ref.float(), out.float()
            denom = ref_f.abs().max().item()
            max_delta = (ref_f - out_f).abs().max().item()
            norm = max_delta / denom if denom > 0 else max_delta
            close = torch.isclose(ref_f, out_f, atol=ATOL, rtol=RTOL)
            pct = close.float().mean().item() * 100.0
            ok = norm <= NORM_TOL
            if verbose:
                print(
                    f"  {'PASS' if ok else 'FAIL'}: {shape['name']} "
                    f"({shape['m']}x{shape['n']}x{shape['k']}) "
                    f"norm={norm:.2e} (tol={NORM_TOL:.0e}) "
                    f"max|d|={max_delta:.4f} max|ref|={denom:.4f} "
                    f"{pct:.3f}% close"
                )
            if not ok:
                failures.append(shape["name"])
        except Exception as e:  # noqa: BLE001
            failures.append(shape["name"])
            if verbose:
                print(f"  FAIL: {shape['name']} - {str(e)[:160]}")

    status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{len(SHAPES)})"
    print(f"Status: {status}")
    print(f"correctness: {'pass' if not failures else 'fail'}")
    assert not failures, f"correctness FAILED for: {failures}"
    return True


def run_benchmark(warmup=10, iters=100, verbose=True):
    import torch

    kmod = _load_module(_KERNEL_DIR, KERNEL_FILE, "flydsl_kernel")
    mmod = _load_module(_KERNEL_DIR, MODEL_FILE, "torch_model")
    if kmod is None or mmod is None:
        print("FAIL: cannot load kernel.py / model.py")
        return {"geomean_latency_ms": -1, "geomean_speedup": -1}

    latencies, speedups, report = [], [], []
    print(f"{'Config (M,N,K)':<28} {'Ref':>10} {'FlyDSL':>10} {'Speedup':>10}")
    print("-" * 62)
    for idx, shape in enumerate(SHAPES):
        m, n, k = shape["m"], shape["n"], shape["k"]
        tiling = {kk: shape[kk] for kk in TILING_KEYS if kk in shape}
        x, weight = _make_inputs(m, n, k)
        xq, x_scale = mmod.pertoken_quant(x)
        wq, w_scale = mmod.pertoken_quant(weight)
        wq_shuf = _checked_preshuffle(kmod, wq)

        protected_inputs = (x, weight, xq, wq, wq_shuf, x_scale, w_scale)
        originals = tuple(v.clone() for v in protected_inputs)
        expected = _quantized_dense_reference(xq, wq, x_scale, w_scale)

        def _call():
            return kmod.flydsl_gemm_a8w8_bpreshuffle(
                xq, wq_shuf, x_scale, w_scale, **tiling
            )

        _retry(_call, what=shape["name"])
        torch.cuda.synchronize()
        for _ in range(warmup):
            _call()
        torch.cuda.synchronize()

        # The PyTorch reference dispatches hipBLASLt, which rejects stream
        # capture in this image. Predetermine an Event-only policy for both
        # sides so the candidate cannot select a different timing method.
        event_reason = "capture_unsafe_hipblaslt_reference"
        timed = TimedRun()
        kernel_ms, kernel_bench_meta = benchmark_cuda_graph_or_events(
            _call,
            warmup=0,
            repetition=iters,
            use_cuda_graph=False,
            fallback_reason=event_reason,
            timed_run=timed,
        )

        kernel_bench_meta.update(verify_timed_run(
            timed, inputs=protected_inputs, originals=originals, expected=expected,
            perturb=lambda: _perturb_preshuffle_scales(x_scale, w_scale),
            reference=lambda: _quantized_dense_reference(xq, wq, x_scale, w_scale),
            compare=_compare_preshuffle_output,
        ))

        ref_ms, ref_bench_meta = benchmark_cuda_graph_or_events(
            lambda: torch.matmul(x, weight.transpose(-1, -2)),
            warmup=0,
            repetition=iters,
            use_cuda_graph=False,
            fallback_reason=event_reason,
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
            "params": {"M": m, "N": n, "K": k, "dtype": "fp8_e4m3", "out": "bf16"},
            "tflops": tflops,
        })
        if verbose:
            print(f"(M={m:>5}, N={n:>5}, K={k:>5}) "
                  f"{ref_ms:>8.4f}ms {kernel_ms:>8.4f}ms {speedup_display}")
        del x, weight, xq, wq, wq_shuf
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
    parser = argparse.ArgumentParser(description="torch2flydsl a8w8 bpreshuffle GEMM harness")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    print("=" * 62)
    print("torch2flydsl GEMM a8w8 b-preshuffle")
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
    mmod = _load_module(_KERNEL_DIR, MODEL_FILE, "torch_model")
    if kmod is None or mmod is None:
        print("FAIL: cannot load kernel.py / model.py")
        return {"geomean_latency_ms": -1, "geomean_speedup": -1}

    latencies, speedups, report = [], [], []
    print(f"{'Config (M,N,K)':<28} {'Ref':>10} {'FlyDSL':>10} {'Speedup':>10}")
    print("-" * 62)
    for idx, shape in enumerate(SHAPES):
        m, n, k = shape["m"], shape["n"], shape["k"]
        tiling = {kk: shape[kk] for kk in TILING_KEYS if kk in shape}
        x, weight = _make_inputs(m, n, k)
        xq, x_scale = mmod.pertoken_quant(x)
        wq, w_scale = mmod.pertoken_quant(weight)
        wq_shuf = _checked_preshuffle(kmod, wq)

        protected_inputs = (x, weight, xq, wq, wq_shuf, x_scale, w_scale)
        originals = tuple(v.clone() for v in protected_inputs)
        expected = _quantized_dense_reference(xq, wq, x_scale, w_scale)

        def _call():
            return kmod.flydsl_gemm_a8w8_bpreshuffle(
                xq, wq_shuf, x_scale, w_scale, **tiling
            )

        _retry(_call, what=shape["name"])
        torch.cuda.synchronize()
        for _ in range(warmup):
            _call()
        torch.cuda.synchronize()

        # The PyTorch reference dispatches hipBLASLt, which rejects stream
        # capture in this image. Predetermine an Event-only policy for both
        # sides so the candidate cannot select a different timing method.
        event_reason = "capture_unsafe_hipblaslt_reference"
        timed = TimedRun()
        kernel_ms, kernel_bench_meta = benchmark_cuda_graph_or_events(
            _call,
            warmup=0,
            repetition=iters,
            use_cuda_graph=False,
            fallback_reason=event_reason,
            timed_run=timed,
        )

        kernel_bench_meta.update(verify_timed_run(
            timed, inputs=protected_inputs, originals=originals, expected=expected,
            perturb=lambda: _perturb_preshuffle_scales(x_scale, w_scale),
            reference=lambda: _quantized_dense_reference(xq, wq, x_scale, w_scale),
            compare=_compare_preshuffle_output,
        ))

        ref_ms, ref_bench_meta = benchmark_cuda_graph_or_events(
            lambda: torch.matmul(x, weight.transpose(-1, -2)),
            warmup=0,
            repetition=iters,
            use_cuda_graph=False,
            fallback_reason=event_reason,
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
            "params": {"M": m, "N": n, "K": k, "dtype": "fp8_e4m3", "out": "bf16"},
            "tflops": tflops,
        })
        if verbose:
            print(f"(M={m:>5}, N={n:>5}, K={k:>5}) "
                  f"{ref_ms:>8.4f}ms {kernel_ms:>8.4f}ms {speedup_display}")
        del x, weight, xq, wq, wq_shuf
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
