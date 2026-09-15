#!/usr/bin/env python3
# GEAK materialized harness bootstrap
import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from _aka_benchmark import benchmark_cuda_graph_or_events_samples
from _timed_contract import checked_call, checked_benchmark
import _contract_oracles as oracle


def benchmark_cuda_graph_or_events(*args, **kwargs):
    samples, metadata = benchmark_cuda_graph_or_events_samples(*args, **kwargs)
    values = sorted(samples)
    midpoint = len(values) // 2
    median_ms = (
        values[midpoint]
        if len(values) % 2
        else (values[midpoint - 1] + values[midpoint]) / 2.0
    )
    return median_ms, metadata

def _find_baseline_kernel_dir():
    """Arena's session owns the frozen baseline; external worktrees are not inputs."""
    return None

def _load_baseline_triton(*args, **kwargs):
    raise RuntimeError("External baseline loading is not part of the v2 task contract")

def _resolve_geak_kernel_dir():
    """Resolve only the local candidate (or the session's frozen task copy)."""
    return os.path.dirname(os.path.abspath(__file__))

def _ensure_geak_package(module_name):
    parts = module_name.split(".")
    for idx in range(1, len(parts)):
        prefix = ".".join(parts[:idx])
        if prefix in sys.modules:
            continue
        pkg = types.ModuleType(prefix)
        pkg.__path__ = []
        sys.modules[prefix] = pkg

def _ensure_geak_aiter_fp8_dtype(module):
    fp8_value = getattr(module, "fp8_dtype", None)
    if fp8_value is None:
        return
    aiter_mod = sys.modules.get("aiter")
    if aiter_mod is None:
        try:
            import aiter as aiter_mod
        except Exception:
            _ensure_geak_package("aiter")
            aiter_mod = sys.modules.get("aiter")
    if aiter_mod is None:
        return
    dtypes_obj = getattr(aiter_mod, "dtypes", None)
    if dtypes_obj is None:
        dtypes_obj = types.SimpleNamespace()
        setattr(aiter_mod, "dtypes", dtypes_obj)
    if getattr(dtypes_obj, "fp8", None) is None:
        setattr(dtypes_obj, "fp8", fp8_value)

def _register_geak_aliases(kernel_dir):
    aliases = ['fused_rms_fp8', 'aiter.ops.triton.fused_fp8_quant']
    entry_file = os.path.join(kernel_dir, "kernel.py")
    if not os.path.isfile(entry_file):
        return
    for alias in aliases:
        if alias in sys.modules:
            existing = getattr(sys.modules[alias], "__file__", None)
            if existing is None or Path(existing).resolve() != Path(entry_file).resolve():
                raise RuntimeError("Candidate module alias resolved outside the task workspace")
            continue
        _ensure_geak_package(alias)
        spec = importlib.util.spec_from_file_location(alias, entry_file)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        spec.loader.exec_module(module)
        _ensure_geak_aiter_fp8_dtype(module)

_KERNEL_DIR = _resolve_geak_kernel_dir()
if _KERNEL_DIR and _KERNEL_DIR not in sys.path:
    sys.path.insert(0, _KERNEL_DIR)
_register_geak_aliases(_KERNEL_DIR)

"""
Test harness for fused_fp8_quant kernel (aiter reference).

Modes: --correctness, --profile, --benchmark, --full-benchmark

The alias bootstrap binds all public wrappers to this task's local kernel.py.
Protected PyTorch oracles cover all four editable entrypoints.
"""
import argparse
import math
import torch
import torch.nn.functional as F

from aiter.ops.triton.fused_fp8_quant import (
    fused_rms_fp8_group_quant, fused_flatten_fp8_group_quant,
    fused_reduce_act_mul_fp8_group_quant, fused_reduce_rms_fp8_group_quant,
)
import aiter

fp8_dtype = aiter.dtypes.fp8


# ============================================================================
# TEST CONFIGURATIONS
# ============================================================================

# (M, N1, N2) -- batch/tokens, hidden dimension 1, hidden dimension 2
ALL_SHAPES = [
    (1, 128, 128),
    (4, 128, 128),
    (1, 128, 4096),
    (8, 128, 128),
    (1, 128, 7168),
    (1, 4096, 4096),
    (1, 128, 8192),
    (1, 4096, 8192),
    (1, 7168, 7168),
    (1, 8192, 8192),
    (32, 128, 128),
    (4, 4096, 4096),
    (8, 4096, 4096),
    (16, 4096, 4096),
    (256, 128, 128),
    (32, 128, 7168),
    (1024, 128, 128),
    (256, 128, 7168),
    (256, 4096, 4096),
    (8192, 128, 128),
    (32, 7168, 7168),
    (256, 7168, 7168),
    (1024, 4096, 4096),
    (1024, 8192, 8192),
    (8192, 7168, 7168),
]

seen = set()
unique_shapes = []
for s in ALL_SHAPES:
    if s not in seen:
        seen.add(s)
        unique_shapes.append(s)
ALL_SHAPES = sorted(unique_shapes, key=lambda s: s[0] * (s[1] + s[2]))

# HARNESS_SHAPES: uniformly sample 25 shapes from ALL_SHAPES
_n_all = len(ALL_SHAPES)
if _n_all <= 25:
    HARNESS_SHAPES = ALL_SHAPES
else:
    _harness_indices = [int(round(i * (_n_all - 1) / 24)) for i in range(25)]
    HARNESS_SHAPES = [ALL_SHAPES[i] for i in _harness_indices]

# PROFILE_SHAPES: exactly 5 shapes evenly spaced
_profile_indices = [int(round(i * (_n_all - 1) / 4)) for i in range(5)]
PROFILE_SHAPES = [ALL_SHAPES[i] for i in _profile_indices]

# For backward compatibility
EVAL_CONFIGS = HARNESS_SHAPES
PROFILE_CONFIGS = PROFILE_SHAPES

RTOL, ATOL = 0.1, 0.1


# ============================================================================
# REFERENCE IMPLEMENTATIONS
# ============================================================================


def rmsnorm(input, weight, eps=1e-6):
    row_norm = input * input
    row_norm = torch.sum(row_norm, dim=-1)
    norm_factor = torch.rsqrt((row_norm / input.shape[1]) + eps)
    rms_norm = input * norm_factor[:, None] * weight[None, :]
    return rms_norm


def per_token_fp8_group_quant(x, dtype_quant, group_size=128):
    DTYPE_MAX = torch.finfo(dtype_quant).max
    M, N = x.shape
    if N % group_size > 0:
        num_pad = group_size - (N % group_size)
        x_reshape = F.pad(x, (0, num_pad, 0, 0), "constant", 0)
        x_reshape = x_reshape.reshape(
            M, (N + group_size - 1) // group_size, group_size
        ).to(torch.float32)
    else:
        x_reshape = x.reshape(M, N // group_size, group_size).to(torch.float32)
    x_max = torch.max(torch.abs(x_reshape), dim=-1, keepdim=True)[0]
    x_max = torch.where(x_max < 1e-10, 1e-10, x_max).to(torch.float32)
    x_scale = x_max / DTYPE_MAX
    scale_recip = 1.0 / x_scale
    x_quant = torch.clamp(x_reshape * scale_recip, -DTYPE_MAX, DTYPE_MAX).to(
        dtype_quant
    )
    x_quant = x_quant.reshape(M, (N + group_size - 1) // group_size * group_size)[:, :N]
    x_scale = x_scale.squeeze(-1)
    return x_quant, x_scale


def upcast(x, s, dtype, group_size=128):
    x_N = x.shape[1]
    x = x.reshape(-1, x_N // group_size, group_size).to(torch.float32) * s.reshape(
        -1, s.shape[1], 1
    )
    x = x.reshape(-1, x_N)
    return x.to(dtype=dtype)


def run_torch_rms_fp8_group_quant(
    x1, w1, eps1, x2, w2, eps2, res1, dtype_quant, group_size
):
    s = x1 + res1
    y1 = rmsnorm(s, w1, eps1)
    y2 = rmsnorm(x2, w2, eps2)
    y1_q, y1_s = per_token_fp8_group_quant(y1, dtype_quant, group_size)
    return (y1_q, y1_s), y1.to(x1.dtype), y2.to(x1.dtype), s.to(x1.dtype)


# ============================================================================
# INPUT GENERATION
# ============================================================================


def generate_inputs(M, N1, N2, dtype=torch.bfloat16):
    """Generate inputs on CPU then move to GPU."""
    torch.manual_seed(42)
    x1 = (torch.randn((M, N1), dtype=dtype, device="cpu") / 10).to("cuda")
    x2 = (torch.randn((M, N2), dtype=dtype, device="cpu") / 10).to("cuda")
    w1 = torch.ones((N1,), dtype=torch.float32, device="cpu").to("cuda")
    w2 = torch.ones((N2,), dtype=torch.float32, device="cpu").to("cuda")
    res1 = (torch.randn((M, N1), dtype=dtype, device="cpu") / 10).to("cuda")
    return x1, w1, x2, w2, res1


# ============================================================================
# TEST HARNESS
# ============================================================================


def _main_contract():
    precise = {}
    def reference(saved):
        # Keep the original rounded-input reference and all of its gates.
        legacy = run_torch_rms_fp8_group_quant(
            saved['x1'], saved['w1'], 1e-6, saved['x2'], saved['w2'], 1e-6,
            saved['res1'], fp8_dtype, 128)
        precise['outputs'] = oracle.rms(saved, fp8_dtype)
        return legacy
    def check(actual, legacy):
        for output, expected in zip(actual[1:], legacy[1:]):
            torch.testing.assert_close(output, expected, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(upcast(*actual[0], dtype=torch.float32),
                                   upcast(*legacy[0], dtype=torch.float32), atol=ATOL, rtol=RTOL)
        # Independently check raw quantized values and scales using FP32
        # accumulation, rather than the legacy early BF16 residual rounding.
        oracle.check_quant(actual[0], precise['outputs'][0], atol=ATOL, rtol=RTOL)
    return reference, check


CONTROL_CASES = [
    {'test_case_id': 'control-rms-optional', 'params': {'variant': 'rms', 'split': 1}},
    {'test_case_id': 'control-flatten', 'params': {'variant': 'flatten', 'split': 1}},
    {'test_case_id': 'control-activation-2d', 'params': {'variant': 'activation', 'split': 1}},
    {'test_case_id': 'control-activation-split3', 'params': {'variant': 'activation', 'split': 3}},
    {'test_case_id': 'control-reduce-rms-2d', 'params': {'variant': 'reduce_rms', 'split': 1}},
    {'test_case_id': 'control-reduce-rms-split3', 'params': {'variant': 'reduce_rms', 'split': 3}},
    {'test_case_id': 'control-reduce-rms-split4', 'params': {'variant': 'reduce_rms', 'split': 4}},
]


def _control_tensor(shape, offset=0):
    # Deterministic, signed, nonuniform data; no changes to the scored RNG stream.
    values = torch.arange(math.prod(shape), device='cuda', dtype=torch.float32)
    return (torch.sin(values * .13 + offset) * .4).reshape(shape).to(torch.bfloat16)


def run_contract_controls():
    for case in CONTROL_CASES:
        variant, split = case['params']['variant'], case['params']['split']
        if variant == 'flatten':
            live = {'x': _control_tensor((3, 2, 128))}
            invoke = lambda: fused_flatten_fp8_group_quant(live['x'], 128, fp8_dtype)
            reference = lambda saved: oracle.quantize(saved['x'].reshape(3, 256), fp8_dtype)
            check = oracle.check_quant
        elif variant == 'activation':
            shape = (3, 512) if split == 1 else (split, 3, 512)
            live = {'x': _control_tensor(shape)}
            if split > 1:
                live['x2'] = _control_tensor((split, 3, 64), .7)
            invoke = lambda: fused_reduce_act_mul_fp8_group_quant(
                live['x'], activation='silu', x2=live.get('x2'), group_size=128,
                dtype_quant=fp8_dtype, dtype=torch.bfloat16)
            reference = lambda saved: oracle.activation_mul(saved, fp8_dtype)
            check = oracle.check_fused
        else:
            shape = (3, 256) if split == 1 else (split, 3, 256)
            live = {'x1': _control_tensor(shape),
                    'w1': torch.linspace(-1.5, 2, 256, device='cuda')}
            if variant == 'rms':
                invoke = lambda: fused_rms_fp8_group_quant(
                    live['x1'], live['w1'], 1e-6, group_size=128, dtype_quant=fp8_dtype)
                reference = lambda saved: oracle.rms(saved, fp8_dtype, show=False)
            else:
                live['res1'] = _control_tensor((3, 256), .4)
                live['x2'] = _control_tensor((3, 128) if split == 1 else (split, 3, 128), .8)
                live['w2'] = torch.linspace(.2, 1.4, 128, device='cuda')
                if split > 1:
                    live['x3'] = _control_tensor((split, 3, 64), 1.2)
                invoke = lambda: fused_reduce_rms_fp8_group_quant(
                    live['x1'], live['w1'], 1e-6,
                    inp2=live['x2'], inp2_weight=live['w2'], inp2_epsilon=1e-6,
                    inp3=live.get('x3'), res1=live['res1'], group_size=128,
                    dtype_quant=fp8_dtype, output_unquantized_inp1=True)
                reference = lambda saved: oracle.rms(saved, fp8_dtype, reduce=True)
            check = oracle.check_fused
        checked_call(invoke, inputs=live, reference=reference, check=check)
        print(case['test_case_id'], 'PASS')


def run_correctness(shapes=None, verbose=True):
    if shapes is None:
        shapes = HARNESS_SHAPES
    if verbose:
        print(f"Running correctness on {len(shapes)} shapes...")

    group_size = 128
    dtype = torch.bfloat16
    results, failures = [], []

    for i, (M, N1, N2) in enumerate(shapes):
        try:
            x1, w1, x2, w2, res1 = generate_inputs(M, N1, N2, dtype)

            readonly = dict(x1=x1, w1=w1, x2=x2, w2=w2, res1=res1)
            reference, check = _main_contract()
            checked_call(
                lambda: fused_rms_fp8_group_quant(
                    x1, w1, 1e-6, inp2=x2, inp2_weight=w2, inp2_epsilon=1e-6,
                    group_size=group_size, dtype_quant=fp8_dtype, res1=res1,
                    output_unquantized_inp1=True),
                inputs=readonly, reference=reference, check=check,
            )

            results.append({"config": (M, N1, N2), "correct": True})
            if verbose:
                print(f"  PASS: ({M}, {N1}, {N2})")

            del x1, x2, w1, w2, res1
            torch.cuda.empty_cache()
        except Exception as e:
            failures.append({"config": (M, N1, N2), "error": str(e)})
            if verbose:
                print(f"  FAIL: ({M}, {N1}, {N2}) - {str(e)[:50]}")

    if verbose:
        print("-" * 62)
        print(
            f"{'Status:':<22} {'ALL PASS' if not failures else f'FAILED ({len(failures)}/{len(shapes)})'}"
        )

    return {
        "correct": len(failures) == 0,
        "num_correct": len(results),
        "num_failed": len(failures),
        "failures": failures,
        "results": results,
    }


def run_profile(shapes=None, warmup=50, iters=200, verbose=True):
    if shapes is None:
        shapes = PROFILE_SHAPES
    group_size = 128
    dtype = torch.bfloat16

    if verbose:
        print(f"Profile: {len(shapes)} config(s), {warmup} warmup, {iters} iter(s)")

    for M, N1, N2 in shapes:
        x1, w1, x2, w2, res1 = generate_inputs(M, N1, N2, dtype)
        for _ in range(warmup):
            _ = fused_rms_fp8_group_quant(
                x1, w1, 1e-6,
                inp2=x2, inp2_weight=w2, inp2_epsilon=1e-6,
                group_size=group_size,
                dtype_quant=fp8_dtype,
                res1=res1,
                output_unquantized_inp1=True,
            )
        torch.cuda.synchronize()
        for _ in range(iters):
            _ = fused_rms_fp8_group_quant(
                x1, w1, 1e-6,
                inp2=x2, inp2_weight=w2, inp2_epsilon=1e-6,
                group_size=group_size,
                dtype_quant=fp8_dtype,
                res1=res1,
                output_unquantized_inp1=True,
            )
        torch.cuda.synchronize()
        if verbose:
            print(f"  ({M},{N1},{N2}) done")
        del x1, x2, w1, w2, res1
        torch.cuda.empty_cache()


def run_benchmark(shapes=None, warmup=50, iters=200, verbose=True):
    """Benchmark kernel vs reference. Uses baseline Triton when available; else PyTorch."""
    if shapes is None:
        shapes = HARNESS_SHAPES
    group_size = 128
    dtype = torch.bfloat16
    baseline_dir = _find_baseline_kernel_dir()
    kernel_dir = _resolve_geak_kernel_dir()
    baseline_fn = None
    if baseline_dir and baseline_dir != kernel_dir:
        baseline_fn = _load_baseline_triton(baseline_dir, "baseline_fused_rms_fp8", "fused_rms_fp8_group_quant")
    ref_label = "baseline_triton" if baseline_fn else "PyTorch"

    latencies = []
    speedups = []
    benchmark_methods = []
    report_cases = []

    print(f"Running benchmark on {len(shapes)} shapes, {warmup} warmup, {iters} iterations each...")
    print(f"  Comparing kernel vs {ref_label}")
    print(f"{'Config (M,N1,N2)':<22} {'Ref':>10} {'Triton':>10} {'Speedup':>10}")
    print("-" * 62)

    for M, N1, N2 in shapes:
        x1, w1, x2, w2, res1 = generate_inputs(M, N1, N2, dtype)

        def run_kernel():
            return fused_rms_fp8_group_quant(
                x1, w1, 1e-6,
                inp2=x2, inp2_weight=w2, inp2_epsilon=1e-6,
                group_size=group_size,
                dtype_quant=fp8_dtype,
                res1=res1,
                output_unquantized_inp1=True,
            )

        reference, check = _main_contract()
        triton_ms, triton_meta = checked_benchmark(
            benchmark_cuda_graph_or_events, run_kernel,
            inputs=dict(x1=x1, w1=w1, x2=x2, w2=w2, res1=res1),
            reference=reference, check=check,
            perturb=lambda saved: {**saved, 'x1': -saved['x1'], 'res1': -saved['res1'], 'x2': -saved['x2']},
            warmup=warmup, repetition=iters,
        )

        def run_reference():
            if baseline_fn is not None:
                return baseline_fn(
                    x1, w1, 1e-6,
                    inp2=x2, inp2_weight=w2, inp2_epsilon=1e-6,
                    group_size=group_size,
                    dtype_quant=fp8_dtype,
                    res1=res1,
                    output_unquantized_inp1=True,
                )
            return run_torch_rms_fp8_group_quant(
                x1, w1, 1e-6, x2, w2, 1e-6, res1, fp8_dtype, group_size
            )

        ref_ms, ref_meta = benchmark_cuda_graph_or_events(
            run_reference, warmup=warmup, repetition=iters,
        )
        methods_match = triton_meta["benchmark_method"] == ref_meta["benchmark_method"]
        speedup = ref_ms / triton_ms if methods_match and triton_ms > 0 else None

        latencies.append(triton_ms)
        if speedup is not None:
            speedups.append(speedup)
        benchmark_methods.append(triton_meta["benchmark_method"])
        report_cases.append({
            "test_case_id": f"M={M} N1={N1} N2={N2}",
            "params": {"M": M, "N1": N1, "N2": N2},
            "execution_time_ms": triton_ms,
            **triton_meta,
        })

        marker = " *" if speedup is not None and speedup > 1.0 else ""
        if verbose:
            speedup_text = f"{speedup:.2f}x" if speedup is not None else "N/A"
            print(f"({M:>6}, {N1:>5}, {N2:>5}){' ':4} {ref_ms:>8.4f}ms {triton_ms:>8.4f}ms {speedup_text:>9s}{marker}", flush=True)

    report_path = Path("build/performance_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report_cases, indent=2))

    log_sum = sum(math.log(l) for l in latencies)
    geomean_latency = math.exp(log_sum / len(latencies))

    methods_consistent = len(speedups) == len(latencies)
    geomean_speedup = (
        math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        if methods_consistent else None
    )

    print("-" * 62)
    print(f"{'Geometric mean latency:':<22} {geomean_latency:.4f} ms")
    print(
        f"{'Geometric mean speedup:':<22} {geomean_speedup:.2f}x"
        if geomean_speedup is not None else
        f"{'Geometric mean speedup:':<22} N/A (timing methods differ)"
    )
    print(f"GEAK_RESULT_LATENCY_MS={geomean_latency:.4f}", flush=True)
    if geomean_speedup is not None:
        print(f"GEAK_RESULT_GEOMEAN_SPEEDUP={geomean_speedup:.4f}", flush=True)
    print(f"GEAK_BENCHMARK_METHOD_CONSISTENT={int(methods_consistent)}")
    print("GEAK_BENCHMARK_METHOD={}".format(
        benchmark_methods[0] if len(set(benchmark_methods)) == 1
        else "mixed:" + ",".join(sorted(set(benchmark_methods)))
    ))

    return {
        "geomean_latency_ms": geomean_latency,
        "geomean_speedup": geomean_speedup,
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fused RMS + FP8 Kernel Test Harness")
    parser.add_argument(
        "--correctness",
        action="store_true",
        help="Run correctness tests on benchmark shapes",
    )
    parser.add_argument(
        "--profile", action="store_true", help="Run minimal profiling workload"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark on HARNESS_SHAPES (25 uniformly sampled)",
    )
    parser.add_argument(
        "--full-benchmark",
        action="store_true",
        help="Run benchmark on ALL_SHAPES (complete set)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Number of warmup iterations (default: 50)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200")),
        help="Number of benchmark iterations (default: GEAK_BENCHMARK_ITERATIONS or 200)",
    )
    args = parser.parse_args()

    print("=" * 62)
    print("Fused RMSNorm + FP8 Quantization Kernel")
    print("=" * 62)

    if args.correctness:
        print("\n[Correctness Mode]")
        run_correctness(HARNESS_SHAPES)
    elif args.profile:
        print("\n[Profile Mode]")
        run_profile(PROFILE_SHAPES, warmup=args.warmup, iters=args.iterations)
    elif args.full_benchmark:
        print("\n[Full Benchmark Mode]")
        run_benchmark(ALL_SHAPES, warmup=args.warmup, iters=args.iterations)
    else:
        print("\n[Benchmark Mode]")
        run_benchmark(HARNESS_SHAPES, warmup=args.warmup, iters=args.iterations)

    print("=" * 62)
