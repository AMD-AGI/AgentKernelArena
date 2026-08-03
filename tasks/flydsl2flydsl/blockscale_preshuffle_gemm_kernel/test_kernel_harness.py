#!/usr/bin/env python3
"""Test harness for FlyDSL blockscale_preshuffle_gemm_kernel (flydsl2flydsl).

Real execution-timing harness for the FP8 A8W8 block-scaled GEMM with 128x128
block scaling (ScaleBlockM=1, ScaleBlockN=128, ScaleBlockK=128).

Kernel API (kernel.py):
    compile_blockscale_preshuffle_gemm(*, M, N, K, tile_m, tile_n, tile_k,
        scale_block_k=128, out_dtype="bf16", ...) -> launch_gemm
    launch_gemm(arg_c, arg_a, arg_b, arg_scale_a, arg_scale_b, i32_m, i32_n, stream)

Tensor layouts:
    arg_a       : A [M, K]  architecture-matched FP8 E4M3
    arg_b       : B preshuffled fp8 from logical B [N, K] via preshuffle_b()
    arg_scale_a : [scale_k, M] float32 (TRANSPOSED), scale_k = K // scale_block_k
    arg_scale_b : [scale_n, scale_k] float32 row-major, scale_n = N // 128
    arg_c       : [M, N] bfloat16 output
    i32_m=M, i32_n=N int32, stream=torch.cuda.current_stream()

Dequant math (per element):
    C[m,n] = sum_kb (sum_{k in block kb} A[m,k]*B[n,k])
             * scale_a[kb, m] * scale_b[n//128, kb]      (kb = k // scale_block_k)

Correctness is checked against an independent PyTorch blockscale reference.
The reference dequantizes each 128-element K block with the configured row and
weight scales, accumulates in float32, and applies the kernel's bf16 output
quantization before comparison.
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import tempfile
from pathlib import Path

from _aka_benchmark import benchmark_cuda_graph_or_events

# ============================================================================
# GEAK bootstrap
# ============================================================================

KERNEL_FILE = "kernel.py"

# The flydsl2flydsl dir (parent of this task dir) holds the shared `kernels`
# package; make `from kernels.fp8_gemm_utils import preshuffle_b` importable.
_TASK_DIR = os.path.dirname(os.path.abspath(__file__))
_FLYDSL2_DIR = os.path.abspath(os.path.join(_TASK_DIR, ".."))
for _p in (_FLYDSL2_DIR, _TASK_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _ensure_writable_flydsl_home():
    """FlyDSL JIT cache lives under ~/.flydsl; redirect HOME when read-only."""
    home = os.path.expanduser("~")
    cache = os.path.join(home, ".flydsl")
    try:
        os.makedirs(cache, exist_ok=True)
        probe = os.path.join(cache, ".write_probe")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
        return
    except OSError:
        pass
    for base in (
        os.environ.get("GEAK_WORK_DIR", "").strip(),
        tempfile.gettempdir(),
        _FLYDSL2_DIR,
    ):
        if not base:
            continue
        try:
            new_home = os.path.join(base, ".flydsl_home")
            os.makedirs(os.path.join(new_home, ".flydsl"), exist_ok=True)
            os.environ["HOME"] = new_home
            return
        except OSError:
            continue


_ensure_writable_flydsl_home()


def _candidate_kernel_dir():
    """Kernel-under-test: $GEAK_WORK_DIR if it has kernel.py, else task dir."""
    work_dir = os.environ.get("GEAK_WORK_DIR", "").strip()
    if work_dir and os.path.isfile(os.path.join(work_dir, KERNEL_FILE)):
        return work_dir
    return _TASK_DIR


def _load_kernel(kernel_dir, alias):
    entry = os.path.join(kernel_dir, KERNEL_FILE)
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


_KERNEL_DIR = _candidate_kernel_dir()

# ============================================================================
# Test shapes: (M, N, K)
#   M multiple of 32 (tile_m), N multiple of 128 (and tile_n=64), K multiple
#   of 256 (tile_k). These satisfy all kernel constraints with the known-good
#   default tiles tile_m=32, tile_n=64, tile_k=256.
# ============================================================================

TILE_M, TILE_N, TILE_K = 32, 64, 256
SCALE_BLOCK_K = 128
OUT_DTYPE = "bf16"

ALL_SHAPES = [
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
]

_n_all = len(ALL_SHAPES)
if _n_all <= 25:
    HARNESS_SHAPES = list(ALL_SHAPES)
else:
    _idx = [int(round(i * (_n_all - 1) / 24)) for i in range(25)]
    HARNESS_SHAPES = [ALL_SHAPES[i] for i in _idx]

_pidx = [int(round(i * (_n_all - 1) / 4)) for i in range(5)]
PROFILE_SHAPES = [ALL_SHAPES[i] for i in _pidx]

RTOL, ATOL = 2e-2, 2e-2

# ============================================================================
# Input generation / reference
# ============================================================================


def _fp8_dtype():
    """Match the architecture's FP8 MFMA interpretation."""
    import torch

    arch = ""
    try:
        from flydsl.runtime.device import get_rocm_arch

        arch = str(get_rocm_arch())
    except Exception:  # noqa: BLE001
        pass
    if arch.startswith("gfx95") and hasattr(torch, "float8_e4m3fn"):
        return torch.float8_e4m3fn
    return getattr(torch, "float8_e4m3fnuz", torch.float8_e4m3fn)


def _rand_fp8(shape, dtype):
    """Random fp8 with |x|>=0.5 so FNUZ NaN byte 0x80 is never produced."""
    import torch

    x = torch.randn(*shape, device="cuda").clamp_(-2, 2)
    sign = torch.where(x < 0, torch.tensor(-1.0, device="cuda"), torch.tensor(1.0, device="cuda"))
    mag = x.abs().clamp_(min=0.5)
    return (sign * mag).to(dtype)


def _make_inputs(M, N, K, seed):
    """Create FP8 inputs + fp32 block scales for one (M, N, K) case.

    Returns a dict of the exact tensors the kernel launch expects, plus the
    logical (unshuffled) fp8 tensors and scales for building references.
    """
    import torch

    torch.manual_seed(seed)
    dev = "cuda"
    fp8 = _fp8_dtype()

    scale_k = K // SCALE_BLOCK_K  # rows of scale_a / cols of scale_b
    scale_n = N // 128            # rows of scale_b

    a_fp8 = _rand_fp8((M, K), fp8)
    b_fp8 = _rand_fp8((N, K), fp8)  # logical B [N, K]

    from kernels.fp8_gemm_utils import preshuffle_b
    b_shuf = preshuffle_b(b_fp8).contiguous()

    # scale_a: [scale_k, M] transposed layout; scale_b: [scale_n, scale_k]
    scale_a = torch.empty(scale_k, M, device=dev, dtype=torch.float32).uniform_(0.5, 1.5)
    scale_b = torch.empty(scale_n, scale_k, device=dev, dtype=torch.float32).uniform_(0.5, 1.5)

    c = torch.zeros(M, N, device=dev, dtype=torch.bfloat16)

    return {
        "M": M, "N": N, "K": K,
        "scale_k": scale_k, "scale_n": scale_n,
        "a_fp8": a_fp8, "b_fp8": b_fp8, "b_shuf": b_shuf,
        "scale_a": scale_a, "scale_b": scale_b, "c": c,
    }


def _compile_and_run_once(mod, flyc, out_c, inp):
    """Compile via flyc.compile (also launches once) and return cached cf."""
    import torch

    launch_gemm = mod.compile_blockscale_preshuffle_gemm(
        M=inp["M"], N=inp["N"], K=inp["K"],
        tile_m=TILE_M, tile_n=TILE_N, tile_k=TILE_K,
        scale_block_k=SCALE_BLOCK_K, out_dtype=OUT_DTYPE,
        use_async_copy=False,
    )
    stream = torch.cuda.current_stream()
    cf = flyc.compile(
        launch_gemm,
        out_c, inp["a_fp8"], inp["b_shuf"], inp["scale_a"], inp["scale_b"],
        inp["M"], inp["N"], stream,
    )
    torch.cuda.synchronize()
    return cf, stream


def _launch_args(inp, out_c, stream):
    return (
        out_c, inp["a_fp8"], inp["b_shuf"], inp["scale_a"], inp["scale_b"],
        inp["M"], inp["N"], stream,
    )


def _torch_blockscale_reference(inp):
    """Block-scaled dequant reference in float32.

    C[m,n] = sum_kb (sum_{k in block kb} A[m,k]*B[n,k])
             * scale_a[kb, m] * scale_b[n//128, kb]
    """
    import torch

    M, N, K = inp["M"], inp["N"], inp["K"]
    scale_k = inp["scale_k"]
    a = inp["a_fp8"].float()                       # [M, K]
    b = inp["b_fp8"].float()                        # [N, K]
    scale_a = inp["scale_a"]                        # [scale_k, M]
    scale_b = inp["scale_b"]                        # [scale_n, scale_k]

    # Expand each block scale over its 128 K elements, then use a normal
    # float32 GEMM. This mirrors the mathematical definition without calling
    # any implementation from the candidate module.
    a_dequant = (
        a.view(M, scale_k, SCALE_BLOCK_K)
        * scale_a.t().unsqueeze(-1)
    ).reshape(M, K)
    sb_full = scale_b.repeat_interleave(128, dim=0)  # [N, scale_k]
    b_dequant = (
        b.view(N, scale_k, SCALE_BLOCK_K)
        * sb_full.unsqueeze(-1)
    ).reshape(N, K)
    return torch.mm(a_dequant, b_dequant.t())


# ============================================================================
# Modes
# ============================================================================


def run_correctness(shapes=None, verbose=True):
    import torch
    import flydsl.compiler as flyc

    if shapes is None:
        shapes = HARNESS_SHAPES
    if verbose:
        print(f"Running correctness on {len(shapes)} shapes...")
        print(f"  candidate kernel dir: {_KERNEL_DIR}")
        print("  oracle type: independent PyTorch blockscale reference")

    cand_mod = _load_kernel(_KERNEL_DIR, "bs_gemm_candidate")
    if cand_mod is None:
        print("FAIL: cannot load candidate kernel.py")
        return {"correct": False, "num_correct": 0, "num_failed": len(shapes), "failures": []}

    results, failures = [], []
    for i, (M, N, K) in enumerate(shapes):
        try:
            inp = _make_inputs(M, N, K, seed=42 + i)

            c_cand = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16)
            _compile_and_run_once(cand_mod, flyc, c_cand, inp)
            torch.cuda.synchronize()

            actual = c_cand.float()
            ref = _torch_blockscale_reference(inp).to(torch.bfloat16).float()
            ok = torch.allclose(actual, ref, atol=ATOL, rtol=RTOL)
            max_err = (actual - ref).abs().max().item()

            if not ok:
                raise AssertionError(f"candidate vs PyTorch max_err={max_err:.4e} > tol")

            results.append({"config": (M, N, K), "correct": True})
            if verbose:
                print(f"  PASS: (M={M}, N={N}, K={K}) max_err={max_err:.4e}")
        except Exception as e:
            failures.append({"config": (M, N, K), "error": str(e)})
            if verbose:
                print(f"  FAIL: (M={M}, N={N}, K={K}) - {str(e)[:120]}")

    if verbose:
        print("-" * 62)
        status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{len(shapes)})"
        print(f"{'Status:':<22} {status}")

    return {
        "correct": len(failures) == 0,
        "num_correct": len(results),
        "num_failed": len(failures),
        "failures": failures,
    }


def run_profile(shapes=None, warmup=10, iters=50, verbose=True):
    import torch
    import flydsl.compiler as flyc

    if shapes is None:
        shapes = PROFILE_SHAPES
    if verbose:
        print(f"Profile: {len(shapes)} config(s), {warmup} warmup, {iters} iter(s)")

    mod = _load_kernel(_KERNEL_DIR, "bs_gemm_candidate")
    if mod is None:
        print("FAIL: cannot load kernel.py")
        return

    for M, N, K in shapes:
        inp = _make_inputs(M, N, K, seed=42)
        c = inp["c"]
        cf, stream = _compile_and_run_once(mod, flyc, c, inp)
        args = _launch_args(inp, c, stream)
        for _ in range(warmup):
            cf(*args)
        torch.cuda.synchronize()
        for _ in range(iters):
            cf(*args)
        torch.cuda.synchronize()
        if verbose:
            print(f"  (M={M}, N={N}, K={K}) done")


def _time_mean_ms(fn, warmup, iters):
    """Graph-first mean GPU time in milliseconds for one callable invocation."""
    return benchmark_cuda_graph_or_events(
        fn, warmup=warmup, repetition=iters
    )


def run_benchmark(shapes=None, warmup=10, iters=100, verbose=True):
    import torch
    import flydsl.compiler as flyc

    if shapes is None:
        shapes = HARNESS_SHAPES

    mod = _load_kernel(_KERNEL_DIR, "bs_gemm_candidate")
    if mod is None:
        print("FAIL: cannot load kernel.py")
        return {"geomean_latency_ms": -1, "geomean_speedup": -1}

    latencies, speedups, report_cases = [], [], []

    print(f"Running benchmark on {len(shapes)} shapes, {warmup} warmup, {iters} iterations...")
    print(f"{'Config (M,N,K)':<28} {'Ref':>10} {'FlyDSL':>10} {'Speedup':>10} {'TFLOP/s':>10}")
    print("-" * 74)

    for idx, (M, N, K) in enumerate(shapes):
        inp = _make_inputs(M, N, K, seed=42)

        # Compile ONCE (outside the timing loop) then time EXECUTION only.
        c = inp["c"]
        cf, stream = _compile_and_run_once(mod, flyc, c, inp)
        args = _launch_args(inp, c, stream)

        # torch reference for speedup display: mm of logical A,B in float.
        a_ref = inp["a_fp8"].float()                 # [M, K]
        b_ref = inp["b_fp8"].float()                 # [N, K] logical

        def kfn():
            cf(*(args[:-1] + (torch.cuda.current_stream(),)))

        def reffn():
            torch.mm(a_ref, b_ref.t())

        kernel_ms, kernel_bench_meta = _time_mean_ms(kfn, warmup, iters)
        ref_ms, ref_bench_meta = _time_mean_ms(reffn, warmup, iters)

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

        flops = 2.0 * M * N * K
        tflops = flops / (kernel_ms * 1e-3) / 1e12

        report_cases.append({
            "test_case_id": f"test_case_{idx}",
            "execution_time_ms": kernel_ms,
            **kernel_bench_meta,
            "reference_benchmark_method": ref_bench_meta["benchmark_method"],
            "benchmark_method_consistent": kernel_bench_meta["benchmark_method"] == ref_bench_meta["benchmark_method"],
            "shape": [M, N, K],
            "params": {"M": M, "N": N, "K": K, "dtype": OUT_DTYPE,
                       "tile_m": TILE_M, "tile_n": TILE_N, "tile_k": TILE_K},
            "tflops": tflops,
            "ref_time_ms": ref_ms,
            "speedup_vs_torch": speedup,
        })

        marker = " *" if speedup is not None and speedup > 1.0 else ""
        if verbose:
            print(
                f"(M={M:>5}, N={N:>5}, K={K:>5})"
                f" {ref_ms:>8.4f}ms {kernel_ms:>8.4f}ms {speedup_display}{marker}"
                f" {tflops:>9.1f}",
                flush=True,
            )

        del inp, c, a_ref, b_ref
        torch.cuda.empty_cache()

    geomean_latency = math.exp(sum(math.log(l) for l in latencies) / len(latencies))
    geomean_speedup = math.exp(sum(math.log(s) for s in speedups) / len(speedups)) if speedups else None
    geomean_speedup_display = (
        format(geomean_speedup, ".2f") + "x"
        if geomean_speedup is not None
        else "N/A"
    )

    build_dir = Path(_KERNEL_DIR) / "build"
    build_dir.mkdir(exist_ok=True)
    with open(build_dir / "performance_report.json", "w") as f:
        json.dump(report_cases, f, indent=2)

    print("-" * 74)
    print(f"{'Geometric mean latency:':<26} {geomean_latency:.4f} ms")
    print(f"{'Geometric mean speedup:':<26} {geomean_speedup_display}")
    print(f"GEAK_RESULT_LATENCY_MS={geomean_latency:.4f}", flush=True)
    if geomean_speedup is not None:
        print(f"GEAK_RESULT_GEOMEAN_SPEEDUP={geomean_speedup:.4f}", flush=True)

    return {"geomean_latency_ms": geomean_latency, "geomean_speedup": geomean_speedup}


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlyDSL Blockscale Preshuffle GEMM Test Harness")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--iterations",
        type=int,
        default=int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "100")),
    )
    args = parser.parse_args()

    print("=" * 62)
    print("FlyDSL Blockscale Preshuffle GEMM (FP8 A8W8, 128x128 block scale)")
    print("=" * 62)

    if args.correctness:
        print("\n[Correctness Mode]")
        result = run_correctness(HARNESS_SHAPES)
        sys.exit(0 if result.get("correct", False) else 1)
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
