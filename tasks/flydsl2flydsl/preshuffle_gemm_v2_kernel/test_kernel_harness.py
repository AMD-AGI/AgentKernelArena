#!/usr/bin/env python3
"""Real execution-timing test harness for FlyDSL preshuffle_gemm_v2 (flydsl2flydsl).

This harness REPLACES the old "compile-smoke" stub. It actually launches the
compiled kernel on the GPU and times kernel EXECUTION (not compilation) using
torch.cuda.Event timers.

Kernel API (see kernel.py):
    compile_preshuffle_gemm_v2(*, N, K, tile_m, tile_n, tile_k,
                               in_dtype="fp8", out_dtype="bf16",
                               waves_per_eu=None, enable_scheduler=True)
      -> launch_gemm(C, A, B, scale_a, scale_b, M, N, stream)

Tensor layout (verified from kernel.py epilogue + kernels/fp8_gemm_utils.py StoreC):
    A:       [M, K]  fp8 (torch.float8_e4m3fn), row-major
    B:       preshuffle_b(B_logical) where B_logical is [N, K] fp8
    C:       [M, N]  bfloat16
    scale_a: [M]     float32  (per-row scale; sa_nbytes = M*4)
    scale_b: [N]     float32  (per-col scale; sb_nbytes = N*4)
    M, N:    int32
    stream:  torch.cuda.current_stream()

The compiled function is obtained via flyc.compile(exe, *args), which compiles
AND runs once; subsequent calls re-launch only (no recompile) -- this is the
same dispatch path used by kernels/tensor_shim.py::_run_compiled.

Oracle: SELF-REFERENCE. The pristine kernel.py shipped in this task directory is
loaded as the oracle; the candidate kernel.py is loaded from $GEAK_WORK_DIR
(fallback: this task directory). Identical inputs are fed to both and the bf16
outputs are compared with a tight torch.allclose.
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import tempfile
from pathlib import Path

# ============================================================================
# Bootstrap: make `from kernels...` import work + locate kernel dirs
# ============================================================================

KERNEL_FILE = "kernel.py"

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# flydsl2flydsl dir is the parent of this task's kernel dir; it contains the
# `kernels` package used by kernel.py (from kernels.fp8_gemm_utils import ...).
_FLYDSL2_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _FLYDSL2_DIR not in sys.path:
    sys.path.insert(0, _FLYDSL2_DIR)


def _ensure_writable_flydsl_home():
    """FlyDSL's JIT writes its compile cache under ``~/.flydsl``. In the
    container HOME may be a read-only mount, which breaks kernel *execution*
    (not compilation). If the default cache dir is not writable, redirect HOME
    to a writable location. This is a no-op when HOME is already writable (e.g.
    when GEAK runs the harness with a writable work dir)."""
    home = os.path.expanduser("~")
    cache = os.path.join(home, ".flydsl")
    try:
        os.makedirs(cache, exist_ok=True)
        probe = os.path.join(cache, ".write_probe")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
        return  # already writable
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


# Must run before any flydsl import (flydsl resolves the cache dir from HOME).
_ensure_writable_flydsl_home()


def _candidate_kernel_dir():
    """Candidate kernel.py: GEAK_WORK_DIR first, else this task dir."""
    work_dir = os.environ.get("GEAK_WORK_DIR", "").strip()
    for c in [work_dir, _THIS_DIR]:
        if c and os.path.isfile(os.path.join(c, KERNEL_FILE)):
            return c
    return _THIS_DIR


def _oracle_kernel_dir():
    """Oracle kernel.py: ALWAYS the pristine copy shipped in this task dir."""
    return _THIS_DIR


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


_CANDIDATE_DIR = _candidate_kernel_dir()
_ORACLE_DIR = _oracle_kernel_dir()

# ============================================================================
# Shapes + tile configs
#
# Constraints (fp8, gfx942), derived from kernel.py:
#   tile_k % 64 == 0,   K   % tile_k == 0,   (K % 64 == 0 for preshuffle_b)
#   tile_n % 64 == 0,   N   % tile_n == 0,   (N % 16 == 0 for preshuffle_b)
#   tile_m % 16 == 0,   M   % tile_m == 0 (M padded by grid, kept exact here)
# ============================================================================

ALL_SHAPES = [
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
]

# Candidate (tile_m, tile_n, tile_k) configs, tried in order until one
# compiles + runs for the given shape.
_TILE_CANDIDATES = [
    (128, 128, 128),
    (128, 128, 64),
    (64, 128, 128),
    (32, 128, 128),
    (16, 64, 256),
    (32, 64, 256),
    (16, 64, 128),
]


def _valid_tiles(M, N, K, tiles):
    tm, tn, tk = tiles
    return (
        tk % 64 == 0
        and K % tk == 0
        and tn % 64 == 0
        and N % tn == 0
        and tm % 16 == 0
        and M % tm == 0
    )


_n_all = len(ALL_SHAPES)
HARNESS_SHAPES = ALL_SHAPES
_pidx = sorted(set(int(round(i * (_n_all - 1) / 2)) for i in range(3)))
PROFILE_SHAPES = [ALL_SHAPES[i] for i in _pidx]

# Tight tolerance: candidate vs pristine self-reference (same byte semantics).
RTOL, ATOL = 2e-2, 2e-2

# Cache of the first working tile config per (M, N, K) shape so correctness and
# benchmark agree and we never recompile during timing.
_CONFIG_CACHE = {}


# ============================================================================
# Input construction + kernel invocation helpers
# ============================================================================


def _fp8_dtype():
    """Match the kernel's fp8 byte interpretation: gfx942 (and other non-gfx950
    CDNA) MFMA uses E4M3 *FNUZ*; gfx950 uses E4M3 *FN*. Feeding the wrong format
    makes byte 0x80 (==-0 in FN) decode as NaN under FNUZ and poisons the GEMM."""
    import torch

    arch = ""
    try:
        from flydsl.runtime.device import get_rocm_arch

        arch = str(get_rocm_arch())
    except Exception:  # noqa: BLE001
        arch = ""
    if arch.startswith("gfx950") and hasattr(torch, "float8_e4m3fn"):
        return torch.float8_e4m3fn
    if hasattr(torch, "float8_e4m3fnuz"):
        return torch.float8_e4m3fnuz
    return torch.float8_e4m3fn


def _rand_fp8(shape, dtype):
    """Small random fp8 with magnitudes floored to |x|>=0.5 so no value rounds
    to the FNUZ NaN code (0x80), keeping GEMM outputs finite and meaningful."""
    import torch

    x = torch.randn(*shape, device="cuda").clamp_(-2, 2)
    sign = torch.where(x < 0, torch.tensor(-1.0, device="cuda"), torch.tensor(1.0, device="cuda"))
    mag = x.abs().clamp_(min=0.5)
    return (sign * mag).to(dtype)


def _make_inputs(mod, M, N, K, seed):
    import torch

    torch.manual_seed(seed)
    fp8 = _fp8_dtype()

    A = _rand_fp8((M, K), fp8)
    B_logical = _rand_fp8((N, K), fp8)
    B = mod.__dict__["preshuffle_b"](B_logical) if "preshuffle_b" in mod.__dict__ else None
    if B is None:
        from kernels.fp8_gemm_utils import preshuffle_b

        B = preshuffle_b(B_logical)
    B = B.contiguous()
    scale_a = torch.empty(M, device="cuda", dtype=torch.float32).uniform_(0.5, 1.5)
    scale_b = torch.empty(N, device="cuda", dtype=torch.float32).uniform_(0.5, 1.5)
    C = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16)
    return A, B_logical, B, scale_a, scale_b, C


def _compile_and_run_once(mod, flyc, C, A, B, scale_a, scale_b, M, N, tiles):
    """Compile the kernel ONCE (flyc.compile also launches once) and return the
    cached CompiledFunction for fast re-launch."""
    tm, tn, tk = tiles
    exe = mod.compile_preshuffle_gemm_v2(
        N=N,
        K=A.shape[1],
        tile_m=tm,
        tile_n=tn,
        tile_k=tk,
        in_dtype="fp8",
        out_dtype="bf16",
        enable_scheduler=True,
    )
    import torch

    stream = torch.cuda.current_stream()
    cf = flyc.compile(exe, C, A, B, scale_a, scale_b, int(M), int(N), stream)
    torch.cuda.synchronize()
    return cf, stream


def _select_config(mod, flyc, M, N, K, seed=0):
    """Find (and cache) the first tile config that compiles + runs for a shape.

    Returns (tiles, cf, tensors) or raises the last error.
    """
    key = (M, N, K)
    tried = []
    candidates = []
    if key in _CONFIG_CACHE:
        candidates.append(_CONFIG_CACHE[key])
    candidates += [t for t in _TILE_CANDIDATES if t not in candidates]

    last_err = None
    for tiles in candidates:
        if not _valid_tiles(M, N, K, tiles):
            continue
        tried.append(tiles)
        try:
            A, B_logical, B, scale_a, scale_b, C = _make_inputs(mod, M, N, K, seed)
            cf, stream = _compile_and_run_once(mod, flyc, C, A, B, scale_a, scale_b, M, N, tiles)
            _CONFIG_CACHE[key] = tiles
            return tiles, cf, stream, (A, B_logical, B, scale_a, scale_b, C)
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise RuntimeError(
        f"No working tile config for (M={M}, N={N}, K={K}); tried {tried}; last error: {last_err}"
    )


# ============================================================================
# Correctness (self-reference oracle)
# ============================================================================


def run_correctness(shapes=None, verbose=True):
    import torch
    import flydsl.compiler as flyc

    if shapes is None:
        shapes = HARNESS_SHAPES
    if verbose:
        print(f"Running correctness on {len(shapes)} shapes (self-reference oracle)...")

    cand = _load_kernel(_CANDIDATE_DIR, "ps_v2_candidate")
    oracle = _load_kernel(_ORACLE_DIR, "ps_v2_oracle")
    if cand is None or oracle is None:
        print("FAIL: cannot load kernel.py (candidate or oracle)")
        return {"correct": False, "num_correct": 0, "num_failed": len(shapes), "failures": []}

    results, failures = [], []
    for i, (M, N, K) in enumerate(shapes):
        try:
            seed = 1234 + i
            # Build identical inputs ONCE and feed both kernels.
            A, B_logical, B, scale_a, scale_b, C_cand = _make_inputs(cand, M, N, K, seed)
            C_oracle = torch.zeros_like(C_cand)

            # Pick a tile config that works for the candidate kernel.
            tiles = None
            last_err = None
            cands = ([_CONFIG_CACHE[(M, N, K)]] if (M, N, K) in _CONFIG_CACHE else [])
            cands += [t for t in _TILE_CANDIDATES if t not in cands]
            for t in cands:
                if not _valid_tiles(M, N, K, t):
                    continue
                try:
                    cf_c, stream = _compile_and_run_once(cand, flyc, C_cand, A, B, scale_a, scale_b, M, N, t)
                    tiles = t
                    _CONFIG_CACHE[(M, N, K)] = t
                    break
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    continue
            if tiles is None:
                raise RuntimeError(f"no working tile config; last error: {last_err}")

            # Oracle uses the SAME tile config + identical inputs.
            cf_o, _ = _compile_and_run_once(oracle, flyc, C_oracle, A, B, scale_a, scale_b, M, N, tiles)
            torch.cuda.synchronize()

            cf = C_cand.float()
            of = C_oracle.float()
            ok = torch.allclose(cf, of, atol=ATOL, rtol=RTOL)
            max_err = (cf - of).abs().max().item()

            if not ok:
                raise AssertionError(f"max_abs_err={max_err:.4e} exceeds atol={ATOL}/rtol={RTOL}")

            results.append({"config": (M, N, K), "tiles": tiles, "correct": True})
            if verbose:
                print(f"  PASS: (M={M}, N={N}, K={K}) tiles={tiles} max_abs_err={max_err:.4e}")
        except Exception as e:  # noqa: BLE001
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


# ============================================================================
# Profile (light execution loop)
# ============================================================================


def run_profile(shapes=None, warmup=10, iters=50, verbose=True):
    import torch
    import flydsl.compiler as flyc

    if shapes is None:
        shapes = PROFILE_SHAPES
    if verbose:
        print(f"Profile: {len(shapes)} config(s), {warmup} warmup, {iters} iter(s)")

    mod = _load_kernel(_CANDIDATE_DIR, "ps_v2_candidate")
    if mod is None:
        print("FAIL: cannot load kernel.py")
        return

    for M, N, K in shapes:
        tiles, cf, stream, tensors = _select_config(mod, flyc, M, N, K)
        A, B_logical, B, scale_a, scale_b, C = tensors
        args = (C, A, B, scale_a, scale_b, int(M), int(N), stream)
        for _ in range(warmup):
            cf(*args)
        torch.cuda.synchronize()
        for _ in range(iters):
            cf(*args)
        torch.cuda.synchronize()
        if verbose:
            print(f"  (M={M}, N={N}, K={K}) tiles={tiles} done")


# ============================================================================
# Benchmark (real kernel-execution timing)
# ============================================================================


def run_benchmark(shapes=None, warmup=10, iters=100, verbose=True):
    import torch
    import flydsl.compiler as flyc

    if shapes is None:
        shapes = HARNESS_SHAPES

    mod = _load_kernel(_CANDIDATE_DIR, "ps_v2_candidate")
    if mod is None:
        print("FAIL: cannot load kernel.py")
        return {"geomean_latency_ms": -1, "geomean_speedup": -1}

    latencies, speedups, report_cases = [], [], []

    print(f"Running benchmark on {len(shapes)} shapes, {warmup} warmup, {iters} iterations...")
    print(f"{'Config (M,N,K)':<26} {'tiles':>16} {'Ref':>10} {'FlyDSL':>10} {'Speedup':>10}")
    print("-" * 80)

    for idx, (M, N, K) in enumerate(shapes):
        try:
            tiles, cf, stream, tensors = _select_config(mod, flyc, M, N, K, seed=42 + idx)
        except Exception as e:  # noqa: BLE001
            print(f"  SKIP (M={M}, N={N}, K={K}): {str(e)[:100]}")
            continue
        A, B_logical, B, scale_a, scale_b, C = tensors
        args = (C, A, B, scale_a, scale_b, int(M), int(N), stream)

        # Warmup (kernel already compiled; this is pure execution).
        for _ in range(warmup):
            cf(*args)
        torch.cuda.synchronize()

        # Time kernel EXECUTION with CUDA events (median over iters).
        kernel_times = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            cf(*args)
            e.record()
            torch.cuda.synchronize()
            kernel_times.append(s.elapsed_time(e))
        kernel_ms = sum(kernel_times) / len(kernel_times)

        # Reference baseline: torch.mm of dequantized operands (for speedup display).
        a_f = A.float()
        b_f = B_logical.float()
        for _ in range(min(warmup, 5)):
            _ = torch.mm(a_f, b_f.T)
        torch.cuda.synchronize()
        ref_times = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            _ = torch.mm(a_f, b_f.T)
            e.record()
            torch.cuda.synchronize()
            ref_times.append(s.elapsed_time(e))
        ref_ms = sum(ref_times) / len(ref_times)

        speedup = ref_ms / kernel_ms if kernel_ms > 0 else 1.0
        latencies.append(kernel_ms)
        speedups.append(speedup)

        flops = 2.0 * M * N * K
        tflops = flops / (kernel_ms * 1e-3) / 1e12

        report_cases.append({
            "test_case_id": f"test_case_{idx}",
            "execution_time_ms": kernel_ms,
            "shape": [M, N, K],
            "params": {"M": M, "N": N, "K": K, "dtype": "fp8", "tiles": list(tiles)},
            "tflops": tflops,
        })

        marker = " *" if speedup > 1.0 else ""
        if verbose:
            print(
                f"(M={M:>5}, N={N:>5}, K={K:>5}) {str(tiles):>16}"
                f" {ref_ms:>8.4f}ms {kernel_ms:>8.4f}ms {speedup:>8.2f}x{marker}",
                flush=True,
            )

        del A, B_logical, B, scale_a, scale_b, C, a_f, b_f
        torch.cuda.empty_cache()

    if not latencies:
        print("FAIL: no shapes produced timings")
        return {"geomean_latency_ms": -1, "geomean_speedup": -1}

    geomean_latency = math.exp(sum(math.log(l) for l in latencies) / len(latencies))
    geomean_speedup = math.exp(sum(math.log(s) for s in speedups) / len(speedups))

    build_dir = Path(_CANDIDATE_DIR) / "build"
    build_dir.mkdir(exist_ok=True)
    with open(build_dir / "performance_report.json", "w") as f:
        json.dump(report_cases, f, indent=2)

    print("-" * 80)
    print(f"{'Geometric mean latency:':<26} {geomean_latency:.4f} ms")
    print(f"{'Geometric mean speedup:':<26} {geomean_speedup:.2f}x")
    print(f"GEAK_RESULT_LATENCY_MS={geomean_latency:.4f}", flush=True)
    print(f"GEAK_RESULT_GEOMEAN_SPEEDUP={geomean_speedup:.4f}", flush=True)

    return {"geomean_latency_ms": geomean_latency, "geomean_speedup": geomean_speedup}


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlyDSL preshuffle_gemm_v2 Kernel Test Harness")
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
    print("FlyDSL preshuffle_gemm_v2 Kernel")
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
