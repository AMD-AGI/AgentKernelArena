#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Test harness for the gemm_a16wfp4 (MXFP4) Triton kernel.

Modes: --correctness, --profile, --benchmark, --full-benchmark
"""
import argparse
import math
import os
import sys
import torch

from kernel import gemm_a16wfp4, is_fp4_avail


# ============================================================================
# CONSTANTS
# ============================================================================

# Specified by the HW and cannot be changed.
SCALE_GROUP_SIZE = 32

DTYPE = torch.bfloat16

# Tolerance defaults — match the previous in-harness assert_close (rtol=1e-2, atol=1e-2).
RTOL, ATOL = 1e-2, 1e-2


# ============================================================================
# SHAPE LISTS
# ============================================================================

# ALL_SHAPES: All unique shapes from test file, sorted by total element count.
ALL_SHAPES = [
    (1, 8192, 1024),
    (1, 1280, 8192),
    (1, 7168, 2048),
    (1, 2112, 7168),
    (1, 4096, 4096),
    (4, 7168, 2048),
    (4, 2112, 7168),
    (8, 7168, 2048),
    (32, 512, 7168),
    (8, 2112, 7168),
    (2, 8192, 8192),
    (32, 8192, 1024),
    (32, 1280, 8192),
    (32, 7168, 2048),
    (32, 2112, 7168),
    (64, 8192, 1024),
    (4, 12288, 12288),
    (64, 1280, 8192),
    (64, 7168, 2048),
    (64, 2112, 7168),
    (128, 8192, 1024),
    (1024, 1024, 1024),
    (128, 1280, 8192),
    (192, 8192, 1024),
    (16, 16384, 6656),
    (128, 7168, 2048),
    (128, 2112, 7168),
    (192, 1280, 8192),
    (8, 16384, 16384),
    (256, 8192, 1024),
    (320, 8192, 1024),
    (256, 1280, 8192),
    (320, 1280, 8192),
    (512, 8192, 1024),
    (512, 1280, 8192),
    (16, 20480, 20480),
    (1024, 8192, 1024),
    (2048, 2048, 2048),
    (1024, 1280, 8192),
    (128, 16384, 6656),
    (2048, 8192, 1024),
    (2048, 1280, 8192),
    (3072, 3072, 3072),
    (4096, 8192, 1024),
    (4096, 1280, 8192),
    (8192, 8192, 1024),
    (4096, 4096, 4096),
    (8192, 1280, 8192),
    (5120, 5120, 5120),
    (16384, 8192, 1024),
    (4864, 4096, 8192),
    # (4864, 8192, 4160),  # Skipped due to compilation error
    (16384, 1280, 8192),
    (6144, 6144, 6144),
    (7168, 7168, 7168),
    (8192, 8192, 8192),
    # (9728, 8192, 65536),  # Too large, may cause OOM
]

# HARNESS_SHAPES: 25 uniformly sampled shapes from ALL_SHAPES.
_n_all = len(ALL_SHAPES)
if _n_all <= 25:
    HARNESS_SHAPES = ALL_SHAPES
else:
    _harness_indices = [int(round(i * (_n_all - 1) / 24)) for i in range(25)]
    HARNESS_SHAPES = [ALL_SHAPES[i] for i in _harness_indices]

# PROFILE_SHAPES: 5 evenly-spaced shapes for profiling.
PROFILE_SHAPES = [
    (1, 8192, 1024),
    (32, 7168, 2048),
    (256, 8192, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
]


# ============================================================================
# PYTORCH REFERENCE (correctness-only)
# ============================================================================

def mxfp4_to_f32(x):
    """Convert MXFP4 packed uint8 to float32."""
    x = x.repeat_interleave(2, dim=-1)
    x[..., ::2] = x[..., ::2] & 0xF
    x[..., 1::2] = x[..., 1::2] >> 4
    mxfp4_list = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ]
    mxfp4_in_f32 = torch.tensor(mxfp4_list, dtype=torch.float32, device="cuda")
    return mxfp4_in_f32[x.long()]


def e8m0_to_f32(x):
    """Convert E8M0 scale to float32."""
    x_f32 = 2 ** (x.to(torch.float32) - 127)
    x_f32[x_f32 == 128] = float("nan")
    return x_f32


def run_torch_reference(x, w, w_scales, dtype):
    """Compute reference output using PyTorch."""
    x_f32 = x.to(torch.float32)
    w_f32 = mxfp4_to_f32(w)
    w_scales_expanded = w_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=-1).to(torch.float32)
    w_scales_f32 = e8m0_to_f32(w_scales_expanded)
    assert w_f32.shape == w_scales_f32.shape
    w_f32 = w_f32 * w_scales_f32
    return torch.mm(x_f32, w_f32.T).to(dtype)


# ============================================================================
# INPUT GENERATION
# ============================================================================

def generate_inputs(M, N, K, dtype=DTYPE):
    """Generate inputs for gemm_a16wfp4 kernel."""
    torch.manual_seed(42)

    # Generate x (bf16 input) — TN layout only
    x_low = torch.randint(0, 16, (M, K // 2), dtype=torch.uint8, device="cuda")
    x_high = torch.randint(0, 16, (M, K // 2), dtype=torch.uint8, device="cuda")
    x_uint8 = x_low | x_high << 4

    # Generate x_scales and convert x to bf16
    x_scales = torch.randint(124, 128, (K // SCALE_GROUP_SIZE, M), dtype=torch.uint8, device="cuda").T
    x_f32 = mxfp4_to_f32(x_uint8)
    x_scales_expanded = x_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=-1).to(torch.float32)
    x_scales_f32 = e8m0_to_f32(x_scales_expanded)
    x_f32 = x_f32 * x_scales_f32
    x = x_f32.to(dtype)

    # Generate w (fp4 weights) — TN layout only
    w_low = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device="cuda")
    w_high = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device="cuda")
    w = w_low | w_high << 4

    # Generate w_scales
    w_scales = torch.randint(124, 128, (K // SCALE_GROUP_SIZE, N), dtype=torch.uint8, device="cuda").T

    # Non-preshuffled deterministic path only.
    return x, w, w, w_scales, w_scales


# ============================================================================
# TEST HARNESS
# ============================================================================

def _label(cfg):
    M, N, K = cfg
    return f"({M}, {N}, {K})"


def run_correctness(shapes=None, verbose=True):
    if shapes is None:
        shapes = HARNESS_SHAPES
    if not is_fp4_avail():
        print("MXFP4 not supported on this architecture, skipping correctness tests")
        return {"correct": True, "num_correct": 0, "num_failed": 0,
                "failures": [], "results": [], "skipped": True}

    if verbose:
        print(f"Running correctness on {len(shapes)} shapes...")

    results, failures = [], []

    for cfg in shapes:
        M, N, K = cfg
        try:
            x, w, w_kernel, w_scales, w_scales_kernel = generate_inputs(M, N, K, DTYPE)
            y = gemm_a16wfp4(x, w_kernel, w_scales_kernel, atomic_add=False, dtype=DTYPE)
            y_ref = run_torch_reference(x, w, w_scales, DTYPE)
            torch.cuda.synchronize()

            torch.testing.assert_close(y, y_ref, atol=ATOL, rtol=RTOL)
            results.append({"config": cfg, "correct": True})
            if verbose:
                print(f"  PASS: {_label(cfg)}")
            del x, w, w_kernel, w_scales, w_scales_kernel, y, y_ref
            torch.cuda.empty_cache()
        except Exception as e:
            failures.append({"config": cfg, "error": str(e)})
            if verbose:
                print(f"  FAIL: {_label(cfg)} - {str(e)[:80]}")

    if verbose:
        print("-" * 62)
        status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{len(shapes)})"
        print(f"{'Status:':<22} {status}")

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
    if not is_fp4_avail():
        print("MXFP4 not supported on this architecture")
        return
    if verbose:
        print(f"Profile: {len(shapes)} config(s), {warmup} warmup, {iters} iter(s)")

    for cfg in shapes:
        M, N, K = cfg
        x, w, w_kernel, w_scales, w_scales_kernel = generate_inputs(M, N, K, DTYPE)
        for _ in range(warmup):
            gemm_a16wfp4(x, w_kernel, w_scales_kernel, atomic_add=False, dtype=DTYPE)
        torch.cuda.synchronize()
        for _ in range(iters):
            gemm_a16wfp4(x, w_kernel, w_scales_kernel, atomic_add=False, dtype=DTYPE)
        torch.cuda.synchronize()
        if verbose:
            print(f"  {_label(cfg)} done")
        del x, w, w_kernel, w_scales, w_scales_kernel
        torch.cuda.empty_cache()


def run_benchmark(shapes=None, warmup=50, iters=200, verbose=True):
    if shapes is None:
        shapes = HARNESS_SHAPES
    if not is_fp4_avail():
        print("MXFP4 not supported on this architecture")
        print("GEAK_RESULT_LATENCY_MS=0.0", flush=True)
        return {"geomean_latency_ms": 0.0, "latencies": [], "skipped": True}

    latencies = []

    print(f"Running benchmark on {len(shapes)} shapes, {warmup} warmup, {iters} iterations each...")
    if verbose:
        print(f"{'Config':<22} {'Triton':>10}")
        print("-" * 34)

    for cfg in shapes:
        M, N, K = cfg
        x, w, w_kernel, w_scales, w_scales_kernel = generate_inputs(M, N, K, DTYPE)

        for _ in range(warmup):
            gemm_a16wfp4(x, w_kernel, w_scales_kernel, atomic_add=False, dtype=DTYPE)
        torch.cuda.synchronize()

        triton_times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            gemm_a16wfp4(x, w_kernel, w_scales_kernel, atomic_add=False, dtype=DTYPE)
            end.record()
            torch.cuda.synchronize()
            triton_times.append(start.elapsed_time(end))

        triton_ms = sorted(triton_times)[len(triton_times) // 2]
        latencies.append(triton_ms)

        if verbose:
            print(f"{_label(cfg):<22} {triton_ms:>8.4f}ms", flush=True)

        del x, w, w_kernel, w_scales, w_scales_kernel
        torch.cuda.empty_cache()

    geomean_latency = math.exp(sum(math.log(l) for l in latencies) / len(latencies))

    print("-" * 34)
    print(f"{'Geometric mean latency:':<22} {geomean_latency:.4f} ms")
    print(f"GEAK_SHAPES_USED={list(range(len(shapes)))}")
    print(f"GEAK_RESULT_LATENCY_MS={geomean_latency:.4f}", flush=True)

    return {"geomean_latency_ms": geomean_latency, "latencies": latencies}


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gemm_a16wfp4 (MXFP4) Test Harness")
    parser.add_argument("--correctness", action="store_true",
                        help="Run correctness tests on HARNESS_SHAPES")
    parser.add_argument("--profile", action="store_true",
                        help="Run minimal profiling workload")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark on HARNESS_SHAPES (25 uniformly sampled)")
    parser.add_argument("--full-benchmark", action="store_true",
                        help="Run benchmark on ALL_SHAPES (complete set)")
    parser.add_argument("--warmup", type=int, default=50,
                        help="Number of warmup iterations (default: 50)")
    parser.add_argument("--iterations", type=int,
                        default=int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200")),
                        help="Number of benchmark iterations (default: GEAK_BENCHMARK_ITERATIONS or 200)")
    args = parser.parse_args()

    print("=" * 62)
    print("gemm_a16wfp4 (MXFP4) Test Harness")
    print("=" * 62)

    if args.correctness:
        print("\n[Correctness Mode]")
        result = run_correctness(HARNESS_SHAPES)
        sys.exit(0 if result["correct"] else 1)
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
