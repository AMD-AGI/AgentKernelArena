#!/usr/bin/env python3
"""
Test harness for the fused gated MLP (SwiGLU) backward Triton kernel.

Modes: --correctness, --profile, --benchmark, --full-benchmark
"""
import argparse
import math
import os
import sys
import torch

from kernel import (
    ff_fused_gated_forward, ff_fused_gated_backward_triton,
    get_inputs,
    EVAL_CONFIGS, PROFILE_CONFIGS, ACTIVATION,
    RTOL, ATOL,
)


# ============================================================================
# SHAPE SUBSETS
# ============================================================================

ALL_SHAPES = EVAL_CONFIGS

_n_all = len(ALL_SHAPES)
if _n_all <= 25:
    HARNESS_SHAPES = ALL_SHAPES
else:
    _harness_indices = [int(round(i * (_n_all - 1) / 24)) for i in range(25)]
    HARNESS_SHAPES = [ALL_SHAPES[i] for i in _harness_indices]

# PROFILE_CONFIGS already provides the profile subset per kernel.py.
PROFILE_SHAPES = PROFILE_CONFIGS


# ============================================================================
# PYTORCH REFERENCE (moved from kernel.py; correctness-only)
# ============================================================================

def silu_backward(x, grad):
    sigmoid_x = torch.sigmoid(x)
    return grad * (sigmoid_x + x * sigmoid_x * (1 - sigmoid_x))


def _pytorch_backward_reference(dy, x, w_up, w_down, h0, h1, a, g, activation='silu'):
    N_half = h0.shape[1]

    dg = torch.matmul(dy, w_down.t())

    if activation == 'silu':
        da = silu_backward(h0, dg * h1)
    else:
        da = dg * h1

    dh0 = da
    dh1 = dg * a

    w_gate = w_up[:N_half, :]
    w_value = w_up[N_half:, :]
    dx = torch.matmul(dh0, w_gate) + torch.matmul(dh1, w_value)

    dw_gate = torch.matmul(dh0.t(), x)
    dw_value = torch.matmul(dh1.t(), x)
    dw_up = torch.cat([dw_gate, dw_value], dim=0)

    dw_down = torch.matmul(g.t(), dy)

    return dx, dw_up, dw_down


# ============================================================================
# TEST HARNESS
# ============================================================================

def _label(cfg):
    M, N, K = cfg
    return f"M={M},N={N},K={K}"


def run_correctness(shapes=None, verbose=True):
    if shapes is None:
        shapes = HARNESS_SHAPES
    if verbose:
        print(f"Running correctness on {len(shapes)} shapes...")

    results, failures = [], []

    for cfg in shapes:
        try:
            M, N, K = cfg
            x, w_up, w_down, dy = get_inputs(M, K, N)
            y, h0, h1, a, g = ff_fused_gated_forward(x, w_up, w_down, ACTIVATION)
            dx_t, dwup_t, dwdown_t = ff_fused_gated_backward_triton(
                dy, x, w_up, w_down, h0, h1, a, g, ACTIVATION)
            dx_r, dwup_r, dwdown_r = _pytorch_backward_reference(
                dy, x, w_up, w_down, h0, h1, a, g, ACTIVATION)
            torch.cuda.synchronize()

            torch.testing.assert_close(dx_t, dx_r, atol=ATOL, rtol=RTOL)
            torch.testing.assert_close(dwup_t, dwup_r, atol=ATOL, rtol=RTOL)
            torch.testing.assert_close(dwdown_t, dwdown_r, atol=ATOL, rtol=RTOL)
            results.append({"config": cfg, "correct": True})
            if verbose:
                print(f"  PASS: {_label(cfg)}")
            del x, w_up, w_down, dy, y, h0, h1, a, g
            del dx_t, dwup_t, dwdown_t, dx_r, dwup_r, dwdown_r
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
    if verbose:
        print(f"Profile: {len(shapes)} config(s), {warmup} warmup, {iters} iter(s)")

    for cfg in shapes:
        M, N, K = cfg
        x, w_up, w_down, dy = get_inputs(M, K, N)
        y, h0, h1, a, g = ff_fused_gated_forward(x, w_up, w_down, ACTIVATION)
        for _ in range(warmup):
            ff_fused_gated_backward_triton(dy, x, w_up, w_down, h0, h1, a, g, ACTIVATION)
        torch.cuda.synchronize()
        for _ in range(iters):
            ff_fused_gated_backward_triton(dy, x, w_up, w_down, h0, h1, a, g, ACTIVATION)
        torch.cuda.synchronize()
        if verbose:
            print(f"  {_label(cfg)} done")
        del x, w_up, w_down, dy, y, h0, h1, a, g
        torch.cuda.empty_cache()


def run_benchmark(shapes=None, warmup=50, iters=200, verbose=True):
    if shapes is None:
        shapes = HARNESS_SHAPES

    latencies = []

    print(f"Running benchmark on {len(shapes)} shapes, {warmup} warmup, {iters} iterations each...")
    if verbose:
        print(f"{'Config':<22} {'Triton':>10}")
        print("-" * 34)

    for cfg in shapes:
        M, N, K = cfg
        x, w_up, w_down, dy = get_inputs(M, K, N)
        # Forward computed ONCE outside the timed loop; the candidate Triton
        # backward consumes the cached intermediates h0/h1/a/g.
        y, h0, h1, a, g = ff_fused_gated_forward(x, w_up, w_down, ACTIVATION)

        for _ in range(warmup):
            ff_fused_gated_backward_triton(dy, x, w_up, w_down, h0, h1, a, g, ACTIVATION)
        torch.cuda.synchronize()

        triton_times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            ff_fused_gated_backward_triton(dy, x, w_up, w_down, h0, h1, a, g, ACTIVATION)
            end.record()
            torch.cuda.synchronize()
            triton_times.append(start.elapsed_time(end))

        triton_ms = sorted(triton_times)[len(triton_times) // 2]
        latencies.append(triton_ms)

        if verbose:
            print(f"{_label(cfg):<22} {triton_ms:>8.4f}ms", flush=True)

        del x, w_up, w_down, dy, y, h0, h1, a, g
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
    parser = argparse.ArgumentParser(description="FF Backward (SwiGLU) Test Harness")
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
    print("FF Backward (SwiGLU) Test Harness")
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
