# SPDX-License-Identifier: Apache-2.0
# Modifications Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

# The kernel in this file is adapted from TritonBench's fast_rms_layernorm:
# https://github.com/thunlp/TritonBench - Apache License 2.0

#  Fast RMS LayerNorm: fused forward, backward, and Gemma-variant Triton kernels.
from __future__ import annotations
import math
import random
import numpy as np
import torch
import torch.nn as nn
import triton
import triton.language as tl


next_power_of_2 = triton.next_power_of_2
MAX_FUSED_SIZE: int = 65536


def calculate_settings(n: int) -> (int, int,):
    BLOCK_SIZE: int = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(f"Cannot launch Triton kernel since n = {n} exceeds "
                           f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}.")
    num_warps: int = 4
    if   BLOCK_SIZE >= 32768: num_warps = 16
    elif BLOCK_SIZE >=  8192: num_warps = 16
    elif BLOCK_SIZE >=  2048: num_warps = 8
    return BLOCK_SIZE, num_warps


@triton.jit
def _rms_layernorm_forward(
    Y, Y_row_stride,
    X, X_row_stride,
    W, W_row_stride,
    r, r_row_stride,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr
):
    """
        Fast RMS Layernorm kernel
        Inspiration from a Triton tutorial:
        https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y += row_idx * Y_row_stride
    X += row_idx * X_row_stride
    r += row_idx * r_row_stride

    X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask=mask, other=0)

    row_var = tl.sum(X_row * X_row, axis=0) / n_cols
    inv_var = tl.math.rsqrt(row_var + eps)
    tl.store(r, inv_var)
    normed = X_row * inv_var
    normed = normed.to(W_row.dtype)
    output = normed * W_row
    tl.store(Y + col_offsets, output, mask=mask)


@triton.heuristics({"GEMMA": lambda args: args["GEMMA"],})
@triton.jit
def _rms_layernorm_backward(
    dY, dY_row_stride,
    X, X_row_stride,
    W, W_row_stride,
    r, r_row_stride,
    dW, dW_row_stride,
    n_cols, eps,
    GEMMA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
        Fast RMS Layernorm kernel for the backward pass
        Inspiration from a Triton tutorial:
        https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dY += row_idx * dY_row_stride
    X  += row_idx *  X_row_stride
    r  += row_idx *  r_row_stride

    dY_row = tl.load(dY + col_offsets, mask=mask, other=0).to(tl.float32)
    X_row  = tl.load(X  + col_offsets, mask=mask, other=0).to(tl.float32)
    W_row  = tl.load(W  + col_offsets, mask=mask, other=0).to(tl.float32)

    inv_var = tl.load(r).to(tl.float32)
    normed = X_row * inv_var

    if GEMMA: dY_W = dY_row * (W_row + 1.0)
    else:     dY_W = dY_row * W_row

    rowsum_dY_normed = tl.sum(dY_W * normed, axis=0)
    output = inv_var/n_cols * (n_cols*dY_W - normed*rowsum_dY_normed)
    tl.store(dY + col_offsets, output, mask=mask)


@triton.jit
def _gemma_rms_layernorm_forward(
    Y, Y_row_stride,
    X, X_row_stride,
    W, W_row_stride,
    r, r_row_stride,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y += row_idx * Y_row_stride
    X += row_idx * X_row_stride
    r += row_idx * r_row_stride

    X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)

    row_var = tl.sum(X_row * X_row, axis=0) / n_cols
    inv_var = tl.math.rsqrt(row_var + eps)
    tl.store(r, inv_var)
    normed = X_row * inv_var
    output = normed * (W_row + 1.0)

    tl.store(Y + col_offsets, output, mask=mask)


class Fast_RMS_Layernorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, eps, gemma=False):
        shape = X.shape
        dim = shape[-1]
        X = X.view(-1, dim)
        n_rows, n_cols = X.shape
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)

        Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device="cuda:0")
        r = torch.empty(n_rows, dtype=torch.float32, device="cuda:0")

        fx = _gemma_rms_layernorm_forward if gemma else _rms_layernorm_forward
        fx[(n_rows,)](
            Y, Y.stride(0),
            X, X.stride(0),
            W, W.stride(0),
            r, r.stride(0),
            n_cols, eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.GEMMA = gemma
        ctx.save_for_backward(X, W, r)
        return Y.view(*shape)

    @staticmethod
    def backward(ctx, dY):
        shape = dY.shape
        dim = shape[-1]
        dY = dY.view(-1, dim)
        X, W, r = ctx.saved_tensors
        n_rows, n_cols = dY.shape
        dW = X

        _rms_layernorm_backward[(n_rows,)](
            dY, dY.stride(0),
            X,  X.stride(0),
            W,  W.stride(0),
            r,  r.stride(0),
            dW, dW.stride(0),
            n_cols, ctx.eps,
            GEMMA=ctx.GEMMA,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
            num_warps=ctx.num_warps,
        )
        dX = dY.view(*shape)
        return dX, None, None, None


def fast_rms_layernorm(layernorm, X, gemma=False):
    W = layernorm.weight
    eps = layernorm.variance_epsilon if \
        hasattr(layernorm, "variance_epsilon") \
        else layernorm.eps
    out = Fast_RMS_Layernorm.apply(X, W, eps, gemma)
    return out


class SimpleLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super(SimpleLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape).cuda())
        self.eps = eps


##################################################################################################################################################

# ============================================================================
# TEST CONFIGURATIONS
# ============================================================================

# (batch, seq_len, hidden_dim)
# Extracted from test_fast_rms_layernorm_with_backward() in the original eval:
#   test_case_1: X=(2,4,8), gemma=False  (forward + backward)
#   test_case_2: X=(2,4,8), gemma=True   (forward + backward)

ALL_SHAPES = [
    (2, 4, 8),
]

HARNESS_SHAPES = ALL_SHAPES[:25]
PROFILE_SHAPES = ALL_SHAPES[:5]

RTOL, ATOL = 1e-2, 1e-2

# For backward compatibility
EVAL_CONFIGS = HARNESS_SHAPES
PROFILE_CONFIGS = PROFILE_SHAPES


# ============================================================================
# TEST HARNESS
# ============================================================================


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_input(shape, seed=42):
    """Create input tensor and layernorm module with fixed seed."""
    set_seed(seed)
    hidden_dim = shape[-1]
    x = torch.randn(*shape, dtype=torch.float32, device='cuda', requires_grad=True)
    layernorm = SimpleLayerNorm(hidden_dim, eps=1e-5)
    return x, layernorm


def rms_layernorm_reference(x, weight, eps=1e-5):
    """PyTorch reference for standard RMS LayerNorm."""
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


def gemma_rms_layernorm_reference(x, weight, eps=1e-5):
    """PyTorch reference for Gemma-variant RMS LayerNorm."""
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * (weight + 1.0)


def run_correctness(shapes, verbose: bool = True) -> dict:
    """Run correctness tests matching the eval test cases exactly.

    Mirrors test_fast_rms_layernorm_with_backward():
      test_case_1: backward grad for gemma=False
      test_case_2: backward grad for gemma=True
    """
    if verbose:
        print(f"Running correctness on {len(shapes)} shapes...")

    results, failures = [], []
    for idx, shape in enumerate(shapes):
        try:
            x, layernorm = make_input(shape, seed=42 + idx)

            output = fast_rms_layernorm(layernorm, x, gemma=False)
            output.mean().backward()
            grad1 = x.grad.clone()
            x.grad.zero_()

            x_ref = x.detach().clone().requires_grad_(True)
            rms_layernorm_reference(x_ref, layernorm.weight, eps=1e-5).mean().backward()
            torch.testing.assert_close(grad1, x_ref.grad, rtol=RTOL, atol=ATOL)

            results.append({"config": shape, "variant": "gemma=False", "correct": True})
            if verbose:
                print(f"  PASS: {shape} gemma=False backward")

            output_g = fast_rms_layernorm(layernorm, x, gemma=True)
            output_g.mean().backward()
            grad2 = x.grad.clone()

            x_ref2 = x.detach().clone().requires_grad_(True)
            gemma_rms_layernorm_reference(x_ref2, layernorm.weight, eps=1e-5).mean().backward()
            torch.testing.assert_close(grad2, x_ref2.grad, rtol=RTOL, atol=ATOL)

            results.append({"config": shape, "variant": "gemma=True", "correct": True})
            if verbose:
                print(f"  PASS: {shape} gemma=True backward")

            del x, layernorm, x_ref, x_ref2
            torch.cuda.empty_cache()
        except Exception as e:
            failures.append({"config": shape, "error": str(e)})
            if verbose:
                print(f"  FAIL: {shape} - {str(e)[:80]}")

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


def run_profile(shapes, warmup: int = 50, iters: int = 200, verbose: bool = True):
    """Run kernel for profiling with proper warmup."""
    if verbose:
        print(f"Profile: {len(shapes)} config(s), {warmup} warmup, {iters} iter(s)")

    for shape in shapes:
        x, layernorm = make_input(shape, seed=42)
        x_bench = x.detach().clone()

        for _ in range(warmup):
            fast_rms_layernorm(layernorm, x_bench, gemma=False)
        torch.cuda.synchronize()

        for _ in range(iters):
            fast_rms_layernorm(layernorm, x_bench, gemma=False)
        torch.cuda.synchronize()

        if verbose:
            print(f"  {shape} done")
        del x, x_bench, layernorm
        torch.cuda.empty_cache()


def run_benchmark(shapes, warmup: int = 50, iters: int = 200, verbose: bool = True) -> dict:
    """Benchmark kernel vs reference; report per-shape speedups and geo-mean."""
    print(
        f"Running benchmark on {len(shapes)} shapes, {warmup} warmup, {iters} iterations each..."
    )
    latencies = []
    speedups = []
    results = []

    if verbose:
        print(
            f"{'Config':<22} {'Reference':>10} {'Kernel':>10} {'Speedup':>10}"
        )
        print("-" * 62)

    for idx, shape in enumerate(shapes):
        x, layernorm = make_input(shape, seed=42 + idx)
        x_bench = x.detach().clone()

        for _ in range(warmup):
            fast_rms_layernorm(layernorm, x_bench, gemma=False)
        torch.cuda.synchronize()

        triton_times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fast_rms_layernorm(layernorm, x_bench, gemma=False)
            end.record()
            torch.cuda.synchronize()
            triton_times.append(start.elapsed_time(end))

        kernel_ms = sorted(triton_times)[len(triton_times) // 2]

        ref_times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            rms_layernorm_reference(x_bench, layernorm.weight, eps=1e-5)
            end.record()
            torch.cuda.synchronize()
            ref_times.append(start.elapsed_time(end))

        ref_ms = sorted(ref_times)[len(ref_times) // 2]

        speedup = ref_ms / kernel_ms if kernel_ms > 0 else float('inf')
        speedups.append(speedup)
        latencies.append(kernel_ms)

        results.append({
            "config": shape,
            "ref_ms": ref_ms,
            "kernel_ms": kernel_ms,
            "speedup": speedup,
        })

        if verbose:
            marker = " *" if speedup > 1.0 else ""
            print(
                f"{str(shape):<22} {ref_ms:>8.4f}ms {kernel_ms:>8.4f}ms {speedup:>8.2f}x{marker}"
            )

        del x, x_bench, layernorm
        torch.cuda.empty_cache()

    log_sum = sum(math.log(t) for t in latencies)
    geomean_latency = math.exp(log_sum / len(latencies))

    log_sum_speedup = sum(math.log(s) for s in speedups)
    geomean_speedup = math.exp(log_sum_speedup / len(speedups))

    if verbose:
        print("-" * 62)
        print(f"{'Geometric mean latency:':<22} {geomean_latency:.4f} ms")
        print(f"{'Geometric mean speedup:':<22} {geomean_speedup:.2f}x")
        print(f"GEAK_RESULT_LATENCY_MS={geomean_latency:.4f}")
        print(f"GEAK_RESULT_SPEEDUP={geomean_speedup:.2f}")

    return {
        "geomean_latency_ms": geomean_latency,
        "geomean_speedup": geomean_speedup,
        "results": results,
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fast RMS LayerNorm Kernel Test Harness")
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
        help="Run benchmark on HARNESS_SHAPES",
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
        default=200,
        help="Number of benchmark iterations (default: 200)",
    )
    args = parser.parse_args()

    print("=" * 62)
    print("Fast RMS LayerNorm Kernel Test Harness")
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
        # Default: benchmark (harness shapes)
        print("\n[Benchmark Mode]")
        run_benchmark(HARNESS_SHAPES, warmup=args.warmup, iters=args.iterations)

    print("=" * 62)
