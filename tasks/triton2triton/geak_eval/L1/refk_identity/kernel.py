#!/usr/bin/env python3
"""
Identity Kernel — Triton implementation extracted via torch.compile(backend='inductor').

Copies an input tensor to an output tensor element-wise.
Triton kernel generated from PyTorch's `output.copy_(input)` on float16 1-D tensors.
"""

import math
import os
import time

import torch
import triton
import triton.language as tl


# ============================================================================
# TRITON KERNEL — extracted from torch.compile inductor output
# ============================================================================


@triton.autotune(
    configs=[
        triton.Config({"XBLOCK": 128}, num_warps=2),
        triton.Config({"XBLOCK": 256}, num_warps=4),
        triton.Config({"XBLOCK": 512}, num_warps=4),
        triton.Config({"XBLOCK": 1024}, num_warps=8),
    ],
    key=["xnumel"],
)
@triton.jit
def _identity_kernel(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    tmp0 = tl.load(in_ptr0 + xindex, xmask).to(tl.float32)
    tl.store(out_ptr0 + xindex, tmp0, xmask)


# ============================================================================
# PYTHON WRAPPER
# ============================================================================


def identity_triton(input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> torch.Tensor:
    xnumel = input_tensor.numel()
    grid = lambda meta: (triton.cdiv(xnumel, meta["XBLOCK"]),)
    _identity_kernel[grid](input_tensor, output_tensor, xnumel)
    return output_tensor


# ============================================================================
# REFERENCE IMPLEMENTATION (pure PyTorch)
# ============================================================================


def identity_pytorch(input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> torch.Tensor:
    output_tensor[...] = input_tensor
    return output_tensor


# ============================================================================
# ENTRY POINTS (for GEAK harness)
# ============================================================================


def triton_op(size, seed):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    data = torch.empty(size, device="cuda", dtype=torch.float16)
    data.uniform_(0, 1, generator=gen)
    output = torch.empty_like(data)
    return identity_triton(data, output)


def torch_op(size, seed):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    data = torch.empty(size, device="cuda", dtype=torch.float16)
    data.uniform_(0, 1, generator=gen)
    output = torch.empty_like(data)
    return identity_pytorch(data, output)


# ============================================================================
# SYNTHETIC INPUT BUILDER
# ============================================================================


def get_inputs(size, seed=42, device="cuda"):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    data = torch.empty(size, device=device, dtype=torch.float16)
    data.uniform_(0, 1, generator=gen)
    output = torch.empty_like(data)
    return data, output


# ============================================================================
# CONFIG SPACE — matches test_submission_harness.py ALL_CONFIGS
# ============================================================================


EVAL_CONFIGS = [
    # tests from task.yml
    {"size": 127, "seed": 4242},
    {"size": 128, "seed": 5236},
    {"size": 129, "seed": 1001},
    {"size": 256, "seed": 5531},
    {"size": 512, "seed": 9173},
    # benchmarks from task.yml
    {"size": 1024, "seed": 54352},
    {"size": 2048, "seed": 93246},
    {"size": 4096, "seed": 6256},
    {"size": 8192, "seed": 8841},
    {"size": 16384, "seed": 6252},
    {"size": 32768, "seed": 52624},
    {"size": 65536, "seed": 125432},
]

PROFILE_CONFIGS = [
    {"size": 1024, "seed": 54352},
    {"size": 8192, "seed": 8841},
    {"size": 65536, "seed": 125432},
]

WARMUP = 50
ITERATIONS = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200"))
RTOL, ATOL = 1e-5, 1e-5


# ============================================================================
# SELF-TEST HARNESS
# ============================================================================


def check_correctness(cfg) -> dict:
    try:
        data, out_triton = get_inputs(**cfg)
        out_ref = torch.empty_like(data)
        identity_triton(data, out_triton)
        identity_pytorch(data, out_ref)
        torch.cuda.synchronize()
        correct = torch.equal(out_triton, out_ref)
        max_diff = torch.max(torch.abs(out_triton.float() - out_ref.float())).item()
        return {"correct": correct, "max_diff": max_diff, "error": None}
    except Exception as e:
        return {"correct": False, "max_diff": float("inf"), "error": str(e)}


def benchmark_config(cfg, warmup=WARMUP, iters=ITERATIONS) -> dict:
    data, output = get_inputs(**cfg)
    for _ in range(warmup):
        identity_triton(data, output)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        identity_triton(data, output)
    torch.cuda.synchronize()
    triton_ms = (time.perf_counter() - start) * 1000 / iters

    output2 = torch.empty_like(data)
    for _ in range(warmup):
        identity_pytorch(data, output2)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        identity_pytorch(data, output2)
    torch.cuda.synchronize()
    torch_ms = (time.perf_counter() - start) * 1000 / iters

    return {"triton_ms": triton_ms, "torch_ms": torch_ms,
            "speedup": torch_ms / triton_ms if triton_ms > 0 else 0.0}


def _config_label(cfg):
    return f"(size={cfg['size']})"


def evaluate(configs=None, warmup=WARMUP, iters=ITERATIONS, verbose=True) -> dict:
    configs = configs or EVAL_CONFIGS
    results, failures = [], []

    if verbose:
        print(f"{'Config':<22} {'Correct':>8} {'Torch':>10} {'Triton':>10} {'Speedup':>10}")
        print("-" * 62)

    for cfg in configs:
        label = _config_label(cfg)
        corr = check_correctness(cfg)
        if not corr["correct"]:
            failures.append({"config": cfg, **corr})
            if verbose:
                err = corr["error"] or f"max_diff={corr['max_diff']:.2e}"
                print(f"{label:<22} {'FAIL':>8}   {err[:30]}")
            continue

        bench = benchmark_config(cfg, warmup=warmup, iters=iters)
        results.append({"config": cfg, "correct": True, **bench})

        if verbose:
            marker = " *" if bench["speedup"] > 1.0 else ""
            print(
                f"{label:<22} {'PASS':>8} "
                f"{bench['torch_ms']:>8.4f}ms {bench['triton_ms']:>8.4f}ms "
                f"{bench['speedup']:>8.2f}x{marker}"
            )

    speedups = [r["speedup"] for r in results]
    geomean = math.prod(speedups) ** (1 / len(speedups)) if speedups else 0.0

    if verbose:
        print("-" * 62)
        status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{len(configs)})"
        print(f"{'Status:':<22} {status}")
        if speedups:
            print(f"{'Speedup (geomean):':<22} {geomean:.2f}x")

    return {
        "correct": len(failures) == 0,
        "num_correct": len(results),
        "num_failed": len(failures),
        "failures": failures,
        "results": results,
        "speedup_geomean": geomean,
    }


def run_profile(configs=None, warmup=5, iters=1, verbose=True):
    configs = configs or PROFILE_CONFIGS
    if verbose:
        print(f"Profile: {len(configs)} config(s)")
    for cfg in configs:
        data, output = get_inputs(**cfg)
        for _ in range(warmup):
            identity_triton(data, output)
        torch.cuda.synchronize()
        for _ in range(iters):
            identity_triton(data, output)
        torch.cuda.synchronize()
        if verbose:
            print(f"  {_config_label(cfg)} done")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Identity Kernel (Triton)")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    print("=" * 62)
    print("Identity Kernel — Triton (torch.compile extracted)")
    print("=" * 62)

    if args.profile:
        print("\n[Profile Mode]")
        run_profile()
    else:
        print("\n[Evaluation]")
        evaluate()

    print("=" * 62)
