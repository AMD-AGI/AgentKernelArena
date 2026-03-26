#!/usr/bin/env python3
"""
GEMM A16W16 Atomic Kernel

GEMM with atomic K-splitting for small-M shapes. Uses Triton kernel with
NUM_KSPLIT>1 and atomic accumulation for improved parallelism.
"""

from typing import Optional
import functools
import os
import json
import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ============================================================================
# ARCH INFO (from aiter.ops.triton.utils._triton.arch_info)
# ============================================================================

AITER_TRITON_CONFIGS_PATH = "/sgl-workspace/aiter/aiter/ops/triton/configs"


@functools.lru_cache(maxsize=1)
def get_arch():
    try:
        arch = triton.runtime.driver.active.get_current_target().arch
    except RuntimeError:
        from jax._src.lib import gpu_triton as triton_kernel_call_lib

        arch = triton_kernel_call_lib.get_arch_details("0")
        arch = arch.split(":")[0]
    return arch


# ============================================================================
# KERNEL REPR (from aiter.ops.triton.utils._triton.kernel_repr)
# ============================================================================


def _sanitize_constexpr_value(value):
    if value is None:
        return "NONE"
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value)
    if isinstance(value, (list, tuple, set)):
        items = sorted(value, key=str) if isinstance(value, set) else value
        sanitized_items = [_sanitize_constexpr_value(item) for item in items]
        joined = "_".join(sanitized_items)
        return joined if joined else "NONE"
    if isinstance(value, str):
        cleaned_value = "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")
        return cleaned_value.upper() if cleaned_value else "NONE"
    cleaned_value = "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")
    return cleaned_value.upper() if cleaned_value else "NONE"


def make_kernel_repr(base_name, config_keys):
    def _repr(specialization):
        constants = specialization.constants
        name_parts = []
        for key in config_keys:
            value = constants.get(key, None)
            symbol = _sanitize_constexpr_value(value)
            name_parts.append(f"{key}_{symbol}")
        if not name_parts:
            return base_name
        suffix = "_".join(name_parts)
        return f"{base_name}_{suffix}"
    return _repr


# ============================================================================
# PID PREPROCESSING (from aiter.ops.triton.utils._triton.pid_preprocessing)
# ============================================================================


@triton.jit
def remap_xcd(pid, GRID_MN, NUM_XCDS: tl.constexpr = 8):
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = (
            tall_xcds * pids_per_xcd
            + (xcd - tall_xcds) * (pids_per_xcd - 1)
            + local_pid
        )
    return pid


@triton.jit
def pid_grid(pid: int, num_pid_m: int, num_pid_n: int, GROUP_SIZE_M: tl.constexpr = 1):
    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


# ============================================================================
# TRITON KERNEL (from aiter.ops.triton._triton_kernels.gemm_a16w16_atomic)
# ============================================================================

_gemm_a16w16_atomic_repr = make_kernel_repr(
    "_gemm_a16_w16_atomic_kernel",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "NUM_KSPLIT",
        "SPLITK_BLOCK_SIZE",
        "cache_modifier",
        "EVEN_K",
        "GRID_MN",
    ],
)


@triton.heuristics(
    {
        "EVEN_K": lambda args: (args["K"] % (args["BLOCK_SIZE_K"]) == 0)
        and (args["SPLITK_BLOCK_SIZE"] % args["BLOCK_SIZE_K"] == 0)
        and (args["K"] % (args["SPLITK_BLOCK_SIZE"]) == 0),
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@triton.jit(repr=_gemm_a16w16_atomic_repr)
def _gemm_a16_w16_atomic_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    cache_modifier: tl.constexpr,
    EVEN_K: tl.constexpr,
    GRID_MN: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    pid_unified = tl.program_id(axis=0)
    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    if NUM_KSPLIT == 1:
        pid = remap_xcd(pid, GRID_MN)
        pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(pid_k >= 0)

    if (pid_k * SPLITK_BLOCK_SIZE) < K:
        num_k_iter = tl.cdiv(SPLITK_BLOCK_SIZE, BLOCK_SIZE_K)

        offs_k = tl.arange(0, BLOCK_SIZE_K)
        offs_k_split = pid_k * (SPLITK_BLOCK_SIZE) + offs_k
        offs_am = (pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        a_ptrs = a_ptr + (
            offs_am[:, None] * stride_am + offs_k_split[None, :] * stride_ak
        )
        b_ptrs = b_ptr + (
            offs_k_split[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        )

        acc_dtype = tl.float32 if c_ptr.type.element_ty != tl.int8 else tl.int32
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        for k in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter):
            if EVEN_K:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs, cache_modifier=cache_modifier)
            else:
                a = tl.load(
                    a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0
                )
                b = tl.load(
                    b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0
                )

            accumulator += tl.dot(a, b, input_precision="ieee")

            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        c = accumulator.to(c_ptr.type.element_ty)

        offs_cm = pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        if NUM_KSPLIT == 1:
            tl.store(c_ptrs, c, mask=c_mask)
        else:
            tl.atomic_add(c_ptrs, c, mask=c_mask, sem="relaxed")


# ============================================================================
# CONFIG LOOKUP (from aiter.ops.triton._triton_kernels.gemm_a16w16_atomic)
# ============================================================================


@functools.lru_cache(maxsize=1024)
def _get_config(M: int, N: int, K: int):
    if not hasattr(_get_config, "_config_dict"):
        dev = get_arch()
        _get_config._config_dict = {}
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-GEMM-A16W16-ATOMIC.json"
        with open(fpath, "r") as file:
            config = json.load(file)
        _get_config._config_dict["default"] = config

    key = f"{N}_{K}"
    if key not in _get_config._config_dict.keys():
        dev = get_arch()
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-GEMM-A16W16-ATOMIC-N={N}-K={K}.json"
        if os.path.exists(fpath):
            with open(fpath, "r") as file:
                config = json.load(file)
                _get_config._config_dict[key] = config
        else:
            key = "default"
            return _get_config._config_dict[key]["any"]
    if M < 32:
        return _get_config._config_dict[key]["small"]
    elif M <= 128:
        BLK_M = triton.next_power_of_2(M)
        if BLK_M == 32:
            return _get_config._config_dict[key]["medium_M32"]
        elif BLK_M == 64:
            return _get_config._config_dict[key]["medium_M64"]
        elif BLK_M == 128:
            return _get_config._config_dict[key]["medium_M128"]
    elif M <= 256:
        return _get_config._config_dict[key]["large"]
    else:
        return _get_config._config_dict[key]["xlarge"]


# ============================================================================
# GEMM WRAPPER (from aiter.ops.triton.gemm.basic.gemm_a16w16_atomic)
# ============================================================================


def gemm_a16w16_atomic(
    x,
    w,
    dtype: Optional[float] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
):
    """
    Computes 16 bit matrix multiplication Y = X @ W^T using atomic operations for split-K reduction.
    """
    w = w.T

    M, K = x.shape
    K, N = w.shape

    if config is None:
        config = _get_config(M, N, K)
    if "NUM_KSPLIT" not in config:
        config["NUM_KSPLIT"] = 1
    if "cache_modifier" not in config:
        config["cache_modifier"] = ""

    if y is None:
        if config["NUM_KSPLIT"] == 1:
            y = torch.empty((M, N), dtype=dtype, device=x.device)
        else:
            y = torch.zeros((M, N), dtype=dtype, device=x.device)

    grid = lambda META: (  # noqa: E731
        triton.cdiv(M, META["BLOCK_SIZE_M"])
        * triton.cdiv(N, META["BLOCK_SIZE_N"])
        * META["NUM_KSPLIT"],
    )
    SPLITK_BLOCK_SIZE = triton.cdiv(K, config["NUM_KSPLIT"])
    config["SPLITK_BLOCK_SIZE"] = SPLITK_BLOCK_SIZE
    _gemm_a16_w16_atomic_kernel[grid](
        x,
        w,
        y,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        y.stride(0),
        y.stride(1),
        **config,
    )

    return y

# ============================================================================
# ENTRY POINTS
# ============================================================================


def triton_op(x, w):
    return gemm_a16w16_atomic(x, w, dtype=torch.float32).to(x.dtype)


def torch_op(x, w):
    """Reference: standard matmul via F.linear (w is NxK, computes x @ w^T)."""
    return F.linear(x, w, bias=None)


# ============================================================================
# TEST CONFIGURATIONS (from GEAK harness test discovery)
# ============================================================================

# (M, N, K)
EVAL_CONFIGS = [
    (1, 1, 1),
    (1, 8192, 1024),
    (32, 256, 7168),
    (64, 256, 7168),
    (32, 8192, 1024),
    (256, 256, 7168),
    (64, 8192, 1024),
    (1024, 1024, 1024),
    (128, 1280, 8192),
    (192, 1280, 8192),
    (256, 1280, 8192),
    (320, 8192, 1024),
    (512, 8192, 1024),
    (2048, 2048, 2048),
    (1024, 8192, 1024),
    (2048, 8192, 1024),
    (3072, 3072, 3072),
    (4096, 1280, 8192),
    (8192, 8192, 1024),
    (8192, 1280, 8192),
    (16384, 8192, 1024),
    (4864, 8192, 4160),
    (16384, 1280, 8192),
    (7168, 7168, 7168),
    (9728, 8192, 65536),
]

PROFILE_CONFIGS = [
    (1, 1, 1),
    (64, 8192, 1024),
    (512, 8192, 1024),
    (8192, 8192, 1024),
    (9728, 8192, 65536),
]

RTOL, ATOL = 1e-1, 1e-1


# ============================================================================
# TEST HARNESS
# ============================================================================


def get_inputs(M, N, K, dtype=torch.bfloat16, device="cuda"):
    x = torch.randn(M, K, dtype=dtype, device=device)
    w = torch.randn(N, K, dtype=dtype, device=device)
    return x, w


def check_correctness(M, N, K) -> dict:
    try:
        x, w = get_inputs(M, N, K)
        res = triton_op(x, w)
        ref = torch_op(x, w)
        correct = torch.allclose(res, ref, rtol=RTOL, atol=ATOL)
        max_diff = torch.max(torch.abs(res - ref)).item() if not correct else 0.0
        return {"correct": correct, "max_diff": max_diff, "error": None}
    except Exception as e:
        return {"correct": False, "max_diff": float("inf"), "error": str(e)}


BASELINE_LATENCIES = {
    (1, 1, 1): 0.0296,
    (1, 8192, 1024): 0.0304,
    (32, 256, 7168): 0.0345,
    (64, 256, 7168): 0.0346,
    (32, 8192, 1024): 0.0303,
    (256, 256, 7168): 0.0347,
    (64, 8192, 1024): 0.0309,
    (1024, 1024, 1024): 0.0366,
    (128, 1280, 8192): 0.1929,
    (192, 1280, 8192): 0.1943,
    (256, 1280, 8192): 0.1965,
    (320, 8192, 1024): 0.04,
    (512, 8192, 1024): 0.0411,
    (2048, 2048, 2048): 0.0632,
    (1024, 8192, 1024): 0.0446,
    (2048, 8192, 1024): 0.0591,
    (3072, 3072, 3072): 0.0938,
    (4096, 1280, 8192): 0.2038,
    (8192, 8192, 1024): 0.2681,
    (8192, 1280, 8192): 0.2654,
    (16384, 8192, 1024): 0.5398,
    (4864, 8192, 4160): 0.4417,
    (16384, 1280, 8192): 0.4478,
    (7168, 7168, 7168): 0.8805,
    (9728, 8192, 65536): 11.011,
}


def benchmark_config(M, N, K, warmup=500, iters=2000) -> dict:
    import time

    cfg_key = (M, N, K)
    x, w = get_inputs(M, N, K)

    for _ in range(warmup):
        triton_op(x, w)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        triton_op(x, w)
    torch.cuda.synchronize()
    triton_ms = (time.perf_counter() - start) * 1000 / iters

    baseline_ms = BASELINE_LATENCIES.get(cfg_key, triton_ms)
    return {"torch_ms": baseline_ms, "triton_ms": triton_ms, "speedup": baseline_ms / triton_ms if triton_ms > 0 else 1.0}


def evaluate(configs=None, warmup=500, iters=2000, verbose=True) -> dict:
    configs = configs or EVAL_CONFIGS
    results, failures = [], []

    if verbose:
        print(f"{'Config (M,N,K)':<25} {'Correct':>8} {'Torch':>10} {'Triton':>10} {'Speedup':>10}")
        print("-" * 65)

    for cfg in configs:
        M, N, K = cfg
        corr = check_correctness(M, N, K)
        if not corr["correct"]:
            failures.append({"config": cfg, **corr})
            if verbose:
                err = corr["error"] or f"max_diff={corr['max_diff']:.2e}"
                print(f"({M},{N},{K}){'':<10} {'FAIL':>8}   {err[:25]}")
            continue

        bench = benchmark_config(M, N, K, warmup, iters)
        results.append({"config": cfg, "correct": True, **bench})
        if verbose:
            marker = " *" if bench["speedup"] > 1.0 else ""
            print(f"({M},{N},{K}){'':<10} {'PASS':>8} {bench['torch_ms']:>8.4f}ms {bench['triton_ms']:>8.4f}ms {bench['speedup']:>8.2f}x{marker}")

    total_baseline = sum(r["torch_ms"] for r in results)
    total_evolved = sum(r["triton_ms"] for r in results)
    speedup = total_baseline / total_evolved if total_evolved > 0 else 0.0

    if verbose:
        print("-" * 65)
        print(f"{'Status:':<25} {'ALL PASS' if not failures else f'FAILED ({len(failures)}/{len(configs)})'}")
        if results:
            print(f"{'Speedup (total):':<25} {speedup:.2f}x")

    return {
        "correct": len(failures) == 0,
        "num_correct": len(results),
        "num_failed": len(failures),
        "failures": failures,
        "results": results,
        "speedup_geomean": speedup,
    }


def run_profile(configs=None, warmup=3, iters=1, verbose=True):
    configs = configs or PROFILE_CONFIGS
    if verbose:
        print(f"Profile: {len(configs)} config(s), {warmup} warmup, {iters} iter(s)")
    for M, N, K in configs:
        x, w = get_inputs(M, N, K)
        for _ in range(warmup):
            triton_op(x, w)
        torch.cuda.synchronize()
        for _ in range(iters):
            triton_op(x, w)
        torch.cuda.synchronize()
        if verbose:
            print(f"  ({M},{N},{K}) done")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GEMM A16W16 Atomic Kernel Test Harness")
    parser.add_argument("--profile", action="store_true", help="Run minimal profiling workload")
    args = parser.parse_args()

    print("=" * 65)
    print("GEMM A16W16 Atomic Kernel")
    print("=" * 65)

    if args.profile:
        print("\n[Profile Mode]")
        run_profile()
    else:
        print("\n[Evaluation]")
        evaluate()

    print("=" * 65)
