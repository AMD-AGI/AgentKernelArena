#!/usr/bin/env python3
"""
FP8 Block-Scale GEMM Kernel — Triton dequantization extracted via torch.compile.

The full pipeline:
  1. Dequantize A (FP8 -> FP32 with per-row block scales)    [Triton]
  2. Dequantize B (FP8 -> FP32 with 2-D block scales)        [Triton]
  3. Matmul A_deq @ B_deq.T                                  [torch.mm]
  4. Cast FP32 -> BF16 and write to output                   [Triton]

Input layout (from reference-kernels/problems/amd/fp8-mm):
  a:       [m, k]           float8_e4m3fnuz, column-major stored
  b:       [n, k]           float8_e4m3fnuz, column-major stored
  a_scale: [m, k // 128]    float32
  b_scale: [n // 128, k // 128] float32
  c:       [m, n]           bfloat16 (pre-allocated output)
"""

import math
import os
import time

import torch
import triton
import triton.language as tl


BLOCK_SHAPE_N = 128
BLOCK_SHAPE_K = 128
BLOCK_SHAPE_N_CONSTEXPR = tl.constexpr(128)
BLOCK_SHAPE_K_CONSTEXPR = tl.constexpr(128)


# ============================================================================
# TRITON KERNEL 1: Dequantize A — fused cast + per-row-block scale multiply
# ============================================================================


@triton.autotune(
    configs=[
        triton.Config({"XBLOCK": 256}, num_warps=4),
        triton.Config({"XBLOCK": 512}, num_warps=4),
        triton.Config({"XBLOCK": 1024}, num_warps=8),
    ],
    key=["xnumel", "scale_k"],
)
@triton.jit
def _dequant_a_kernel(
    a_ptr,        # [m, k] fp8, contiguous
    a_scale_ptr,  # [m, scale_k] fp32, contiguous
    out_ptr,      # [m, k] fp32, contiguous
    xnumel,       # m * k
    k: tl.constexpr,
    scale_k: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    row = xindex // k
    col = xindex % k
    scale_col = col // BLOCK_SHAPE_K_CONSTEXPR
    scale_col = tl.where(scale_col < scale_k, scale_col, scale_k - 1)
    a_val = tl.load(a_ptr + xindex, xmask).to(tl.float32)
    s_val = tl.load(a_scale_ptr + row * scale_k + scale_col, xmask)
    tl.store(out_ptr + xindex, a_val * s_val, xmask)


# ============================================================================
# TRITON KERNEL 2: Dequantize B — fused cast + 2D block scale with permute
# ============================================================================


@triton.autotune(
    configs=[
        triton.Config({"XBLOCK": 32, "YBLOCK": 32}, num_warps=4),
        triton.Config({"XBLOCK": 64, "YBLOCK": 16}, num_warps=4),
        triton.Config({"XBLOCK": 128, "YBLOCK": 8}, num_warps=4),
    ],
    key=["n", "k"],
)
@triton.jit
def _dequant_b_kernel(
    b_ptr,        # [n, k] fp8, col-major (stride: [1, n])
    b_scale_ptr,  # [scale_n, scale_k] fp32, contiguous
    out_ptr,      # [n, k] fp32, contiguous (row-major)
    n,
    k,
    b_stride_row,
    b_stride_col,
    scale_n: tl.constexpr,
    scale_k: tl.constexpr,
    XBLOCK: tl.constexpr,
    YBLOCK: tl.constexpr,
):
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < n
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < k

    row = yindex  # n dimension
    col = xindex  # k dimension
    sn = row // BLOCK_SHAPE_N_CONSTEXPR
    sn = tl.where(sn < scale_n, sn, scale_n - 1)
    sk = col // BLOCK_SHAPE_K_CONSTEXPR
    sk = tl.where(sk < scale_k, sk, scale_k - 1)

    b_val = tl.load(b_ptr + row * b_stride_row + col * b_stride_col,
                    ymask & xmask).to(tl.float32)
    s_val = tl.load(b_scale_ptr + sn * scale_k + sk, ymask & xmask)
    tl.store(out_ptr + row * k + col, b_val * s_val, ymask & xmask)


# ============================================================================
# TRITON KERNEL 3: Cast FP32 -> BF16 into output
# ============================================================================


@triton.autotune(
    configs=[
        triton.Config({"XBLOCK": 256}, num_warps=4),
        triton.Config({"XBLOCK": 512}, num_warps=4),
        triton.Config({"XBLOCK": 1024}, num_warps=8),
    ],
    key=["xnumel"],
)
@triton.jit
def _cast_to_bf16_kernel(in_ptr, out_ptr, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    tmp0 = tl.load(in_ptr + xindex, xmask)
    tl.store(out_ptr + xindex, tmp0.to(tl.bfloat16), xmask)


# ============================================================================
# PYTHON WRAPPER — full pipeline
# ============================================================================


def fp8_blockwise_mm_triton(a, b, a_scale, b_scale, c):
    a_cont = a.contiguous()
    a_scale_c = a_scale.contiguous()
    b_scale_c = b_scale.contiguous()

    m, k = a_cont.shape
    n = b.shape[0]
    scale_n = b_scale_c.shape[0]
    scale_k_a = a_scale_c.shape[1]
    scale_k_b = b_scale_c.shape[1]

    # 1. Dequantize A
    a_deq = torch.empty((m, k), dtype=torch.float32, device=a.device)
    grid_a = lambda meta: (triton.cdiv(m * k, meta["XBLOCK"]),)
    _dequant_a_kernel[grid_a](a_cont, a_scale_c, a_deq, m * k,
                              k=k, scale_k=scale_k_a)

    # 2. Dequantize B
    b_deq = torch.empty((n, k), dtype=torch.float32, device=b.device)
    grid_b = lambda meta: (triton.cdiv(k, meta["XBLOCK"]),
                           triton.cdiv(n, meta["YBLOCK"]))
    _dequant_b_kernel[grid_b](b, b_scale_c, b_deq, n, k,
                              b.stride(0), b.stride(1),
                              scale_n=scale_n, scale_k=scale_k_b)

    # 3. Matmul (via torch.mm / hipBLAS)
    result_f32 = torch.mm(a_deq, b_deq.T)

    # 4. Cast to BF16 into output
    mn = m * n
    grid_c = lambda meta: (triton.cdiv(mn, meta["XBLOCK"]),)
    _cast_to_bf16_kernel[grid_c](result_f32.view(-1), c.view(-1), mn)

    return c


# ============================================================================
# REFERENCE IMPLEMENTATION (pure PyTorch — same as submission.py)
# ============================================================================


def fp8_blockwise_mm_pytorch(a, b, a_scale, b_scale, c):
    a_c = a.contiguous()
    a_s = a_scale.contiguous()
    b_s = b_scale.contiguous()

    m, k = a_c.shape
    n = b.shape[0]
    block_n, block_k = BLOCK_SHAPE_N, BLOCK_SHAPE_K
    sn = b_s.shape[0]
    sk = b_s.shape[1]

    a_sc = a_s.unsqueeze(-1).repeat(1, 1, block_k).reshape(m, sk * block_k)[:, :k]
    a_deq = a_c.to(a_sc.dtype) * a_sc

    b_sc = (b_s.view(-1, 1).repeat(1, block_n * block_k)
            .view(sn, sk, block_n, block_k)
            .permute(0, 2, 1, 3)
            .reshape(sn * block_n, sk * block_k))[:n, :k]
    b_deq = b.to(b_sc.dtype) * b_sc

    c[...] = (a_deq @ b_deq.T).to(torch.bfloat16)
    return c


# ============================================================================
# ENTRY POINTS (for GEAK harness)
# ============================================================================


def triton_op(m, n, k, seed):
    data = _generate_input(m, n, k, seed)
    return fp8_blockwise_mm_triton(*data)


def torch_op(m, n, k, seed):
    data = _generate_input(m, n, k, seed)
    return fp8_blockwise_mm_pytorch(*data)


# ============================================================================
# SYNTHETIC INPUT BUILDER (matches reference.py generate_input)
# ============================================================================


def _generate_input(m, n, k, seed, device="cuda"):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    block_n, block_k = BLOCK_SHAPE_N, BLOCK_SHAPE_K
    scale_n = (n + block_n - 1) // block_n
    scale_k = (k + block_k - 1) // block_k

    a = torch.randn((k, m), dtype=torch.bfloat16, device=device, generator=gen).to(
        torch.float8_e4m3fnuz
    )
    b = torch.randn((k, n), dtype=torch.bfloat16, device=device, generator=gen).to(
        torch.float8_e4m3fnuz
    )
    a_scale = torch.randn([scale_k, m], dtype=torch.float32, device=device, generator=gen)
    b_scale = torch.randn([scale_k, scale_n], dtype=torch.float32, device=device, generator=gen)
    c = torch.zeros((m, n), dtype=torch.bfloat16, device=device)
    return (a.T, b.T, a_scale.T, b_scale.T, c)


def get_inputs(m, n, k, seed=42, device="cuda"):
    return _generate_input(m, n, k, seed, device)


# ============================================================================
# CONFIG SPACE — matches test_submission_harness.py
# ============================================================================


TEST_CONFIGS = [
    {"m": 64, "n": 64, "k": 128, "seed": 6635},
    {"m": 64, "n": 1536, "k": 7168, "seed": 6635},
    {"m": 64, "n": 3072, "k": 1536, "seed": 1236},
    {"m": 64, "n": 576, "k": 7168, "seed": 542},
    {"m": 96, "n": 7168, "k": 256, "seed": 1234},
    {"m": 96, "n": 7168, "k": 2048, "seed": 4153},
    {"m": 96, "n": 4608, "k": 7168, "seed": 412},
    {"m": 128, "n": 7168, "k": 2304, "seed": 624},
    {"m": 128, "n": 512, "k": 7168, "seed": 2514},
    {"m": 512, "n": 4096, "k": 512, "seed": 543},
    {"m": 512, "n": 1536, "k": 7168, "seed": 12341},
]

BENCHMARK_CONFIGS = [
    {"m": 1024, "n": 1536, "k": 7168, "seed": 8135},
    {"m": 1024, "n": 3072, "k": 1536, "seed": 6251},
    {"m": 1024, "n": 576, "k": 7168, "seed": 12346},
    {"m": 1024, "n": 7168, "k": 256, "seed": 5364},
    {"m": 1024, "n": 7168, "k": 2048, "seed": 6132},
    {"m": 1024, "n": 4608, "k": 7168, "seed": 7531},
    {"m": 1024, "n": 7168, "k": 2304, "seed": 12345},
    {"m": 1024, "n": 512, "k": 7168, "seed": 6563},
    {"m": 1024, "n": 4096, "k": 512, "seed": 17512},
    {"m": 6144, "n": 1536, "k": 7168, "seed": 6543},
    {"m": 6144, "n": 3072, "k": 1536, "seed": 234},
    {"m": 6144, "n": 576, "k": 7168, "seed": 9863},
    {"m": 6144, "n": 7168, "k": 256, "seed": 764243},
    {"m": 6144, "n": 7168, "k": 2048, "seed": 76547},
    {"m": 6144, "n": 4608, "k": 7168, "seed": 65436},
    {"m": 6144, "n": 7168, "k": 2304, "seed": 452345},
    {"m": 6144, "n": 512, "k": 7168, "seed": 12341},
    {"m": 6144, "n": 4096, "k": 512, "seed": 45245},
]

EVAL_CONFIGS = TEST_CONFIGS + BENCHMARK_CONFIGS

PROFILE_CONFIGS = [
    {"m": 64, "n": 64, "k": 128, "seed": 6635},
    {"m": 1024, "n": 7168, "k": 2048, "seed": 6132},
    {"m": 6144, "n": 4608, "k": 7168, "seed": 65436},
]

WARMUP = 50
ITERATIONS = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200"))
RTOL, ATOL = 2e-2, 1e-3


# ============================================================================
# SELF-TEST HARNESS
# ============================================================================


def check_correctness(cfg) -> dict:
    try:
        data = get_inputs(**cfg)
        a, b, a_scale, b_scale, c_triton = data
        c_ref = c_triton.clone()

        fp8_blockwise_mm_triton(a, b, a_scale, b_scale, c_triton)
        fp8_blockwise_mm_pytorch(a, b, a_scale, b_scale, c_ref)
        torch.cuda.synchronize()

        correct = torch.allclose(c_triton.float(), c_ref.float(), rtol=RTOL, atol=ATOL)
        max_diff = torch.max(torch.abs(c_triton.float() - c_ref.float())).item()
        return {"correct": correct, "max_diff": max_diff, "error": None}
    except Exception as e:
        return {"correct": False, "max_diff": float("inf"), "error": str(e)}


def benchmark_config(cfg, warmup=WARMUP, iters=ITERATIONS) -> dict:
    data = get_inputs(**cfg)
    a, b, a_scale, b_scale, c = data

    for _ in range(warmup):
        c_t = c.clone()
        fp8_blockwise_mm_triton(a, b, a_scale, b_scale, c_t)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        c_t = c.clone()
        fp8_blockwise_mm_triton(a, b, a_scale, b_scale, c_t)
    torch.cuda.synchronize()
    triton_ms = (time.perf_counter() - start) * 1000 / iters

    for _ in range(warmup):
        c_r = c.clone()
        fp8_blockwise_mm_pytorch(a, b, a_scale, b_scale, c_r)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        c_r = c.clone()
        fp8_blockwise_mm_pytorch(a, b, a_scale, b_scale, c_r)
    torch.cuda.synchronize()
    torch_ms = (time.perf_counter() - start) * 1000 / iters

    return {"triton_ms": triton_ms, "torch_ms": torch_ms,
            "speedup": torch_ms / triton_ms if triton_ms > 0 else 0.0}


def _config_label(cfg):
    return f"(M={cfg['m']},N={cfg['n']},K={cfg['k']})"


def evaluate(configs=None, warmup=WARMUP, iters=ITERATIONS, verbose=True) -> dict:
    configs = configs or TEST_CONFIGS[:5]
    results, failures = [], []

    if verbose:
        print(f"{'Config':<26} {'Correct':>8} {'Torch':>10} {'Triton':>10} {'Speedup':>10}")
        print("-" * 66)

    for cfg in configs:
        label = _config_label(cfg)
        corr = check_correctness(cfg)
        if not corr["correct"]:
            failures.append({"config": cfg, **corr})
            if verbose:
                err = corr["error"] or f"max_diff={corr['max_diff']:.2e}"
                print(f"{label:<26} {'FAIL':>8}   {err[:30]}")
            continue

        bench = benchmark_config(cfg, warmup=warmup, iters=iters)
        results.append({"config": cfg, "correct": True, **bench})

        if verbose:
            marker = " *" if bench["speedup"] > 1.0 else ""
            print(
                f"{label:<26} {'PASS':>8} "
                f"{bench['torch_ms']:>8.3f}ms {bench['triton_ms']:>8.3f}ms "
                f"{bench['speedup']:>8.2f}x{marker}"
            )

    speedups = [r["speedup"] for r in results]
    geomean = math.prod(speedups) ** (1 / len(speedups)) if speedups else 0.0

    if verbose:
        print("-" * 66)
        status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{len(configs)})"
        print(f"{'Status:':<26} {status}")
        if speedups:
            print(f"{'Speedup (geomean):':<26} {geomean:.2f}x")

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
        data = get_inputs(**cfg)
        a, b, a_scale, b_scale, c = data
        for _ in range(warmup):
            ct = c.clone()
            fp8_blockwise_mm_triton(a, b, a_scale, b_scale, ct)
        torch.cuda.synchronize()
        for _ in range(iters):
            ct = c.clone()
            fp8_blockwise_mm_triton(a, b, a_scale, b_scale, ct)
        torch.cuda.synchronize()
        if verbose:
            print(f"  {_config_label(cfg)} done")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FP8 Block-Scale GEMM (Triton dequant)")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    print("=" * 66)
    print("FP8 Block-Scale GEMM — Triton dequant + torch.mm")
    print("=" * 66)

    if args.profile:
        print("\n[Profile Mode]")
        run_profile()
    else:
        print("\n[Evaluation]")
        evaluate()

    print("=" * 66)
