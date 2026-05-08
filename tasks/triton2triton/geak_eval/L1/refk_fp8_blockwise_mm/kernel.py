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
# ENTRY POINT (callable form of the kernel for downstream profilers)
# ============================================================================


def triton_op(m, n, k, seed):
    data = _generate_input(m, n, k, seed)
    return fp8_blockwise_mm_triton(*data)


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

RTOL, ATOL = 2e-2, 1e-3
