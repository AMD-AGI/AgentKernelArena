# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""a16w8 block-scaled GEMM Triton kernel (bf16 activations, fp8 weights with 128x128 fp32 scales).

Provenance: ROCm/aiter, aiter/ops/triton/gemm/basic/gemm_a16w8_blockscale.py (launcher) and
aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w8_blockscale.py (device kernel
`_gemm_a16w8_blockscale_kernel`). The preshuffle kernel/launcher, the split-K reduce kernel and the PREQUANT
(fused activation fp8 quantization) branch are dropped; only the non-split-K (`NUM_KSPLIT == 1`),
non-prequant path is kept. The on-disk tuned-config lookup is replaced by the NUM_KSPLIT == 1 rows of
configs/gemm/gfx950-GEMM-A16W8_BLOCKSCALE.json (BLOCK_SIZE_K == GROUP_K == 128). The pid helper (`pid_grid`) and the
constexpr-aware kernel-naming helper are inlined, so the module depends only on `triton` + `torch`.

Op (w8-only block-scale dequant matmul):
    Y = X @ W^T with 16-bit activations X [M, K] and 8-bit weights W [N, K]
    (transposed to [K, N] before launch). The per-block W_scale
    [ceil(N/128), ceil(K/128)] is applied inside the K loop (W is upcast to the
    activation dtype, scaled, and accumulated in fp32), output written in `dtype`
    (default bf16). On gfx942 the arch-appropriate fp8 weight type is e4m3fnuz.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Inlined helper utils (pid grid, kernel repr)
# ---------------------------------------------------------------------------
def _sanitize_constexpr_value(value):
    if value is None:
        return "NONE"
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else str(value)
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")
    return cleaned.upper() if cleaned else "NONE"


def make_kernel_repr(base_name, config_keys):
    """Inlined from utils/_triton/kernel_repr.py (constexpr-aware kernel naming)."""

    def _repr(specialization):
        constants = specialization.constants
        parts = [
            f"{key}_{_sanitize_constexpr_value(constants.get(key, None))}"
            for key in config_keys
        ]
        return f"{base_name}_{'_'.join(parts)}" if parts else base_name

    return _repr


@triton.jit
def pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M: tl.constexpr = 1):
    """Inlined from pid_preprocessing.pid_grid (1D->2D grouped pid map)."""
    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        tl.assume(group_size_m >= 0)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


_gemm_a16w8_blockscale_repr = make_kernel_repr(
    "_gemm_a16w8_blockscale_kernel",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "num_warps",
        "num_stages",
        "waves_per_eu",
        "matrix_instr_nonkdim",
        "cache_modifier",
        "NUM_KSPLIT",
        "SPLITK_BLOCK_SIZE",
        "EVEN_K",
        "GRID_MN",
    ],
)


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@triton.jit(repr=_gemm_a16w8_blockscale_repr)
def _gemm_a16w8_blockscale_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_ck,
    stride_cm,
    stride_cn,
    stride_bscale_k,
    stride_bscale_n,
    # Meta-parameters
    GROUP_K: tl.constexpr,
    GROUP_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    GRID_MN: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
    waves_per_eu: tl.constexpr,
    matrix_instr_nonkdim: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    """
    Computes the 8 bit (w8) block-scale matmul C = A x B.

    A: (M, K) 16-bit activations. B: (K, N) 8-bit weights.
    B_scale: (*scale_k, **scale_n), scale_k = ceil(K/GROUP_K), scale_n = ceil(N/GROUP_N).
    For this kernel implementation, GROUP_K must equal BLOCK_K.
    """

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_ck > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_bscale_k > 0)
    tl.assume(stride_bscale_n > 0)

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid_unified = tl.program_id(axis=0)
    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    if NUM_KSPLIT == 1:
        pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(pid_k >= 0)

    if (pid_k * SPLITK_BLOCK_SIZE) < K:

        # SPLITK_BLOCK_SIZE = tl.cdiv(K, NUM_KSPLIT)
        num_k_iter = tl.cdiv(SPLITK_BLOCK_SIZE, BLOCK_SIZE_K)

        # Create pointers for first block of A and B input matrices
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        offs_k_split = pid_k * SPLITK_BLOCK_SIZE + offs_k
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        a_ptrs = a_ptr + (
            offs_am[:, None] * stride_am + offs_k_split[None, :] * stride_ak
        )
        b_ptrs = b_ptr + (
            offs_k_split[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        )

        # Create pointers for the scales
        offs_ks = (pid_k * SPLITK_BLOCK_SIZE) // GROUP_K
        offs_bsn = offs_bn // GROUP_N
        b_scale_ptrs = (
            b_scale_ptr + offs_ks * stride_bscale_k + offs_bsn * stride_bscale_n
        )
        offs_ks_step = BLOCK_SIZE_K // GROUP_K

        acc_dtype = tl.float32 if c_ptr.type.element_ty != tl.int8 else tl.int32
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        for k in tl.range(
            pid_k * num_k_iter, (pid_k + 1) * num_k_iter, num_stages=num_stages
        ):
            # Load the next block of A and B, generate a mask by checking the K dimension.
            # If it is out of bounds, set it to 0.
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

            b_scale = tl.load(b_scale_ptrs)

            b = b.to(a_ptr.type.element_ty)
            accumulator += tl.dot(a, b) * b_scale[None, :]

            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

            b_scale_ptrs += offs_ks_step * stride_bscale_k

        c = accumulator.to(c_ptr.type.element_ty)

        # Write back the block of the output matrix C with masks.
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        c_ptrs = (
            c_ptr
            + stride_cm * offs_cm[:, None]
            + stride_cn * offs_cn[None, :]
            + pid_k * stride_ck
        )
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)


def _get_config(M: int):
    """NUM_KSPLIT == 1 rows of the upstream gfx950 table (M_LEQ_256, M_LEQ_512, any).

    The upstream rows for M <= 128 use split-K plus a reduce kernel, which is not carried over; those sizes use
    the M_LEQ_256 row. BLOCK_SIZE_K is 128 so GROUP_K (the weight scale block) equals BLOCK_SIZE_K.
    """
    if M <= 256:
        return {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8,
            "num_warps": 2, "num_stages": 2, "waves_per_eu": 1, "matrix_instr_nonkdim": 16,
            "cache_modifier": None, "NUM_KSPLIT": 1,
        }
    if M <= 512:
        return {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1,
            "num_warps": 8, "num_stages": 2, "waves_per_eu": 1, "matrix_instr_nonkdim": 16,
            "cache_modifier": ".cg", "NUM_KSPLIT": 1,
        }
    return {
        "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 32,
        "num_warps": 8, "num_stages": 2, "waves_per_eu": 1, "matrix_instr_nonkdim": 16,
        "cache_modifier": None, "NUM_KSPLIT": 1,
    }


def run(*, x: torch.Tensor, w: torch.Tensor, w_scale: torch.Tensor, out: torch.Tensor) -> None:
    """out = x @ dequant(w)^T: x [M, K] bf16, w [N, K] fp8, w_scale [ceil(N/128), ceil(K/128)] fp32."""
    M, K = x.shape
    N, _ = w.shape

    # Transpose w and w_scale (views).
    w = w.T  # (K, N)
    w_scale = w_scale.T  # (scale_k, scale_n)

    config = _get_config(M)
    config["SPLITK_BLOCK_SIZE"] = triton.cdiv(K, config["NUM_KSPLIT"])

    # Scale block sizes
    config["GROUP_K"] = triton.next_power_of_2(triton.cdiv(K, w_scale.shape[0]))
    config["GROUP_N"] = triton.next_power_of_2(triton.cdiv(N, w_scale.shape[1]))

    assert config["GROUP_K"] == config["BLOCK_SIZE_K"], "GROUP_K must equal BLOCK_SIZE_K"

    grid = lambda META: (  # noqa: E731
        META["NUM_KSPLIT"]
        * triton.cdiv(M, META["BLOCK_SIZE_M"])
        * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    _gemm_a16w8_blockscale_kernel[grid](
        x,
        w,
        out,
        w_scale,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        0,
        out.stride(0),
        out.stride(1),
        w_scale.stride(0),
        w_scale.stride(1),
        **config,
    )
