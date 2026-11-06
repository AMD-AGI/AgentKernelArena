# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Standalone MXFP8 activation x FP8 weight scaled GEMM Triton kernel (gfx950-only).

Provenance: ROCm/aiter, aiter.ops.triton.gemm.basic.gemm_afp8wfp8
(`gemm_afp8wfp8`) and its device kernel `_gemm_afp8wfp8_kernel`
(aiter.ops.triton._triton_kernels.gemm.basic.gemm_afp8wfp8). The preshuffle
kernel/launcher and the split-K reduce kernel are dropped; only the non-split-K
(`NUM_KSPLIT == 1`) triton path is kept, and the on-disk tuned-config lookup is
replaced by a static config (BLOCK_SIZE_K == 128 so the 128x128 W-scale layout is
addressed correctly). The pid helpers (`remap_xcd`, `pid_grid`) and the
constexpr-aware kernel-naming helper are inlined, so the module depends only on
`triton` + `torch`.

Op:
    Y = X @ W^T with MXFP8 activation scales (1x32 e8m0 a-scales [M, K//32]) and
    FP8 weight block scales (128x128 e8m0 w-scales [N//128, K//128]), fp32
    accumulation via `tl.dot_scaled` (format "e4m3"), output written in `dtype`
    (default bf16).

ARCH NOTE: `tl.dot_scaled` (microscale MX matmul) lowers to a scaled-MFMA
instruction that exists only on CDNA4 (gfx950); on CDNA3 (gfx942) it fails to
compile ("Unsupported DotScaleOp"). This task is therefore tagged
`supported_archs: [gfx950]` and the harness arch-guard SKIPs (exit 0) on gfx942.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Inlined helper utils (XCD remap, pid grid, kernel repr)
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
def remap_xcd(pid, GRID_MN, NUM_XCDS: tl.constexpr = 8):
    """Inlined from pid_preprocessing.remap_xcd (XCD-balanced pid remap)."""
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    tall_xcds = GRID_MN % NUM_XCDS
    if tall_xcds == 0:
        tall_xcds = tl.cast(NUM_XCDS, tall_xcds.type)
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


_gemm_afp8wfp8_repr = make_kernel_repr(
    "_gemm_afp8wfp8_kernel",
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
    ],
)


@triton.heuristics(
    {
        "EVEN_K": lambda args: (args["K"] % args["BLOCK_SIZE_K"] == 0),
    }
)
@triton.jit(repr=_gemm_afp8wfp8_repr)
def _gemm_afp8wfp8_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scales_ptr,
    b_scales_ptr,
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
    stride_asm,
    stride_ask,
    stride_bsn,
    stride_bsk,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
    waves_per_eu: tl.constexpr,
    matrix_instr_nonkdim: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    """
    Kernel for computing the matmul C = A x B.
    A and B inputs are FP8 e4m3 (1 byte per element).
    A_scales are e8m0 (uint8) with shape (M, K // 32).
    B_scales are stored compact e8m0 (uint8) with shape (N // 128, K // 128),
    representing 128x128 weight blocks. Broadcast inside kernel to (N, K // 32).
    A has shape (M, K), B has shape (K, N) and C has shape (M, N).
    Output dtype is determined by c_ptr (bf16 or fp16).
    When NUM_KSPLIT > 1, K is split into NUM_KSPLIT partitions of
    SPLITK_BLOCK_SIZE elements and the partial result for partition pid_k is
    written to c_ptr + pid_k * stride_ck; a downstream reduce kernel sums them.
    """

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_asm > 0)
    tl.assume(stride_ask > 0)
    tl.assume(stride_bsk > 0)
    tl.assume(stride_bsn > 0)

    GRID_MN = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)

    pid_unified = tl.program_id(axis=0)
    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    if NUM_KSPLIT == 1:
        pid = remap_xcd(pid, GRID_MN, NUM_XCDS=8)
        pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(pid_k >= 0)

    # Scale group sizes
    SCALE_GROUP_SIZE: tl.constexpr = 32  # A: per 32 elements along K
    B_SCALE_K_GROUP: tl.constexpr = 128  # B: per 128 along K
    B_SCALE_N_GROUP: tl.constexpr = 128  # B: per 128 along N

    if (pid_k * SPLITK_BLOCK_SIZE) < K:
        # K-block iteration range for this split (absolute block indices).
        num_k_iter = tl.cdiv(SPLITK_BLOCK_SIZE, BLOCK_SIZE_K)

        # Create pointers for first block of A and B input matrices. The K
        # offset is the absolute start of this split's K range.
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

        # A-scale pointers: per-row (M) and per scale group (K // 32). Shift
        # along the K-scale axis by the split's start in scale groups.
        offs_ks_a = tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_SIZE)
        offs_ks_a_split = pid_k * (SPLITK_BLOCK_SIZE // SCALE_GROUP_SIZE) + offs_ks_a
        a_scale_ptrs = (
            a_scales_ptr
            + offs_am[:, None] * stride_asm
            + offs_ks_a_split[None, :] * stride_ask
        )

        # B-scale pointers: compact (N // 128, K // 128) — broadcast inside the kernel
        # Each scale covers a 128(N) x 128(K) block. Computed per-iteration below
        # using absolute K (so split-K naturally addresses the right b-scale block).
        offs_bsn = offs_bn // B_SCALE_N_GROUP  # (BLOCK_SIZE_N,)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        offs_scale_k_a = tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_SIZE)

        for k in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter):
            # K base for this iteration (in elements, absolute).
            k_base = k * BLOCK_SIZE_K

            # ---- Load A scales (M, BLOCK_SIZE_K // 32) ----
            if EVEN_K:
                a_scales = tl.load(a_scale_ptrs)
            else:
                a_scale_mask = offs_scale_k_a[None, :] < (
                    K // SCALE_GROUP_SIZE - k * (BLOCK_SIZE_K // SCALE_GROUP_SIZE)
                )
                a_scales = tl.load(a_scale_ptrs, mask=a_scale_mask, other=127)

            # ---- Load and broadcast B scales (BLOCK_SIZE_N, BLOCK_SIZE_K // 32) ----
            offs_bsk = (
                k_base + offs_scale_k_a * SCALE_GROUP_SIZE
            ) // B_SCALE_K_GROUP  # (BLOCK_SIZE_K // 32,)
            b_scale_ptrs = (
                b_scales_ptr
                + offs_bsn[:, None] * stride_bsn
                + offs_bsk[None, :] * stride_bsk
            )
            if EVEN_K:
                b_scales = tl.load(b_scale_ptrs, cache_modifier=cache_modifier)
            else:
                # OOB along K: load with the same mask as a-scales
                b_scale_mask = offs_scale_k_a[None, :] < (
                    K // SCALE_GROUP_SIZE - k * (BLOCK_SIZE_K // SCALE_GROUP_SIZE)
                )
                b_scales = tl.load(
                    b_scale_ptrs,
                    mask=b_scale_mask,
                    other=127,
                    cache_modifier=cache_modifier,
                )

            # ---- Load A, B data ----
            if EVEN_K:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs, cache_modifier=cache_modifier)
            else:
                a = tl.load(
                    a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0
                )
                b = tl.load(
                    b_ptrs,
                    mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                    other=0,
                    cache_modifier=cache_modifier,
                )

            accumulator = tl.dot_scaled(
                a, a_scales, "e4m3", b, b_scales, "e4m3", accumulator
            )

            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
            a_scale_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_SIZE) * stride_ask

        c = accumulator.to(c_ptr.type.element_ty)

        # Write back the block of the output matrix C with masks. For
        # NUM_KSPLIT > 1, each pid_k writes to a separate slab of c_ptr.
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


def _get_config(M: int, N: int, K: int):
    """Static (non-split-K) config replacing the on-disk tuned-config lookup.

    A single robust MXFP8 tile is used with NUM_KSPLIT == 1 (no split-K reduce
    pass). BLOCK_SIZE_K is fixed to 128 so the 128x128 W-scale blocks are
    addressed correctly. SPLITK_BLOCK_SIZE is set to K by the host launcher.
    """
    block_m = 128 if M >= 128 else max(16, triton.next_power_of_2(M))
    block_n = 128
    return {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 8,
        "NUM_KSPLIT": 1,
        "cache_modifier": "",
        "matrix_instr_nonkdim": 16,
        "num_warps": 4,
        "num_stages": 2,
        "waves_per_eu": 0,
    }


def run(*, a, a_scale, w, w_scale, out):
    """Y = A @ W^T written into `out` ([M, N], bf16) with the static non-split-K config."""
    M, K = a.shape
    N, K_w = w.shape
    assert K == K_w, f"K mismatch: a has K={K}, w has K={K_w}"

    # Transposed (K, N) view of the weight; tl.dot_scaled with format "e4m3"
    # expects uint8-typed operands, so reinterpret the FP8 buffers (bit-identical views).
    w_t = w.T.view(torch.uint8)
    a_u8 = a.view(torch.uint8)

    config = _get_config(M, N, K)
    config["SPLITK_BLOCK_SIZE"] = triton.cdiv(K, config["NUM_KSPLIT"])

    grid = lambda META: (  # noqa: E731
        META["NUM_KSPLIT"] * triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    _gemm_afp8wfp8_kernel[grid](
        a_u8,
        w_t,
        out,
        a_scale,
        w_scale,
        M,
        N,
        K,
        a_u8.stride(0),
        a_u8.stride(1),
        w_t.stride(0),
        w_t.stride(1),
        out.stride(0),
        out.stride(0),
        out.stride(1),
        a_scale.stride(0),
        a_scale.stride(1),
        w_scale.stride(0),
        w_scale.stride(1),
        **config,
    )
