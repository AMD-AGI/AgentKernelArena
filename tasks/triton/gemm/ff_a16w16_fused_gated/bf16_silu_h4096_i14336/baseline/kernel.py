# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused gated feed-forward block, Y = (act(X @ W_up[:N/2]^T) * (X @ W_up[N/2:]^T)) @ W_down, from ROCm/aiter.

Device kernel: aiter/ops/triton/_triton_kernels/gemm/feed_forward/ff_a16w16_fused_gated.py (unchanged apart from the two
imports, which are now the sibling files of this package). Launcher: aiter/ops/triton/gemm/feed_forward/
ff_a16w16_fused_gated.py (``ff_a16w16_fused_gated``), reduced to ``run()``. Tile configuration:
configs/gemm/gfx950-FF-A16W16-fused.json, selected with the M_LEQ_x / M_GEQ_x / any order and the
``bounds=(4, 8, 64, 4096)`` of the upstream ``_get_config``. No N=28672, K=4096 specialization exists for this kernel.
The kernel accumulates partial products of Y with atomic adds, so ``run()`` zeroes the output first (the launcher
allocates it with ``torch.zeros`` for the same reason).
"""

import triton
import triton.language as tl

from .pid_preprocessing import pid_grid, remap_xcd
from .activation import _silu


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@triton.jit
def _ff_a16w16_fused_gated(
    x_ptr,
    w1_ptr,
    w2_ptr,
    y_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_w1k,
    stride_w1n,
    stride_w2n,
    stride_w2k,
    stride_ym,
    stride_yk,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    GRID_MN: tl.constexpr,
    cache_modifier: tl.constexpr,
    activation: tl.constexpr,
    use_activation: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """

    tl.assume(stride_xm > 0)
    tl.assume(stride_xk > 0)
    tl.assume(stride_w1k > 0)
    tl.assume(stride_w1n > 0)
    tl.assume(stride_w2k > 0)
    tl.assume(stride_w2n > 0)
    tl.assume(stride_ym > 0)
    tl.assume(stride_yk > 0)

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    remap_xcd(pid, GRID_MN)

    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    # Create pointers for first block of x and w1 input matrices
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_xm = pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    x_ptrs = x_ptr + (offs_xm[:, None] * stride_xm + offs_k[None, :] * stride_xk)

    acc_dtype = tl.float32 if y_ptr.type.element_ty != tl.int8 else tl.int32

    """
    Our effective block size is actually BLOCK_N // 2.
    Per Triton program, we compute the matmul for TWO tiles of C of shape (BLOCK_M, BLOCK_N // 2) -
    one on the left side of C and one on the right side.
    """
    offs_w1n0 = pid_n.to(tl.int64) * (BLOCK_SIZE_N // 2) + tl.arange(
        0, BLOCK_SIZE_N // 2
    )
    offs_w1n1 = (
        pid_n.to(tl.int64) * (BLOCK_SIZE_N // 2) + tl.arange(0, BLOCK_SIZE_N // 2)
    ) + (N // 2)
    w1n0_ptrs = w1_ptr + (
        offs_k[:, None] * stride_w1k + offs_w1n0[None, :] * stride_w1n
    )
    w1n1_ptrs = w1_ptr + (
        offs_k[:, None] * stride_w1k + offs_w1n1[None, :] * stride_w1n
    )
    acc0 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N // 2), dtype=acc_dtype)
    acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N // 2), dtype=acc_dtype)

    for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        if EVEN_K:
            x = tl.load(x_ptrs, mask=offs_xm[:, None] < M)
            w1n0 = tl.load(
                w1n0_ptrs,
                mask=offs_w1n0[None, :] < (N // 2),
                cache_modifier=cache_modifier,
            )
            w1n1 = tl.load(
                w1n1_ptrs,
                mask=offs_w1n1[None, :] < N,
                cache_modifier=cache_modifier,
            )
        else:
            x = tl.load(
                x_ptrs,
                mask=(offs_xm[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                other=0.0,
            )
            w1n0 = tl.load(
                w1n0_ptrs,
                mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K)
                & (offs_w1n0[None, :] < (N // 2)),
                other=0.0,
                cache_modifier=cache_modifier,
            )
            w1n1 = tl.load(
                w1n1_ptrs,
                mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K)
                & (offs_w1n1[None, :] < N),
                other=0.0,
                cache_modifier=cache_modifier,
            )

        acc0 = tl.dot(x, w1n0, acc=acc0)
        acc1 = tl.dot(x, w1n1, acc=acc1)

        # Advance the ptrs to the next K block.
        x_ptrs += BLOCK_SIZE_K * stride_xk
        w1n0_ptrs += BLOCK_SIZE_K * stride_w1k
        w1n1_ptrs += BLOCK_SIZE_K * stride_w1k

    if use_activation:
        acc0 = activation(acc0)

    acc_gated = acc0 * acc1
    acc_gated = acc_gated.to(w2_ptr.type.element_ty)

    offs_w2n = pid_n.to(tl.int64) * (BLOCK_SIZE_N // 2) + tl.arange(
        0, BLOCK_SIZE_N // 2
    )

    w2_ptrs = w2_ptr + (offs_w2n[:, None] * stride_w2n + offs_k[None, :] * stride_w2k)

    offs_ym = pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    y_ptrs = y_ptr + (offs_ym[:, None] * stride_ym + offs_k[None, :] * stride_yk)

    # Stagger k-loop start position based on N block index (to minimize contention)
    k_cyclic_offset = pid_n % tl.cdiv(K, BLOCK_SIZE_K)
    w2_ptrs += k_cyclic_offset * stride_w2k * BLOCK_SIZE_K
    y_ptrs += k_cyclic_offset * stride_yk * BLOCK_SIZE_K

    for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
        if EVEN_K:
            w2 = tl.load(
                w2_ptrs,
                mask=offs_w2n[:, None] < (N // 2),
            )
        else:
            w2 = tl.load(
                w2_ptrs,
                mask=(offs_w2n[:, None] < (N // 2))
                & ((offs_k[None, :] + k_cyclic_offset * BLOCK_SIZE_K) < K),
                other=0.0,
            )
        partial_sum_y = tl.dot(acc_gated, w2)
        # tl.device_print("w2:", w2)
        # tl.device_print("partial y:", partial_sum_y)
        y_mask = (offs_ym[:, None] < M) & (
            (offs_k[None, :] + BLOCK_SIZE_K * k_cyclic_offset) < K
        )
        tl.atomic_add(y_ptrs, partial_sum_y, mask=y_mask, sem="relaxed", scope="gpu")
        # tl.store(y_ptrs, partial_sum_y, mask=y_mask)
        k_cyclic_offset += 1
        if k_cyclic_offset >= tl.cdiv(K, BLOCK_SIZE_K):
            k_cyclic_offset = 0
            w2_ptrs -= BLOCK_SIZE_K * stride_w2k * (tl.cdiv(K, BLOCK_SIZE_K) - 1)
            y_ptrs -= BLOCK_SIZE_K * stride_yk * (tl.cdiv(K, BLOCK_SIZE_K) - 1)
        else:
            w2_ptrs += BLOCK_SIZE_K * stride_w2k
            y_ptrs += BLOCK_SIZE_K * stride_yk


_CONFIGS = {
    "M_LEQ_4": dict(BLOCK_SIZE_M=4, BLOCK_SIZE_N=256, BLOCK_SIZE_K=256, GROUP_SIZE_M=1, num_warps=8, num_stages=1, waves_per_eu=3, matrix_instr_nonkdim=16, cache_modifier='.cg', kpack=1),
    "M_LEQ_8": dict(BLOCK_SIZE_M=8, BLOCK_SIZE_N=256, BLOCK_SIZE_K=256, GROUP_SIZE_M=1, num_warps=8, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=16, cache_modifier='.cg', kpack=1),
    "M_LEQ_64": dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=256, BLOCK_SIZE_K=256, GROUP_SIZE_M=1, num_warps=8, num_stages=1, waves_per_eu=2, matrix_instr_nonkdim=16, cache_modifier='.cg', kpack=1),
    "any": dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=256, BLOCK_SIZE_K=256, GROUP_SIZE_M=1, num_warps=8, num_stages=1, waves_per_eu=2, matrix_instr_nonkdim=16, cache_modifier=None, kpack=1),
}

_BOUNDS = (4, 8, 64, 4096)

# The upstream M_LEQ_8 row (BLOCK 8x256x256, num_stages=2) needs about 270 KB of shared memory for the x, gate and up
# tiles of its two pipeline stages (278464 bytes reported by the Triton compiler on gfx950), above the 163840-byte
# limit, so it cannot be launched. Batches of 5 to 8 rows use the M_LEQ_4 row of the same table instead (same N and K
# tiles and warps, BLOCK_SIZE_M 4, one stage, about 133 KB).
_ROW_FALLBACK = {"M_LEQ_8": "M_LEQ_4"}


def _get_config(M):
    """Selection of utils/gemm_config_utils.py:_get_gemm_config_cached for a config without N/K specialization."""
    for bound in _BOUNDS:
        key = f"M_LEQ_{bound}"
        if M <= bound and key in _CONFIGS:
            return dict(_CONFIGS[_ROW_FALLBACK.get(key, key)])
    for bound in reversed(_BOUNDS):
        key = f"M_GEQ_{bound}"
        if M >= bound and key in _CONFIGS:
            return dict(_CONFIGS[key])
    return dict(_CONFIGS["any"])


def run(*, x, w_up, w_down, out):
    M, K = x.shape
    N = w_up.shape[0]
    w1 = w_up.T  # (N, K) -> (K, N), a view
    config = _get_config(M)
    out.zero_()  # the kernel adds partial sums into the output
    grid = lambda META: (  # noqa: E731
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    _ff_a16w16_fused_gated[grid](
        x,
        w1,
        w_down,
        out,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w1.stride(0),
        w1.stride(1),
        w_down.stride(0),
        w_down.stride(1),
        out.stride(0),
        out.stride(1),
        activation=_silu,
        use_activation=True,
        **config,
    )
