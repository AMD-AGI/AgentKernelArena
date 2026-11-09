# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Gated 16-bit GEMM, Y = act(X @ W[:N/2]^T) * (X @ W[N/2:]^T), from ROCm/aiter.

Device kernel: aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16_gated.py (unchanged apart from the three
imports, which are now the sibling files of this package). Launcher: aiter/ops/triton/gemm/basic/gemm_a16w16_gated.py
(``gemm_a16w16_gated``), reduced to ``run()``. Tile configuration: configs/gemm/gfx950-GEMM-A16W16-gated.json, selected
with the M_LEQ_x / M_GEQ_x / any order and the ``bounds=(64, 128, 256, 512, 2048)`` of the upstream ``_get_config``.
No N=28672, K=4096 specialization exists for this kernel, so the default file applies.
"""

import triton
import triton.language as tl

from .kernel_repr import make_kernel_repr
from .pid_preprocessing import pid_grid, remap_xcd
from .activation import _silu

_gemm_a16w16_gated_repr = make_kernel_repr(
    "_gemm_a16_w16_gated_kernel",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "EVEN_K",
        "GRID_MN",
        "cache_modifier",
        "activation",
        "use_activation",
    ],
)


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@triton.jit(repr=_gemm_a16w16_gated_repr)
def _gemm_a16_w16_gated_kernel(
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

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

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

    # Create pointers for first block of A and B input matrices
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_am = (pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)

    acc_dtype = tl.float32 if c_ptr.type.element_ty != tl.int8 else tl.int32

    """
    Our effective block size is actually BLOCK_N // 2.
    Per Triton program, we compute the matmul for TWO tiles of C of shape (BLOCK_M, BLOCK_N // 2) -
    one on the left side of C and one on the right side.
    """
    offs_bn0 = (
        pid_n.to(tl.int64) * (BLOCK_SIZE_N // 2) + tl.arange(0, BLOCK_SIZE_N // 2)
    ) % (N // 2)
    offs_bn1 = (
        (pid_n.to(tl.int64) * (BLOCK_SIZE_N // 2) + tl.arange(0, BLOCK_SIZE_N // 2))
        % (N // 2)
    ) + (N // 2)
    b0_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn0[None, :] * stride_bn)
    b1_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn1[None, :] * stride_bn)
    acc0 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N // 2), dtype=acc_dtype)
    acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N // 2), dtype=acc_dtype)

    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), num_stages=1):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        if EVEN_K:
            a = tl.load(a_ptrs)
            b0 = tl.load(b0_ptrs, cache_modifier=cache_modifier)
            b1 = tl.load(b1_ptrs, cache_modifier=cache_modifier)
        else:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b0 = tl.load(
                b0_ptrs,
                mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                other=0.0,
                cache_modifier=cache_modifier,
            )
            b1 = tl.load(
                b1_ptrs,
                mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                other=0.0,
                cache_modifier=cache_modifier,
            )

        acc0 = tl.dot(a, b0, acc=acc0)
        acc1 = tl.dot(a, b1, acc=acc1)

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b0_ptrs += BLOCK_SIZE_K * stride_bk
        b1_ptrs += BLOCK_SIZE_K * stride_bk

    if use_activation:
        acc0 = activation(acc0)

    acc_gated = acc0 * acc1
    c = acc_gated.to(c_ptr.type.element_ty)

    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n.to(tl.int64) * (BLOCK_SIZE_N // 2) + tl.arange(0, BLOCK_SIZE_N // 2)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < (N // 2))
    tl.store(c_ptrs, c, mask=c_mask)


_CONFIGS = {
    "M_LEQ_8": dict(BLOCK_SIZE_M=8, BLOCK_SIZE_N=16, BLOCK_SIZE_K=256, GROUP_SIZE_M=1, num_warps=1, num_stages=3, waves_per_eu=2, matrix_instr_nonkdim=16, cache_modifier='.cg'),
    "M_LEQ_16": dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=128, GROUP_SIZE_M=8, num_warps=2, num_stages=3, waves_per_eu=1, matrix_instr_nonkdim=16, cache_modifier='.cg'),
    "M_LEQ_32": dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=128, GROUP_SIZE_M=1, num_warps=2, num_stages=3, waves_per_eu=2, matrix_instr_nonkdim=16, cache_modifier='.cg'),
    "M_LEQ_64": dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128, GROUP_SIZE_M=1, num_warps=4, num_stages=3, waves_per_eu=2, matrix_instr_nonkdim=16, cache_modifier='.cg'),
    "M_LEQ_128": dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=128, GROUP_SIZE_M=1, num_warps=8, num_stages=3, waves_per_eu=1, matrix_instr_nonkdim=16, cache_modifier='.cg'),
    "M_LEQ_256": dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=128, GROUP_SIZE_M=1, num_warps=8, num_stages=2, waves_per_eu=1, matrix_instr_nonkdim=16, cache_modifier='.cg'),
    "M_LEQ_512": dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=64, BLOCK_SIZE_K=128, GROUP_SIZE_M=1, num_warps=8, num_stages=2, waves_per_eu=4, matrix_instr_nonkdim=16, cache_modifier='.cg'),
    "M_LEQ_2048": dict(BLOCK_SIZE_M=256, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64, GROUP_SIZE_M=1, num_warps=8, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=16, cache_modifier=None),
    "any": dict(BLOCK_SIZE_M=256, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64, GROUP_SIZE_M=16, num_warps=8, num_stages=1, waves_per_eu=2, matrix_instr_nonkdim=16, cache_modifier=None),
}

_BOUNDS = (64, 128, 256, 512, 2048)


def _get_config(M):
    """Selection of utils/gemm_config_utils.py:_get_gemm_config_cached for a config without N/K specialization."""
    for bound in _BOUNDS:
        key = f"M_LEQ_{bound}"
        if M <= bound and key in _CONFIGS:
            return dict(_CONFIGS[key])
    for bound in reversed(_BOUNDS):
        key = f"M_GEQ_{bound}"
        if M >= bound and key in _CONFIGS:
            return dict(_CONFIGS[key])
    return dict(_CONFIGS["any"])


def run(*, x, w, out):
    M, K = x.shape
    N = w.shape[0]
    wt = w.T  # (N, K) -> (K, N), a view
    config = _get_config(M)
    grid = lambda META: (  # noqa: E731
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    _gemm_a16_w16_gated_kernel[grid](
        x,
        wt,
        out,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        wt.stride(0),
        wt.stride(1),
        out.stride(0),
        out.stride(1),
        activation=_silu,
        use_activation=True,
        **config,
    )
