# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.


"""Block-scaled fp8 GEMM fused with the MLA key/value split and concatenation from ROCm/aiter (Triton): the product
x @ w^T is viewed as [M, D, S1 + S2]; the first S1 columns of each head are concatenated with a shared rope key y [M, D, S3]
into k, the last S2 become v.

Provenance: the device kernels of aiter/ops/triton/_triton_kernels/gemm/fused/fused_gemm_a8w8_blockscale_split_cat.py
(`_fused_gemm_a8w8_blockscale_split_cat` and `_fused_gemm_a8w8_blockscale_split_cat_reduce`; the preshuffle kernel is
dropped), the non-preshuffled launcher `fused_gemm_a8w8_blockscale_split_cat` of
aiter/ops/triton/gemm/fused/fused_gemm_a8w8_blockscale_split_cat.py, and `remap_xcd` and `pid_grid` of
aiter/ops/triton/utils/_triton/pid_preprocessing.py. The tuned table is the shape-specialized gfx950 file
configs/gemm/gfx950-GEMM-A8W8_BLOCKSCALE-N=32768-K=512.json, inlined with the get_gemm_config lookup order. Every row has
NUM_KSPLIT 1, so the reduce kernel is never launched for this shape; it is kept with the kernel it belongs to.
"""


import torch
import triton
import triton.language as tl


@triton.jit
def remap_xcd(pid, GRID_MN, NUM_XCDS: tl.constexpr = 8):
    ## pid remapping on xcds
    # Number of pids per XCD in the new arrangement
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    # When GRID_MN cannot divide NUM_XCDS, some xcds will have
    # pids_per_xcd pids, the other will have pids_per_xcd - 1 pids.
    # We calculate the number of xcds that have pids_per_xcd pids as
    # tall_xcds
    tall_xcds = GRID_MN % NUM_XCDS
    if tall_xcds == 0:
        tall_xcds = tl.cast(NUM_XCDS, tall_xcds.type)
    # Compute current XCD and local pid within the XCD
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    # Calculate new pid based on the new grouping
    # Note that we need to consider the following two cases:
    # 1. the current pid is on a tall xcd
    # 2. the current pid is on a short xcd
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
    """
    Maps 1D pid to 2D grid coords (pid_m, pid_n).

    Args:
        - pid: 1D pid
        - num_pid_m: grid m size
        - num_pid_n: grid n size
        - GROUP_SIZE_M: tl.constexpr: default is 1
    """
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


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@triton.jit
def _fused_gemm_a8w8_blockscale_split_cat(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    y_ptr,
    c1_ptr,
    c2_ptr,
    a_scale_ptr,
    b_scale_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    D,
    S1,
    S2,
    S3,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_a_m` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_a_m,
    stride_a_k,
    stride_b_k,
    stride_b_n,
    stride_y_m,
    stride_y_d,
    stride_y_s,
    stride_c1_k,
    stride_c1_m,
    stride_c1_n,
    stride_c2_m,
    stride_c2_d,
    stride_c2_s,
    stride_ascale_m,
    stride_ascale_k,
    stride_bscale_k,
    stride_bscale_n,
    # Meta-parameters
    GROUP_K: tl.constexpr,
    GROUP_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_S3: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    GRID_MN: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    tl.assume(stride_a_m > 0)
    tl.assume(stride_a_k > 0)
    tl.assume(stride_b_k > 0)
    tl.assume(stride_b_n > 0)
    tl.assume(stride_y_m > 0)
    tl.assume(stride_y_s > 0)
    tl.assume(stride_c1_k > 0)
    tl.assume(stride_c1_m > 0)
    tl.assume(stride_c1_n > 0)
    tl.assume(stride_c2_d > 0)
    tl.assume(stride_c2_m > 0)
    tl.assume(stride_c2_s > 0)
    tl.assume(stride_ascale_m > 0)
    tl.assume(stride_ascale_k > 0)
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
        remap_xcd(pid, GRID_MN)
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
        # ^ Number of K blocks within our split-K partition

        # Create pointers for first block of A and B input matrices
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        offs_k_split = pid_k * SPLITK_BLOCK_SIZE + offs_k
        offs_a_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_b_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        a_ptrs = a_ptr + (
            offs_a_m[:, None] * stride_a_m + offs_k_split[None, :] * stride_a_k
        )
        b_ptrs = b_ptr + (
            offs_k_split[:, None] * stride_b_k + offs_b_n[None, :] * stride_b_n
        )

        # Create pointers for the scales
        offs_k_scale = (pid_k * SPLITK_BLOCK_SIZE) // GROUP_K
        a_scale_ptrs = (
            a_scale_ptr + offs_a_m * stride_ascale_m + offs_k_scale * stride_ascale_k
        )
        offs_b_scale_n = offs_b_n // GROUP_N
        b_scale_ptrs = (
            b_scale_ptr
            + offs_k_scale * stride_bscale_k
            + offs_b_scale_n * stride_bscale_n
        )
        offs_ks_step = BLOCK_SIZE_K // GROUP_K

        acc_dtype = tl.float32 if c1_ptr.type.element_ty != tl.int8 else tl.int32
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        for k in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter):
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

            a_scale = tl.load(a_scale_ptrs)
            b_scale = tl.load(b_scale_ptrs)

            # Perform dot operation and apply scale
            accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]

            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_a_k
            b_ptrs += BLOCK_SIZE_K * stride_b_k

            # k_cur = k * BLOCK_SIZE_K // GROUP_K
            # k_nxt = (k + 1) * BLOCK_SIZE_K // GROUP_K
            # offs_ks = k_nxt - k_cur
            a_scale_ptrs += offs_ks_step * stride_ascale_k
            b_scale_ptrs += offs_ks_step * stride_bscale_k

        c = accumulator.to(c1_ptr.type.element_ty)  # [BLOCK_SIZE_M, BLOCK_SIZE_N]

        if NUM_KSPLIT == 1:
            offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
            offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
            offs_d = offs_n // (S1 + S2)
            offs_s = offs_n % (S1 + S2)

            # Write back the block of the output matrix C1 with masks.
            c1_ptrs = (
                c1_ptr
                + stride_c1_m * offs_m[:, None]
                + stride_c1_k * offs_d[None, :]
                + stride_c1_n * offs_s[None, :]
            )
            c1_mask = (
                (offs_m[:, None] < M) & (offs_d[None, :] < D) & (offs_s[None, :] < S1)
            )
            tl.store(c1_ptrs, c, mask=c1_mask)

            # Write back the block of the output matrix C2 with masks.
            c2_ptrs = (
                c2_ptr
                + stride_c2_m * offs_m[:, None]
                + stride_c2_d * offs_d[None, :]
                + stride_c2_s * (offs_s[None, :] - S1)
            )
            c2_mask = (
                (offs_m[:, None] < M)
                & (offs_d[None, :] < D)
                & (offs_s[None, :] >= S1)
                & (offs_s[None, :] < S1 + S2)
            )
            tl.store(c2_ptrs, c, mask=c2_mask)

            # Handle y
            offs_n = pid_n * BLOCK_SIZE_S3 + tl.arange(0, BLOCK_SIZE_S3).to(tl.int64)
            offs_d = offs_n // S3
            offs_s = offs_n % S3

            # Load y
            y_ptrs = (
                y_ptr
                + stride_y_m * offs_m[:, None]
                + stride_y_d * offs_d[None, :]
                + stride_y_s * offs_s[None, :]
            )
            y_mask = (
                (offs_m[:, None] < M) & (offs_d[None, :] < D) & (offs_s[None, :] < S3)
            )
            y = tl.load(y_ptrs, mask=y_mask)

            # Concat y to the output matrix C1.
            c1_ptrs = (
                c1_ptr
                + stride_c1_m * offs_m[:, None]
                + stride_c1_k * offs_d[None, :]
                + stride_c1_n * (offs_s[None, :] + S1)
            )
            tl.store(c1_ptrs, y, mask=y_mask)
        else:
            # SPLIT K
            # Write back the block of the output matrix C with masks.
            offs_c1_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
            offs_c1_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
            c1_ptrs = (
                c1_ptr
                + stride_c1_m * offs_c1_m[:, None]
                + stride_c1_n * offs_c1_n[None, :]
                + stride_c1_k * pid_k
            )
            c1_mask = (offs_c1_m[:, None] < M) & (offs_c1_n[None, :] < N)
            tl.store(c1_ptrs, c, mask=c1_mask)


@triton.jit
def _fused_gemm_a8w8_blockscale_split_cat_reduce(
    c_ptr,
    c1_ptr,
    c2_ptr,
    y_ptr,
    M,
    N,
    D,
    S1,
    S2,
    S3,
    stride_c_k,
    stride_c_m,
    stride_c_n,
    stride_c1_m,
    stride_c1_d,
    stride_c1_s,
    stride_c2_m,
    stride_c2_d,
    stride_c2_s,
    stride_y_m,
    stride_y_d,
    stride_y_s,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_S3: tl.constexpr,
    ACTUAL_KSPLIT: tl.constexpr,
    MAX_KSPLIT: tl.constexpr,
):
    tl.assume(stride_c_k > 0)
    tl.assume(stride_c_m > 0)
    tl.assume(stride_c_n > 0)
    tl.assume(stride_c1_m > 0)
    tl.assume(stride_c1_d > 0)
    tl.assume(stride_c1_s > 0)
    tl.assume(stride_c2_m > 0)
    tl.assume(stride_c2_d > 0)
    tl.assume(stride_c2_s > 0)
    tl.assume(stride_y_m > 0)
    tl.assume(stride_y_s > 0)

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    tl.assume(pid_m > 0)
    tl.assume(pid_n > 0)

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, MAX_KSPLIT)
    c_ptrs = (
        c_ptr
        + (offs_k[:, None, None] * stride_c_k)
        + (offs_m[None, :, None] * stride_c_m)
        + (offs_n[None, None, :] * stride_c_n)
    )

    if ACTUAL_KSPLIT == MAX_KSPLIT:
        c = tl.load(c_ptrs)
    else:
        c = tl.load(c_ptrs, mask=offs_k[:, None, None] < ACTUAL_KSPLIT, other=0.0)
    c = tl.sum(c, axis=0)

    c = c.to(c1_ptr.type.element_ty)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_d = offs_n // (S1 + S2)
    offs_s = offs_n % (S1 + S2)

    # store c to output matrix c1
    c1_ptrs = (
        c1_ptr
        + stride_c1_m * offs_m[:, None]
        + stride_c1_d * offs_d[None, :]
        + stride_c1_s * offs_s[None, :]
    )
    c1_mask = (offs_m[:, None] < M) & (offs_d[None, :] < D) & (offs_s[None, :] < S1)
    tl.store(c1_ptrs, c, mask=c1_mask)

    # store c to output matrix c2
    c2_ptrs = (
        c2_ptr
        + stride_c2_m * offs_m[:, None]
        + stride_c2_d * offs_d[None, :]
        + stride_c2_s * (offs_s[None, :] - S1)
    )
    c2_mask = (
        (offs_m[:, None] < M)
        & (offs_d[None, :] < D)
        & (offs_s[None, :] >= S1)
        & (offs_s[None, :] < S1 + S2)
    )
    tl.store(c2_ptrs, c, mask=c2_mask)

    # handle y
    offs_n = pid_n * BLOCK_SIZE_S3 + tl.arange(0, BLOCK_SIZE_S3)
    offs_d = offs_n // S3
    offs_s = offs_n % S3

    # load y
    y_ptrs = (
        y_ptr
        + stride_y_m * offs_m[:, None]
        + stride_y_d * offs_d[None, :]
        + stride_y_s * offs_s[None, :]
    )
    y_mask = (offs_m[:, None] < M) & (offs_d[None, :] < D) & (offs_s[None, :] < S3)
    y = tl.load(y_ptrs, mask=y_mask)

    # concat y to c1
    c1_ptrs = (
        c1_ptr
        + stride_c1_m * offs_m[:, None]
        + stride_c1_d * offs_d[None, :]
        + stride_c1_s * (offs_s[None, :] + S1)
    )
    tl.store(c1_ptrs, y, mask=y_mask)


# configs/gemm/gfx950-GEMM-A8W8_BLOCKSCALE-N=32768-K=512.json
_CONFIGS = {
    # Task change: the upstream M_LEQ_8 row (BLOCK_SIZE_M 8, BLOCK_SIZE_N 16, one warp) is removed, so M <= 8 uses the
    # M_LEQ_16 row (16-row tiles, the smallest MFMA height); see the README.
    "M_LEQ_16": dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=32, BLOCK_SIZE_K=128, GROUP_SIZE_M=4, num_warps=1, num_stages=3, waves_per_eu=1, matrix_instr_nonkdim=16, cache_modifier='.cg', NUM_KSPLIT=1),
    "M_LEQ_32": dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=128, GROUP_SIZE_M=1, num_warps=1, num_stages=3, waves_per_eu=1, matrix_instr_nonkdim=16, cache_modifier='.cg', NUM_KSPLIT=1),
    "M_LEQ_64": dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=32, BLOCK_SIZE_K=128, GROUP_SIZE_M=8, num_warps=1, num_stages=3, waves_per_eu=1, matrix_instr_nonkdim=16, cache_modifier='.cg', NUM_KSPLIT=1),
    "M_LEQ_128": dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=128, GROUP_SIZE_M=1, num_warps=1, num_stages=2, waves_per_eu=1, matrix_instr_nonkdim=16, cache_modifier='.cg', NUM_KSPLIT=1),
    "M_LEQ_256": dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128, GROUP_SIZE_M=1, num_warps=4, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=16, cache_modifier=None, NUM_KSPLIT=1),
    "M_LEQ_512": dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=128, GROUP_SIZE_M=1, num_warps=1, num_stages=2, waves_per_eu=1, matrix_instr_nonkdim=16, cache_modifier=None, NUM_KSPLIT=1),
    "any": dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128, GROUP_SIZE_M=16, num_warps=2, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=32, cache_modifier=None, NUM_KSPLIT=1),
}


_STANDARD_M_BOUNDS = (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)


def _table_config(M):
    """Row of the tuned table for M, in the lookup order of AITER's get_gemm_config."""
    for bound in _STANDARD_M_BOUNDS:
        key = f"M_LEQ_{bound}"
        if M <= bound and key in _CONFIGS:
            return dict(_CONFIGS[key])
    for bound in reversed(_STANDARD_M_BOUNDS):
        key = f"M_GEQ_{bound}"
        if M >= bound and key in _CONFIGS:
            return dict(_CONFIGS[key])
    return dict(_CONFIGS["any"])


def _get_config(M, N, K):
    # Upstream: get_gemm_config("GEMM-A8W8_BLOCKSCALE", M, N, K); the table above is its N=32768, K=512 file.
    return _table_config(M)


def run(*, x, x_scale, w, w_scale, k_pe, k, v) -> None:
    """(x @ w^T) viewed as [M, D, S1 + S2]; k = cat(first S1, k_pe broadcast over D), v = last S2 (wrapper's c1, c2)."""
    M, K = x.shape
    N, K = w.shape
    D = v.shape[1]
    S2 = v.shape[2]
    S3 = k_pe.shape[1]
    S1 = k.shape[2] - S3
    # The wrapper takes y [M, D, S3]; the rope key is shared by every head, so it is broadcast without a copy.
    y = k_pe.unsqueeze(1).expand(M, D, S3)

    # Transpose w and w_scale
    w = w.T  # (K, N)
    w_scale = w_scale.T  # (scale_k, scale_n)

    config = _get_config(M, N, K)

    c1 = k
    c2 = v

    config["SPLITK_BLOCK_SIZE"] = triton.cdiv(
        K, config["NUM_KSPLIT"]
    )  # How big each split_k partition is
    if config["NUM_KSPLIT"] > 1:
        c_pp = torch.empty(
            (config["NUM_KSPLIT"], M, N),
            dtype=torch.float32,
            device=x.device,
        )
    else:
        c_pp = None

    # If block size is greater than split k size, shrink the block size
    if config["BLOCK_SIZE_K"] > config["SPLITK_BLOCK_SIZE"]:
        config["BLOCK_SIZE_K"] = triton.next_power_of_2(config["SPLITK_BLOCK_SIZE"])
        if config["BLOCK_SIZE_K"] > config["SPLITK_BLOCK_SIZE"]:
            config["BLOCK_SIZE_K"] = config["BLOCK_SIZE_K"] // 4
    config["BLOCK_SIZE_K"] = max(
        config["BLOCK_SIZE_K"], 16
    )  # minimum block size is 16 for perf

    # Scale block sizes
    config["GROUP_K"] = triton.next_power_of_2(
        triton.cdiv(K, w_scale.shape[0])
    )  # scale_block_size_k
    config["GROUP_N"] = triton.next_power_of_2(
        triton.cdiv(N, w_scale.shape[1])
    )  # scale_block_size_n

    # S3 block sizes
    config["BLOCK_SIZE_S3"] = triton.next_power_of_2(
        triton.cdiv(D * S3, triton.cdiv(N, config["BLOCK_SIZE_N"]))
    )

    assert (
        config["GROUP_K"] == config["BLOCK_SIZE_K"]
    ), "GROUP_K must equal BLOCK_SIZE_K"

    grid = lambda META: (  # noqa: E731
        (
            META["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"])
        ),  # Effective launch grid dims: [NUM_KSPLIT, NUM_M_BLOCKS, NUM_N_BLOCKS]
    )
    _fused_gemm_a8w8_blockscale_split_cat[grid](
        x,
        w,
        y,
        c1 if config["NUM_KSPLIT"] == 1 else c_pp,
        c2,
        x_scale,
        w_scale,
        M,
        N,
        K,
        D,
        S1,
        S2,
        S3,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        c1.stride(1) if config["NUM_KSPLIT"] == 1 else c_pp.stride(0),
        c1.stride(0) if config["NUM_KSPLIT"] == 1 else c_pp.stride(1),
        c1.stride(2) if config["NUM_KSPLIT"] == 1 else c_pp.stride(2),
        c2.stride(0),
        c2.stride(1),
        c2.stride(2),
        x_scale.stride(0),
        x_scale.stride(1),
        w_scale.stride(0),
        w_scale.stride(1),
        **config,
    )

    if config["NUM_KSPLIT"] > 1:
        REDUCE_BLOCK_SIZE_M = 32
        REDUCE_BLOCK_SIZE_N = 32
        REDUCE_BLOCK_SIZE_S3 = triton.next_power_of_2(
            triton.cdiv(D * S3, triton.cdiv(N, REDUCE_BLOCK_SIZE_N))
        )
        ACTUAL_KSPLIT = triton.cdiv(K, config["SPLITK_BLOCK_SIZE"])

        grid_reduce = (
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
        )
        _fused_gemm_a8w8_blockscale_split_cat_reduce[grid_reduce](
            c_pp,
            c1,
            c2,
            y,
            M,
            N,
            D,
            S1,
            S2,
            S3,
            c_pp.stride(0),
            c_pp.stride(1),
            c_pp.stride(2),
            c1.stride(0),
            c1.stride(1),
            c1.stride(2),
            c2.stride(0),
            c2.stride(1),
            c2.stride(2),
            y.stride(0),
            y.stride(1),
            y.stride(2),
            REDUCE_BLOCK_SIZE_M,
            REDUCE_BLOCK_SIZE_N,
            REDUCE_BLOCK_SIZE_S3,
            ACTUAL_KSPLIT,
            triton.next_power_of_2(config["NUM_KSPLIT"]),
        )
