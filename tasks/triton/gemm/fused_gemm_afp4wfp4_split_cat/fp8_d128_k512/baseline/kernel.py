# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


"""MXFP4 GEMM fused with the MLA key/value split and concatenation from ROCm/aiter (Triton): the product x @ w^T is
viewed as [M, D, S1 + S2]; the first S1 columns of each head are concatenated with a shared rope key y [M, D, S3] into k, the
last S2 become v. The outputs take the dtype of the output tensors (float8_e4m3fn in the scenario).

Provenance: the device kernels of aiter/ops/triton/_triton_kernels/gemm/fused/fused_gemm_afp4wfp4_split_cat.py
(`_fused_gemm_afp4wfp4_split_cat` and `_fused_gemm_afp4wfp4_split_cat_reduce`; the preshuffle kernel is dropped), the
non-preshuffled launcher `fused_gemm_afp4wfp4_split_cat` of aiter/ops/triton/gemm/fused/fused_gemm_afp4wfp4_split_cat.py,
`get_splitk` of aiter/ops/triton/gemm/basic/gemm_afp4wfp4.py, and `remap_xcd`, `pid_grid` and `make_kernel_repr` of
aiter/ops/triton/utils/_triton/. The tuned table is the shape-specialized gfx950 file
configs/gfx950/triton/gemm/GEMM-AFP4WFP4-N=32768-K=512.json, inlined with the get_gemm_config lookup order. The table's NUM_KSPLIT 4 for M <= 8
collapses to a single split under get_splitk for K = 512, so the reduce kernel is never launched for this shape; it is kept
with the kernel it belongs to.
"""


import torch
import triton
import triton.language as tl


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

    # for lists, tuples, sets - recursively join each
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


def make_kernel_repr(base_name, config_keys, name_key=None):
    # When name_key is set, the base name is taken from the matching constexpr
    # kwarg at call time (falling back to base_name if missing/empty). Lets a
    # single shared kernel produce caller-specific names in compiled artifacts.
    def _repr(specialization):
        constants = specialization.constants

        name = base_name
        if name_key is not None:
            override = constants.get(name_key, None)
            if override:
                cleaned = "".join(
                    ch if ch.isalnum() or ch == "_" else "_" for ch in str(override)
                )
                if cleaned:
                    name = cleaned

        name_parts = []
        for key in config_keys:
            value = constants.get(key, None)
            symbol = _sanitize_constexpr_value(value)
            name_parts.append(f"{key}_{symbol}")

        if not name_parts:
            return name

        suffix = "_".join(name_parts)
        return f"{name}_{suffix}"

    return _repr


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


def get_splitk(K: int, BLOCK_SIZE_K: int, NUM_KSPLIT: int):
    # heuristics for make "EVEN_K == True" as much as possible
    NUM_KSPLIT_STEP = 2
    BLOCK_SIZE_K_STEP = 2
    SPLITK_BLOCK_SIZE = (
        triton.cdiv((2 * triton.cdiv(K, NUM_KSPLIT)), BLOCK_SIZE_K) * BLOCK_SIZE_K
    )
    while NUM_KSPLIT > 1 and BLOCK_SIZE_K > 16:
        if (
            K % (SPLITK_BLOCK_SIZE // 2) == 0
            and SPLITK_BLOCK_SIZE % BLOCK_SIZE_K == 0
            and K % (BLOCK_SIZE_K // 2) == 0
        ):
            break
        elif K % (SPLITK_BLOCK_SIZE // 2) != 0 and NUM_KSPLIT > 1:
            NUM_KSPLIT = NUM_KSPLIT // NUM_KSPLIT_STEP
        elif SPLITK_BLOCK_SIZE % BLOCK_SIZE_K != 0:
            if NUM_KSPLIT > 1:
                NUM_KSPLIT = NUM_KSPLIT // NUM_KSPLIT_STEP
            elif BLOCK_SIZE_K > 16:
                BLOCK_SIZE_K = BLOCK_SIZE_K // BLOCK_SIZE_K_STEP
        elif K % (BLOCK_SIZE_K // 2) != 0 and BLOCK_SIZE_K > 16:
            BLOCK_SIZE_K = BLOCK_SIZE_K // BLOCK_SIZE_K_STEP
        else:
            break

        SPLITK_BLOCK_SIZE = (
            triton.cdiv((2 * triton.cdiv(K, NUM_KSPLIT)), BLOCK_SIZE_K) * BLOCK_SIZE_K
        )

    # re-ensuring NUM_KSPLIT is the correct value
    NUM_KSPLIT = triton.cdiv(K, (SPLITK_BLOCK_SIZE // 2))

    return SPLITK_BLOCK_SIZE, BLOCK_SIZE_K, NUM_KSPLIT


_fused_gemm_afp4wfp4_split_cat_repr = make_kernel_repr(
    "_fused_gemm_afp4wfp4_split_cat",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "NUM_KSPLIT",
        "SPLITK_BLOCK_SIZE",
        "EVEN_K",
        "cache_modifier",
    ],
)


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % (args["BLOCK_SIZE_K"] // 2) == 0,
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@triton.jit(repr=_fused_gemm_afp4wfp4_split_cat_repr)
def _fused_gemm_afp4wfp4_split_cat(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    y_ptr,
    c1_ptr,
    c2_ptr,
    a_scales_ptr,
    b_scales_ptr,
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
    stride_ascales_m,
    stride_ascales_k,
    stride_bscales_k,
    stride_bscales_n,
    # Meta-parameters
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
    tl.assume(stride_ascales_m > 0)
    tl.assume(stride_ascales_k > 0)
    tl.assume(stride_bscales_k > 0)
    tl.assume(stride_bscales_n > 0)

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid_unified = tl.program_id(axis=0)
    # remap so that XCDs get continous chunks of pids (of CHUNK_SIZE).
    pid_unified = remap_xcd(pid_unified, GRID_MN * NUM_KSPLIT, NUM_XCDS=8)

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

    SCALE_GROUP_SIZE: tl.constexpr = 32

    if (pid_k * SPLITK_BLOCK_SIZE // 2) < K:
        # SPLITK_BLOCK_SIZE = tl.cdiv(K, NUM_KSPLIT)
        num_k_iter = tl.cdiv(SPLITK_BLOCK_SIZE // 2, BLOCK_SIZE_K // 2)
        # ^ Number of K blocks within our split-K partition

        # Create pointers for first block of A and B input matrices
        offs_k = tl.arange(0, BLOCK_SIZE_K // 2)
        offs_k_split = pid_k * SPLITK_BLOCK_SIZE // 2 + offs_k
        offs_a_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_b_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        a_ptrs = a_ptr + (
            offs_a_m[:, None] * stride_a_m + offs_k_split[None, :] * stride_a_k
        )
        b_ptrs = b_ptr + (
            offs_k_split[:, None] * stride_b_k + offs_b_n[None, :] * stride_b_n
        )

        # Create pointers for the first block of A and B scales
        offs_ks = (pid_k * (SPLITK_BLOCK_SIZE // SCALE_GROUP_SIZE)) + tl.arange(
            0, BLOCK_SIZE_K // SCALE_GROUP_SIZE
        )
        a_scales_ptrs = (
            a_scales_ptr
            + offs_a_m[:, None] * stride_ascales_m
            + offs_ks[None, :] * stride_ascales_k
        )
        # B scales are N x K even though B operand is K x N.
        b_scales_ptrs = (
            b_scales_ptr
            + offs_b_n[:, None] * stride_bscales_n
            + offs_ks[None, :] * stride_bscales_k
        )

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter):
            a_scales = tl.load(a_scales_ptrs)
            b_scales = tl.load(b_scales_ptrs, cache_modifier=cache_modifier)

            # Load the next block of A and B, generate a mask by checking the K dimension.
            # If it is out of bounds, set it to 0.
            if EVEN_K:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs, cache_modifier=cache_modifier)
            else:
                a = tl.load(
                    a_ptrs,
                    mask=offs_k[None, :] < K - k * (BLOCK_SIZE_K // 2),
                    other=0.0,
                )
                b = tl.load(
                    b_ptrs,
                    mask=offs_k[:, None] < K - k * (BLOCK_SIZE_K // 2),
                    other=0.0,
                )

            # Perform dot operation and apply scales
            accumulator = tl.dot_scaled(
                a, a_scales, "e2m1", b, b_scales, "e2m1", accumulator
            )

            # Advance the ptrs to the next K block.
            a_ptrs += (BLOCK_SIZE_K // 2) * stride_a_k
            b_ptrs += (BLOCK_SIZE_K // 2) * stride_b_k

            # k_cur = k * BLOCK_SIZE_K // GROUP_K
            # k_nxt = (k + 1) * BLOCK_SIZE_K // GROUP_K
            # offs_ks = k_nxt - k_cur
            a_scales_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_SIZE) * stride_ascales_k
            b_scales_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_SIZE) * stride_bscales_k

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
def _fused_gemm_afp4wfp4_split_cat_reduce(
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


# configs/gfx950/triton/gemm/GEMM-AFP4WFP4-N=32768-K=512.json
_CONFIGS = {
    "M_LEQ_8": dict(BLOCK_SIZE_M=8, BLOCK_SIZE_N=64, BLOCK_SIZE_K=512, GROUP_SIZE_M=1, num_warps=4, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=32, cache_modifier='.cg', NUM_KSPLIT=4),
    "M_LEQ_16": dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=1024, GROUP_SIZE_M=4, num_warps=4, num_stages=3, waves_per_eu=1, matrix_instr_nonkdim=16, cache_modifier='.cg', NUM_KSPLIT=1),
    "M_LEQ_32": dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=128, BLOCK_SIZE_K=256, GROUP_SIZE_M=1, num_warps=4, num_stages=2, waves_per_eu=3, matrix_instr_nonkdim=16, cache_modifier='.cg', NUM_KSPLIT=1),
    "M_LEQ_64": dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=256, GROUP_SIZE_M=1, num_warps=4, num_stages=2, waves_per_eu=4, matrix_instr_nonkdim=16, cache_modifier='.cg', NUM_KSPLIT=1),
    "M_LEQ_128": dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=256, GROUP_SIZE_M=1, num_warps=4, num_stages=3, waves_per_eu=2, matrix_instr_nonkdim=16, cache_modifier='.cg', NUM_KSPLIT=1),
    "M_LEQ_256": dict(BLOCK_SIZE_M=256, BLOCK_SIZE_N=128, BLOCK_SIZE_K=256, GROUP_SIZE_M=1, num_warps=8, num_stages=2, waves_per_eu=1, matrix_instr_nonkdim=16, cache_modifier='.cg', NUM_KSPLIT=1),
    "M_LEQ_512": dict(BLOCK_SIZE_M=256, BLOCK_SIZE_N=256, BLOCK_SIZE_K=256, GROUP_SIZE_M=2, num_warps=8, num_stages=2, waves_per_eu=1, matrix_instr_nonkdim=32, cache_modifier=None, NUM_KSPLIT=1),
    "any": dict(BLOCK_SIZE_M=256, BLOCK_SIZE_N=256, BLOCK_SIZE_K=256, GROUP_SIZE_M=2, num_warps=8, num_stages=2, waves_per_eu=1, matrix_instr_nonkdim=32, cache_modifier=None, NUM_KSPLIT=1),
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
    # Upstream: the basic afp4wfp4 _get_config, get_gemm_config("GEMM-AFP4WFP4", M, N, 2 * K); the table above is its
    # N=32768, K=512 file.
    return _table_config(M)


def run(*, x, x_scale, w, w_scale, k_pe, k, v) -> None:
    """(x @ w^T) viewed as [M, D, S1 + S2]; k = cat(first S1, k_pe broadcast over D), v = last S2 (wrapper's c1, c2)."""
    M, K = x.shape  # K counts packed bytes (two fp4 values each)
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
        SPLITK_BLOCK_SIZE, BLOCK_SIZE_K, NUM_KSPLIT = get_splitk(
            K, config["BLOCK_SIZE_K"], config["NUM_KSPLIT"]
        )

        config["SPLITK_BLOCK_SIZE"] = SPLITK_BLOCK_SIZE
        config["BLOCK_SIZE_K"] = BLOCK_SIZE_K
        config["NUM_KSPLIT"] = NUM_KSPLIT

    if config["BLOCK_SIZE_K"] >= 2 * K:
        config["BLOCK_SIZE_K"] = triton.next_power_of_2(2 * K)
        config["SPLITK_BLOCK_SIZE"] = 2 * K
        config["NUM_KSPLIT"] = 1

    if config["NUM_KSPLIT"] > 1:
        c_pp = torch.empty(
            (config["NUM_KSPLIT"], M, N),
            dtype=torch.float32,
            device=x.device,
        )
    else:
        config["SPLITK_BLOCK_SIZE"] = 2 * K
        c_pp = None

    # S3 block sizes
    config["BLOCK_SIZE_S3"] = triton.next_power_of_2(
        triton.cdiv(D * S3, triton.cdiv(N, config["BLOCK_SIZE_N"]))
    )

    grid = lambda META: (  # noqa: E731
        (
            META["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"])
        ),  # Effective launch grid dims: [NUM_KSPLIT, NUM_M_BLOCKS, NUM_N_BLOCKS]
    )
    _fused_gemm_afp4wfp4_split_cat[grid](
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
        ACTUAL_KSPLIT = triton.cdiv(K, (config["SPLITK_BLOCK_SIZE"] // 2))

        grid_reduce = (
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
        )
        _fused_gemm_afp4wfp4_split_cat_reduce[grid_reduce](
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
