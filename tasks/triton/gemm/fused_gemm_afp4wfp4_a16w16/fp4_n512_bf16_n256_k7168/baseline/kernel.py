# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


"""Fused MXFP4 GEMM and bf16 GEMM from ROCm/aiter (Triton): one launch computes the MXFP4 gate-up projection
(x_fp4 @ w_fp4^T, e2m1 codes with e8m0 scales per 32 values of K) and the bf16 MoE router gate (x_bf16 @ w_bf16^T).

Provenance: the device kernels of aiter/ops/triton/_triton_kernels/gemm/fused/fused_gemm_afp4wfp4_a16w16.py
(`_fused_gemm_afp4wfp4_a16w16_kernel` and `_fused_gemm_afp4wfp4_a16w16_reduce_kernel`; the preshuffle kernel is dropped),
the non-preshuffled path of the launcher `fused_gemm_afp4wfp4_a16w16` of aiter/ops/triton/gemm/fused/fused_gemm_afp4wfp4_a16w16.py
(its ahead-of-time metadata lookup is dropped), `get_splitk` of aiter/ops/triton/gemm/basic/gemm_afp4wfp4.py, and `remap_xcd`,
`pid_grid` and `make_kernel_repr` of aiter/ops/triton/utils/_triton/. The tuned table is the shape-specialized gfx950 file
configs/gemm/gfx950-FUSED-GEMM-AFP4WFP4-A16W16-N4=512-N16=256-K=7168.json, inlined with the get_gemm_config lookup order.
Split-K writes float32 partial sums that the reduce kernel adds.
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


_fused_gemm_afp4wfp4_a16w16_repr = make_kernel_repr(
    "_fused_gemm_afp4wfp4_a16w16_kernel",
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
    ],
)


@triton.heuristics(
    {
        "EVEN_K": lambda args: (args["K"] % (args["BLOCK_SIZE_K"] // 2) == 0)
        and (args["SPLITK_BLOCK_SIZE"] % args["BLOCK_SIZE_K"] == 0)
        and (args["K"] % (args["SPLITK_BLOCK_SIZE"] // 2) == 0),
        "GRID_MN_FP4": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N_fp4"], args["BLOCK_SIZE_N"]),
        "GRID_MN_BF16": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N_bf16"], args["BLOCK_SIZE_N"]),
    }
)
@triton.jit(repr=_fused_gemm_afp4wfp4_a16w16_repr)
def _fused_gemm_afp4wfp4_a16w16_kernel(
    # Pointers to matrices
    a_fp4_ptr,
    b_fp4_ptr,
    bias_fp4_ptr,
    a_fp4_scale_ptr,
    b_fp4_scale_ptr,
    c_fp4_ptr,
    a_bf16_ptr,
    b_bf16_ptr,
    bias_bf16_ptr,
    c_bf16_ptr,
    # Matrix dimensions
    M,
    N_fp4,
    N_bf16,
    K,
    stride_a_fp4_m,
    stride_a_fp4_k,
    stride_b_fp4_k,
    stride_b_fp4_n,
    stride_a_fp4_scale_m,
    stride_a_fp4_scale_k,
    stride_b_fp4_scale_n,
    stride_b_fp4_scale_k,
    stride_c_fp4_k,
    stride_c_fp4_m,
    stride_c_fp4_n,
    stride_a_bf16_m,
    stride_a_bf16_k,
    stride_b_bf16_k,
    stride_b_bf16_n,
    stride_c_bf16_k,
    stride_c_bf16_m,
    stride_c_bf16_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    ADD_BIAS_FP4: tl.constexpr,
    ADD_BIAS_BF16: tl.constexpr,
    EVEN_K: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
    waves_per_eu: tl.constexpr,
    matrix_instr_nonkdim: tl.constexpr,
    GRID_MN_FP4: tl.constexpr,
    GRID_MN_BF16: tl.constexpr,
    SKIP_REDUCE: tl.constexpr,
    cache_modifier: tl.constexpr,
):

    tl.assume(stride_a_fp4_m > 0)
    tl.assume(stride_a_fp4_k > 0)
    tl.assume(stride_b_fp4_k > 0)
    tl.assume(stride_b_fp4_n > 0)
    tl.assume(stride_c_fp4_k > 0)
    tl.assume(stride_c_fp4_m > 0)
    tl.assume(stride_c_fp4_n > 0)
    tl.assume(stride_a_fp4_scale_m > 0)
    tl.assume(stride_a_fp4_scale_k > 0)
    tl.assume(stride_b_fp4_scale_k > 0)
    tl.assume(stride_b_fp4_scale_n > 0)

    tl.assume(stride_a_bf16_m > 0)
    tl.assume(stride_a_bf16_k > 0)
    tl.assume(stride_b_bf16_k > 0)
    tl.assume(stride_b_bf16_n > 0)
    tl.assume(stride_c_bf16_m > 0)
    tl.assume(stride_c_bf16_n > 0)

    SCALE_GROUP_SIZE: tl.constexpr = 32
    GRID_MN: tl.constexpr = GRID_MN_FP4 + GRID_MN_BF16

    pid_unified = tl.program_id(axis=0)
    pid_unified = remap_xcd(pid_unified, GRID_MN * NUM_KSPLIT, NUM_XCDS=8)

    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n_fp4 = tl.cdiv(N_fp4, BLOCK_SIZE_N)
    num_pid_n_bf16 = tl.cdiv(N_bf16, BLOCK_SIZE_N)
    num_pid_n = num_pid_n_fp4 + num_pid_n_bf16

    if NUM_KSPLIT == 1:
        pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(pid_k >= 0)

    if (pid_k * SPLITK_BLOCK_SIZE // 2) < K:

        num_k_iter = tl.cdiv(SPLITK_BLOCK_SIZE // 2, BLOCK_SIZE_K // 2)

        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M

        if pid_n < num_pid_n_fp4:
            offs_k_fp4 = tl.arange(0, BLOCK_SIZE_K // 2)
            offs_k_fp4_split = pid_k * (SPLITK_BLOCK_SIZE // 2) + offs_k_fp4
            offs_b_fp4_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N_fp4
            a_fp4_ptrs = a_fp4_ptr + (
                offs_am[:, None] * stride_a_fp4_m
                + offs_k_fp4_split[None, :] * stride_a_fp4_k
            )
            b_fp4_ptrs = b_fp4_ptr + (
                offs_k_fp4_split[:, None] * stride_b_fp4_k
                + offs_b_fp4_n[None, :] * stride_b_fp4_n
            )

            offs_k_fp4_scale = (
                pid_k * (SPLITK_BLOCK_SIZE // SCALE_GROUP_SIZE)
            ) + tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_SIZE)
            a_fp4_scale_ptrs = (
                a_fp4_scale_ptr
                + offs_am[:, None] * stride_a_fp4_scale_m
                + offs_k_fp4_scale[None, :] * stride_a_fp4_scale_k
            )
            # B scales are N x K even though B operand is K x N.
            b_fp4_scale_ptrs = (
                b_fp4_scale_ptr
                + offs_b_fp4_n[:, None] * stride_b_fp4_scale_n
                + offs_k_fp4_scale[None, :] * stride_b_fp4_scale_k
            )

            if ADD_BIAS_FP4:
                if NUM_KSPLIT == 1 or (SKIP_REDUCE and pid_k == 0):
                    accumulator_fp4 = tl.load(bias_fp4_ptr + offs_b_fp4_n).to(
                        dtype=tl.float32
                    )
                    accumulator_fp4 = tl.broadcast_to(
                        accumulator_fp4[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N)
                    )
                else:
                    accumulator_fp4 = tl.zeros(
                        (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32
                    )
            else:
                accumulator_fp4 = tl.zeros(
                    (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32
                )

            for k in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter):
                a_scale = tl.load(a_fp4_scale_ptrs)
                b_scale = tl.load(b_fp4_scale_ptrs, cache_modifier=cache_modifier)

                if EVEN_K:
                    a = tl.load(a_fp4_ptrs)
                    b = tl.load(b_fp4_ptrs, cache_modifier=cache_modifier)
                else:
                    a = tl.load(
                        a_fp4_ptrs,
                        mask=offs_k_fp4[None, :] < K - k * (BLOCK_SIZE_K // 2),
                        other=0.0,
                    )
                    b = tl.load(
                        b_fp4_ptrs,
                        mask=offs_k_fp4[:, None] < K - k * (BLOCK_SIZE_K // 2),
                        other=0.0,
                        cache_modifier=cache_modifier,
                    )

                accumulator_fp4 = tl.dot_scaled(
                    a, a_scale, "e2m1", b, b_scale, "e2m1", acc=accumulator_fp4
                )

                a_fp4_ptrs += (BLOCK_SIZE_K // 2) * stride_a_fp4_k
                b_fp4_ptrs += (BLOCK_SIZE_K // 2) * stride_b_fp4_k
                a_fp4_scale_ptrs += (
                    BLOCK_SIZE_K // SCALE_GROUP_SIZE
                ) * stride_a_fp4_scale_k
                b_fp4_scale_ptrs += (
                    BLOCK_SIZE_K // SCALE_GROUP_SIZE
                ) * stride_b_fp4_scale_k

            c_fp4 = accumulator_fp4.to(c_fp4_ptr.type.element_ty)

            offs_cm = pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(
                tl.int64
            )
            offs_c_fp4_n = pid_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(
                0, BLOCK_SIZE_N
            ).to(tl.int64)
            c_fp4_ptrs = (
                c_fp4_ptr
                + stride_c_fp4_m * offs_cm[:, None]
                + stride_c_fp4_n * offs_c_fp4_n[None, :]
                + pid_k * stride_c_fp4_k
            )
            c_fp4_mask = (offs_cm[:, None] < M) & (offs_c_fp4_n[None, :] < N_fp4)
            tl.store(c_fp4_ptrs, c_fp4, mask=c_fp4_mask)
        else:
            pid_n -= num_pid_n_fp4
            offs_k_bf16 = tl.arange(0, BLOCK_SIZE_K)
            offs_k_bf16_split = pid_k * SPLITK_BLOCK_SIZE + offs_k_bf16
            K = 2 * K

            offs_b_bf16_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N_bf16
            a_ptrs = a_bf16_ptr + (
                offs_am[:, None] * stride_a_bf16_m
                + offs_k_bf16_split[None, :] * stride_a_bf16_k
            )
            b_ptrs = b_bf16_ptr + (
                offs_k_bf16_split[:, None] * stride_b_bf16_k
                + offs_b_bf16_n[None, :] * stride_b_bf16_n
            )

            if ADD_BIAS_BF16:
                if NUM_KSPLIT == 1 or (SKIP_REDUCE and pid_k == 0):
                    accumulator_bf16 = tl.load(bias_bf16_ptr + offs_b_bf16_n).to(
                        dtype=tl.float32
                    )
                    accumulator_bf16 = tl.broadcast_to(
                        accumulator_bf16[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N)
                    )
                else:
                    accumulator_bf16 = tl.zeros(
                        (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32
                    )
            else:
                accumulator_bf16 = tl.zeros(
                    (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32
                )

            for k in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter):
                if EVEN_K:
                    a = tl.load(a_ptrs)
                    b = tl.load(b_ptrs, cache_modifier=cache_modifier)
                else:
                    a = tl.load(
                        a_ptrs,
                        mask=offs_k_bf16[None, :] < K - k * BLOCK_SIZE_K,
                        other=0.0,
                    )
                    b = tl.load(
                        b_ptrs,
                        mask=offs_k_bf16[:, None] < K - k * BLOCK_SIZE_K,
                        other=0.0,
                        cache_modifier=cache_modifier,
                    )

                accumulator_bf16 = tl.dot(a, b, acc=accumulator_bf16)

                a_ptrs += BLOCK_SIZE_K * stride_a_bf16_k
                b_ptrs += BLOCK_SIZE_K * stride_b_bf16_k

            c_bf16 = accumulator_bf16.to(c_bf16_ptr.type.element_ty)

            # Write back the block of the output matrix C with masks.
            offs_cm = pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(
                tl.int64
            )
            offs_c_bf16_n = pid_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(
                0, BLOCK_SIZE_N
            ).to(tl.int64)
            c_bf16_ptrs = (
                c_bf16_ptr
                + stride_c_bf16_m * offs_cm[:, None]
                + stride_c_bf16_n * offs_c_bf16_n[None, :]
                + pid_k * stride_c_bf16_k
            )
            c_bf16_mask = (offs_cm[:, None] < M) & (offs_c_bf16_n[None, :] < N_bf16)
            tl.store(c_bf16_ptrs, c_bf16, mask=c_bf16_mask)


_gemm_afp4wfp4_a16w16_reduce_repr = make_kernel_repr(
    "_fused_gemm_afp4wfp4_a16w16_reduce_kernel",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "ACTUAL_KSPLIT",
        "MAX_KSPLIT",
        "ADD_BIAS_FP4",
        "ADD_BIAS_BF16",
    ],
)


@triton.heuristics({})  # dummy heuristics to invoke kernel re-naming
@triton.jit(repr=_gemm_afp4wfp4_a16w16_reduce_repr)
def _fused_gemm_afp4wfp4_a16w16_reduce_kernel(
    bias_fp4_ptr,
    c_fp4_in_ptr,
    c_fp4_out_ptr,
    bias_bf16_ptr,
    c_bf16_in_ptr,
    c_bf16_out_ptr,
    M,
    N_fp4,
    N_bf16,
    stride_c_fp4_in_k,
    stride_c_fp4_in_m,
    stride_c_fp4_in_n,
    stride_c_fp4_out_m,
    stride_c_fp4_out_n,
    stride_c_bf16_in_k,
    stride_c_bf16_in_m,
    stride_c_bf16_in_n,
    stride_c_bf16_out_m,
    stride_c_bf16_out_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    ACTUAL_KSPLIT: tl.constexpr,
    MAX_KSPLIT: tl.constexpr,
    ADD_BIAS_FP4: tl.constexpr,
    ADD_BIAS_BF16: tl.constexpr,
):

    tl.assume(stride_c_fp4_in_k > 0)
    tl.assume(stride_c_fp4_in_m > 0)
    tl.assume(stride_c_fp4_in_n > 0)
    tl.assume(stride_c_fp4_out_m > 0)
    tl.assume(stride_c_fp4_out_n > 0)

    tl.assume(stride_c_bf16_in_k > 0)
    tl.assume(stride_c_bf16_in_m > 0)
    tl.assume(stride_c_bf16_in_n > 0)
    tl.assume(stride_c_bf16_out_m > 0)
    tl.assume(stride_c_bf16_out_n > 0)

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M

    num_pid_n_fp4 = tl.cdiv(N_fp4, BLOCK_SIZE_N)
    offs_k = tl.arange(0, MAX_KSPLIT)
    acc_dtype = tl.float32 if c_fp4_in_ptr.type.element_ty != tl.int8 else tl.int32

    if pid_n < num_pid_n_fp4:
        offs_fp4_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N_fp4
        c_fp4_in_ptrs = (
            c_fp4_in_ptr
            + (offs_k[:, None, None] * stride_c_fp4_in_k)
            + (offs_m[None, :, None] * stride_c_fp4_in_m)
            + (offs_fp4_n[None, None, :] * stride_c_fp4_in_n)
        )

        if ACTUAL_KSPLIT == MAX_KSPLIT:
            c_fp4 = tl.load(c_fp4_in_ptrs)
        else:
            c_fp4 = tl.load(
                c_fp4_in_ptrs, mask=offs_k[:, None, None] < ACTUAL_KSPLIT, other=0.0
            )
        c_fp4 = tl.sum(c_fp4, axis=0)
        if ADD_BIAS_FP4:
            bias_fp4 = tl.load(bias_fp4_ptr + offs_fp4_n).to(dtype=acc_dtype)
            bias_fp4 = tl.broadcast_to(bias_fp4[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N))
            c_fp4 += bias_fp4

        c_fp4 = c_fp4.to(c_fp4_out_ptr.type.element_ty)

        c_fp4_out_ptrs = (
            c_fp4_out_ptr
            + (offs_m[:, None] * stride_c_fp4_out_m)
            + (offs_fp4_n[None, :] * stride_c_fp4_out_n)
        )

        tl.store(c_fp4_out_ptrs, c_fp4)
    else:
        pid_n -= num_pid_n_fp4

        offs_bf16_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N_bf16
        c_bf16_in_ptrs = (
            c_bf16_in_ptr
            + (offs_k[:, None, None] * stride_c_bf16_in_k)
            + (offs_m[None, :, None] * stride_c_bf16_in_m)
            + (offs_bf16_n[None, None, :] * stride_c_bf16_in_n)
        )

        if ACTUAL_KSPLIT == MAX_KSPLIT:
            c_bf16 = tl.load(c_bf16_in_ptrs)
        else:
            c_bf16 = tl.load(
                c_bf16_in_ptrs, mask=offs_k[:, None, None] < ACTUAL_KSPLIT, other=0.0
            )
        c_bf16 = tl.sum(c_bf16, axis=0)
        if ADD_BIAS_BF16:
            bias_bf16 = tl.load(bias_bf16_ptr + offs_bf16_n).to(dtype=acc_dtype)
            bias_bf16 = tl.broadcast_to(
                bias_bf16[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N)
            )
            c_bf16 += bias_bf16

        c_bf16 = c_bf16.to(c_bf16_out_ptr.type.element_ty)

        c_bf16_out_ptrs = (
            c_bf16_out_ptr
            + (offs_m[:, None] * stride_c_bf16_out_m)
            + (offs_bf16_n[None, :] * stride_c_bf16_out_n)
        )
        c_bf16_mask = (offs_m[:, None] < M) & (offs_bf16_n[None, :] < N_bf16)
        tl.store(c_bf16_out_ptrs, c_bf16, mask=c_bf16_mask)


# configs/gemm/gfx950-FUSED-GEMM-AFP4WFP4-A16W16-N4=512-N16=256-K=7168.json
_CONFIGS = {
    "M_LEQ_8": dict(BLOCK_SIZE_M=8, BLOCK_SIZE_N=16, BLOCK_SIZE_K=256, GROUP_SIZE_M=1, num_warps=2, num_stages=2, waves_per_eu=1, matrix_instr_nonkdim=16, cache_modifier='.cg', NUM_KSPLIT=14),
    "M_LEQ_16": dict(BLOCK_SIZE_M=8, BLOCK_SIZE_N=16, BLOCK_SIZE_K=256, GROUP_SIZE_M=1, num_warps=2, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=16, cache_modifier=None, NUM_KSPLIT=16),
    "M_LEQ_32": dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=512, GROUP_SIZE_M=1, num_warps=8, num_stages=2, waves_per_eu=6, matrix_instr_nonkdim=16, cache_modifier=None, NUM_KSPLIT=7),
    "M_LEQ_64": dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=16, BLOCK_SIZE_K=256, GROUP_SIZE_M=1, num_warps=8, num_stages=2, waves_per_eu=1, matrix_instr_nonkdim=16, cache_modifier='.cg', NUM_KSPLIT=7),
    "M_LEQ_128": dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=32, BLOCK_SIZE_K=256, GROUP_SIZE_M=1, num_warps=2, num_stages=2, waves_per_eu=4, matrix_instr_nonkdim=16, cache_modifier=None, NUM_KSPLIT=4),
    "M_LEQ_256": dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=256, GROUP_SIZE_M=1, num_warps=4, num_stages=3, waves_per_eu=2, matrix_instr_nonkdim=16, cache_modifier=None, NUM_KSPLIT=1),
    "any": dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=64, BLOCK_SIZE_K=128, GROUP_SIZE_M=8, num_warps=8, num_stages=3, waves_per_eu=2, matrix_instr_nonkdim=16, cache_modifier=None, NUM_KSPLIT=1),
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


def _get_config(M, N_fp4, N_bf16, K):
    # Custom file naming upstream: N4={N_fp4}-N16={N_bf16}-K={2*K}; the table above is that file for (512, 256, 7168).
    return _table_config(M)


def run(*, x_fp4, x_fp4_scale, w_fp4, w_fp4_scale, x_bf16, w_bf16, y_a4w4, y_a16w16) -> None:
    """y_a4w4 = MXFP4 x_fp4 @ w_fp4^T and y_a16w16 = x_bf16 @ w_bf16^T (the wrapper's y_fp4 and y_bf16)."""
    M, K = x_fp4.shape  # K counts packed bytes (two fp4 values each)
    N_fp4, K = w_fp4.shape
    M, _ = x_bf16.shape
    N_bf16, _ = w_bf16.shape

    # Transpose w
    w_fp4 = w_fp4.T
    w_bf16 = w_bf16.T

    config = _get_config(M, N_fp4, N_bf16, K)

    if config["NUM_KSPLIT"] > 1:
        SPLITK_BLOCK_SIZE, BLOCK_SIZE_K, NUM_KSPLIT = get_splitk(
            K, config["BLOCK_SIZE_K"], config["NUM_KSPLIT"]
        )

        config["SPLITK_BLOCK_SIZE"] = SPLITK_BLOCK_SIZE
        config["BLOCK_SIZE_K"] = BLOCK_SIZE_K
        config["NUM_KSPLIT"] = NUM_KSPLIT
        config["NUM_KSPLIT"] = triton.cdiv(K, config["SPLITK_BLOCK_SIZE"] // 2)
    else:
        config["SPLITK_BLOCK_SIZE"] = 2 * K

    if config["BLOCK_SIZE_K"] >= 2 * K:
        config["BLOCK_SIZE_K"] = triton.next_power_of_2(2 * K)
        config["SPLITK_BLOCK_SIZE"] = 2 * K
        config["NUM_KSPLIT"] = 1

    if config["NUM_KSPLIT"] > 1:
        y_fp4_pp = torch.empty(
            (config["NUM_KSPLIT"], M, N_fp4),
            dtype=torch.float32,
            device=x_fp4.device,
        )
        y_bf16_pp = torch.empty(
            (config["NUM_KSPLIT"], M, N_bf16),
            dtype=torch.float32,
            device=x_bf16.device,
        )
    else:
        y_fp4_pp = None
        y_bf16_pp = None

    config["BLOCK_SIZE_N"] = max(config["BLOCK_SIZE_N"], 32)

    grid = lambda META: (  # noqa: E731
        (
            META["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * (
                triton.cdiv(N_fp4, META["BLOCK_SIZE_N"])
                + triton.cdiv(N_bf16, META["BLOCK_SIZE_N"])
            )
        ),
    )
    _fused_gemm_afp4wfp4_a16w16_kernel[grid](
        x_fp4,
        w_fp4,
        None,
        x_fp4_scale,
        w_fp4_scale,
        y_a4w4 if config["NUM_KSPLIT"] == 1 else y_fp4_pp,
        x_bf16,
        w_bf16,
        None,
        y_a16w16 if config["NUM_KSPLIT"] == 1 else y_bf16_pp,
        M,
        N_fp4,
        N_bf16,
        K,
        x_fp4.stride(0),
        x_fp4.stride(1),
        w_fp4.stride(0),
        w_fp4.stride(1),
        x_fp4_scale.stride(0),
        x_fp4_scale.stride(1),
        w_fp4_scale.stride(0),
        w_fp4_scale.stride(1),
        0 if config["NUM_KSPLIT"] == 1 else y_fp4_pp.stride(0),
        y_a4w4.stride(0) if config["NUM_KSPLIT"] == 1 else y_fp4_pp.stride(1),
        y_a4w4.stride(1) if config["NUM_KSPLIT"] == 1 else y_fp4_pp.stride(2),
        x_bf16.stride(0),
        x_bf16.stride(1),
        w_bf16.stride(0),
        w_bf16.stride(1),
        0 if config["NUM_KSPLIT"] == 1 else y_bf16_pp.stride(0),
        y_a16w16.stride(0) if config["NUM_KSPLIT"] == 1 else y_bf16_pp.stride(1),
        y_a16w16.stride(1) if config["NUM_KSPLIT"] == 1 else y_bf16_pp.stride(2),
        ADD_BIAS_FP4=False,
        ADD_BIAS_BF16=False,
        SKIP_REDUCE=False,
        **config,
    )

    if config["NUM_KSPLIT"] > 1:
        REDUCE_BLOCK_SIZE_M = 32
        REDUCE_BLOCK_SIZE_N = 32
        ACTUAL_KSPLIT = triton.cdiv(K, config["SPLITK_BLOCK_SIZE"] // 2)

        grid_reduce = (
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N_fp4, REDUCE_BLOCK_SIZE_N)
            + triton.cdiv(N_bf16, REDUCE_BLOCK_SIZE_N),
        )
        _fused_gemm_afp4wfp4_a16w16_reduce_kernel[grid_reduce](
            None,
            y_fp4_pp,
            y_a4w4,
            None,
            y_bf16_pp,
            y_a16w16,
            M,
            N_fp4,
            N_bf16,
            y_fp4_pp.stride(0),
            y_fp4_pp.stride(1),
            y_fp4_pp.stride(2),
            y_a4w4.stride(0),
            y_a4w4.stride(1),
            y_bf16_pp.stride(0),
            y_bf16_pp.stride(1),
            y_bf16_pp.stride(2),
            y_a16w16.stride(0),
            y_a16w16.stride(1),
            REDUCE_BLOCK_SIZE_M,
            REDUCE_BLOCK_SIZE_N,
            ACTUAL_KSPLIT,
            triton.next_power_of_2(config["NUM_KSPLIT"]),
            ADD_BIAS_FP4=False,
            ADD_BIAS_BF16=False,
        )
