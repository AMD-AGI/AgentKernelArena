# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


"""Block-scaled fp8 GEMM with a fused mul-add epilogue from ROCm/aiter (Triton): out = a * b + x @ w^T (fuse_type 1)
with 1x128 activation and 128x128 weight scales.

Provenance: the device kernels of aiter/ops/triton/_triton_kernels/gemm/fused/fused_gemm_a8w8_blockscale_mul_add.py
(`_fused_gemm_a8w8_blockscale_mul_add_kernel` and its reduce kernel), the launcher `fused_gemm_a8w8_blockscale_mul_add` of
aiter/ops/triton/gemm/fused/fused_gemm_a8w8_blockscale_mul_add.py, `compute_splitk_params` of
aiter/ops/triton/utils/gemm_config_utils.py, and `pid_grid` and `make_kernel_repr` of aiter/ops/triton/utils/_triton/. For
fuse_type 1 on gfx950 upstream selects the base GEMM-A8W8_BLOCKSCALE family, so the tuned table is
configs/gemm/gfx950-GEMM-A8W8_BLOCKSCALE-N=7168-K=256.json, inlined with the get_gemm_config lookup order. Every row has
NUM_KSPLIT 1, so the reduce kernel is never launched for this shape; it is kept with the kernel it belongs to.
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


def add_default_gemm_config_params(config: dict) -> dict:
    """
    this fn ensures that all configs have required default values.

    Args:
        config: Dictionary containing GEMM configuration parameters.

    Returns:
        same object as input
    """
    if "NUM_KSPLIT" not in config:
        config["NUM_KSPLIT"] = 1

    # adding default cache_modifier if not present as some kernels need this
    if "cache_modifier" not in config and "BLOCK_SIZE_K" in config:
        config["cache_modifier"] = None

    return config


def compute_splitk_params(config: dict, K: int) -> dict:
    """
    this fn calculates the SPLITK_BLOCK_SIZE and adjusts BLOCK_SIZE_K
    if necessary based on the NUM_KSPLIT value in the config.

    Args:
        config: Dictionary containing GEMM configuration parameters.
        K: K dimension of the GEMM operation (must be positive)

    Returns:
        same object as input
    """
    assert K > 0, "K must be positive"

    add_default_gemm_config_params(config)

    config["SPLITK_BLOCK_SIZE"] = triton.cdiv(K, config["NUM_KSPLIT"])

    if "BLOCK_SIZE_K" in config:
        # If NUM_KSPLIT makes K too small, then BLOCK_K will decrease to be smaller than
        # GROUP_K.
        while (
            config["NUM_KSPLIT"] > 1
            and config["BLOCK_SIZE_K"] > config["SPLITK_BLOCK_SIZE"]
        ):
            config["NUM_KSPLIT"] = max(config["NUM_KSPLIT"] // 2, 1)
            config["SPLITK_BLOCK_SIZE"] = triton.cdiv(K, config["NUM_KSPLIT"])

        # If BLOCK_SIZE_K is still too large with NUM_KSPLIT=1, fix it to equal K dim.
        if config["BLOCK_SIZE_K"] > config["SPLITK_BLOCK_SIZE"]:
            config["BLOCK_SIZE_K"] = triton.next_power_of_2(config["SPLITK_BLOCK_SIZE"])

            if config["BLOCK_SIZE_K"] > config["SPLITK_BLOCK_SIZE"]:
                config["BLOCK_SIZE_K"] = config["BLOCK_SIZE_K"] // 2

        config["BLOCK_SIZE_K"] = max(config["BLOCK_SIZE_K"], 16)

        # Round the SPLITK_BLOCK_SIZE to multiple of BLOCK_SIZE_K and update NUM_KSPLIT to again.
        if config["NUM_KSPLIT"] > 1 and (
            config["SPLITK_BLOCK_SIZE"] % config["BLOCK_SIZE_K"] != 0
        ):
            config["SPLITK_BLOCK_SIZE"] = (
                triton.cdiv(config["SPLITK_BLOCK_SIZE"], config["BLOCK_SIZE_K"])
                * config["BLOCK_SIZE_K"]
            )
            config["NUM_KSPLIT"] = triton.cdiv(K, config["SPLITK_BLOCK_SIZE"])

    return config


_fused_gemm_a8w8_blockscale_mul_add_repr = make_kernel_repr(
    "_fused_gemm_a8w8_blockscale_mul_add_kernel",
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
        "EVEN_K": lambda args: (args["K"] % args["BLOCK_SIZE_K"] == 0)
        and (args["SPLITK_BLOCK_SIZE"] % args["BLOCK_SIZE_K"] == 0)
        and (args["K"] % args["SPLITK_BLOCK_SIZE"] == 0),
    }
)
@triton.jit(repr=_fused_gemm_a8w8_blockscale_mul_add_repr)
def _fused_gemm_a8w8_blockscale_mul_add_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    c_a_ptr,
    c_b_ptr,
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
    stride_ascale_m,
    stride_ascale_k,
    stride_bscale_k,
    stride_bscale_n,
    stride_cam,
    stride_can,
    stride_cbm,
    stride_cbn,
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
    IS_A_SCALAR: tl.constexpr,
    IS_B_SCALAR: tl.constexpr,
    IS_A_TENSOR: tl.constexpr,
    IS_B_TENSOR: tl.constexpr,
    FUSE_TYPE: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
    waves_per_eu: tl.constexpr,
    matrix_instr_nonkdim: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    """
    Kernel for computing the matmul C = A x B.
    A and B inputs are in the microscale fp4 (mxfp4) format.
    A_scales and B_scales are in e8m0 format.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_ascale_m > 0)
    tl.assume(stride_ascale_k > 0)
    tl.assume(stride_bscale_k > 0)
    tl.assume(stride_bscale_n > 0)
    tl.assume(stride_cam > 0)
    tl.assume(stride_can > 0)
    tl.assume(stride_cbm > 0)
    tl.assume(stride_cbn > 0)

    tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid_unified = tl.program_id(axis=0)
    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    if NUM_KSPLIT == 1:
        # TODO: fix remap
        # remap_xcd(pid, GRID_MN)

        pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(pid_k >= 0)

    if (pid_k * SPLITK_BLOCK_SIZE) < K:

        num_k_iter = tl.cdiv(SPLITK_BLOCK_SIZE, BLOCK_SIZE_K)

        # Create pointers for first block of A and B input matrices
        # The BLOCK sizes are of the elements and in fp4 we pack 2 per uint8 container.
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
        offs_k_scale = (pid_k * SPLITK_BLOCK_SIZE) // GROUP_K
        a_scale_ptrs = (
            a_scale_ptr + offs_am * stride_ascale_m + offs_k_scale * stride_ascale_k
        )
        offs_b_scale_n = offs_bn // GROUP_N
        b_scale_ptrs = (
            b_scale_ptr
            + offs_k_scale * stride_bscale_k
            + offs_b_scale_n * stride_bscale_n
        )
        offs_ks_step = BLOCK_SIZE_K // GROUP_K

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter):
            # Load the next block of A and B, generate a mask by checking the K dimension.
            # If it is out of bounds, set it to 0.
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

            a_scale = tl.load(a_scale_ptrs)
            b_scale = tl.load(b_scale_ptrs)

            # Perform dot operation and apply scale
            accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]

            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

            # k_cur = k * BLOCK_SIZE_K // GROUP_K
            # k_nxt = (k + 1) * BLOCK_SIZE_K // GROUP_K
            # offs_ks = k_nxt - k_cur
            a_scale_ptrs += offs_ks_step * stride_ascale_k
            b_scale_ptrs += offs_ks_step * stride_bscale_k

        # Write back the block of the output matrix C with masks.
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

        if NUM_KSPLIT == 1:
            if IS_A_SCALAR and IS_A_TENSOR:
                c_a = tl.load(c_a_ptr)
            elif IS_A_SCALAR:
                c_a = c_a_ptr
            else:
                c_a = tl.load(
                    c_a_ptr
                    + stride_cam * offs_cm[:, None]
                    + stride_can * offs_cn[None, :],
                    mask=c_mask,
                )
            c_a = c_a.to(tl.float32)

            if IS_B_SCALAR and IS_B_TENSOR:
                c_b = tl.load(c_b_ptr)
            elif IS_B_SCALAR:
                c_b = c_b_ptr
            else:
                c_b = tl.load(
                    c_b_ptr
                    + stride_cbm * offs_cm[:, None]
                    + stride_cbn * offs_cn[None, :],
                    mask=c_mask,
                )
            c_b = c_b.to(tl.float32)

            if FUSE_TYPE == 0:
                accumulator = c_a * accumulator + c_b
            else:
                accumulator = c_b * c_a + accumulator

        c = accumulator.to(c_ptr.type.element_ty)

        c_ptrs = (
            c_ptr
            + stride_cm * offs_cm[:, None]
            + stride_cn * offs_cn[None, :]
            + pid_k * stride_ck
        )
        tl.store(c_ptrs, c, mask=c_mask)


_fused_gemm_a8w8_blockscale_mul_add_reduce_repr = make_kernel_repr(
    "_fused_gemm_a8w8_blockscale_mul_add_reduce_kernel",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "ACTUAL_KSPLIT",
        "MAX_KSPLIT",
    ],
)


@triton.heuristics({})  # dummy heuristics to invoke kernel re-naming
@triton.jit(repr=_fused_gemm_a8w8_blockscale_mul_add_reduce_repr)
def _fused_gemm_a8w8_blockscale_mul_add_reduce_kernel(
    c_in_ptr,
    c_out_ptr,
    c_a_ptr,
    c_b_ptr,
    M,
    N,
    stride_c_in_k,
    stride_c_in_m,
    stride_c_in_n,
    stride_c_out_m,
    stride_c_out_n,
    stride_cam,
    stride_can,
    stride_cbm,
    stride_cbn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    ACTUAL_KSPLIT: tl.constexpr,
    MAX_KSPLIT: tl.constexpr,
    IS_A_SCALAR: tl.constexpr,
    IS_B_SCALAR: tl.constexpr,
    IS_A_TENSOR: tl.constexpr,
    IS_B_TENSOR: tl.constexpr,
    FUSE_TYPE: tl.constexpr,
):

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, MAX_KSPLIT)
    c_in_ptrs = (
        c_in_ptr
        + (offs_k[:, None, None] * stride_c_in_k)
        + (offs_m[None, :, None] * stride_c_in_m)
        + (offs_n[None, None, :] * stride_c_in_n)
    )

    if ACTUAL_KSPLIT == MAX_KSPLIT:
        c = tl.load(c_in_ptrs)
    else:
        c = tl.load(c_in_ptrs, mask=offs_k[:, None, None] < ACTUAL_KSPLIT)
    c = tl.sum(c, axis=0)

    if IS_A_SCALAR and IS_A_TENSOR:
        c_a = tl.load(c_a_ptr)
    elif IS_A_SCALAR:
        c_a = c_a_ptr
    else:
        c_a = tl.load(
            c_a_ptr + stride_cam * offs_m[:, None] + stride_can * offs_n[None, :]
        )
    c_a = c_a.to(tl.float32)

    if IS_B_SCALAR and IS_B_TENSOR:
        c_b = tl.load(c_b_ptr)
    elif IS_B_SCALAR:
        c_b = c_b_ptr
    else:
        c_b = tl.load(
            c_b_ptr + stride_cbm * offs_m[:, None] + stride_cbn * offs_n[None, :]
        )
    c_b = c_b.to(tl.float32)

    if FUSE_TYPE == 0:
        c = c_a * c + c_b
    else:
        c = c_b * c_a + c
    c = c.to(c_out_ptr.type.element_ty)

    c_out_ptrs = (
        c_out_ptr
        + (offs_m[:, None] * stride_c_out_m)
        + (offs_n[None, :] * stride_c_out_n)
    )

    tl.store(c_out_ptrs, c)


# configs/gemm/gfx950-GEMM-A8W8_BLOCKSCALE-N=7168-K=256.json
_CONFIGS = {
    "M_LEQ_8": dict(BLOCK_SIZE_M=4, BLOCK_SIZE_N=32, BLOCK_SIZE_K=128, GROUP_SIZE_M=1, num_warps=1, num_stages=3, waves_per_eu=1, matrix_instr_nonkdim=16, cache_modifier='.cg', NUM_KSPLIT=1),
    "M_LEQ_16": dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=128, GROUP_SIZE_M=4, num_warps=1, num_stages=3, waves_per_eu=6, matrix_instr_nonkdim=16, cache_modifier=None, NUM_KSPLIT=1),
    "M_LEQ_32": dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=128, GROUP_SIZE_M=4, num_warps=4, num_stages=3, waves_per_eu=4, matrix_instr_nonkdim=16, cache_modifier='.cg', NUM_KSPLIT=1),
    "M_LEQ_64": dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=128, GROUP_SIZE_M=8, num_warps=1, num_stages=3, waves_per_eu=2, matrix_instr_nonkdim=16, cache_modifier=None, NUM_KSPLIT=1),
    "M_LEQ_128": dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=128, GROUP_SIZE_M=4, num_warps=8, num_stages=3, waves_per_eu=8, matrix_instr_nonkdim=16, cache_modifier=None, NUM_KSPLIT=1),
    "M_LEQ_256": dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=128, GROUP_SIZE_M=1, num_warps=4, num_stages=3, waves_per_eu=2, matrix_instr_nonkdim=16, cache_modifier='.cg', NUM_KSPLIT=1),
    "M_LEQ_512": dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=128, GROUP_SIZE_M=4, num_warps=1, num_stages=2, waves_per_eu=1, matrix_instr_nonkdim=16, cache_modifier=None, NUM_KSPLIT=1),
    "any": dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128, GROUP_SIZE_M=1, num_warps=8, num_stages=2, waves_per_eu=4, matrix_instr_nonkdim=16, cache_modifier=None, NUM_KSPLIT=1),
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


# fuse_type 1: out = a * b + (x @ w^T), the routed-expert scaling and shared-expert sum of a DeepSeek MoE layer.
_FUSE_TYPE = 1


def _get_config(M, N, K, fuse_type=_FUSE_TYPE):
    # On gfx950 upstream routes fuse_type=1 to the base GEMM-A8W8_BLOCKSCALE config (the dedicated mul_add tile is
    # miscompiled for fuse_type=1 on Triton 3.8); the table above is that family's file for N=7168, K=256.
    return _table_config(M)


def run(*, x, x_scales, w, w_scales, a, b, out) -> None:
    """out = a * b + dequant(x) @ dequant(w)^T: x [M, K], w [N, K] fp8 with block scales; a a scalar, b [M, N]."""
    fuse_type = _FUSE_TYPE
    if isinstance(a, (float, int)):
        IS_A_SCALAR = True
        IS_A_TENSOR = False
    elif isinstance(a, torch.Tensor) and a.is_contiguous():
        IS_A_TENSOR = True
        if a.numel() == 1:
            IS_A_SCALAR = True
        else:
            IS_A_SCALAR = False
    if isinstance(b, (float, int)):
        IS_B_SCALAR = True
        IS_B_TENSOR = False
    elif isinstance(b, torch.Tensor) and b.is_contiguous():
        IS_B_TENSOR = True
        if b.numel() == 1:
            IS_B_SCALAR = True
        else:
            IS_B_SCALAR = False

    M, K = x.shape
    N, K = w.shape

    # Transpose w
    w = w.T
    w_scales = w_scales.T  # (scale_k, scale_n)

    config = _get_config(M, N, K, fuse_type)

    config["SPLITK_BLOCK_SIZE"] = triton.cdiv(
        K, config["NUM_KSPLIT"]
    )  # How big each split_k partition is
    if config["NUM_KSPLIT"] > 1:
        y_pp = torch.empty(
            (config["NUM_KSPLIT"], M, N),
            dtype=torch.float32,
            device=x.device,
        )
    else:
        y_pp = None

    compute_splitk_params(config, K)

    # Scale block sizes
    config["GROUP_K"] = triton.next_power_of_2(
        triton.cdiv(K, w_scales.shape[0])
    )  # scale_block_size_k
    config["GROUP_N"] = triton.next_power_of_2(
        triton.cdiv(N, w_scales.shape[1])
    )  # scale_block_size_n

    assert (
        config["GROUP_K"] == config["BLOCK_SIZE_K"]
    ), "GROUP_K must equal BLOCK_SIZE_K"

    grid = lambda META: (  # noqa: E731
        (
            META["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"])
        ),
    )
    _fused_gemm_a8w8_blockscale_mul_add_kernel[grid](
        x,
        w,
        out if config["NUM_KSPLIT"] == 1 else y_pp,
        x_scales,
        w_scales,
        a,
        b,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        0 if config["NUM_KSPLIT"] == 1 else y_pp.stride(0),
        out.stride(0) if config["NUM_KSPLIT"] == 1 else y_pp.stride(1),
        out.stride(1) if config["NUM_KSPLIT"] == 1 else y_pp.stride(2),
        x_scales.stride(0),
        x_scales.stride(1),
        w_scales.stride(0),
        w_scales.stride(1),
        0 if IS_A_SCALAR else a.stride(0),
        0 if IS_A_SCALAR else a.stride(1),
        0 if IS_B_SCALAR else b.stride(0),
        0 if IS_B_SCALAR else b.stride(1),
        IS_A_SCALAR=IS_A_SCALAR,
        IS_B_SCALAR=IS_B_SCALAR,
        IS_A_TENSOR=IS_A_TENSOR,
        IS_B_TENSOR=IS_B_TENSOR,
        FUSE_TYPE=fuse_type,
        **config,
    )

    if config["NUM_KSPLIT"] > 1:
        REDUCE_BLOCK_SIZE_M = 32
        REDUCE_BLOCK_SIZE_N = 32
        ACTUAL_KSPLIT = triton.cdiv(K, config["SPLITK_BLOCK_SIZE"])

        grid_reduce = (
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
        )
        _fused_gemm_a8w8_blockscale_mul_add_reduce_kernel[grid_reduce](
            y_pp,
            out,
            a,
            b,
            M,
            N,
            y_pp.stride(0),
            y_pp.stride(1),
            y_pp.stride(2),
            out.stride(0),
            out.stride(1),
            0 if IS_A_SCALAR else a.stride(0),
            0 if IS_A_SCALAR else a.stride(1),
            0 if IS_B_SCALAR else b.stride(0),
            0 if IS_B_SCALAR else b.stride(1),
            REDUCE_BLOCK_SIZE_M,
            REDUCE_BLOCK_SIZE_N,
            ACTUAL_KSPLIT,
            triton.next_power_of_2(config["NUM_KSPLIT"]),
            IS_A_SCALAR=IS_A_SCALAR,
            IS_B_SCALAR=IS_B_SCALAR,
            IS_A_TENSOR=IS_A_TENSOR,
            IS_B_TENSOR=IS_B_TENSOR,
            FUSE_TYPE=fuse_type,
        )
