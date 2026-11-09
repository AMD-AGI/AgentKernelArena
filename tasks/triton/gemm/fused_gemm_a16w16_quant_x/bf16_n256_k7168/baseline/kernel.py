# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


"""bf16 GEMM fused with MXFP8 activation quantization from ROCm/aiter (Triton): one launch computes y = x @ w^T and
the per-1x32 MXFP8 quantization of x (float8_e4m3fn codes and e8m0 scale bytes).

Provenance: the device kernel of aiter/ops/triton/_triton_kernels/gemm/fused/fused_gemm_a16w16_quant_x.py
(`_fused_gemm_a16w16_quant_x_kernel`), the launcher `fused_gemm_a16w16_quant_x` of
aiter/ops/triton/gemm/fused/fused_gemm_a16w16_quant_x.py (bias and activation removed), `_mxfp8_quant_op` of
aiter/ops/triton/_triton_kernels/quant/quant.py, `_gemm_splitk_reduce_kernel` of
aiter/ops/triton/_triton_kernels/common/splitk_reduce.py, `compute_splitk_params` of aiter/ops/triton/utils/gemm_config_utils.py,
and `remap_xcd`, `pid_grid` and `make_kernel_repr` of aiter/ops/triton/utils/_triton/. The tuned table is the
shape-specialized gfx950 file configs/gemm/gfx950-GEMM-A16W16-N=256-K=7168.json, inlined with the get_gemm_config lookup
order. Split-K (NUM_KSPLIT 7 to 14 for M <= 512 in the table) writes float32 partial sums that the reduce kernel adds.
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


_gemm_splitk_reduce_repr = make_kernel_repr(
    "_gemm_splitk_reduce_kernel",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "ACTUAL_KSPLIT",
        "MAX_KSPLIT",
        "ADD_BIAS",
        "activation",
        "use_activation",
    ],
    name_key="KERNEL_NAME",
)


@triton.jit(repr=_gemm_splitk_reduce_repr)
def _gemm_splitk_reduce_kernel(
    c_in_ptr,
    c_out_ptr,
    bias_ptr,
    M,
    N,
    stride_c_in_k,
    stride_c_in_m,
    stride_c_in_n,
    stride_c_out_m,
    stride_c_out_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    ACTUAL_KSPLIT: tl.constexpr,
    MAX_KSPLIT: tl.constexpr,
    ADD_BIAS: tl.constexpr,
    activation: tl.constexpr,
    use_activation: tl.constexpr,
    KERNEL_NAME: tl.constexpr = "_gemm_splitk_reduce_kernel",
):
    tl.assume(stride_c_in_k > 0)
    tl.assume(stride_c_in_m > 0)
    tl.assume(stride_c_in_n > 0)
    tl.assume(stride_c_out_m > 0)
    tl.assume(stride_c_out_n > 0)

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Tell the AMD backend pid * stride stays non-negative so it can lower
    # the loads/stores to buffer ops instead of generic global ops.
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
        c = tl.load(c_in_ptrs, mask=offs_k[:, None, None] < ACTUAL_KSPLIT, other=0.0)
    c = tl.sum(c, axis=0)

    if ADD_BIAS:
        acc_dtype = tl.float32 if c_in_ptr.type.element_ty != tl.int8 else tl.int32
        bias = tl.load(bias_ptr + offs_n).to(dtype=acc_dtype)
        bias = tl.broadcast_to(bias[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N))
        c += bias

    if use_activation:
        c = activation(c)

    c = c.to(c_out_ptr.type.element_ty)

    c_out_ptrs = (
        c_out_ptr
        + (offs_m[:, None] * stride_c_out_m)
        + (offs_n[None, :] * stride_c_out_n)
    )

    tl.store(c_out_ptrs, c)


@triton.jit
def _mxfp8_quant_op(x_grouped, QUANT_AXIS: tl.constexpr):
    """Shared MXFP8 (1x32 e8m0) scale derivation.

    Given a fp32 tile where the QUANT_AXIS dim is sized QUANT_BLOCK_SIZE (=32),
    returns (scale_e8m0, quant_scale): the per-group uint8 e8m0 scale and the
    matching fp32 multiplicative scale. Both outputs keep QUANT_AXIS with size 1
    so they broadcast against the input for in-place quantization.
    """
    amax = tl.max(tl.abs(x_grouped), axis=QUANT_AXIS, keep_dims=True)
    amax_i32 = amax.to(tl.int32, bitcast=True)
    amax_i32 = (amax_i32 + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    amax_p2 = amax_i32.to(tl.float32, bitcast=True)
    scale_unbiased = tl.log2(amax_p2).floor() - 8
    scale_unbiased = tl.clamp(scale_unbiased, min=-127, max=127)
    scale_e8m0 = (scale_unbiased.to(tl.int32) + 127).to(tl.uint8)
    quant_scale = tl.exp2(-scale_unbiased)
    return scale_e8m0, quant_scale


_QUANT_BLOCK_SIZE = 32


_fused_gemm_a16w16_quant_x_repr = make_kernel_repr(
    "_fused_gemm_a16w16_quant_x_kernel",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "NUM_KSPLIT",
        "SPLITK_BLOCK_SIZE",
        "QUANT_BLOCK_SIZE",
        "EVEN_K",
        "EVEN_MN",
        "cache_modifier",
        "activation",
        "use_activation",
        "ADD_BIAS",
        "SKIP_REDUCE",
    ],
)


@triton.heuristics(
    {
        "EVEN_K": lambda args: (args["K"] % (args["SPLITK_BLOCK_SIZE"]) == 0)
        and (args["SPLITK_BLOCK_SIZE"] % args["BLOCK_SIZE_K"] == 0),
        "EVEN_MN": lambda args: (args["M"] % args["BLOCK_SIZE_M"] == 0)
        and (args["N"] % args["BLOCK_SIZE_N"] == 0),
    }
)
@triton.jit(
    repr=_fused_gemm_a16w16_quant_x_repr,
    do_not_specialize=["M", "N"],
)
def _fused_gemm_a16w16_quant_x_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    a_quant_ptr,
    a_scale_ptr,
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
    stride_a_quant_m,
    stride_a_quant_k,
    stride_a_scale_m,
    stride_a_scale_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_MN: tl.constexpr,
    cache_modifier: tl.constexpr,
    activation: tl.constexpr,
    use_activation: tl.constexpr,
    ADD_BIAS: tl.constexpr,
    SKIP_REDUCE: tl.constexpr,
):
    """Kernel that computes C = A x B and also emits an MXFP8-quantized A.

    The grid is laid out as a single 1D program-id space split into two
    contiguous regions:

      * `[0, NUM_KSPLIT * num_pid_m * num_pid_n)` runs the GEMM (identical to
        the unfused a16w16 kernel).
      * `[NUM_KSPLIT * num_pid_m * num_pid_n, GEMM_GRID + num_pid_m * num_pid_k_copy)`
        runs the per-1x32 MXFP8 quantization of A: each program handles one
        `BLOCK_SIZE_M x BLOCK_SIZE_K` tile of A, derives per-1x32 e8m0 scales,
        and writes both the FP8 values and the uint8 scales.

    BLOCK_SIZE_K must be a multiple of QUANT_BLOCK_SIZE (=32) so that each tile
    contains a whole number of MXFP8 groups per row.
    """

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_ck > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_a_quant_m > 0)
    tl.assume(stride_a_quant_k > 0)
    tl.assume(stride_a_scale_m > 0)
    tl.assume(stride_a_scale_n > 0)

    pid_unified = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k_copy = tl.cdiv(K, BLOCK_SIZE_K)
    GEMM_GRID = num_pid_m * num_pid_n * NUM_KSPLIT

    if pid_unified < GEMM_GRID:
        # ---- GEMM branch ----------------------------------------------------
        pid_unified = remap_xcd(pid_unified, GEMM_GRID, NUM_XCDS=8)
        pid_k = pid_unified % NUM_KSPLIT
        pid = pid_unified // NUM_KSPLIT

        if NUM_KSPLIT == 1:
            pid_m, pid_n = pid_grid(
                pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M
            )
        else:
            pid_m = pid // num_pid_n
            pid_n = pid % num_pid_n

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)
        tl.assume(pid_k >= 0)

        split_k_start = pid_k * SPLITK_BLOCK_SIZE
        if split_k_start < K:
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            offs_k_split = split_k_start + offs_k
            if EVEN_MN:
                offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            else:
                offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
                offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

            a_ptrs = a_ptr + (
                offs_am[:, None] * stride_am + offs_k_split[None, :] * stride_ak
            )
            b_ptrs = b_ptr + (
                offs_k_split[:, None] * stride_bk + offs_bn[None, :] * stride_bn
            )

            acc_dtype = tl.float32 if c_ptr.type.element_ty != tl.int8 else tl.int32
            if ADD_BIAS:
                if NUM_KSPLIT == 1 or (SKIP_REDUCE and pid_k == 0):
                    accumulator = tl.load(bias_ptr + offs_bn).to(dtype=acc_dtype)
                    accumulator = tl.broadcast_to(
                        accumulator[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N)
                    )
                else:
                    accumulator = tl.zeros(
                        (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype
                    )
            else:
                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

            split_k_end = tl.minimum(split_k_start + SPLITK_BLOCK_SIZE, K)
            k_span = split_k_end - split_k_start
            num_k_iter = tl.cdiv(k_span, BLOCK_SIZE_K)

            for k in range(num_k_iter):
                if EVEN_K:
                    a = tl.load(a_ptrs)
                    b = tl.load(b_ptrs, cache_modifier=cache_modifier)
                else:
                    a = tl.load(
                        a_ptrs,
                        mask=offs_k[None, :] < k_span - k * BLOCK_SIZE_K,
                        other=0.0,
                    )
                    b = tl.load(
                        b_ptrs,
                        mask=offs_k[:, None] < k_span - k * BLOCK_SIZE_K,
                        other=0.0,
                        cache_modifier=cache_modifier,
                    )
                accumulator += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K * stride_ak
                b_ptrs += BLOCK_SIZE_K * stride_bk

            if use_activation and NUM_KSPLIT == 1:
                accumulator = activation(accumulator)

            c = accumulator.to(c_ptr.type.element_ty)
            offs_cm = pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = pid_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = (
                c_ptr
                + stride_cm * offs_cm[:, None]
                + stride_cn * offs_cn[None, :]
                + pid_k * stride_ck
            )
            if EVEN_MN:
                tl.store(c_ptrs, c)
            else:
                c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
                tl.store(c_ptrs, c, mask=c_mask)
    else:
        # ---- MXFP8 quant branch --------------------------------------------
        pid_copy = pid_unified - GEMM_GRID
        pid_m = pid_copy // num_pid_k_copy
        pid_k = pid_copy % num_pid_k_copy

        tl.assume(pid_m >= 0)
        tl.assume(pid_k >= 0)

        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=mask, other=0.0).to(tl.float32)

        # Per-1x32 MXFP8 quant. Group along K within this tile.
        n_groups: tl.constexpr = BLOCK_SIZE_K // QUANT_BLOCK_SIZE
        a_2d = tl.reshape(a, (BLOCK_SIZE_M, n_groups, QUANT_BLOCK_SIZE))
        scale_e8m0, quant_scale = _mxfp8_quant_op(a_2d, QUANT_AXIS=2)

        qa_2d = a_2d * quant_scale
        qa = tl.reshape(qa_2d, (BLOCK_SIZE_M, BLOCK_SIZE_K))
        a_quant_ptrs = a_quant_ptr + (
            offs_m[:, None] * stride_a_quant_m + offs_k[None, :] * stride_a_quant_k
        )
        tl.store(a_quant_ptrs, qa.to(a_quant_ptr.type.element_ty), mask=mask)

        # Store scales: shape (M, K // QUANT_BLOCK_SIZE).
        offs_s_n = pid_k * n_groups + tl.arange(0, n_groups)
        scale_2d = tl.reshape(scale_e8m0, (BLOCK_SIZE_M, n_groups))
        a_scale_ptrs = a_scale_ptr + (
            offs_m[:, None] * stride_a_scale_m + offs_s_n[None, :] * stride_a_scale_n
        )
        scale_mask = (offs_m[:, None] < M) & (
            offs_s_n[None, :] < (K // QUANT_BLOCK_SIZE)
        )
        tl.store(a_scale_ptrs, scale_2d, mask=scale_mask)


# configs/gemm/gfx950-GEMM-A16W16-N=256-K=7168.json
_CONFIGS = {
    "M_LEQ_8": dict(BLOCK_SIZE_M=8, BLOCK_SIZE_N=16, BLOCK_SIZE_K=256, GROUP_SIZE_M=1, num_warps=4, num_stages=3, waves_per_eu=1, matrix_instr_nonkdim=16, cache_modifier='.cg', NUM_KSPLIT=14),
    "M_LEQ_16": dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=16, BLOCK_SIZE_K=256, GROUP_SIZE_M=1, num_warps=4, num_stages=3, waves_per_eu=2, matrix_instr_nonkdim=16, cache_modifier='.cg', NUM_KSPLIT=14),
    "M_LEQ_32": dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=256, GROUP_SIZE_M=1, num_warps=8, num_stages=2, waves_per_eu=8, matrix_instr_nonkdim=16, cache_modifier=None, NUM_KSPLIT=16),
    "M_LEQ_64": dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=128, GROUP_SIZE_M=1, num_warps=4, num_stages=3, waves_per_eu=4, matrix_instr_nonkdim=16, cache_modifier=None, NUM_KSPLIT=7),
    "M_LEQ_128": dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=256, GROUP_SIZE_M=1, num_warps=4, num_stages=3, waves_per_eu=1, matrix_instr_nonkdim=16, cache_modifier=None, NUM_KSPLIT=8),
    "M_LEQ_256": dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128, GROUP_SIZE_M=1, num_warps=8, num_stages=3, waves_per_eu=2, matrix_instr_nonkdim=16, cache_modifier=None, NUM_KSPLIT=8),
    "M_LEQ_512": dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=128, GROUP_SIZE_M=1, num_warps=8, num_stages=3, waves_per_eu=6, matrix_instr_nonkdim=16, cache_modifier=None, NUM_KSPLIT=8),
    "any": dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128, GROUP_SIZE_M=1, num_warps=8, num_stages=3, waves_per_eu=4, matrix_instr_nonkdim=16, cache_modifier=None, NUM_KSPLIT=1),
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
    # Upstream: get_gemm_config("GEMM-A16W16", M, N, K) followed by compute_splitk_params; the table above is the
    # N=256, K=7168 file.
    return compute_splitk_params(_table_config(M), K)


def run(*, x, w, y, x_quant, x_scale) -> None:
    """y = x @ w^T and the per-1x32 MXFP8 quantization of x (e4m3fn codes x_quant, e8m0 bytes x_scale)."""
    M, K = x.shape
    N, K = w.shape
    w = w.T

    config = _get_config(M, N, K)

    assert (
        config["BLOCK_SIZE_K"] % _QUANT_BLOCK_SIZE == 0
    ), f"BLOCK_SIZE_K={config['BLOCK_SIZE_K']} must be a multiple of {_QUANT_BLOCK_SIZE}"

    if config["NUM_KSPLIT"] > 1:
        y_pp = torch.empty(
            (config["NUM_KSPLIT"], M, N),
            dtype=torch.float32,
            device=x.device,
        )
    else:
        y_pp = None

    grid = lambda META: (  # noqa: E731
        META["NUM_KSPLIT"]
        * triton.cdiv(M, META["BLOCK_SIZE_M"])
        * triton.cdiv(N, META["BLOCK_SIZE_N"])
        + triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(K, META["BLOCK_SIZE_K"]),
    )
    _fused_gemm_a16w16_quant_x_kernel[grid](
        x,
        w,
        None,
        y if config["NUM_KSPLIT"] == 1 else y_pp,
        x_quant,
        x_scale,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        0 if config["NUM_KSPLIT"] == 1 else y_pp.stride(0),
        y.stride(0) if config["NUM_KSPLIT"] == 1 else y_pp.stride(1),
        y.stride(1) if config["NUM_KSPLIT"] == 1 else y_pp.stride(2),
        x_quant.stride(0),
        x_quant.stride(1),
        x_scale.stride(0),
        x_scale.stride(1),
        activation="",
        use_activation=False,
        ADD_BIAS=False,
        SKIP_REDUCE=False,
        QUANT_BLOCK_SIZE=_QUANT_BLOCK_SIZE,
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
        _gemm_splitk_reduce_kernel[grid_reduce](
            y_pp,
            y,
            None,
            M,
            N,
            y_pp.stride(0),
            y_pp.stride(1),
            y_pp.stride(2),
            y.stride(0),
            y.stride(1),
            REDUCE_BLOCK_SIZE_M,
            REDUCE_BLOCK_SIZE_N,
            ACTUAL_KSPLIT,
            triton.next_power_of_2(config["NUM_KSPLIT"]),
            ADD_BIAS=False,
            activation="",
            use_activation=False,
            KERNEL_NAME="_fused_gemm_a16w16_quant_x_reduce_kernel",
        )
