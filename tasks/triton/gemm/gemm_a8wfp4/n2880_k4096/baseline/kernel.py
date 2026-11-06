# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""a8wfp4 GEMM from ROCm/aiter: fp8 (e4m3) activations with a per-token scale times MXFP4 weights (Triton).

Provenance: aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8wfp4.py (device kernel, `_get_splitk`),
aiter/ops/triton/gemm/basic/gemm_a8wfp4.py (launcher), aiter/ops/triton/_triton_kernels/common/splitk_reduce.py
(`_gemm_splitk_reduce_kernel`), `remap_xcd`, `pid_grid` and `make_kernel_repr` from
aiter/ops/triton/utils/_triton/, and the generic gfx950 table configs/gemm/gfx950-GEMM-A8WFP4.json (AITER has
no file specialized to this N and K), inlined with the get_gemm_config lookup order. Split-K (NUM_KSPLIT 4 for
M <= 128 in the table, adjusted by `_get_splitk`) writes float32 partial sums that the reduce kernel adds.
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


_gemm_a8wfp4_repr = make_kernel_repr(
    "_gemm_a8wfp4_kernel",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "NUM_KSPLIT",
        "SPLITK_BLOCK_SIZE",
        "EVEN_K",
        "GRID_MN",
        "RAW_MASKED_LOADS",
        "cache_modifier",
    ],
)


@triton.heuristics(
    {
        "EVEN_K": lambda args: (args["K"] % (args["BLOCK_SIZE_K"] // 2) == 0)
        and (args["SPLITK_BLOCK_SIZE"] % args["BLOCK_SIZE_K"] == 0)
        and (args["K"] % (args["SPLITK_BLOCK_SIZE"] // 2) == 0),
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@triton.jit(repr=_gemm_a8wfp4_repr)
def _gemm_a8wfp4_kernel(
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
    GRID_MN: tl.constexpr,
    RAW_MASKED_LOADS: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    """
    Kernel for computing the matmul C = A x B.
    A is in fp8 e4m3 format.
    B is in the microscale fp4 (mxfp4) format.
    A_scales and B_scales are in e8m0 format.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
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
    # We assume 32 elements along K share the same scale.
    SCALE_GROUP_SIZE: tl.constexpr = 32

    if (pid_k * SPLITK_BLOCK_SIZE) < K:
        num_k_iter = tl.cdiv(SPLITK_BLOCK_SIZE, BLOCK_SIZE_K)

        # Set up base A offsets
        offs_am_raw = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_am = offs_am_raw % M

        # Load A scales once (they're per-row)
        a_scale_ptrs = a_scales_ptr + offs_am * stride_asm
        if RAW_MASKED_LOADS:
            a_scale_mask = offs_am < M
            a_scales = tl.load(a_scale_ptrs, mask=a_scale_mask)
        else:
            a_scales = tl.load(a_scale_ptrs)
        a_ones_scale = tl.full(
            (BLOCK_SIZE_M, BLOCK_SIZE_K // SCALE_GROUP_SIZE), 127, dtype=tl.uint8
        )  # 1.0 in e8m0

        # Set up base B offsets
        offs_bn_raw = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_bn = offs_bn_raw % N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(num_k_iter):
            # Load A inside the loop with correct K offset
            offs_ak = tl.arange(0, BLOCK_SIZE_K) + k * BLOCK_SIZE_K  # Add k offset
            offs_ak_split = pid_k * SPLITK_BLOCK_SIZE + offs_ak
            a_ptrs = a_ptr + (
                offs_am[:, None] * stride_am + offs_ak_split[None, :] * stride_ak
            )
            if RAW_MASKED_LOADS:
                a_mask = (offs_am_raw[:, None] < M) & (offs_ak_split[None, :] < K)
                a = tl.load(a_ptrs, mask=a_mask)
            else:
                a = tl.load(a_ptrs)

            # B loading stays mostly the same, but fix the offsets
            offs_bk = tl.arange(0, BLOCK_SIZE_K // 2) + k * (
                BLOCK_SIZE_K // 2
            )  # Add k offset
            offs_bk_split = pid_k * (SPLITK_BLOCK_SIZE // 2) + offs_bk
            b_ptrs = b_ptr + (
                offs_bk_split[:, None] * stride_bk + offs_bn[None, :] * stride_bn
            )
            offs_ks = (
                (pid_k * (SPLITK_BLOCK_SIZE // SCALE_GROUP_SIZE))
                + k * (BLOCK_SIZE_K // SCALE_GROUP_SIZE)
                + tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_SIZE)
            )
            b_scale_ptrs = (
                b_scales_ptr
                + offs_bn[:, None] * stride_bsn
                + offs_ks[None, :] * stride_bsk
            )
            if RAW_MASKED_LOADS:
                b_k_mask = offs_bk_split[:, None] < (K // 2)
                b_n_mask = offs_bn_raw[None, :] < N
                b_mask = b_k_mask & b_n_mask
                if EVEN_K:
                    b = tl.load(b_ptrs, mask=b_mask, cache_modifier=cache_modifier)
                else:
                    b = tl.load(b_ptrs, mask=b_mask, other=0)
                bs_k_mask = offs_ks[None, :] < (K // SCALE_GROUP_SIZE)
                bs_n_scale_mask = offs_bn_raw[:, None] < N
                bs_mask = bs_k_mask & bs_n_scale_mask
                b_scales = tl.load(b_scale_ptrs, mask=bs_mask, other=0)
            else:
                if EVEN_K:
                    b = tl.load(b_ptrs, cache_modifier=cache_modifier)
                else:
                    b_mask = offs_bk[:, None] < K - k * (BLOCK_SIZE_K // 2)
                    b = tl.load(b_ptrs, mask=b_mask, other=0)
                b_scales = tl.load(b_scale_ptrs)
            accumulator = tl.dot_scaled(
                a, a_ones_scale, "e4m3", b, b_scales, "e2m1", acc=accumulator
            )

        # Scale by a_scales at the end
        c = (accumulator * a_scales[:, None]).to(c_ptr.type.element_ty)

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


# configs/gemm/gfx950-GEMM-A8WFP4.json (generic table; no N=2880, K=4096 file exists).
_CONFIGS = {
    "M_LEQ_8": dict(BLOCK_SIZE_M=8, BLOCK_SIZE_N=64, BLOCK_SIZE_K=256, GROUP_SIZE_M=1, num_warps=4, num_stages=3, waves_per_eu=2, matrix_instr_nonkdim=16, cache_modifier='.cg', NUM_KSPLIT=4),
    "M_LEQ_16": dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=256, GROUP_SIZE_M=1, num_warps=4, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=16, cache_modifier='.cg', NUM_KSPLIT=4),
    "M_LEQ_32": dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=64, BLOCK_SIZE_K=256, GROUP_SIZE_M=1, num_warps=8, num_stages=3, waves_per_eu=1, matrix_instr_nonkdim=16, cache_modifier='.cg', NUM_KSPLIT=4),
    "M_LEQ_64": dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=256, GROUP_SIZE_M=1, num_warps=8, num_stages=2, waves_per_eu=1, matrix_instr_nonkdim=16, cache_modifier='.cg', NUM_KSPLIT=4),
    "M_LEQ_128": dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=256, GROUP_SIZE_M=1, num_warps=8, num_stages=3, waves_per_eu=2, matrix_instr_nonkdim=16, cache_modifier='.cg', NUM_KSPLIT=4),
    "M_LEQ_256": dict(BLOCK_SIZE_M=256, BLOCK_SIZE_N=32, BLOCK_SIZE_K=128, GROUP_SIZE_M=8, num_warps=4, num_stages=3, waves_per_eu=1, matrix_instr_nonkdim=16, cache_modifier='.cg', NUM_KSPLIT=1),
    "M_LEQ_512": dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128, GROUP_SIZE_M=4, num_warps=8, num_stages=3, waves_per_eu=4, matrix_instr_nonkdim=16, cache_modifier=None, NUM_KSPLIT=1),
    "any": dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=256, BLOCK_SIZE_K=128, GROUP_SIZE_M=16, num_warps=4, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=16, cache_modifier=None, NUM_KSPLIT=1),
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


def _get_splitk(K: int, BLOCK_SIZE_K: int, NUM_KSPLIT: int):
    # heuristics for make "EVEN_K == True" as much as possible
    NUM_KSPLIT_STEP = 4
    BLOCK_SIZE_K_STEP = 4
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

    return SPLITK_BLOCK_SIZE, BLOCK_SIZE_K, NUM_KSPLIT


def _get_config(M, N, K):
    config = _table_config(M)
    if M <= 128:
        SPLITK_BLOCK_SIZE, BLOCK_SIZE_K, NUM_KSPLIT = _get_splitk(K, config["BLOCK_SIZE_K"], config["NUM_KSPLIT"])
        config["SPLITK_BLOCK_SIZE"] = SPLITK_BLOCK_SIZE
        config["BLOCK_SIZE_K"] = BLOCK_SIZE_K
        config["NUM_KSPLIT"] = NUM_KSPLIT
    else:
        config["SPLITK_BLOCK_SIZE"] = 2 * K
    return config


def run(*, a, a_scales, w, w_scales, out) -> None:
    """out = (a * a_scales) @ dequant(w)^T: a [M, K] e4m3, a_scales [M, 1] f32, w [N, K // 2] e2m1, w_scales [N, K // 32]."""
    M, K = a.shape
    N, K_packed = w.shape
    w = w.T  # the kernel expects (K // 2, N)

    config = _get_config(M, N, K)

    y_pp = None
    if M <= 128:
        # Disable split-K when M < BLOCK_SIZE_M (upstream workaround for incorrect partial sums on gfx950).
        if M < config["BLOCK_SIZE_M"]:
            config["NUM_KSPLIT"] = 1
            config["SPLITK_BLOCK_SIZE"] = 2 * K
        if config["NUM_KSPLIT"] > 1:
            y_pp = torch.empty((config["NUM_KSPLIT"], M, N), dtype=torch.float32, device=out.device)
    else:
        assert config.get("NUM_KSPLIT", 1) == 1, "split-K is not supported for M > 128"
        config["SPLITK_BLOCK_SIZE"] = 2 * K

    grid = lambda META: (  # noqa: E731
        config["NUM_KSPLIT"] * triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    y_final = out if config["NUM_KSPLIT"] == 1 else y_pp
    stride_ck, stride_cm, stride_cn = (
        (0, out.stride(0), out.stride(1)) if config["NUM_KSPLIT"] == 1 else y_pp.stride()
    )
    _gemm_a8wfp4_kernel[grid](
        a,
        w,
        y_final,
        a_scales,
        w_scales,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        w.stride(0),
        w.stride(1),
        stride_ck,
        stride_cm,
        stride_cn,
        a_scales.stride(0),
        a_scales.stride(1),
        w_scales.stride(0),
        w_scales.stride(1),
        RAW_MASKED_LOADS=True,
        **config,
    )

    if config["NUM_KSPLIT"] > 1:
        REDUCE_BLOCK_SIZE_M = 16
        REDUCE_BLOCK_SIZE_N = 64  # float32 partial sums
        ACTUAL_KSPLIT = triton.cdiv(K, config["SPLITK_BLOCK_SIZE"])
        grid_reduce = (
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
        )
        _gemm_splitk_reduce_kernel[grid_reduce](
            y_pp,
            out,
            None,
            M,
            N,
            y_pp.stride(0),
            y_pp.stride(1),
            y_pp.stride(2),
            out.stride(0),
            out.stride(1),
            REDUCE_BLOCK_SIZE_M,
            REDUCE_BLOCK_SIZE_N,
            ACTUAL_KSPLIT,
            config["NUM_KSPLIT"],
            ADD_BIAS=False,
            activation="",
            use_activation=False,
            KERNEL_NAME="_gemm_afp4_wfp4_reduce_kernel",
        )
