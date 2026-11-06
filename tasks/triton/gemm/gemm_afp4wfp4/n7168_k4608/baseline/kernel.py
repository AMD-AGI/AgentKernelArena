# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""afp4wfp4 GEMM from ROCm/aiter: MXFP4 activations times MXFP4 weights, both with e8m0 1x32 scales (Triton).

Provenance: aiter/ops/triton/_triton_kernels/gemm/basic/gemm_afp4wfp4.py (device kernel), the launcher and
`get_splitk` in aiter/ops/triton/gemm/basic/gemm_afp4wfp4.py (`gemm_afp4wfp4_`),
aiter/ops/triton/_triton_kernels/common/splitk_reduce.py (`_gemm_splitk_reduce_kernel`), `remap_xcd`, `pid_grid`
and `make_kernel_repr` from aiter/ops/triton/utils/_triton/, and the generic gfx950 table
configs/gfx950/triton/gemm/GEMM-AFP4WFP4.json (AITER has no file specialized to this N and K), inlined with the
get_gemm_config lookup order. The preshuffle kernels are dropped. Split-K (NUM_KSPLIT 4 for M <= 8 in the table,
adjusted by `get_splitk`) writes float32 partial sums that the reduce kernel adds.
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


_gemm_afp4wfp4_repr = make_kernel_repr(
    "_gemm_afp4wfp4_kernel",
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
    }
)
@triton.jit(repr=_gemm_afp4wfp4_repr)
def _gemm_afp4wfp4_kernel(
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
    tl.assume(stride_asm > 0)
    tl.assume(stride_ask > 0)
    tl.assume(stride_bsk > 0)
    tl.assume(stride_bsn > 0)

    GRID_MN = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)

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
    # We assume 32 elements along K share the same scale.
    SCALE_GROUP_SIZE: tl.constexpr = 32

    if (pid_k * SPLITK_BLOCK_SIZE // 2) < K:

        num_k_iter = tl.cdiv(SPLITK_BLOCK_SIZE // 2, BLOCK_SIZE_K // 2)

        # Create pointers for first block of A and B input matrices
        # The BLOCK sizes are of the elements and in fp4 we pack 2 per uint8 container.
        offs_k = tl.arange(0, BLOCK_SIZE_K // 2)
        offs_k_split = pid_k * (SPLITK_BLOCK_SIZE // 2) + offs_k
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        a_ptrs = a_ptr + (
            offs_am[:, None] * stride_am + offs_k_split[None, :] * stride_ak
        )
        b_ptrs = b_ptr + (
            offs_k_split[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        )
        # Create pointers for the first block of A and B scales
        offs_ks = (pid_k * (SPLITK_BLOCK_SIZE // SCALE_GROUP_SIZE)) + tl.arange(
            0, BLOCK_SIZE_K // SCALE_GROUP_SIZE
        )
        a_scale_ptrs = (
            a_scales_ptr + offs_am[:, None] * stride_asm + offs_ks[None, :] * stride_ask
        )
        # B scales are N x K even though B operand is K x N.
        b_scale_ptrs = (
            b_scales_ptr + offs_bn[:, None] * stride_bsn + offs_ks[None, :] * stride_bsk
        )

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        offs_scale_k = tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_SIZE)

        for k in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter):
            # Load scales, masking OOB when K is not aligned to BLOCK_SIZE_K.
            # OOB scales could contain 0xFF (NaN in e8m0) which propagates
            # through dot_scaled even when the corresponding data is zero.
            if EVEN_K:
                a_scales = tl.load(a_scale_ptrs)
                b_scales = tl.load(b_scale_ptrs, cache_modifier=cache_modifier)
            else:
                scale_mask = offs_scale_k[None, :] < (
                    2 * K // SCALE_GROUP_SIZE - k * (BLOCK_SIZE_K // SCALE_GROUP_SIZE)
                )
                a_scales = tl.load(a_scale_ptrs, mask=scale_mask, other=127)
                b_scales = tl.load(
                    b_scale_ptrs,
                    mask=scale_mask,
                    other=127,
                    cache_modifier=cache_modifier,
                )

            # Load the next block of A and B, generate a mask by checking the K dimension.
            # If it is out of bounds, set it to 0.
            if EVEN_K:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs, cache_modifier=cache_modifier)
            else:
                a = tl.load(
                    a_ptrs, mask=offs_k[None, :] < K - k * (BLOCK_SIZE_K // 2), other=0
                )
                b = tl.load(
                    b_ptrs,
                    mask=offs_k[:, None] < K - k * (BLOCK_SIZE_K // 2),
                    other=0,
                    cache_modifier=cache_modifier,
                )

            accumulator = tl.dot_scaled(
                a, a_scales, "e2m1", b, b_scales, "e2m1", accumulator
            )

            # Advance the ptrs to the next K block.
            a_ptrs += (BLOCK_SIZE_K // 2) * stride_ak
            b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
            a_scale_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_SIZE) * stride_ask
            b_scale_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_SIZE) * stride_bsk

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


# configs/gfx950/triton/gemm/GEMM-AFP4WFP4.json (generic table; no N=7168, K=4608 file exists).
_CONFIGS = {
    "M_LEQ_8": dict(BLOCK_SIZE_M=8, BLOCK_SIZE_N=64, BLOCK_SIZE_K=512, GROUP_SIZE_M=1, num_warps=4, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=32, cache_modifier='.cg', NUM_KSPLIT=4),
    "M_LEQ_16": dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=1024, GROUP_SIZE_M=4, num_warps=4, num_stages=3, waves_per_eu=1, matrix_instr_nonkdim=16, cache_modifier='.cg', NUM_KSPLIT=1),
    "M_LEQ_32": dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=1024, GROUP_SIZE_M=4, num_warps=4, num_stages=3, waves_per_eu=1, matrix_instr_nonkdim=16, cache_modifier='.cg', NUM_KSPLIT=1),
    "M_LEQ_64": dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=512, GROUP_SIZE_M=1, num_warps=4, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=16, cache_modifier=None, NUM_KSPLIT=1),
    "M_LEQ_128": dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=512, GROUP_SIZE_M=1, num_warps=4, num_stages=3, waves_per_eu=1, matrix_instr_nonkdim=16, cache_modifier=None, NUM_KSPLIT=1),
    "M_LEQ_256": dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=64, BLOCK_SIZE_K=512, GROUP_SIZE_M=1, num_warps=4, num_stages=3, waves_per_eu=1, matrix_instr_nonkdim=16, cache_modifier=None, NUM_KSPLIT=1),
    "M_LEQ_512": dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=256, GROUP_SIZE_M=8, num_warps=4, num_stages=3, waves_per_eu=2, matrix_instr_nonkdim=16, cache_modifier='.cg', NUM_KSPLIT=1),
    "any": dict(BLOCK_SIZE_M=256, BLOCK_SIZE_N=256, BLOCK_SIZE_K=256, GROUP_SIZE_M=2, num_warps=4, num_stages=2, waves_per_eu=1, matrix_instr_nonkdim=32, cache_modifier=None, NUM_KSPLIT=1),
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


def _get_config(M, N, K):
    return _table_config(M)


def run(*, a, a_scales, w, w_scales, out) -> None:
    """out = dequant(a) @ dequant(w)^T: a [M, K // 2] and w [N, K // 2] packed e2m1, scales [., K // 32] e8m0."""
    M, K = a.shape  # K counts packed bytes (two fp4 values each)
    N, K = w.shape
    w = w.T  # the kernel expects (K // 2, N)

    config = _get_config(M, N, K)

    if config["NUM_KSPLIT"] > 1:
        SPLITK_BLOCK_SIZE, BLOCK_SIZE_K, NUM_KSPLIT = get_splitk(K, config["BLOCK_SIZE_K"], config["NUM_KSPLIT"])
        config["SPLITK_BLOCK_SIZE"] = SPLITK_BLOCK_SIZE
        config["BLOCK_SIZE_K"] = BLOCK_SIZE_K
        config["NUM_KSPLIT"] = NUM_KSPLIT

    if config["BLOCK_SIZE_K"] >= 2 * K:
        config["BLOCK_SIZE_K"] = triton.next_power_of_2(2 * K)
        config["SPLITK_BLOCK_SIZE"] = 2 * K
        config["NUM_KSPLIT"] = 1
    config["BLOCK_SIZE_K"] = max(config["BLOCK_SIZE_K"], 128)

    if config["NUM_KSPLIT"] > 1:
        y_pp = torch.empty((config["NUM_KSPLIT"], M, N), dtype=torch.float32, device=out.device)
    else:
        config["SPLITK_BLOCK_SIZE"] = 2 * K
        y_pp = None

    grid = lambda META: (  # noqa: E731
        META["NUM_KSPLIT"] * triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    _gemm_afp4wfp4_kernel[grid](
        a,
        w,
        out if config["NUM_KSPLIT"] == 1 else y_pp,
        a_scales,
        w_scales,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        w.stride(0),
        w.stride(1),
        0 if config["NUM_KSPLIT"] == 1 else y_pp.stride(0),
        out.stride(0) if config["NUM_KSPLIT"] == 1 else y_pp.stride(1),
        out.stride(1) if config["NUM_KSPLIT"] == 1 else y_pp.stride(2),
        a_scales.stride(0),
        a_scales.stride(1),
        w_scales.stride(0),
        w_scales.stride(1),
        **config,
    )

    if config["NUM_KSPLIT"] > 1:
        REDUCE_BLOCK_SIZE_M = 16
        REDUCE_BLOCK_SIZE_N = 64  # float32 partial sums
        ACTUAL_KSPLIT = triton.cdiv(K, config["SPLITK_BLOCK_SIZE"] // 2)
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
            triton.next_power_of_2(config["NUM_KSPLIT"]),
            ADD_BIAS=False,
            activation="",
            use_activation=False,
            KERNEL_NAME="_gemm_afp4_wfp4_reduce_kernel",
        )
