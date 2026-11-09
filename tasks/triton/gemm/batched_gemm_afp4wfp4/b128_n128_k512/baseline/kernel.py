# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Batched afp4wfp4 GEMM from ROCm/aiter: MXFP4 activations times MXFP4 weights per batch element, e8m0 1x32 scales.

Provenance (ROCm/aiter at the revision recorded in task.yaml): the device kernel and `_get_config` of
aiter/ops/triton/_triton_kernels/gemm/batched/batched_gemm_afp4wfp4.py, the launcher of
aiter/ops/triton/gemm/batched/batched_gemm_afp4wfp4.py, `remap_xcd`, `pid_grid` and `make_kernel_repr` from
aiter/ops/triton/utils/_triton/, and the N=128, K=512 table
aiter/ops/triton/configs/gemm/gfx950-BATCHED_GEMM-AFP4WFP4-N=128-K=512.json inlined in the lookup order of
aiter/ops/triton/utils/gemm_config_utils.py. Every row of that table has NUM_KSPLIT == 1, so the split-K
reduce kernel, `get_splitk` and the float32 partial-sum buffer are dropped.
"""
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


_batched_gemm_afp4_wfp4_repr = make_kernel_repr(
    "_batched_gemm_afp4_wfp4_kernel",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "NUM_KSPLIT",
        "SPLITK_BLOCK_SIZE",
        "EVEN_K",
        "GRID_MN",
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
@triton.jit(repr=_batched_gemm_afp4_wfp4_repr)
def _batched_gemm_afp4_wfp4_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scales_ptr,
    b_scales_ptr,
    M,
    N,
    K,
    stride_in_ab,
    stride_in_am,
    stride_in_ak,
    stride_in_bb,
    stride_in_bk,
    stride_in_bn,
    stride_in_cb,
    stride_in_ck,
    stride_in_cm,
    stride_in_cn,
    stride_in_asb,
    stride_in_asm,
    stride_in_ask,
    stride_in_bsb,
    stride_in_bsn,
    stride_in_bsk,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    GRID_MN: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A and B inputs are in the microscale fp4 (mxfp4) format.
    A_scales and B_scales are in e8m0 format.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """

    tl.assume(stride_in_ab > 0)
    tl.assume(stride_in_am > 0)
    tl.assume(stride_in_ak > 0)
    tl.assume(stride_in_bb > 0)
    tl.assume(stride_in_bk > 0)
    tl.assume(stride_in_bn > 0)
    tl.assume(stride_in_cb > 0)
    tl.assume(stride_in_cm > 0)
    tl.assume(stride_in_cn > 0)
    tl.assume(stride_in_asb > 0)
    tl.assume(stride_in_asm > 0)
    tl.assume(stride_in_ask > 0)
    tl.assume(stride_in_bsb > 0)
    tl.assume(stride_in_bsk > 0)
    tl.assume(stride_in_bsn > 0)

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid_batch = tl.program_id(axis=0)
    pid_unified = tl.program_id(axis=1)
    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # Cast batch id and batch dimension strides to int64 to avoid int32 overflow during offset calculation
    # Note: If you're attempting to cast strides to int64 to prevent integer overflow, use `tl.cast` instead of `.to()`.
    # See https://github.com/ROCm/aiter/pull/597 for rationale
    # stride_ab = tl.cast(stride_ab, tl.int64)
    # stride_bb = tl.cast(stride_bb, tl.int64)
    # stride_cb = tl.cast(stride_cb, tl.int64)
    # pid_batch = tl.cast(pid_batch, tl.int64)

    stride_ab = tl.cast(stride_in_ab, tl.int64)
    stride_am = tl.cast(stride_in_am, tl.int64)
    stride_ak = tl.cast(stride_in_ak, tl.int64)
    stride_bb = tl.cast(stride_in_bb, tl.int64)
    stride_bk = tl.cast(stride_in_bk, tl.int64)
    stride_bn = tl.cast(stride_in_bn, tl.int64)
    stride_cb = tl.cast(stride_in_cb, tl.int64)
    stride_ck = tl.cast(stride_in_ck, tl.int64)
    stride_cm = tl.cast(stride_in_cm, tl.int64)
    stride_cn = tl.cast(stride_in_cn, tl.int64)
    stride_asb = tl.cast(stride_in_asb, tl.int64)
    stride_asm = tl.cast(stride_in_asm, tl.int64)
    stride_ask = tl.cast(stride_in_ask, tl.int64)
    stride_bsb = tl.cast(stride_in_bsb, tl.int64)
    stride_bsk = tl.cast(stride_in_bsk, tl.int64)
    stride_bsn = tl.cast(stride_in_bsn, tl.int64)

    if NUM_KSPLIT == 1:
        remap_xcd(pid, GRID_MN)

        pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    tl.assume(pid_batch >= 0)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
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
            pid_batch * stride_ab
            + offs_am[:, None] * stride_am
            + offs_k_split[None, :] * stride_ak
        )
        b_ptrs = b_ptr + (
            pid_batch * stride_bb
            + offs_k_split[:, None] * stride_bk
            + offs_bn[None, :] * stride_bn
        )
        # Create pointers for the first block of A and B scales
        offs_ks = (pid_k * (SPLITK_BLOCK_SIZE // SCALE_GROUP_SIZE)) + tl.arange(
            0, BLOCK_SIZE_K // SCALE_GROUP_SIZE
        )
        a_scale_ptrs = (
            a_scales_ptr
            + pid_batch * stride_asb
            + offs_am[:, None] * stride_asm
            + offs_ks[None, :] * stride_ask
        )
        # B scales are N x K even though B operand is K x N.
        b_scale_ptrs = (
            b_scales_ptr
            + pid_batch * stride_bsb
            + offs_bn[:, None] * stride_bsn
            + offs_ks[None, :] * stride_bsk
        )

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter):
            a_scales = tl.load(a_scale_ptrs)
            b_scales = tl.load(b_scale_ptrs)
            # a_scales = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_K//SCALE_GROUP_SIZE), 127, dtype=tl.uint8)
            # b_scales = tl.full((BLOCK_SIZE_N, BLOCK_SIZE_K//SCALE_GROUP_SIZE), 127, dtype=tl.uint8)
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
                    b_ptrs, mask=offs_k[:, None] < K - k * (BLOCK_SIZE_K // 2), other=0
                )

            accumulator = tl.dot_scaled(
                a, a_scales, "e2m1", b, b_scales, "e2m1", acc=accumulator
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
            + pid_batch * stride_cb
            + stride_cm * offs_cm[:, None]
            + stride_cn * offs_cn[None, :]
            + pid_k * stride_ck
        )
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)


_CONFIGS = {
    "M_LEQ_16": dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128, GROUP_SIZE_M=1, num_warps=4, num_stages=2, waves_per_eu=6, matrix_instr_nonkdim=16, cache_modifier='.cg', NUM_KSPLIT=1),
    "M_LEQ_32": dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=32, BLOCK_SIZE_K=512, GROUP_SIZE_M=1, num_warps=4, num_stages=2, waves_per_eu=4, matrix_instr_nonkdim=16, cache_modifier='.cg', NUM_KSPLIT=1),
    "M_LEQ_64": dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128, GROUP_SIZE_M=1, num_warps=4, num_stages=3, waves_per_eu=1, matrix_instr_nonkdim=16, cache_modifier='.cg', NUM_KSPLIT=1),
    "M_LEQ_128": dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128, GROUP_SIZE_M=1, num_warps=4, num_stages=3, waves_per_eu=1, matrix_instr_nonkdim=16, cache_modifier='.cg', NUM_KSPLIT=1),
    "M_LEQ_256": dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128, GROUP_SIZE_M=2, num_warps=4, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=16, cache_modifier=None, NUM_KSPLIT=1),
    "any": dict(BLOCK_SIZE_M=256, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128, GROUP_SIZE_M=32, num_warps=8, num_stages=2, waves_per_eu=1, matrix_instr_nonkdim=16, cache_modifier=None, NUM_KSPLIT=1),
}


_M_BOUNDS = (4, 8, 16, 32, 64, 128, 256, 320, 512, 1024, 2048, 4096, 8192)


def _table_config(M):
    """Row of the tuned table for M, in the lookup order of AITER's get_gemm_config (custom bounds of _get_config)."""
    for bound in _M_BOUNDS:
        key = f"M_LEQ_{bound}"
        if M <= bound and key in _CONFIGS:
            return dict(_CONFIGS[key])
    for bound in reversed(_M_BOUNDS):
        key = f"M_GEQ_{bound}"
        if M >= bound and key in _CONFIGS:
            return dict(_CONFIGS[key])
    return dict(_CONFIGS["any"])


def _get_config(M: int, N: int, K: int):
    # K counts packed bytes (two fp4 values each). NUM_KSPLIT is 1 in every row of the table.
    config = _table_config(M)
    config["SPLITK_BLOCK_SIZE"] = 2 * K
    if config["BLOCK_SIZE_K"] >= 2 * K:
        config["BLOCK_SIZE_K"] = triton.next_power_of_2(2 * K)
        config["SPLITK_BLOCK_SIZE"] = 2 * K
    return config


def run(*, x, x_scales, w, w_scales, out) -> None:
    """out[b] = dequant(x[b]) @ dequant(w[b])^T: x [B, M, K // 2] and w [B, N, K // 2] packed e2m1, scales [., ., K // 32] e8m0."""
    w = w.transpose(1, 2)  # the kernel expects (B, K // 2, N)
    Batch, M, K = x.shape  # K counts packed bytes
    _, _, N = w.shape
    config = _get_config(M, N, K)

    grid = lambda META: (  # noqa: E731
        Batch,
        META["NUM_KSPLIT"] * triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    _batched_gemm_afp4_wfp4_kernel[grid](
        x,
        w,
        out,
        x_scales,
        w_scales,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        w.stride(0),
        w.stride(1),
        w.stride(2),
        out.stride(0),
        0,
        out.stride(1),
        out.stride(2),
        x_scales.stride(0),
        x_scales.stride(1),
        x_scales.stride(2),
        w_scales.stride(0),
        w_scales.stride(1),
        w_scales.stride(2),
        **config,
    )
