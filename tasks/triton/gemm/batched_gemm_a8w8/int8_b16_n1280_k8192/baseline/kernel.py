# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Batched int8 GEMM with per-token and per-channel scales, from ROCm/aiter
(aiter/ops/triton/_triton_kernels/gemm/batched/batched_gemm_a8w8.py and its launcher
aiter/ops/triton/gemm/batched/batched_gemm_a8w8.py).

The tuned configuration table is the gfx950 file configs/gemm/gfx950-BATCHED_GEMM-A8W8.json, inlined
with the upstream M_LEQ / M_GEQ / any lookup order.
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


_batched_gemm_a8w8_repr = make_kernel_repr(
    "_batched_gemm_a8w8_kernel",
    [
        "HAS_BIAS",
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "EVEN_K",
        "GRID_MN",
    ],
)


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@triton.jit(repr=_batched_gemm_a8w8_repr)
def _batched_gemm_a8w8_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    bias_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    stride_ascaleb,
    stride_bscaleb,
    stride_biasb,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    GRID_MN: tl.constexpr,
):
    """
    Note: this is Triton jited function and not meant to be called directly. Call batched_gemm_a8w8 function
    below

    Computes the matmul C[i] = A[i] x B[i] and applies a conversion scale for every i in a given batch.
    Optionally, adds a bias to each result.

    The conversion scale for each matmul is received in the form of two 1D tensors that are multiplied to form a
    2D one before being applied.

    Key parameters:
    - A: Batch tensor A with shape (B, M, K).
    - B: Batch tensor B with shape (B, K, N).
    - C: Batch tensor C with shape (B, M, N).
    - A_scale: First scale batch tensor with shape (B, M, 1).
    - B_scale: Second scale batch tensor with shape (B, 1, N).
    - Bias: Bias batch tensor with shape (B, 1, N).
    """

    tl.assume(stride_ab > 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bb > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cb > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_ascaleb > 0)
    tl.assume(stride_bscaleb > 0)
    tl.assume(stride_biasb > 0)

    # -----------------------------------------------------------
    # Get batch program id
    batch_id = tl.program_id(axis=0)
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    # Cast batch id and batch dimension strides to int64 to avoid int32 overflow during offset calculation
    # Note: If you're attempting to cast strides to int64 to prevent integer overflow, use `tl.cast` instead of `.to()`.
    # See https://github.com/ROCm/aiter/pull/597 for rationale
    batch_id = tl.cast(batch_id, tl.int64)
    stride_ab = tl.cast(stride_ab, tl.int64)
    stride_bb = tl.cast(stride_bb, tl.int64)
    stride_cb = tl.cast(stride_cb, tl.int64)

    # Create pointers for first block of A and B input matrices
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    a_ptrs = a_ptr + (
        batch_id * stride_ab
        + offs_am[:, None] * stride_am
        + offs_k[None, :] * stride_ak
    )
    b_ptrs = b_ptr + (
        batch_id * stride_bb
        + offs_k[:, None] * stride_bk
        + offs_bn[None, :] * stride_bn
    )

    # Create pointers for the scale tensors and load them
    offs_a_scale = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_b_scale = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    a_scale = tl.load(a_scale_ptr + batch_id * stride_ascaleb + offs_a_scale)
    b_scale = tl.load(b_scale_ptr + batch_id * stride_bscaleb + offs_b_scale)

    acc_dtype = tl.float32 if c_ptr.type.element_ty != tl.int8 else tl.int32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

    for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        accumulator += tl.dot(a, b)

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Apply scale
    accumulator *= a_scale[:, None] * b_scale[None, :]

    # Add bias
    if HAS_BIAS:
        offs_bias = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        bias = tl.load(bias_ptr + batch_id * stride_biasb + offs_bias)
        accumulator = accumulator.to(bias_ptr.type.element_ty) + bias[None, :]

    c = accumulator.to(c_ptr.type.element_ty)

    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = (
        c_ptr
        + stride_cb * batch_id
        + stride_cm * offs_cm[:, None]
        + stride_cn * offs_cn[None, :]
    )
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(c_ptrs, c, mask=c_mask)


_CONFIGS = {
    "M_LEQ_128": dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, GROUP_SIZE_M=1, num_warps=4,
                      num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=16),
    "M_LEQ_256": dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, GROUP_SIZE_M=1, num_warps=4,
                      num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=16),
    "M_LEQ_512": dict(BLOCK_SIZE_M=256, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64, GROUP_SIZE_M=1, num_warps=4,
                      num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=16),
    "M_LEQ_1024": dict(BLOCK_SIZE_M=256, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64, GROUP_SIZE_M=1, num_warps=4,
                       num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=16),
    "M_LEQ_2048": dict(BLOCK_SIZE_M=256, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64, GROUP_SIZE_M=1, num_warps=4,
                       num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=16),
    "M_GEQ_4096": dict(BLOCK_SIZE_M=256, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64, GROUP_SIZE_M=1, num_warps=4,
                       num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=16),
    "any": dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32, GROUP_SIZE_M=1, num_warps=8, num_stages=2,
                waves_per_eu=2, matrix_instr_nonkdim=16),
}
_STANDARD_M_BOUNDS = (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)


def _get_config(M):
    """The upstream get_gemm_config lookup over the table above."""
    for bound in _STANDARD_M_BOUNDS:
        key = f"M_LEQ_{bound}"
        if M <= bound and key in _CONFIGS:
            return dict(_CONFIGS[key])
    for bound in reversed(_STANDARD_M_BOUNDS):
        key = f"M_GEQ_{bound}"
        if M >= bound and key in _CONFIGS:
            return dict(_CONFIGS[key])
    return dict(_CONFIGS["any"])


def run(*, x, x_scale, w, w_scale, out):
    batch, M, K = x.shape
    N = w.shape[1]
    wt = w.transpose(1, 2)  # (B, N, K) -> (B, K, N), a view
    config = _get_config(M)
    grid = lambda META: (  # noqa: E731
        batch,
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    _batched_gemm_a8w8_kernel[grid](
        x, wt, out, x_scale, w_scale, None,
        M, N, K,
        x.stride(0), x.stride(1), x.stride(2),
        wt.stride(0), wt.stride(1), wt.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        x_scale.stride(0), w_scale.stride(0), 0,
        False,
        **config,
    )
