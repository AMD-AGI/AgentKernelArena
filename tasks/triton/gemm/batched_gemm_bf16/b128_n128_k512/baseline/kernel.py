# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Batched bf16 GEMM Y[b] = X[b] @ W[b]^T from ROCm/aiter
(aiter/ops/triton/_triton_kernels/gemm/batched/batched_gemm_bf16.py and its launcher
aiter/ops/triton/gemm/batched/batched_gemm_bf16.py, Triton backend).

The tuned configuration is the gfx950 file configs/gemm/gfx950-BATCHED_GEMM-A16W16.json, inlined with the
upstream M_LEQ / M_GEQ / any lookup order. Both of its entries use NUM_KSPLIT = 1, so the split-K reduction
path of the launcher is not carried over.
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


_batched_gemm_bf16_repr = make_kernel_repr(
    "_batched_gemm_bf16_kernel",
    [
        "HAS_BIAS",
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "NUM_KSPLIT",
        "SPLITK_BLOCK_SIZE",
        "EVEN_K",
        "cache_modifier",
        "num_warps",
        "num_stages",
        "waves_per_eu",
    ],
)


@triton.heuristics(
    {
        "EVEN_K": lambda args: (args["K"] % args["SPLITK_BLOCK_SIZE"] == 0)
        and (args["SPLITK_BLOCK_SIZE"] % args["BLOCK_SIZE_K"] == 0),
    }
)
@triton.jit(repr=_batched_gemm_bf16_repr)
def _batched_gemm_bf16_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
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
    # stride along the split-K (partial-sum) dimension of the output; 0 when
    # NUM_KSPLIT == 1 (the output is the final (B, M, N) tensor).
    stride_ck,
    stride_biasb,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    cache_modifier: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
    waves_per_eu: tl.constexpr,
):
    """
    Note: this is Triton jited function and not meant to be called directly. Call batched_gemm_bf16 function
    below

    Computes the matmul C[i] = A[i] x B[i] for every i in a given batch and optionally adds a bias to each result.

    When NUM_KSPLIT > 1 the K dimension is split across NUM_KSPLIT program
    groups: each group reduces a SPLITK_BLOCK_SIZE-wide slice of K into its own
    partial-sum plane and the planes are summed by the split-K reduce kernel.

    Key parameters:
    - A: Batch tensor A with shape (B, M, K).
    - B: Batch tensor B with shape (B, K, N).
    - C: Batch tensor C with shape (B, M, N), or partial sums with shape
         (NUM_KSPLIT, B, M, N) when NUM_KSPLIT > 1.
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
    tl.assume(stride_biasb > 0)

    # -----------------------------------------------------------
    # Get batch program id
    batch_id = tl.program_id(axis=0)
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid_unified = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT

    if NUM_KSPLIT == 1:
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
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(pid_k >= 0)

    # Cast batch id and batch dimension strides to int64 to avoid int32 overflow during offset calculation
    # Note: If you're attempting to cast strides to int64 to prevent integer overflow, use `tl.cast` instead of `.to()`.
    # See https://github.com/ROCm/aiter/pull/597 for rationale
    batch_id = tl.cast(batch_id, tl.int64)
    stride_ab = tl.cast(stride_ab, tl.int64)
    stride_bb = tl.cast(stride_bb, tl.int64)
    stride_cb = tl.cast(stride_cb, tl.int64)

    split_k_start = pid_k * SPLITK_BLOCK_SIZE
    if split_k_start < K:
        # Create pointers for first block of A and B input matrices
        # Cast M/N offsets to int64 to avoid int32 overflow in offs*stride
        # products (offs_am*stride_am, offs_cm*stride_cm). Large per-batch
        # matrices (M*K or M*N > 2^31) overflow otherwise.
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        offs_k_split = split_k_start + offs_k
        offs_am = tl.cast(
            (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M, tl.int64
        )
        offs_bn = tl.cast(
            (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N, tl.int64
        )
        a_ptrs = a_ptr + (
            batch_id * stride_ab
            + offs_am[:, None] * stride_am
            + offs_k_split[None, :] * stride_ak
        )
        b_ptrs = b_ptr + (
            batch_id * stride_bb
            + offs_k_split[:, None] * stride_bk
            + offs_bn[None, :] * stride_bn
        )

        acc_dtype = tl.float32 if c_ptr.type.element_ty != tl.int8 else tl.int32
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        split_k_end = tl.minimum(split_k_start + SPLITK_BLOCK_SIZE, K)
        k_span = split_k_end - split_k_start
        num_k_iter = tl.cdiv(k_span, BLOCK_SIZE_K)

        for k in range(num_k_iter):
            # Load the next block of A and B, generate a mask by checking the K dimension.
            # If it is out of bounds, set it to 0.
            if EVEN_K:
                b = tl.load(b_ptrs, cache_modifier=cache_modifier)
                a = tl.load(a_ptrs)
            else:
                b = tl.load(
                    b_ptrs,
                    mask=offs_k[:, None] < k_span - k * BLOCK_SIZE_K,
                    other=0.0,
                    cache_modifier=cache_modifier,
                )
                a = tl.load(
                    a_ptrs, mask=offs_k[None, :] < k_span - k * BLOCK_SIZE_K, other=0.0
                )

            accumulator = tl.dot(a, b, acc=accumulator)

            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        # Add bias only on the non-split path; for NUM_KSPLIT > 1 the bias is
        # added once during the split-K reduction so it is not counted per-split.
        if HAS_BIAS and NUM_KSPLIT == 1:
            offs_bias = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
            bias = tl.load(bias_ptr + batch_id * stride_biasb + offs_bias).to(
                dtype=acc_dtype
            )
            accumulator += bias[None, :]

        c = accumulator.to(c_ptr.type.element_ty)

        # Write back the block of the output matrix C with masks.
        # int64 to avoid int32 overflow in offs_cm*stride_cm for large M*N.
        offs_cm = tl.cast(pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M), tl.int64)
        offs_cn = tl.cast(pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N), tl.int64)
        c_ptrs = (
            c_ptr
            + pid_k * stride_ck
            + stride_cb * batch_id
            + stride_cm * offs_cm[:, None]
            + stride_cn * offs_cn[None, :]
        )
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

        tl.store(c_ptrs, c, mask=c_mask)


_CONFIGS = {
    "M_GEQ_4096": dict(BLOCK_SIZE_M=256, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64, GROUP_SIZE_M=4, num_warps=8,
                       num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=16, NUM_KSPLIT=1),
    "any": dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32, GROUP_SIZE_M=1, num_warps=8, num_stages=2,
                waves_per_eu=2, matrix_instr_nonkdim=16, NUM_KSPLIT=1),
}
_STANDARD_M_BOUNDS = (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)


def _get_config(M, K):
    """get_gemm_config over the table above, then compute_splitk_params for NUM_KSPLIT = 1."""
    config = None
    for bound in _STANDARD_M_BOUNDS:
        if M <= bound and f"M_LEQ_{bound}" in _CONFIGS:
            config = dict(_CONFIGS[f"M_LEQ_{bound}"])
            break
    if config is None:
        for bound in reversed(_STANDARD_M_BOUNDS):
            if M >= bound and f"M_GEQ_{bound}" in _CONFIGS:
                config = dict(_CONFIGS[f"M_GEQ_{bound}"])
                break
    if config is None:
        config = dict(_CONFIGS["any"])
    config["cache_modifier"] = None
    config["SPLITK_BLOCK_SIZE"] = triton.cdiv(K, config["NUM_KSPLIT"])
    if config["BLOCK_SIZE_K"] > config["SPLITK_BLOCK_SIZE"]:
        config["BLOCK_SIZE_K"] = triton.next_power_of_2(config["SPLITK_BLOCK_SIZE"])
        if config["BLOCK_SIZE_K"] > config["SPLITK_BLOCK_SIZE"]:
            config["BLOCK_SIZE_K"] = config["BLOCK_SIZE_K"] // 2
    config["BLOCK_SIZE_K"] = max(config["BLOCK_SIZE_K"], 16)
    return config


def run(*, x, w, out):
    batch, M, K = x.shape
    N = w.shape[1]
    wt = w.transpose(1, 2)  # (B, N, K) -> (B, K, N), a view
    config = _get_config(M, K)
    grid = lambda META: (  # noqa: E731
        batch,
        META["NUM_KSPLIT"] * triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    _batched_gemm_bf16_kernel[grid](
        x, wt, out, None,
        M, N, K,
        x.stride(0), x.stride(1), x.stride(2),
        wt.stride(0), wt.stride(1), wt.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        0,
        0,
        False,
        **config,
    )
