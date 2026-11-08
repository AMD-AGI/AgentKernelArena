# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Batched a8w8 GEMM from ROCm/aiter: bfloat16 activations quantized per token and 128-wide K group to fp8 inside the
kernel, fp8 weights with one per-tensor scale.

Provenance (ROCm/aiter at the revision recorded in task.yaml), all under aiter/ops/triton/: the device kernel of
_triton_kernels/gemm/batched/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py, the launcher
of gemm/batched/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py, `make_kernel_repr` from
utils/_triton/kernel_repr.py, and the N=128, K=512 table of configs/gemm/
gfx950-BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT-N=128-K=512.json inlined in the lookup
order of utils/gemm_config_utils.py.
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


_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_repr = make_kernel_repr(
    "_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_kernel",
    [
        "HAS_BIAS",
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "EVEN_K",
        "EVEN_MN",
        "cache_modifier",
    ],
)


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "EVEN_MN": lambda args: (args["M"] % args["BLOCK_SIZE_M"] == 0)
        and (args["N"] % args["BLOCK_SIZE_N"] == 0),
    }
)
@triton.jit(
    repr=_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_repr,
    do_not_specialize=["M", "N"],
)
def _batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
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
    stride_in_ab,
    stride_in_am,
    stride_in_ak,
    stride_in_bb,
    stride_in_bk,
    stride_in_bn,
    stride_in_cb,
    stride_in_cm,
    stride_in_cn,
    stride_in_biasb,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
    DTYPE_MIN: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_MN: tl.constexpr,
    cache_modifier: tl.constexpr,
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

    stride_ab = tl.cast(stride_in_ab, tl.int64)
    stride_am = tl.cast(stride_in_am, tl.int64)
    stride_ak = tl.cast(stride_in_ak, tl.int64)
    stride_bb = tl.cast(stride_in_bb, tl.int64)
    stride_bk = tl.cast(stride_in_bk, tl.int64)
    stride_bn = tl.cast(stride_in_bn, tl.int64)
    stride_cb = tl.cast(stride_in_cb, tl.int64)
    stride_cm = tl.cast(stride_in_cm, tl.int64)
    stride_cn = tl.cast(stride_in_cn, tl.int64)
    stride_biasb = tl.cast(stride_in_biasb, tl.int64)

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

    batch_id = tl.cast(batch_id, tl.int64)
    pid_m = tl.cast(pid_m, tl.int64)
    pid_n = tl.cast(pid_n, tl.int64)

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(batch_id >= 0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    if EVEN_MN:
        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    else:
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
    one_over_DTYPE_MAX = 1.0 / DTYPE_MAX
    b_scale = tl.load(b_scale_ptr)

    acc_dtype = tl.float32 if c_ptr.type.element_ty != tl.int8 else tl.int32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

    for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs, cache_modifier=cache_modifier)
        else:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        m = tl.maximum(tl.max(tl.abs(a), axis=-1), 1e-10)[:, None]
        a_scale = m.to(tl.float32) * one_over_DTYPE_MAX
        a_scale_recip = 1.0 / a_scale
        a = tl.clamp(a * a_scale_recip, DTYPE_MIN, DTYPE_MAX).to(b_ptr.dtype.element_ty)

        accumulator += tl.dot(a, b) * a_scale

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    accumulator *= b_scale

    if HAS_BIAS:
        if EVEN_MN:
            offs_bias = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        else:
            offs_bias = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        bias = tl.load(bias_ptr + batch_id * stride_biasb + offs_bias)
        accumulator = accumulator.to(bias_ptr.type.element_ty) + bias[None, :]

    c = accumulator.to(c_ptr.type.element_ty)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = (
        c_ptr
        + stride_cb * batch_id
        + stride_cm * offs_cm[:, None]
        + stride_cn * offs_cn[None, :]
    )
    if EVEN_MN:
        tl.store(c_ptrs, c)
    else:
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)


_CONFIGS = {
    "M_LEQ_16": dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=32, GROUP_SIZE_M=1, num_warps=8, num_stages=2, waves_per_eu=4, matrix_instr_nonkdim=16, cache_modifier='.cg'),
    "M_LEQ_32": dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, GROUP_SIZE_M=1, num_warps=4, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=16, cache_modifier='.cg'),
    "M_LEQ_64": dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=64, GROUP_SIZE_M=1, num_warps=8, num_stages=2, waves_per_eu=1, matrix_instr_nonkdim=16, cache_modifier='.cg'),
    "M_LEQ_128": dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, GROUP_SIZE_M=1, num_warps=8, num_stages=2, waves_per_eu=1, matrix_instr_nonkdim=32, cache_modifier=None),
    "M_LEQ_256": dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, GROUP_SIZE_M=1, num_warps=8, num_stages=2, waves_per_eu=1, matrix_instr_nonkdim=32, cache_modifier=None),
    "any": dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=128, GROUP_SIZE_M=1, num_warps=8, num_stages=2, waves_per_eu=2, matrix_instr_nonkdim=16, cache_modifier='.cg'),
}


_STANDARD_M_BOUNDS = (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)

_GROUP_SIZE = 128  # K group of the per-token activation quantization (the launcher's group_size argument)
_DTYPE_MAX = 448.0  # torch.finfo(torch.float8_e4m3fn).max, the launcher's DTYPE_MAX for the e4m3fn weights


def _get_config(M: int, N: int, K: int):
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


def run(*, x, wq, w_scale, out) -> None:
    """out[m, b] = (quantize(x[m, b]) @ wq[b]^T) * w_scale: x [M, B, K] bf16, wq [B, N, K] fp8 (e4m3fn), w_scale [1].

    The launch is batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(X=x, WQ=wq.transpose(-1, -2),
    w_scale=w_scale, group_size=128, YQ=out, transpose_bm=True, transpose_bm_in=True) of the MLA value up-projection:
    the activations are token-major and the output is written as [M, B, N].
    """
    M, B, K = x.shape
    N = wq.shape[1]
    WQ = wq.transpose(1, 2)

    config = _get_config(M, N, K)
    config["BLOCK_SIZE_K"] = _GROUP_SIZE
    config["kpack"] = 1

    grid = lambda META: (  # noqa: E731
        B,
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    _batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_kernel[grid](
        x,
        WQ,
        out,
        w_scale,
        None,
        M,
        N,
        K,
        x.stride(1),  # transpose_bm_in: x is [M, B, K]
        x.stride(0),
        x.stride(2),
        WQ.stride(0),
        WQ.stride(1),
        WQ.stride(2),
        out.stride(1),  # transpose_bm: out is [M, B, N]
        out.stride(0),
        out.stride(2),
        0,
        False,
        DTYPE_MAX=_DTYPE_MAX,
        DTYPE_MIN=-_DTYPE_MAX,
        **config,
    )
