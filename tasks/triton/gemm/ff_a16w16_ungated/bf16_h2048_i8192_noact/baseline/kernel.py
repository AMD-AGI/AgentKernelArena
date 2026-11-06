# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Standalone 16-bit (bf16/fp16) ungated feed-forward block Triton kernel.

Provenance: ported from aiter.ops.triton.gemm.feed_forward.ff_a16w16
(`ff_a16w16_nogate`), which composes two `gemm_a16w16` matmuls, plus the basic
16-bit GEMM device kernel `_gemm_a16_w16_kernel`
(aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16w16) and the elementwise
activation helpers (aiter.ops.triton._triton_kernels.activation). The XCD-remap /
grouped pid helpers (`remap_xcd`, `pid_grid`), the constexpr-aware kernel-naming
helper, and `_get_activation_from_str` are inlined; only the non-split-K
(`NUM_KSPLIT == 1`) triton path is kept (the gluon/gfx1250 backend and the split-K
reduce kernel are dropped), so the module depends only on `triton` + `torch`.

Op (transformer FFN, ungated):
    intermediate = act(X @ W_up^T)
    Y            = intermediate @ W_down^T
with X = [M, K] activations, W_up = [N, K], W_down = [K, N] (passed as [K, N] and
transposed internally), fp32 accumulation, and bf16/fp16 output. `activation` is
one of {"gelu", "gelu_tanh", "silu", "silu_exp2", "relu", None}.
"""

from typing import Optional

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Inlined helper utils (XCD remap, pid grid, kernel repr)
# ---------------------------------------------------------------------------
def _sanitize_constexpr_value(value):
    if value is None:
        return "NONE"
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else str(value)
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")
    return cleaned.upper() if cleaned else "NONE"


def make_kernel_repr(base_name, config_keys):
    """Inlined from utils/_triton/kernel_repr.py (constexpr-aware kernel naming)."""

    def _repr(specialization):
        constants = specialization.constants
        parts = [
            f"{key}_{_sanitize_constexpr_value(constants.get(key, None))}"
            for key in config_keys
        ]
        return f"{base_name}_{'_'.join(parts)}" if parts else base_name

    return _repr


@triton.jit
def remap_xcd(pid, GRID_MN, NUM_XCDS: tl.constexpr = 8):
    """Inlined from pid_preprocessing.remap_xcd (XCD-balanced pid remap)."""
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    tall_xcds = GRID_MN % NUM_XCDS
    if tall_xcds == 0:
        tall_xcds = tl.cast(NUM_XCDS, tall_xcds.type)
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
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
def pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M: tl.constexpr = 1):
    """Inlined from pid_preprocessing.pid_grid (1D->2D grouped pid map)."""
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


# ---------------------------------------------------------------------------
# Inlined activation helpers (from _triton_kernels/activation.py)
# ---------------------------------------------------------------------------
@triton.jit
def _silu_exp2(x):
    return x / (1.0 + tl.exp2(-(x * 1.44269504089)))


@triton.jit
def _silu(x):
    return _silu_exp2(x)


@triton.jit
def _tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _gelu(x):
    M_SQRT1_2 = 0.70710678118654752440
    ALPHA = M_SQRT1_2
    return 0.5 * x * (1.0 + tl.erf(x * ALPHA))


@triton.jit
def _gelu_tanh(x):
    M_SQRT2 = 1.41421356237309504880
    M_2_SQRTPI = 1.12837916709551257390
    BETA = M_SQRT2 * M_2_SQRTPI * 0.5
    KAPPA = 0.044715
    x_cube = x * x * x
    inner = BETA * (x + KAPPA * x_cube)
    return 0.5 * x * (1.0 + _tanh(inner))


@triton.jit
def _relu(x):
    return tl.maximum(0.0, x)


def _get_activation_from_str(activation: str):
    mapping = {
        "gelu": _gelu,
        "gelu_tanh": _gelu_tanh,
        "silu": _silu,
        "silu_exp2": _silu_exp2,
        "relu": _relu,
    }
    return mapping[activation]


_gemm_a16w16_repr = make_kernel_repr(
    "_gemm_a16_w16_kernel",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "NUM_KSPLIT",
        "SPLITK_BLOCK_SIZE",
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
    repr=_gemm_a16w16_repr,
    do_not_specialize=["M", "N"],
)
def _gemm_a16_w16_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
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
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_MN: tl.constexpr,
    cache_modifier: tl.constexpr,
    activation: tl.constexpr,
    use_activation: tl.constexpr,
    ADD_BIAS: tl.constexpr,
    SKIP_REDUCE: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_ck > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid_unified = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_unified = remap_xcd(pid_unified, num_pid_m * num_pid_n * NUM_KSPLIT, NUM_XCDS=8)
    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT

    if NUM_KSPLIT == 1:
        pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(pid_k >= 0)

    split_k_start = pid_k * SPLITK_BLOCK_SIZE
    if split_k_start < K:
        # Create pointers for first block of A and B input matrices
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
                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        else:
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        split_k_end = tl.minimum(split_k_start + SPLITK_BLOCK_SIZE, K)
        k_span = split_k_end - split_k_start
        num_k_iter = tl.cdiv(k_span, BLOCK_SIZE_K)

        for k in range(num_k_iter):
            # Load the next block of A and B, generate a mask by checking the K dimension.
            # If it is out of bounds, set it to 0.
            if EVEN_K:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs, cache_modifier=cache_modifier)
            else:
                a = tl.load(
                    a_ptrs, mask=offs_k[None, :] < k_span - k * BLOCK_SIZE_K, other=0.0
                )
                b = tl.load(
                    b_ptrs,
                    mask=offs_k[:, None] < k_span - k * BLOCK_SIZE_K,
                    other=0.0,
                    cache_modifier=cache_modifier,
                )
            accumulator = tl.dot(a, b, acc=accumulator)
            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        if use_activation and NUM_KSPLIT == 1:
            accumulator = activation(accumulator)

        # Write back the block of the output matrix C with masks.
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


def _get_config(M: int, N: int, K: int):
    """Static (non-split-K) config replacing the on-disk tuned-config lookup.

    A single robust bf16 tile is used with NUM_KSPLIT == 1 so the standalone
    kernel needs no config files and no split-K reduce pass. SPLITK_BLOCK_SIZE
    covers the whole K in one split.
    """
    block_m = 128 if M >= 128 else max(16, triton.next_power_of_2(M))
    block_n = 128 if N >= 128 else max(16, triton.next_power_of_2(N))
    block_k = 64 if K >= 64 else max(16, triton.next_power_of_2(K))
    return {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": block_k,
        "GROUP_SIZE_M": 8,
        "NUM_KSPLIT": 1,
        "SPLITK_BLOCK_SIZE": max(block_k, triton.next_power_of_2(K)),
        "cache_modifier": "",
        "num_warps": 4,
        "num_stages": 2,
        "waves_per_eu": 0,
    }


def gemm_a16w16(
    x,
    w,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
    activation: Optional[str] = None,
):
    """Computes 16 bit matrix multiplication Y = act(X @ W^T) (triton path)."""
    assert x.shape[1] == w.shape[1], "Incompatible matrix shapes."
    if not isinstance(dtype, torch.dtype):
        dtype = torch.bfloat16

    M, K = x.shape
    N, K = w.shape
    w = w.T

    if config is None:
        config = _get_config(M, N, K)

    if y is None:
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    grid = lambda META: (  # noqa: E731
        (
            META["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"])
        ),
    )
    _gemm_a16_w16_kernel[grid](
        x,
        w,
        None,
        y,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        0,
        y.stride(0),
        y.stride(1),
        activation=_get_activation_from_str(activation) if activation else "",
        use_activation=activation is not None,
        ADD_BIAS=False,
        SKIP_REDUCE=False,
        **config,
    )

    return y


def ff_a16w16_nogate(
    x,
    w_up,
    w_down,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    intermediate: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
    activation: Optional[str] = None,
):
    """
    Ungated feed-forward block: Y = act(X @ W_up^T) @ W_down^T.

    x: torch.Tensor (M, K)
    w_up: torch.Tensor (N, K)
    w_down: torch.Tensor (N, K)
    intermediate: torch.Tensor (M, N)
    y: torch.Tensor (M, K)
    activation: One of ("silu", "relu", "gelu", "gelu_tanh", "silu_exp2", None)
    """
    # Shape checks
    assert (
        x.shape[1] == w_up.shape[1] == w_down.shape[1]
    ), f"Incompatible matrix shapes: x:{x.shape}, w_up:{w_up.shape}, w_down:{w_down.shape}"
    assert (
        w_up.shape[0] == w_down.shape[0]
    ), f"Incompatible matrix shapes: w_up:{w_up.shape}, w_down:{w_down.shape}"

    M, K = x.shape
    N = w_up.shape[0]

    if intermediate is None:
        intermediate = torch.empty((M, N), dtype=dtype, device=x.device)

    intermediate = gemm_a16w16(
        x,
        w_up,
        dtype=dtype,
        y=intermediate,
        config=config,
        activation=activation,
    )

    if y is None:
        y = torch.empty((M, K), dtype=dtype, device=x.device)
    y = gemm_a16w16(intermediate, w_down.T, dtype=dtype, config=config, y=y)

    return y


def run(*, x, w_up, w_down, out):
    """Arena entry: out = (x @ w_up^T) @ w_down^T, both GEMMs in bf16 with fp32 accumulation.

    w_down is the [hidden, intermediate] weight of the down projection; the production function takes
    it as the transposed [intermediate, hidden] view.
    """
    ff_a16w16_nogate(x, w_up, w_down.T, torch.bfloat16, y=out, activation=None)
