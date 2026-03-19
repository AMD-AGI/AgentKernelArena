# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

# The kernel in this file is adapted from FlagGems' topk:
# https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/ops/topk.py

#  Top-K on GPU:  1-stage (tiny rows) + 2-stage (large rows) Triton kernels,
from __future__ import annotations
from typing import Tuple
import math
import torch
import triton
import triton.language as tl
import triton.language.core as core
from triton.language.standard import _log2, zeros_like


class AiterTritonLogger:
    def info(self, *args, **kwargs):
        pass


_LOGGER = AiterTritonLogger()


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


def make_kernel_repr(base_name, config_keys):
    def _repr(specialization):
        constants = specialization.constants
        name_parts = []

        for key in config_keys:
            value = constants.get(key, None)
            symbol = _sanitize_constexpr_value(value)
            name_parts.append(f"{key}_{symbol}")

        if not name_parts:
            return base_name

        suffix = "_".join(name_parts)
        return f"{base_name}_{suffix}"

    return _repr


_topk_kernel_repr = make_kernel_repr(
    "_topk_kernel",
    [
        "M",
        "K",
        "BLOCK",
    ],
)

_topk_stage1_kernel_repr = make_kernel_repr(
    "topk_stage1_kernel",
    [
        "N",
        "CHUNK_SIZE",
        "DESCENDING",
    ],
)

_topk_stage2_kernel_repr = make_kernel_repr(
    "topk_stage2_kernel",
    [
        "k",
        "N",
        "BLOCK_SIZE",
        "DESCENDING",
    ],
)


# 1-STAGE KERNEL (tiny rows)
@triton.jit(repr=_topk_kernel_repr)
def _topk_kernel(
    X,
    OUT_V,
    OUT_I,
    stride_xm,
    stride_ovm,
    stride_oim,
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    FILL_VALUE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_ptr = X + pid * stride_xm
    offs = tl.arange(0, BLOCK)
    mask = offs < M
    # FILL_VALUE = tl.constexpr(torch.finfo(torch.float32).min)
    vals = tl.load(row_ptr + offs, mask=mask, other=FILL_VALUE).to(tl.float32)
    idxs = offs.to(tl.int64)

    out_v_ptr = OUT_V + pid * stride_ovm
    out_i_ptr = OUT_I + pid * stride_oim

    # unrolled exactly K iterations -- no break/continue needed
    for j in core.static_range(0, K):
        vmax = tl.max(vals, axis=0)
        eq = vals == vmax
        big = tl.where(
            eq, tl.zeros_like(idxs), tl.zeros_like(idxs) + BLOCK
        )  # BLOCK as int64
        arg = tl.min(idxs + big, axis=0)

        tl.store(out_v_ptr + j, vmax)
        tl.store(out_i_ptr + j, arg)

        vals = tl.where(idxs == arg, FILL_VALUE, vals)


# 2-STAGE KERNEL (large rows)
@triton.jit(repr=_topk_stage1_kernel_repr)
def topk_stage1_kernel(
    y_ptr,
    index_ptr,
    x_ptr,
    k,
    N: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    DESCENDING: tl.constexpr,
    FILL_VALUE: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_chunk_idx = tl.program_id(1)
    chunk_num = tl.num_programs(1)

    y_ptr += cur_batch * chunk_num * k + cur_chunk_idx * k
    index_ptr += cur_batch * chunk_num * k + cur_chunk_idx * k

    chunk_offset = cur_chunk_idx * CHUNK_SIZE
    x_ptr += cur_batch * N + chunk_offset

    cols = tl.arange(0, CHUNK_SIZE)
    mask = (chunk_offset + cols) < N

    x_val = tl.load(x_ptr + cols, mask=mask, other=FILL_VALUE).to(tl.float32)
    for k_idx in range(k):
        if DESCENDING:
            chunk_select_val, chunk_select_idx = tl.max(
                x_val, axis=0, return_indices=True
            )
        else:
            chunk_select_val, chunk_select_idx = tl.min(
                x_val, axis=0, return_indices=True
            )

        tl.store(y_ptr + k_idx, chunk_select_val)
        tl.store(index_ptr + k_idx, chunk_select_idx + chunk_offset)

        x_val = tl.where(
            cols == chunk_select_idx,
            FILL_VALUE,
            x_val,
        )


@triton.jit
def _compare_and_swap(x, ids, flip, i: core.constexpr, n_dims: core.constexpr):
    n_outer: core.constexpr = x.numel >> n_dims
    shape: core.constexpr = [n_outer * 2**i, 2, 2 ** (n_dims - i - 1)]

    y = core.reshape(x, shape)
    y_idx = core.reshape(ids, shape)

    # slice left/right with 'stride' 2**(n_dims - i - 1)
    mask = core.arange(0, 2)[None, :, None]
    left = core.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape).to(x.dtype)
    right = core.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape).to(x.dtype)
    left = core.reshape(left, x.shape)
    right = core.reshape(right, x.shape)

    left_idx = core.broadcast_to(tl.sum(y_idx * (1 - mask), 1)[:, None, :], shape).to(
        ids.dtype
    )
    right_idx = core.broadcast_to(tl.sum(y_idx * mask, 1)[:, None, :], shape).to(
        ids.dtype
    )
    left_idx = core.reshape(left_idx, ids.shape)
    right_idx = core.reshape(right_idx, ids.shape)

    # actual compare-and-swap
    if core.constexpr(x.dtype.primitive_bitwidth) == 8:
        idtype = core.int8
    elif core.constexpr(x.dtype.primitive_bitwidth) == 16:
        idtype = core.int16
    elif core.constexpr(x.dtype.primitive_bitwidth) == 32:
        idtype = core.int32
    elif core.constexpr(x.dtype.primitive_bitwidth) == 64:
        idtype = core.int64
    else:
        raise ValueError("Unsupported dtype")

    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)

    cond = (left > right) ^ flip
    ret = ix ^ core.where(cond, ileft ^ iright, zeros_like(ix))

    if core.constexpr(ids.dtype.primitive_bitwidth) == 8:
        idx_dtype = core.int8
    elif core.constexpr(ids.dtype.primitive_bitwidth) == 16:
        idx_dtype = core.int16
    elif core.constexpr(ids.dtype.primitive_bitwidth) == 32:
        idx_dtype = core.int32
    elif core.constexpr(ids.dtype.primitive_bitwidth) == 64:
        idx_dtype = core.int64
    else:
        raise ValueError("Unsupported dtype")

    ileft_idx = left_idx.to(idx_dtype, bitcast=True)
    iright_idx = right_idx.to(idx_dtype, bitcast=True)
    ix_idx = ids.to(idx_dtype, bitcast=True)
    ret_idx = ix_idx ^ core.where(cond, ileft_idx ^ iright_idx, zeros_like(ix_idx))

    return ret.to(x.dtype, bitcast=True), ret_idx.to(ids.dtype, bitcast=True)


@triton.jit
def _bitonic_merge(
    x, ids, stage: core.constexpr, order: core.constexpr, n_dims: core.constexpr
):
    """
    order_type 0 == ascending
    order_type 1 == descending
    order_type 2 == alternating
    """
    n_outer: core.constexpr = x.numel >> n_dims
    core.static_assert(stage <= n_dims)
    # flip denotes whether to re-arrange sub-sequences of elements in ascending or
    # descending order.
    # if flip = 00000000... then all elements will be re-arranged ascendingly at this stage
    # if flip = 00110011... then all the elements will be re-arranged alternatingly (with
    # a stride of 2) at this stage
    if order == 2:
        shape: core.constexpr = [n_outer * 2 ** (n_dims - 1 - stage), 2, 2**stage]
        flip = core.reshape(
            core.broadcast_to(core.arange(0, 2)[None, :, None], shape), x.shape
        )
    else:
        flip = order
    # perform `stage` rounds of `compare-and-swap`
    for i in core.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids


@triton.jit
def argsort(x, ids, dim: tl.constexpr, descending: core.constexpr):
    # handle default dimension or check that it is the most minor dim
    _dim: core.constexpr = dim
    n_dims: core.constexpr = _log2(x.shape[_dim])
    for i in core.static_range(1, n_dims + 1):
        x, ids = _bitonic_merge(x, ids, i, 2 if i < n_dims else descending, n_dims)
    return x, ids


@triton.jit(repr=_topk_stage2_kernel_repr)
def topk_stage2_kernel(
    y_ptr,
    index_ptr,
    chunk_x,
    chunk_index,
    k: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DESCENDING: tl.constexpr,
    FILL_VALUE: tl.constexpr,
    MASK_INDEX_VAL: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    chunk_x += cur_batch * N
    chunk_index += cur_batch * N
    y_ptr += cur_batch * k
    index_ptr += cur_batch * k

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # FILL_VALUE = tl.constexpr(
    #    torch.finfo(torch.float32).min if DESCENDING else torch.finfo(torch.float32).max
    # )
    # mask_index_val = (
    #    tl.constexpr(torch.iinfo(torch.int32).min)
    #    if DESCENDING
    #    else tl.constexpr(torch.iinfo(torch.int32).max)
    # )

    chunk_x_val = tl.load(chunk_x + cols, mask=mask, other=FILL_VALUE).to(tl.float32)
    chunk_index_val = tl.load(chunk_index + cols, mask=mask, other=MASK_INDEX_VAL).to(
        tl.int32
    )

    sorted_chunk_x, sorted_chunk_index = argsort(
        chunk_x_val, chunk_index_val, 0, descending=DESCENDING
    )
    tl.store(y_ptr + cols, sorted_chunk_x, mask=cols < k)
    tl.store(index_ptr + cols, sorted_chunk_index, mask=cols < k)


# Pre-computed block size lookup table for next power of 2
_BLOCK_TABLE = {}
for _m in range(1, 8193):
    _b = max(16, _m)
    _b -= 1
    _b |= _b >> 1
    _b |= _b >> 2
    _b |= _b >> 4
    _b |= _b >> 8
    _b |= _b >> 16
    _b += 1
    if _b > 8192:
        _b = 8192
    _BLOCK_TABLE[_m] = _b

# Pre-computed num_warps lookup tuned for AMD MI300X
_WARPS_TABLE = {16: 1, 32: 1, 64: 1, 128: 2, 256: 4, 512: 4, 1024: 8, 2048: 8, 4096: 16, 8192: 16}

# Cache frequently used constants
_FILL_MIN = torch.finfo(torch.float32).min
_FILL_MAX = torch.finfo(torch.float32).max
_MASK_IDX_MIN = torch.iinfo(torch.int32).min
_MASK_IDX_MAX = torch.iinfo(torch.int32).max

# Stage1 kernel num_warps tuned for AMD MI300X (separate from 1-stage kernel)
_STAGE1_WARPS = {256: 4, 512: 4, 1024: 8, 2048: 8, 4096: 8, 8192: 8}


def one_stage_topk(
    x: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, M = x.shape
    BLOCK = _BLOCK_TABLE.get(max(M, k), 8192)

    dev = x.device
    out_v = torch.empty((B, k), device=dev, dtype=x.dtype)
    out_i = torch.empty((B, k), device=dev, dtype=torch.int64)

    nw = _WARPS_TABLE.get(BLOCK, 8)
    # Single pipeline stage - kernel is compute-bound (iterative max-reduce)
    ns = 1

    _topk_kernel[(B,)](
        x,
        out_v,
        out_i,
        M,   # stride_xm for contiguous
        k,   # stride_ovm for contiguous
        k,   # stride_oim for contiguous
        M=M,
        K=k,
        BLOCK=BLOCK,
        FILL_VALUE=_FILL_MIN,
        num_warps=nw,
        num_stages=ns,
    )
    return out_v, out_i


def two_stage_topk(x, k, dim=-1, largest=True):
    descending = largest

    topk_elem_cnt = x.shape[dim]
    batch_size = x.shape[0] if x.ndim == 2 else math.prod(x.shape) // topk_elem_cnt

    # Larger chunks = fewer chunks = smaller stage2 sort = faster
    if topk_elem_cnt <= 4096:
        chunk_size = 2048
    elif topk_elem_cnt <= 16384:
        chunk_size = 4096
    else:
        chunk_size = 8192

    if chunk_size < k:
        chunk_size = triton.next_power_of_2(k)

    chunk_num = (topk_elem_cnt + chunk_size - 1) // chunk_size

    dev = x.device
    total_stage1 = batch_size * chunk_num * k
    stage1_out = torch.empty(total_stage1, device=dev, dtype=x.dtype)
    stage1_out_idx = torch.empty(total_stage1, device=dev, dtype=torch.int64)

    out_shape = x.shape[:-1] + (k,)
    stage2_out = torch.empty(out_shape, device=dev, dtype=x.dtype)
    stage2_out_idx = torch.empty(out_shape, device=dev, dtype=torch.int64)

    fill_val = _FILL_MIN if descending else _FILL_MAX
    mask_idx = _MASK_IDX_MIN if descending else _MASK_IDX_MAX

    # num_warps=8 is optimal for CHUNK_SIZE=8192 on MI300X (benchmarked)
    stage1_nw = _STAGE1_WARPS.get(chunk_size, 8)
    topk_stage1_kernel[
        batch_size,
        chunk_num,
    ](
        stage1_out,
        stage1_out_idx,
        x,
        k,
        topk_elem_cnt,
        chunk_size,
        descending,
        fill_val,
        num_warps=stage1_nw,
    )
    stage2_elem_cnt = chunk_num * k
    BLOCK_SIZE = _BLOCK_TABLE.get(stage2_elem_cnt, triton.next_power_of_2(stage2_elem_cnt))

    stage2_nw = _WARPS_TABLE.get(BLOCK_SIZE, 4)
    topk_stage2_kernel[batch_size,](
        stage2_out,
        stage2_out_idx,
        stage1_out,
        stage1_out_idx,
        k,
        stage2_elem_cnt,
        BLOCK_SIZE,
        descending,
        fill_val,
        mask_idx,
        num_warps=stage2_nw,
    )

    return (stage2_out, stage2_out_idx)


# For dispatcher - increased to handle larger rows in 1-stage for better perf
MAX_TINY_ROW = 8192

"""
Triton Top-K operator
=========================================

Selects the "k" largest elements (and their indices) along the "last"
dimension of a 2-D input tensor.  A fast path and a hierarchical path are
chosen automatically based on the row length "M".

Algorithm selection
-------------------
- 1-stage kernel - used when M <= 1024 ("tiny" rows).
  Each row is processed by one Triton launch.
- 2-stage kernel - used when M > 1024 ("large" rows).
  The row is first tiled, each tile computes a local Top-K, and the partial
  results are merged in a second stage.

Interface & constraints
-----------------------
1. Only the last dimension can be reduced.
2. Input must be a 2-D tensor of shape (B, M).
3. Exactly k largest elements are returned.
4. Returned values are **sorted in descending order.

Returns
-------
(values, indices) - both tensors have shape (B, k) and reside on the
same device as the input.

"""


def topk(
    x: torch.Tensor,
    k: int,
    *,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
    tiny_row_thresh: int = MAX_TINY_ROW,
):
    """
    Selects k largest elements along last dimension using 1-stage or 2-stage algorithm.

    Args:
        x (torch.Tensor): Input tensor with shape (B, M). Must be 2D.
        k (int): Number of top elements to select.
        dim (int): Dimension to reduce. Must be -1 (last dimension).
        largest (bool): Select largest elements. Must be True.
        sorted (bool): Return sorted results. Must be True.
        tiny_row_thresh (int): Threshold for choosing 1-stage vs 2-stage algorithm.

    Returns:
        tuple: (values, indices) both with shape (B, k), sorted in descending order.
    """
    if not x.is_contiguous():
        x = x.contiguous()

    row_len = x.shape[-1]
    if row_len <= tiny_row_thresh:
        return one_stage_topk(x, k)
    else:
        return two_stage_topk(x, k, dim=dim, largest=largest)


def triton_op(x, k):
    """Main TopK entry point - streamlined for performance."""
    row_len = x.shape[-1]
    if row_len <= MAX_TINY_ROW:
        return one_stage_topk(x, k)
    return two_stage_topk(x, k)


def torch_op(x, k):
    return torch.topk(x, k, dim=-1, largest=True, sorted=True)

##################################################################################################################################################

# ============================================================================
# TEST CONFIGURATIONS
# ============================================================================

# (B, M, K) -- batch_size, hidden_size, topk
# Extracted from aiter's tests:
#   op_tests/triton_tests/test_topk.py:
#     BATCH_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 16, 1335]
#     DIM2 = [16, 128256]
#     K = [2, 8]
#   op_tests/op_benchmarks/triton/bench_topk.py:
#     BATCH_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 16, 1335]
#     DIM2S = (16, 128, 256, 128256)
#     KS = (2, 8)

ALL_SHAPES = [
    (1, 16, 2), (1, 16, 8), (2, 16, 2), (2, 16, 8), (3, 16, 2), (3, 16, 8),
    (4, 16, 2), (4, 16, 8), (5, 16, 2), (5, 16, 8), (6, 16, 2), (6, 16, 8),
    (7, 16, 2), (7, 16, 8), (1, 128, 2), (1, 128, 8), (8, 16, 2), (8, 16, 8),
    (1, 256, 2), (1, 256, 8), (2, 128, 2), (2, 128, 8), (16, 16, 2), (16, 16, 8),
    (3, 128, 2), (3, 128, 8), (2, 256, 2), (2, 256, 8), (4, 128, 2), (4, 128, 8),
    (5, 128, 2), (5, 128, 8), (3, 256, 2), (3, 256, 8), (6, 128, 2), (6, 128, 8),
    (7, 128, 2), (7, 128, 8), (4, 256, 2), (4, 256, 8), (8, 128, 2), (8, 128, 8),
    (5, 256, 2), (5, 256, 8), (6, 256, 2), (6, 256, 8), (7, 256, 2), (7, 256, 8),
    (8, 256, 2), (8, 256, 8), (16, 128, 2), (16, 128, 8), (16, 256, 2), (16, 256, 8),
    (1335, 16, 2), (1335, 16, 8), (1, 128256, 2), (1, 128256, 8), (1335, 128, 2),
    (1335, 128, 8), (2, 128256, 2), (2, 128256, 8), (1335, 256, 2), (1335, 256, 8),
    (3, 128256, 2), (3, 128256, 8), (4, 128256, 2), (4, 128256, 8), (5, 128256, 2),
    (5, 128256, 8), (6, 128256, 2), (6, 128256, 8), (7, 128256, 2), (7, 128256, 8),
    (8, 128256, 2), (8, 128256, 8), (16, 128256, 2), (16, 128256, 8),
    (1335, 128256, 2), (1335, 128256, 8),
]

# HARNESS_SHAPES: 25 uniformly sampled from ALL_SHAPES
_n_all = len(ALL_SHAPES)
_harness_indices = [int(round(i * (_n_all - 1) / 24)) for i in range(25)]
HARNESS_SHAPES = [ALL_SHAPES[i] for i in _harness_indices]

# PROFILE_SHAPES: 5 evenly-spaced from ALL_SHAPES
_profile_indices = [int(round(i * (_n_all - 1) / 4)) for i in range(5)]
PROFILE_SHAPES = [ALL_SHAPES[i] for i in _profile_indices]

RTOL, ATOL = 1.3e-6, 1e-4

# For backward compatibility
EVAL_CONFIGS = HARNESS_SHAPES
PROFILE_CONFIGS = PROFILE_SHAPES


# ============================================================================
# TEST HARNESS
# ============================================================================


def make_input(batch, hidden, seed=42):
    """Create input tensor on CPU with fixed seed, then move to GPU."""
    torch.manual_seed(seed)
    x_cpu = torch.randn(batch, hidden, dtype=torch.float32)
    return x_cpu.to("cuda")


def reference_topk(x, k, largest=True):
    """Torch reference on CPU."""
    return torch.topk(x.cpu(), k, dim=-1, largest=largest)


def run_correctness(shapes, verbose: bool = True) -> dict:
    if verbose:
        print(f"Running correctness on {len(shapes)} shapes...")

    results, failures = [], []
    for idx, (batch, hidden, k) in enumerate(shapes):
        try:
            x = make_input(batch, hidden, seed=42 + idx)
            ref_val, ref_idx = reference_topk(x, k, largest=True)
            res_val, res_idx = triton_op(x, k)

            res_val_cpu = res_val.cpu()
            res_idx_cpu = res_idx.cpu()

            torch.testing.assert_close(
                res_val_cpu,
                ref_val.to(torch.float32),
                atol=ATOL * hidden,
                rtol=RTOL,
            )
            gathered_res = torch.gather(x.cpu(), 1, res_idx_cpu)
            gathered_ref = torch.gather(x.cpu(), 1, ref_idx)
            torch.testing.assert_close(
                gathered_res,
                gathered_ref.to(torch.float32),
                atol=ATOL * hidden,
                rtol=RTOL,
            )

            results.append({"config": (batch, hidden, k), "correct": True})
            if verbose:
                print(f"  PASS: ({batch}, {hidden}), k={k}")

            del x, res_val, res_idx
            torch.cuda.empty_cache()
        except Exception as e:
            failures.append({"config": (batch, hidden, k), "error": str(e)})
            if verbose:
                print(f"  FAIL: ({batch}, {hidden}), k={k} - {str(e)[:50]}")

    if verbose:
        print("-" * 62)
        print(
            f"{'Status:':<22} {'ALL PASS' if not failures else f'FAILED ({len(failures)}/{len(shapes)})'}"
        )

    return {
        "correct": len(failures) == 0,
        "num_correct": len(results),
        "num_failed": len(failures),
        "failures": failures,
        "results": results,
    }


def run_profile(shapes, warmup: int = 50, iters: int = 200, verbose: bool = True):
    if verbose:
        print(f"Profile: {len(shapes)} config(s), {warmup} warmup, {iters} iter(s)")

    for batch, hidden, k in shapes:
        x = torch.randn(batch, hidden, dtype=torch.float32, device="cpu").to("cuda")
        for _ in range(warmup):
            triton_op(x, k)
        torch.cuda.synchronize()
        for _ in range(iters):
            triton_op(x, k)
        torch.cuda.synchronize()
        if verbose:
            print(f"  ({batch}, {hidden}), k={k} done")
        del x
        torch.cuda.empty_cache()


def run_benchmark(shapes, warmup: int = 50, iters: int = 200, verbose: bool = True) -> dict:
    print(
        f"Running benchmark on {len(shapes)} shapes, {warmup} warmup, {iters} iterations each..."
    )
    latencies = []
    speedups = []
    results = []

    if verbose:
        print(
            f"{'Config (B,M,K)':<22} {'PyTorch':>10} {'Triton':>10} {'Speedup':>10}"
        )
        print("-" * 62)

    for idx, (batch, hidden, k) in enumerate(shapes):
        x = make_input(batch, hidden, seed=42 + idx)

        for _ in range(warmup):
            triton_op(x, k)
        torch.cuda.synchronize()

        triton_times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            triton_op(x, k)
            end.record()
            torch.cuda.synchronize()
            triton_times.append(start.elapsed_time(end))

        triton_ms = sorted(triton_times)[len(triton_times) // 2]

        torch_times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            torch_op(x, k)
            end.record()
            torch.cuda.synchronize()
            torch_times.append(start.elapsed_time(end))

        torch_ms = sorted(torch_times)[len(torch_times) // 2]

        speedup = torch_ms / triton_ms if triton_ms > 0 else 1.0
        speedups.append(speedup)
        latencies.append(triton_ms)

        results.append({
            "config": (batch, hidden, k),
            "torch_ms": torch_ms,
            "triton_ms": triton_ms,
            "speedup": speedup,
        })

        if verbose:
            marker = " *" if speedup > 1.0 else ""
            print(
                f"({batch}, {hidden}), k={k}{' ':4} {torch_ms:>8.4f}ms {triton_ms:>8.4f}ms {speedup:>8.2f}x{marker}"
            )

        del x
        torch.cuda.empty_cache()

    log_sum = sum(math.log(t) for t in latencies)
    geomean_latency = math.exp(log_sum / len(latencies))

    log_sum_speedup = sum(math.log(s) for s in speedups)
    geomean_speedup = math.exp(log_sum_speedup / len(speedups))

    if verbose:
        print("-" * 62)
        print(f"{'Geometric mean latency:':<22} {geomean_latency:.4f} ms")
        print(f"{'Geometric mean speedup:':<22} {geomean_speedup:.2f}x")
        print(f"GEAK_RESULT_LATENCY_MS={geomean_latency:.4f}")
        print(f"GEAK_RESULT_SPEEDUP={geomean_speedup:.2f}")

    return {
        "geomean_latency_ms": geomean_latency,
        "geomean_speedup": geomean_speedup,
        "results": results,
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TopK Kernel Test Harness")
    parser.add_argument(
        "--correctness",
        action="store_true",
        help="Run correctness tests on benchmark shapes",
    )
    parser.add_argument(
        "--profile", action="store_true", help="Run minimal profiling workload"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark on HARNESS_SHAPES (25 uniformly sampled)",
    )
    parser.add_argument(
        "--full-benchmark",
        action="store_true",
        help="Run benchmark on ALL_SHAPES (complete set)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Number of warmup iterations (default: 50)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Number of benchmark iterations (default: 200)",
    )
    args = parser.parse_args()

    print("=" * 62)
    print("TopK Kernel Test Harness")
    print("=" * 62)

    if args.correctness:
        print("\n[Correctness Mode]")
        run_correctness(HARNESS_SHAPES)
    elif args.profile:
        print("\n[Profile Mode]")
        run_profile(PROFILE_SHAPES, warmup=args.warmup, iters=args.iterations)
    elif args.full_benchmark:
        print("\n[Full Benchmark Mode]")
        run_benchmark(ALL_SHAPES, warmup=args.warmup, iters=args.iterations)
    else:
        # Default: benchmark (harness shapes)
        print("\n[Benchmark Mode]")
        run_benchmark(HARNESS_SHAPES, warmup=args.warmup, iters=args.iterations)

    print("=" * 62)
