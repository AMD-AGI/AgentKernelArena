# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

# Provenance and modifications are recorded in task.yaml.
# The kernel in this file is adapted from FlagGems' topk:
# https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/ops/topk.py

#  Top-K on GPU:  1-stage (tiny rows) + 2-stage (large rows) Triton kernels,
from __future__ import annotations
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

# Rows up to this length use the one-stage kernel (8192 in the earlier arena task; the vocabulary row here
# is longer, so the two-stage path is always taken).
MAX_TINY_ROW = 8192


def one_stage_topk(x, k, out_v, out_i):
    B, M = x.shape
    BLOCK = _BLOCK_TABLE.get(max(M, k), 8192)

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


def two_stage_topk(x, k, stage2_out, stage2_out_idx, dim=-1, largest=True):
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


def run(*, x, k, out_values, out_indices):
    if x.shape[-1] <= MAX_TINY_ROW:
        one_stage_topk(x, k, out_values, out_indices)
    else:
        two_stage_topk(x, k, out_values, out_indices)
