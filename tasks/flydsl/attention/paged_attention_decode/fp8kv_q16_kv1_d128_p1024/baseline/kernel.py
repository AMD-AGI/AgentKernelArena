# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""FlyDSL paged-attention decode with persistent scheduling, FP8 KV cache.

The entry point of this baseline. run() is the block-1024 persistent-worklist branch of pa_decode_ps_launch
(ROCm/FlyDSL kernels/attention/pa_decode_fp8.py, tag v0.3.2): one launch of the persistent decode kernel of
pa_metadata.py (compile_pa_decode_metadata) that walks the worklist, then the deterministic FlyDSL reduce
(pa_metadata_reduce) that merges the split partials. The worklist (work_*, reduce_*, partition_indptr) is built
once per decode step in production by get_pa_metadata, so it is an input of the task; see ../README.md.
"""

import torch

from .pa_metadata import compile_pa_decode_metadata, compile_pa_metadata_reduce


def run(
    *,
    query,
    key_cache,
    value_cache,
    key_scale,
    value_scale,
    context_lengths,
    kv_indptr,
    kv_page_indices,
    partition_indptr,
    work_indptr,
    work_info,
    reduce_indptr,
    reduce_final_map,
    reduce_partial_map,
    softmax_scale,
    out,
):
    num_query_heads = query.shape[1]
    num_kv_heads = key_cache.shape[1]
    head_size = query.shape[-1]
    query_length = query.shape[0] // context_lengths.shape[0]
    query_group_size = num_query_heads // num_kv_heads
    trans_v = len(value_cache.shape) == 5
    per_token_kv = key_scale.ndim > 1
    num_sm = work_indptr.shape[0] - 1
    num_partials = reduce_partial_map.shape[0]

    # Partial buffers sized for the largest worklist: the first query_length rows are reserved by the kernel.
    partial_output = torch.empty(
        ((num_partials + 1) * query_length, 1, num_query_heads, head_size), dtype=torch.float32, device=query.device
    )
    partial_lse = torch.empty(
        ((num_partials + 1) * query_length, 1, num_query_heads, 1), dtype=torch.float32, device=query.device
    )
    stride_po_partial = query_length * num_query_heads * head_size
    stride_pl_partial = query_length * num_query_heads
    stride_po_ql = num_query_heads * head_size
    stride_pl_ql = num_query_heads
    if per_token_kv:
        stride_ks_block = key_scale.stride(0)
        stride_ks_head = key_scale.stride(1)
    else:
        stride_ks_block = 0
        stride_ks_head = 0

    stream = torch.cuda.current_stream()
    compiled = compile_pa_decode_metadata(
        softmax_scale=float(softmax_scale),
        trans_v=trans_v,
        query_group_size=query_group_size,
        per_token_kv=per_token_kv,
        query_length=query_length,
        query_input_dtype="bf16",
        head_dim=int(head_size),
        block_size=int(key_cache.shape[-2]),
        output_dtype_str="bf16",
    )
    compiled["launch"](
        out.data_ptr(),
        partial_output.data_ptr(),
        partial_lse.data_ptr(),
        query.data_ptr(),
        key_cache.data_ptr(),
        value_cache.data_ptr(),
        context_lengths.data_ptr(),
        key_scale.data_ptr(),
        value_scale.data_ptr(),
        work_indptr.data_ptr(),
        work_info.data_ptr(),
        kv_page_indices.data_ptr(),
        kv_indptr.data_ptr(),
        partition_indptr.data_ptr(),
        query.stride(0),
        query.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        value_cache.stride(0),
        value_cache.stride(1),
        out.stride(0),
        out.stride(1),
        stride_po_partial,
        stride_pl_partial,
        stride_ks_block,
        stride_ks_head,
        stride_po_ql,
        stride_pl_ql,
        num_sm,
        stream,
    )

    reduce_compiled = compile_pa_metadata_reduce(
        query_length=int(query_length),
        num_query_heads=int(num_query_heads),
        head_dim=int(head_size),
        output_dtype_str="bf16",
    )
    reduce_compiled["launch"](
        out,
        partial_output[query_length:],
        partial_lse[query_length:],
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        out.stride(0),
        out.stride(1),
        num_query_heads * head_size,
        num_query_heads,
        reduce_indptr.numel() - 1,
        stream,
    )
