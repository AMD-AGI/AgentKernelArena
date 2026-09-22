"""Additional unscored public-branch controls from PR105.

Source revision: 0acf65b3a967ef1025dbfc5fd4b415b259e3bd43
Source path: tasks/triton2triton/vllm/triton_chunked_prefill_paged_decode/scripts/task_runner.py
Original scored inputs, gates and timing remain in scripts/task_runner.py.
"""
import sys
import os
import json
import argparse
import importlib.util
EXTRA_CORRECTNESS_CASES = [{'name': 'unaligned_shuffled_strided_x4_head80', 'seq_lens': [1, 17, 31], 'query_lens': [1, 1, 1], 'num_query_heads': 12, 'num_kv_heads': 3, 'head_size': 80, 'block_size': 16, 'x_factor': 4, 'page_mapping': 'shuffled', 'strided_layout': True, 'filter_by_query_len': False}, {'name': 'mixed_query_filter_sliding_x16_head96', 'seq_lens': [33, 70, 47, 95], 'query_lens': [1, 3, 1, 2], 'num_query_heads': 8, 'num_kv_heads': 2, 'head_size': 96, 'block_size': 32, 'x_factor': 16, 'page_mapping': 'reversed', 'strided_layout': False, 'filter_by_query_len': True, 'sliding_window': 17}, {'name': 'unaligned_alibi_mha', 'seq_lens': [23, 49], 'query_lens': [1, 1], 'num_query_heads': 4, 'num_kv_heads': 4, 'head_size': 64, 'block_size': 16, 'x_factor': 8, 'page_mapping': 'shuffled', 'strided_layout': False, 'filter_by_query_len': False, 'use_alibi': True}]

def make_extended_test_data(case, device='cuda', dtype=None):
    """Build feature-focused inputs, including non-contiguous padded views."""
    import torch
    if dtype is None:
        dtype = torch.float16
    seq_lens_list = case['seq_lens']
    query_lens_list = case['query_lens']
    num_seqs = len(seq_lens_list)
    num_query_heads = case['num_query_heads']
    num_kv_heads = case['num_kv_heads']
    head_size = case['head_size']
    block_size = case['block_size']
    x_factor = case['x_factor']
    strided_layout = case['strided_layout']
    assert len(query_lens_list) == num_seqs
    assert num_query_heads % num_kv_heads == 0
    assert head_size % x_factor == 0
    if not case['filter_by_query_len']:
        assert all((length == 1 for length in query_lens_list))
    total_tokens = sum(query_lens_list)
    if strided_layout:
        query_storage = torch.randn(total_tokens, num_query_heads, head_size + 3, device=device, dtype=dtype)
        query = query_storage[..., :head_size]
        output_storage = torch.full((total_tokens, num_query_heads, head_size + 5), -7.0, device=device, dtype=dtype)
        output = output_storage[..., :head_size]
    else:
        query = torch.randn(total_tokens, num_query_heads, head_size, device=device, dtype=dtype)
        output = torch.full_like(query, -7.0)
    blocks_per_seq = [(length + block_size - 1) // block_size for length in seq_lens_list]
    logical_block_count = sum(blocks_per_seq)
    total_blocks = logical_block_count + 4
    if strided_layout:
        key_storage = torch.randn(total_blocks, num_kv_heads, head_size // x_factor, block_size, x_factor + 3, device=device, dtype=dtype)
        key_cache = key_storage[..., :x_factor]
        value_storage = torch.randn(total_blocks, num_kv_heads, head_size, block_size + 3, device=device, dtype=dtype)
        value_cache = value_storage[..., :block_size]
        assert not query.is_contiguous()
        assert not output.is_contiguous()
        assert not key_cache.is_contiguous()
        assert not value_cache.is_contiguous()
    else:
        key_cache = torch.randn(total_blocks, num_kv_heads, head_size // x_factor, block_size, x_factor, device=device, dtype=dtype)
        value_cache = torch.randn(total_blocks, num_kv_heads, head_size, block_size, device=device, dtype=dtype)
    max_blocks_per_seq = max(blocks_per_seq)
    table_storage = torch.zeros(num_seqs, max_blocks_per_seq + 2, device=device, dtype=torch.int32)
    block_table = table_storage[:, :max_blocks_per_seq]
    if case['page_mapping'] == 'shuffled':
        physical_blocks = torch.randperm(total_blocks, device=device)
    elif case['page_mapping'] == 'reversed':
        physical_blocks = torch.arange(total_blocks - 1, -1, -1, device=device)
    else:
        raise ValueError(f'Unknown page mapping: {case['page_mapping']}')
    block_offset = 0
    for seq_idx, block_count in enumerate(blocks_per_seq):
        block_table[seq_idx, :block_count] = physical_blocks[block_offset:block_offset + block_count]
        block_offset += block_count
    seq_lens = torch.tensor(seq_lens_list, device=device, dtype=torch.int32)
    query_lens = torch.tensor(query_lens_list, device=device, dtype=torch.int32)
    query_start_loc = torch.cat((torch.zeros(1, device=device, dtype=torch.int32), query_lens.cumsum(dim=0)))
    scale = 1.0 / head_size ** 0.5
    alibi_slopes = None
    if case.get('use_alibi', False):
        alibi_slopes = torch.linspace(0.02, 0.16, num_query_heads, device=device, dtype=torch.float32)
    return (query, output, key_cache, value_cache, block_table, seq_lens, query_start_loc, scale, alibi_slopes)

def reference_extended_attention(query, initial_output, key_cache, value_cache, block_table, seq_lens, query_start_loc, scale, block_size, x_factor, filter_by_query_len, sliding_window=0, alibi_slopes=None):
    """Reference for filtered decode, sliding-window, and ALiBi modes."""
    import torch
    num_seqs = len(seq_lens)
    num_query_heads = query.shape[1]
    head_size = query.shape[2]
    num_kv_heads = key_cache.shape[1]
    num_queries_per_kv = num_query_heads // num_kv_heads
    assert key_cache.shape[2] * x_factor == head_size
    output = initial_output.float().clone()
    for seq_idx in range(num_seqs):
        query_start = int(query_start_loc[seq_idx].item())
        query_stop = int(query_start_loc[seq_idx + 1].item())
        if filter_by_query_len and query_stop - query_start > 1:
            continue
        query_idx = query_start if filter_by_query_len else seq_idx
        seq_len = int(seq_lens[seq_idx].item())
        token_offsets = torch.arange(seq_len, device=query.device)
        logical_blocks = token_offsets // block_size
        physical_blocks = block_table[seq_idx, logical_blocks].long()
        block_offsets = token_offsets % block_size
        gathered_k = key_cache[physical_blocks, :, :, block_offsets, :].reshape(seq_len, num_kv_heads, head_size)
        gathered_v = value_cache[physical_blocks, :, :, block_offsets]
        for query_head in range(num_query_heads):
            kv_head = query_head // num_queries_per_kv
            scores = query[query_idx, query_head].float() @ gathered_k[:, kv_head].float().T * scale
            context_len = seq_len - 1
            if sliding_window > 0:
                scores = scores.masked_fill(context_len - token_offsets >= sliding_window, float('-inf'))
            if alibi_slopes is not None:
                scores += alibi_slopes[query_head] * (token_offsets - context_len)
            probabilities = torch.softmax(scores, dim=0)
            output[query_idx, query_head] = probabilities @ gathered_v[:, kv_head].float()
    return output.to(query.dtype)
EXTRA_CASES = [{'name': 'unaligned_shuffled_strided_x4_head80', 'seq_lens': [1, 17, 31], 'query_lens': [1, 1, 1], 'num_query_heads': 12, 'num_kv_heads': 3, 'head_size': 80, 'block_size': 16, 'x_factor': 4, 'page_mapping': 'shuffled', 'strided_layout': True, 'filter_by_query_len': False}, {'name': 'mixed_query_filter_sliding_x16_head96', 'seq_lens': [33, 70, 47, 95], 'query_lens': [1, 3, 1, 2], 'num_query_heads': 8, 'num_kv_heads': 2, 'head_size': 96, 'block_size': 32, 'x_factor': 16, 'page_mapping': 'reversed', 'strided_layout': False, 'filter_by_query_len': True, 'sliding_window': 17}, {'name': 'unaligned_alibi_mha', 'seq_lens': [23, 49], 'query_lens': [1, 1], 'num_query_heads': 4, 'num_kv_heads': 4, 'head_size': 64, 'block_size': 16, 'x_factor': 8, 'page_mapping': 'shuffled', 'strided_layout': False, 'filter_by_query_len': False, 'use_alibi': True}]

def run_control(index, load_module):
    if type(index) is not int or not 0 <= index < len(EXTRA_CASES):
        raise ValueError('Unknown upstream correctness control')
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return (False, f'Failed to load module: {e}')
    device = 'cuda'
    dtype = torch.float16
    for i, case in enumerate(EXTRA_CASES, start=0):
        if i != index + 0:
            continue
        case_name = case['name']
        try:
            torch.manual_seed(142 + i)
            query, output, key_cache, value_cache, block_table, seq_lens, qsl, scale, alibi_slopes = make_extended_test_data(case, device, dtype)
            initial_output = output.clone()
            mod.chunked_prefill_paged_decode(query, output, key_cache, value_cache, block_table, seq_lens, qsl, scale, alibi_slopes=alibi_slopes, sliding_window=case.get('sliding_window', 0), filter_by_query_len=case['filter_by_query_len'])
            torch.cuda.synchronize()
            ref = reference_extended_attention(query, initial_output, key_cache, value_cache, block_table, seq_lens, qsl, scale, case['block_size'], case['x_factor'], case['filter_by_query_len'], sliding_window=case.get('sliding_window', 0), alibi_slopes=alibi_slopes)
            if not torch.allclose(output.float(), ref.float(), atol=0.01, rtol=0.01):
                max_diff = (output.float() - ref.float()).abs().max().item()
                return (False, f'Case {case_name}: max diff = {max_diff:.6f}')
        except Exception as e:
            return (False, f'Case {case_name}: exception: {e}')
    return (True, None)
