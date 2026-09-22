"""Additional unscored public-branch controls from PR105.

Source revision: 0acf65b3a967ef1025dbfc5fd4b415b259e3bd43
Source path: tasks/triton2triton/vllm/triton_compute_slot_mappings/scripts/task_runner.py
Original scored inputs, gates and timing remain in scripts/task_runner.py.
"""
import sys
import os
import json
import argparse
import importlib.util
ADDITIONAL_CORRECTNESS_CASES = [{'name': 'uneven_empty_permuted_int64', 'query_lengths': [0, 1, 9, 17, 3], 'idx_mapping': [6, 0, 4, 2, 7], 'position_starts': [0, 7, 31, 4093, 1024], 'block_size': 8, 'max_num_reqs': 8, 'max_num_blocks': 520, 'block_table_row_padding': 5, 'max_num_tokens': 47, 'index_dtype': 'int64', 'block_table_dtype': 'int64'}, {'name': 'multi_tile_data_and_padding', 'query_lengths': [1025, 0, 6], 'idx_mapping': [2, 5, 1], 'position_starts': [8190, 0, 65533], 'block_size': 64, 'max_num_reqs': 6, 'max_num_blocks': 1025, 'block_table_row_padding': 7, 'max_num_tokens': 3082, 'index_dtype': 'int32', 'block_table_dtype': 'int32'}, {'name': 'all_empty_multi_tile_padding', 'query_lengths': [0, 0, 0], 'idx_mapping': [2, 0, 1], 'position_starts': [0, 0, 0], 'block_size': 8, 'max_num_reqs': 3, 'max_num_blocks': 1, 'block_table_row_padding': 3, 'max_num_tokens': 2053, 'index_dtype': 'int32', 'block_table_dtype': 'int32'}]

def reference_compute_slot_mappings(idx_mapping, query_start_loc, positions, block_table, block_size):
    import torch
    num_reqs = idx_mapping.shape[0]
    num_tokens = positions.shape[0]
    slot_mappings = torch.full((num_tokens,), -1, dtype=torch.int64)
    for b in range(num_reqs):
        req_idx = idx_mapping[b].item()
        start = query_start_loc[b].item()
        end = query_start_loc[b + 1].item()
        for t in range(start, end):
            p = positions[t].item()
            block_idx = p // block_size
            block_off = p % block_size
            block_num = block_table[req_idx, block_idx].item()
            slot_mappings[t] = block_num * block_size + block_off
    return slot_mappings

def validate_slot_mappings(result, ref, case_name):
    """Validate the token prefix returned by compute_slot_mappings."""
    import torch
    if result.shape != ref.shape:
        return f'{case_name}: output shape {tuple(result.shape)} expected {tuple(ref.shape)}'
    if result.dtype != torch.int64:
        return f'{case_name}: output dtype {result.dtype} expected torch.int64'
    result_cpu = result.cpu()
    if not torch.equal(result_cpu, ref):
        diff_mask = result_cpu != ref
        first_diff = diff_mask.nonzero(as_tuple=True)[0][0].item()
        return f'{case_name}: mismatch at index {first_diff}, got {result_cpu[first_diff].item()} expected {ref[first_diff].item()}'
    return None

def validate_padding(mod, idx_mapping, query_start_loc, positions, block_table, block_size, max_num_tokens, case_name):
    """Launch into a poisoned full buffer and directly validate all padding."""
    import torch
    num_tokens = positions.shape[0]
    padding_poison = 123456789
    full_result = torch.full((max_num_tokens,), padding_poison, dtype=torch.int64, device=positions.device)
    mod._compute_slot_mappings_kernel[idx_mapping.shape[0] + 1,](num_tokens, max_num_tokens, idx_mapping, query_start_loc, positions, block_table, block_table.stride(0), block_size, full_result, PAD_ID=-1, TRITON_BLOCK_SIZE=1024)
    padding = full_result[num_tokens:].cpu()
    if not torch.all(padding == -1).item():
        first_diff = (padding != -1).nonzero(as_tuple=True)[0][0].item()
        return f'{case_name}: padding mismatch at index {num_tokens + first_diff}, got {padding[first_diff].item()} expected -1'
    return None
EXTRA_CASES = [{'name': 'uneven_empty_permuted_int64', 'query_lengths': [0, 1, 9, 17, 3], 'idx_mapping': [6, 0, 4, 2, 7], 'position_starts': [0, 7, 31, 4093, 1024], 'block_size': 8, 'max_num_reqs': 8, 'max_num_blocks': 520, 'block_table_row_padding': 5, 'max_num_tokens': 47, 'index_dtype': 'int64', 'block_table_dtype': 'int64'}, {'name': 'multi_tile_data_and_padding', 'query_lengths': [1025, 0, 6], 'idx_mapping': [2, 5, 1], 'position_starts': [8190, 0, 65533], 'block_size': 64, 'max_num_reqs': 6, 'max_num_blocks': 1025, 'block_table_row_padding': 7, 'max_num_tokens': 3082, 'index_dtype': 'int32', 'block_table_dtype': 'int32'}, {'name': 'all_empty_multi_tile_padding', 'query_lengths': [0, 0, 0], 'idx_mapping': [2, 0, 1], 'position_starts': [0, 0, 0], 'block_size': 8, 'max_num_reqs': 3, 'max_num_blocks': 1, 'block_table_row_padding': 3, 'max_num_tokens': 2053, 'index_dtype': 'int32', 'block_table_dtype': 'int32'}]

def run_control(index, load_module):
    if type(index) is not int or not 0 <= index < len(EXTRA_CASES):
        raise ValueError('Unknown upstream correctness control')
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return (False, f'Failed to load module: {e}')
    device = 'cuda'
    for i, case in enumerate(EXTRA_CASES, start=0):
        if i != index + 0:
            continue
        case_name = case['name']
        try:
            torch.manual_seed(1042 + i)
            query_lengths = case['query_lengths']
            query_start_values = [0]
            for query_len in query_lengths:
                query_start_values.append(query_start_values[-1] + query_len)
            index_dtype = getattr(torch, case['index_dtype'])
            idx_mapping = torch.tensor(case['idx_mapping'], dtype=index_dtype, device=device)
            query_start_loc = torch.tensor(query_start_values, dtype=index_dtype, device=device)
            position_values = []
            for position_start, query_len in zip(case['position_starts'], query_lengths):
                position_values.extend(range(position_start, position_start + query_len))
            positions = torch.tensor(position_values, dtype=torch.int64, device=device)
            table_storage = torch.randint(0, 10000, (case['max_num_reqs'], case['max_num_blocks'] + case['block_table_row_padding']), dtype=getattr(torch, case['block_table_dtype']), device=device)
            block_table = table_storage[:, :case['max_num_blocks']]
            result = mod.compute_slot_mappings(idx_mapping, query_start_loc, positions, block_table, case['block_size'], case['max_num_tokens'])
            torch.cuda.synchronize()
            ref = reference_compute_slot_mappings(idx_mapping.cpu(), query_start_loc.cpu(), positions.cpu(), block_table.cpu(), case['block_size'])
            error = validate_slot_mappings(result, ref, case_name)
            if error:
                return (False, error)
            error = validate_padding(mod, idx_mapping, query_start_loc, positions, block_table, case['block_size'], case['max_num_tokens'], case_name)
            if error:
                return (False, error)
        except Exception as e:
            return (False, f'{case_name}: exception: {e}')
    return (True, None)
