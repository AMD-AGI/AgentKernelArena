"""Additional unscored public-branch controls from PR105.

Source revision: 0acf65b3a967ef1025dbfc5fd4b415b259e3bd43
Source path: tasks/triton2triton/vllm/triton_prepare_mrope_positions/scripts/task_runner.py
Original scored inputs, gates and timing remain in scripts/task_runner.py.
"""
import sys
import os
import json
import argparse
import importlib.util
TARGETED_CORRECTNESS_CASES = [{'name': 'mixed_mapped_boundary_lengths', 'max_model_len': 64, 'idx_mapping': [6, 2, 7, 0, 5, 3], 'query_lens': [0, 1, 3, 1, 0, 7], 'prefill_lens': [5, 9, 12, 1, 4, 11], 'num_computed_tokens': [0, 8, 12, 1, 4, 4]}, {'name': 'mixed_multi_tile_requests', 'max_model_len': 4096, 'idx_mapping': [3, 0], 'query_lens': [1025, 2051], 'prefill_lens': [4096, 2045], 'num_computed_tokens': [3071, 2045]}]

def reference_prepare_mrope(mrope_positions, prefill_mrope_positions, max_model_len, prefill_mrope_delta, idx_mapping, query_start_loc, prefill_lens, num_computed_tokens):
    import torch
    mrope_positions = mrope_positions.clone()
    num_reqs = idx_mapping.shape[0]
    for b in range(num_reqs):
        req_state_idx = idx_mapping[b].item()
        prefill_len = prefill_lens[req_state_idx].item()
        num_computed = num_computed_tokens[req_state_idx].item()
        is_prefill = num_computed < prefill_len
        qstart = query_start_loc[b].item()
        qend = query_start_loc[b + 1].item()
        qlen = qend - qstart
        delta = prefill_mrope_delta[req_state_idx].item()
        for k in range(qlen):
            orig_pos = num_computed + k
            for j in range(3):
                if is_prefill:
                    pos = prefill_mrope_positions[req_state_idx * 3 + j, orig_pos].item()
                else:
                    pos = orig_pos + delta
                mrope_positions[j, qstart + k] = pos
    return mrope_positions

def check_mrope_case(mod, case_name, mrope_positions, prefill_mrope_positions, max_model_len, prefill_mrope_delta, idx_mapping, query_start_loc, prefill_lens, num_computed_tokens):
    import torch
    ref = reference_prepare_mrope(mrope_positions.cpu(), prefill_mrope_positions.cpu(), max_model_len, prefill_mrope_delta.cpu(), idx_mapping.cpu(), query_start_loc.cpu(), prefill_lens.cpu(), num_computed_tokens.cpu())
    mod.prepare_mrope_positions(mrope_positions, prefill_mrope_positions, max_model_len, prefill_mrope_delta, idx_mapping, query_start_loc, prefill_lens, num_computed_tokens)
    torch.cuda.synchronize()
    actual = mrope_positions.cpu()
    if torch.equal(actual, ref):
        return (True, None)
    first_diff = (actual != ref).nonzero()[0]
    dim = first_diff[0].item()
    token = first_diff[1].item()
    return (False, f'{case_name}: mismatch at [{dim},{token}] got {actual[dim, token].item()} expected {ref[dim, token].item()}')

def run_targeted_correctness_case(mod, case, seed):
    import torch
    device = 'cuda'
    idx_mapping_values = case['idx_mapping']
    query_lens = case['query_lens']
    prefill_lens_by_batch = case['prefill_lens']
    num_computed_by_batch = case['num_computed_tokens']
    max_model_len = case['max_model_len']
    num_reqs = len(idx_mapping_values)
    if not len(query_lens) == len(prefill_lens_by_batch) == len(num_computed_by_batch) == num_reqs:
        raise ValueError('targeted case arrays must have one entry per request')
    if len(set(idx_mapping_values)) != num_reqs:
        raise ValueError('targeted cases require unique request-state mappings')
    max_num_reqs = max(num_reqs + 8, max(idx_mapping_values) + 1)
    idx_mapping = torch.tensor(idx_mapping_values, dtype=torch.int32, device=device)
    query_starts = [0]
    for query_len in query_lens:
        if query_len < 0:
            raise ValueError('query lengths must be nonnegative')
        query_starts.append(query_starts[-1] + query_len)
    query_start_loc = torch.tensor(query_starts, dtype=torch.int32, device=device)
    prefill_lens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
    num_computed_tokens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
    for batch_idx, req_state_idx in enumerate(idx_mapping_values):
        prefill_len = prefill_lens_by_batch[batch_idx]
        num_computed = num_computed_by_batch[batch_idx]
        query_len = query_lens[batch_idx]
        if not 0 <= prefill_len <= max_model_len:
            raise ValueError('prefill length is outside max_model_len')
        if not 0 <= num_computed <= max_model_len:
            raise ValueError('num_computed is outside max_model_len')
        if num_computed + query_len > max_model_len:
            raise ValueError('request tokens exceed max_model_len')
        if num_computed < prefill_len and num_computed + query_len > prefill_len:
            raise ValueError('prefill query extends beyond its prefill length')
        prefill_lens[req_state_idx] = prefill_len
        num_computed_tokens[req_state_idx] = num_computed
    torch.manual_seed(seed)
    prefill_mrope_positions = torch.randint(0, max_model_len, (max_num_reqs * 3, max_model_len), dtype=torch.int32, device=device)
    prefill_mrope_delta = torch.randint(-10, 10, (max_num_reqs,), dtype=torch.int32, device=device)
    mrope_positions = torch.full((3, query_starts[-1] + 1), -(1 << 40), dtype=torch.int64, device=device)
    return check_mrope_case(mod, case['name'], mrope_positions, prefill_mrope_positions, max_model_len, prefill_mrope_delta, idx_mapping, query_start_loc, prefill_lens, num_computed_tokens)
EXTRA_CASES = [{'name': 'mixed_mapped_boundary_lengths', 'max_model_len': 64, 'idx_mapping': [6, 2, 7, 0, 5, 3], 'query_lens': [0, 1, 3, 1, 0, 7], 'prefill_lens': [5, 9, 12, 1, 4, 11], 'num_computed_tokens': [0, 8, 12, 1, 4, 4]}, {'name': 'mixed_multi_tile_requests', 'max_model_len': 4096, 'idx_mapping': [3, 0], 'query_lens': [1025, 2051], 'prefill_lens': [4096, 2045], 'num_computed_tokens': [3071, 2045]}]

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
        try:
            ok, err = run_targeted_correctness_case(mod, case, 100 + i)
            if not ok:
                return (False, err)
        except Exception as e:
            return (False, f'Case {case['name']}: exception: {e}')
    return (True, None)
