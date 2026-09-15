"""Additional unscored public-branch controls from PR105.

Source revision: 0acf65b3a967ef1025dbfc5fd4b415b259e3bd43
Source path: tasks/triton2triton/vllm/triton_prepare_prefill_inputs/scripts/task_runner.py
Original scored inputs, gates and timing remain in scripts/task_runner.py.
"""
import sys
import os
import json
import argparse
import importlib.util
CORRECTNESS_CASES = [{'name': 'variable_lengths_sparse_mapping', 'max_num_reqs': 16, 'max_seq_len': 512, 'idx_mapping': [9, 2, 14, 5, 11, 0], 'query_lens': [0, 1, 7, 64, 255, 257], 'prefill_lens': [6, 14, 37, 95, 297, 310], 'num_computed_tokens': [5, 13, 29, 31, 41, 53]}, {'name': 'completed_prefill_early_return', 'max_num_reqs': 12, 'max_seq_len': 64, 'idx_mapping': [8, 1, 10, 4], 'query_lens': [3, 0, 5, 2], 'prefill_lens': [12, 15, 19, 12], 'num_computed_tokens': [12, 16, 20, 9]}]

def reference_prepare_prefill_inputs(idx_mapping, query_start_loc, all_token_ids, prefill_len, num_computed_tokens):
    """CPU reference implementation."""
    import torch
    num_reqs = idx_mapping.shape[0]
    total_tokens = int(query_start_loc[-1].item())
    input_ids = torch.zeros(total_tokens, dtype=torch.int32, device='cpu')
    next_prefill_tokens = torch.zeros(all_token_ids.shape[0], dtype=torch.int32, device='cpu')
    for b in range(num_reqs):
        req_state_idx = idx_mapping[b].item()
        plen = prefill_len[req_state_idx].item()
        num_computed = num_computed_tokens[req_state_idx].item()
        if num_computed >= plen:
            continue
        qstart = query_start_loc[b].item()
        qend = query_start_loc[b + 1].item()
        qlen = qend - qstart
        for k in range(qlen):
            input_ids[qstart + k] = all_token_ids[req_state_idx, num_computed + k]
        next_pos = num_computed + qlen
        if next_pos < plen:
            next_prefill_tokens[req_state_idx] = all_token_ids[req_state_idx, next_pos]
    return (input_ids, next_prefill_tokens)
EXTRA_CASES = [{'name': 'variable_lengths_sparse_mapping', 'max_num_reqs': 16, 'max_seq_len': 512, 'idx_mapping': [9, 2, 14, 5, 11, 0], 'query_lens': [0, 1, 7, 64, 255, 257], 'prefill_lens': [6, 14, 37, 95, 297, 310], 'num_computed_tokens': [5, 13, 29, 31, 41, 53]}, {'name': 'completed_prefill_early_return', 'max_num_reqs': 12, 'max_seq_len': 64, 'idx_mapping': [8, 1, 10, 4], 'query_lens': [3, 0, 5, 2], 'prefill_lens': [12, 15, 19, 12], 'num_computed_tokens': [12, 16, 20, 9]}]

def run_control(index, load_module):
    if type(index) is not int or not 0 <= index < len(EXTRA_CASES):
        raise ValueError('Unknown upstream correctness control')
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return (False, f'Failed to load module: {e}')
    device = 'cuda'
    for _control_index, case in enumerate(EXTRA_CASES, start=0):
        if _control_index != index + 0:
            continue
        name = case['name']
        try:
            max_num_reqs = case['max_num_reqs']
            max_seq_len = case['max_seq_len']
            idx_mapping = torch.tensor(case['idx_mapping'], dtype=torch.int32, device=device)
            query_lens = torch.tensor(case['query_lens'], dtype=torch.int32, device=device)
            query_start_loc = torch.zeros(len(case['query_lens']) + 1, dtype=torch.int32, device=device)
            query_start_loc[1:] = torch.cumsum(query_lens, dim=0)
            all_token_ids = torch.arange(max_num_reqs * max_seq_len, dtype=torch.int32, device=device).reshape(max_num_reqs, max_seq_len) % 31999 + 1
            prefill_len = torch.full((max_num_reqs,), max_seq_len, dtype=torch.int32, device=device)
            num_computed_tokens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
            prefill_len[idx_mapping.long()] = torch.tensor(case['prefill_lens'], dtype=torch.int32, device=device)
            num_computed_tokens[idx_mapping.long()] = torch.tensor(case['num_computed_tokens'], dtype=torch.int32, device=device)
            total_tokens = int(query_start_loc[-1].item())
            input_ids = torch.zeros(total_tokens, dtype=torch.int32, device=device)
            next_prefill_tokens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
            mod.prepare_prefill_inputs(input_ids, next_prefill_tokens, idx_mapping, query_start_loc, all_token_ids, prefill_len, num_computed_tokens)
            torch.cuda.synchronize()
            ref_ids, ref_next = reference_prepare_prefill_inputs(idx_mapping.cpu(), query_start_loc.cpu(), all_token_ids.cpu(), prefill_len.cpu(), num_computed_tokens.cpu())
            if not torch.equal(input_ids.cpu(), ref_ids):
                return (False, f'Case {name}: input_ids mismatch')
            if not torch.equal(next_prefill_tokens.cpu(), ref_next):
                return (False, f'Case {name}: next_prefill_tokens mismatch')
        except Exception as e:
            return (False, f'Case {name}: exception: {e}')
    return (True, None)
