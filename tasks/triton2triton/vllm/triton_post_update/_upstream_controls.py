"""Unscored PR105 control inputs; original performance remains unchanged.
Source revision: 0acf65b3a967ef1025dbfc5fd4b415b259e3bd43
Source path: tasks/triton2triton/vllm/triton_post_update/scripts/task_runner.py
"""
import sys
import os
import json
import argparse
import importlib.util

def reference_post_update(idx_mapping, num_computed_tokens, last_sampled_tokens, output_bin_counts, sampled_tokens, num_sampled, num_rejected, query_start_loc, all_token_ids, total_len):
    import torch
    num_reqs = idx_mapping.shape[0]
    num_computed_tokens = num_computed_tokens.clone()
    last_sampled_tokens = last_sampled_tokens.clone()
    output_bin_counts = output_bin_counts.clone()
    all_token_ids = all_token_ids.clone()
    total_len = total_len.clone()
    for r in range(num_reqs):
        req_state_idx = idx_mapping[r].item()
        tlen = total_len[req_state_idx].item()
        ns = num_sampled[r].item()
        if ns > 0:
            tok = sampled_tokens[r, ns - 1].item()
            last_sampled_tokens[req_state_idx] = tok
            total_len[req_state_idx] = tlen + ns
        for j in range(ns):
            tok = sampled_tokens[r, j].item()
            output_bin_counts[req_state_idx, tok] += 1
            all_token_ids[req_state_idx, tlen + j] = tok
        qstart = query_start_loc[r].item()
        qend = query_start_loc[r + 1].item()
        qlen = qend - qstart
        nr = num_rejected[r].item()
        nc = num_computed_tokens[req_state_idx].item()
        num_computed_tokens[req_state_idx] = nc + qlen - nr
    return (num_computed_tokens, last_sampled_tokens, output_bin_counts, all_token_ids, total_len)
EXTRA_CASES = [{'name': 'multi_program_tail', 'requests': 70, 'max_requests': 149, 'vocab': 257, 'max_model_len': 96, 'max_sampled': 5, 'mapping': '(request*43+17)%149'}]

def run_control(index, load_module):
    if type(index) is not int or not 0 <= index < len(EXTRA_CASES):
        raise ValueError('Unknown upstream correctness control')
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return (False, f'Failed to load module: {e}')
    device = 'cuda'
    case_label = 'Targeted multi-program tail'
    if index == 0:
        try:
            num_reqs = 70
            max_num_reqs = 149
            vocab_size = 257
            max_model_len = 96
            max_num_sampled = 5
            query_len = max_num_sampled
            req_ids = torch.arange(num_reqs, dtype=torch.int32, device=device)
            idx_mapping = (req_ids * 43 + 17) % max_num_reqs
            query_start_loc = torch.arange(num_reqs + 1, dtype=torch.int32, device=device) * query_len
            state_ids = torch.arange(max_num_reqs, dtype=torch.int32, device=device)
            num_computed_tokens = 10 + state_ids % 40
            last_sampled_tokens = (state_ids * 11 + 3) % vocab_size
            output_bin_counts = torch.full((max_num_reqs, vocab_size), 3, dtype=torch.int32, device=device)
            repeated_tokens = ((req_ids * 13 + 5) % vocab_size).unsqueeze(1)
            sampled_tokens = repeated_tokens.expand(num_reqs, max_num_sampled).clone()
            num_sampled = req_ids % (max_num_sampled + 1)
            num_rejected = query_len - num_sampled
            all_token_ids = torch.full((max_num_reqs, max_model_len), -1, dtype=torch.int32, device=device)
            total_len = 8 + state_ids % 17
            nct_g = num_computed_tokens.clone()
            lst_g = last_sampled_tokens.clone()
            obc_g = output_bin_counts.clone()
            ati_g = all_token_ids.clone()
            tl_g = total_len.clone()
            mod.post_update(idx_mapping, nct_g, lst_g, obc_g, sampled_tokens, num_sampled, num_rejected, query_start_loc, ati_g, tl_g)
            torch.cuda.synchronize()
            ref = reference_post_update(idx_mapping.cpu(), num_computed_tokens.cpu(), last_sampled_tokens.cpu(), output_bin_counts.cpu(), sampled_tokens.cpu(), num_sampled.cpu(), num_rejected.cpu(), query_start_loc.cpu(), all_token_ids.cpu(), total_len.cpu())
            outputs = (nct_g, lst_g, obc_g, ati_g, tl_g)
            output_names = ('num_computed_tokens', 'last_sampled_tokens', 'output_bin_counts', 'all_token_ids', 'total_len')
            for name, actual, expected in zip(output_names, outputs, ref):
                if not torch.equal(actual.cpu(), expected):
                    return (False, f'{case_label}: {name} mismatch')
        except Exception as e:
            return (False, f'{case_label}: exception: {e}')
    return (True, None)
