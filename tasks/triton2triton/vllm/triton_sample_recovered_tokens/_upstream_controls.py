"""Unscored PR105 control inputs; original performance remains unchanged.
Source revision: 0acf65b3a967ef1025dbfc5fd4b415b259e3bd43
Source path: tasks/triton2triton/vllm/triton_sample_recovered_tokens/scripts/task_runner.py
"""
import sys, os, json, argparse, importlib.util
CORRECTNESS_EDGE_CASES = [(1, 1, 3, [1]), (5, 3, 511, [0, 1, 0, 3, 0])]

def reference_sample_recovered_tokens(cu_num_draft_tokens, draft_token_ids, draft_probs, target_probs, q, vocab_size):
    import torch
    batch_size = cu_num_draft_tokens.shape[0]
    out = torch.empty_like(draft_token_ids)
    start = 0
    for req in range(batch_size):
        end = cu_num_draft_tokens[req].item()
        for idx in range(start, end):
            if draft_probs is None:
                prob = target_probs[idx].clone()
                prob[draft_token_ids[idx]] = 0
            else:
                prob = torch.maximum(target_probs[idx] - draft_probs[idx], torch.zeros_like(target_probs[idx]))
            out[idx] = torch.argmax(prob / q[req]).to(out.dtype)
        start = end
    return out
EXTRA_CASES = [{'name': 'singleton', 'shape': [1, 1, 3], 'lengths': [1]}, {'name': 'empty_ragged', 'shape': [5, 3, 511], 'lengths': [0, 1, 0, 3, 0]}, {'name': 'recovered_distribution'}]

def run_control(index, load_module):
    if type(index) is not int or not 0 <= index < len(EXTRA_CASES):
        raise ValueError('Unknown upstream correctness control')
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return (False, f'Failed to load module: {e}')
    device = 'cuda'
    for i, (batch_size, max_draft, vocab_size, num_per_req) in enumerate(CORRECTNESS_EDGE_CASES):
        if i + 0 != index:
            continue
        case_name = f'Edge shape {i + 1}'
        try:
            torch.manual_seed(100 + i)
            cu = torch.cumsum(torch.tensor(num_per_req, dtype=torch.int32, device=device), dim=0)
            total = sum(num_per_req)
            draft_ids = torch.randint(0, vocab_size, (total,), dtype=torch.int32, device=device)
            draft_probs = torch.rand(total, vocab_size, device=device)
            draft_probs = draft_probs / draft_probs.sum(-1, keepdim=True)
            target_probs = torch.rand(total, vocab_size, device=device)
            target_probs = target_probs / target_probs.sum(-1, keepdim=True)
            q = torch.empty(batch_size, vocab_size, device=device).exponential_()
            result = mod.sample_recovered_tokens(cu, draft_ids, draft_probs, target_probs, q, max_draft, vocab_size)
            ref = reference_sample_recovered_tokens(cu, draft_ids, draft_probs, target_probs, q, vocab_size)
            if not torch.equal(result, ref):
                return (False, f'{case_name}: mismatch with draft_probs path')
            result_no_draft = mod.sample_recovered_tokens(cu, draft_ids, None, target_probs, q, max_draft, vocab_size)
            ref_no_draft = reference_sample_recovered_tokens(cu, draft_ids, None, target_probs, q, vocab_size)
            if not torch.equal(result_no_draft, ref_no_draft):
                return (False, f'{case_name}: mismatch with NO_DRAFT_PROBS path')
            torch.cuda.synchronize()
            assert result.shape == (total,), f'Wrong shape: {result.shape}'
            assert result.min() >= 0 and result.max() < vocab_size
        except Exception as e:
            return (False, f'{case_name}: exception: {e}')
    if index == 2:
        try:
            batch_size, max_draft, vocab_size = (3, 2, 513)
            cu = torch.tensor([1, 1, 3], dtype=torch.int32, device=device)
            draft_ids = torch.tensor([0, 5, 0], dtype=torch.int32, device=device)
            draft_probs = torch.zeros(3, vocab_size, device=device)
            target_probs = torch.zeros_like(draft_probs)
            draft_probs[0, 0] = 1.0
            target_probs[0, 1] = 0.5
            target_probs[0, 3] = 0.5
            draft_probs[1, 5] = 1.0
            target_probs[1, 5] = 1.0
            draft_probs[2, 0] = 1.0
            target_probs[2, vocab_size - 1] = 1.0
            q = torch.ones(batch_size, vocab_size, device=device)
            expected = torch.tensor([1, 0, vocab_size - 1], dtype=torch.int32, device=device)
            ref = reference_sample_recovered_tokens(cu, draft_ids, draft_probs, target_probs, q, vocab_size)
            ref_no_draft = reference_sample_recovered_tokens(cu, draft_ids, None, target_probs, q, vocab_size)
            if not torch.equal(ref, expected) or not torch.equal(ref_no_draft, expected):
                return (False, 'Tie/padding case: reference did not produce expected tokens')
            result = mod.sample_recovered_tokens(cu, draft_ids, draft_probs, target_probs, q, max_draft, vocab_size)
            if not torch.equal(result, expected):
                return (False, 'Tie/padding case: mismatch with draft_probs path')
            result_no_draft = mod.sample_recovered_tokens(cu, draft_ids, None, target_probs, q, max_draft, vocab_size)
            if not torch.equal(result_no_draft, expected):
                return (False, 'Tie/padding case: mismatch with NO_DRAFT_PROBS path')
            torch.cuda.synchronize()
        except Exception as e:
            return (False, f'Tie/padding case: exception: {e}')
    return (True, None)
