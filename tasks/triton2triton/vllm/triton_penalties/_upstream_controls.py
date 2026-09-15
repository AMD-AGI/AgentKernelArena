"""Additional unscored public-branch controls from PR105.

Source revision: 0acf65b3a967ef1025dbfc5fd4b415b259e3bd43
Source path: tasks/triton2triton/vllm/triton_penalties/scripts/task_runner.py
Original scored inputs, gates and timing remain in scripts/task_runner.py.
"""
import sys, os, json, argparse, importlib.util
TARGETED_CORRECTNESS_CASES = [{'name': 'small_vocab_speculative_mapped_penalties_fp16', 'vocab': 7, 'dtype': 'float16', 'idx_mapping': [2, 2, 2, 0, 0, 0, 1, 1, 1, 3], 'token_ids': [6, 5, 5, 0, 2, 6, 4, 0, 0, 3], 'local_pos': [0, 1, 2, 0, 1, 2, 0, 1, 2, 0], 'num_speculative_tokens': 2, 'penalties': [(1.25, 0.0, 0.0), (1.0, 0.5, 0.0), (1.0, 0.0, 0.75), (1.0, 0.0, 0.0)], 'prompt_tokens': [[0, 3, 6], [1], [2], [0, 6]], 'output_counts': [(0, 1, 2), (0, 5, 1), (1, 0, 3), (1, 6, 1), (2, 2, 1), (3, 4, 3)]}, {'name': 'packed_mask_and_tile_tails_bf16', 'vocab': 8209, 'dtype': 'bfloat16', 'idx_mapping': [1, 0, 1], 'token_ids': [8208, 8192, 31], 'local_pos': [0, 0, 0], 'num_speculative_tokens': 0, 'penalties': [(1.2, 0.35, 0.15), (1.3, 0.0, 0.0)], 'prompt_tokens': [[0, 31, 32, 8191, 8192, 8208], [1, 63, 8193, 8208]], 'output_counts': [(0, 31, 1), (0, 8192, 2), (0, 8208, 3), (1, 0, 1), (1, 8191, 4), (1, 8193, 2)]}]

def unpack_prompt_mask(packed_mask_row, vocab_size):
    import torch
    out = torch.zeros(vocab_size, dtype=torch.bool, device=packed_mask_row.device)
    for tok in range(vocab_size):
        out[tok] = packed_mask_row[tok // 32] >> tok % 32 & 1 != 0
    return out

def reference_apply_penalties(logits, idx_mapping, token_ids, local_pos, repetition_penalty, frequency_penalty, presence_penalty, prompt_bin_mask, output_bin_counts, num_speculative_tokens):
    import torch
    out = logits.clone().float()
    num_tokens, vocab_size = out.shape
    for token_idx in range(num_tokens):
        state_idx = idx_mapping[token_idx].item()
        rep = repetition_penalty[state_idx].item()
        freq = frequency_penalty[state_idx].item()
        pres = presence_penalty[state_idx].item()
        if rep == 1.0 and freq == 0.0 and (pres == 0.0):
            continue
        counts = output_bin_counts[state_idx].to(torch.int32).clone()
        if num_speculative_tokens > 0:
            pos = local_pos[token_idx].item()
            start_idx = token_idx - pos
            for prev_pos in range(pos):
                prev_token = token_ids[start_idx + prev_pos + 1].item()
                counts[prev_token] += 1
        output_mask = counts > 0
        prompt_mask = unpack_prompt_mask(prompt_bin_mask[state_idx], vocab_size)
        if rep != 1.0:
            scale_mask = prompt_mask | output_mask
            scale = torch.where(scale_mask, torch.tensor(rep, device=out.device), torch.tensor(1.0, device=out.device))
            out[token_idx] = torch.where(out[token_idx] > 0, out[token_idx] / scale, out[token_idx] * scale)
        out[token_idx] -= freq * counts.float()
        out[token_idx] -= pres * output_mask.float()
    return out.to(logits.dtype)

def run_targeted_correctness_case(mod, case, seed):
    import torch
    device = 'cuda'
    vocab = case['vocab']
    dtype = getattr(torch, case['dtype'])
    idx_mapping = torch.tensor(case['idx_mapping'], dtype=torch.int32, device=device)
    token_ids = torch.tensor(case['token_ids'], dtype=torch.int32, device=device)
    local_pos = torch.tensor(case['local_pos'], dtype=torch.int32, device=device)
    num_states = len(case['penalties'])
    penalties = torch.tensor(case['penalties'], dtype=torch.float32, device=device)
    rep_pen = penalties[:, 0].contiguous()
    freq_pen = penalties[:, 1].contiguous()
    pres_pen = penalties[:, 2].contiguous()
    prompt_mask = torch.zeros(num_states, (vocab + 31) // 32, dtype=torch.int64)
    for state_idx, prompt_tokens in enumerate(case['prompt_tokens']):
        for token_id in prompt_tokens:
            prompt_mask[state_idx, token_id // 32] |= 1 << token_id % 32
    prompt_mask = prompt_mask.to(dtype=torch.int32, device=device)
    output_counts = torch.zeros(num_states, vocab, dtype=torch.int32, device=device)
    for state_idx, token_id, count in case['output_counts']:
        output_counts[state_idx, token_id] = count
    torch.manual_seed(seed)
    logits = torch.randn(len(case['idx_mapping']), vocab, dtype=torch.float32, device=device).to(dtype)
    logits[:, 0] = 2.0
    logits[:, -1] = -2.0
    ref = reference_apply_penalties(logits, idx_mapping, token_ids, local_pos, rep_pen, freq_pen, pres_pen, prompt_mask, output_counts, case['num_speculative_tokens'])
    mod.apply_penalties(logits, idx_mapping, token_ids, local_pos, rep_pen, freq_pen, pres_pen, prompt_mask, output_counts, case['num_speculative_tokens'])
    torch.cuda.synchronize()
    if not torch.allclose(logits, ref, atol=0.01, rtol=0.01):
        return (False, (logits - ref).abs().max().item())
    return (True, None)
EXTRA_CASES = [{'name': 'small_vocab_speculative_mapped_penalties_fp16', 'vocab': 7, 'dtype': 'float16', 'idx_mapping': [2, 2, 2, 0, 0, 0, 1, 1, 1, 3], 'token_ids': [6, 5, 5, 0, 2, 6, 4, 0, 0, 3], 'local_pos': [0, 1, 2, 0, 1, 2, 0, 1, 2, 0], 'num_speculative_tokens': 2, 'penalties': [(1.25, 0.0, 0.0), (1.0, 0.5, 0.0), (1.0, 0.0, 0.75), (1.0, 0.0, 0.0)], 'prompt_tokens': [[0, 3, 6], [1], [2], [0, 6]], 'output_counts': [(0, 1, 2), (0, 5, 1), (1, 0, 3), (1, 6, 1), (2, 2, 1), (3, 4, 3)]}, {'name': 'packed_mask_and_tile_tails_bf16', 'vocab': 8209, 'dtype': 'bfloat16', 'idx_mapping': [1, 0, 1], 'token_ids': [8208, 8192, 31], 'local_pos': [0, 0, 0], 'num_speculative_tokens': 0, 'penalties': [(1.2, 0.35, 0.15), (1.3, 0.0, 0.0)], 'prompt_tokens': [[0, 31, 32, 8191, 8192, 8208], [1, 63, 8193, 8208]], 'output_counts': [(0, 31, 1), (0, 8192, 2), (0, 8208, 3), (1, 0, 1), (1, 8191, 4), (1, 8193, 2)]}]

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
            ok, max_diff = run_targeted_correctness_case(mod, case, 100 + i)
            if not ok:
                return (False, f'Case {case['name']}: max diff = {max_diff}')
        except Exception as e:
            return (False, f'Case {case['name']}: exception: {e}')
    return (True, None)
