"""Unscored PR105 control inputs; original performance remains unchanged.
Source revision: 0acf65b3a967ef1025dbfc5fd4b415b259e3bd43
Source path: tasks/triton2triton/vllm/triton_bad_words/scripts/task_runner.py
"""
import sys, os, json, argparse, importlib.util
EXTRA_CASES = [{'name': 'irregular_speculative', 'vocab': 96, 'mapping': [2, 2, 0, 1, 1, 1], 'prefix_lengths': [1, 2, 3, 4]}, {'name': 'empty_grids', 'calls': [{'logits': [0, 32], 'bad_words': 1}, {'logits': [2, 32], 'bad_words': 0}]}]

def run_control(index, load_module):
    if type(index) is not int or not 0 <= index < len(EXTRA_CASES):
        raise ValueError('Unknown upstream correctness control')
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return (False, f'Failed to load module: {e}')
    device = 'cuda'

    def reference_bad_words(logits, idx_mapping, bad_word_ids, offsets, num_bw, all_token_ids, prompt_len, total_len, input_ids, local_pos):
        """Small scalar reference for targeted irregular-shape cases."""
        ref = logits.clone()
        for logit_idx in range(logits.shape[0]):
            req_idx = int(idx_mapping[logit_idx].item())
            pos = int(local_pos[logit_idx].item())
            first_pos = logit_idx - pos
            prompt = int(prompt_len[req_idx].item())
            output_len = int(total_len[req_idx].item()) - prompt
            effective_len = output_len + pos
            for bw_idx in range(int(num_bw[req_idx].item())):
                start = int(offsets[req_idx, bw_idx].item())
                end = int(offsets[req_idx, bw_idx + 1].item())
                prefix_len = end - start - 1
                if prefix_len > effective_len:
                    continue
                matched = True
                for prefix_idx in range(prefix_len):
                    expected = int(bad_word_ids[req_idx, start + prefix_idx].item())
                    actual_pos = effective_len - prefix_len + prefix_idx
                    if actual_pos >= output_len:
                        actual = int(input_ids[first_pos + actual_pos - output_len].item())
                    else:
                        actual = int(all_token_ids[req_idx, prompt + actual_pos].item())
                    if expected != actual:
                        matched = False
                        break
                if matched:
                    last_token = int(bad_word_ids[req_idx, end - 1].item())
                    ref[logit_idx, last_token] = float('-inf')
        return ref

    def exact_error(name, actual, expected):
        if torch.equal(actual, expected):
            return None
        mismatch = (actual != expected).nonzero()
        first = mismatch[0].tolist()
        return f'{name}: {mismatch.shape[0]} mismatched value(s); first at {first}: got {actual[tuple(first)].item()}, expected {expected[tuple(first)].item()}'
    multi_cases = [(4, 256, 2, 2), (8, 512, 3, 2), (6, 1024, 3, 3)]
    OUTPUT_LEN = 10
    PROMPT_LEN = 10
    if index == 0:
        try:
            vocab = 96
            logits = torch.arange(6 * vocab, device=device, dtype=torch.float32).reshape(6, vocab).to(torch.float16) / 100
            idx_mapping = torch.tensor([2, 2, 0, 1, 1, 1], dtype=torch.int32, device=device)
            local_pos = torch.tensor([0, 1, 0, 0, 1, 2], dtype=torch.int32, device=device)
            bad_word_ids = torch.zeros((4, 8), dtype=torch.int32, device=device)
            bad_word_ids[0, :7] = torch.tensor([11, 21, 22, 20, 21, 23, 34], dtype=torch.int32, device=device)
            bad_word_ids[1, :6] = torch.tensor([41, 42, 43, 42, 44, 45], dtype=torch.int32, device=device)
            bad_word_ids[2, :8] = torch.tensor([51, 61, 62, 63, 61, 62, 64, 66], dtype=torch.int32, device=device)
            offsets = torch.tensor([[0, 1, 3, 7, 7], [0, 3, 6, 6, 6], [0, 1, 4, 8, 8], [0, 0, 0, 0, 0]], dtype=torch.int32, device=device)
            num_bw = torch.tensor([3, 2, 3, 0], dtype=torch.int32, device=device)
            prompt_len = torch.tensor([2, 1, 3, 0], dtype=torch.int32, device=device)
            total_len = torch.tensor([4, 2, 5, 0], dtype=torch.int32, device=device)
            all_token_ids = torch.zeros((4, 8), dtype=torch.int32, device=device)
            all_token_ids[0, 2:4] = torch.tensor([20, 21], dtype=torch.int32, device=device)
            all_token_ids[1, 1] = 41
            all_token_ids[2, 3:5] = torch.tensor([61, 62], dtype=torch.int32, device=device)
            input_ids = torch.tensor([64, 90, 91, 42, 44, 92], dtype=torch.int32, device=device)
            ref = reference_bad_words(logits, idx_mapping, bad_word_ids, offsets, num_bw, all_token_ids, prompt_len, total_len, input_ids, local_pos)
            mod.apply_bad_words(logits, idx_mapping, bad_word_ids, offsets, num_bw, all_token_ids, prompt_len, total_len, input_ids, local_pos, 4)
            torch.cuda.synchronize()
            err = exact_error('Irregular speculative case', logits, ref)
            if err:
                return (False, err)
        except Exception as e:
            return (False, f'Irregular speculative case: exception: {e}')
    if index == 1:
        try:
            logits = torch.empty((0, 32), dtype=torch.float32, device=device)
            idx_mapping = torch.empty((0,), dtype=torch.int32, device=device)
            bad_word_ids = torch.tensor([[7]], dtype=torch.int32, device=device)
            offsets = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
            num_bw = torch.tensor([1], dtype=torch.int32, device=device)
            all_token_ids = torch.zeros((1, 4), dtype=torch.int32, device=device)
            prompt_len = torch.tensor([1], dtype=torch.int32, device=device)
            total_len = torch.tensor([1], dtype=torch.int32, device=device)
            input_ids = torch.empty((0,), dtype=torch.int32, device=device)
            local_pos = torch.empty((0,), dtype=torch.int32, device=device)
            mod.apply_bad_words(logits, idx_mapping, bad_word_ids, offsets, num_bw, all_token_ids, prompt_len, total_len, input_ids, local_pos, 1)
            logits = torch.arange(64, dtype=torch.float32, device=device).reshape(2, 32)
            ref = logits.clone()
            idx_mapping = torch.zeros((2,), dtype=torch.int32, device=device)
            bad_word_ids = torch.empty((1, 0), dtype=torch.int32, device=device)
            offsets = torch.zeros((1, 1), dtype=torch.int32, device=device)
            num_bw = torch.zeros((1,), dtype=torch.int32, device=device)
            all_token_ids = torch.zeros((1, 4), dtype=torch.int32, device=device)
            prompt_len = torch.tensor([1], dtype=torch.int32, device=device)
            total_len = torch.tensor([1], dtype=torch.int32, device=device)
            input_ids = torch.zeros((2,), dtype=torch.int32, device=device)
            local_pos = torch.tensor([0, 1], dtype=torch.int32, device=device)
            mod.apply_bad_words(logits, idx_mapping, bad_word_ids, offsets, num_bw, all_token_ids, prompt_len, total_len, input_ids, local_pos, 0)
            torch.cuda.synchronize()
            err = exact_error('Zero-bad-words case', logits, ref)
            if err:
                return (False, err)
        except Exception as e:
            return (False, f'Zero-sized case: exception: {e}')
    return (True, None)
