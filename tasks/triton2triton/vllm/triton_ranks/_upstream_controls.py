"""Additional unscored public-branch controls from PR105.

Source revision: 0acf65b3a967ef1025dbfc5fd4b415b259e3bd43
Source path: tasks/triton2triton/vllm/triton_ranks/scripts/task_runner.py
Original scored inputs, gates and timing remain in scripts/task_runner.py.
"""
import sys, os, json, argparse, importlib.util
ADDITIONAL_CORRECTNESS_CASES = [{'name': 'masked_tail', 'batch': 3, 'vocab': 8193, 'dtype': 'float32', 'pattern': 'all_ties'}, {'name': 'above_32768', 'batch': 2, 'vocab': 40003, 'dtype': 'float32', 'pattern': 'high_tail'}, {'name': 'large_batch_small_vocab_fp16_ties', 'batch': 65, 'vocab': 257, 'dtype': 'float16', 'pattern': 'repeated_values'}]
EXTRA_CASES = [{'name': 'masked_tail', 'batch': 3, 'vocab': 8193, 'dtype': 'float32', 'pattern': 'all_ties'}, {'name': 'above_32768', 'batch': 2, 'vocab': 40003, 'dtype': 'float32', 'pattern': 'high_tail'}, {'name': 'large_batch_small_vocab_fp16_ties', 'batch': 65, 'vocab': 257, 'dtype': 'float16', 'pattern': 'repeated_values'}]

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
        name = case['name']
        batch = case['batch']
        vocab = case['vocab']
        dtype = getattr(torch, case['dtype'])
        try:
            if case['pattern'] == 'all_ties':
                logits = torch.zeros(batch, vocab, device=device, dtype=dtype)
                token_ids = torch.full((batch,), vocab - 1, dtype=torch.int64, device=device)
            elif case['pattern'] == 'high_tail':
                logits = torch.zeros(batch, vocab, device=device, dtype=dtype)
                logits[:, 32768:] = 1
                token_ids = torch.tensor([0, vocab - 1], dtype=torch.int64, device=device)
            else:
                values = (torch.arange(vocab, device=device) % 7 - 3).to(dtype)
                logits = values.unsqueeze(0).expand(batch, -1).contiguous()
                token_ids = (torch.arange(batch, dtype=torch.int64, device=device) * 37 + vocab - 1) % vocab
            result = mod.compute_ranks(logits, token_ids)
            torch.cuda.synchronize()
            ref = torch.zeros(batch, dtype=torch.int64, device=device)
            for b in range(batch):
                x = logits[b, token_ids[b].item()]
                ref[b] = (logits[b] >= x).sum().item()
            if not torch.equal(result, ref):
                return (False, f'Additional case {name}: mismatch')
        except Exception as e:
            return (False, f'Additional case {name}: exception: {e}')
    return (True, None)
