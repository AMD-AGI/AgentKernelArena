"""Unscored PR105 public-branch controls, ported from pinned main.

The original runner still owns every existing correctness and performance path.
These additional calls use the current checked candidate loader. No benchmark
helper or original workload is replaced.
Source revision: 0acf65b3a967ef1025dbfc5fd4b415b259e3bd43
Source path: tasks/triton2triton/vllm/triton_topk_log_softmax/scripts/task_runner.py
"""
import sys, os, json, argparse, importlib.util
TEST_SHAPES = [(4, 256, 3), (8, 1024, 5), (16, 4096, 10), (32, 8192, 20), (64, 32768, 10)]
CORRECTNESS_CASES = [{'name': f'random_{i + 1}', 'shape': shape, 'dtype': 'float32', 'values': 'random'} for i, shape in enumerate(TEST_SHAPES)] + [{'name': 'singleton_vocab', 'shape': (1, 1, 1), 'dtype': 'float32', 'values': 'singleton'}, {'name': 'non_power_of_two_below_block', 'shape': (3, 1000, 7), 'dtype': 'float32', 'values': 'random'}, {'name': 'non_power_of_two_above_block_fp16_edges', 'shape': (2, 1025, 5), 'dtype': 'float16', 'values': 'numerical_edges'}]

def _make_correctness_inputs(torch, case, seed, device):
    batch, vocab, ntok = case['shape']
    dtype = getattr(torch, case['dtype'])
    torch.manual_seed(seed)
    if case['values'] == 'random':
        logits = torch.randn(batch, vocab, device=device, dtype=dtype)
        token_ids = torch.randint(0, vocab, (batch, ntok), dtype=torch.int64, device=device)
    elif case['values'] == 'singleton':
        logits = torch.tensor([[12345.0]], device=device, dtype=dtype)
        token_ids = torch.tensor([[0]], device=device, dtype=torch.int64)
    elif case['values'] == 'numerical_edges':
        logits = torch.empty((batch, vocab), device=device, dtype=dtype)
        logits[0].fill_(-80.0)
        logits[0, 0] = 80.0
        logits[0, vocab // 2] = 79.5
        logits[0, -1] = 80.0
        logits[1].fill_(10000.0)
        token_ids = torch.tensor([[0, 1, vocab // 2, vocab - 2, vocab - 1], [0, 1, vocab // 2, vocab - 2, vocab - 1]], device=device, dtype=torch.int64)
    else:
        raise ValueError(f'Unknown correctness value pattern: {case['values']}')
    return (logits, token_ids)
EXTRA_CASES = [{'name': 'singleton_vocab', 'shape': (1, 1, 1), 'dtype': 'float32', 'values': 'singleton'}, {'name': 'non_power_of_two_below_block', 'shape': (3, 1000, 7), 'dtype': 'float32', 'values': 'random'}, {'name': 'non_power_of_two_above_block_fp16_edges', 'shape': (2, 1025, 5), 'dtype': 'float16', 'values': 'numerical_edges'}]

def run_control(index, load_module):
    if type(index) is not int or not 0 <= index < len(EXTRA_CASES):
        raise ValueError('Unknown upstream correctness control')
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return (False, f'Failed to load module: {e}')
    device = 'cuda'
    for i, case in enumerate(CORRECTNESS_CASES):
        if i != index + 5:
            continue
        try:
            logits, token_ids = _make_correctness_inputs(torch, case, seed=42 + i, device=device)
            result = mod.compute_token_logprobs(logits, token_ids)
            torch.cuda.synchronize()
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            ref = log_probs.gather(1, token_ids)
            if not torch.allclose(result, ref, atol=0.01, rtol=0.01):
                return (False, f'Case {i + 1} ({case['name']}): max diff = {(result - ref).abs().max().item()}')
        except Exception as e:
            return (False, f'Case {i + 1} ({case['name']}): exception: {e}')
    return (True, None)
