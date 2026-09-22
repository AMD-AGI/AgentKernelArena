"""Unscored PR105 public-branch controls, ported from pinned main.

The original runner still owns every existing correctness and performance path.
These additional calls use the current checked candidate loader. No benchmark
helper or original workload is replaced.
Source revision: 0acf65b3a967ef1025dbfc5fd4b415b259e3bd43
Source path: tasks/triton2triton/vllm/triton_apply_grammar_bitmask/scripts/task_runner.py
"""
import sys
import os
import json
import argparse
import importlib.util
TEST_SHAPES = [(4, 1024), (8, 4096), (16, 8192), (32, 32000), (64, 65536)]
CORRECTNESS_CASES = [(*shape, None) for shape in TEST_SHAPES] + [(3, 37, (5, 1, 3)), (4, 8209, (6, 2, 5, 0))]

def reference_apply_grammar_bitmask(logits, logits_indices, bitmask, vocab_size):
    """CPU reference: unpack bitmask and apply to logits."""
    import torch
    logits = logits.clone()
    num_masks = bitmask.shape[0]
    for m in range(num_masks):
        logits_idx = logits_indices[m].item()
        for v in range(vocab_size):
            word_idx = v // 32
            bit_idx = v % 32
            bit = bitmask[m, word_idx].item() >> bit_idx & 1
            if bit == 0:
                logits[logits_idx, v] = float('-inf')
    return logits
EXTRA_CASES = [(3, 37, (5, 1, 3)), (4, 8209, (6, 2, 5, 0))]

def run_control(index, load_module):
    if type(index) is not int or not 0 <= index < len(EXTRA_CASES):
        raise ValueError('Unknown upstream correctness control')
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return (False, f'Failed to load module: {e}')
    device = 'cuda'
    for i, (num_masks, vocab_size, indices) in enumerate(CORRECTNESS_CASES):
        if i != index + 5:
            continue
        try:
            torch.manual_seed(42 + i)
            num_total_logits = num_masks + 4
            logits = torch.randn(num_total_logits, vocab_size, device=device, dtype=torch.float32)
            if indices is None:
                logits_indices = torch.arange(num_masks, dtype=torch.int32, device=device)
            else:
                logits_indices = torch.tensor(indices, dtype=torch.int32, device=device)
            bitmask_words = (vocab_size + 31) // 32
            bitmask = torch.randint(0, 2 ** 31, (num_masks, bitmask_words), dtype=torch.int32, device=device)
            logits_gpu = logits.clone()
            mod.apply_grammar_bitmask(logits_gpu, logits_indices, bitmask, vocab_size)
            torch.cuda.synchronize()
            ref = reference_apply_grammar_bitmask(logits.cpu(), logits_indices.cpu(), bitmask.cpu(), vocab_size)
            logits_gpu_cpu = logits_gpu.cpu()
            ref_neginf = ref.isinf() & (ref < 0)
            gpu_neginf = logits_gpu_cpu.isinf() & (logits_gpu_cpu < 0)
            if not torch.equal(ref_neginf, gpu_neginf):
                return (False, f'Shape {i + 1}: bitmask application mismatch')
            non_inf_mask = ~ref_neginf
            if not torch.allclose(logits_gpu_cpu[non_inf_mask], ref[non_inf_mask]):
                return (False, f'Shape {i + 1}: non-masked values changed')
            selected_rows = torch.zeros(num_total_logits, dtype=torch.bool)
            selected_rows[logits_indices.cpu().to(torch.long)] = True
            if not torch.equal(logits_gpu_cpu[~selected_rows], logits.cpu()[~selected_rows]):
                return (False, f'Shape {i + 1}: unselected logits rows changed')
        except Exception as e:
            return (False, f'Shape {i + 1}: exception: {e}')
    return (True, None)
