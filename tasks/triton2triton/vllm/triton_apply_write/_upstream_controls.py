"""Unscored PR105 control inputs; original performance remains unchanged.
Source revision: 0acf65b3a967ef1025dbfc5fd4b415b259e3bd43
Source path: tasks/triton2triton/vllm/triton_apply_write/scripts/task_runner.py
"""
import sys
import os
import json
import argparse
import importlib.util

def reference_apply_write(output, write_indices, write_starts, write_contents, write_cu_lens):
    import torch
    output = output.clone()
    n = write_indices.shape[0]
    for i in range(n):
        row_idx = write_indices[i].item()
        start_idx = write_starts[i].item()
        cu_start = write_cu_lens[i - 1].item() if i > 0 else 0
        cu_end = write_cu_lens[i].item()
        content_len = cu_end - cu_start
        for j in range(content_len):
            output[row_idx, start_idx + j] = write_contents[cu_start + j]
    return output
EXTRA_CASES = [{'name': 'zero_writes', 'output': [4, 64], 'lengths': []}, {'name': 'zero_length_segments', 'output': [10, 64], 'lengths': [0, 0, 5, 0, 9, 0, 3, 0]}, {'name': 'block_cap_tails', 'output': [6, 4096], 'lengths': [1, 1024, 1025, 2049]}]

def run_control(index, load_module):
    if type(index) is not int or not 0 <= index < len(EXTRA_CASES):
        raise ValueError('Unknown upstream correctness control')
    import torch
    try:
        mod = load_module()
        mod = _guarded_module(mod, 'apply_write', ('output',))
    except Exception as e:
        return (False, f'Failed to load module: {e}')
    device = 'cuda'

    def check_case(case_name, output, write_indices, write_starts, write_contents, cu_lens):
        output_gpu = output.clone()
        mod.apply_write(output_gpu, write_indices, write_starts, write_contents, cu_lens)
        torch.cuda.synchronize()
        ref = reference_apply_write(output.cpu(), write_indices.cpu(), write_starts.cpu(), write_contents.cpu(), cu_lens.cpu())
        if not torch.equal(output_gpu.cpu(), ref):
            return (False, f'{case_name}: mismatch')
        return (True, None)
    boundary_cases = [{'name': 'zero writes', 'output': torch.randint(-100, 100, (4, 64), dtype=torch.int32, device=device), 'write_indices': torch.empty(0, dtype=torch.int32, device=device), 'write_starts': torch.empty(0, dtype=torch.int32, device=device), 'content_lens': []}, {'name': 'zero-length segments', 'output': torch.randint(-100, 100, (10, 64), dtype=torch.int32, device=device), 'write_indices': torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int32, device=device), 'write_starts': torch.tensor([3, 7, 11, 17, 23, 31, 37, 41], dtype=torch.int32, device=device), 'content_lens': [0, 0, 5, 0, 9, 0, 3, 0]}, {'name': 'skewed lengths at the block cap', 'output': torch.randint(-100, 100, (6, 4096), dtype=torch.int32, device=device), 'write_indices': torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=device), 'write_starts': torch.tensor([4095, 3072, 1536, 2047], dtype=torch.int32, device=device), 'content_lens': [1, 1024, 1025, 2049]}]
    for _extra_index, case in enumerate(boundary_cases):
        if _extra_index + 0 != index:
            continue
        try:
            content_lens = torch.tensor(case['content_lens'], dtype=torch.int32, device=device)
            cu_lens = torch.cumsum(content_lens, dim=0).to(torch.int32)
            total_content = int(cu_lens[-1].item()) if cu_lens.numel() else 0
            write_contents = torch.randint(1, 10000, (total_content,), dtype=torch.int32, device=device)
            ok, err = check_case(case['name'], case['output'], case['write_indices'], case['write_starts'], write_contents, cu_lens)
            if not ok:
                return (False, err)
        except Exception as e:
            return (False, f'{case['name']}: exception: {e}')
    return (True, None)

def _guarded_module(module, symbol, mutable_names):
    """Freeze all read-only control inputs around this public candidate call."""
    import inspect
    import torch
    original = getattr(module, symbol)
    signature = inspect.signature(original)

    def tensors(value):
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, (tuple, list)):
            for item in value:
                yield from tensors(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from tensors(item)

    def invoke(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        readonly = [value for name, arg in bound.arguments.items()
                    if name not in mutable_names for value in tensors(arg)]
        saved = [value.detach().clone() for value in readonly]
        try:
            result = original(*args, **kwargs)
            for value, initial in zip(readonly, saved):
                if (value.shape != initial.shape or value.dtype != initial.dtype
                        or value.device != initial.device or not torch.equal(
                            value.contiguous().reshape(-1).view(torch.uint8),
                            initial.contiguous().reshape(-1).view(torch.uint8))):
                    raise AssertionError('Candidate modified a read-only control input')
            return result
        finally:
            with torch.no_grad():
                for value, initial in zip(readonly, saved):
                    value.copy_(initial)

    class ControlModule:
        def __getattr__(self, name):
            return invoke if name == symbol else getattr(module, name)
    return ControlModule()
