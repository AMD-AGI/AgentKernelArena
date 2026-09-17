"""Additional unscored public-branch controls from PR105.

Source revision: 0acf65b3a967ef1025dbfc5fd4b415b259e3bd43
Source path: tasks/triton2triton/vllm/triton_awq_dequantize/scripts/task_runner.py
Original scored inputs, gates and timing remain in scripts/task_runner.py.
"""
import sys
import os
import json
import argparse
import importlib.util
TEST_SHAPES = [(64, 8, 32), (128, 16, 32), (128, 16, 64), (256, 32, 128), (256, 32, 64)]
CORRECTNESS_TEST_CASES = [{'K': K, 'N_packed': N_packed, 'group_size': group_size} for K, N_packed, group_size in TEST_SHAPES] + [{'K': 33, 'N_packed': 9, 'group_size': 33, 'scale_dtype': 'float32', 'signed_packed': True}, {'K': 96, 'N_packed': 33, 'group_size': 32, 'scale_dtype': 'bfloat16'}, {'K': 512, 'N_packed': 65, 'group_size': 64}, {'K': 1024, 'N_packed': 128, 'group_size': 128}]

def reference_awq_dequantize(qweight, scales, zeros, group_size):
    """CPU reference: unpack 4-bit AWQ weights and dequantize."""
    import torch
    K, N_packed = qweight.shape
    N = N_packed * 8
    awq_order = [0, 4, 1, 5, 2, 6, 3, 7]
    result = torch.zeros((K, N), dtype=scales.dtype, device='cpu')
    qweight_cpu = qweight.cpu().to(torch.int32)
    zeros_cpu = zeros.cpu().to(torch.int32)
    scales_cpu = scales.cpu().float()
    for row in range(K):
        group_idx = row // group_size
        for col_packed in range(N_packed):
            packed_val = qweight_cpu[row, col_packed].item()
            zero_packed = zeros_cpu[group_idx, col_packed].item()
            for bit_idx in range(8):
                awq_idx = awq_order[bit_idx]
                weight_val = packed_val >> awq_idx * 4 & 15
                zero_val = zero_packed >> awq_idx * 4 & 15
                out_col = col_packed * 8 + bit_idx
                scale_val = scales_cpu[group_idx, out_col].item()
                result[row, out_col] = (weight_val - zero_val) * scale_val
    return result.to(scales.dtype)

def make_packed_int32(shape, device, signed_top_nibbles=False):
    """Create packed AWQ words, optionally covering every signed top nibble."""
    import torch
    if not signed_top_nibbles:
        return torch.randint(0, 2 ** 31, shape, device=device, dtype=torch.int32)
    lower_bits = torch.randint(0, 2 ** 28, shape, device=device, dtype=torch.int32)
    top_nibbles = torch.arange(lower_bits.numel(), device=device, dtype=torch.int64).reshape(shape).remainder(8).add(8)
    return (lower_bits.to(torch.int64) | top_nibbles << 28).to(torch.int32)
EXTRA_CASES = [{'K': 33, 'N_packed': 9, 'group_size': 33, 'scale_dtype': 'float32', 'signed_packed': True}, {'K': 96, 'N_packed': 33, 'group_size': 32, 'scale_dtype': 'bfloat16'}, {'K': 512, 'N_packed': 65, 'group_size': 64}, {'K': 1024, 'N_packed': 128, 'group_size': 128}]

def run_control(index, load_module):
    if type(index) is not int or not 0 <= index < len(EXTRA_CASES):
        raise ValueError('Unknown upstream correctness control')
    import torch
    try:
        mod = load_module()
        mod = _guarded_module(mod, 'awq_dequantize_triton', ())
    except Exception as e:
        return (False, f'Failed to load module: {e}')
    device = 'cuda'
    for i, case in enumerate(EXTRA_CASES, start=5):
        if i != index + 5:
            continue
        K = case['K']
        N_packed = case['N_packed']
        group_size = case['group_size']
        dtype = getattr(torch, case.get('scale_dtype', 'float16'))
        signed_packed = case.get('signed_packed', False)
        try:
            torch.manual_seed(42 + i)
            N = N_packed * 8
            num_groups = K // group_size
            qweight = make_packed_int32((K, N_packed), device, signed_top_nibbles=signed_packed)
            scales = torch.randn(num_groups, N, device=device, dtype=dtype).abs() * 0.1 + 0.01
            zeros = make_packed_int32((num_groups, N_packed), device, signed_top_nibbles=signed_packed)
            result = mod.awq_dequantize_triton(qweight, scales, zeros)
            torch.cuda.synchronize()
            ref = reference_awq_dequantize(qweight, scales, zeros, group_size)
            ref = ref.to(device)
            if not torch.allclose(result, ref, atol=0.01, rtol=0.01):
                max_diff = (result - ref).abs().max().item()
                return (False, f'Shape {i + 1} (K={K}, N_packed={N_packed}, G={group_size}, dtype={dtype}, signed_packed={signed_packed}): max diff = {max_diff:.6f}')
        except Exception as e:
            return (False, f'Shape {i + 1} (K={K}, N_packed={N_packed}, G={group_size}, dtype={dtype}, signed_packed={signed_packed}): exception: {e}')
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
