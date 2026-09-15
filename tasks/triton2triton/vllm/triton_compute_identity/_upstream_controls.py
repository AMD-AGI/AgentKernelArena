"""Unscored PR105 public-branch controls, ported from pinned main.

The original runner still owns every existing correctness and performance path.
These additional calls use the current checked candidate loader. No benchmark
helper or original workload is replaced.
Source revision: 0acf65b3a967ef1025dbfc5fd4b415b259e3bd43
Source path: tasks/triton2triton/vllm/triton_compute_identity/scripts/task_runner.py
"""
import sys, os, json, argparse, importlib.util
TEST_SHAPES = [(32, 256, 2), (64, 512, 2), (128, 1024, 2), (256, 2048, 4), (512, 4096, 2)]
CORRECTNESS_CASES = [(*shape, 'positive') for shape in TEST_SHAPES] + [(33, 768, 2, 'positive'), (31, 1280, 4, 'cancellation')]

def reference_compute_identity(hidden_states, expert_scales, top_k):
    """CPU reference: hidden_states * sum(expert_scales, dim=-1, keepdim=True)."""
    import torch
    scale_sum = expert_scales.sum(dim=-1, keepdim=True)
    return (hidden_states.float() * scale_sum.float()).to(hidden_states.dtype)
EXTRA_CASES = [(33, 768, 2, 'positive'), (31, 1280, 4, 'cancellation')]

def run_control(index, load_module):
    if type(index) is not int or not 0 <= index < len(EXTRA_CASES):
        raise ValueError('Unknown upstream correctness control')
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return (False, f'Failed to load module: {e}')
    device = 'cuda'
    for i, (num_tokens, hidden_dim, top_k, scale_pattern) in enumerate(CORRECTNESS_CASES):
        if i != index + 5:
            continue
        try:
            torch.manual_seed(42 + i)
            hidden_states = torch.randn(num_tokens, hidden_dim, device=device, dtype=torch.float16)
            if scale_pattern == 'cancellation':
                magnitudes = torch.rand(num_tokens, 2, device=device, dtype=torch.float32) + 0.5
                residual = torch.linspace(-0.001, 0.001, num_tokens, device=device)
                expert_scales = torch.stack((magnitudes[:, 0], -magnitudes[:, 0], magnitudes[:, 1], -magnitudes[:, 1] + residual), dim=1)
            else:
                expert_scales = torch.randn(num_tokens, top_k, device=device, dtype=torch.float32).abs() * 0.5
            result = mod.compute_identity(hidden_states, expert_scales, top_k)
            torch.cuda.synchronize()
            ref = reference_compute_identity(hidden_states, expert_scales, top_k).to(device)
            if not torch.allclose(result.float(), ref.float(), atol=0.01, rtol=0.01):
                max_diff = (result.float() - ref.float()).abs().max().item()
                return (False, f'Shape {i + 1}: max diff = {max_diff:.6f}')
        except Exception as e:
            return (False, f'Shape {i + 1}: exception: {e}')
    return (True, None)
