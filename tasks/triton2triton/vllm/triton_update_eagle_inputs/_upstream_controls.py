"""Additional unscored public-branch controls from PR105.

Source revision: 0acf65b3a967ef1025dbfc5fd4b415b259e3bd43
Source path: tasks/triton2triton/vllm/triton_update_eagle_inputs/scripts/task_runner.py
Original scored inputs, gates and timing remain in scripts/task_runner.py.
"""
import sys, os, json, argparse, importlib.util
TEST_SHAPES = [(4, 256, 2048), (8, 512, 4096), (16, 768, 4096), (32, 1024, 8192), (64, 2048, 8192)]
CORRECTNESS_CASES = [*((shape, False) for shape in TEST_SHAPES), ((4, 1536, 2048), True)]

def make_inputs(num_reqs, hidden_size, max_model_len, device='cpu', boundary_values=False):
    import torch
    torch.manual_seed(42)
    draft_tokens = torch.randint(0, 32000, (num_reqs,), dtype=torch.int64)
    output_hs = torch.randn(num_reqs, hidden_size, dtype=torch.float16)
    input_ids = torch.zeros(num_reqs, dtype=torch.int64)
    positions = torch.randint(0, max_model_len - 2, (num_reqs,), dtype=torch.int32)
    input_hs = torch.zeros(num_reqs, hidden_size, dtype=torch.float16)
    seq_lens = torch.randint(1, max_model_len - 1, (num_reqs,), dtype=torch.int32)
    if boundary_values:
        if num_reqs < 4:
            raise ValueError('boundary cases require at least four requests')
        positions[:4] = torch.tensor([max_model_len - 2, max_model_len - 1, 0, max_model_len - 3], dtype=positions.dtype)
        seq_lens[:4] = torch.tensor([max_model_len - 1, max_model_len, 1, max_model_len - 2], dtype=seq_lens.dtype)
    if device != 'cpu':
        draft_tokens = draft_tokens.to(device)
        output_hs = output_hs.to(device)
        input_ids = input_ids.to(device)
        positions = positions.to(device)
        input_hs = input_hs.to(device)
        seq_lens = seq_lens.to(device)
    return (draft_tokens, output_hs, input_ids, positions, input_hs, seq_lens)

def reference(draft_tokens, output_hs, input_ids, positions, input_hs, seq_lens, max_ml):
    import torch
    num_reqs = draft_tokens.shape[0]
    input_ids = input_ids.clone()
    positions = positions.clone()
    input_hs = input_hs.clone()
    seq_lens = seq_lens.clone()
    for r in range(num_reqs):
        input_ids[r] = draft_tokens[r]
        input_hs[r] = output_hs[r]
        positions[r] = min(positions[r].item() + 1, max_ml - 1)
        seq_lens[r] = min(seq_lens[r].item() + 1, max_ml)
    return (input_ids, positions, input_hs, seq_lens)
EXTRA_CASES = [((4, 1536, 2048), True)]

def run_control(index, load_module):
    if type(index) is not int or not 0 <= index < len(EXTRA_CASES):
        raise ValueError('Unknown upstream correctness control')
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return (False, f'Failed to load module: {e}')
    device = 'cuda'
    for i, ((nr, hs, mml), boundary_values) in enumerate(EXTRA_CASES, start=5):
        if i != index + 5:
            continue
        try:
            gpu_inputs = make_inputs(nr, hs, mml, device, boundary_values)
            cpu_inputs = make_inputs(nr, hs, mml, 'cpu', boundary_values)
            mod.update_eagle_inputs(gpu_inputs[0], gpu_inputs[1], gpu_inputs[2], gpu_inputs[3], gpu_inputs[4], gpu_inputs[5], mml)
            torch.cuda.synchronize()
            ref = reference(cpu_inputs[0], cpu_inputs[1], cpu_inputs[2], cpu_inputs[3], cpu_inputs[4], cpu_inputs[5], mml)
            if not torch.equal(gpu_inputs[2].cpu(), ref[0]):
                return (False, f'Shape {i + 1}: input_ids mismatch')
            if not torch.equal(gpu_inputs[3].cpu(), ref[1]):
                return (False, f'Shape {i + 1}: positions mismatch')
            if not torch.allclose(gpu_inputs[4].cpu().float(), ref[2].float(), atol=0.001, rtol=0.001):
                return (False, f'Shape {i + 1}: hidden states mismatch')
            if not torch.equal(gpu_inputs[5].cpu(), ref[3]):
                return (False, f'Shape {i + 1}: seq_lens mismatch')
        except Exception as e:
            return (False, f'Shape {i + 1}: exception: {e}')
    return (True, None)
