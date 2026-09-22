"""Additional unscored public-branch controls from PR105.

Source revision: 0acf65b3a967ef1025dbfc5fd4b415b259e3bd43
Source path: tasks/triton2triton/vllm/triton_prepare_eagle_docode/scripts/task_runner.py
Original scored inputs, gates and timing remain in scripts/task_runner.py.
"""
import sys, os, json, argparse, importlib.util
TEST_SHAPES = [(4, 64, 256, 2048, 8), (8, 128, 512, 4096, 16), (16, 256, 768, 4096, 32), (32, 512, 1024, 8192, 64), (64, 1024, 2048, 8192, 128)]
CORRECTNESS_TEST_SHAPES = TEST_SHAPES + [(16, 256, 2049, 8192, 256), (17, 272, 4096, 8192, 257)]

def make_inputs(num_reqs, total_tokens, hidden_size, max_model_len, max_num_reqs, device='cpu'):
    import torch
    torch.manual_seed(42)
    draft_tokens = torch.randint(0, 32000, (num_reqs,), dtype=torch.int64)
    output_hidden_states = torch.randn(total_tokens, hidden_size, dtype=torch.float16)
    last_token_indices = torch.randint(0, total_tokens, (num_reqs,), dtype=torch.int64)
    target_seq_lens = torch.randint(10, max_model_len // 2, (num_reqs,), dtype=torch.int32)
    num_rejected = torch.randint(0, 3, (num_reqs,), dtype=torch.int32)
    positions = torch.randint(0, max_model_len - 2, (max(total_tokens, max_num_reqs),), dtype=torch.int32)
    seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32)
    query_start_loc = torch.zeros(max_num_reqs + 1, dtype=torch.int32)
    input_ids = torch.zeros(max(total_tokens, max_num_reqs), dtype=torch.int64)
    input_hidden_states = torch.zeros(max(total_tokens, max_num_reqs), hidden_size, dtype=torch.float16)
    if device != 'cpu':
        draft_tokens = draft_tokens.to(device)
        output_hidden_states = output_hidden_states.to(device)
        last_token_indices = last_token_indices.to(device)
        target_seq_lens = target_seq_lens.to(device)
        num_rejected = num_rejected.to(device)
        positions = positions.to(device)
        seq_lens = seq_lens.to(device)
        query_start_loc = query_start_loc.to(device)
        input_ids = input_ids.to(device)
        input_hidden_states = input_hidden_states.to(device)
    return (draft_tokens, output_hidden_states, last_token_indices, target_seq_lens, num_rejected, positions, seq_lens, query_start_loc, input_ids, input_hidden_states)

def reference(draft_tokens, output_hs, last_ti, target_sl, num_rej, positions, seq_lens, qsl, input_ids, input_hs, max_ml, max_nr):
    import torch
    num_reqs = draft_tokens.shape[0]
    hidden_size = output_hs.shape[-1]
    positions = positions.clone()
    seq_lens = seq_lens.clone()
    qsl = qsl.clone()
    input_ids = input_ids.clone()
    input_hs = input_hs.clone()
    for r in range(num_reqs):
        input_ids[r] = draft_tokens[r]
        src = last_ti[r].item()
        input_hs[r] = output_hs[src]
        pos = min(positions[r].item() + 1, max_ml - 1)
        positions[r] = pos
        sl = target_sl[r].item() - num_rej[r].item()
        sl = min(sl + 1, max_ml)
        seq_lens[r] = sl
    for i in range(max_nr + 1):
        if i < num_reqs:
            qsl[i] = i
        else:
            qsl[i] = num_reqs
    for i in range(num_reqs, max_nr):
        seq_lens[i] = 0
    return (positions, seq_lens, qsl, input_ids, input_hs)
EXTRA_CASES = [(16, 256, 2049, 8192, 256), (17, 272, 4096, 8192, 257)]

def run_control(index, load_module):
    if type(index) is not int or not 0 <= index < len(EXTRA_CASES):
        raise ValueError('Unknown upstream correctness control')
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return (False, f'Failed to load module: {e}')
    device = 'cuda'
    for i, (nr, tt, hs, mml, mnr) in enumerate(EXTRA_CASES, start=5):
        if i != index + 5:
            continue
        try:
            inputs_gpu = make_inputs(nr, tt, hs, mml, mnr, device)
            inputs_cpu = make_inputs(nr, tt, hs, mml, mnr, 'cpu')
            mod.prepare_eagle_decode(inputs_gpu[0], inputs_gpu[1], inputs_gpu[2], inputs_gpu[3], inputs_gpu[4], inputs_gpu[5], inputs_gpu[6], inputs_gpu[7], inputs_gpu[8], inputs_gpu[9], mml, mnr)
            torch.cuda.synchronize()
            ref = reference(inputs_cpu[0], inputs_cpu[1], inputs_cpu[2], inputs_cpu[3], inputs_cpu[4], inputs_cpu[5], inputs_cpu[6], inputs_cpu[7], inputs_cpu[8], inputs_cpu[9], mml, mnr)
            if not torch.equal(inputs_gpu[8][:nr].cpu(), ref[3][:nr]):
                return (False, f'Shape {i + 1}: input_ids mismatch')
            if not torch.equal(inputs_gpu[5][:nr].cpu(), ref[0][:nr]):
                return (False, f'Shape {i + 1}: positions mismatch')
            if not torch.equal(inputs_gpu[6].cpu(), ref[1]):
                return (False, f'Shape {i + 1}: seq_lens mismatch')
            if not torch.equal(inputs_gpu[7].cpu(), ref[2]):
                return (False, f'Shape {i + 1}: query_start_loc mismatch')
            if not torch.allclose(inputs_gpu[9][:nr].cpu().float(), ref[4][:nr].float(), atol=0.001, rtol=0.001):
                return (False, f'Shape {i + 1}: hidden states mismatch')
        except Exception as e:
            return (False, f'Shape {i + 1}: exception: {e}')
    return (True, None)
