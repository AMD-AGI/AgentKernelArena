"""Additional unscored public-branch controls from PR105.

Source revision: 0acf65b3a967ef1025dbfc5fd4b415b259e3bd43
Source path: tasks/triton2triton/vllm/triton_ep_scatter_2/scripts/task_runner.py
Original scored inputs, gates and timing remain in scripts/task_runner.py.
"""
import sys, os, json, argparse, importlib.util
TEST_SHAPES = [(16, 64, 4, 2), (32, 128, 8, 2), (64, 256, 8, 2), (128, 512, 16, 2), (256, 512, 8, 2)]
CORRECTNESS_CASES = [(f'baseline_{i + 1}', *shape, 'float16', False) for i, shape in enumerate(TEST_SHAPES)] + [('single_token_boundary', 1, 1, 1, 1, 'float32', False), ('non_power_of_two_topk_1', 37, 65, 5, 1, 'float32', True), ('non_power_of_two_topk_3', 129, 257, 8, 3, 'float16', True), ('above_program_cap', 8209, 33, 8, 4, 'float16', True)]

def round_up_128(x):
    return (x + 127) // 128 * 128
EXTRA_CASES = [('single_token_boundary', 1, 1, 1, 1, 'float32', False), ('non_power_of_two_topk_1', 37, 65, 5, 1, 'float32', True), ('non_power_of_two_topk_3', 129, 257, 8, 3, 'float16', True), ('above_program_cap', 8209, 33, 8, 4, 'float16', True)]

def run_control(index, load_module):
    if type(index) is not int or not 0 <= index < len(EXTRA_CASES):
        raise ValueError('Unknown upstream correctness control')
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return (False, f'Failed to load module: {e}')
    device = 'cuda'
    for i, case in enumerate(EXTRA_CASES, start=5):
        if i != index + 5:
            continue
        case_name, num_tokens, hidden_size, num_experts, topk, dtype_name, negative_assignments = case
        try:
            torch.manual_seed(42 + i)
            dtype = getattr(torch, dtype_name)
            recv_x = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
            recv_topk = torch.randint(0, num_experts, (num_tokens, topk), device=device, dtype=torch.int32)
            if negative_assignments:
                recv_topk.view(-1)[::5] = -1
            recv_topk_cpu = recv_topk.cpu()
            valid_assignments_cpu = recv_topk_cpu[recv_topk_cpu >= 0]
            counts = torch.bincount(valid_assignments_cpu.to(torch.int64), minlength=num_experts).to(torch.int32)
            aligned = [round_up_128(c.item()) for c in counts]
            total = sum(aligned)
            starts = []
            s = 0
            for a in aligned:
                starts.append(s)
                s += a
            initial_expert_start_loc = torch.tensor(starts, device=device, dtype=torch.int32)
            expert_start_loc = initial_expert_start_loc.clone()
            output_tensor = torch.zeros(total, hidden_size, device=device, dtype=dtype)
            output_index = torch.full((num_tokens, topk), -1, device=device, dtype=torch.int32)
            mod.ep_scatter_2(recv_x, recv_topk, expert_start_loc, output_tensor, output_index)
            torch.cuda.synchronize()
            valid_mask = recv_topk >= 0
            invalid_mask = ~valid_mask
            if invalid_mask.any() and (not torch.all(output_index[invalid_mask] == -1)):
                return (False, f'Case {case_name}: negative assignment modified output_index')
            valid_indices = output_index[valid_mask].to(torch.int64)
            valid_experts = recv_topk[valid_mask].to(torch.int64)
            region_starts = initial_expert_start_loc[valid_experts].to(torch.int64)
            counts_device = counts.to(device=device, dtype=torch.int64)
            region_ends = region_starts + counts_device[valid_experts]
            in_expert_region = (valid_indices >= region_starts) & (valid_indices < region_ends)
            if not torch.all(in_expert_region):
                return (False, f'Case {case_name}: output_index outside assigned expert region')
            if torch.unique(valid_indices).numel() != valid_indices.numel():
                return (False, f'Case {case_name}: duplicate output_index values')
            token_ids = torch.arange(num_tokens, device=device, dtype=torch.int64)
            token_ids = token_ids[:, None].expand(num_tokens, topk)[valid_mask]
            if not torch.equal(output_tensor[valid_indices], recv_x[token_ids]):
                return (False, f'Case {case_name}: scattered token data mismatch')
            expected_final_counters = initial_expert_start_loc + counts.to(device)
            if not torch.equal(expert_start_loc, expected_final_counters):
                return (False, f'Case {case_name}: final expert counters mismatch')
        except Exception as e:
            return (False, f'Case {case_name}: exception: {e}')
    return (True, None)
