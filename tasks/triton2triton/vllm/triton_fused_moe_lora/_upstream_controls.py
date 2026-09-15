"""Additional unscored public-branch controls from PR105.

Source revision: 0acf65b3a967ef1025dbfc5fd4b415b259e3bd43
Source path: tasks/triton2triton/vllm/triton_fused_moe_lora/scripts/task_runner.py
Original scored inputs, gates and timing remain in scripts/task_runner.py.
"""
import sys, os, json, argparse, importlib.util
CORRECTNESS_CASES = [{'name': 'naive_irregular_slices_offset', 'shape': (7, 70, 5, 13, 45, 3, 3), 'num_slices': 2, 'offset': 5, 'dtype': 'float16', 'sorted_assignment': False, 'mul_routed_weight': False, 'include_invalid_routes': True}, {'name': 'sorted_routed_weight_bfloat16', 'shape': (9, 65, 3, 17, 33, 3, 3), 'num_slices': 1, 'offset': 0, 'dtype': 'bfloat16', 'sorted_assignment': True, 'mul_routed_weight': True, 'include_invalid_routes': True}, {'name': 'split_k2_no_l2_cache', 'shape': (5, 70, 3, 11, 37, 2, 2), 'num_slices': 1, 'offset': 0, 'dtype': 'float16', 'sorted_assignment': False, 'mul_routed_weight': False, 'include_invalid_routes': False, 'shrink_split_k': 2, 'use_b_l2_cache': False}]

def reference_fused_moe_lora(qcurr_hidden_states, lora_a_stacked, lora_b_stacked, topk_weights, expert_ids, token_lora_mapping, top_k_num, adapter_enabled, mul_routed_weight):
    """CPU reference: per-token shrink then expand with MoE expert routing.
    expert_ids contains the canonical flat route-to-expert mapping, regardless
    of whether the kernel launch uses naive or sorted assignment."""
    import torch
    M = topk_weights.shape[0]
    K = qcurr_hidden_states.shape[1]
    num_slices = len(lora_a_stacked)
    max_lora_rank = lora_a_stacked[0].shape[2]
    w1_out_dim = lora_b_stacked[0].shape[2]
    out_dim = num_slices * w1_out_dim
    num_tokens = M * top_k_num
    intermediate = torch.zeros(num_slices, M, top_k_num, max_lora_rank, dtype=torch.float32, device=qcurr_hidden_states.device)
    for token_idx in range(M):
        lora_id = token_lora_mapping[token_idx].item()
        if lora_id == -1:
            continue
        if adapter_enabled[lora_id].item() == 0:
            continue
        for k in range(top_k_num):
            flat_idx = token_idx * top_k_num + k
            exp_id = expert_ids[flat_idx].item()
            if exp_id == -1:
                continue
            input_idx = flat_idx if mul_routed_weight else token_idx
            inp = qcurr_hidden_states[input_idx].float()
            for s in range(num_slices):
                wa = lora_a_stacked[s][lora_id, exp_id].float()
                intermediate[s, token_idx, k] = inp @ wa.T
    output = torch.zeros(M, top_k_num, out_dim, dtype=qcurr_hidden_states.dtype, device=qcurr_hidden_states.device).float()
    for token_idx in range(M):
        lora_id = token_lora_mapping[token_idx].item()
        if lora_id == -1:
            continue
        if adapter_enabled[lora_id].item() == 0:
            continue
        for k in range(top_k_num):
            flat_idx = token_idx * top_k_num + k
            exp_id = expert_ids[flat_idx].item()
            if exp_id == -1:
                continue
            for s in range(num_slices):
                inter = intermediate[s, token_idx, k].float()
                wb = lora_b_stacked[s][lora_id, exp_id].float()
                result = inter @ wb.T
                if mul_routed_weight:
                    result *= topk_weights[token_idx, k].item()
                col_start = s * w1_out_dim
                col_end = col_start + w1_out_dim
                output[token_idx, k, col_start:col_end] += result
    return output.to(qcurr_hidden_states.dtype)

def _make_sorted_assignment(route_expert_ids, token_lora_mapping, num_loras, block_size, device):
    """Pack flat token routes into the block-sorted format consumed by the kernel."""
    import torch
    num_routes = route_expert_ids.numel()
    top_k = num_routes // token_lora_mapping.numel()
    rows = []
    expert_rows = []
    padded_counts = []
    route_experts_cpu = route_expert_ids.cpu().tolist()
    token_loras_cpu = token_lora_mapping.cpu().tolist()
    for lora_id in range(num_loras):
        row = []
        row_experts = []
        grouped_routes = {}
        for route_id, expert_id in enumerate(route_experts_cpu):
            token_id = route_id // top_k
            if token_loras_cpu[token_id] == lora_id:
                grouped_routes.setdefault(expert_id, []).append(route_id)
        for expert_id in sorted(grouped_routes):
            routes = grouped_routes[expert_id]
            for start in range(0, len(routes), block_size):
                block = routes[start:start + block_size]
                row.extend(block)
                row.extend([num_routes] * (block_size - len(block)))
                row_experts.append(expert_id)
        rows.append(row)
        expert_rows.append(row_experts)
        padded_counts.append(len(row))
    max_blocks = max(1, max((len(row) for row in expert_rows)))
    row_width = max_blocks * block_size
    sorted_token_ids = torch.full((num_loras, row_width), num_routes, dtype=torch.int64, device=device)
    sorted_expert_ids = torch.full((num_loras, max_blocks), -1, dtype=torch.int64, device=device)
    for lora_id, (row, row_experts) in enumerate(zip(rows, expert_rows)):
        if row:
            sorted_token_ids[lora_id, :len(row)] = torch.tensor(row, dtype=torch.int64, device=device)
            sorted_expert_ids[lora_id, :len(row_experts)] = torch.tensor(row_experts, dtype=torch.int64, device=device)
    num_tokens_post_padded = torch.tensor(padded_counts, dtype=torch.int32, device=device)
    return (sorted_token_ids, sorted_expert_ids, num_tokens_post_padded)

def make_correctness_data(case, device, seed):
    """Build a correctness-only case without changing benchmark inputs."""
    import torch
    M, K, num_experts, lora_rank, out_dim, num_loras, top_k = case['shape']
    num_slices = case['num_slices']
    mul_routed_weight = case['mul_routed_weight']
    dtype = getattr(torch, case['dtype'])
    torch.manual_seed(seed)
    input_rows = M * top_k if mul_routed_weight else M
    value_scale = 0.2
    qcurr = torch.randn(input_rows, K, device=device, dtype=dtype) * value_scale
    topk_weights = torch.linspace(0.2, 1.1, M * top_k, device=device, dtype=torch.float32).reshape(M, top_k)
    lora_a = [torch.randn(num_loras, num_experts, lora_rank, K, device=device, dtype=dtype) * value_scale for _ in range(num_slices)]
    lora_b = [torch.randn(num_loras, num_experts, out_dim, lora_rank, device=device, dtype=dtype) * value_scale for _ in range(num_slices)]
    route_expert_ids = torch.arange(M * top_k, device=device, dtype=torch.int64) % num_experts
    token_lora_mapping = torch.arange(M, device=device, dtype=torch.int64) % num_loras
    adapter_enabled = torch.ones(num_loras, device=device, dtype=torch.int32)
    if case['include_invalid_routes']:
        token_lora_mapping[0] = -1
        if num_loras > 1:
            token_lora_mapping[1] = 1
            adapter_enabled[1] = 0
        route_expert_ids[2 * top_k + (top_k - 1)] = -1
    if case['sorted_assignment']:
        sorted_token_ids, expert_ids, num_tokens_post_padded = _make_sorted_assignment(route_expert_ids, token_lora_mapping, num_loras, 64, device)
        active_ids = [num_loras - 1] + list(range(num_loras - 1))
        lora_ids = torch.tensor(active_ids, device=device, dtype=torch.int64)
        num_active_loras = len(active_ids)
    else:
        sorted_token_ids = None
        expert_ids = route_expert_ids
        num_tokens_post_padded = None
        lora_ids = torch.arange(num_loras, device=device, dtype=torch.int64)
        num_active_loras = num_loras
    offset = case['offset']
    target_width = num_slices * out_dim
    output_width = offset + target_width + (3 if offset else 0)
    output = torch.full((M, top_k, output_width), -0.75, device=device, dtype=dtype)
    output[:, :, offset:offset + target_width] = 0
    return {'output': output, 'qcurr': qcurr, 'lora_a': lora_a, 'lora_b': lora_b, 'topk_weights': topk_weights, 'sorted_token_ids': sorted_token_ids, 'expert_ids': expert_ids, 'reference_expert_ids': route_expert_ids, 'num_tokens_post_padded': num_tokens_post_padded, 'token_lora_mapping': token_lora_mapping, 'max_lora_rank': lora_rank, 'top_k_num': top_k, 'lora_ids': lora_ids, 'num_active_loras': num_active_loras, 'adapter_enabled': adapter_enabled}

def prepare_direct_launch(mod, output, qcurr_hidden_states, lora_a_stacked, lora_b_stacked, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded, token_lora_mapping, max_lora_rank, top_k_num, lora_ids, num_active_loras, adapter_enabled, mul_routed_weight=False, offset=0, shrink_split_k=1, use_b_l2_cache=True):
    """Prepare stable shrink/expand launches for graph-first benchmarking.

    The public wrapper allocates pointer tables and an intermediate tensor on
    every call. Those operations are deliberately hoisted here, along with all
    views, grids, strides, scalar arguments, and Triton meta-parameters. The
    returned callables only enqueue already-prepared device operations.
    """
    import torch
    assert len(lora_a_stacked) == len(lora_b_stacked) > 0
    assert topk_weights.dim() == qcurr_hidden_states.dim() == 2
    device = qcurr_hidden_states.device
    num_slices = len(lora_a_stacked)
    w1_lora_a_stacked = lora_a_stacked[0]
    w1_lora_b_stacked = lora_b_stacked[0]
    num_experts = w1_lora_a_stacked.shape[1]
    shrink_n = max_lora_rank
    num_tokens_base = topk_weights.shape[0]
    shrink_k = qcurr_hidden_states.shape[1]
    num_tokens = num_tokens_base * top_k_num
    output_dim = w1_lora_b_stacked.shape[2]
    shrink_block_size_m = 64
    shrink_block_size_n = min(64, mod._next_power_of_2(shrink_n))
    shrink_block_size_k = 32
    shrink_group_size_m = 8
    shrink_num_warps = 4
    shrink_num_stages = 3
    expand_block_size_m = 64
    expand_block_size_n = 64
    expand_block_size_k = max(16, min(32, mod._next_power_of_2(shrink_n)))
    expand_group_size_m = 8
    expand_num_warps = 4
    expand_num_stages = 3
    em = sorted_token_ids.shape[1] if sorted_token_ids is not None else num_tokens * shrink_block_size_m
    grid_lora_dim, stride_tl, stride_el = mod._adjust_kernel_inputs(num_active_loras, sorted_token_ids, expert_ids)
    grid_lora_dim2, stride_tl2, stride_el2 = mod._adjust_kernel_inputs(num_active_loras, sorted_token_ids, expert_ids)
    lora_a_ptrs = mod._get_ptr(lora_a_stacked, device)
    lora_b_ptrs = mod._get_ptr(lora_b_stacked, device)
    intermediate = torch.zeros((num_slices, num_tokens_base, top_k_num, max_lora_rank), dtype=output.dtype, device=device)
    intermediate_flat = intermediate.view(-1, intermediate.shape[3])
    out_view = output[:, :, offset:offset + num_slices * output_dim]

    def _ceil_div(value, divisor):
        return (value + divisor - 1) // divisor
    shrink_grid = (shrink_split_k * _ceil_div(em, shrink_block_size_m) * _ceil_div(shrink_n, shrink_block_size_n), num_slices, grid_lora_dim)
    expand_grid = (_ceil_div(em, expand_block_size_m) * _ceil_div(output_dim, expand_block_size_n), num_slices, grid_lora_dim2)
    shrink_args = (qcurr_hidden_states, lora_a_ptrs, intermediate, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded, token_lora_mapping, shrink_n, shrink_k, em, num_tokens, num_experts, top_k_num, lora_ids, adapter_enabled, w1_lora_a_stacked.shape[0], qcurr_hidden_states.stride(0), qcurr_hidden_states.stride(1), w1_lora_a_stacked.stride(0), w1_lora_a_stacked.stride(1), w1_lora_a_stacked.stride(3), w1_lora_a_stacked.stride(2), intermediate.stride(2), intermediate.stride(3), stride_tl, stride_el)
    shrink_meta = {'slice_a_size': qcurr_hidden_states.numel(), 'slice_c_size': intermediate.numel() // num_slices, 'num_slice_a': 1, 'num_slice_c': num_slices, 'token_mapping_factor': 1 if mul_routed_weight else top_k_num, 'naive_block_assignment': sorted_token_ids is None, 'MUL_ROUTED_WEIGHT': False, 'ADD_INPUTS': False, 'USE_B_L2_CACHE': use_b_l2_cache, 'IS_PRIMARY': True, 'BLOCK_SIZE_M': shrink_block_size_m, 'BLOCK_SIZE_N': shrink_block_size_n, 'BLOCK_SIZE_K': shrink_block_size_k, 'GROUP_SIZE_M': shrink_group_size_m, 'SPLIT_K': shrink_split_k, 'USE_GDC': False, 'launch_pdl': False, 'num_warps': shrink_num_warps, 'num_stages': shrink_num_stages}
    expand_args = (intermediate_flat, lora_b_ptrs, out_view, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded, token_lora_mapping, output_dim, max_lora_rank, em, num_tokens, num_experts, top_k_num, lora_ids, adapter_enabled, w1_lora_b_stacked.shape[0], intermediate_flat.stride(0), intermediate_flat.stride(1), w1_lora_b_stacked.stride(0), w1_lora_b_stacked.stride(1), w1_lora_b_stacked.stride(3), w1_lora_b_stacked.stride(2), out_view.stride(1), out_view.stride(2), stride_tl2, stride_el2)
    expand_meta = {'slice_a_size': intermediate_flat.numel() // num_slices, 'slice_c_size': output_dim * out_view.stride(2), 'num_slice_a': num_slices, 'num_slice_c': num_slices, 'token_mapping_factor': 1, 'naive_block_assignment': sorted_token_ids is None, 'MUL_ROUTED_WEIGHT': mul_routed_weight, 'ADD_INPUTS': True, 'USE_B_L2_CACHE': use_b_l2_cache, 'IS_PRIMARY': False, 'BLOCK_SIZE_M': expand_block_size_m, 'BLOCK_SIZE_N': expand_block_size_n, 'BLOCK_SIZE_K': expand_block_size_k, 'GROUP_SIZE_M': expand_group_size_m, 'SPLIT_K': 1, 'USE_GDC': False, 'launch_pdl': False, 'num_warps': expand_num_warps, 'num_stages': expand_num_stages}
    shrink_launcher = mod.fused_moe_lora_kernel[shrink_grid]
    expand_launcher = mod.fused_moe_lora_kernel[expand_grid]
    reset_output = output.zero_
    reset_intermediate = intermediate.zero_

    def launch_shrink():
        shrink_launcher(*shrink_args, **shrink_meta)

    def launch_expand():
        expand_launcher(*expand_args, **expand_meta)

    def launch_reset_expand():
        reset_output()
        launch_expand()

    def launch_fused_no_reset():
        launch_shrink()
        launch_expand()

    def launch_fused():
        reset_output()
        if shrink_split_k > 1:
            reset_intermediate()
        launch_fused_no_reset()
    return {'fused': launch_fused, 'fused_no_reset': launch_fused_no_reset, 'shrink': launch_shrink, 'reset_expand': launch_reset_expand, 'reset_output': reset_output, 'reset_intermediate': reset_intermediate, 'output': output, 'intermediate': intermediate, 'lora_a_ptrs': lora_a_ptrs, 'lora_b_ptrs': lora_b_ptrs}
EXTRA_CASES = [{'name': 'naive_irregular_slices_offset', 'shape': (7, 70, 5, 13, 45, 3, 3), 'num_slices': 2, 'offset': 5, 'dtype': 'float16', 'sorted_assignment': False, 'mul_routed_weight': False, 'include_invalid_routes': True}, {'name': 'sorted_routed_weight_bfloat16', 'shape': (9, 65, 3, 17, 33, 3, 3), 'num_slices': 1, 'offset': 0, 'dtype': 'bfloat16', 'sorted_assignment': True, 'mul_routed_weight': True, 'include_invalid_routes': True}, {'name': 'split_k2_no_l2_cache', 'shape': (5, 70, 3, 11, 37, 2, 2), 'num_slices': 1, 'offset': 0, 'dtype': 'float16', 'sorted_assignment': False, 'mul_routed_weight': False, 'include_invalid_routes': False, 'shrink_split_k': 2, 'use_b_l2_cache': False}]

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
        try:
            data = make_correctness_data(case, device, 100 + i)
            with _readonly_control(data):
                output = data['output']
                offset = case['offset']
                ref = reference_fused_moe_lora(data['qcurr'], data['lora_a'], data['lora_b'], data['topk_weights'], data['reference_expert_ids'], data['token_lora_mapping'], data['top_k_num'], data['adapter_enabled'], case['mul_routed_weight']).to(device)
                target_width = ref.shape[2]
                expected = output.clone()
                expected[:, :, offset:offset + target_width] = ref
                mod.fused_moe_lora(output, data['qcurr'], data['lora_a'], data['lora_b'], data['topk_weights'], data['sorted_token_ids'], data['expert_ids'], data['num_tokens_post_padded'], data['token_lora_mapping'], data['max_lora_rank'], data['top_k_num'], data['lora_ids'], data['num_active_loras'], data['adapter_enabled'], mul_routed_weight=case['mul_routed_weight'], offset=offset)
                torch.cuda.synchronize()
                if not torch.allclose(output.float(), expected.float(), atol=0.05, rtol=0.05):
                    max_diff = (output.float() - expected.float()).abs().max().item()
                    return (False, f'Case {name}: wrapper max diff = {max_diff:.6f}')
                direct = prepare_direct_launch(mod, output, data['qcurr'], data['lora_a'], data['lora_b'], data['topk_weights'], data['sorted_token_ids'], data['expert_ids'], data['num_tokens_post_padded'], data['token_lora_mapping'], data['max_lora_rank'], data['top_k_num'], data['lora_ids'], data['num_active_loras'], data['adapter_enabled'], mul_routed_weight=case['mul_routed_weight'], offset=offset, shrink_split_k=case.get('shrink_split_k', 1), use_b_l2_cache=case.get('use_b_l2_cache', True))
                direct['fused']()
                torch.cuda.synchronize()
                direct_expected = torch.zeros_like(output)
                direct_expected[:, :, offset:offset + target_width] = ref
                if not torch.allclose(output.float(), direct_expected.float(), atol=0.05, rtol=0.05):
                    max_diff = (output.float() - direct_expected.float()).abs().max().item()
                    return (False, f'Case {name}: direct max diff = {max_diff:.6f}')
        except Exception as e:
            return (False, f'Case {name}: exception: {e}')
    return (True, None)

from contextlib import contextmanager


@contextmanager
def _readonly_control(data):
    """Guard wrapper and direct-JIT branch inputs against contamination."""
    import torch
    def walk(value):
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, (tuple, list)):
            for item in value:
                yield from walk(item)
    values = [value for key, item in data.items() if key != 'output'
              for value in walk(item)]
    saved = [value.clone() for value in values]
    try:
        yield
        for value, initial in zip(values, saved):
            if (value.shape != initial.shape or value.dtype != initial.dtype
                    or value.device != initial.device or not torch.equal(value, initial)):
                raise AssertionError('MoE LoRA control modified read-only input')
    finally:
        for value, initial in zip(values, saved):
            value.copy_(initial)
