"""Additional unscored public-branch controls from PR105.

Source revision: 0acf65b3a967ef1025dbfc5fd4b415b259e3bd43
Source path: tasks/triton2triton/vllm/triton_lora_shrink/scripts/task_runner.py
Original scored inputs, gates and timing remain in scripts/task_runner.py.
"""
import sys, os, json, argparse, importlib.util
TEST_SHAPES = [(16, 64, 8, 2, 1), (32, 128, 16, 4, 1), (64, 256, 16, 4, 2), (128, 512, 32, 8, 1), (256, 1024, 32, 8, 2)]
CORRECTNESS_CASES = [{'name': f'default_{i + 1}', 'shape': shape} for i, shape in enumerate(TEST_SHAPES)] + [{'name': 'mixed_split_first_partial', 'shape': (128, 33, 16, 4, 1)}, {'name': 'mixed_split_last_inactive_bfloat16_4d_inactive_lora', 'shape': (128, 223, 16, 4, 2), 'dtype': 'bfloat16', 'weight_ndim': 4, 'include_inactive_lora': True}]

def reference_lora_shrink(inputs, lora_a_weights, token_indices, num_tokens_per_lora, lora_token_start_loc, lora_ids, scaling):
    """CPU reference for LoRA shrink (A) operation."""
    import torch
    num_slices = len(lora_a_weights)
    M = inputs.shape[0]
    lora_rank = lora_a_weights[0].shape[-2]
    output = torch.zeros(num_slices, M, lora_rank, device=inputs.device, dtype=torch.float32)
    for lora_idx in range(lora_ids.shape[0]):
        lora_id = lora_ids[lora_idx].item()
        if lora_id == -1:
            continue
        n_tokens = num_tokens_per_lora[lora_idx].item()
        start = lora_token_start_loc[lora_idx].item()
        for t in range(n_tokens):
            token_id = token_indices[start + t].item()
            for s in range(num_slices):
                w = lora_a_weights[s]
                if w.ndim == 4:
                    w = w.squeeze(1)
                inp = inputs[token_id].float()
                weight = w[lora_id].float()
                out_row = inp @ weight.T
                output[s, token_id] = out_row * scaling
    return output.to(inputs.dtype)

def make_test_data(M, hidden_size, lora_rank, num_loras, num_slices, device, seed, dtype='float16', weight_ndim=3, include_inactive_lora=False):
    import torch
    torch.manual_seed(seed)
    tensor_dtype = getattr(torch, dtype)
    inputs = torch.randn(M, hidden_size, device=device, dtype=tensor_dtype) * 0.1
    lora_a_weights = []
    for _ in range(num_slices):
        weight_shape = (num_loras, lora_rank, hidden_size)
        if weight_ndim == 4:
            weight_shape = (num_loras, 1, lora_rank, hidden_size)
        w = torch.randn(*weight_shape, device=device, dtype=tensor_dtype) * 0.1
        lora_a_weights.append(w)
    output_tensor = torch.zeros(num_slices, M, lora_rank, device=device, dtype=torch.float32)
    token_lora_mapping = torch.randint(0, num_loras, (M,), device=device, dtype=torch.int64)
    lora_ids_list = list(range(num_loras))
    if include_inactive_lora:
        token_lora_mapping[0] = -1
        lora_ids_list.insert(0, -1)
    lora_ids = torch.tensor(lora_ids_list, device=device, dtype=torch.int64)
    sorted_indices = []
    num_tokens_list = []
    for lid in lora_ids_list:
        mask = token_lora_mapping == lid
        indices = mask.nonzero(as_tuple=True)[0]
        sorted_indices.append(indices)
        num_tokens_list.append(len(indices))
    token_indices_sorted = torch.cat(sorted_indices).to(device)
    num_tokens_per_lora = torch.tensor(num_tokens_list, device=device, dtype=torch.int64)
    cumsum = [0]
    for n in num_tokens_list:
        cumsum.append(cumsum[-1] + n)
    lora_token_start_loc = torch.tensor(cumsum, device=device, dtype=torch.int64)
    num_active_loras = len(lora_ids_list)
    scaling = 0.5
    return (inputs, lora_a_weights, output_tensor, token_lora_mapping, token_indices_sorted, num_tokens_per_lora, lora_token_start_loc, lora_ids, num_active_loras, scaling)
EXTRA_CASES = [{'name': 'mixed_split_first_partial', 'shape': (128, 33, 16, 4, 1)}, {'name': 'mixed_split_last_inactive_bfloat16_4d_inactive_lora', 'shape': (128, 223, 16, 4, 2), 'dtype': 'bfloat16', 'weight_ndim': 4, 'include_inactive_lora': True}]

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
        M, hidden_size, lora_rank, num_loras, num_slices = case['shape']
        try:
            inputs, lora_a_weights, output_tensor, token_lora_mapping, token_indices_sorted, num_tokens_per_lora, lora_token_start_loc, lora_ids, num_active_loras, scaling = make_test_data(M, hidden_size, lora_rank, num_loras, num_slices, device, 42 + i, dtype=case.get('dtype', 'float16'), weight_ndim=case.get('weight_ndim', 3), include_inactive_lora=case.get('include_inactive_lora', False))
            mod.lora_shrink(inputs, lora_a_weights, output_tensor, token_lora_mapping, token_indices_sorted, num_tokens_per_lora, lora_token_start_loc, lora_ids, num_active_loras, scaling)
            torch.cuda.synchronize()
            ref = reference_lora_shrink(inputs, lora_a_weights, token_indices_sorted, num_tokens_per_lora, lora_token_start_loc, lora_ids, scaling).to(device)
            if not torch.allclose(output_tensor.float(), ref.float(), atol=0.05, rtol=0.05):
                max_diff = (output_tensor.float() - ref.float()).abs().max().item()
                return (False, f'Case {case['name']} (M={M}, K={hidden_size}): max diff = {max_diff:.6f}')
        except Exception as e:
            return (False, f'Case {case['name']}: exception: {e}')
    return (True, None)
