"""Additional unscored public-branch controls from PR105.

Source revision: 0acf65b3a967ef1025dbfc5fd4b415b259e3bd43
Source path: tasks/triton2triton/vllm/triton_silu_mul_fp8_quant_dg/scripts/task_runner.py
Original scored inputs, gates and timing remain in scripts/task_runner.py.
"""
import sys, os, json, argparse, importlib.util
TEST_SHAPES = [(4, 16, 128, 128), (4, 32, 256, 128), (8, 32, 512, 128), (8, 64, 1024, 128), (16, 64, 1024, 128)]
CORRECTNESS_CASES = [(*shape, 'float16', False) for shape in TEST_SHAPES] + [(4, 9, 320, 64, 'bfloat16', True), (4, 7, 768, 256, 'float16', False)]

def reference_silu_mul_fp8(y, tokens_per_expert, group_size, fp8_dtype):
    """CPU reference for the activation, scales, and quantized output."""
    import torch
    E, T, H2 = y.shape
    H = H2 // 2
    G = (H + group_size - 1) // group_size
    fp8_max = torch.finfo(fp8_dtype).max
    results_float = torch.zeros(E, T, H, dtype=torch.float32)
    results_q = torch.zeros(E, T, H, dtype=fp8_dtype)
    results_s = torch.zeros(E, T, G, dtype=torch.float32)
    for e in range(E):
        nt = tokens_per_expert[e].item()
        if nt == 0:
            continue
        gate = y[e, :nt, :H].float()
        up = y[e, :nt, H:].float()
        results_float[e, :nt] = gate * torch.sigmoid(gate) * up
        for g in range(G):
            start = g * group_size
            end = min(start + group_size, H)
            values = results_float[e, :nt, start:end]
            scale = values.abs().amax(dim=-1).clamp_min(1e-10) / fp8_max
            results_s[e, :nt, g] = scale
            results_q[e, :nt, start:end] = torch.clamp(values / scale[:, None], torch.finfo(fp8_dtype).min, fp8_max).to(fp8_dtype)
    return (results_float, results_q, results_s)
EXTRA_CASES = [(4, 9, 320, 64, 'bfloat16', True), (4, 7, 768, 256, 'float16', False)]

def run_control(index, load_module):
    if type(index) is not int or not 0 <= index < len(EXTRA_CASES):
        raise ValueError('Unknown upstream correctness control')
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return (False, f'Failed to load module: {e}')
    device = 'cuda'
    try:
        expected_fp8_dtype = torch.float8_e4m3fnuz
        _ = torch.tensor([1.0]).to(expected_fp8_dtype)
    except (RuntimeError, AttributeError):
        expected_fp8_dtype = torch.float8_e4m3fn
    for i, case in enumerate(EXTRA_CASES, start=5):
        if i != index + 5:
            continue
        E, T, H, group_size, dtype_name, non_contiguous = case
        try:
            torch.manual_seed(42 + i)
            input_dtype = getattr(torch, dtype_name)
            if non_contiguous:
                backing = torch.randn(T, E, 4 * H, device=device, dtype=input_dtype) * 0.5
                y = backing.permute(1, 0, 2)[..., ::2]
                if y.is_contiguous():
                    return (False, f'Shape {i + 1}: input unexpectedly became contiguous')
            else:
                y = torch.randn(E, T, 2 * H, device=device, dtype=input_dtype) * 0.5
            boundary_counts = torch.tensor([0, 1, T - 1, T], device=device, dtype=torch.int32)
            tokens_per_expert = boundary_counts.repeat((E + 3) // 4)[:E]
            y_q, y_s = mod.silu_mul_fp8_quant(y, tokens_per_expert, group_size)
            torch.cuda.synchronize()
            G = (H + group_size - 1) // group_size
            if y_q.shape != (E, T, H) or y_s.shape != (E, T, G):
                return (False, f'Shape {i + 1}: output shapes are {tuple(y_q.shape)} and {tuple(y_s.shape)}, expected {(E, T, H)} and {(E, T, G)}')
            if y_q.dtype != expected_fp8_dtype:
                return (False, f'Shape {i + 1}: quantized dtype is {y_q.dtype}, expected {expected_fp8_dtype}')
            if y_s.dtype != torch.float32:
                return (False, f'Shape {i + 1}: scale dtype is {y_s.dtype}, expected torch.float32')
            ref_float, ref_q, ref_s = reference_silu_mul_fp8(y.cpu(), tokens_per_expert.cpu(), group_size, expected_fp8_dtype)
            valid_tokens = torch.arange(T)[None, :] < tokens_per_expert.cpu()[:, None]
            actual_q = y_q.float().cpu()[valid_tokens]
            expected_q = ref_q.float()[valid_tokens]
            q_mismatch = actual_q != expected_q
            if torch.any(q_mismatch):
                mismatch_count = q_mismatch.sum().item()
                max_diff = (actual_q - expected_q).abs().max().item()
                return (False, f'Shape {i + 1}: y_q has {mismatch_count} mismatches (max_diff={max_diff:.4f})')
            actual_s = y_s.cpu()[valid_tokens]
            expected_s = ref_s[valid_tokens]
            if not torch.allclose(actual_s, expected_s, atol=1e-08, rtol=1e-05):
                max_diff = (actual_s - expected_s).abs().max().item()
                return (False, f'Shape {i + 1}: y_s max_diff={max_diff:.4e}')
            expanded_s = y_s.cpu().repeat_interleave(group_size, dim=-1)[..., :H]
            deq = y_q.float().cpu() * expanded_s
            if not torch.allclose(deq[valid_tokens], ref_float[valid_tokens], atol=0.5, rtol=0.2):
                max_diff = (deq[valid_tokens] - ref_float[valid_tokens]).abs().max().item()
                return (False, f'Shape {i + 1}: dequantized output max_diff={max_diff:.4f}')
        except Exception as e:
            return (False, f'Shape {i + 1}: exception: {e}')
    return (True, None)
