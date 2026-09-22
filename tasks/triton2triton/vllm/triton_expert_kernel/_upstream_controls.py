"""Unscored PR105 public-branch controls, ported from pinned main.

The original runner still owns every existing correctness and performance path.
These additional calls use the current checked candidate loader. No benchmark
helper or original workload is replaced.
Source revision: 0acf65b3a967ef1025dbfc5fd4b415b259e3bd43
Source path: tasks/triton2triton/vllm/triton_expert_kernel/scripts/task_runner.py
"""
import sys, os, json, argparse, importlib.util
TEST_SHAPES = [(32, 64, 32), (64, 128, 64), (128, 256, 128), (256, 512, 256), (512, 1024, 512)]
CORRECTNESS_CASES = [*[(f'aligned_{M}x{K}x{N}', M, K, N, 'float16', 'contiguous') for M, K, N in TEST_SHAPES], ('sub_tile_tails', 7, 13, 11, 'float16', 'contiguous'), ('multi_tile_tails_wide', 65, 97, 129, 'float16', 'contiguous'), ('strided_bfloat16_tails', 73, 45, 19, 'bfloat16', 'strided')]
EXTRA_CASES = [('sub_tile_tails', 7, 13, 11, 'float16', 'contiguous'), ('multi_tile_tails_wide', 65, 97, 129, 'float16', 'contiguous'), ('strided_bfloat16_tails', 73, 45, 19, 'bfloat16', 'strided')]

def run_control(index, load_module):
    if type(index) is not int or not 0 <= index < len(EXTRA_CASES):
        raise ValueError('Unknown upstream correctness control')
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return (False, f'Failed to load module: {e}')
    device = 'cuda'
    for i, (case_name, M, K, N, dtype_name, layout) in enumerate(CORRECTNESS_CASES):
        if i != index + 5:
            continue
        try:
            torch.manual_seed(42 + i)
            dtype = getattr(torch, dtype_name)
            if layout == 'strided':
                A_storage = torch.randn(M, 2 * K, device=device, dtype=dtype) * 0.1
                B_storage = torch.randn(2 * N, K, device=device, dtype=dtype) * 0.1
                A = A_storage[:, ::2]
                B = B_storage[::2, :].T
                assert not A.is_contiguous() and (not B.is_contiguous())
            else:
                A = torch.randn(M, K, device=device, dtype=dtype) * 0.1
                B = torch.randn(K, N, device=device, dtype=dtype) * 0.1
            result = mod.expert_gemm(A, B)
            torch.cuda.synchronize()
            ref = (A.float() @ B.float()).to(dtype)
            if result.shape != ref.shape or result.dtype != ref.dtype:
                return (False, f'{case_name}: expected shape/dtype {ref.shape}/{ref.dtype}, got {result.shape}/{result.dtype}')
            if not torch.allclose(result.float(), ref.float(), atol=0.05, rtol=0.05):
                max_diff = (result.float() - ref.float()).abs().max().item()
                return (False, f'{case_name}: max diff = {max_diff:.6f}')
        except Exception as e:
            return (False, f'{case_name}: exception: {e}')
    return (True, None)
