"""Unscored PR105 control inputs; original performance remains unchanged.
Source revision: 0acf65b3a967ef1025dbfc5fd4b415b259e3bd43
Source path: tasks/triton2triton/vllm/triton_ssd_chunk_cumsum/scripts/task_runner.py
"""
import sys, os, json, argparse, importlib.util
import math
CORRECTNESS_EDGE_CASES = [{'name': 'variable_chunks_odd_heads_clamped', 'seqlen': 121, 'nheads': 7, 'chunk_size': 48, 'cu_chunk_seqlens': (0, 48, 79, 121), 'has_bias': True, 'softplus': False, 'dt_limit': (0.01, 0.06), 'dtype': 'float32', 'strided': False}, {'name': 'short_chunks_strided_float16', 'seqlen': 67, 'nheads': 5, 'chunk_size': 33, 'cu_chunk_seqlens': (0, 1, 34, 58, 67), 'has_bias': True, 'softplus': True, 'dt_limit': (0.0, float('inf')), 'dtype': 'float16', 'strided': True}]

def ref_softplus(x):
    import torch
    return torch.where(x <= 20.0, torch.log1p(torch.exp(x)), x)

def reference(dt, A, chunk_size, cu, dt_bias, softplus, dt_limit=(0.0, float('inf'))):
    import torch
    seqlen, nheads = dt.shape
    nchunks = len(cu) - 1
    dt_f = dt.cpu().float()
    A_f = A.cpu().float()
    dt_out = torch.zeros(nheads, nchunks, chunk_size, dtype=torch.float32)
    dA_cumsum = torch.zeros(nheads, nchunks, chunk_size, dtype=torch.float32)
    cu_cpu = cu.cpu()
    for c in range(nchunks):
        s, e = (cu_cpu[c].item(), cu_cpu[c + 1].item())
        clen = e - s
        for h in range(nheads):
            dt_chunk = dt_f[s:e, h].clone()
            if dt_bias is not None:
                dt_chunk += dt_bias.cpu().float()[h]
            if softplus:
                dt_chunk = ref_softplus(dt_chunk)
            dt_chunk = dt_chunk.clamp(min=dt_limit[0], max=dt_limit[1])
            dt_out[h, c, :clen] = dt_chunk
            dA = dt_out[h, c] * A_f[h]
            dA_cumsum[h, c] = torch.cumsum(dA, 0)
    return (dA_cumsum, dt_out)
EXTRA_CASES = [{'name': 'variable_chunks_odd_heads_clamped', 'lengths': [0, 48, 79, 121], 'heads': 7, 'chunk_size': 48, 'limits': [0.01, 0.06]}, {'name': 'short_chunks_strided_float16', 'lengths': [0, 1, 34, 58, 67], 'heads': 5, 'chunk_size': 33}]

def run_control(index, load_module):
    if type(index) is not int or not 0 <= index < len(EXTRA_CASES):
        raise ValueError('Unknown upstream correctness control')
    import torch
    try:
        mod = load_module()
        mod = _guarded_module(mod, 'chunk_cumsum_fwd', ())
    except Exception as e:
        return (False, f'Load failed: {e}')
    device = 'cuda'
    cases = []
    cases.extend(CORRECTNESS_EDGE_CASES)
    for i, case in enumerate(cases):
        if i + 0 != index:
            continue
        try:
            seqlen = case['seqlen']
            nheads = case['nheads']
            chunk_size = case['chunk_size']
            has_bias = case['has_bias']
            softplus = case['softplus']
            dt_limit = case['dt_limit']
            dtype = getattr(torch, case['dtype'])
            torch.manual_seed(47 + i)
            if case['strided']:
                dt_storage = torch.randn(seqlen * 2, nheads * 2, device=device, dtype=dtype) * 0.1
                dt = dt_storage[::2, ::2]
                A_storage = -torch.rand(nheads * 2, device=device, dtype=torch.float32) * 0.5
                A = A_storage[::2]
                if has_bias:
                    dt_bias_storage = torch.randn(nheads * 2, device=device, dtype=torch.float32) * 0.01
                    dt_bias = dt_bias_storage[::2]
                else:
                    dt_bias = None
            else:
                dt = torch.randn(seqlen, nheads, device=device, dtype=dtype) * 0.1
                A = -torch.rand(nheads, device=device, dtype=torch.float32) * 0.5
                dt_bias = torch.randn(nheads, device=device, dtype=torch.float32) * 0.01 if has_bias else None
            cu = torch.tensor(case['cu_chunk_seqlens'], device=device, dtype=torch.int32)
            dA_cs, dt_out = mod.chunk_cumsum_fwd(dt, A, chunk_size, cu, dt_bias=dt_bias, dt_softplus=softplus, dt_limit=dt_limit)
            ref_dA, ref_dt = reference(dt, A, chunk_size, cu, dt_bias, softplus, dt_limit)
            ref_dA = ref_dA.to(device)
            ref_dt = ref_dt.to(device)
            if not torch.allclose(dA_cs, ref_dA, atol=0.001, rtol=0.001):
                diff = (dA_cs - ref_dA).abs().max().item()
                return (False, f'{case['name']} dA_cumsum: max diff={diff}')
            if not torch.allclose(dt_out, ref_dt, atol=0.001, rtol=0.001):
                diff = (dt_out - ref_dt).abs().max().item()
                return (False, f'{case['name']} dt_out: max diff={diff}')
        except Exception as e:
            return (False, f'{case['name']}: {e}')
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
