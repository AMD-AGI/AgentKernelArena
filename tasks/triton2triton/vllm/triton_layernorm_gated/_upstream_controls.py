"""Unscored PR105 public-branch controls, ported from pinned main.

The original runner still owns every existing correctness and performance path.
These additional calls use the current checked candidate loader. No benchmark
helper or original workload is replaced.
Source revision: 0acf65b3a967ef1025dbfc5fd4b415b259e3bd43
Source path: tasks/triton2triton/vllm/triton_layernorm_gated/scripts/task_runner.py
"""
import sys, os, json, argparse, importlib.util
TEST_SHAPES = [(32, 128, False, True, False), (64, 256, True, False, False), (128, 512, True, True, True), (256, 1024, False, True, True), (512, 2048, True, False, True)]
CORRECTNESS_CASES = [{'name': 'norm_before_gate_false_fp32_tail_out', 'M': 7, 'N': 130, 'is_rms': False, 'has_bias': True, 'has_z': True, 'dtype': 'float32', 'group_size': None, 'norm_before_gate': False, 'explicit_out': True}, {'name': 'grouped_bf16_tail', 'M': 5, 'N': 390, 'is_rms': False, 'has_bias': False, 'has_z': False, 'dtype': 'bfloat16', 'group_size': 130, 'norm_before_gate': True, 'explicit_out': False}]

def reference(x, weight, bias, eps, z, is_rms, group_size=None, norm_before_gate=True):
    import torch
    M, N = x.shape
    if group_size is None:
        group_size = N
    ngroups = N // group_size
    x_f = x.float().reshape(M, ngroups, group_size)
    if z is not None and (not norm_before_gate):
        z_f = z.float().reshape(M, ngroups, group_size)
        x_f = x_f * z_f * torch.sigmoid(z_f)
    if is_rms:
        mean = None
        var = (x_f ** 2).mean(-1, keepdim=True)
        x_hat = x_f * torch.rsqrt(var + eps)
    else:
        grouped_mean = x_f.mean(-1, keepdim=True)
        var = ((x_f - grouped_mean) ** 2).mean(-1, keepdim=True)
        x_hat = (x_f - grouped_mean) * torch.rsqrt(var + eps)
        mean = grouped_mean.squeeze(-1).T.contiguous().flatten()
    rstd = torch.rsqrt(var + eps).squeeze(-1).T.contiguous().flatten()
    y = x_hat * weight.float().reshape(ngroups, group_size)
    if bias is not None:
        y = y + bias.float().reshape(ngroups, group_size)
    if z is not None and norm_before_gate:
        z_f = z.float().reshape(M, ngroups, group_size)
        y = y * z_f * torch.sigmoid(z_f)
    return (y.reshape(M, N).to(x.dtype), mean, rstd)
EXTRA_CASES = [{'name': 'norm_before_gate_false_fp32_tail_out', 'M': 7, 'N': 130, 'is_rms': False, 'has_bias': True, 'has_z': True, 'dtype': 'float32', 'group_size': None, 'norm_before_gate': False, 'explicit_out': True}, {'name': 'grouped_bf16_tail', 'M': 5, 'N': 390, 'is_rms': False, 'has_bias': False, 'has_z': False, 'dtype': 'bfloat16', 'group_size': 130, 'norm_before_gate': True, 'explicit_out': False}]

def run_control(index, load_module):
    if type(index) is not int or not 0 <= index < len(EXTRA_CASES):
        raise ValueError('Unknown upstream correctness control')
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return (False, f'Load failed: {e}')
    device = 'cuda'
    cases = [{'name': f'base_{i}', 'M': M, 'N': N, 'is_rms': is_rms, 'has_bias': has_bias, 'has_z': has_z, 'dtype': 'float16', 'group_size': None, 'norm_before_gate': True, 'explicit_out': False} for i, (M, N, is_rms, has_bias, has_z) in enumerate(TEST_SHAPES)] + CORRECTNESS_CASES
    for i, case in enumerate(cases):
        if i != index + 5:
            continue
        try:
            torch.manual_seed(42 + i)
            dtype = getattr(torch, case['dtype'])
            x = torch.randn(case['M'], case['N'], device=device, dtype=dtype)
            w = torch.randn(case['N'], device=device, dtype=dtype)
            b = torch.randn(case['N'], device=device, dtype=dtype) if case['has_bias'] else None
            z = torch.randn_like(x) if case['has_z'] else None
            supplied_out = torch.full_like(x, torch.nan) if case['explicit_out'] else None
            eps = 1e-05
            out, mean, rstd = mod.layer_norm_fwd(x, w, b, eps, z=z, out=supplied_out, group_size=case['group_size'], norm_before_gate=case['norm_before_gate'], is_rms_norm=case['is_rms'])
            ref_out, ref_mean, ref_rstd = reference(x, w, b, eps, z, case['is_rms'], group_size=case['group_size'], norm_before_gate=case['norm_before_gate'])
            if supplied_out is not None and out.data_ptr() != supplied_out.data_ptr():
                return (False, f'{case['name']}: did not return the supplied out buffer')
            if not torch.allclose(out, ref_out, atol=0.01, rtol=0.01):
                diff = (out - ref_out).abs().max().item()
                return (False, f'{case['name']}: output max diff={diff}')
            if case['is_rms']:
                if mean is not None:
                    return (False, f'{case['name']}: RMSNorm returned a mean buffer')
            else:
                if mean is None:
                    return (False, f'{case['name']}: LayerNorm did not return mean')
                if mean.shape != ref_mean.shape or mean.dtype != ref_mean.dtype:
                    return (False, f'{case['name']}: mean metadata mismatch')
                if not torch.allclose(mean, ref_mean, atol=1e-05, rtol=0.0001):
                    diff = (mean - ref_mean).abs().max().item()
                    return (False, f'{case['name']}: mean max diff={diff}')
            if rstd.shape != ref_rstd.shape or rstd.dtype != ref_rstd.dtype:
                return (False, f'{case['name']}: rstd metadata mismatch')
            if not torch.allclose(rstd, ref_rstd, atol=1e-05, rtol=0.0001):
                diff = (rstd - ref_rstd).abs().max().item()
                return (False, f'{case['name']}: rstd max diff={diff}')
        except Exception as e:
            return (False, f'{case['name']}: {e}')
    return (True, None)
