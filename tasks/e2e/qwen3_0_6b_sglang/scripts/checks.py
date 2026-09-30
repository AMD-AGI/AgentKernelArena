"""Task-owned RMSNorm numerical oracle, independent of candidate kernels."""
import ast
import importlib.util
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def source_policy(*, target_only=False):
    """Reject ordinary configuration/file/dispatch edits in a kernel submission.

    This is a narrow source contract, not a Python security sandbox. Runtime
    settings and rank evidence are checked separately by the protected adapter.
    """
    tree = ast.parse((ROOT / 'source/rmsnorm.py').read_text())
    allowed = {'torch', 'triton', 'aiter', 'math', 'typing', 'functools'}
    if target_only:
        allowed.remove('aiter')
    forbidden = {'open', 'eval', 'exec', 'compile', '__import__', 'getattr', 'setattr',
                 'globals', 'locals', 'vars', 'breakpoint'}
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases.update({item.asname or item.name: item.name for item in node.names})
        elif isinstance(node, ast.ImportFrom) and node.module:
            aliases.update({item.asname or item.name: node.module+'.'+item.name for item in node.names})
    def qualified(node):
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return qualified(node.value)+'.'+node.attr
        return ''
    allocation_and_metadata = {'torch.empty', 'torch.empty_like', 'torch.empty_strided',
        'torch.zeros', 'torch.zeros_like', 'torch.finfo', 'torch.iinfo', 'torch.device',
        'torch.is_tensor', 'torch.cuda.current_device', 'torch.cuda.get_device_properties'}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or '']
            if any(name.split('.')[0] not in allowed for name in names):
                raise ValueError('Kernel source imported an undeclared dependency')
        if isinstance(node, ast.Name) and node.id in forbidden:
            raise ValueError('Kernel source uses a prohibited runtime access')
        if isinstance(node, ast.Attribute) and (node.attr.startswith(('_', 'set_', 'enable_', 'disable_')) or isinstance(node.ctx, (ast.Store, ast.Del))):
            raise ValueError('Kernel source changes runtime attributes or uses reflective access')
        if target_only and isinstance(node, ast.Call):
            name = qualified(node.func)
            if name.startswith('torch.') and name not in allocation_and_metadata:
                raise ValueError('Torch is limited to allocation and metadata; implement computation in Triton')


def operator_checks(lock):
    source_policy()
    import torch
    torch.manual_seed(lock['correctness']['seed'])
    spec = importlib.util.spec_from_file_location('candidate', ROOT / 'source/rmsnorm.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    evidence = []
    for rows in (1, 7, 8, 64, 128, 1024, 2048, 16384):
        for width in (128, 1024):
            x = torch.randn((rows, width), device='cuda', dtype=torch.float16)
            residual = torch.randn_like(x)
            weight = torch.randn(width, device='cuda', dtype=x.dtype)
            for eps in (1e-6, 1e-5):
                def reference(value):
                    value = value.float()
                    return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype)
                original, wcopy = x.clone(), weight.clone()
                expected = reference(x)
                actual = module.rms_norm(x, weight, eps)
                torch.testing.assert_close(weight, wcopy, rtol=0, atol=0)
                torch.testing.assert_close(actual, expected, rtol=lock['correctness']['rtol'], atol=lock['correctness']['atol'])
                torch.testing.assert_close(x, original, rtol=0, atol=0)
                output, added = torch.empty_like(x), torch.empty_like(x)
                rcopy = residual.clone()
                # Construct the oracle before calling editable code.
                expected_added = (x.float() + residual.float()).to(x.dtype)
                expected_norm = reference(expected_added)
                module.fused_add_rms_norm(output, x, residual, added, weight, eps)
                torch.testing.assert_close(weight, wcopy, rtol=0, atol=0)
                torch.testing.assert_close(added, expected_added, rtol=0, atol=0)
                torch.testing.assert_close(output, expected_norm, rtol=lock['correctness']['rtol'], atol=lock['correctness']['atol'])
                torch.testing.assert_close(residual, rcopy, rtol=0, atol=0)
                torch.testing.assert_close(x, original, rtol=0, atol=0)
                evidence.append(dict(shape=[rows, width], eps=eps))
    torch.cuda.synchronize()
    return evidence



def target_kernels():
    tree = ast.parse((ROOT/'source/rmsnorm.py').read_text())
    aliases = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            aliases.update({item.asname or item.name: item.name for item in node.names})
        if isinstance(node, ast.ImportFrom) and node.module:
            aliases.update({item.asname or item.name: node.module+'.'+item.name for item in node.names})
    def qualified(node):
        if isinstance(node, ast.Call):
            return qualified(node.func)
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return qualified(node.value)+'.'+node.attr
        return ''
    names = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)
             and any(qualified(d) == 'triton.jit' for d in node.decorator_list)]
    if not names:
        raise ValueError('Final candidate must implement and execute a Triton GPU kernel')
    return names


def compile_candidate(lock):
    import torch
    names = target_kernels()
    cases = operator_checks(lock)
    module = sys.modules['candidate']
    x = torch.randn((8, 1024), device='cuda', dtype=torch.float16)
    weight = torch.ones(1024, device='cuda', dtype=x.dtype)
    inputs = {'rms_norm': (x, weight, 1e-6),
              'fused_add_rms_norm': (torch.empty_like(x), x, torch.randn_like(x),
                                     torch.empty_like(x), weight, 1e-6)}
    observed = {}
    for function, args in inputs.items():
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as profile:
            getattr(module, function)(*args)
            torch.cuda.synchronize()
        events = sorted({event.name for event in profile.events()
                         if event.device_type == torch.autograd.DeviceType.CUDA})
        if not any(name in event for name in names for event in events):
            raise ValueError(f'{function} did not execute a declared candidate Triton kernel')
        observed[function] = events
    return dict(operator_cases=cases, candidate_gpu_kernels=observed)
