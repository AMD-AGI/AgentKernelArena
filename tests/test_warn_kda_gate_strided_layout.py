"""CPU pointer-model negatives and protected-wrapper layout controls."""

from pathlib import Path
import subprocess
import sys

import pytest


TASK = Path(__file__).resolve().parents[1] / "tasks/triton2triton/vllm/triton_kda_gate"


def _run(code):
    result = subprocess.run([sys.executable, "-c", code], cwd=TASK,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("fault", [None, "g", "A", "g_bias", "storage_hole",
                                    "signed_zero_view", "signed_zero_hole"])
def test_strided_known_answers_reject_flat_pointer_reads_and_storage_writes(fault):
    code = r'''
from types import SimpleNamespace
import torch
from scripts import task_runner as h, semantic_controls as c
from scripts.contract_checks import ContractFailure, NumericalMismatch

cases = list(c.control_cases('cpu'))
assert len(cases) == 8
assert [x['name'] for x in cases[-3:]] == [
    'strided_g_h4_d32', 'strided_A_h4_d32', 'strided_g_bias_h4_d32']
for case in cases[-3:]:
    assert len(case['storage_owners']) == 1
    owner = next(iter(case['storage_owners'].values()))
    assert owner.numel() == case['kwargs'][case['name'].removeprefix('strided_').removesuffix('_h4_d32')].numel() * 2
    torch.testing.assert_close(h.reference(**case['kwargs']), case['expected'], atol=.001, rtol=.001)
assert len(c.reference_controls(h)) == 8

if FAULT == 'signed_zero_hole':
    original_cases = c.control_cases
    def zero_hole_cases(device):
        for case in original_cases(device):
            if case['name'] == 'strided_g_h4_d32':
                case['storage_owners']['g_storage'][1::2].zero_()
            yield case
    c.control_cases = zero_hole_cases

def candidate(**kwargs):
    kwargs = dict(kwargs)
    if FAULT == 'g' and kwargs['g'].stride(-1) == 2:
        g = kwargs['g']
        kwargs['g'] = torch.as_strided(g, g.shape, (256, 128, 1),
                                      storage_offset=g.storage_offset())
    elif FAULT == 'A' and kwargs['A'].stride(0) == 2:
        A = kwargs['A']
        kwargs['A'] = torch.as_strided(A, A.shape, (1,),
                                      storage_offset=A.storage_offset())
    elif FAULT == 'g_bias' and kwargs['g_bias'] is not None and kwargs['g_bias'].stride(0) == 2:
        bias = kwargs['g_bias']
        kwargs['g_bias'] = torch.as_strided(bias, bias.shape, (1,),
                                           storage_offset=bias.storage_offset())
    elif FAULT == 'storage_hole' and kwargs['g'].stride(-1) == 2:
        kwargs['g']._base[1] += 1
    elif FAULT == 'signed_zero_view' and kwargs['A'].stride(0) == 2:
        assert kwargs['A'][0].item() == 0.0
        kwargs['A'][0] = -0.0
    elif FAULT == 'signed_zero_hole' and kwargs['g'].stride(-1) == 2:
        assert kwargs['g']._base[1].item() == 0.0
        kwargs['g']._base[1] = -0.0
    return h.reference(**kwargs)

if FAULT is None:
    assert len(c.run_controls(SimpleNamespace(fused_kda_gate=candidate), 'cpu')) == 8
else:
    try:
        c.run_controls(SimpleNamespace(fused_kda_gate=candidate), 'cpu')
    except (NumericalMismatch, ContractFailure) as exc:
        expected_type = (NumericalMismatch if FAULT in ('g', 'A', 'g_bias')
                         else ContractFailure)
        assert isinstance(exc, expected_type), (type(exc), exc)
        if FAULT.startswith('signed_zero_'):
            assert 'bytes changed' in str(exc)
    else:
        raise AssertionError('Strided flat-read or backing-storage mutation passed')
'''.replace("FAULT", repr(fault))
    _run(code)


@pytest.mark.parametrize("operand", ["g", "A", "g_bias"])
def test_public_wrapper_normalizes_only_needed_views_without_gpu(operand):
    code = r'''
import ast
from math import ceil
from pathlib import Path
import torch
from scripts import task_runner as h, semantic_controls as c

source = Path('source/triton_kda_gate.py')
tree = ast.parse(source.read_text())
wrapper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'fused_kda_gate')
seen = []
class FakeJIT:
    def __getitem__(self, grid):
        def launch(g, A, y, bias, beta, threshold, T, H, D, **meta):
            seen.append((g.stride(), A.stride(), bias.stride() if bias is not None else None,
                         g.data_ptr(), A.data_ptr(), bias.data_ptr() if bias is not None else None))
            assert meta['HAS_BIAS'] is (bias is not None)
            ref = h.reference(g, A, D, g_bias=bias, beta_val=beta, threshold=threshold)
            y.copy_(ref.reshape_as(y))
        return launch
namespace = {'torch': torch, 'ceil': ceil, 'kda_gate_fwd_kernel': FakeJIT()}
exec(compile(ast.Module(body=[wrapper], type_ignores=[]), str(source), 'exec'), namespace)
fused = namespace['fused_kda_gate']
case = next(x for x in c.control_cases('cpu') if x['name'] == 'strided_' + OPERAND + '_h4_d32')
kwargs = case['kwargs']
backing = next(iter(case['storage_owners'].values()))
before = backing.clone()
output = fused(**kwargs)
torch.testing.assert_close(output, case['expected'], atol=.001, rtol=.001)
assert torch.equal(backing, before)
assert len(seen) == 1
assert seen[0][:3] == ((128, 1), (1,), (1,))

# The original scored path has contiguous tensors and receives no new copy.
plain = {key: value.contiguous() if isinstance(value, torch.Tensor) else value
         for key, value in kwargs.items()}
seen.clear()
fused(**plain)
assert seen[0][3:] == (plain['g'].data_ptr(), plain['A'].data_ptr(),
                        plain['g_bias'].data_ptr())

# Preserve the old view-based domain boundary: a nonflattenable transpose is
# rejected before the fake kernel dispatch rather than silently broadening it.
nonflattenable_g = torch.randn(2, 3, 128).transpose(0, 1)
count_before = len(seen)
try:
    fused(nonflattenable_g, plain['A'], 32, plain['g_bias'])
except RuntimeError as exc:
    assert 'view size is not compatible' in str(exc)
else:
    raise AssertionError('Previously rejected nonflattenable g was accepted')
assert len(seen) == count_before
'''.replace("OPERAND", repr(operand))
    _run(code)
