"""CPU gates for unscored expand and chunk-output public-domain controls."""

import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
VLLM = ROOT / 'tasks/triton2triton/vllm'


def _load_expand():
    path = VLLM / 'triton_expand/_arena_checks.py'
    spec = importlib.util.spec_from_file_location('expand_domain_checks', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('broken', [False, True])
def test_nan_replacement_key_controls_reject_isnan_matching(broken):
    checks = _load_expand()
    seen = set()

    def candidate(x, counts, total, old=0, new=0):
        output = checks.reference(x, counts, total, old, new)
        if x.is_floating_point() and isinstance(old, float) and old != old:
            seen.add((x.dtype, counts.dtype))
            if broken:
                output = torch.where(torch.isnan(output),
                                     torch.full_like(output, new), output)
        return output

    module = SimpleNamespace(expand_batch_to_tokens=candidate)
    harness = SimpleNamespace(load_module=lambda: module)
    with checks.checked_modules(harness):
        if broken:
            with pytest.raises(AssertionError, match='differs'):
                harness.load_module().expand_batch_to_tokens(
                    torch.tensor([2, 5]), torch.tensor([1, 2]), 2)
        else:
            harness.load_module().expand_batch_to_tokens(
                torch.tensor([2, 5]), torch.tensor([1, 2]), 2)
    assert len(seen) == (1 if broken else 8)
    assert module.expand_batch_to_tokens is candidate


def test_multi_tile_value_control_has_independent_answer_and_rejects_dropped_tile():
    code = '''
from types import SimpleNamespace
import torch
from scripts import task_runner as h, semantic_controls as c
cases = [x for x in c.control_cases('cpu') if x['name'] == 'gated_multi_tile_value_width']
assert len(cases) == 1
case = cases[0]
assert case['kwargs']['v'].shape[-1] == 160
assert case['kwargs']['v'].shape[1] == 20
torch.testing.assert_close(h.reference(**case['kwargs']), case['expected'], atol=.05, rtol=.05)
assert not torch.allclose(torch.zeros_like(case['expected'][...,128:]), case['expected'][...,128:], atol=.05, rtol=.05)
names = [item['name'] for item in c.control_cases('cpu')]
assert names[:5] == [
    'ungated_default_scale_partial_chunks',
    'gated_inter_intra_partial_chunks',
    'gated_scored_length',
    'batched_three_heads_mixed_width_partial_chunk',
    'gated_multi_tile_value_width',
]
assert names[5:] == ['ungated_multi_tile_key_width', 'gated_multi_tile_key_width']
assert [row['control'] for row in c.reference_controls(h)] == names
assert [row['control'] for row in c.run_controls(
    SimpleNamespace(chunk_fwd_o=lambda **kw: h.reference(**kw)), 'cpu')] == names
def dropped(**kw):
    out = h.reference(**kw)
    if kw['v'].shape[-1] > 128: out[...,128:] = 0
    return out
try:
    c.run_controls(SimpleNamespace(chunk_fwd_o=dropped), 'cpu')
except c.ContractFailure as exc:
    assert 'numerical mismatch' in str(exc).lower()
else:
    raise AssertionError('Dropped second tile passed task controls')
'''
    result = subprocess.run([sys.executable, '-c', code],
                            cwd=VLLM / 'triton_fla_chunk_fwd_o',
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
