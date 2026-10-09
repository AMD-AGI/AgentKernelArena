"""Reject copy-value and chunk-branch gaps found by task validation."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

TASKS = Path(__file__).resolve().parents[1] / 'tasks/triton2triton/vllm'


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_exact_copy_accepts_nan_and_rejects_changed_bits(dtype):
    checks = load_file(TASKS / 'triton_expand/_arena_checks.py', 'copy_value_checks')
    x = torch.tensor([float('nan'), float('inf'), -float('inf'), -0.0, 1.25], dtype=dtype)
    counts = torch.tensor([1, 2, 3, 4, 6], dtype=torch.int64)
    expected = checks.reference(x, counts, 6, -float('inf'), 6.125)
    checks.check_output(expected.clone(), expected)
    checks.unchanged((x, counts), (x.clone(), counts.clone()))
    assert torch.isnan(expected[0]) and torch.isposinf(expected[1])
    assert expected[2] == 6.125 and torch.signbit(expected[3])
    for index, wrong_value in [(0, 0.0), (1, -float('inf')), (3, 0.0), (5, 1.5)]:
        wrong = expected.clone()
        wrong[index] = wrong_value
        with pytest.raises(AssertionError, match='differs'):
            checks.check_output(wrong, expected)
    changed = x.clone()
    changed[3] = 0.0
    with pytest.raises(AssertionError, match='read-only'):
        checks.unchanged((changed, counts), (x, counts))


@pytest.mark.parametrize('broken', [False, True])
def test_public_expand_controls_reject_nonfinite_corruption(broken):
    checks = load_file(TASKS / 'triton_expand/_arena_checks.py', 'public_copy_checks')
    seen = set()

    def expand(x, counts, total, old=0, new=0):
        output = checks.reference(x, counts, total, old, new)
        if x.is_floating_point() and torch.isnan(x).any():
            seen.add((x.dtype, counts.dtype))
            if broken:
                output.nan_to_num_()
        return output

    module = SimpleNamespace(expand_batch_to_tokens=expand)
    harness = SimpleNamespace(load_module=lambda: module)

    def invoke():
        with checks.checked_modules(harness):
            harness.load_module().expand_batch_to_tokens(
                torch.tensor([2, 5]), torch.tensor([1, 2]), 2)

    if broken:
        with pytest.raises(AssertionError, match='differs'):
            invoke()
    else:
        invoke()
        assert len(seen) == 8
    assert module.expand_batch_to_tokens is expand


def test_kda_chunk_controls_reject_untested_branch_shortcuts(monkeypatch):
    root = TASKS / 'triton_kda_gla_fwd_o'
    for name in list(sys.modules):
        if name == 'scripts' or name.startswith('scripts.'):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.chdir(root)
    runner = load_file(root / 'scripts/task_runner.py', 'chunk_domain_runner')
    controls = runner.semantic_controls
    cases = [case for case in controls.control_cases('cpu')
             if case['kwargs']['chunk_size'] in (32, 128)]
    assert {case['kwargs']['chunk_size'] for case in cases} == {32, 128}
    for case in cases:
        kwargs = case['kwargs']
        assert kwargs['q'].shape[1] % kwargs['chunk_size'] != 0
        controls.check_outputs(runner.reference(**kwargs), case['expected'],
                               atol=case['atol'], rtol=case['rtol'])
        monkeypatch.setattr(controls, 'control_cases', lambda device, c=case: iter([c]))
        good = SimpleNamespace(kda_gla_fwd_o=lambda **kw: runner.reference(**kw))
        assert controls.run_controls(good, 'cpu')[0]['status'] == 'PASS'

        def wrong_chunk(**kw):
            value = runner.reference(**kw)
            if kw['chunk_size'] in (32, 128):
                value.zero_()
            return value

        with pytest.raises(controls.ContractFailure, match='numerical mismatch'):
            controls.run_controls(SimpleNamespace(kda_gla_fwd_o=wrong_chunk), 'cpu')
