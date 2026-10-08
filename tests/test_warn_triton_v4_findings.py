"""Regression controls for task-contract findings from GPU validation."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / 'tasks/triton2triton/vllm'


def load(task, relative):
    path = TASKS / task / relative
    spec = importlib.util.spec_from_file_location('gpu_findings_' + task, path)
    module = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    try:
        spec.loader.exec_module(module)
    finally:
        os.chdir(cwd)
    return module


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_expand_requires_exact_float_values(dtype):
    checks = load('triton_expand', '_arena_checks.py')
    expected = torch.tensor([1.25, -2.5, 6.125], dtype=dtype)
    checks.check_output(expected.clone(), expected)
    with pytest.raises(AssertionError):
        checks.check_output(expected + .005, expected)
    with pytest.raises(AssertionError):
        checks.check_output(expected + .5, expected)
    with pytest.raises(AssertionError):
        checks.check_output(torch.full_like(expected, float('nan')), expected)


def test_expand_bool_controls_cover_both_count_dtypes_and_replacement():
    checks = load('triton_expand', '_arena_checks.py')
    cases = [(x, cu, old, new) for x, cu, old, new in checks.dtype_controls('cpu')
             if x.dtype == torch.bool]
    assert len(cases) == 4
    assert {cu.dtype for _, cu, _, _ in cases} == {torch.int32, torch.int64}
    for count_dtype in (torch.int32, torch.int64):
        matching = [(x, cu, old, new) for x, cu, old, new in cases if cu.dtype == count_dtype]
        assert [(old, new) for _, _, old, new in matching] == [(False, False), (False, True)]
        mixed = checks.reference(*matching[0][:2], 132, *matching[0][2:])
        replaced = checks.reference(*matching[1][:2], 132, *matching[1][2:])
        assert mixed.dtype == replaced.dtype == torch.bool
        assert mixed.any() and (~mixed).any()
        assert replaced.all()


@pytest.mark.parametrize('broken', [None, 'integer_only', 'int32_truncation', 'bool_only'])
def test_expand_public_control_rejects_dtype_shortcuts(broken):
    checks = load('triton_expand', '_arena_checks.py')
    seen = set()
    calls = []
    def expand(x, cu, num_tokens, old=0, new=0):
        seen.add((x.dtype, cu.dtype))
        calls.append((x.dtype, cu.dtype, old, new))
        output = checks.reference(x, cu, num_tokens, old, new)
        if broken == 'integer_only' and x.is_floating_point():
            output = output.to(torch.int64).to(x.dtype)
        if broken == 'int32_truncation' and x.dtype == torch.int64:
            output = output.to(torch.int32).to(x.dtype)
        if broken == 'bool_only' and x.dtype == torch.bool:
            output = ~output
        return output
    module = SimpleNamespace(expand_batch_to_tokens=expand)
    harness = SimpleNamespace(load_module=lambda: module)
    def run():
        with checks.checked_modules(harness):
            result = harness.load_module().expand_batch_to_tokens(
                torch.tensor([3, 9], dtype=torch.int32),
                torch.tensor([2, 4], dtype=torch.int32), 4)
            assert result.tolist() == [3, 3, 9, 9]
    if broken:
        with pytest.raises(AssertionError):
            run()
    else:
        run()
        assert len(seen) == 20
        assert len(calls) == 29  # ragged, 22 dtype, 5 stride controls, original
    assert module.expand_batch_to_tokens is expand


def test_moe_topk_one_partial_tiles_have_observable_contributions():
    h = load('triton_fused_moe', 'scripts/task_runner.py')
    inputs, options = h.control_inputs('small_topk_one', 'cpu')
    assert inputs['A'].shape == (3, 7)
    assert inputs['B'].shape == (2, 5, 7)
    assert inputs['ids'].shape == (3, 1)
    expected = h.reference(inputs, options)
    h.check_output(expected.clone(), expected)
    with pytest.raises(AssertionError):
        h.check_output(torch.zeros_like(expected), expected)
    with pytest.raises(AssertionError):
        h.check_output(h.reference(inputs, {'mul_routed_weight': False}), expected)
    manifest = json.loads((TASKS / 'triton_fused_moe/workloads.json').read_text())
    assert [row['params']['control'] for row in manifest['cases'] if 'control' in row['params']] == list(h.CONTROL_CASES)


def test_moe_mmk_declared_control_dtypes_match_real_inputs():
    h = load('triton_moe_mmk', 'scripts/task_runner.py')
    manifest = json.loads((TASKS / 'triton_moe_mmk/workloads.json').read_text())
    controls = [row for row in manifest['cases'] if 'control' in row['params']]
    assert len(controls) == 5
    for row in controls:
        values = h.control_inputs(row['params']['control'], 'cpu')
        actual = {str(value.dtype).removeprefix('torch.') for value in values.values()
                  if isinstance(value, torch.Tensor)}
        assert set(row['dtype'].split(',')) == actual


@pytest.mark.parametrize('operand', ['a', 'b'])
def test_ssd_controls_reject_ignored_token_and_group_strides(operand):
    h = load('triton_ssd_bmm', 'scripts/task_runner.py')
    c = load('triton_ssd_bmm', 'scripts/semantic_controls.py')
    case = next(case for case in c.control_cases('cpu')
                if case['name'].endswith('_stride_' + operand))
    kwargs = dict(case['kwargs'])
    kwargs.pop('output_dtype')
    value = kwargs[operand]
    assert value.stride(-1) == 1 and not value.is_contiguous()
    c.check_outputs(h.reference_bmm(**kwargs), case['expected'], atol=case['atol'], rtol=case['rtol'])
    kwargs[operand] = torch.as_strided(value, value.shape, torch.empty(value.shape).stride())
    with pytest.raises(c.ContractFailure):
        c.check_outputs(h.reference_bmm(**kwargs), case['expected'], atol=case['atol'], rtol=case['rtol'])
