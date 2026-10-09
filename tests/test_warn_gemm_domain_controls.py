"""CPU negative controls for the task-local, correctness-only GEMM API probes."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch


TASK = Path(__file__).resolve().parents[1] / 'tasks/instruction2triton/rocmbench/gemm'
SPEC = importlib.util.spec_from_file_location('_gemm_domain_controls_test', TASK / '_arena_domain_controls.py')
checks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checks)


def _public_gemm(a, b, c, a_scale, b_scale, *, scale_a8_b8=None, activation='', defect=None):
    left = a.float()
    right = b.float()
    if defect == 'fp32_downcast':
        left = a.half().float()
        right = b.half().float()
    if defect == 'fp8_truncate':
        right = b.float().to(torch.int8).float()
    if defect == 'ignore_a_stride':
        left = a.as_strided(a.shape, (a.shape[1], 1)).float()
    if defect == 'ignore_b_stride':
        right = b.as_strided(b.shape, (b.shape[1], 1)).float()
    if scale_a8_b8 == 'block':
        k = left.shape[1]
        n = right.shape[1]
        if a_scale is not None and defect != 'ignore_block_a':
            expanded_a = a_scale.float().repeat_interleave(128, 1)[:, :k]
            left = left * expanded_a
        if defect != 'ignore_block_b':
            expanded_b = b_scale.float().repeat_interleave(128, 0).repeat_interleave(128, 1)[:k, :n]
            right = right * expanded_b
    result = left @ right
    if scale_a8_b8 == 'tensor':
        if defect != 'ignore_tensor_a':
            result = result * a_scale
        if defect != 'ignore_tensor_b':
            result = result * b_scale
    if activation == 'leaky_relu' and defect != 'ignore_activation':
        result = torch.where(result + 1 >= 0, result + 1, (result + 1) * .01)
    if defect == 'zero':
        result.zero_()
    if defect == 'no_write':
        return
    if defect == 'ignore_c_stride':
        c.as_strided(c.shape, (c.shape[1], 1)).copy_(result.to(c.dtype))
    else:
        c.copy_(result.to(c.dtype))
    if defect == 'mutate_a':
        a.zero_()
    if defect == 'mutate_scale' and b_scale is not None:
        b_scale.zero_()


def _module(defect=None):
    return SimpleNamespace(
        e4m3_type=torch.float8_e4m3fnuz,
        matmul=lambda *args, **kwargs: _public_gemm(*args, **kwargs, defect=defect),
    )


@pytest.mark.parametrize('name', checks.CONTROL_NAMES)
def test_public_gemm_control_accepts_independent_reference(name):
    checks.run_control(_module(), name, device='cpu')


@pytest.mark.parametrize('name,defect', [
    ('bf16_column_a', 'ignore_a_stride'),
    ('fp32_column_b_strided_output', 'ignore_b_stride'),
    ('fp32_column_b_strided_output', 'ignore_c_stride'),
    ('fp32_column_b_strided_output', 'fp32_downcast'),
    ('fp16_both_inputs_strided', 'ignore_a_stride'),
    ('fp16_both_inputs_strided', 'ignore_b_stride'),
    ('fp8_tensor_scale', 'ignore_tensor_a'),
    ('fp8_tensor_scale', 'ignore_tensor_b'),
    ('fp8_tensor_scale', 'fp8_truncate'),
    ('int8_int32', 'zero'),
    ('fp16_block_scale', 'ignore_block_a'),
    ('fp16_block_scale', 'ignore_block_b'),
    ('fp16_block_b_only', 'ignore_block_b'),
    ('leaky_relu', 'ignore_activation'),
    ('leaky_relu', 'no_write'),
    ('leaky_relu', 'mutate_a'),
    ('fp16_block_scale', 'mutate_scale'),
])
def test_public_gemm_control_rejects_wrong_branch_or_mutation(name, defect):
    with pytest.raises((AssertionError, RuntimeError)):
        checks.run_control(_module(defect), name, device='cpu')


def test_domain_manifest_is_separate_from_the_original_22_and_unscored():
    cases = json.loads((TASK / 'workloads.json').read_text())['cases']
    original = [row for row in cases if row['params']['function'] != 'domain_control']
    controls = [row for row in cases if row['params']['function'] == 'domain_control']
    assert len(original) == 22
    assert sum('performance' in row['checks'] for row in original) == 11
    assert len(controls) == len(checks.CONTROL_NAMES)
    assert {row['params']['arguments']['name'] for row in controls} == set(checks.CONTROL_NAMES)
    assert all(row['checks'] == ['correctness'] for row in controls)
    assert len({row['test_case_id'] for row in cases}) == len(cases)


@pytest.mark.parametrize('bad_control', [None, 'fp8_tensor_scale'])
def test_adapter_reports_every_domain_control_for_both_roles(monkeypatch, bad_control):
    monkeypatch.syspath_prepend(str(TASK))
    spec = importlib.util.spec_from_file_location('_gemm_adapter_domain_test', TASK / '_arena_eval.py')
    adapter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = adapter
    spec.loader.exec_module(adapter)
    collected = json.loads((TASK / 'workloads.json').read_text())
    plugin = adapter.ReportPlugin(collected, 'correctness')
    assert len(plugin.expected) == 22 and len(plugin.controls) == 8
    calls = []

    def run_control(module, name, *, device='cuda'):
        calls.append(name)
        if name == bad_control:
            raise AssertionError('wrong public GEMM branch')

    import _arena_domain_controls
    monkeypatch.setattr(_arena_domain_controls, 'run_control', run_control)

    def fake_pytest_main(args, *, plugins):
        selected = plugins[0]
        selected.module = _module()
        for key in selected.expected:
            selected.rows[key]['status'] = 'PASS'
        return 0

    monkeypatch.setattr(pytest, 'main', fake_pytest_main)
    for role in ('baseline', 'candidate'):
        calls.clear()
        result = adapter.evaluate(role, 'correctness')
        assert calls == list(checks.CONTROL_NAMES)
        assert len(result['cases']) == 30
        failed = [row for row in result['cases'] if row['status'] != 'PASS']
        assert len(failed) == int(bad_control is not None)
        assert (result['status'] == 'FAIL') == (bad_control is not None)
        if failed:
            assert failed[0]['params']['arguments']['name'] == bad_control
