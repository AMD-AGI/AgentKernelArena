"""CPU controls for unscored reverse-range public dtype coverage."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
TASKS = (
    'triton2triton/rocmbench/easy/test_reverse_range',
)


def reference(task):
    path = ROOT / 'tasks' / task / '_arena_reference.py'
    spec = importlib.util.spec_from_file_location('reverse_dtype_reference', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeKernel:
    def __init__(self, fault=None):
        self.fault = fault
    def __getitem__(self, grid):
        assert grid == (1,)
        def run(source, output):
            expected = torch.flip(source[1:513], [0])
            if self.fault == source.dtype:
                if source.dtype == torch.int64:
                    expected = expected.to(torch.int32).to(torch.int64)
                elif source.dtype == torch.float64:
                    expected = expected.float().double()
                elif source.dtype == torch.int32:
                    expected = torch.flip(source[:512], [0])
                else:
                    expected = ~expected
            output.copy_(expected)
            if self.fault == 'mutate':
                source[-1] = 0
        return run


@pytest.mark.parametrize('task', TASKS)
def test_original_scored_cases_and_unscored_dtype_oracles(task):
    manifest = json.loads((ROOT / 'tasks' / task / 'workloads.json').read_text())
    rows = manifest['cases']
    assert len(rows) == 10 and sum('performance' in row['checks'] for row in rows) == 9
    assert {row['params']['arguments']['dtype_str'] for row in rows
            if row['params']['function'] == 'test_performance'} == {'fp16', 'bf16', 'fp32'}
    assert reference(task).check_dtype_controls(
        type('Module', (), {'reverse_range': FakeKernel()})(), device='cpu') == [
        'int32', 'int64', 'float64', 'bool']


@pytest.mark.parametrize('task', TASKS)
@pytest.mark.parametrize('fault', [torch.int32, torch.int64, torch.float64, torch.bool, 'mutate'])
def test_rejects_dtype_specific_wrong_kernel_and_input_write(task, fault):
    module = type('Module', (), {'reverse_range': FakeKernel(fault)})()
    with pytest.raises((AssertionError, ValueError)):
        reference(task).check_dtype_controls(module, device='cpu')


@pytest.mark.parametrize('task', TASKS)
@pytest.mark.parametrize('fault', [None, torch.int64])
def test_correctness_action_runs_public_dtype_control(task, fault, monkeypatch):
    directory = ROOT / 'tasks' / task
    monkeypatch.syspath_prepend(str(directory))
    cpu_reference = reference(task)
    check = cpu_reference.check_dtype_controls
    cpu_reference.check_dtype_controls = lambda module: check(module, device='cpu')
    monkeypatch.setitem(sys.modules, '_arena_reference', cpu_reference)
    path = directory / '_arena_eval.py'
    spec = importlib.util.spec_from_file_location('reverse_dtype_eval', path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    def fake_pytest_main(args, plugins):
        plugin = plugins[0]
        plugin.operator_module = type('Module', (), {'reverse_range': FakeKernel(fault)})()
        for row in plugin.rows.values():
            row['status'] = 'PASS'
            row.pop('reason', None)
        return 0
    monkeypatch.setattr(pytest, 'main', fake_pytest_main)
    result = adapter.evaluate('candidate', 'correctness')
    if fault is None:
        assert result['status'] == 'PASS'
        assert result['cases'][0]['metrics']['unscored_dtype_controls'] == [
            'int32', 'int64', 'float64', 'bool']
    else:
        assert result['status'] == 'FAIL'
        assert 'Exact output mismatch' in result['reason']
