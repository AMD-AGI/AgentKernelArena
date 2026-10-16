"""Actual protected zero-control plumbing on CPU; no GPU qualification."""
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest
import torch

from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness
from src.task_spec import load_task_spec

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / 'tasks/torch2flydsl/per_token_i8_quant_kernel'


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def control():
    return module(TASK / 'scripts/int8_zero_controls.py', '_int8_zero_control')


@pytest.fixture
def harness():
    model = module(TASK / 'model.py', '_int8_zero_model')
    replay = module(TASK / 'scripts/replay_checks.py', '_int8_zero_replay')
    tree = ast.parse((TASK / 'test_kernel_harness.py').read_text())
    names = {'_compare', '_checked_quant_pair', '_compare_quant_outputs'}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    ns = {'CODE_TOL': 1, 'SCALE_RTOL': 1e-3}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<actual-protected-comparators>', 'exec'), ns)
    h = SimpleNamespace(**ns, _KERNEL_DIR=str(TASK), MODEL_FILE='model.py', KERNEL_FILE='kernel.py',
                        KERNEL_ENTRY='flydsl_per_token_i8_quant', ARENA_PROVIDED_BASELINE=False,
                        require_unchanged=replay.require_unchanged)
    h.model_module = model
    return h


def zero_pair(x):
    return (torch.zeros_like(x, dtype=torch.int8), torch.ones((x.shape[0], 1), dtype=torch.float32))


@pytest.mark.parametrize('provided', [True, False])
@pytest.mark.parametrize('behavior', ['correct', 'scale_zero', 'wrong_codes', 'nonfinite', 'shape',
                                    'dtype', 'mutate', 'raise'])
def test_real_zero_control_checks_actual_candidate_and_restores_on_failure(control, harness, provided, behavior):
    seen = []

    def candidate(x):
        seen.append(x)
        pair = zero_pair(x)
        if behavior == 'scale_zero': pair[1].zero_()
        if behavior == 'wrong_codes': pair[0].fill_(2)
        if behavior == 'nonfinite': pair[1].fill_(float('nan'))
        if behavior == 'shape': pair = (pair[0], pair[1].reshape(-1))
        if behavior == 'dtype': pair = (pair[0].to(torch.int32), pair[1])
        if behavior == 'mutate': x.fill_(1)
        if behavior == 'raise':
            x.fill_(1)
            raise RuntimeError('candidate failed')
        return pair

    kmod = SimpleNamespace(flydsl_per_token_i8_quant=candidate)
    harness.ARENA_PROVIDED_BASELINE = provided
    harness._load_module = lambda directory, filename, alias: harness.model_module if filename == 'model.py' else kmod
    if provided or behavior == 'correct':
        result = control.run(harness, device='cpu')
        assert result['test_case_id'] == 'control_all_zero_rows'
        assert result['status'] == 'PASS'
    else:
        with pytest.raises((AssertionError, RuntimeError)):
            control.run(harness, device='cpu')
    assert len(seen) == (0 if provided else 1)
    if seen:
        assert seen[0].shape == (3, 4096) and seen[0].dtype == torch.bfloat16
        assert torch.count_nonzero(seen[0]) == 0
    assert harness.CODE_TOL == 1 and harness.SCALE_RTOL == 1e-3


def test_real_zero_control_rejects_missing_candidate(control, harness):
    harness._load_module = lambda directory, filename, alias: harness.model_module if filename == 'model.py' else None
    with pytest.raises(RuntimeError, match='no baseline fallback'):
        control.run(harness, device='cpu')


@pytest.mark.parametrize('provided', [True, False])
def test_actual_action_binding_runs_new_control_inside_existing_candidate_audit(control, harness, monkeypatch, provided):
    checks = module(TASK / 'scripts/candidate_checks.py', '_int8_actual_candidate_checks')
    monkeypatch.setitem(sys.modules, 'scripts.candidate_checks', checks)
    actions = module(TASK / 'scripts/task_actions.py', '_int8_actual_task_actions')
    events = []
    harness.ARENA_PROVIDED_BASELINE = provided
    harness.run_correctness = lambda verbose=True: events.append('original_correctness') or True
    # Only the profiling surface is simulated; the real preparation mode,
    # action binding, protected zero control and numerical comparisons execute.
    ns = {'__name__': 'flydsl.compiler.jit_function'}
    exec('class JitFunction:\n    def __init__(self, fn): self.fn = fn\n    def __call__(self, *args): return self.fn(*args)\n', ns)
    candidate = ns['JitFunction'](lambda x: events.append('zero_candidate') or zero_pair(x))
    kmod = SimpleNamespace(flydsl_per_token_i8_quant=candidate)

    def load(directory, filename, alias):
        if filename == 'model.py': return harness.model_module
        if filename == 'kernel.py': return kmod
        if filename == 'scripts/int8_zero_controls.py':
            return SimpleNamespace(run=lambda h: events.append('zero_control') or control.run(h, device='cpu'))
        raise AssertionError(filename)

    harness._load_module = load
    original_loader = harness._load_module
    observed = actions.check(harness)
    assert events[:2] == ['original_correctness', 'zero_control']
    assert ('zero_candidate' in events) is (not provided)
    assert bool(observed) is (not provided)
    assert harness._load_module is original_loader


def test_manifest_adds_only_an_unscored_explicit_zero_case(control):
    cases = json.loads((TASK / 'cases.json').read_text())['cases']
    assert cases[-1] == control.CASE
    assert len(cases) == 6
    assert [row['params'] for row in cases[:5]] == [
        {'m': 1, 'n': 4096, 'seed': 20260401}, {'m': 16, 'n': 8192, 'seed': 20260401},
        {'m': 128, 'n': 4096, 'seed': 20260401}, {'m': 256, 'n': 8192, 'seed': 20260401},
        {'m': 1024, 'n': 8192, 'seed': 20260401}]
    assert sum('performance' in row['checks'] for row in cases) == 5


def test_new_zero_helper_is_an_immutable_task_input(tmp_path):
    path = tmp_path / 'task'
    shutil.copytree(TASK, path, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    spec = load_task_spec(path / 'config.yaml', task_id='torch2flydsl/per_token_i8_quant_kernel')
    snapshot = snapshot_workspace_harness(path, task_spec=spec)
    source = path / 'scripts/int8_zero_controls.py'
    assert 'scripts/int8_zero_controls.py' in snapshot.digests
    source.write_bytes(source.read_bytes() + b'\n')
    with pytest.raises(RuntimeError, match='[Hh]arness|protected'):
        verify_workspace_harness(snapshot, discard_added=False)
