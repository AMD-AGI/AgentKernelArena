"""Finite FLA geometry and real adapter/guard contracts; CPU is not GPU proof."""
import ast
from copy import deepcopy
import hashlib
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
from triton_signal_migration_helpers import FLA_GEOMETRY_CONTROLS, without_fla_geometry

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / 'tasks/triton2triton/vllm/triton_fla_fused_recurrent'
KERNEL = 'fused_recurrent_gated_delta_rule_fwd_kernel'
PUBLIC = 'fused_recurrent_gated_delta_rule_fwd'
FAULT_IDS = {
    'k_tail': {0, 1}, 'v_tail': {0, 1}, 'v_second_tile': {0, 1, 2},
    'extra_head': {0, 1, 2}, 'initial_state': {1, 2},
}


@pytest.fixture
def task(monkeypatch):
    for name in list(sys.modules):
        if name == 'scripts' or name.startswith('scripts.'):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(TASK))
    monkeypatch.chdir(TASK)
    spec = importlib.util.spec_from_file_location('_geometry_adapter', TASK / '_arena_eval.py')
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    h = adapter.load_harness()
    monkeypatch.setattr(adapter, 'load_harness', lambda: h)
    return h, adapter


def cpu_module(controls, fault=None):
    def operation(**kwargs):
        # Exercise the original general reference, not the new closed form.
        actual = controls.reference_outputs(**kwargs)
        keys, values, heads = kwargs['q'].shape[-1], kwargs['v'].shape[-1], kwargs['q'].shape[-2]
        if fault == 'k_tail' and keys & (keys - 1):
            actual[0].zero_()
        if fault == 'v_tail' and values & (values - 1):
            actual[0][..., values // 32 * 32:].zero_()
        if fault == 'v_second_tile':
            actual[0][..., 32:].zero_()
        if fault == 'extra_head' and heads > 2:
            actual[0][:, :, 2:].zero_()
        if fault == 'initial_state' and kwargs['initial_state'] is not None:
            actual = controls.reference_outputs(**{**kwargs, 'initial_state': None})
        if fault in ('q_input', 'h0_input'):
            value = kwargs['q' if fault == 'q_input' else 'initial_state']
            if value is not None:
                value.zero_()
        return actual
    return SimpleNamespace(**{PUBLIC: operation})


def cpu_factories(monkeypatch):
    for name in ('arange', 'zeros', 'empty', 'full'):
        original = getattr(torch, name)
        def factory(*args, _original=original, **kwargs):
            if kwargs.get('device') == 'cuda':
                kwargs['device'] = 'cpu'
            return _original(*args, **kwargs)
        monkeypatch.setattr(torch, name, factory)


def test_new_manifest_is_exactly_additive_and_old_runner_bytes_are_pinned(task):
    h, adapter = task
    manifest = adapter.load_manifest()
    assert h.GEOMETRY_CONTROLS == manifest['geometry_controls'] == FLA_GEOMETRY_CONTROLS
    old = without_fla_geometry(manifest)
    # This pins the complete previously GPU-run source-v6 manifest, not merely
    # a subset of seed or shape fields. Every old case remains byte-equivalent.
    assert hashlib.sha256((json.dumps(old, indent=2) + '\n').encode()).hexdigest() == (
        'b61bea1624ee1299130960d34b8616b3e61b07e324b9c2c81f761b7488121efa')
    assert len(manifest['cases']) == 13
    assert sum('performance' in row['checks'] for row in manifest['cases']) == 5
    source = (TASK / 'scripts/task_runner.py').read_text()
    additions = ['GEOMETRY_CONTROLS = semantic_controls.GEOMETRY_CONTROLS\n\n',
        '    if case_index is not None and case_index >= 30000:\n'
        '        return semantic_controls.run_geometry_control(case_index - 30000, load_module)\n']
    for addition in additions:
        assert source.count(addition) == 1
        source = source.replace(addition, '', 1)
    assert hashlib.sha256(source.encode()).hexdigest() == 'bcb65a713dabe0d1e5d805e419c8746f6cad986c970a0af21b98a801b07f9ee9'
    assert (h.WARMUP_ITERATIONS, h.BENCHMARK_ITERATIONS) == (10, 100)
    assert hashlib.sha256((TASK / 'source/triton_fla_fused_recurrent.py').read_bytes()).hexdigest() == (
        '931a0358c40f99eb3359afc054db4c89978d641f7252b123111e69f341db60e8')


def test_every_old_semantic_control_and_reference_math_node_is_preserved():
    tree = ast.parse((TASK / 'scripts/semantic_controls.py').read_text())
    removed = []
    for node in list(tree.body):
        if (isinstance(node, ast.FunctionDef) and node.name in {'geometry_case', 'run_geometry_control'}
                or isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == 'GEOMETRY_CONTROLS'):
            tree.body.remove(node)
            removed.append(node)
    assert len(removed) == 3
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    controls = functions['reference_controls']
    assert ast.unparse(controls.body[-2].iter) == 'range(len(GEOMETRY_CONTROLS))'
    controls.body.pop(-2)
    signal = functions['run_signal_controls']
    assert ast.unparse(signal.body[0].test) == 'case_index >= 30000'
    signal.body.pop(0)
    assert hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest() == (
        '7097e546e9647d76332666e0a73753d82c1e09115776eba6c8a13a99283deb38')


@pytest.mark.parametrize('index', range(3))
def test_closed_form_matches_both_original_independent_references(task, index):
    h, _ = task
    c = h.semantic_controls.geometry_case(index, 'cpu')
    output, history = c['expected']
    torch.testing.assert_close(output, h.reference(**c['kwargs']), atol=2e-5, rtol=2e-6)
    result = h.semantic_controls.reference_outputs(**c['kwargs'])
    for got, expected in zip(result, c['expected']):
        torch.testing.assert_close(got, expected, atol=2e-5, rtol=2e-6)
    h.semantic_controls.require_signal(c['expected'])
    assert bool(torch.all(output != 0))
    assert bool(torch.all(history[..., -1] != 0))
    if c['kwargs']['initial_state'] is not None:
        assert bool(torch.all(history[..., -2] != 0))
    assert h.semantic_controls.run_geometry_control(index, lambda: cpu_module(h.semantic_controls), 'cpu') == (True, None)


@pytest.mark.parametrize('fault', FAULT_IDS)
@pytest.mark.parametrize('index', range(3))
def test_specific_omissions_are_rejected_by_actual_geometry_gate(task, fault, index):
    h, _ = task
    ok, reason = h.semantic_controls.run_geometry_control(index, lambda: cpu_module(h.semantic_controls, fault), 'cpu')
    assert ok is (index not in FAULT_IDS[fault]), reason
    if not ok:
        assert 'NumericalMismatch: Output numerical mismatch:' in reason


@pytest.mark.parametrize('fault', ['q_input', 'h0_input'])
@pytest.mark.parametrize('index', range(3))
def test_geometry_inputs_are_readonly(task, fault, index):
    h, _ = task
    ok, reason = h.semantic_controls.run_geometry_control(index, lambda: cpu_module(h.semantic_controls, fault), 'cpu')
    should_fail = fault == 'q_input' or index != 0
    assert ok is not should_fail
    if should_fail:
        assert 'Read-only input changed:' in reason


@pytest.mark.parametrize('index', range(3))
def test_real_host_wrapper_preserves_geometry_layout_and_launch_contract(task, index):
    h, _ = task
    case = h.semantic_controls.geometry_case(index, 'cpu')
    wrapper = next(node for node in ast.parse((TASK / 'source/triton_fla_fused_recurrent.py').read_text()).body
                   if isinstance(node, ast.FunctionDef) and node.name == PUBLIC)
    seen = []
    class Launch:
        def __getitem__(self, grid):
            def launch(**kw):
                seen.append((grid, kw))
                # CPU stand-in for device execution: this tests the unmodified
                # actual wrapper's allocations/launch ABI, never the GPU kernel.
                expected = h.semantic_controls.reference_outputs(*(kw[key] for key in ('q', 'k', 'v', 'g', 'beta')),
                    kw['scale'], kw['h0'])
                kw['o'].squeeze(0).copy_(expected[0])
                kw['ht'].copy_(expected[1])
            return launch
    namespace = {'torch': torch, 'triton': SimpleNamespace(next_power_of_2=lambda x: 1 << (x - 1).bit_length(),
                  cdiv=lambda x, y: (x + y - 1) // y), KERNEL: Launch()}
    exec(compile(ast.Module(body=[wrapper], type_ignores=[]), '<actual FLA wrapper>', 'exec'), namespace)
    got = namespace[PUBLIC](**case['kwargs'])
    for actual, expected in zip(got, case['expected']):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-6)
    grid, kw = seen[0]
    batch, _, heads, keys, values = FLA_GEOMETRY_CONTROLS[index]['shape']
    assert grid == (1, 2, batch * heads)
    assert kw['BK'] == (32 if keys == 17 else 64) and kw['BV'] == 32
    assert (kw['B'], kw['H'], kw['K'], kw['V']) == (batch, heads, keys, values)
    assert kw['num_warps'] == 1 and kw['num_stages'] == 3


@pytest.mark.parametrize('role', ['baseline', 'candidate'])
def test_actual_adapter_routes_all_new_geometry_rows_for_both_roles(task, monkeypatch, role):
    h, adapter = task
    cpu_factories(monkeypatch)
    monkeypatch.setattr(h, 'load_module', lambda: cpu_module(h.semantic_controls))
    original = h.run_correctness
    original_signal = h.run_signal_controls
    seen = []
    def check(*, case_index):
        seen.append(case_index)
        return original(case_index=case_index) if case_index >= 30000 else (True, None)
    monkeypatch.setattr(h, 'run_correctness', check)
    monkeypatch.setattr(h, 'run_signal_controls', lambda *, case_index: original_signal(case_index=case_index)
                        if case_index >= 30000 else {'status': 'PASS'})
    monkeypatch.setattr(h, 'run_semantic_controls', lambda: [])
    result = adapter.evaluate(role, 'correctness')
    assert result['status'] == 'PASS', result
    assert seen == [row['params']['case_index'] for row in result['cases']]
    assert len(seen) == 13 and seen[-3:] == [30000, 30001, 30002]
    assert all(row['metrics']['unscored_signal_control']['independent_closed_form'] for row in result['cases'][-3:])


@pytest.mark.parametrize('field', ['shape', 'case_index', 'initial_state', 'scale', 'dtype', 'checks', 'missing'])
def test_adapter_rejects_manifest_geometry_drift_before_candidate_execution(task, monkeypatch, field):
    h, adapter = task
    data = deepcopy(adapter.load_manifest())
    if field in ('shape', 'case_index', 'initial_state', 'scale'):
        data['geometry_controls'][0][field] = 0
    elif field == 'missing':
        data['cases'].pop()
    else:
        data['cases'][-1][field] = 'invalid'
    monkeypatch.setattr(adapter, 'load_manifest', lambda: data)
    monkeypatch.setattr(h, 'load_module', lambda: pytest.fail('Drift must fail before candidate execution'))
    result = adapter.evaluate('candidate', 'correctness')
    assert result['status'] == 'FAIL' and result['failure_kind'] == 'execution_failure'
    assert 'geometry' in result['reason'].lower()


@pytest.mark.parametrize('index', [-1, 3, True, 0.0])
def test_invalid_geometry_index_does_not_load_candidate(task, index):
    h, _ = task
    ok, reason = h.semantic_controls.run_geometry_control(index, lambda: pytest.fail('Invalid index loaded candidate'), 'cpu')
    assert ok is False and 'Invalid geometry control index' in reason


@pytest.mark.parametrize('change', ['device_kernel', 'new_helper', 'wrapper', 'import', 'control', 'manifest'])
def test_actual_harness_guard_protects_geometry_and_host_boundary(tmp_path, change):
    workspace = tmp_path / 'task'
    shutil.copytree(TASK, workspace, ignore=shutil.ignore_patterns('__pycache__'))
    spec = load_task_spec(workspace / 'config.yaml', task_id='triton2triton/vllm/triton_fla_fused_recurrent')
    snapshot = snapshot_workspace_harness(workspace, task_spec=spec)
    source = workspace / 'source/triton_fla_fused_recurrent.py'
    tree = ast.parse(source.read_text())
    if change in ('device_kernel', 'wrapper'):
        name = KERNEL if change == 'device_kernel' else PUBLIC
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        node.body.append(ast.Pass())
    elif change == 'new_helper':
        tree.body.extend(ast.parse('@triton.jit\ndef geometry_helper(x):\n    return x\n').body)
    elif change == 'import':
        tree.body.append(ast.Import(names=[ast.alias(name='os')]))
    elif change == 'control':
        with (workspace / 'scripts/semantic_controls.py').open('a') as handle:
            handle.write('\nGEOMETRY_CONTROLS = []\n')
    elif change == 'manifest':
        (workspace / 'workloads.json').write_text('{}\n')
    source.write_text(ast.unparse(ast.fix_missing_locations(tree)))
    if change in ('device_kernel', 'new_helper'):
        verify_workspace_harness(snapshot, discard_added=False)
    else:
        with pytest.raises(RuntimeError, match='[Hh]arness|[Pp]rotected|policy'):
            verify_workspace_harness(snapshot, discard_added=False)
