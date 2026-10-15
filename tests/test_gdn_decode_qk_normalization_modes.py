"""CPU reference, default-option dispatch and protected coverage contracts."""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import types

import pytest
import torch
from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness
from src.task_spec import load_task_spec
from src.task_protocol import parse_command_result

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / 'tasks/triton2flydsl/sglang/gdn_fused_recurrent_decode'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def functions(names, namespace):
    tree = ast.parse((TASK / 'test_kernel_harness.py').read_text())
    nodes = [deepcopy(n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.fix_missing_locations(ast.Module(nodes, type_ignores=[])), '<protected GDN harness CPU>', 'exec'), namespace)


def support(monkeypatch):
    package = types.ModuleType('scripts')
    package.__path__ = [str(TASK / 'scripts')]
    monkeypatch.setitem(sys.modules, 'scripts', package)
    controls = load(TASK / 'scripts/normalization_controls.py', '_gdn_false_controls')
    monkeypatch.setitem(sys.modules, 'scripts.normalization_controls', controls)
    replay = load(TASK / 'scripts/replay_checks.py', '_gdn_false_replay')
    ns = {'require_unchanged': replay.require_unchanged}
    functions({'reference_decode', '_checked_recurrent_output', '_require_unused_state', '_compare_recurrent_output'}, ns)
    return controls, replay, ns


def test_false_reference_matches_independent_closed_forms_and_rejects_forced_normalization(monkeypatch):
    controls, replay, ns = support(monkeypatch)
    records = controls.reference_self_test(ns['reference_decode'], ns['_compare_recurrent_output'])
    assert [r['control'] for r in records] == ['control_qk_false_explicit', 'control_qk_false_default']
    assert all(r['known_answer'] == 'PASS' and r['negative_output'] == 'rejected' for r in records)
    for row in controls.manifest_cases():
        inp, expected = controls.make_case(row, 'cpu')
        originals = {name: value.clone() for name, value in inp.items() if isinstance(value, torch.Tensor)}
        actual = ns['reference_decode'](inp, use_qk_l2norm_in_kernel=False)
        for got, ref in zip(actual, expected):
            assert torch.equal(got, ref)
        for name, value in originals.items():
            assert torch.equal(inp[name], value)
        assert not torch.allclose(ns['reference_decode'](inp)[0], expected[0], atol=.02, rtol=.01)


@pytest.mark.parametrize('behavior', ['correct', 'always_normalize', 'output_only', 'unused_state', 'input_mutation', 'bad_alias'])
def test_real_extra_controls_enforce_output_state_readonly_and_public_return_aliases(behavior, monkeypatch):
    controls, replay, ns = support(monkeypatch)
    calls = []
    def implementation(**kwargs):
        calls.append(dict(kwargs))
        inp = {'B': 2, 'H': 1, 'HV': 2, 'K': 128, 'V': 128,
               'mixed_qkv': kwargs['mixed_qkv'], 'a': kwargs['a'], 'b': kwargs['b'],
               'A_log': kwargs['A_log'], 'dt_bias': kwargs['dt_bias'],
               'scale': kwargs['scale'], 'ssm_states': kwargs['initial_state'],
               'cache_indices': kwargs['ssm_state_indices']}
        actual = ns['reference_decode'](inp, use_qk_l2norm_in_kernel=behavior == 'always_normalize')
        kwargs['out'].copy_(actual[0])
        if behavior != 'output_only':
            kwargs['initial_state'].copy_(actual[1])
        if behavior == 'unused_state':
            kwargs['initial_state'][1].fill_(10)
        if behavior == 'input_mutation':
            kwargs['mixed_qkv'].zero_()
        if behavior == 'bad_alias':
            return kwargs['out'].clone(), kwargs['initial_state']
        return kwargs['out'], kwargs['initial_state']
    mod = SimpleNamespace(fused_recurrent_gated_delta_rule_packed_decode=implementation)
    rows = controls.run_cases(mod, ns['_compare_recurrent_output'], replay.require_unchanged, device='cpu')
    assert len(rows) == 2 and [r['shape_id'] for r in rows] == [11, 12]
    assert all(r['passed'] is (behavior == 'correct') for r in rows)
    assert calls[0]['use_qk_l2norm_in_kernel'] is False
    assert 'use_qk_l2norm_in_kernel' not in calls[1]
    # Inputs are restored even after an actual input-mutation fixture fails.
    if behavior == 'input_mutation':
        assert all(torch.count_nonzero(call['mixed_qkv']) > 0 for call in calls)


def test_public_initial_wrapper_default_and_explicit_flag_dispatch(monkeypatch):
    controls, _, _ = support(monkeypatch)
    calls = []
    class Kernel:
        def __getitem__(self, grid):
            def launch(**kwargs): calls.append((grid, kwargs))
            return launch
    triton, language = types.ModuleType('triton'), types.ModuleType('triton.language')
    language.constexpr = object()
    triton.jit = lambda fn: Kernel()
    triton.cdiv = lambda a, b: (a + b - 1) // b
    triton.next_power_of_2 = lambda n: 1 << (n - 1).bit_length()
    monkeypatch.setitem(sys.modules, 'triton', triton)
    monkeypatch.setitem(sys.modules, 'triton.language', language)
    module = load(TASK / 'gdn_fused_recurrent_decode.py', '_gdn_actual_initial_wrapper')
    inp, _ = controls.make_case(controls.manifest_cases()[0], 'cpu')
    out = torch.empty(2, 1, 2, 128, dtype=torch.bfloat16)
    kwargs = dict(mixed_qkv=inp['mixed_qkv'], a=inp['a'], b=inp['b'], A_log=inp['A_log'],
                  dt_bias=inp['dt_bias'], scale=inp['scale'], initial_state=inp['ssm_states'],
                  out=out, ssm_state_indices=inp['cache_indices'])
    for flag in [None, False, True]:
        supplied = dict(kwargs)
        if flag is not None: supplied['use_qk_l2norm_in_kernel'] = flag
        returned = module.fused_recurrent_gated_delta_rule_packed_decode(**supplied)
        assert returned[0] is out and returned[1] is inp['ssm_states']
        assert calls[-1][0] == (4, 4)
        assert calls[-1][1]['USE_QK_L2NORM_IN_KERNEL'] is (False if flag is None else flag)


def test_manifest_and_actual_action_bind_twelve_correctness_but_ten_timed_rows(monkeypatch):
    controls, _, _ = support(monkeypatch)
    manifest = json.loads((TASK / 'cases.json').read_text())['cases']
    actions = load(TASK / 'scripts/task_actions.py', '_gdn_false_actions')
    assert manifest[10:] == controls.manifest_cases()
    assert len(manifest) == actions.CORRECTNESS_COUNT == 12
    assert len([r for r in manifest if 'performance' in r['checks']]) == 10
    assert set(actions.PERFORMANCE_IDS.values()) == {r['test_case_id'] for r in manifest[:10]}
    rows = [{'shape_id': i, 'passed': True} for i in range(1, 13)]
    h = SimpleNamespace(TASK_DIR=str(TASK), TEST_SHAPES=actions.EXPECTED_SHAPES,
                        run_correctness=lambda: (True, None, rows))
    actions.select_role(h, 'baseline', False)
    assert actions.check(h) == []
    rows[-1]['passed'] = False
    with pytest.raises(RuntimeError, match='Invalid or failed correctness case'):
        actions.check(h)


def test_actual_guard_keeps_new_control_and_manifest_protected(tmp_path):
    workspace = tmp_path / 'task'
    shutil.copytree(TASK, workspace)
    spec = load_task_spec(workspace / 'config.yaml', task_id='triton2flydsl/sglang/gdn_fused_recurrent_decode')
    guard = snapshot_workspace_harness(workspace, task_spec=spec)
    source = workspace / 'gdn_fused_recurrent_decode.py'
    source.write_text(source.read_text() + '\n# allowed candidate implementation edit\n')
    verify_workspace_harness(guard, discard_added=False)
    helper = workspace / 'scripts/normalization_controls.py'
    helper.write_text(helper.read_text() + '\n_unpermitted_oracle_edit = True\n')
    with pytest.raises(RuntimeError):
        verify_workspace_harness(guard, discard_added=False)


def test_real_cpu_validate_task_emits_twelve_manifest_rows_and_false_known_answers():
    process = subprocess.run([sys.executable, 'scripts/evaluate.py', 'validate-task'],
        cwd=TASK, text=True, capture_output=True, timeout=30)
    result = parse_command_result(process.stdout, role='task', action='validate-task', returncode=process.returncode)
    assert result.passed, result.reason
    assert len(result.cases) == 12
    controls = result.metadata['reference_controls']
    assert len(controls) == 6
    assert [r['control'] for r in controls[-2:]] == ['control_qk_false_explicit', 'control_qk_false_default']
    assert all(r['known_answer'] == 'PASS' and r['negative_output'] == 'rejected' for r in controls)
