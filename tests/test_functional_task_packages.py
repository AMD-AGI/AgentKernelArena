"""CPU packaging checks for functional task drafts; these do not qualify kernels."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from src.perf_helper_materialization import (
    canonical_aka_helper,
    materialize_perf_helpers_in_workspace,
)
from src.task_protocol import CaseManifest, parse_command_result
from src.task_spec import load_task_spec, resolve_task_path

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / 'tasks/Aiter-task'
TASKS = sorted(p.parents[1] for p in SUITE.glob('*/scripts/workload.json'))
ORIGINAL_PACKAGES = '9ab5ddb238c4704985604285d9491e0d1820c9a5'
_A8W8 = 'gemm_a8w8_blockwise_scaled_blk128x128_nt_obfloat16_bshuf16x16_'
# Packages added after the original publication, built from the same bundle.
ADDED_PACKAGES = {_A8W8 + 'asraw_n1024_k4096', _A8W8 + 'aslogical_n1024_k4096'}
# AITER's tuned asm split-K row for M=128 in this model config file.
DIAGNOSTIC_EVIDENCE = {name: '65246705468a77baacc29af9831825efdbba78b8aab5e324d484463f4ddfea97'
                       for name in ADDED_PACKAGES}
SHARED_HARNESS = ('scripts/task_api.py', 'scripts/task_inputs.py', 'scripts/task_policy.py',
                  'scripts/task_runner.py', 'scripts/task_timing.py', 'scripts/export_solution.py',
                  'test_kernel_harness.py', 'kernel.py')
DSV4_FAMILY_POLICY = 'deepseek-v4-flash family policy'


def test_each_task_has_exactly_one_supported_workload_layout():
    assert TASKS
    for config in SUITE.glob('*/config.yaml'):
        root = config.parent
        assert sum((root / path).is_file() for path in
                   ('workload.json', 'scripts/workload.json')) == 1, root.name


@pytest.mark.parametrize('task', TASKS, ids=lambda t: t.name)
def test_config_paths_and_complete_manifest(task):
    spec = load_task_spec(task / 'config.yaml', task_id=f'Aiter-task/{task.name}')
    config = spec.to_mapping()
    assert spec.candidate.initial_state == 'unimplemented'
    assert spec.candidate.language == 'flydsl'
    assert config['candidate']['editable'] == ['kernel.py']
    assert [(e.file, e.kind, e.symbol) for e in spec.candidate.entrypoints] == [
        ('kernel.py', 'builder', f'build_{task.name}_module')]
    assert spec.baseline.kind == 'provided'
    assert spec.baseline.correctness_policy == 'diagnostic'
    assert spec.baseline.diagnostic_reason.startswith(DSV4_FAMILY_POLICY)
    readme = (task / 'BUNDLE_README.md').read_text()
    assert '## Baseline numerical policy' in readme
    if task.name in DIAGNOSTIC_EVIDENCE:
        assert DIAGNOSTIC_EVIDENCE[task.name] in spec.baseline.diagnostic_reason
        assert '## Production baseline numerical evidence' in readme
    else:
        assert '## Production baseline numerical evidence' not in readme
    assert [(e['format'], e['output'], e['command']) for e in config['exports']] == [
        ('sikl-solution', 'artifacts/solution.json', ['python3', 'scripts/export_solution.py'])]
    assert config['platform_support']['required_arch'] == 'gfx950'
    assert {(a.role, a.action) for a in spec.actions} == {
        ('task', 'validate-task'),
        *((role, action) for role in ('baseline', 'candidate')
          for action in ('compile', 'correctness', 'performance')),
    }
    for path in (config['instructions'] + config['baseline']['source_files']
                 + config['candidate']['editable']
                 + [config['evaluation']['workloads'], config['evaluation']['runner'][1]]):
        assert resolve_task_path(task, path, must_exist=True).is_file()
    data = json.loads((task / config['evaluation']['workloads']).read_text())
    assert data['definition']['name'] == task.name
    assert len(data['rows']) == len(data['cases']) >= 13
    rows_by_id = {row['workload']['uuid']: row for row in data['rows']}
    cases_by_id = {case['test_case_id']: case for case in data['cases']}
    assert len(rows_by_id) == len(data['rows'])
    assert len(cases_by_id) == len(data['cases'])
    # Runtime-length cases may extend a workload, but must retain every
    # originally published shape and case identity.
    shown = subprocess.run([
        'git', 'show',
        f'{ORIGINAL_PACKAGES}:{task.relative_to(ROOT)}/scripts/workload.json',
    ], cwd=ROOT, capture_output=True, text=True)
    if task.name in ADDED_PACKAGES:
        assert shown.returncode != 0
        original = {'rows': [], 'cases': []}
    else:
        original = json.loads(shown.stdout)
    for row in original['rows']:
        current = rows_by_id[row['workload']['uuid']]
        for key in ('definition', 'solution', 'evaluation'):
            assert current[key] == row[key]
        for key, value in row['workload'].items():
            assert current['workload'][key] == value
    for case in original['cases']:
        current = cases_by_id[case['test_case_id']]
        for key, value in case.items():
            if key == 'params':
                for parameter, setting in value.items():
                    assert current[key][parameter] == setting
            else:
                assert current[key] == value
    assert [row['workload']['uuid'] for row in data['rows']] == [
        case['test_case_id'] for case in data['cases']]
    result = parse_command_result('ARENA_EVAL_RESULT=' + json.dumps({
        'protocol': 'arena-eval-v1', 'role': 'task', 'action': 'validate-task',
        'status': 'PASS', 'cases': data['cases'],
    }), role='task', action='validate-task', returncode=0)
    manifest = CaseManifest.from_result(result)
    assert len(manifest.cases) == len(data['cases'])
    assert all(set(case['checks']) == {'correctness', 'performance'}
               for case in manifest.cases)


@pytest.mark.parametrize('relative', SHARED_HARNESS)
def test_harness_copies_are_identical(relative):
    assert len({(task / relative).read_bytes() for task in TASKS}) == 1


@pytest.mark.parametrize('task', TASKS, ids=lambda t: t.name)
def test_initial_candidate_is_the_unimplemented_target(task):
    tree = ast.parse((task / 'kernel.py').read_text())
    assert not any(isinstance(node, (ast.FunctionDef, ast.ClassDef)) for node in tree.body)
    assert not (task / 'source').exists()
    template = json.loads((task / 'solution.json').read_text())
    assert template['definition'] == task.name
    assert template['spec']['entry_point'] == ''
    assert template['sources'] == [{'path': '', 'content': ''}]
    assert template['spec']['target'] == [{'arch': 'gfx950', 'hardware_id': 'MI355X'}]
    data = json.loads((task / 'scripts/workload.json').read_text())
    assert data['bundle_readme'] == (task / 'BUNDLE_README.md').read_text()
    held = data['policy']['persistent_inputs']
    tensors = {name for name, spec in data['definition']['inputs'].items() if spec.get('shape') is not None}
    assert held and set(held) < tensors


@pytest.mark.parametrize('task', TASKS, ids=lambda t: t.name)
def test_materialized_copy_is_self_contained(task, tmp_path):
    copy = tmp_path / 'isolated'
    shutil.copytree(task, copy)
    assert not (copy / 'scripts/_aka_benchmark.py').exists()
    changed = materialize_perf_helpers_in_workspace(copy)
    helper = copy / 'scripts/_aka_benchmark.py'
    assert helper in changed
    assert helper.read_text() == canonical_aka_helper(ROOT)
    assert materialize_perf_helpers_in_workspace(copy) == []
    for path in copy.rglob('*.py'):
        tree = ast.parse(path.read_text(), filename=str(path))
        compile(tree, str(path), 'exec')
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert (node.module or '').split('.')[0] not in {'src', 'agents'}
            elif isinstance(node, ast.Import):
                assert not {'src', 'agents'} & {n.name.split('.')[0] for n in node.names}


@pytest.mark.parametrize('task', TASKS, ids=lambda t: t.name)
def test_isolated_cli_rejects_missing_gpu_with_all_case_evidence(task, tmp_path):
    pytest.importorskip('torch')
    copy = tmp_path / 'isolated'
    shutil.copytree(task, copy)
    materialize_perf_helpers_in_workspace(copy)
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    result = subprocess.run([
        sys.executable, '-c',
        "import runpy, sys, torch; torch.cuda.is_available = lambda: False; "
        "sys.argv = ['scripts/task_runner.py', 'validate-task']; "
        "runpy.run_path('scripts/task_runner.py', run_name='__main__')",
    ], cwd=copy, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 1, result.stderr
    report = parse_command_result(result.stdout, role='task', action='validate-task',
                                  returncode=result.returncode)
    assert not report.passed
    assert 'compatible ROCm GPU' in report.reason
    data = json.loads((copy / 'scripts/workload.json').read_text())
    assert [case['test_case_id'] for case in report.cases] == [
        case['test_case_id'] for case in data['cases']]
    assert all(case['status'] == 'FAIL' for case in report.cases)
