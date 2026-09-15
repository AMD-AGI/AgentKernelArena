"""Protected operator semantics and packaging invariants for the 21 SIKL tasks."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from src.task_spec import load_task_spec

ROOT = Path(__file__).resolve().parents[1]
SIKL_ROOT = ROOT / 'tasks' / 'SIKL-task'
TASKS = sorted(p.parent for p in SIKL_ROOT.glob('*/config.yaml'))
VAR_AXIS = {'gemm': 'm', 'moe': 'num_tokens'}
SHARED_TEMPLATE_FILES = (
    'kernel.py', 'test_kernel_harness.py', 'scripts/task_inputs.py',
    'scripts/task_initialize.py', 'scripts/task_compare.py', 'scripts/task_reference.py',
    'scripts/task_baseline.py', 'scripts/task_measure.py', 'scripts/task_contract.py',
    'scripts/evaluate.py', 'scripts/export_solution.py', 'scripts/task_validation.py', 'README.md',
)
# SHA256 from the original PR's bundled callbacks before schema migration.
# A migration must not quietly change input distributions or acceptance gates.
CALLBACK_SHA256 = {
    'gemm': {
        'task_baseline': 'eeaf2f83a06a298b2b7c0fe2c2cad75bd4cb4da89ddf6b583f31a6b3de6de66e',
        'task_compare': 'b3237750010516954281047cc26b7045d9a276ed5322f2104a5cecb770f89b1a',
        'task_initialize': '60688988b28b137ee8cfe328077d4c4dba6d7bf5835a002062f03a88a298b89b',
        'task_reference': '4ce9f4075562e14cd48ec6e6c20e6f80deae18f34e213dd6891432001314c663',
    },
    'moe': {
        'task_baseline': '3d27b3fe67b8d030aa1fd13287fcdef7d68e657a47a169469292c22f1dc4600e',
        'task_compare': '37f0e083eb7654da9d3f36da03d3e9648d0845de9ca1ec1282da96d3d4db708c',
        'task_initialize': '80e50d81def9ac0c6f85390155e23eec48c173e8a6b754eaa465e9ed65f2120e',
        'task_reference': 'de096e2726bb4e81e4e748748b044bcaf99576070ace1f0b1662eee763f5d072',
    },
}


def _config(task):
    return yaml.safe_load((task / 'config.yaml').read_text())


def _workload(task):
    return json.loads((task / 'workload.json').read_text())


def test_suite_keeps_all_21_tasks_and_273_cases():
    assert len(TASKS) == 21
    assert sum(_workload(t)['op_type'] == 'gemm' for t in TASKS) == 17
    assert sum(_workload(t)['op_type'] == 'moe' for t in TASKS) == 4
    assert sum(len(_workload(t)['cases']) for t in TASKS) == 273


@pytest.mark.parametrize('op_type', sorted(VAR_AXIS))
@pytest.mark.parametrize('relative', SHARED_TEMPLATE_FILES)
def test_family_copies_are_identical(op_type, relative):
    tasks = [t for t in TASKS if _workload(t)['op_type'] == op_type]
    assert len({(t / relative).read_bytes() for t in tasks}) == 1


@pytest.mark.parametrize('task', TASKS, ids=lambda t: t.name)
def test_one_v2_config_and_no_agent_dependency(task):
    spec = load_task_spec(task / 'config.yaml', task_id=str(task.relative_to(ROOT / 'tasks')))
    assert spec.candidate.language == 'flydsl'
    assert spec.candidate.initial_state == 'unimplemented'
    assert spec.baseline.kind == 'provided'
    assert len(spec.actions) == 7
    config = _config(task)
    assert 'task_type' not in config and 'rewrite_source_file' not in config
    assert not (task / 'scripts/forge_driver.py').exists()
    assert not (task / 'definition.yaml').exists()
    assert len(spec.candidate.entrypoints) == 1
    assert spec.candidate.entrypoints[0].kind == 'builder'
    assert 'builder_symbol' not in _workload(task)  # One declaration, no derivation.
    for path in task.rglob('*.py'):
        source = path.read_text()
        assert 'KERNELFORGE_' not in source
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert (node.module or '').split('.')[0] not in {'src', 'agents', 'kernelforge'}
            elif isinstance(node, ast.Import):
                assert not {'src', 'agents', 'kernelforge'} & {n.name.split('.')[0] for n in node.names}


@pytest.mark.parametrize('task', TASKS, ids=lambda t: t.name)
def test_production_source_is_separate_and_documented(task):
    config = _config(task)
    acquisition, = config['workspace']['sources']
    assert acquisition['kind'] == 'image'
    assert Path(acquisition['image_path']).is_absolute()
    assert acquisition['destination'] != Path(acquisition['image_path']).name
    for source in config['baseline']['source_files']:
        assert source.startswith(acquisition['destination'] + '/')
        assert source in (task / 'README.md').read_text()
        assert source not in config['candidate']['editable']
    assert acquisition['exclude'] == ['jit', '__pycache__']


@pytest.mark.parametrize('task', TASKS, ids=lambda t: t.name)
def test_original_callbacks_and_case_sampling_are_unchanged(task):
    workload = _workload(task)
    kind = workload['op_type']
    for module, sha in CALLBACK_SHA256[kind].items():
        assert hashlib.sha256((task / 'scripts' / f'{module}.py').read_bytes()).hexdigest() == sha
    cases = workload['cases']
    assert [case[VAR_AXIS[kind]] for case in cases] == [2**i for i in range(13)]
    assert len({case['uuid'] for case in cases}) == 13
    for case in cases:
        assert set(case) == {'case_id', 'uuid', VAR_AXIS[kind]}
    assert workload['seed'] == 0
    assert workload['bench'] == {'warmup': 20, 'repetition': 100, 'target_ms': 1.0}
    assert workload['gate_policy']
    for forbidden in ('gate_multiplier', 'gate_floor', 'atol', 'rtol', 'snr_threshold'):
        assert forbidden not in workload
    inputs = (task / 'scripts/task_inputs.py').read_text()
    assert 'task_initialize.run(inputs, seed=SEED)' in inputs
    assert 'task_initialize.run(inputs, seed=REFILL_SEED)' in inputs
    assert 'task_compare.run(got, expected)' in inputs


@pytest.mark.parametrize('task', TASKS, ids=lambda t: t.name)
def test_stub_and_unfilled_solution_are_not_accepted_results(task):
    assert not any(isinstance(n, (ast.FunctionDef, ast.ClassDef)) for n in ast.parse((task / 'kernel.py').read_text()).body)
    template = json.loads((task / 'solution.json').read_text())
    assert template['definition'] == _workload(task)['definition']
    assert template['spec']['entry_point'] == ''
    assert template['sources'] == [{'path': '', 'content': ''}]
    assert template['spec']['target'] == [{'arch': 'gfx950', 'hardware_id': 'MI355X'}]
    config = _config(task)
    assert config['exports'][0]['output'] != 'solution.json'
    assert config['exports'][0]['command'] == ['python3', 'scripts/export_solution.py']
    for path in ('_aka_benchmark.py', 'scripts/_aka_benchmark.py'):
        assert not (task / path).exists()  # Canonical helper is materialized, not forked.
