"""The ordinary workspace guard owns trusted task contracts across optimization."""
from pathlib import Path

import pytest
import yaml

from src.harness_guard import (
    describe_workspace_harness,
    snapshot_workspace_harness,
    verify_workspace_harness,
)


def make_task(root):
    root.mkdir(parents=True, exist_ok=True)
    config = {
        'task_type': 'triton2triton',
        'source_file_path': ['source/kernel.py'],
        'target_kernel_functions': ['kernel'],
        'performance_command': ['python3 scripts/task_runner.py performance'],
        'trusted_evaluation': {
            'schema_version': 1,
            'case_manifest': 'cases.json',
            'contract_file': 'ut/evaluation_contract.py',
            'source_guard': 'ut/source_guard.py',
            'fixture_manifest': 'fixtures/EXTERNAL-MANIFEST.json',
            'reference_sources': {'source/kernel.py': 'ut/reference/kernel.py'},
        },
    }
    (root / 'config.yaml').write_text(yaml.safe_dump(config))
    files = {
        'source/kernel.py': 'def kernel():\n    return 1\n',
        'source/native_wrapper.py': 'def invoke():\n    return kernel()\n',
        'scripts/task_runner.py': 'print("protected entrypoint")\n',
        'ut/evaluation_contract.py': 'WARMUPS = 10\nSAMPLES = 100\n',
        'ut/source_guard.py': 'def validate_sources(*args): return True\n',
        'ut/runtime.py': 'def measure(): return real_device_time()\n',
        'ut/reference/kernel.py': 'def kernel():\n    return 1\n',
        'ut/reference.py': 'def reference(x): return x * 2\n',
        'ut/build/layout.py': 'SORTED_LAYOUT = 32\n',
        'cases.json': '{"cases": ["small", "large"]}\n',
        'task_definition.json': '{"tolerance": 0.02}\n',
        'SOURCE-PROVENANCE.json': '{"image": "pinned"}\n',
        'fixtures/EXTERNAL-MANIFEST.json': '{"assets": ["payload.bin"]}\n',
        'fixtures/payload.bin': 'captured-storage',
        'fixtures/logs/control.json': '{"work": 64}\n',
    }
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return root


PROTECTED = ['ut/runtime.py', 'ut/reference.py', 'ut/source_guard.py', 'ut/evaluation_contract.py',
             'ut/reference/kernel.py', 'cases.json', 'task_definition.json', 'SOURCE-PROVENANCE.json',
             'fixtures/EXTERNAL-MANIFEST.json', 'fixtures/payload.bin', 'source/native_wrapper.py',
             'ut/build/layout.py', 'fixtures/logs/control.json']


@pytest.mark.parametrize('with_task_root', [False, True])
@pytest.mark.parametrize('relative', PROTECTED)
def test_changed_contract_helpers_data_and_host_wrappers_reject_scoring(tmp_path, with_task_root, relative):
    original = make_task(tmp_path / 'original')
    workspace = make_task(tmp_path / 'workspace')
    snapshot = snapshot_workspace_harness(workspace, **({'task_root': original} if with_task_root else {}))
    assert relative in snapshot.digests
    (workspace / relative).write_text('agent-supplied replacement')
    with pytest.raises(RuntimeError, match='kernel score is rejected'):
        verify_workspace_harness(snapshot)


@pytest.mark.parametrize('with_task_root', [False, True])
def test_declared_gpu_source_remains_editable_and_reported_boundary_matches_snapshot(tmp_path, with_task_root):
    original = make_task(tmp_path / 'original')
    workspace = make_task(tmp_path / 'workspace')
    snapshot = snapshot_workspace_harness(workspace, **({'task_root': original} if with_task_root else {}))
    description = describe_workspace_harness(workspace)
    assert set(description['protected_paths']) == set(snapshot.digests)
    assert 'source/kernel.py' not in snapshot.digests
    (workspace / 'source/kernel.py').write_text('def kernel():\n    probe = 0\n    return 1 + probe\n')
    verify_workspace_harness(snapshot)


def test_deleting_required_metadata_fails_closed(tmp_path):
    workspace = make_task(tmp_path)
    snapshot = snapshot_workspace_harness(workspace)
    (workspace / 'cases.json').unlink()
    with pytest.raises(RuntimeError, match='Missing or unsafe|deleted='):
        verify_workspace_harness(snapshot)


def test_changing_opt_in_cannot_drop_preexisting_immutable_paths(tmp_path):
    workspace = make_task(tmp_path)
    snapshot = snapshot_workspace_harness(workspace)
    (workspace / 'config.yaml').write_text('task_type: triton2triton\n')
    (workspace / 'ut/runtime.py').write_text('def measure(): return 0.000001\n')
    with pytest.raises(RuntimeError, match='ut/runtime.py'):
        verify_workspace_harness(snapshot)


def test_only_runtime_outputs_are_excluded_and_new_contract_helpers_are_removed(tmp_path):
    workspace = make_task(tmp_path)
    for name in ('build/scripts/generated.py', '.validator_audit/tests/stream.log',
                 '.validator_torch_extensions/build/extension.so', 'ut/__pycache__/cached.pyc'):
        p = workspace / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('runtime output')
    snapshot = snapshot_workspace_harness(workspace)
    assert not any(name.startswith(('build/', '.validator_')) or '__pycache__' in name
                   for name in snapshot.digests)
    (workspace / '.validator_audit/tests/stream.log').write_text('new runtime output')
    injected = workspace / 'ut/injected_reference.py'
    injected.write_text('def reference(): return 0\n')
    verify_workspace_harness(snapshot)
    assert not injected.exists()


def test_contract_file_cannot_be_declared_editable(tmp_path):
    workspace = make_task(tmp_path)
    path = workspace / 'config.yaml'
    config = yaml.safe_load(path.read_text())
    config['source_file_path'].append('ut/evaluation_contract.py')
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(RuntimeError, match='declared editable'):
        snapshot_workspace_harness(workspace)


HEADKERNEL = Path(__file__).resolve().parents[1] / 'tasks/headkernel'
TRUSTED_TASKS = [p.parent for p in sorted(HEADKERNEL.glob('*/config.yaml'))
                 if yaml.safe_load(p.read_text()).get('trusted_evaluation')]


@pytest.mark.parametrize('task', TRUSTED_TASKS, ids=lambda p: p.name)
def test_real_trusted_packages_expose_the_same_default_and_optimizer_boundary(task):
    config = yaml.safe_load((task / 'config.yaml').read_text())
    descriptor = config['trusted_evaluation']
    default = snapshot_workspace_harness(task)
    ordinary = snapshot_workspace_harness(task, task_root=task)
    reported = set(describe_workspace_harness(task)['protected_paths'])
    assert set(default.digests) == set(ordinary.digests) == reported
    critical = {descriptor['case_manifest'], descriptor['contract_file'], descriptor['source_guard'],
                *descriptor['reference_sources'].values()}
    if 'fixture_manifest' in descriptor:
        critical.add(descriptor['fixture_manifest'])
    critical.update(p.relative_to(task).as_posix() for p in (task / 'ut').rglob('*')
                    if p.is_file() and '__pycache__' not in p.parts)
    assert critical <= reported
    assert not set(config['source_file_path']) & reported
