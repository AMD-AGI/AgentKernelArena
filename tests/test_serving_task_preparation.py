"""First-class recipe discovery and preparation; no GPU qualification claims."""
import hashlib
import compileall
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

from src import magpie_catalog as catalog
from src.serving_runtime import selected_recipe_task_ids, selected_serving_tasks
from src.serving_task_preparation import GENERATOR_INPUTS, prepare_recipe_task
from src.task_spec import TaskConfigError, load_task_spec
from src.tasks import get_task_config
from src.tools.sync_perf_helpers import audit_task_benchmark_entrypoints
from test_magpie_workload_catalog import binding

ROOT = Path(__file__).resolve().parents[1]
CATALOG = catalog.load_catalog()


def inventory(directory):
    return {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob('*') if p.is_file()}


@pytest.fixture
def recipe_project(tmp_path, monkeypatch):
    root = tmp_path/'project'
    for name in GENERATOR_INPUTS:
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, path)
    entry = next(e for e in CATALOG['entries']
                 if e['id'] == 'sglang_mi355x/benchmark_sglang_qwen3_next_80b_fp8')
    snapshot = {**CATALOG, 'entries': [entry]}
    catalog.export_tasks(snapshot, root/'tasks/e2e/magpie', root=root)
    task_id = 'e2e/magpie/' + entry['id']
    source = root/'tasks'/task_id
    profile = tmp_path/'binding.json'
    profile.write_text(json.dumps(binding(entry)))
    config = tmp_path/'run.yaml'
    config.write_text(yaml.safe_dump(dict(tasks=[task_id],
        serving_runtime=dict(bindings={task_id: 'binding.json'}))))
    cache = tmp_path/'cache'
    monkeypatch.setenv('AKA_SERVING_TASK_CACHE', str(cache))
    return root, task_id, source, profile, config, cache


def test_all_catalog_workloads_are_real_discoverable_task_directories():
    expected = {'e2e/magpie/' + e['id'] for e in CATALOG['entries']}
    discovered = get_task_config(str(ROOT/'tasks'), 'e2e/magpie')
    assert set(discovered) == expected
    assert len(discovered) >= 50
    for task_id, path in discovered.items():
        spec = load_task_spec(Path(path), task_id=task_id)
        assert spec.candidate.language == 'gpu'
        assert spec.candidate.editable[0].scope == 'tree'
        assert spec.to_mapping()['evaluation']['measurement']['recipe'] == 'recipe.json'
    assert catalog.export_tasks(CATALOG, ROOT/'tasks/e2e/magpie', check=True) == len(expected)


def test_export_preserves_edits_unless_update_is_explicit(tmp_path):
    snapshot = {**CATALOG, 'entries': CATALOG['entries'][:1]}
    catalog.export_tasks(snapshot, tmp_path)
    path = tmp_path/snapshot['entries'][0]['id']/'README.md'
    path.write_text('Maintainer edit that must not be silently discarded.\n')
    with pytest.raises(ValueError, match='differs'):
        catalog.export_tasks(snapshot, tmp_path)
    assert path.read_text().startswith('Maintainer edit')
    catalog.export_tasks(snapshot, tmp_path, update=True)
    assert catalog.export_tasks(snapshot, tmp_path, check=True) == 1


@pytest.mark.parametrize('changed', ['scripts/evaluate.py', 'upstream.yaml'])
def test_helper_audit_checks_recipe_identity_and_canonical_bootstrap(recipe_project, changed):
    root, task_id, source, profile, config, cache = recipe_project
    counts, problems = audit_task_benchmark_entrypoints(root)
    assert counts == {'serving_recipe': 1} and not problems
    path = source/changed
    path.write_bytes(path.read_bytes() + b'\n# Different declaration.\n')
    counts, problems = audit_task_benchmark_entrypoints(root)
    assert not counts and len(problems) == 1


def test_single_kernel_task_discovery_never_prepares_serving_runtime(tmp_path, monkeypatch):
    root = tmp_path/'project'
    task = root/'tasks/hip2hip/example'
    task.mkdir(parents=True)
    (task/'config.yaml').write_text(yaml.safe_dump(dict(schema_version=2,
        candidate=dict(language='hip', editable=['kernel.hip']),
        evaluation=dict(runner=['python3', 'evaluate.py']))))
    config = tmp_path/'run.yaml'
    config.write_text(yaml.safe_dump(dict(tasks=['hip2hip/example'])))
    cache = tmp_path/'cache'
    monkeypatch.setenv('AKA_SERVING_TASK_CACHE', str(cache))
    before = inventory(root)
    assert selected_serving_tasks(config, root) == {}
    assert selected_recipe_task_ids(config, root) == []
    assert inventory(root) == before and not cache.exists()


def test_selected_recipe_prepares_unchanged_standard_task_outside_checkout(recipe_project):
    root, task_id, source, profile, config, cache = recipe_project
    before = inventory(root)
    tasks = selected_serving_tasks(config, root)
    prepared, spec, lock = tasks[task_id]
    assert prepared.is_relative_to(cache)
    assert not prepared.is_relative_to(root)
    assert inventory(root) == before
    assert not (source/'runtime.lock.json').exists()
    assert spec.task_id == task_id
    assert 'recipe' not in spec.to_mapping()['evaluation']['measurement']
    assert spec.to_mapping()['evaluation']['measurement']['pairs'] == 3
    assert spec.to_mapping()['evaluation']['measurement']['minimum_speedup_ratio'] == 1.05
    assert lock['candidate_bundle']['scope'] == 'whole_model'
    assert (prepared/'scripts/evaluate.py').read_bytes() == (root/'src/tools/perf/catalog_serving.py').read_bytes()
    assert (prepared/'scripts/_aka_serving.py').read_bytes() == (root/'src/tools/perf/serving.py').read_bytes()
    assert (prepared/'upstream.yaml').read_bytes() == (source/'upstream.yaml').read_bytes()
    original = yaml.safe_load((source/'upstream.yaml').read_text())['benchmark']
    fixed = yaml.safe_load((prepared/'benchmark.yaml').read_text())['benchmark']
    for name in ('TP', 'EP', 'CONC', 'ISL', 'OSL', 'RANDOM_RANGE_RATIO', 'EXTRA_SGLANG_ARGS'):
        if name in original['envs']:
            assert fixed['envs'][name] == original['envs'][name]
    assert selected_recipe_task_ids(config, root) == [task_id]


def test_prepared_package_reused_without_rerendering(recipe_project, monkeypatch):
    root, task_id, source, profile, config, cache = recipe_project
    first = selected_serving_tasks(config, root)[task_id][0]
    def unexpected(*args, **kwargs):
        pytest.fail('unchanged prepared task should be verified and reused')
    monkeypatch.setattr(catalog, 'materialize', unexpected)
    assert selected_serving_tasks(config, root)[task_id][0] == first


def test_source_bytecode_is_ignored_but_never_copied_into_executable_package(recipe_project):
    root, task_id, source, profile, config, cache = recipe_project
    assert compileall.compile_dir(source, quiet=1)
    assert list(source.rglob('*.pyc'))
    prepared = selected_serving_tasks(config, root)[task_id][0]
    assert not list(prepared.rglob('*.pyc'))
    bytecode = prepared/'scripts/__pycache__/evaluate.cpython-312.pyc'
    bytecode.parent.mkdir()
    bytecode.write_bytes(b'untrusted executable cache')
    with pytest.raises(ValueError, match='cache changed'):
        selected_serving_tasks(config, root)


@pytest.mark.parametrize('name', ['config.yaml', 'runtime.lock.json', 'scripts/_aka_serving.py', 'correctness.json'])
def test_modified_cache_is_rejected_without_overwriting_evidence(recipe_project, name):
    root, task_id, source, profile, config, cache = recipe_project
    first = selected_serving_tasks(config, root)[task_id][0]
    changed = first/name
    changed.write_text('Changed cached content\n')
    with pytest.raises(ValueError, match='cache changed'):
        selected_serving_tasks(config, root)
    assert changed.read_text() == 'Changed cached content\n'


def test_binding_change_creates_new_immutable_task_preserving_old_one(recipe_project):
    root, task_id, source, profile, config, cache = recipe_project
    old = selected_serving_tasks(config, root)[task_id][0]
    old_files = inventory(old)
    data = json.loads(profile.read_text())
    data['inputs']['correctness.json'][0] = 'Another independently frozen probe.'
    profile.write_text(json.dumps(data))
    new = selected_serving_tasks(config, root)[task_id][0]
    assert old != new and inventory(old) == old_files
    assert json.loads((new/'correctness.json').read_text())[0] == data['inputs']['correctness.json'][0]


def test_helper_update_cannot_reuse_an_older_prepared_evaluator(recipe_project):
    root, task_id, source, profile, config, cache = recipe_project
    old = selected_serving_tasks(config, root)[task_id][0]
    previous = (old/'scripts/_aka_serving.py').read_bytes()
    helper = root/'src/tools/perf/serving.py'
    helper.write_bytes(helper.read_bytes() + b'\n# Reusable evaluator revision.\n')
    new = selected_serving_tasks(config, root)[task_id][0]
    assert new != old
    assert (old/'scripts/_aka_serving.py').read_bytes() == previous
    assert (new/'scripts/_aka_serving.py').read_bytes() == helper.read_bytes()


def test_missing_binding_is_actionable_and_does_not_prepare_anything(recipe_project):
    root, task_id, source, profile, config, cache = recipe_project
    config.write_text(yaml.safe_dump(dict(tasks=[task_id])))
    before = inventory(root)
    with pytest.raises(ValueError, match='serving_runtime.bindings'):
        selected_serving_tasks(config, root)
    assert inventory(root) == before and not cache.exists()


@pytest.mark.parametrize('name', ['config.yaml', 'upstream.yaml', 'source/patches.json', 'recipe.json'])
def test_authored_recipe_mutation_is_not_silently_ignored(recipe_project, name):
    root, task_id, source, profile, config, cache = recipe_project
    path = source/name
    if name.endswith('.yaml'):
        path.write_text(path.read_text() + '\n# Unreviewed source change.\n')
    else:
        data = json.loads(path.read_text())
        data['unreviewed_change'] = True
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        selected_serving_tasks(config, root)
    assert not cache.exists()


def test_source_symlink_cannot_redirect_recipe_preparation(recipe_project):
    root, task_id, source, profile, config, cache = recipe_project
    path = source/'upstream.yaml'
    outside = root.parent/'outside.yaml'
    path.rename(outside)
    path.symlink_to(outside)
    with pytest.raises((ValueError, TaskConfigError)):
        selected_serving_tasks(config, root)
    assert not cache.exists()


def test_recipe_and_lock_cannot_be_declared_editable(tmp_path):
    for field in ('recipe', 'runtime_lock'):
        config = catalog.task_config(recipe='recipe.json')
        config['evaluation']['measurement'][field] = 'source/protected.json'
        path = tmp_path/'config.yaml'
        path.write_text(yaml.safe_dump(config))
        with pytest.raises(TaskConfigError, match='protected'):
            load_task_spec(path, task_id='e2e/recipe')


def test_readonly_metadata_fallback_explains_required_host_python(recipe_project, monkeypatch):
    root, task_id, source, profile, config, cache = recipe_project
    monkeypatch.setenv('AKA_SERVING_METADATA_READONLY', '1')
    with pytest.raises(ValueError, match='AKA_SERVING_PYTHON'):
        selected_serving_tasks(config, root)
    assert not cache.exists()


def test_bypassing_preparation_never_reports_a_passing_task(recipe_project):
    root, task_id, source, profile, config, cache = recipe_project
    result = subprocess.run([sys.executable, str(source/'scripts/evaluate.py'), 'validate-task'],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert 'serving_runtime.bindings' in result.stderr
    assert 'ARENA_EVAL_RESULT=' not in result.stdout


def test_docker_mounts_only_selected_recipe_packages_readonly(tmp_path):
    script = (ROOT/'src/scripts/docker_benchmark.sh').read_text()
    start = script.index('append_serving_task_mounts() {')
    end = script.index('\n}\n', start) + 3
    function = script[start:end]
    task_id = 'e2e/magpie/sglang_mi355x/example'
    template_root = tmp_path/'template root with spaces'
    task = template_root/task_id
    task.mkdir(parents=True)
    (task/'config.yaml').write_text('schema_version: 2\n')
    command = '''
die() { echo "$*" >&2; exit 1; }
docker_args=(-v /checkout:/workspace)
SERVING_RUNTIME_DIR=/run/serving
SERVING_TASK_TEMPLATE_DIR=$1
SERVING_RECIPE_TASKS=$2
CONTAINER_WORKDIR=/workspace
append_serving_task_mounts
printf '%s\\0' "${docker_args[@]}"
'''
    result = subprocess.run(['bash', '-c', function+command, 'test', str(template_root), task_id],
                            capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split(b'\0')[:-1] == [b'-v', b'/checkout:/workspace', b'-v',
        f'{task}:/workspace/tasks/{task_id}:ro'.encode()]
    for invalid in ('e2e/magpie/missing', '../escape', 'bad:mount'):
        result = subprocess.run(['bash', '-c', function+command, 'test', str(template_root), invalid],
                                capture_output=True)
        assert result.returncode != 0
    result = subprocess.run(['bash', '-c', function+command, 'test', str(template_root), ''],
                            capture_output=True)
    assert result.returncode == 0
    assert result.stdout.split(b'\0')[:-1] == [b'-v', b'/checkout:/workspace']
