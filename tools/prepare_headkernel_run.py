#!/usr/bin/env python3
"""Prepare a clean Arena checkout and verified fixtures before any agent runs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.tools.trusted_task_eval import stage_trusted_task


def require(value, message):
    if not value:
        raise ValueError(message)


def selected_tasks(catalog, requested):
    active = {row['task']: row for row in catalog['tasks'] if row.get('scope') == 'refreshed_sg520'}
    names = list(requested) if requested else list(active)
    require(names and len(names) == len(set(names)), 'Select distinct refreshed task IDs')
    require(all(name in active for name in names), 'Task is outside the refreshed selection; Qwen uses its preserved setup')
    return [active[name] for name in names]


def install_prepared_task(arena, task_path, stage, originals):
    """Swap the whole verified tree, retaining the clean Git tree for review."""
    relative = Path(task_path)
    require(not relative.is_absolute() and relative.parts[0] == 'tasks' and '..' not in relative.parts,
            'Use a repository-relative task path')
    receipt = json.loads((stage / 'staging_receipt.json').read_text())
    require(receipt['status'] == 'staged_not_evaluated' and receipt['task_path'] == task_path,
            'Staged task receipt does not match the selection')
    source, destination = stage / 'task', arena / relative
    require(source.is_dir() and not source.is_symlink() and destination.is_dir() and not destination.is_symlink(),
            'Prepared and Git task roots must be regular directories')
    saved = originals / relative
    saved.parent.mkdir(parents=True, exist_ok=True)
    require(not saved.exists(), 'Original task was already preserved')
    destination.rename(saved)
    try:
        source.rename(destination)
    except BaseException:
        saved.rename(destination)
        raise
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=ROOT)
    p.add_argument('--commit', required=True, help='full trusted Git commit')
    p.add_argument('--output', type=Path, required=True, help='new directory outside the source checkout')
    p.add_argument('--scratch-dir', type=Path, required=True)
    p.add_argument('--task', action='append', help='headkernel/TASK; repeat, or omit for all 18 refreshed tasks')
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--use-manifest-oci-prefixes', action='store_true', help='approve the exact committed task prefixes')
    source.add_argument('--fixture-local-mirror', type=Path, help='object-key layout; supported for one task at a time')
    p.add_argument('--timeout', type=int, default=7200)
    a = p.parse_args()
    require(not Path('/.dockerenv').exists(), 'Prepare on the trusted host before launching an agent container')
    require(re.fullmatch('[0-9a-f]{40}', a.commit), 'Provide the full trusted commit SHA')
    repo, output = a.repo.resolve(), a.output.absolute()
    require(not output.exists() and not output.is_relative_to(repo), 'Use a new output directory outside the source checkout')
    output.mkdir(parents=True)
    arena = output / 'arena'
    subprocess.run(['git','clone','--quiet','--no-hardlinks','--no-checkout',str(repo),str(arena)], check=True)
    subprocess.run(['git','-C',str(arena),'checkout','--quiet','--detach',a.commit], check=True)
    catalog = json.loads((arena / 'tools/headkernel-runtime-targets.json').read_text())
    tasks = selected_tasks(catalog, a.task)
    require(a.fixture_local_mirror is None or len(tasks) == 1, 'A local mirror must select one task explicitly')
    prepared = []
    for row in tasks:
        task_path = 'tasks/' + row['task']
        manifest_path = row.get('fixture_manifest')
        if manifest_path is None:
            prepared.append({'task':row['task'], 'inputs':'self_contained_declared_generated_cases'})
            continue
        manifest_file = arena / manifest_path
        require(hashlib.sha256(manifest_file.read_bytes()).hexdigest() == row['fixture_manifest_sha256'],
                'Catalog fixture manifest pin differs from the trusted commit')
        manifest = json.loads(manifest_file.read_text())
        stage = output / 'stages' / row['task'].replace('/', '__')
        result = stage_trusted_task(repo=repo, commit=a.commit, task_path=task_path,
            output=stage, scratch_dir=a.scratch_dir,
            fixture_local_mirror=a.fixture_local_mirror,
            fixture_oci_prefix=manifest['oci_prefix'] if a.use_manifest_oci_prefixes else None,
            fixture_max_files=max(16384, len(manifest['assets'])),
            fixture_max_bytes=max(64 << 30, sum(item['bytes'] for item in manifest['assets'])),
            timeout=a.timeout)
        require(result['trusted_commit'] == a.commit, 'Stage receipt belongs to another trusted commit')
        receipt = install_prepared_task(arena, task_path, stage, output / 'original-tasks')
        prepared.append({'task':row['task'], 'stage_receipt':str((stage/'staging_receipt.json').relative_to(output)),
                         'package_sha256':receipt['package_sha256']})
    subprocess.run(['git','-C',str(arena),'diff','--exit-code','--','tasks'], check=True)
    import yaml
    config = {'agent':{'template':'task_validator'}, 'tasks':[row['task'] for row in tasks],
              'target_gpu_model':'MI355X', 'log_directory':'logs', 'workspace_directory_prefix':'workspace'}
    config_path = arena / 'example_configs/prepared_headkernel_run.yaml'
    require(not config_path.exists(), 'Prepared run config already exists')
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    report = {'status':'staged_not_evaluated', 'trusted_commit':a.commit, 'tasks':prepared,
              'arena':'arena', 'run_config':'example_configs/prepared_headkernel_run.yaml',
              'runtime_image':catalog['images']['sglang_v0520']['pull_reference'],
              'qualification':False, 'suite_ready':False, 'GPU_actions':False}
    (output/'PREPARATION.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
