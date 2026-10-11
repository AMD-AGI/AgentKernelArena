#!/usr/bin/env python3
"""Prepare a clean Arena checkout and verified fixtures before any agent runs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.tools.trusted_task_eval import stage_trusted_task
from src.tools import prepare_head_kernel_artifacts as legacy_artifacts

LEGACY_PROTOCOL = 'headkernel_artifacts_v1'
LEGACY_MANIFEST = 'tools/headkernel-artifacts.json'
TRUSTED_PROTOCOL = 'trusted_task_v1'
PREPARATION_SCOPES = {'refreshed_sg520', 'source_refreshed_sg520'}


def require(value, message):
    if not value:
        raise ValueError(message)


def selected_tasks(catalog, requested):
    rows = [row for row in catalog['tasks'] if row.get('scope') in PREPARATION_SCOPES]
    active = {row['task']: row for row in rows}
    require(len(active) == len(rows), 'Duplicate task IDs in the preparation catalog')
    names = list(requested) if requested else list(active)
    require(names and len(names) == len(set(names)), 'Select distinct catalog task IDs')
    require(all(name in active for name in names), 'Task is outside the explicit preparation catalog selection')
    selected = [active[name] for name in names]
    for row in selected:
        preparation_protocol(row)
    return selected


def preparation_protocol(row):
    """A supported scope never substitutes for an explicit input protocol."""
    protocol = row.get('preparation_protocol')
    require(protocol in (TRUSTED_PROTOCOL, LEGACY_PROTOCOL),
            'Task requires an explicit supported preparation protocol')
    if row.get('scope') == 'source_refreshed_sg520':
        require(protocol == LEGACY_PROTOCOL,
                'Source-refreshed Qwen setup requires headkernel_artifacts_v1')
    if protocol == LEGACY_PROTOCOL:
        legacy_task_name(row['task'])
        require('fixture_manifest' in row and row['fixture_manifest'] is None,
                'Legacy preparation must explicitly set fixture_manifest to null')
        require(row.get('legacy_artifact_manifest') == LEGACY_MANIFEST,
                'Legacy preparation must name the committed original-artifact registry')
        digest = row.get('legacy_artifact_manifest_sha256')
        require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest),
                'Legacy preparation requires a full artifact-registry SHA-256 pin')
    return protocol


def legacy_task_name(task):
    relative = Path(task)
    require(len(relative.parts) == 2 and relative.parts[0] == 'headkernel'
            and relative.name not in ('.', '..'), 'Legacy artifacts require headkernel/TASK')
    return relative.name


def prepare_legacy_task(arena, row, *, runtime_image, use_manifest_oci_prefixes, fixture_local_mirror=None):
    """Hydrate registered original fixtures through their existing verified helper.

    This protocol does not interpret an old GEAK capture as an EXTERNAL-MANIFEST
    or create a trusted-evaluation declaration. Its receipt asserts preparation
    of the committed historical contract only, never qualification or a newer
    workload topology.
    """
    require(row.get('preparation_protocol') == LEGACY_PROTOCOL,
            'Legacy tasks require an explicit headkernel_artifacts_v1 preparation protocol')
    require(row.get('fixture_manifest') is None,
            'Legacy artifacts cannot be passed to the EXTERNAL-MANIFEST staging path')
    require(row.get('legacy_artifact_manifest') == LEGACY_MANIFEST,
            'Legacy protocol must name the committed original-artifact registry')
    manifest_path = arena / LEGACY_MANIFEST
    require(manifest_path.is_file() and not manifest_path.is_symlink(), 'Legacy artifact registry must be a regular file')
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    require(digest == row.get('legacy_artifact_manifest_sha256'), 'Legacy artifact registry pin differs from the trusted commit')
    name = legacy_task_name(row['task'])
    task = arena / 'tasks/headkernel' / name
    config = yaml.safe_load((task / 'config.yaml').read_text())
    require(config.get('headkernel', {}).get('docker') == runtime_image,
            'Legacy task runtime image differs from the selected catalog image')
    require(config.get('trusted_evaluation') is None, 'A trusted-evaluation task must use its own staging protocol')
    manifest = legacy_artifacts.load_manifest(manifest_path)
    artifacts = legacy_artifacts.select_artifacts(manifest, [name])
    suite = arena / 'tasks/headkernel'
    if fixture_local_mirror is not None:
        actions = legacy_artifacts.prepare_artifacts(artifacts, suite, mirror=fixture_local_mirror)
    else:
        require(use_manifest_oci_prefixes, 'Approve the committed original OCI prefix or supply a local mirror')
        actions = legacy_artifacts.download_artifacts(artifacts, suite, manifest.oci_remote_root)
    legacy_artifacts.prepare_artifacts(artifacts, suite, verify_only=True)
    return {'task':row['task'], 'preparation_protocol':LEGACY_PROTOCOL,
            'legacy_artifact_manifest':LEGACY_MANIFEST, 'legacy_artifact_manifest_sha256':digest,
            'fixture_oci_prefix':manifest.oci_remote_root, 'actions':actions,
            'artifacts':[{'path':a.path, 'bytes':a.size_bytes, 'sha256':a.sha256} for a in artifacts],
            'inputs':'verified_original_fixtures' if artifacts else 'explicitly_registered_fixture_free_task',
            'workload_scope':'preserved_committed_task_contract', 'qualification':False}


def requires_legacy_protocol(arena, row):
    """Reject a scope-only catalog edit that would silently omit original data."""
    if not row['task'].startswith('headkernel/'):
        return False
    name = legacy_task_name(row['task'])
    manifest = legacy_artifacts.load_manifest(arena / LEGACY_MANIFEST)
    config = yaml.safe_load((arena / 'tasks/headkernel' / name / 'config.yaml').read_text())
    return name in manifest.tasks and config.get('trusted_evaluation') is None


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
    p.add_argument('--task', action='append', help='task path relative to tasks/; repeat, or omit for all 23 explicitly registered tasks')
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
        protocol = preparation_protocol(row)
        if protocol == LEGACY_PROTOCOL:
            prepared.append(prepare_legacy_task(arena, row,
                runtime_image=catalog['images']['sglang_v0520']['pull_reference'],
                use_manifest_oci_prefixes=a.use_manifest_oci_prefixes,
                fixture_local_mirror=a.fixture_local_mirror))
            continue
        require(not requires_legacy_protocol(arena, row),
                'Registered legacy task requires explicit headkernel_artifacts_v1 preparation; scope alone is insufficient')
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
