"""Prepare serving recipe tasks before any optimizer or GPU action starts.

Task directories contain the workload and its v2 declaration. Runtime-specific
bindings stay in run configuration. Prepared packages are content-addressed,
verified on reuse and mounted read-only at the original task locations. Neither
the task checkout nor an existing experiment is rewritten during preparation.
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile

GENERATOR_INPUTS = (
    'src/magpie_catalog.py', 'src/serving_task_preparation.py', 'src/measurement.py',
    'src/perf_helper_materialization.py',
    'src/tools/perf/catalog_serving.py', 'src/tools/perf/serving_bundle.py',
    'src/tools/perf/serving_backends.py', 'src/tools/perf/serving_scenarios.py',
    'src/tools/perf/serving_request_plan.py', 'src/tools/perf/serving.py',
    'src/tools/perf/catalog_checks.py', 'src/tools/perf/catalog_instructions.md',
    'tasks/e2e/MAGPIE_LICENSE', 'tasks/e2e/qwen3_next_80b_sglang/scripts/quality.py',
    'tasks/e2e/qwen3_next_80b_sglang/QUALITY_LICENSE',
)


def recipe_path(config: dict) -> str | None:
    return config.get('evaluation', {}).get('measurement', {}).get('recipe')


def _digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _json(value) -> bytes:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()


def _files(directory: Path, *, source_names=None) -> dict[str, str]:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError('Serving task package must be a real directory')
    files = {}
    for path in sorted(directory.rglob('*')):
        relative = path.relative_to(directory)
        # Source-tree compilation and editor caches never enter the prepared
        # package. The executable cache below stays strict: accepting extra
        # bytecode there would undermine its source-content identity.
        if source_names is not None and (set(relative.parts) & {'__pycache__', '.pytest_cache'}
                                         or path.name == '.DS_Store'):
            continue
        if path.is_symlink():
            raise ValueError('Serving task packages must not contain symlinks')
        if path.is_file():
            name = relative.as_posix()
            if source_names is not None and name not in source_names:
                raise ValueError('Unexpected file in serving recipe task: ' + name)
            files[name] = _digest(path.read_bytes())
        elif not path.is_dir():
            raise ValueError('Serving task package contains a non-regular file')
    return files


def recipe_identity(directory: Path, task_id: str, descriptor_name='recipe.json') -> dict:
    """Dependency-free source identity, shared by preparation and helper auditing."""
    def source_file(name):
        if not isinstance(name, str) or not name or PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts:
            raise ValueError('Serving recipe source path escapes the task')
        path = directory/name
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError('Serving recipe source path escapes the task')
        return path
    descriptor = json.loads(source_file(descriptor_name).read_text())
    if (not isinstance(descriptor, dict)
            or set(descriptor) != {'version', 'kind', 'repository', 'revision', 'id',
                                   'upstream_path', 'sha256', 'source'}
            or type(descriptor.get('version')) is not int or descriptor['version'] != 1
            or descriptor.get('kind') != 'magpie'):
        raise ValueError('Unsupported serving recipe descriptor')
    if not re.fullmatch(r'[0-9a-f]{40}', descriptor.get('revision', '')):
        raise ValueError('Serving recipe requires an immutable upstream revision')
    if descriptor['repository'] != 'https://github.com/AMD-AGI/Magpie':
        raise ValueError('Magpie recipe requires its declared upstream repository')
    entry = {key: descriptor[key] for key in ('id', 'upstream_path', 'sha256')}
    upstream = PurePosixPath(entry['upstream_path'])
    entry['source'] = source_file(descriptor['source']).read_text()
    if (_digest(entry['source'].encode()) != entry['sha256']
            or upstream.suffix not in {'.yaml', '.yml'}
            or upstream.relative_to('examples/benchmarks').with_suffix('').as_posix() != entry['id']
            or task_id != 'e2e/magpie/' + entry['id']):
        raise ValueError('Serving recipe identity or original workload bytes changed')
    return dict(version=1, repository=descriptor['repository'],
                revision=descriptor['revision'], entries=[entry])


def recipe_catalog(directory: Path, config: dict, task_id: str, *, root: Path) -> dict:
    from . import magpie_catalog as catalog

    value = recipe_identity(directory, task_id, recipe_path(config))
    entry = value['entries'][0]
    expected = catalog.recipe_task_files(value, entry, root=root)
    if _files(directory, source_names=expected) != {name: _digest(body) for name, body in expected.items()}:
        raise ValueError('Serving recipe task differs from its generated declaration; '
                         'review the change and regenerate task definitions before running')
    return value


def binding_paths(config: dict, config_path: Path) -> dict[str, Path]:
    from .task_spec import relative_path

    settings = config.get('serving_runtime', {})
    if not isinstance(settings, dict) or set(settings) - {'bindings'}:
        raise ValueError('serving_runtime only supports bindings')
    bindings = settings.get('bindings', {})
    if not isinstance(bindings, dict):
        raise ValueError('serving_runtime.bindings must map task IDs to JSON binding files')
    paths = {}
    for task_id, value in bindings.items():
        relative_path(task_id, 'serving_runtime.bindings task ID')
        if not isinstance(value, str) or not value.strip():
            raise ValueError('Serving runtime binding path must be a nonempty string')
        path = Path(value)
        paths[task_id] = path if path.is_absolute() else config_path.parent/path
    return paths


def prepare_recipe_task(directory: Path, config: dict, task_id: str, binding_path: Path,
                        *, cache: Path, root: Path) -> Path:
    from . import magpie_catalog as catalog
    from .task_spec import load_task_spec

    if os.environ.get('AKA_SERVING_METADATA_READONLY') == '1':
        raise ValueError('Serving recipe tasks require host Python with PyYAML. '
                         'Set AKA_SERVING_PYTHON to a dedicated Python environment.')
    snapshot = recipe_catalog(directory, config, task_id, root=root)
    binding = json.loads(binding_path.read_text())
    # Include every generator input, not just the recipe and runtime profile.
    # A helper or license update must not silently reuse an older task package.
    source_names = catalog.recipe_task_files(snapshot, snapshot['entries'][0], root=root)
    identity = dict(version=1, task_id=task_id, recipe_files=_files(directory, source_names=source_names),
                    binding_sha256=_digest(_json(binding)),
                    helper_files={name: _digest((root/name).read_bytes()) for name in GENERATOR_INPUTS})
    cache = cache.expanduser().resolve()
    if cache.is_relative_to(root.resolve()):
        raise ValueError('Prepared serving task cache must be outside the checkout')
    key = _digest(_json(identity))
    destination = cache/key

    def verify():
        if destination.is_symlink():
            raise ValueError('Prepared serving task cache must not be a symlink')
        manifest_path = destination/'manifest.json'
        if manifest_path.is_symlink():
            raise ValueError('Prepared serving task manifest must not be a symlink')
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('identity') != identity or manifest.get('files') != _files(destination/'task'):
            raise ValueError('Prepared serving task cache changed; preserve it and use a new cache root')
        return destination/'task'

    if destination.exists():
        return verify()
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.prepare-task-', dir=cache) as temp:
        staging = Path(temp)/'package'
        staging.mkdir()
        task = catalog.materialize(snapshot, snapshot['entries'][0]['id'], binding,
                                   staging/'task', root=root)
        load_task_spec(task/'config.yaml', task_id=task_id)
        manifest = dict(identity=identity, files=_files(task))
        (staging/'manifest.json').write_bytes(_json(manifest) + b'\n')
        try:
            staging.rename(destination)
        except OSError as exc:
            if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY) or not destination.is_dir():
                raise
            # Another preparer may have completed the identical snapshot.
            # Never overwrite its package or accept it without verification.
    return verify()
