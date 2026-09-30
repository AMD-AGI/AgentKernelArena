"""Explicit, host-only acquisition of immutable serving dependencies and weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request

from .serving_runtime import selected_serving_tasks


def file_digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def acquire(url, target):
    with urllib.request.urlopen(url, timeout=120) as source, target.open('wb') as out:
        shutil.copyfileobj(source, out)


def verify(directory, identity):
    manifest = json.loads((directory / 'artifact_manifest.json').read_text())
    if manifest['identity'] != identity:
        raise ValueError('Cached runtime identity changed')
    files = {str(p.relative_to(directory)): file_digest(p) for p in directory.rglob('*')
             if p.is_file() and p.name != 'artifact_manifest.json'}
    if files != manifest['files']:
        raise ValueError('Cached runtime content changed')


def prepare(lock, cache):
    identity = {key: lock[key] for key in ('model', 'dependencies')}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    destination = cache / key
    if destination.exists():
        verify(destination, identity)
        return destination
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.prepare-', dir=cache) as tmp:
        directory = Path(tmp)
        for name, dep in lock['dependencies'].items():
            archive = directory / (name + '.tar.gz')
            acquire(dep['repository'].removesuffix('.git') + '/archive/' + dep['revision'] + '.tar.gz', archive)
            staging = directory / ('extract-' + name)
            with tarfile.open(archive) as source:
                for member in source:
                    path = (staging / member.name).resolve()
                    if not path.is_relative_to(staging.resolve()) or not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                        raise ValueError('Unsafe archive member')
                    if member.issym() or member.islnk():
                        target = (path.parent / member.linkname) if member.issym() else (staging / member.linkname)
                        if Path(member.linkname).is_absolute() or not target.resolve().is_relative_to(staging.resolve()):
                            raise ValueError('Archive link escapes source root')
                source.extractall(staging, filter='data')
            roots = list(staging.iterdir())
            if len(roots) != 1:
                raise ValueError('Unexpected source archive layout')
            roots[0].rename(directory / name)
            staging.rmdir()
            archive.unlink()
        model = directory / 'model'
        model.mkdir()
        base = 'https://huggingface.co/' + lock['model']['id']
        # Exact revision, never the model's moving default branch.
        with urllib.request.urlopen('https://huggingface.co/api/models/' + lock['model']['id'] + '/revision/' + lock['model']['revision']) as response:
            details = json.load(response)
        if details['sha'] != lock['model']['revision']:
            raise ValueError('Model revision mismatch')
        for item in details['siblings']:
            name = item['rfilename']
            target = model / name
            if not target.resolve().is_relative_to(model.resolve()):
                raise ValueError('Unsafe model filename')
            if target.suffix not in {'.json', '.safetensors', '.txt', '.model'} and target.name != 'LICENSE':
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            acquire(base + '/resolve/' + lock['model']['revision'] + '/' + name, target)
        manifest = dict(identity=identity, files={str(p.relative_to(directory)): file_digest(p)
                                                for p in directory.rglob('*') if p.is_file()})
        (directory / 'artifact_manifest.json').write_text(json.dumps(manifest, indent=2))
        directory.rename(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path.home() / '.cache/aka-serving')
    args = parser.parse_args()
    for task_id, (_, _, lock) in selected_serving_tasks(args.config, Path.cwd()).items():
        print(task_id, prepare(lock, args.cache))


if __name__ == '__main__':
    main()
