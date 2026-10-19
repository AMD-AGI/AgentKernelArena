"""Explicit, host-only acquisition of immutable serving dependencies and weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile
import tempfile
import urllib.request
from urllib.parse import urlsplit

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


def runtime_files(directory):
    # Host-prepared assets may be read-only shared directories. Walk each root
    # explicitly so a top-level model symlink cannot disappear from validation.
    paths = []
    for child in directory.iterdir():
        paths.extend(child.rglob('*') if child.is_dir() else [child])
    return [p for p in paths if p.is_file() and p != directory/'artifact_manifest.json'
            and not (p.relative_to(directory).parts[0] == 'model'
                     and any(part.startswith('.') for part in p.relative_to(directory).parts[1:]))]


def verify(directory, identity):
    manifest = json.loads((directory / 'artifact_manifest.json').read_text())
    if manifest['identity'] != identity:
        raise ValueError('Cached runtime identity changed')
    files = {str(p.relative_to(directory)): file_digest(p) for p in runtime_files(directory)}
    if files != manifest['files']:
        raise ValueError('Cached runtime content changed')


def asset_identity(lock):
    identity = {key: lock[key] for key in ('model', 'dependencies')}
    if lock.get('assets'):
        identity['assets'] = lock['assets']
    return identity


class VerifiedRuntime:
    """Bind every action to the original immutable asset inventory and contents.

    Metadata is useful for rejecting changes early, but is not a content cache:
    rapid writes can share a ctime tick on container and network filesystems.
    Hash on every action, outside benchmark timing and inside its setup budget.
    This assumes a trusted host/filesystem, not protection against host root.
    """
    def __init__(self):
        self.verified = {}

    @staticmethod
    def fingerprint(directory):
        rows = []
        for path in [directory/'artifact_manifest.json', *runtime_files(directory)]:
            value = path.stat()
            rows.append((str(path.relative_to(directory)), value.st_dev, value.st_ino,
                         value.st_size, value.st_mtime_ns, value.st_ctime_ns))
        return tuple(sorted(rows))

    def verify(self, directory, identity):
        before = self.fingerprint(directory)
        key = str(directory.resolve())
        record = (json.dumps(identity, sort_keys=True), before,
                  file_digest(directory/'artifact_manifest.json'))
        if key in self.verified:
            if self.verified[key] != record:
                raise ValueError('Verified runtime assets changed during the experiment')
        try:
            verify(directory, identity)
        except ValueError as exc:
            if key in self.verified:
                raise ValueError('Verified runtime assets changed during the experiment') from exc
            raise
        if self.fingerprint(directory) != before:
            raise ValueError('Runtime assets changed during content verification')
        self.verified[key] = record


def validate_assets(assets):
    """Content identity belongs in the task; offline host locations do not."""
    for name, asset in assets.items():
        path = Path(name)
        if not name or path.is_absolute() or '..' in path.parts:
            raise ValueError('Runtime asset escapes its declared root')
        if not re.fullmatch(r'[0-9a-f]{64}', asset.get('sha256', '')):
            raise ValueError('Runtime asset requires a fixed SHA-256')
        if asset.get('source') == 'provided' and 'url' not in asset:
            continue
        url = urlsplit(asset.get('url', ''))
        if asset.get('source') or url.scheme != 'https' or not url.netloc or url.username or url.password:
            raise ValueError('Asset requires a public HTTPS URL or source: provided')


def prepare(lock, cache, asset_files=None):
    assets = lock.get('assets', {})
    validate_assets(assets)
    asset_files = asset_files or {}
    if set(asset_files) - set(assets):
        raise ValueError('Host asset file is not declared by the selected task')
    identity = asset_identity(lock)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    destination = cache / key
    if destination.exists():
        verify(destination, identity)
        return destination
    for name, asset in assets.items():
        if asset.get('source') == 'provided' and name not in asset_files:
            raise ValueError('Provide the declared offline asset with --asset-file ' + name + '=PATH')
        if name in asset_files and file_digest(Path(asset_files[name])) != asset['sha256']:
            raise ValueError('Provided runtime asset differs from its fixed digest')
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
            # Tokenizer templates, processor assets and pinned custom-model
            # Python are part of model semantics too. The recipe/image decides
            # whether remote model code is trusted; scoring stays offline.
            if target.suffix not in {'.json', '.safetensors', '.bin', '.txt', '.model',
                                      '.jinja', '.py', '.tiktoken', '.bpe', '.vocab'} and not target.name.startswith('LICENSE'):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            acquire(base + '/resolve/' + lock['model']['revision'] + '/' + name, target)
        for name, asset in lock.get('assets', {}).items():
            target = directory/'assets'/name
            if Path(name).is_absolute() or not target.resolve().is_relative_to((directory/'assets').resolve()):
                raise ValueError('Runtime asset escapes its declared root')
            target.parent.mkdir(parents=True, exist_ok=True)
            if name in asset_files:
                shutil.copyfile(asset_files[name], target)
            else:
                acquire(asset['url'], target)
            if file_digest(target) != asset['sha256']:
                raise ValueError('Downloaded runtime asset differs from its fixed digest')
        manifest = dict(identity=identity, files={str(p.relative_to(directory)): file_digest(p)
                                                for p in directory.rglob('*') if p.is_file()})
        (directory / 'artifact_manifest.json').write_text(json.dumps(manifest, indent=2))
        directory.rename(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path.home() / '.cache/aka-serving')
    parser.add_argument('--asset-file', action='append', default=[], metavar='NAME=PATH',
                        help='Host-only file for a declared hash-pinned runtime asset')
    args = parser.parse_args()
    asset_files = {}
    for value in args.asset_file:
        name, sep, path = value.partition('=')
        if not sep or not name or not path or name in asset_files:
            parser.error('--asset-file requires a unique NAME=PATH')
        asset_files[name] = Path(path)
    tasks = selected_serving_tasks(args.config, Path.cwd())
    declared = {name for _, _, lock in tasks.values() for name in lock.get('assets', {})}
    if set(asset_files) - declared:
        parser.error('--asset-file name is not declared by any selected task')
    for task_id, (_, _, lock) in tasks.items():
        selected = {name: path for name, path in asset_files.items() if name in lock.get('assets', {})}
        print(task_id, prepare(lock, args.cache, selected))


if __name__ == '__main__':
    main()
