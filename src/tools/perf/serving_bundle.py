"""Task-local, multi-file compute replacements for a pinned serving runtime.

This is an auditable submission contract, not a security sandbox. Python and
native GPU code execute in the evaluator's isolated, credential-free container.
The manifest chooses compute boundaries; it cannot replace the serving harness.
"""
from __future__ import annotations

import ast
import functools
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import types

_ORIGINALS = {}
_ACTIVE = None
_SUFFIXES = {'.py', '.json', '.h', '.hpp', '.cuh', '.cpp', '.cc', '.cu', '.hip'}
_NAME = re.compile(r'[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)*\Z')


def source_files(root: Path, *, max_bytes: int, max_files: int = 256) -> dict[str, Path]:
    """Do not accept links, executables, build products, or unbounded trees."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Kernel bundle must be a regular directory')
    files, size = {}, 0
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('Kernel bundle must not contain symlinks')
        if '__pycache__' in path.relative_to(root).parts:
            continue
        if path.is_dir():
            continue
        relative = path.relative_to(root).as_posix()
        if not path.is_file() or path.suffix not in _SUFFIXES:
            raise ValueError('Unsupported kernel bundle file: ' + relative)
        size += path.stat().st_size
        files[relative] = path
        if size > max_bytes or len(files) > max_files:
            raise ValueError('Kernel bundle exceeds the declared source budget')
    if not files:
        raise ValueError('Kernel bundle is empty')
    return files


class Bundle:
    def __init__(self, task: Path, lock: dict):
        self.task, self.lock = Path(task), lock
        declaration = lock['candidate_bundle']
        for name in ('root', 'manifest'):
            value = declaration[name]
            if not isinstance(value, str) or not value or PurePosixPath(value).is_absolute() or '..' in PurePosixPath(value).parts:
                raise ValueError('Kernel bundle paths must stay inside the task')
        self.root = self.task / declaration['root']
        self.files = source_files(self.root, max_bytes=lock['limits']['max_candidate_source_bytes'])
        self.hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in self.files.items()}
        self.digest = hashlib.sha256(json.dumps(self.hashes, sort_keys=True).encode()).hexdigest()
        manifest = json.loads(self.files[declaration['manifest']].read_text())
        if not isinstance(manifest, dict) or set(manifest) != {'version', 'patches'} or type(manifest['version']) is not int or manifest['version'] != 1:
            raise ValueError('Invalid kernel replacement manifest')
        self.patches = manifest['patches']
        if not isinstance(self.patches, list) or not 1 <= len(self.patches) <= 128:
            raise ValueError('Declare between 1 and 128 compute replacements')
        seen = set()
        for patch in self.patches:
            if not isinstance(patch, dict) or set(patch) != {'module', 'symbol', 'file', 'callable'}:
                raise ValueError('Invalid compute replacement fields')
            if any(not isinstance(patch[k], str) or not _NAME.fullmatch(patch[k])
                   for k in ('module', 'symbol', 'callable')):
                raise ValueError('Invalid compute replacement name')
            if not any(patch['module'] == prefix or patch['module'].startswith(prefix + '.')
                       for prefix in declaration['module_prefixes']):
                raise ValueError('Replacement target is outside the declared compute modules')
            if any(part.startswith(('set_', 'enable_', 'disable_', 'load_', 'save_', 'register_')) or part.endswith('_config')
                   for part in patch['symbol'].split('.')):
                raise ValueError('Configuration mutations are not compute replacements')
            if patch['module'].startswith('sglang.srt.models.') and not patch['symbol'].split('.')[-1].startswith('forward'):
                raise ValueError('Model replacements must use a forward compute boundary')
            path = PurePosixPath(patch['file'])
            if path.is_absolute() or '..' in path.parts or path.as_posix() not in self.files or path.suffix != '.py':
                raise ValueError('Replacement entrypoint must be a Python file inside the bundle')
            key = (patch['module'], patch['symbol'])
            if key in seen:
                raise ValueError('Duplicate compute replacement')
            seen.add(key)
        location = hashlib.sha256(str(self.root.resolve()).encode()).hexdigest()[:16]
        self.package = '_aka_candidate_' + self.digest + '_' + location

    def check_sources(self):
        # Deliberately allow numerical libraries and kernel dispatch glue. Do not
        # turn this into a mandatory profiler or a Triton-only task.
        libraries = {'torch', 'triton', 'aiter', 'sgl_kernel', 'math', 'typing',
                     'functools', 'dataclasses', 'collections', 'itertools', 'enum', 'operator', '__future__'}
        forbidden = {'open', 'eval', 'exec', 'compile', '__import__', 'globals', 'locals',
                     'vars', 'getattr', 'setattr', 'delattr', 'breakpoint'}
        runtime = {'os', 'sys', 'subprocess', 'environ', 'socket', 'pathlib', 'shutil',
                   'builtins', 'ctypes', 'importlib', 'multiprocessing', 'set_default_dtype',
                   'set_default_device', 'manual_seed', 'set_num_threads', 'set_grad_enabled'}
        for name, path in self.files.items():
            if path.suffix != '.py':
                continue
            tree = ast.parse(path.read_text(), filename=name)
            compile(tree, name, 'exec')
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or '']
                    if isinstance(node, ast.ImportFrom) and node.level:
                        continue  # The private package search path is only this bundle.
                    for module in imports:
                        allowed = module.split('.')[0] in libraries or any(
                            module == p or module.startswith(p + '.')
                            for p in self.lock['candidate_bundle']['module_prefixes'])
                        if module == '_aka_serving_bundle' and isinstance(node, ast.ImportFrom):
                            allowed = all(a.name in {'original', 'load_extension'} for a in node.names)
                        if not allowed:
                            raise ValueError('Undeclared implementation dependency: ' + module)
                if isinstance(node, ast.Name) and (node.id in forbidden or node.id.startswith('__')):
                    raise ValueError('Kernel source accesses protected runtime state')
                if isinstance(node, ast.Attribute) and (node.attr in runtime or node.attr.startswith(
                        ('__', 'set_', 'enable_', 'disable_'))):
                    raise ValueError('Kernel source accesses protected runtime state')
                if isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)):
                    raise ValueError('Use the replacement manifest instead of mutating runtime attributes')
        return dict(bundle_sha256=self.digest, source_hashes=self.hashes, replacements=self.patches)

    def function(self, patch):
        if self.package not in sys.modules:
            package = types.ModuleType(self.package)
            package.__path__ = [str(self.root)]
            sys.modules[self.package] = package
            prefix = self.package + '.'
            class SourceLoader(importlib.machinery.SourceFileLoader):
                def get_code(self, fullname):
                    # Never execute a timestamp-valid but source-stale pyc.
                    return self.source_to_code(self.get_data(self.path), self.path)
            class SourceFinder(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if not fullname.startswith(prefix):
                        return None
                    spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
                    if spec is not None and isinstance(spec.loader, importlib.machinery.SourceFileLoader):
                        spec.loader = SourceLoader(fullname, spec.origin)
                    return spec
            sys.meta_path.insert(0, SourceFinder())
        path = PurePosixPath(patch['file'])
        module_name = self.package + '.' + '.'.join(path.with_suffix('').parts)
        module = importlib.import_module(module_name)
        function = resolve(module, patch['callable'])
        if not callable(function) or inspect.isclass(function):
            raise ValueError('Replacement must be a function, not a class or configuration object')
        return function


def resolve(module, symbol):
    value = module
    for part in symbol.split('.'):
        value = getattr(value, part)
    return value


def original(module: str, symbol: str):
    """Call an unchanged compute function from a replacement without recursion."""
    key = (module, symbol)
    if key not in _ORIGINALS:
        raise ValueError('original() requires a target declared in the replacement manifest')
    return _ORIGINALS[key]


def load_extension(name: str, sources: list[str]):
    """Build submitted HIP/C++ files using the pinned toolchain and local /tmp."""
    if _ACTIVE is None or not re.fullmatch('[A-Za-z][A-Za-z0-9_]*', name) or not sources:
        raise ValueError('Invalid extension declaration')
    paths = []
    for source in sources:
        if source not in _ACTIVE.files or Path(source).suffix not in {'.cpp', '.cc', '.cu', '.hip'}:
            raise ValueError('Extension source must belong to the candidate bundle')
        paths.append(str(_ACTIVE.files[source]))
    from torch.utils.cpp_extension import load
    return load(name=name + '_' + _ACTIVE.digest[:16], sources=paths,
                extra_cflags=['-O3'], extra_cuda_cflags=['-O3'], verbose=False)


def record(bundle, function, *, observer=False):
    import dataclasses
    import torch
    from sglang.srt.server_args import get_global_server_args
    evidence = dict(pid=os.getpid(), rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
                    function=function, bundle_sha256=bundle.digest, observer=observer)
    if observer:
        evidence.update(server_settings=dataclasses.asdict(get_global_server_args()),
            runtime_settings={k: v for k, v in sorted(os.environ.items()) if k.startswith(
                ('SGLANG_', 'VLLM_', 'HIP_', 'ROCR_', 'HSA_', 'RCCL_', 'NCCL_', 'CUDA_',
                 'AITER_', 'TORCH_', 'PYTORCH_', 'FLYDSL_', 'AMD_'))})
    with (bundle.task / 'rank_calls.jsonl').open('a') as stream:
        stream.write(json.dumps(evidence) + '\n')


def instrument(bundle, function, label, *, observer=False):
    seen = set()
    @functools.wraps(function)
    def call(*args, **kwargs):
        result = function(*args, **kwargs)
        if os.getpid() not in seen:
            record(bundle, label, observer=observer)
            seen.add(os.getpid())
        return result
    return call


def install(task: Path, lock: dict):
    global _ACTIVE
    bundle = Bundle(task, lock)
    bundle.check_sources()
    _ACTIVE = bundle
    observer = lock['candidate_bundle']['observer']
    modules = {p['module'] for p in bundle.patches} | {observer['module']}

    def apply(module):
        patches = [p for p in bundle.patches if p['module'] == module.__name__]
        # Save all originals before importing any candidate helpers.
        for patch in patches:
            parts = patch['symbol'].split('.')
            parent = resolve(module, '.'.join(parts[:-1])) if len(parts) > 1 else module
            descriptor = inspect.getattr_static(parent, parts[-1])
            target = descriptor.__func__ if isinstance(descriptor, (classmethod, staticmethod)) else resolve(module, patch['symbol'])
            if not callable(target) or inspect.isclass(target):
                raise ValueError('Target must be a compute function or method')
            _ORIGINALS[(patch['module'], patch['symbol'])] = target
        for patch in patches:
            parts = patch['symbol'].split('.')
            parent = resolve(module, '.'.join(parts[:-1])) if len(parts) > 1 else module
            descriptor = inspect.getattr_static(parent, parts[-1])
            replacement = instrument(bundle, bundle.function(patch), patch['module'] + ':' + patch['symbol'])
            if isinstance(descriptor, staticmethod):
                replacement = staticmethod(replacement)
            elif isinstance(descriptor, classmethod):
                replacement = classmethod(replacement)
            setattr(parent, parts[-1], replacement)
        if module.__name__ == observer['module']:
            parts = observer['symbol'].split('.')
            parent = resolve(module, '.'.join(parts[:-1])) if len(parts) > 1 else module
            target = getattr(parent, parts[-1])
            setattr(parent, parts[-1], instrument(bundle, target, 'model_forward', observer=True))

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname not in modules:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec is None or spec.loader is None:
                raise ImportError('Declared compute module unavailable: ' + fullname)
            loader = spec.loader
            class Loader(importlib.abc.Loader):
                def create_module(self, spec):
                    return loader.create_module(spec)
                def exec_module(self, module):
                    loader.exec_module(module)
                    apply(module)
            spec.loader = Loader()
            return spec
    sys.meta_path.insert(0, Finder())
    for name in sorted(modules & sys.modules.keys()):
        apply(sys.modules[name])
    return bundle


def effective_function(task: Path, lock: dict, module: str, symbol: str):
    """Resolve an existing operator oracle without requiring that operator to be edited."""
    global _ACTIVE
    bundle = Bundle(task, lock)
    _ACTIVE = bundle
    original_module = importlib.import_module(module)
    function = resolve(original_module, symbol)
    _ORIGINALS[(module, symbol)] = function
    patch = next((p for p in bundle.patches if (p['module'], p['symbol']) == (module, symbol)), None)
    return bundle.function(patch) if patch else function


def audit(task: Path, lock: dict):
    bundle = Bundle(task, lock)
    calls = [json.loads(line) for line in (task / 'rank_calls.jsonl').read_text().splitlines()]
    if not calls or any(row.get('bundle_sha256') != bundle.digest for row in calls):
        raise ValueError('Server loaded a different kernel bundle')
    observers = [row for row in calls if row.get('observer') is True]
    if {row['rank'] for row in observers} != set(range(lock['gpu_count'])):
        raise ValueError('Missing model execution evidence for a tensor-parallel rank')
    if len(observers) != lock['gpu_count']:
        raise ValueError('Duplicate model execution observers for a tensor-parallel rank')
    expected = {p['module'] + ':' + p['symbol'] for p in bundle.patches}
    actual = {row['function'] for row in calls if not row.get('observer')}
    if actual != expected:
        raise ValueError('A declared compute replacement was not executed')
    if any(row['server_settings'].get('random_seed') != lock['server_seed'] for row in observers):
        raise ValueError('A server rank changed the locked random seed')
    return calls
