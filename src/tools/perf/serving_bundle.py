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
_COMPUTE_NAME = re.compile(r'[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\Z')


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
        minimum = 0 if declaration.get('allow_empty') is True else 1
        if not isinstance(self.patches, list) or not minimum <= len(self.patches) <= 128:
            raise ValueError(f'Declare between {minimum} and 128 compute replacements')
        seen = set()
        whole_model = declaration.get('scope') == 'whole_model'
        for patch in self.patches:
            if not isinstance(patch, dict) or set(patch) != {'module', 'symbol', 'file', 'callable'}:
                raise ValueError('Invalid compute replacement fields')
            names = _COMPUTE_NAME if whole_model else _NAME
            if any(not isinstance(patch[k], str) or not names.fullmatch(patch[k])
                   for k in ('module', 'symbol', 'callable')):
                raise ValueError('Invalid compute replacement name')
            if not any(patch['module'] == prefix or patch['module'].startswith(prefix + '.')
                       for prefix in declaration['module_prefixes']):
                raise ValueError('Replacement target is outside the declared compute modules')
            if any(patch['module'] == prefix or patch['module'].startswith(prefix + '.')
                   for prefix in declaration.get('protected_modules', [])):
                raise ValueError('Evaluator configuration observation modules are protected')
            if not whole_model and any(part.startswith(('set_', 'enable_', 'disable_', 'load_', 'save_', 'register_')) or part.endswith('_config')
                                       for part in patch['symbol'].split('.')):
                raise ValueError('Configuration mutations are not compute replacements')
            if (declaration.get('scope') != 'whole_model'
                    and patch['module'].startswith('sglang.srt.models.')
                    and not patch['symbol'].split('.')[-1].startswith('forward')):
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
        libraries = {'torch', 'triton', 'aiter', 'sgl_kernel', 'flydsl', 'wave_lang', 'math', 'typing',
                     'functools', 'dataclasses', 'collections', 'itertools', 'enum', 'operator', '__future__'}
        forbidden = {'open', 'eval', 'exec', 'compile', '__import__', 'globals', 'locals',
                     'vars', 'getattr', 'setattr', 'delattr', 'breakpoint'}
        whole_model = self.lock['candidate_bundle'].get('scope') == 'whole_model'
        if whole_model:
            # Model/fusion helpers may own tensor state and dispatch through
            # attributes. Scope/semantics are mandatory independent-review gates;
            # lexical setter bans would also reject real KV-cache kernels.
            forbidden -= {'vars', 'getattr', 'setattr', 'delattr'}
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
                if isinstance(node, ast.Attribute) and (node.attr in runtime or (
                        not whole_model and node.attr.startswith(('__', 'set_', 'enable_', 'disable_')))):
                    raise ValueError('Kernel source accesses protected runtime state')
                if not whole_model and isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)):
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


def record(bundle, function, *, observer=False, model=None, execution='direct_call', configuration_inputs=None):
    import dataclasses
    import torch
    evidence = dict(pid=os.getpid(), rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
                    function=function, bundle_sha256=bundle.digest, observer=observer,
                    backend=bundle.lock.get('framework', 'sglang'), execution=execution)
    if observer:
        if configuration_inputs is not None:
            evidence['configuration_inputs'] = configuration_inputs
        if bundle.lock.get('framework'):
            from _aka_serving_backends import runtime_settings
            settings = runtime_settings(bundle.lock, model)
        else:
            from sglang.srt.server_args import get_global_server_args
            settings = dataclasses.asdict(get_global_server_args())
        evidence.update(server_settings=settings,
            runtime_settings={k: v for k, v in sorted(os.environ.items()) if k.startswith(
                ('SGLANG_', 'VLLM_', 'HIP_', 'ROCR_', 'HSA_', 'RCCL_', 'NCCL_', 'CUDA_',
                 'AITER_', 'TORCH_', 'PYTORCH_', 'FLYDSL_', 'AMD_', 'ATOM_',
                 'FLASH_ATTENTION_', 'GPU_MAX_HW_QUEUES'))})
    with (bundle.task / 'rank_calls.jsonl').open('a') as stream:
        stream.write(json.dumps(evidence) + '\n')


def instrument(bundle, function, label, *, observer=False):
    seen = set()
    torch = sys.modules.get('torch')
    compiler = getattr(torch, 'compiler', None)
    compiled_mark = None
    if not observer and compiler is not None:
        def mark_trace():
            if os.getpid() not in seen:
                record(bundle, label, execution='compiled_trace')
                seen.add(os.getpid())
        # This records tracing separately from GPU execution. The enclosing
        # model call still has to execute and pass the per-rank audit. Metadata
        # I/O must not become part of the compiled compute graph or break it.
        compiled_mark = torch._dynamo.assume_constant_result(mark_trace)
    @functools.wraps(function)
    def call(*args, **kwargs):
        inputs = None
        if observer and bundle.lock.get('framework') in {'vllm', 'atom'} and os.getpid() not in seen:
            from _aka_serving_backends import runtime_inputs
            inputs = runtime_inputs(bundle.lock, args[0])
        result = function(*args, **kwargs)
        if compiled_mark is not None and compiler.is_compiling():
            compiled_mark()
        elif os.getpid() not in seen:
            if observer and bundle.lock.get('framework') in {'vllm', 'atom'}:
                record(bundle, label, observer=True, model=args[0], configuration_inputs=inputs)
            else:
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
    generic_observer = observer.get('kind') == 'model_classes'
    modules = {p['module'] for p in bundle.patches}
    if generic_observer:
        names = observer.get('classes')
        if not isinstance(names, list) or not names or any(not isinstance(n, str) or not n for n in names):
            raise ValueError('Model observer requires explicit architecture class names')
    else:
        modules.add(observer['module'])

    def observes(name):
        return generic_observer and any(name == prefix or name.startswith(prefix + '.')
            for prefix in bundle.lock['candidate_bundle']['module_prefixes'])

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
        if not generic_observer and module.__name__ == observer['module']:
            parts = observer['symbol'].split('.')
            parent = resolve(module, '.'.join(parts[:-1])) if len(parts) > 1 else module
            target = getattr(parent, parts[-1])
            setattr(parent, parts[-1], instrument(bundle, target, 'model_forward', observer=True))
        elif generic_observer:
            entrypoint = observer.get('entrypoint', 'forward')
            if entrypoint not in {'forward', '__call__'}:
                raise ValueError('Unsupported model execution observation boundary')
            for name in observer['classes']:
                cls = vars(module).get(name)
                if inspect.isclass(cls) and cls.__module__ == module.__name__ and callable(getattr(cls, 'forward', None)):
                    # vLLM compiles forward; observe its enclosing call without
                    # tracing file/config inspection into the compute graph.
                    target = getattr(cls, entrypoint)
                    setattr(cls, entrypoint, instrument(bundle, target, 'model_forward', observer=True))

    class Finder(importlib.abc.MetaPathFinder):
        resolving = set()
        def find_spec(self, fullname, path=None, target=None):
            if fullname not in modules and not observes(fullname):
                return None
            if fullname in self.resolving:
                return None
            # Runtime images commonly install engines as editable packages.
            # Their top-level package is resolved by an editable meta finder,
            # not PathFinder. Preserve that import mechanism before wrapping it.
            self.resolving.add(fullname)
            try:
                spec = None
                for finder in tuple(sys.meta_path):
                    if finder is self or not hasattr(finder, 'find_spec'):
                        continue
                    spec = finder.find_spec(fullname, path, target)
                    if spec is not None:
                        break
            finally:
                self.resolving.remove(fullname)
            if spec is None or spec.loader is None:
                # Namespace/optional packages should retain normal import
                # behavior. Missing required target/model calls fail the audit.
                return spec
            loader = spec.loader
            class Loader(importlib.abc.Loader):
                def __getattr__(self, name):
                    # python -m and importlib.resources use optional loader
                    # interfaces in addition to exec_module. Preserve them.
                    return getattr(loader, name)
                def create_module(self, spec):
                    return loader.create_module(spec)
                def exec_module(self, module):
                    loader.exec_module(module)
                    apply(module)
            spec.loader = Loader()
            return spec
    sys.meta_path.insert(0, Finder())
    for name in sorted(n for n in sys.modules if n in modules or observes(n)):
        apply(sys.modules[name])
    return bundle


def effective_function(task: Path, lock: dict, module: str, symbol: str):
    """Resolve an existing operator oracle without requiring that operator to be edited."""
    global _ACTIVE
    bundle = Bundle(task, lock)
    _ACTIVE = bundle
    original_module = importlib.import_module(module)
    function = original_target(original_module, symbol)
    _ORIGINALS[(module, symbol)] = function
    patch = next((p for p in bundle.patches if (p['module'], p['symbol']) == (module, symbol)), None)
    return bundle.function(patch) if patch else function


def original_target(module, symbol):
    parts = symbol.split('.')
    parent = resolve(module, '.'.join(parts[:-1])) if len(parts) > 1 else module
    descriptor = inspect.getattr_static(parent, parts[-1])
    return descriptor.__func__ if isinstance(descriptor, (classmethod, staticmethod)) else getattr(parent, parts[-1])


def original_sources(bundle):
    """Expose pinned image compute source to the independent semantic reviewer.

    An empty baseline manifest otherwise contains no implementation to compare
    against. Native/builtin targets are identified explicitly when Python source
    is unavailable; the reviewer must treat missing evidence as uncertainty.
    """
    result, total, modules = [], 0, {}
    def module_text(name):
        nonlocal total
        module = importlib.import_module(name)
        path = inspect.getsourcefile(module)
        if not path:
            raise TypeError('Module has no inspectable Python source')
        text = Path(path).read_text()
        value = hashlib.sha256(text.encode()).hexdigest()
        if name in modules:
            if modules[name] != value:
                raise ValueError('Original module changed while freezing review evidence')
            return None, value
        total += len(text.encode())
        if total > 4*1024*1024:
            raise ValueError('Original compute source exceeds review budget')
        modules[name] = value
        return text, value
    for patch in bundle.patches:
        module = importlib.import_module(patch['module'])
        target = original_target(module, patch['symbol'])
        _ORIGINALS[(patch['module'], patch['symbol'])] = target
        defining = getattr(target, '__module__', None) or patch['module']
        row = dict(module=patch['module'], symbol=patch['symbol'], defining_module=defining)
        try:
            lines, start = inspect.getsourcelines(target)
            source = ''.join(lines)
            total += len(source.encode())
            if total > 4*1024*1024:
                raise ValueError('Original compute source exceeds review budget')
            row.update(source=source, start_line=start,
                sha256=hashlib.sha256(source.encode()).hexdigest())
        except (OSError, TypeError):
            row['source_unavailable'] = True
        for name, prefix in [(patch['module'], 'module'), (defining, 'defining_module')]:
            if prefix == 'defining_module' and name == patch['module']:
                continue
            try:
                text, value = module_text(name)
                row[prefix+'_sha256'] = value
                if text is not None:
                    row[prefix+'_source'] = text
            except (ImportError, OSError, TypeError, UnicodeError):
                row[prefix+'_source_unavailable'] = True
        result.append(row)
    return result


def audit(task: Path, lock: dict):
    bundle = Bundle(task, lock)
    calls = [json.loads(line) for line in (task / 'rank_calls.jsonl').read_text().splitlines()]
    if not calls or any(row.get('bundle_sha256') != bundle.digest for row in calls):
        raise ValueError('Server loaded a different kernel bundle')
    observers = [row for row in calls if row.get('observer') is True]
    if any(row.get('execution', 'direct_call') != 'direct_call' for row in observers):
        raise ValueError('Compilation alone is not model execution evidence')
    if any(row.get('execution', 'direct_call') not in {'direct_call', 'compiled_trace'} for row in calls):
        raise ValueError('Unknown compute observation type')
    if {row['rank'] for row in observers} != set(range(lock['gpu_count'])):
        raise ValueError('Missing model execution evidence for a tensor-parallel rank')
    if len(observers) != lock['gpu_count']:
        raise ValueError('Duplicate model execution observers for a tensor-parallel rank')
    expected = {p['module'] + ':' + p['symbol'] for p in bundle.patches}
    actual = {row['function'] for row in calls if not row.get('observer')}
    if actual != expected:
        raise ValueError('A declared compute replacement was not executed or traced')
    if 'server_seed' in lock and any(row['server_settings'].get('random_seed') != lock['server_seed'] for row in observers):
        raise ValueError('A server rank changed the locked random seed')
    return calls
