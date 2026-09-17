"""Bind the declared, protected AITER Triton source without unrelated package init."""
import hashlib
import importlib.machinery
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parent


def verify_dependency():
    declaration = json.loads((ROOT / 'runtime-dependencies.json').read_text())
    root = (ROOT / declaration['source_root']).resolve(strict=True)
    if not root.is_relative_to(ROOT) or root == ROOT:
        raise ValueError('AITER dependency must be contained in the task workspace')
    for relative, expected in declaration['files'].items():
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('AITER dependency path escaped its declared source')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'AITER source identity mismatch: {relative}')
    return root


def bind_dependency():
    root = verify_dependency()
    # Only these real Python source namespaces are required by kernel.py.
    # Bypass AITER's unrelated quant/comms/C++ package initializers; every
    # implementation module below is loaded normally from the copied source.
    packages = {
        'aiter': root,
        'aiter.ops': root,
        'aiter.ops.triton': root,
        'aiter.ops.triton.utils': root / 'utils',
        'aiter.ops.triton.utils._triton': root / 'utils/_triton',
        'aiter.ops.triton._triton_kernels': root / '_triton_kernels',
        'aiter.ops.triton._triton_kernels.attention': root / '_triton_kernels/attention',
    }
    for name, directory in packages.items():
        existing = sys.modules.get(name)
        if existing is not None and getattr(existing, '_arena_source_root', None) != str(root):
            raise RuntimeError(f'AITER namespace already bound outside this task: {name}')
    for name, directory in packages.items():
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(directory)]
            package.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
            package.__spec__.submodule_search_locations = package.__path__
            package._arena_source_root = str(root)
            sys.modules[name] = package
            parent, _, child = name.rpartition('.')
            if parent:
                setattr(sys.modules[parent], child, package)
    return root


if __name__ == '__main__':
    if sys.argv[1:] != ['--check']:
        raise SystemExit('Usage: python3 _aiter_dependency.py --check')
    verify_dependency()
    print('Declared AITER Triton source identity verified')
