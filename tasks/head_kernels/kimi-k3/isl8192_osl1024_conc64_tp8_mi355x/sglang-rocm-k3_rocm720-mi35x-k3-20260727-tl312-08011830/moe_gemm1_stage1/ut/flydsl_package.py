"""Load only the captured MoE module under independent package namespaces.

The original package __init__ reexports unrelated attention/GEMM kernels. A
standalone MoE task needs the MoE dependency closure, not those runtime features.
All source module bodies remain the captured originals.
"""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import types


def load_moe_package(root, name, *, frozen_manifest=None):
    root = Path(root).resolve()
    if frozen_manifest is not None:
        with Path(frozen_manifest).open() as stream:
            manifest = json.load(stream)
        expected = manifest["baseline_files"]
        if "moe_kernels.py" not in expected:
            raise RuntimeError("frozen FlyDSL manifest does not identify the MoE entry module")
        for relative, checksum in expected.items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root):
                raise RuntimeError("frozen FlyDSL dependency escapes its package")
            if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
                raise RuntimeError(f"frozen FlyDSL dependency changed: {relative}")
    # Keep the captured package's compiler availability/version check without
    # importing its unrelated public API reexports.
    from packaging.version import Version
    flydsl = importlib.import_module("flydsl")
    version = getattr(flydsl, "__version__", None)
    if version is None or Version(version.split("+")[0]) < Version("0.2.4"):
        raise ImportError(f"captured MoE requires FlyDSL >=0.2.4, found {version!r}")
    if name in sys.modules:
        raise RuntimeError(f"FlyDSL namespace is already bound: {name}")
    package = types.ModuleType(name)
    package.__file__ = str(root / "__init__.py")
    package.__package__ = name
    package.__path__ = [str(root)]
    package.__spec__ = importlib.util.spec_from_loader(name, loader=None, is_package=True)
    sys.modules[name] = package
    try:
        return importlib.import_module(name + ".moe_kernels")
    except BaseException:
        for key in list(sys.modules):
            if key == name or key.startswith(name + "."):
                sys.modules.pop(key, None)
        raise


REQUIRED_BINDING_FILES = ('flydsl_package.py', 'dependency_manifest.json', 'baseline_src/flydsl/moe_kernels.py', 'kernel_src/flydsl/moe_kernels.py')

def validate_layout(task_ut_dir):
    """Validate the direct binding inputs without importing GPU packages."""
    from pathlib import Path
    task_ut = Path(task_ut_dir).resolve()
    task_root = task_ut.parent
    for relative in REQUIRED_BINDING_FILES:
        path = (task_ut / relative).resolve()
        if not path.is_relative_to(task_root) or not path.is_file():
            raise RuntimeError(f"independent Kimi binding input is missing or external: ut/{relative}")
    return True
