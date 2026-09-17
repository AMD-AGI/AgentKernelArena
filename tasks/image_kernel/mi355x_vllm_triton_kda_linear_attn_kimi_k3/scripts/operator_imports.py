"""Load the pinned standalone operator without initializing an inference model.

These package namespaces select real workspace files. The KDA and FLA leaf
modules, including their Triton functions, are imported normally and unchanged.
The skipped initializers only export unrelated full-model/server interfaces.
"""
import importlib.machinery
from pathlib import Path
import sys
import types


PACKAGES = (
    "vllm.models", "vllm.models.kimi_k3", "vllm.models.kimi_k3.amd",
    "vllm.models.kimi_k3.amd.ops", "vllm.models.kimi_k3.amd.ops.third_party",
    "vllm.third_party.flash_linear_attention.ops",
)


def configure_operator_packages(root):
    root = Path(root).resolve(strict=True)
    for name in PACKAGES:
        path = root.joinpath(*name.split(".")).resolve(strict=True)
        if not path.is_dir() or not path.is_relative_to(root):
            raise ValueError(f"Operator package escapes its workspace: {name}")
        existing = sys.modules.get(name)
        if existing is not None:
            locations = [Path(p).resolve(strict=True) for p in getattr(existing, "__path__", ())]
            if locations != [path]:
                raise RuntimeError(f"Operator package was already imported from another source: {name}")
            continue
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        module.__file__ = str(path / "__init__.py")
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        sys.modules[name] = module
