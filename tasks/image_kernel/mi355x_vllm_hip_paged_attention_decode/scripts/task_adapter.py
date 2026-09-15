"""Task-owned binding of the original harness to explicit role/action execution."""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MODULE_BINDINGS = [('aiter', 'aiter')]
LOCAL_LOADER = None
LOCAL_MODULE = None
FROZEN_REFERENCES = []
REQUIRED_ASSETS = []
META_ROOT = 'aiter_meta'
FORCE_REBUILD = True


def setup():
    # Copy oracle code before baseline capture, never refresh it from a candidate
    # on resume. The framework protects these files together with the harness.
    for source, target in FROZEN_REFERENCES:
        src, dst = ROOT / source, ROOT / target
        if not src.is_file():
            raise FileNotFoundError(f"Missing provided reference source: {source}")
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
    for required in REQUIRED_ASSETS:
        if not (ROOT / required).exists():
            raise FileNotFoundError(f"Required task image asset is absent: {required}")


def validate_workloads(harness):
    data = json.loads((ROOT / "workloads.json").read_text())
    actual = {"correctness": harness.CASES,
              "performance": getattr(harness, "PERF_CASES", None)}
    if json.loads(json.dumps(actual)) != data["original_cases"]:
        raise ValueError("Protected manifest and harness case definitions disagree")


def prepare(harness):
    for required in REQUIRED_ASSETS:
        if not (ROOT / required).exists():
            raise FileNotFoundError(f"Required task image asset is absent: {required}")
    # Cache paths belong to this role's workspace. An inherited cache/build root
    # must not cause a candidate to load a binary from the baseline or image.
    for key, subdir in {
        "AITER_JIT_DIR": "jit", "AITER_ROOT_DIR": "aiter_root",
        "TRITON_CACHE_DIR": "triton", "TORCH_EXTENSIONS_DIR": "torch_extensions",
        "FLYDSL_RUNTIME_CACHE_DIR": "flydsl", "FLYDSL_DUMP_DIR": "flydsl_dump",
        "FLYDSL_AUTOTUNE_CACHE_DIR": "flydsl_autotune",
    }.items():
        os.environ[key] = str(ROOT / "build" / subdir)
    if META_ROOT:
        os.environ["AITER_META_DIR"] = str(ROOT / META_ROOT)
        if not (ROOT / META_ROOT / "csrc").is_dir():
            raise FileNotFoundError("Declared AITER C++ source tree is unavailable")
    if FORCE_REBUILD:
        # AITER's ctypes path can reuse a .so despite AITER_REBUILD. Start an
        # empty per-action build root; compilation stays outside device timing.
        cache_root = ROOT / "build"
        cache_root.mkdir(exist_ok=True)
        action_cache = Path(tempfile.mkdtemp(prefix="hip-action-", dir=cache_root))
        os.environ["AITER_JIT_DIR"] = str(action_cache / "jit")
        os.environ["AITER_ROOT_DIR"] = str(action_cache / "template")
        os.environ["AITER_REBUILD"] = "1"
    # New tasks use the container's selected Python. Never re-exec into an
    # arbitrary other installation, or silently use its installed candidate.
    configure = getattr(harness, "_configure_runtime", None) or harness._configure
    configure()
    for module_name, relative in MODULE_BINDINGS:
        module = importlib.import_module(module_name)
        path = Path(module.__file__).resolve()
        expected = (ROOT / relative).resolve(strict=True)
        if path != expected and not (expected.is_dir() and path.is_relative_to(expected)):
            raise RuntimeError(f"{module_name} resolved outside the declared implementation: {path}")
    if LOCAL_LOADER:
        module = getattr(harness, LOCAL_LOADER)()
        path = Path(module.__file__).resolve()
        if path != (ROOT / LOCAL_MODULE).resolve(strict=True):
            raise RuntimeError("Kernel loader did not load the declared candidate file")
    for _, target in FROZEN_REFERENCES:
        if not (ROOT / target).is_file():
            raise FileNotFoundError("Task setup has not captured the protected numerical reference")

    import yaml
    from source_build import watch
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    targets = [item if isinstance(item, str) else item["path"] for item in cfg["candidate"]["editable"]]
    # paged_attention_rocm builds csrc.cpp_itfs templates, not jit.core modules.
    return watch(ROOT, targets, META_ROOT, template=True)


def run_correctness(harness):
    harness.run_correctness()
