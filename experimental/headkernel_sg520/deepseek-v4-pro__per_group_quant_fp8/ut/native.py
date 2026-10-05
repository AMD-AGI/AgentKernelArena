"""Production AITER baseline and independently built task-local native candidate."""

import functools
from contextlib import contextmanager
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]


def configure_workspace():
    (ROOT / "build").mkdir(parents=True, exist_ok=True)


@contextmanager
def _candidate_build_scope(core):
    """Keep the production cache intact; isolate only candidate setup/build.

    Used synchronously before any measurements in this task's own process.
    """
    destination = str(ROOT / "build/aiter_jit")
    previous_env = os.environ.get("AITER_JIT_DIR")
    previous_build_dir = core.bd_dir
    previous_path = list(sys.path)
    try:
        os.environ["AITER_JIT_DIR"] = destination
        core.get_user_jit_dir.cache_clear()
        core.get_user_jit_dir()
        core.bd_dir = str(Path(destination) / "build")
        yield
    finally:
        if previous_env is None:
            os.environ.pop("AITER_JIT_DIR", None)
        else:
            os.environ["AITER_JIT_DIR"] = previous_env
        core.bd_dir = previous_build_dir
        core.get_user_jit_dir.cache_clear()
        core.get_user_jit_dir()
        sys.path[:] = previous_path


def baseline():
    configure_workspace()
    import aiter.ops.quant as production
    from aiter.jit import core

    provenance = json.loads((ROOT / "provenance/SOURCE.json").read_text())
    expected = {Path(x["path"]).name: x["sha256"] for x in provenance["native_source_files"]}
    for path, name in [(Path(production.__file__), "quant.py"),
                       (Path(core.AITER_CSRC_DIR) / "kernels/quant_kernels.cu", "quant_kernels.cu")]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected[name]:
            raise RuntimeError(f"production runtime source differs from the observed SG520 source: {name}")
    return production.per_group_quant_hip


def _stage_native_sources():
    frozen = ROOT / "ut/native"
    include = frozen / "include"
    sources = [ROOT / "source/quant_kernels.cu", frozen / "quant_entry_pybind.cu"]
    headers = sorted(p for p in include.rglob("*") if p.is_file())
    digest = hashlib.sha256()
    for path in sources + headers:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    identity = digest.hexdigest()
    stage = ROOT / "build/native_source" / identity
    stage.mkdir(parents=True, exist_ok=True)
    targets = {p.resolve(): stage / "include" / p.relative_to(include) for p in headers}
    targets.update({p.resolve(): stage / p.name for p in sources})
    pattern = re.compile(r'#include\s*[<"]([^>"\n]+)[>"]')
    for source, destination in targets.items():
        def replace(match):
            name = match.group(1)
            for option in (source.parent / name, include / name):
                target = targets.get(option.resolve())
                if target is not None:
                    return '#include "' + str(target) + '"'
            return match.group(0)  # ROCm, Torch, CK and system runtime headers.
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(pattern.sub(replace, source.read_text()))
    return identity, [str(stage / p.name) for p in sources], stage


@functools.lru_cache(maxsize=1)
def candidate():
    configure_workspace()
    from aiter.jit import core

    identity, sources, stage = _stage_native_sources()
    module_name = "module_quant_aka_candidate_" + identity[:20]
    with _candidate_build_scope(core):
        try:
            module = core.get_module(module_name)
        except ModuleNotFoundError:
            options = core.get_args_of_build("module_quant")
            if options.get("third_party"):
                raise RuntimeError("this draft requires installed native dependencies; it never fetches them")
            options.update(md_name=module_name, srcs=sources,
                           extra_include=options["extra_include"] + [str(stage / "include")])
            accepted = inspect.signature(core.build_module).parameters
            core.build_module(**{k: v for k, v in options.items() if k in accepted})
            module = core.get_module(module_name)
    if module_name not in Path(module.__file__).name:
        raise RuntimeError("candidate did not load the task-specific native extension")
    convert, tensor_type, raw_stream, current_device = core._pybind_develop_hooks()

    def invoke_native(name, *args, **kwargs):
        args = tuple(convert(value) if isinstance(value, tensor_type) else value for value in args)
        kwargs = {key: convert(value) if isinstance(value, tensor_type) else value for key, value in kwargs.items()}
        module._set_current_hip_stream(raw_stream(current_device()))
        return getattr(module, name)(*args, **kwargs)

    spec = importlib.util.spec_from_file_location("_frozen_quant_abi_" + identity[:20], ROOT / "ut/abi_wrapper.py")
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    wrapper.dynamic_per_token_scaled_quant = functools.partial(invoke_native, "dynamic_per_token_scaled_quant")
    def reject_unobserved_branch(*args, **kwargs):
        raise ValueError("the E8M0 branch is outside this frozen FP32-scale ABI")
    wrapper.dynamic_per_group_scaled_quant = reject_unobserved_branch
    (ROOT / "build/NATIVE-BUILD.json").write_text(json.dumps({
        "candidate_module": module_name, "extension_path": module.__file__,
        "extension_sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        "source_tree_sha256": identity, "sources": sources,
        "build_recipe": "current AITER module_quant flags, unchanged quant_kernels.cu and only the observed native export",
        "production_namespace_rebound": False,
    }, indent=2) + "\n")
    return wrapper.per_group_quant_hip
