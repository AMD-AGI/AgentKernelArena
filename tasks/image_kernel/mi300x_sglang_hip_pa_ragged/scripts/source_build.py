"""Attest task-local HIP compilation without changing operator/timing calls.

This records compilation inputs, not complete GPU instruction provenance. A
runtime dispatch that never builds any declared target is rejected, including a
CK task silently dispatching an unrelated precompiled or FlyDSL implementation.
"""
from __future__ import annotations

import importlib
import inspect
from pathlib import Path


class BuildEvidence:
    def __init__(self, root, targets):
        self.root = Path(root).resolve()
        self.targets = {str(p): (self.root / p).resolve() for p in targets}
        self.compiled = set()
        self.modules = set()

    def record(self, module, inputs):
        for value in inputs:
            source = Path(value).resolve()
            if not source.is_relative_to(self.root):
                continue
            for relative, target in self.targets.items():
                if source == target or (source.is_dir() and target.is_relative_to(source)):
                    self.compiled.add(relative)
                    self.modules.add(str(module))

    def wrap(self, module, name, source_fields, name_field):
        original = getattr(module, name)
        signature = inspect.signature(original)

        def compile_and_record(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            # Copy before calling: some compiler helpers mutate these lists.
            paths = [p for field in source_fields for p in (bound.arguments.get(field) or [])]
            result = original(*args, **kwargs)
            self.record(bound.arguments[name_field], paths)
            return result

        setattr(module, name, compile_and_record)

    def finish(self):
        if not self.compiled:
            raise RuntimeError("Runtime did not compile any declared HIP candidate source; "
                               "installed binaries or a different dispatch are not a candidate")
        return {"compiled_candidate_paths": sorted(self.compiled),
                "build_modules": sorted(self.modules),
                "coverage": "build inputs; not exhaustive kernel execution attestation"}


def watch(root, targets, meta_root, *, template=False):
    evidence = BuildEvidence(root, targets)
    if template:
        utils = importlib.import_module("csrc.cpp_itfs.utils")
        if Path(utils.AITER_CORE_DIR).resolve() != (Path(root) / meta_root).resolve():
            raise RuntimeError("Template compiler source root is outside the declared workspace")
        evidence.wrap(utils, "compile_lib", ("includes", "sources"), "folder")
    else:
        core = importlib.import_module("aiter.jit.core")
        expected = (Path(root) / meta_root / "csrc").resolve()
        if Path(core.AITER_CSRC_DIR).resolve() != expected:
            raise RuntimeError("AITER compiler ignored the declared C++ source tree")
        evidence.wrap(core, "build_module", ("srcs",), "md_name")

        def declared_third_party(name):
            # Setup provides dependencies. The compiler must not git-clone,
            # reset a source checkout, or alter global git settings during eval.
            locations = {"ComposableKernel": Path(root) / meta_root / "3rdparty/composable_kernel",
                         "HipKittens": Path(root) / meta_root / "3rdparty/HipKittens"}
            path = locations.get(name)
            if path is None or not path.is_dir() or not any(path.iterdir()):
                raise RuntimeError(f"Missing declared compiler dependency: {name}")
            if name == "HipKittens" and Path(getattr(core, "HIP_KITTENS_DIR", "")).resolve() != path.resolve():
                raise RuntimeError("HipKittens compiler dependency is not task-local")

        core.clone_3rdparty = declared_third_party
    return evidence
