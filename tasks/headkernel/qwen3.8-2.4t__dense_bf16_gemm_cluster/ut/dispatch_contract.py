"""Bind the declared dense source body without reusing an existing AITER op."""
from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
META = json.loads((HERE / "meta.json").read_text())
_native_module = None
_native_callable = None
_candidate_module = None
_candidate_callable = None
_candidate_identity = None
_candidate_calls = 0


def _require(value, message):
    if not value:
        raise RuntimeError(message)


def load_candidate_module(name, path, native):
    """Remove only duplicate custom-op registration, retaining the source body.

    torch_compile_guard may return the first registered op under this name and
    discard a later candidate. This trusted loader compiles the declared source
    with that one decorator removed; it never changes the on-disk candidate.
    """
    global _native_module, _native_callable, _candidate_module
    global _candidate_callable, _candidate_identity
    spec = META["candidate_bind"]
    path = Path(path).resolve()
    declared = (HERE / spec["file"]).resolve()
    _require(name == spec["impl_module"] and spec["impl_attr"] == "gemm_a16w16"
             and spec["target"] == "aiter.tuned_gemm:gemm_a16w16", "Unexpected dense source binding")
    _require(path.is_relative_to(HERE) and declared.is_relative_to(HERE.parent),
             "Dense source must remain inside the task")
    _require(_candidate_callable is None, "Dense candidate was already bound in this worker")
    raw = path.read_bytes()
    _require(raw == declared.read_bytes(), "Overlay source differs from the declared editable source")
    tree = ast.parse(raw, filename=str(path))
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "gemm_a16w16"]
    _require(len(definitions) == 1, "Dense source must define the dispatcher exactly once")
    target = definitions[0]
    decorators = target.decorator_list
    _require(len(decorators) == 1 and isinstance(decorators[0], ast.Call)
             and isinstance(decorators[0].func, ast.Name)
             and decorators[0].func.id == "torch_compile_guard",
             "The dense registration decorator must retain its declared form")
    target.decorator_list = []
    _native_module, _native_callable = native, native.gemm_a16w16
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    exec(compile(tree, str(path), "exec"), module.__dict__)
    function = module.gemm_a16w16
    _require(hasattr(function, "__code__") and Path(function.__code__.co_filename).resolve() == path
             and function.__globals__ is module.__dict__, "Dense dispatcher does not execute its declared source body")
    _candidate_module, _candidate_callable = module, function
    _candidate_identity = {"module": name, "qualname": function.__qualname__, "file": str(path),
                           "source_sha256": hashlib.sha256(raw).hexdigest(),
                           "registration_decorator_removed_by_protected_loader": True}
    return module


def _bound_candidate():
    _require(_candidate_callable is not None, "Dense candidate overlay has not bound a source body")
    native = importlib.import_module("aiter.tuned_gemm")
    _require(native is _native_module and native.gemm_a16w16 is _candidate_callable
             and _candidate_module.gemm_a16w16 is _candidate_callable,
             "Dense source callable binding changed after overlay installation")
    return _candidate_module


def candidate_identity():
    _bound_candidate()
    return dict(_candidate_identity)


def candidate_dispatch_module():
    """Configuration and solMap must come from the body actually invoked."""
    return _bound_candidate()


def native_baseline_callable():
    _bound_candidate()
    return _native_callable


def candidate_call_count():
    return _candidate_calls


def call_candidate(*args, **kwargs):
    global _candidate_calls
    _bound_candidate()
    result = _candidate_callable(*args, **kwargs)
    _candidate_calls += 1
    return result
