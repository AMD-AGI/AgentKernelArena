###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################
"""Detect candidate code that rebinds evaluator, timing or comparison objects.

Python-family candidates run in the evaluator's process, so the static audit
cannot stop every reflective route to the objects that time and judge them
(for example ``torch.sys.modules[...]`` and ``torch.builtins.setattr``). This
module records the identity of those objects before candidate code is imported
and verifies them after the import, after every candidate call and after every
timing sample. A changed binding fails the action as harness tampering.

What is covered:

- every module of the evaluator package (including the generated
  ``_aka_benchmark`` timing helper), its global bindings and the attributes of
  the classes it defines, and its ``sys.modules`` entry;
- the callables, classes and submodules of ``torch``, ``torch.cuda``,
  ``builtins`` and the standard-library modules the evaluator uses, and the
  attributes of the PyTorch tensor, event, stream and graph classes;
- ``triton.runtime.jit.JITFunction`` when Triton is loaded (the launch observer
  relies on it);
- the interpreter's trace and profile hooks, ``gc.callbacks`` and the standard
  output streams.

New attributes are allowed: Python and PyTorch add lazily imported submodules.
A process cannot protect itself completely from code it runs, so this is a
detector for rebinding, not a sandbox.
"""
from __future__ import annotations

import builtins
import gc
import importlib
import json
import math
import statistics
import sys
import time
import types
from typing import Any, Callable

# Evaluator modules imported before the snapshot, so lazily imported ones are covered.
EVALUATOR_MODULES = ("audit", "compare", "contract", "controls", "evaluate", "loader", "values",
                     "integrity", "_aka_benchmark")
# Module globals the evaluator itself updates: name -> (module, class) the value
# must be an exact instance of, or None.
MUTABLE_GLOBALS = {"evaluate": {"_FLUSH": ("_aka_benchmark", "ReadOnlyCacheFlush")}}
STDLIB_MODULES = (builtins, gc, importlib, json, math, statistics, time, types)
SYS_FUNCTIONS = ("settrace", "setprofile", "gettrace", "getprofile", "exit", "modules")
_TRACKED_KINDS = (types.FunctionType, types.BuiltinFunctionType, types.MethodType, types.ModuleType,
                  type, staticmethod, classmethod, property)


class TamperError(RuntimeError):
    """A protected evaluator or runtime binding changed while candidate code could run."""


def _is_tracked(value: Any) -> bool:
    return isinstance(value, _TRACKED_KINDS) or callable(value)


def _bindings(namespace: dict, *, everything: bool) -> dict[str, int]:
    return {name: id(value) for name, value in list(namespace.items())
            if everything or _is_tracked(value)}


class HarnessIntegrity:
    """A snapshot of protected bindings; ``check(stage)`` raises on any change."""

    def __init__(self, package: str):
        self.package = package
        self._namespaces: list[tuple[str, dict, dict[str, int]]] = []
        self._held: list[Any] = []          # keep snapshotted objects alive so ids stay unique
        self._mutable: dict[tuple[int, str], type | None] = {}
        self._modules: dict[str, int] = {}
        modules = {}
        for name in EVALUATOR_MODULES:
            try:
                modules[name] = importlib.import_module(f"{package}.{name}")
            except ImportError:
                continue    # _aka_benchmark exists only in a materialized workspace
        for short, module in modules.items():
            self._modules[module.__name__] = id(module)
            allowed = MUTABLE_GLOBALS.get(short, {})
            namespace = vars(module)
            bindings = _bindings(namespace, everything=True)
            for name, (owner, cls_name) in allowed.items():
                bindings.pop(name, None)
                owner_module = modules.get(owner)
                self._mutable[(id(namespace), name)] = getattr(owner_module, cls_name, None)
            self._track(module.__name__, namespace, bindings)
            for value in list(namespace.values()):
                if isinstance(value, type) and value.__module__ == module.__name__:
                    self._track(f"{module.__name__}.{value.__name__}", vars(value),
                                _bindings(vars(value), everything=True))
        package_module = sys.modules.get(package)
        if package_module is not None:
            self._modules[package] = id(package_module)
            self._track(package, vars(package_module), _bindings(vars(package_module), everything=False))
        for module in STDLIB_MODULES:
            self._track(module.__name__, vars(module), _bindings(vars(module), everything=False))
        self._track("sys", vars(sys), {name: id(getattr(sys, name)) for name in SYS_FUNCTIONS})
        self._track_runtime()
        self._hooks = _hook_state()

    def _track(self, label: str, namespace: dict, bindings: dict[str, int]) -> None:
        self._held.append([namespace.get(name) for name in bindings])
        self._namespaces.append((label, namespace, bindings))

    def _track_runtime(self) -> None:
        import torch

        for label, module in (("torch", torch), ("torch.cuda", torch.cuda)):
            self._track(label, vars(module), _bindings(vars(module), everything=False))
        for cls in (torch.Tensor, torch.cuda.Event, torch.cuda.Stream, torch.cuda.CUDAGraph):
            self._track(f"{cls.__module__}.{cls.__qualname__}", vars(cls),
                        _bindings(vars(cls), everything=True))
        jit = sys.modules.get("triton.runtime.jit")
        if jit is not None and hasattr(jit, "JITFunction"):
            cls = jit.JITFunction
            self._track("triton.runtime.jit.JITFunction", vars(cls), _bindings(vars(cls), everything=True))

    # Defaults bind the helpers now, so later rebinding of module or class names
    # cannot change what an existing check runs.
    def changes(self, _hook_state=None, _id=id, _dict=dict, _type=type, _modules=sys.modules) -> list[str]:
        _hook_state = _hook_state or _HOOK_STATE
        found = []
        for name, expected in self._modules.items():
            if _id(_modules.get(name)) != expected:
                found.append(f"sys.modules[{name!r}] was replaced")
        for label, namespace, bindings in self._namespaces:
            current = _dict(namespace)       # a stable copy; the mapping may be mutated concurrently
            for name, expected in bindings.items():
                if name not in current:
                    found.append(f"{label}.{name} was deleted")
                elif _id(current[name]) != expected:
                    found.append(f"{label}.{name} was rebound")
            for (namespace_id, name), cls in self._mutable.items():
                if namespace_id == id(namespace) and name in current:
                    value = current[name]
                    if value is not None and (cls is None or _type(value) is not cls):
                        found.append(f"{label}.{name} holds a {_type(value).__name__}")
        hooks = _hook_state()
        for key, expected in self._hooks.items():
            if hooks[key] is not expected and hooks[key] != expected:
                found.append(f"interpreter {key} hook changed")
        return found

def _hook_state() -> dict[str, Any]:
    return {"trace": sys.gettrace(), "profile": sys.getprofile(),
            "gc.callbacks": tuple(id(c) for c in gc.callbacks),
            "stdout": id(sys.stdout), "stderr": id(sys.stderr)}


_HOOK_STATE = _hook_state


def snapshot(package: str) -> Callable[[str], None]:
    """Snapshot protected bindings now and return a check function for them.

    The returned closure holds the snapshot's bound ``changes`` method, so
    rebinding names in this module or class later cannot disable it. Callers
    keep the closure instead of looking anything up again.
    """
    changes = HarnessIntegrity(package).changes
    error, hook_state = TamperError, _hook_state

    def check(stage: str) -> None:
        found = changes(_hook_state=hook_state)
        if found:
            raise error(f"Harness tampering detected after {stage}: {sorted(found)[:6]}")

    return check
