"""Reject known source-level timer, comparator and answer-file attacks.

This is an integrity monitor inside the permissive task process, not an OS or
Python security sandbox. Install it before importing any editable candidate.
"""
from __future__ import annotations

import builtins
import hashlib
import io
import json
import math
import marshal
import os
from pathlib import Path
import sys
import types

_GETFRAME = sys._getframe
_HASH = hashlib.sha256
_DUMP = marshal.dumps
_CODE_DIGESTS = {}
ACTIVE_GUARD = None


class IntegrityError(RuntimeError):
    pass


class _FixedIntegrityType(type):
    def __setattr__(cls, name, value):
        if getattr(cls, name, None) is value:
            return
        raise IntegrityError("candidate changed a protected integrity-monitor class")

    def __delattr__(cls, name):
        raise IntegrityError("candidate deleted a protected integrity-monitor method")


class _GuardAccess(metaclass=_FixedIntegrityType):
    """Expose comparison/check operations without exposing mutable guard state."""
    __slots__ = ("__compare", "__check")

    def __init__(self, compare, check):
        object.__setattr__(self, "_GuardAccess__compare", compare)
        object.__setattr__(self, "_GuardAccess__check", check)

    def __setattr__(self, name, value):
        raise IntegrityError("candidate changed the protected integrity-monitor interface")

    def compare(self, *args):
        return self.__compare(*args)

    def check(self):
        return self.__check()


def _fingerprint(value):
    code = getattr(value, "__code__", None)
    if code is None:
        return None
    entry = _CODE_DIGESTS.get(id(code))
    if entry is None:
        entry = (code, _HASH(_DUMP(code)).hexdigest())
        _CODE_DIGESTS[id(code)] = entry
    return (entry[1], repr(value.__defaults__),
            repr(value.__kwdefaults__),
            tuple(id(cell.cell_contents) for cell in (value.__closure__ or ())))


class RuntimeIntegrity(metaclass=_FixedIntegrityType):
    def __setattr__(self, name, value):
        if name != "active" and hasattr(self, name):
            raise IntegrityError(f"candidate changed protected integrity-monitor state: {name}")
        object.__setattr__(self, name, value)

    def __init__(self, task_root, torch, benchmark, harness, overlay=None):
        self.root = Path(task_root).resolve()
        self.overlay = Path(overlay).resolve() if overlay else None
        self.modules = {"torch": torch, "_aka_benchmark": benchmark}
        self.bindings = []
        self.harness_fingerprints = {}
        self.paths = {}
        self._candidate_codes = {}
        self.active = False
        self._import = builtins.__import__
        self._getprofile = sys.getprofile
        self._setprofile = sys.setprofile
        self._profile_hook = self._profile
        self._import_hook = self._checked_import
        self._audit_hook = self._audit
        self._reference_correct = harness.correct

        # Bind the real event method before candidate import. The canonical
        # helper retains its validation logic, but receives a proxy whose
        # elapsed_time cannot be replaced via torch.cuda.Event monkeypatching.
        original_helper = benchmark._event_elapsed_ms
        original_elapsed = torch.cuda.Event.elapsed_time

        class BoundEvent:
            def __init__(self, event):
                self.event = event

            def elapsed_time(self, end):
                return original_elapsed(self.event, end)

        def trusted_event_elapsed(start, end):
            return original_helper(BoundEvent(start), end)

        benchmark._event_elapsed_ms = trusted_event_elapsed

        for owner, names in (
            (torch.cuda, ("Event", "Stream", "CUDAGraph", "graph", "stream", "synchronize",
                          "current_stream", "is_available")),
            (torch.cuda.Event, ("elapsed_time", "record", "synchronize")),
            (torch.cuda.Stream, ("wait_stream", "synchronize")),
            (torch.cuda.CUDAGraph, ("capture_begin", "capture_end", "replay")),
            (torch, ("isfinite", "isnan", "allclose", "isclose", "equal", "all", "any", "load", "save")),
            (torch.Tensor, ("float", "pow", "mean", "sqrt", "clamp_min", "abs", "div", "max",
                            "item", "all", "equal", "__sub__", "__le__", "__add__", "__mul__",
                            "__truediv__", "__bool__")),
            (torch.testing, ("assert_close",)),
            (builtins, ("open",)),
            (io, ("open",)),
            (os, ("open",)),
            (math, ("isfinite",)),
            (json, ("load", "loads", "dump", "dumps")),
            (sys, ("setprofile", "getprofile", "addaudithook", "settrace")),
            (benchmark, ("torch",)),
        ):
            for name in names:
                if hasattr(owner, name):
                    value = getattr(owner, name)
                    self.bindings.append((owner, name, value, _fingerprint(value)))
        for name, value in vars(benchmark).copy().items():
            if isinstance(value, types.FunctionType):
                self.bindings.append((benchmark, name, value, _fingerprint(value)))
        for name in ("correct", "_correct_one", "flatten_outputs", "to_device_like", "_torch"):
            value = getattr(harness, name, None)
            if value is not None:
                self.harness_fingerprints[name] = _fingerprint(value)
        for owner in (RuntimeIntegrity, _GuardAccess, _FixedIntegrityType):
            for name, value in vars(owner).items():
                if isinstance(value, types.FunctionType):
                    self.bindings.append((owner, name, value, _fingerprint(value)))
        monitor_module = sys.modules.get(__name__)
        if monitor_module is not None:
            self.modules[__name__] = monitor_module
            for name in ("RuntimeIntegrity", "IntegrityError", "_GuardAccess", "_fingerprint"):
                value = getattr(monitor_module, name)
                self.bindings.append((monitor_module, name, value, _fingerprint(value)))
        self._protected_functions = frozenset(
            id(value) for _, _, value, _ in self.bindings if hasattr(value, "__code__")) | {
                id(self._reference_correct)}
        object.__setattr__(self, "bindings", tuple(self.bindings))
        object.__setattr__(self, "modules", types.MappingProxyType(self.modules))
        object.__setattr__(self, "harness_fingerprints", types.MappingProxyType(self.harness_fingerprints))
        self._access = _GuardAccess(self.compare, self.check)

    def _candidate_filename(self, filename):
        if filename in self.paths:
            return self.paths[filename]
        try:
            path = Path(filename).resolve()
            candidate = path.is_relative_to(self.root / "source")
            candidate |= path.is_relative_to(self.root / "ut/kernel_src")
            if self.overlay and path.is_relative_to(self.overlay):
                candidate |= path.name != "sitecustomize.py"
        except (OSError, ValueError, RuntimeError):
            candidate = False
        self.paths[filename] = candidate
        return candidate

    def _candidate_on_stack(self):
        frame = _GETFRAME(1)
        while frame:
            if self._candidate_frame(frame):
                return True
            frame = frame.f_back
        return False

    def _candidate_frame(self, frame):
        return (id(frame.f_code) in self._candidate_codes
                or self._candidate_filename(frame.f_code.co_filename))

    def _mark_candidate_code(self, code):
        if not isinstance(code, types.CodeType):
            return
        self._candidate_codes[id(code)] = code
        for value in code.co_consts:
            if isinstance(value, types.CodeType):
                self._mark_candidate_code(value)

    def check(self):
        for name, module in self.modules.items():
            if sys.modules.get(name) is not module:
                raise IntegrityError(f"candidate replaced protected module {name}")
        for owner, name, original, fingerprint in self.bindings:
            value = getattr(owner, name, None)
            if value is not original or _fingerprint(value) != fingerprint:
                raise IntegrityError(f"candidate changed trusted timing/comparison primitive: {name}")
        # Several UTs legitimately reload this same protected file under the
        # same name. Require its exact code/defaults and module globals, rather
        # than accepting replacement comparators or rejecting equivalent loads.
        harness = sys.modules.get("harness_lib")
        for name, fingerprint in self.harness_fingerprints.items():
            value = getattr(harness, name, None)
            if (_fingerprint(value) != fingerprint
                    or getattr(value, "__globals__", None) is not vars(harness)):
                raise IntegrityError(f"candidate changed protected comparator harness_lib.{name}")
        if self.active:
            if ACTIVE_GUARD is not self._access:
                raise IntegrityError("candidate replaced the protected integrity-monitor interface")
            if builtins.__import__ is not self._import_hook:
                raise IntegrityError("candidate disabled protected-module import monitoring")
            if self._getprofile() is not self._profile_hook:
                raise IntegrityError("candidate disabled runtime integrity monitoring")

    def _profile(self, frame, event, arg):
        if event in {"call", "return"} and self._candidate_frame(frame):
            self.check()

    def compare(self, out, ref, tol):
        self.check()
        result = self._reference_correct(out, ref, tol)
        self.check()
        return result

    def _checked_import(self, name, globals=None, locals=None, fromlist=(), level=0):
        protected = ("_aka_benchmark", "harness_lib", "task_runner", "_bench",
                     "runtime_integrity", "_trusted_worker", "_aka_frozen_baseline_bindings")
        if (self.active and name.split(".", 1)[0] in protected
                and self._candidate_on_stack()):
            raise IntegrityError(f"editable candidate imported protected evaluation module {name}")
        return self._import(name, globals, locals, fromlist, level)

    def _protected_path(self, value):
        if not isinstance(value, (str, bytes, os.PathLike)):
            return False
        try:
            path = Path(os.fsdecode(value)).resolve()
        except (OSError, ValueError, RuntimeError):
            return False
        if not path.is_relative_to(self.root):
            return False
        relative = path.relative_to(self.root)
        if relative.parts[0] == "scripts" or relative.name in {"config.yaml", "config.yml"}:
            return True
        if relative.parts[0] == "ut":
            # Editable aliases resolve into source/. The generated overlay is
            # implementation code, except for its trusted setup/manifest.
            if self.overlay and path.is_relative_to(self.overlay):
                return path.name in {"sitecustomize.py", "_overlay_manifest.json"}
            return True
        # Fresh candidate-written result files are no more trustworthy than
        # stale ones. Include atomic-write siblings of every score/report path.
        protected_reports = ("_benchmark_reference", "_bench_raw", "performance_report",
                             "baseline_perf", "optimized_perf", "task_result",
                             "validation_report", "compile_report", "correctness_report")
        return relative.name.startswith("_worker_completion_") or any(
                   relative.name == name or relative.name.startswith(name + ".")
                   for name in protected_reports)

    def _audit(self, event, args):
        if event not in {"open", "os.remove", "os.rmdir", "os.mkdir", "os.chmod",
                         "os.truncate", "os.rename", "os.link", "os.symlink",
                         "exec", "object.__setattr__"}:
            return
        if not self.active or not self._candidate_on_stack():
            return
        if event == "exec":
            # Generated functions keep their candidate provenance even when
            # compile() supplies a filename outside source/ (or <generated>).
            self._mark_candidate_code(args[0])
            return
        if (event == "object.__setattr__" and args[1] in {"__code__", "__defaults__", "__kwdefaults__"}
                and id(args[0]) in self._protected_functions):
            raise IntegrityError("candidate changed protected integrity or measurement function code")
        if event == "open" and self._protected_path(args[0]):
            raise IntegrityError("editable candidate accessed a protected harness or oracle file")
        if event in {"os.remove", "os.rmdir", "os.mkdir", "os.chmod", "os.truncate"}:
            if args and self._protected_path(args[0]):
                raise IntegrityError("editable candidate changed a protected task path")
        if event in {"os.rename", "os.link", "os.symlink"}:
            if any(self._protected_path(path) for path in args[:2]):
                raise IntegrityError("editable candidate changed a protected task path")

    def install(self):
        global ACTIVE_GUARD
        self.check()
        ACTIVE_GUARD = self._access
        self.active = True
        builtins.__import__ = self._import_hook
        sys.addaudithook(self._audit_hook)
        self._setprofile(self._profile_hook)

    def close(self):
        global ACTIVE_GUARD
        try:
            self.check()
        finally:
            self.active = False
            ACTIVE_GUARD = None
            self._setprofile(None)
            builtins.__import__ = self._import
