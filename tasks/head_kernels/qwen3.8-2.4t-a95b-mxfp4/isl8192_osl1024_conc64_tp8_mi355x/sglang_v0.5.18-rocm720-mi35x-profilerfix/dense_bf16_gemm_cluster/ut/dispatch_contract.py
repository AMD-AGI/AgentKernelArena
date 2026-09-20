"""Select the retained five-row AITER dispatch table before runtime imports."""
from __future__ import annotations

import ast
import csv
import importlib.util
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
META = json.loads((HERE / "meta.json").read_text())
ENVIRONMENT_KEY = "AITER_CONFIG_GEMM_BF16"
_prepared = None
_baseline_callable = None
_candidate_source = None
_candidate_calls = 0


def config_matches(config, case):
    return (
        config.get("libtype") == case["expected_backend"]
        and int(config.get("solidx", -1)) == int(case["expected_solidx"])
        and int(config.get("splitK", 0) or 0) == int(case["expected_split_k"])
        and str(config.get("kernelName", "") or "") == case["expected_kernel"]
    )


def prepare_environment():
    """Pure-stdlib setup; no torch/AITER import can precede this in the runner."""
    path = HERE / META["dispatch_config"]
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(HERE):
        raise RuntimeError("dense dispatch table must be a task-local regular file")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != META["dispatch_config_sha256"]:
        raise RuntimeError("frozen dense dispatch table SHA-256 mismatch")
    rows = list(csv.DictReader(raw.decode("utf-8").splitlines()))
    expected = {(int(case["m"]), int(case["n"]), int(case["k"])): case
                for case in META["workload"]["cases"]}
    if len(rows) != len(expected) or len(expected) != 5:
        raise RuntimeError("dense dispatch must contain the exact five retained rows")
    seen = set()
    for row in rows:
        shape = tuple(int(row[key]) for key in ("M", "N", "K"))
        if shape in seen or shape not in expected:
            raise RuntimeError("dense dispatch contains a duplicate or unrecorded shape")
        seen.add(shape)
        case = expected[shape]
        if (row["gfx"] != "gfx950" or int(row["cu_num"]) != 256
                or any(row[key] != "False" for key in ("bias", "scaleAB", "bpreshuffle"))
                or row["dtype"] != "torch.bfloat16" or row["outdtype"] != "torch.bfloat16"
                or not config_matches(row, case)):
            raise RuntimeError(f"frozen dispatch row differs from {case['ledger_id']}")
    os.environ[ENVIRONMENT_KEY] = str(path)
    return path


def _implementation(native):
    module = sys.modules.get(META["candidate_bind"]["impl_module"])
    if module is None:
        return native
    if native.gemm_a16w16 is not module.gemm_a16w16:
        raise RuntimeError("installed dense callable does not bind the declared candidate module")
    return module


def prepare_runtime():
    """Warm native configuration caches and retain candidate backend functions.

    Configuration helpers are protected control code, not editable kernel
    targets. Sharing their warmed native instances avoids an editable source
    frame opening the protected CSV during a numerical call.
    """
    global _prepared
    native = importlib.import_module("aiter.tuned_gemm")
    implementation = _implementation(native)
    if _prepared is not None:
        old_native, old_implementation, table_path, getter, table_getter = _prepared
        if (native is not old_native or implementation is not old_implementation
                or os.environ.get(ENVIRONMENT_KEY) != str(table_path)
                or native.get_GEMM_A16W16_config is not getter
                or native.get_GEMM_A16W16_config_ is not table_getter
                or implementation.get_GEMM_A16W16_config is not getter
                or implementation.get_GEMM_A16W16_config_ is not table_getter):
            raise RuntimeError("prepared dense dispatch binding changed")
        return native
    table_path = prepare_environment()
    core = importlib.import_module("aiter.jit.core")
    # get_config_file() is a process-lifetime lru_cache in the pinned AITER API.
    # Clearing it also supports direct UT/selection runs after an earlier import.
    core.AITER_CONFIGS.get_config_file.cache_clear()
    observed = Path(core.AITER_CONFIGS.AITER_CONFIG_GEMM_BF16_FILE).resolve()
    if observed != table_path.resolve():
        raise RuntimeError(f"AITER did not select the frozen dense table: {observed}")
    getter = native.get_GEMM_A16W16_config
    table_getter = native.get_GEMM_A16W16_config_
    table_getter.cache_clear()
    getter.cache_clear()
    table_getter()
    for case in META["workload"]["cases"]:
        config = getter(int(case["m"]), int(case["n"]), int(case["k"]), False,
                        "torch.bfloat16", "torch.bfloat16", False, False)
        if not config_matches(config, case):
            raise RuntimeError(f"live dispatcher config drift for {case['ledger_id']}: {config}")
    if implementation is not native:
        # Leave solMap and every backend function on the actual candidate.
        implementation.get_GEMM_A16W16_config = getter
        implementation.get_GEMM_A16W16_config_ = table_getter
    _prepared = (native, implementation, table_path, getter, table_getter)
    return native


def dispatch_table(native):
    """Observe the table actually reached by the currently bound callable."""
    if prepare_runtime() is not native:
        raise RuntimeError("dense dispatch module identity changed")
    return _implementation(native).solMap


def load_candidate_module(name, path, native):
    """Execute the candidate body instead of reusing an existing torch.ops op.

    AITER's torch_compile_guard drops a later function when the op name already
    exists. Remove only that registration decorator from the declared dispatcher;
    function bodies, argument defaults, and the on-disk source remain unchanged.
    """
    global _baseline_callable, _candidate_source
    path = Path(path).resolve()
    if name != META["candidate_bind"]["impl_module"] or not path.is_relative_to(HERE.parent):
        raise RuntimeError("candidate binding must use the declared task-local module")
    if _candidate_source is not None:
        raise RuntimeError("dense candidate was already bound in this worker")
    _baseline_callable = native.gemm_a16w16
    tree = ast.parse(path.read_text(), filename=str(path))
    found = 0
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "gemm_a16w16":
            found += 1
            node.decorator_list = [decorator for decorator in node.decorator_list
                                   if not (isinstance(decorator, ast.Call)
                                           and isinstance(decorator.func, ast.Name)
                                           and decorator.func.id == "torch_compile_guard")]
    if found != 1:
        raise RuntimeError("candidate must define the fixed GEMM dispatcher once")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(tree, str(path), "exec"), module.__dict__)
    function = module.gemm_a16w16
    if not hasattr(function, "__code__") or Path(function.__code__.co_filename).resolve() != path:
        raise RuntimeError("candidate dispatcher did not bind its actual source body")
    _candidate_source = function
    return module


def baseline_callable():
    native = prepare_runtime()
    return _baseline_callable if _baseline_callable is not None else native.gemm_a16w16


def baseline_dispatch_table(native):
    if prepare_runtime() is not native:
        raise RuntimeError("native baseline module identity changed")
    return native.solMap


def candidate_call_count():
    return _candidate_calls


def call_candidate(*args, **kwargs):
    global _candidate_calls
    native = prepare_runtime()
    if _candidate_source is None or native.gemm_a16w16 is not _candidate_source:
        raise RuntimeError("candidate source dispatcher is not the bound callable")
    result = _candidate_source(*args, **kwargs)
    _candidate_calls += 1
    return result
