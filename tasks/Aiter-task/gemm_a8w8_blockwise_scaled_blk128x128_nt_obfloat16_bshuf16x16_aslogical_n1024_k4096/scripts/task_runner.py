#!/usr/bin/env python3
"""Protected arena-eval-v1 runner for functional SIKL tasks with a FlyDSL builder candidate."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml

from scripts.task_api import (assert_unmodified, compare_outputs, dimensions, load_solution, snapshot,
                              validate_inputs)
from scripts.task_inputs import call_varying_draws, make_inputs, persistent_inputs
from scripts.task_timing import time_row

ALLOWED_IMPORTS = {"__future__", "typing", "collections", "dataclasses", "enum", "functools",
                   "itertools", "math", "operator", "numbers", "abc", "types", "torch", "flydsl"}
# Library matrix products and the normalizations these operators fuse.
LIBRARY_COMPUTE = {"matmul", "mm", "bmm", "einsum", "linear", "addmm", "addbmm", "baddbmm",
                   "tensordot", "_scaled_mm", "scaled_mm"}
# FlyDSL exposes math intrinsics such as rsqrt and exp under the same names, so
# these are rejected only when reached through a torch binding.
TORCH_COMPUTE = {"softmax", "log_softmax", "sigmoid", "rsqrt", "logsumexp", "rms_norm",
                 "layer_norm", "normalize", "silu"}


def _torch_aliases(tree):
    aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "torch":
                    aliases.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "torch":
            aliases.update(alias.asname or alias.name for alias in node.names)
    return aliases


def _attribute_root(node):
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def assert_source_independent(source):
    """Reject protected or library-operator imports, including members and aliases.

    This is a static guard, not proof against reflection or arbitrary hostile
    Python. The numerical checks of every timed invocation also apply.
    """
    tree = ast.parse(source)
    torch_names = _torch_aliases(tree)
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise RuntimeError("candidate imports a protected task/local module")
            prefix = node.module or ""
            names = [prefix, *(f"{prefix}.{a.name}" for a in node.names)]
        for name in names:
            parts = name.split(".")
            if parts[0] == "scripts" or parts[0] not in ALLOWED_IMPORTS:
                raise RuntimeError(f"candidate imports a protected task or unsupported module: {name}")
            if parts[0] == "torch" and any(p in LIBRARY_COMPUTE | TORCH_COMPUTE for p in parts):
                raise RuntimeError(f"candidate imports library operator computation: {name}")
        if isinstance(node, (ast.BinOp, ast.AugAssign)) and isinstance(node.op, ast.MatMult):
            raise RuntimeError("candidate uses the library matrix multiplication operator")
        if isinstance(node, ast.Attribute):
            if node.attr in LIBRARY_COMPUTE:
                raise RuntimeError(f"candidate references library operator computation: {node.attr}")
            if node.attr in TORCH_COMPUTE and _attribute_root(node) in torch_names:
                raise RuntimeError(f"candidate references torch operator computation: {node.attr}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in ("__import__", "eval", "exec", "compile", "open"):
                raise RuntimeError(f"candidate uses unsupported dynamic access: {node.func.id}")


def candidate_entry(config):
    entries = config["candidate"]["entrypoints"]
    if len(entries) != 1 or entries[0]["kind"] != "builder" or not entries[0]["symbol"].isidentifier():
        raise ValueError("This task requires one declared builder entrypoint")
    return entries[0], ROOT / entries[0]["file"]


def initial_state(config):
    """The declared candidate's state, read from its file without executing it."""
    entry, path = candidate_entry(config)
    if not path.exists():
        return "unimplemented"
    tree = ast.parse(path.read_text())
    if any(isinstance(node, ast.FunctionDef) and node.name == entry["symbol"] for node in tree.body):
        return "implemented"
    # A shipped stub holds only a docstring and __future__ imports; anything
    # else is broken candidate code, not an initial generation target.
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        raise RuntimeError("The candidate file contains code but does not define its builder")
    return "unimplemented"


def load_builder(config):
    entry, path = candidate_entry(config)
    source = path.read_text()
    compile(source, str(path), "exec")
    assert_source_independent(source)
    if initial_state(config) != "implemented":
        raise RuntimeError(f"Candidate is unimplemented: {entry['symbol']} is missing")
    spec = importlib.util.spec_from_file_location("_sikl_candidate", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    builder = getattr(module, entry["symbol"], None)
    if not callable(builder):
        raise RuntimeError("Declared candidate builder is not callable")
    return builder


def build_launch(builder, definition, row):
    launch = builder(**dimensions(definition, row))
    if not callable(launch):
        raise RuntimeError("Candidate builder did not return a callable launch")
    return launch


def wrong_output(expected):
    if isinstance(expected, dict):
        return {name: wrong_output(value) for name, value in expected.items()}
    if isinstance(expected, (tuple, list)):
        return type(expected)(wrong_output(value) for value in expected)
    if expected.is_floating_point():
        # Finite, same-shape/dtype adversarial values exercise the numerical
        # rule rather than merely the NaN/shape guard.
        return torch.where(expected >= 0, -torch.ones_like(expected), torch.ones_like(expected)) * 1000
    return torch.bitwise_not(expected)


def validate_row(definition, row, policy, reference, values, device="cuda"):
    expected = reference(**values)
    if compare_outputs(expected, expected, definition, row, device)["status"] != "PASS":
        raise ValueError("Comparison rejected the reference output")
    if compare_outputs(wrong_output(expected), expected, definition, row, device)["status"] == "PASS":
        raise ValueError("Comparison accepted deliberately incorrect finite outputs")
    again = make_inputs(definition, row, policy, device=device)
    assert_unmodified(again, snapshot(values))
    held = persistent_inputs(definition, policy)
    first, second = call_varying_draws(values, definition, row, policy, [1, 2], device)
    if all(torch.equal(first[name], second[name]) for name in first):
        raise ValueError("Call-varying draws do not vary")
    return {"reference_self_check": True, "wrong_output_rejected": True, "deterministic_inputs": True,
            "persistent_inputs": list(held)}


def runtime_evidence(contract):
    versions = {"torch": torch.__version__, "rocm": torch.version.hip}
    for name in ("flydsl", "aiter"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    baseline = ROOT / "scripts/baseline/main.py"
    return {"runtime_versions": versions, "device": torch.cuda.get_device_name(0),
            "arch": torch.cuda.get_device_properties(0).gcnArchName,
            "baseline_wrapper_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest()}


def run_row(action, role, row, contract, reference, baseline, builder, diagnostic):
    """One workload row's result; every row runs independently of the others."""
    definition, policy = contract["definition"], contract["policy"]
    values = make_inputs(definition, row, policy)
    validate_inputs(values, definition, row, "cuda")
    if action == "validate-task":
        return {"status": "PASS", "metadata": validate_row(definition, row, policy, reference, values)}
    call = baseline if role == "baseline" else build_launch(builder, definition, row)
    if action == "performance":
        return time_row(call, reference, values, definition, row, policy, role=role,
                        baseline_diagnostic=diagnostic)
    before = snapshot(values)
    expected = reference(**values)
    got = call(**values)
    torch.cuda.synchronize()
    assert_unmodified(values, before)
    verdict = compare_outputs(got, expected, definition, row, "cuda")
    if action == "correctness" or verdict.get("failure_kind") == "output_contract":
        return verdict
    # Compilation executes one real specialization; numerical acceptance is
    # the correctness action's verdict.
    return {"status": "PASS", "metadata": {"runtime_specialization_executed": True}}


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    return value


def run(role, action):
    report = {"protocol": "arena-eval-v1", "role": role, "action": action,
              "status": "PASS", "cases": [], "metadata": {}}
    try:
        config = yaml.safe_load((ROOT / "config.yaml").read_text())
        contract = json.loads((ROOT / "scripts/workload.json").read_text())
        report["cases"] = [dict(case, status="FAIL", reason="Not executed") for case in contract["cases"]]
        if action == "validate-task":
            report["metadata"]["candidate_state"] = initial_state(config)
        builder = load_builder(config) if role == "candidate" else None
        if not torch.cuda.is_available() or not torch.version.hip:
            raise RuntimeError("This task requires a compatible ROCm GPU")
        report["metadata"].update(runtime_evidence(contract))
        reference = load_solution(ROOT / "scripts/reference", contract["reference_spec"]["entry_point"])
        baseline = (load_solution(ROOT / "scripts/baseline", contract["baseline_spec"]["entry_point"])
                    if role == "baseline" else None)
        diagnostic = config["baseline"].get("correctness_policy") == "diagnostic"
        for row, result in zip(contract["rows"], report["cases"]):
            try:
                result.update(run_row(action, role, row, contract, reference, baseline, builder, diagnostic))
                if result["status"] == "PASS":
                    result.pop("reason", None)
            except Exception as error:
                traceback.print_exc(file=sys.stderr)
                result.update(status="FAIL", failure_kind="execution_error", reason=f"{type(error).__name__}: {error}")
            print(f"{role} {action}: {row['workload']['uuid']} {result['status']}", file=sys.stderr, flush=True)
        failed = [case for case in report["cases"] if case["status"] != "PASS"]
        if failed:
            kinds = {case.get("failure_kind", "execution_error") for case in failed}
            report.update(status="FAIL", reason=f"{len(failed)}/{len(report['cases'])} cases failed; see per-case evidence",
                          failure_kind=next(iter(kinds)) if len(kinds) == 1 else "multiple_failures")
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        for case in report["cases"]:
            if case["status"] != "PASS":
                case.update(failure_kind="execution_error", reason=f"{type(error).__name__}: {error}")
        report.update(status="FAIL", reason=f"{type(error).__name__}: {error}", failure_kind="execution_error")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("validate-task", "baseline", "candidate"))
    parser.add_argument("action", nargs="?", choices=("compile", "correctness", "performance"))
    args = parser.parse_args(argv)
    if (args.role == "validate-task") != (args.action is None):
        parser.error("Use validate-task or <baseline|candidate> <compile|correctness|performance>")
    role, action = ("task", "validate-task") if args.role == "validate-task" else (args.role, args.action)
    report = json_safe(run(role, action))
    print("ARENA_EVAL_RESULT=" + json.dumps(report, allow_nan=False), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
