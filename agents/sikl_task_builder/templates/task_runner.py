#!/usr/bin/env python3
"""Protected arena-eval-v1 runner for functional SIKL tasks."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from scripts.task_api import (assert_outputs, assert_unmodified, clone_inputs,
                              load_solution, outputs, poison_outputs, validate_inputs)
from scripts.task_inputs import make_inputs, refill_inputs


class TimedRun:
    def _bind(self, rerun, outputs=None):
        self.rerun = rerun
        self.outputs = outputs


def candidate():
    path = ROOT / "source" / "kernel.py"
    spec = importlib.util.spec_from_file_location("aka_candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "run", None)):
        raise ValueError("Candidate must expose callable run(**kwargs)")
    return module.run


def wrong_output(expected):
    if isinstance(expected, dict):
        return {name: wrong_output(value) for name, value in expected.items()}
    if isinstance(expected, (tuple, list)):
        return type(expected)(wrong_output(value) for value in expected)
    if expected.is_floating_point():
        # Finite, same-shape/dtype adversarial values exercise the numerical
        # rule rather than merely testing the NaN/shape guard.
        return torch.where(expected >= 0, -torch.ones_like(expected), torch.ones_like(expected)) * 1000
    return torch.bitwise_not(expected)


def validate_case(definition, row, policy, reference, values, device="cuda"):
    expected = reference(**clone_inputs(values))
    assert_outputs(expected, expected, definition, row, policy, device)
    try:
        assert_outputs(wrong_output(expected), expected, definition, row, policy, device)
    except AssertionError:
        pass
    else:
        raise ValueError("Comparison accepted deliberately incorrect finite outputs")
    again = make_inputs(definition, row, policy, device=device)
    validate_inputs(again, definition, row, device)
    assert_unmodified(values, again)
    return {"reference_self_check": True, "wrong_output_rejected": True,
            "deterministic_inputs": True}


def measure_case(launch, reference, values, definition, row, policy, device="cuda"):
    from _aka_benchmark import benchmark_cuda_graph_or_events
    pristine = clone_inputs(values)
    expected = reference(**clone_inputs(values))
    replay = TimedRun()
    elapsed, timing = benchmark_cuda_graph_or_events(
        lambda: launch(**values), warmup=policy["warmup"],
        repetition=policy["repetition"], target_ms=policy["target_ms"], timed_run=replay,
    )
    assert_unmodified(pristine, values)
    poison_outputs(replay.outputs, expected, values, definition, row, device)
    assert_outputs(replay.rerun(), expected, definition, row, policy, device)
    assert_unmodified(pristine, values)
    # A correct capture over the original inputs must not conceal cached host
    # work or a stale output copy. Validate the same measured replay after refill.
    refill_inputs(values, definition, row, policy, device=device)
    validate_inputs(values, definition, row, device)
    changed = clone_inputs(values)
    expected = reference(**clone_inputs(values))
    poison_outputs(replay.outputs, expected, values, definition, row, device)
    assert_outputs(replay.rerun(), expected, definition, row, policy, device)
    assert_unmodified(changed, values)
    return {"execution_time_ms": elapsed, "benchmark_method": timing["benchmark_method"],
            "metadata": {"device_timing": timing, "timed_output_checked": True,
                         "exact_graph_replay_validated": True, "refilled_input_replay_validated": True}}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("validate-task", "baseline", "candidate"))
    parser.add_argument("action", nargs="?", choices=("compile", "correctness", "performance"))
    args = parser.parse_args(argv)
    if (args.role == "validate-task") != (args.action is None):
        parser.error("Use validate-task or <baseline|candidate> <compile|correctness|performance>")
    role, action = ("task", "validate-task") if args.role == "validate-task" else (args.role, args.action)
    report = {"protocol": "arena-eval-v1", "role": role, "action": action,
              "status": "PASS", "cases": [], "metadata": {}}
    try:
        contract = json.loads((ROOT / "scripts/workload.json").read_text())
        definition, policy = contract["definition"], contract["policy"]
        report["cases"] = [dict(case, status="FAIL", reason="Not executed") for case in contract["cases"]]
        if action == "validate-task":
            for result in report["cases"]:
                result["checks"] = ["correctness", "performance"]
        if not torch.cuda.is_available() or not torch.version.hip:
            raise RuntimeError("This task requires a compatible ROCm GPU")
        for path in ROOT.rglob("*.py"):
            if "__pycache__" not in path.parts and not (role == "baseline" and path.is_relative_to(ROOT / "source")):
                compile(path.read_text(), str(path), "exec")
        reference = load_solution(ROOT / "scripts/reference", contract["reference_spec"]["entry_point"])
        # Baseline actions never import candidate code. Task validation inspects
        # and actually executes the initial candidate separately below.
        launch = (load_solution(ROOT / "scripts/baseline", contract["baseline_spec"]["entry_point"])
                  if role == "baseline" else candidate())
        for row, result in zip(contract["rows"], report["cases"]):
            values = make_inputs(definition, row, policy)
            validate_inputs(values, definition, row, "cuda")
            pristine = clone_inputs(values)
            if action == "validate-task":
                result["metadata"] = validate_case(definition, row, policy, reference, values)
                # Importing a stub is insufficient evidence for implemented.
                outputs(launch(**values), definition, row, "cuda")
            elif action == "correctness":
                expected = reference(**clone_inputs(values))
                assert_outputs(launch(**values), expected, definition, row, policy, "cuda")
            elif action == "compile":
                # Exercise lazy GPU compilation for every declared case.
                outputs(launch(**values), definition, row, "cuda")
            else:
                result.update(measure_case(launch, reference, values, definition, row, policy))
            if action != "performance":
                assert_unmodified(pristine, values)
            torch.cuda.synchronize()
            result.update(status="PASS")
            result.pop("reason", None)
            print(f"{role} {action}: {row['workload']['uuid']} PASS", flush=True)
            del values, pristine
        if action == "validate-task":
            report["metadata"]["candidate_state"] = "implemented"
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        report.update(status="FAIL", reason=f"{type(error).__name__}: {error}")
    print("ARENA_EVAL_RESULT=" + json.dumps(report, allow_nan=False), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
