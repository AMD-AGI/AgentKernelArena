#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Task-local explicit v2 role/actions around protected native HIP harnesses."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def manifest():
    data = json.loads((ROOT / "workload.json").read_text())
    cases = data["cases"]
    if not cases or len({row["test_case_id"] for row in cases}) != len(cases):
        raise ValueError("Empty or duplicate independent case manifest")
    return data


def checked_performance(expected, measured):
    if isinstance(measured, tuple):
        measured, error = measured
        if error:
            raise RuntimeError(error)
    if len(measured) != len(expected):
        raise RuntimeError("Native benchmark omitted declared cases")
    by_id = {row["test_case_id"]: row for row in measured}
    if len(by_id) != len(measured) or set(by_id) != {row["test_case_id"] for row in expected}:
        raise RuntimeError("Native benchmark case IDs differ from the independent manifest")
    result = []
    for row in expected:
        actual = by_id[row["test_case_id"]]
        if actual.get("params") != row["params"]:
            raise RuntimeError("Native benchmark changed a declared case identity")
        latency = actual.get("execution_time_ms")
        if type(latency) not in (int, float) or not math.isfinite(latency) or latency <= 0:
            raise RuntimeError("Native benchmark did not produce finite positive device time")
        if actual.get("benchmark_method") not in ("cuda_graph", "cuda_event_fallback"):
            raise RuntimeError("Native benchmark did not establish a supported device timing method")
        result.append({**row, "status": "PASS", "execution_time_ms": latency,
                       "benchmark_method": actual["benchmark_method"], "metadata": {"native_benchmark": actual}})
    return result


def run(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", nargs="+")
    args = parser.parse_args(argv)
    role, action = ("task", "validate-task") if args.operation == ["validate-task"] else tuple(args.operation)
    result = {"protocol": "arena-eval-v1", "role": role, "action": action, "status": "FAIL", "cases": []}
    cases = []
    try:
        if (role, action) not in {("task", "validate-task"), *((r, a) for r in ("baseline", "candidate") for a in ("compile", "correctness", "performance"))}:
            raise ValueError("Unsupported role/action")
        os.chdir(ROOT)
        (ROOT / "build").mkdir(exist_ok=True)
        os.environ["TORCH_EXTENSIONS_DIR"] = str(ROOT / "build" / "torch_extensions")
        data = manifest()
        cases = data["cases"]
        import task_runner as harness
        if [list(shape) for shape in harness.TEST_SHAPES] != data["test_shapes"]:
            raise ValueError("Protected harness cases no longer match workload manifest")
        # Source-dependent policy was evaluated on the original implementation.
        # A candidate cannot downgrade the frozen baseline's timing method.
        if "graph_policy" in data:
            harness.HIP_GRAPH_ENABLED = data["graph_policy"]["enabled"]
            harness.HIP_GRAPH_FALLBACK_REASON = data["graph_policy"]["reason"]
        import reference_checks
        if action == "validate-task":
            reference_checks.self_test(harness)
            for relative in data["candidate_files"]:
                path = (ROOT / relative).resolve()
                if not path.is_relative_to(ROOT) or not path.is_file() or not path.read_text().strip():
                    raise ValueError("Initial native candidate is missing, empty, or outside task")
            result["metadata"] = {"candidate_state": "implemented", "case_count": len(cases)}
            result["cases"] = [{**row, "status": "PASS", "checks": ["correctness", "performance"]} for row in cases]
        elif action == "compile":
            # The original harness invokes hipcc or torch's native extension build.
            # A text match, Python syntax check, or old binary alone cannot pass.
            ok, error = harness.run_compile()
            if not ok:
                raise RuntimeError(error or "Native compilation failed")
        elif action == "correctness":
            ok, error = harness.run_correctness()
            if not ok:
                raise RuntimeError(error or "Original native correctness check failed")
            reference_checks.check_additional_paths(harness)
            result["cases"] = [{**row, "status": "PASS"} for row in cases]
        else:
            result["cases"] = checked_performance(cases, harness.run_performance())
        result["status"] = "PASS"
    except Exception as exc:
        result.update(reason=f"{type(exc).__name__}: {exc}", failure_kind="execution_error")
        result["cases"] = [] if action == "compile" else [
            {**row, "status": "FAIL", "failure_kind": "not_completed",
             **({"checks": ["correctness", "performance"]} if action == "validate-task" else {})}
            for row in cases]
    print("ARENA_EVAL_RESULT=" + json.dumps(result, allow_nan=False))
    return 0 if result["status"] == "PASS" else 1


def main(argv=None):
    # Even argument/dependency errors belong to this invocation's envelope.
    tokens = list(sys.argv[1:] if argv is None else argv)
    try:
        return run(tokens)
    except (Exception, SystemExit) as exc:
        role = tokens[-2] if len(tokens) >= 2 and tokens[-2] in ("baseline", "candidate") else "task"
        action = tokens[-1] if role != "task" else "validate-task"
        result = {"protocol": "arena-eval-v1", "role": role, "action": action,
                  "status": "FAIL", "cases": [], "failure_kind": "execution_error",
                  "reason": f"{type(exc).__name__}: {exc}"}
        print("ARENA_EVAL_RESULT=" + json.dumps(result, allow_nan=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
