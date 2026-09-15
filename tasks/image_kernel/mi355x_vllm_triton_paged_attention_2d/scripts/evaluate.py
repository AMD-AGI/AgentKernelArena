#!/usr/bin/env python3
"""Self-contained arena-eval-v1 entrypoint; baseline uses its assigned snapshot."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "ARENA_EVAL_RESULT="


def load_harness():
    spec = importlib.util.spec_from_file_location("arena_image_harness", ROOT / "scripts/task_runner.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def manifest():
    data = json.loads((ROOT / "workloads.json").read_text())
    json.dumps(data, allow_nan=False)
    rows = data["cases"]
    ids = [row["test_case_id"] for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("Workload manifest requires nonempty unique case IDs")
    for row in rows:
        checks = row["checks"]
        if not checks or set(checks) - {"correctness", "performance"}:
            raise ValueError("Invalid manifest checks")
        if "performance" in checks and "correctness" not in checks:
            raise ValueError("Scored case has no correctness coverage")
    return deepcopy(rows)


def case_result(case, status):
    return {key: deepcopy(value) for key, value in case.items() if key != "checks"} | {"status": status}


def validate_performance(raw, expected):
    if not isinstance(raw, list):
        raise ValueError("Harness did not return fresh performance rows")
    by_id = {row["test_case_id"]: row for row in raw}
    if len(by_id) != len(raw) or set(by_id) != {c["test_case_id"] for c in expected}:
        raise ValueError("Measured case IDs differ from the protected manifest")
    result = []
    for case in expected:
        measured = by_id[case["test_case_id"]]
        # Some original harnesses express shape only in metadata. Preserve the
        # independent manifest identity, and reject conflicting explicit shapes.
        if "shape" in measured and measured["shape"] != case.get("shape"):
            raise ValueError(f"Measured shape changed: {case['test_case_id']}")
        latency = measured["execution_time_ms"]
        if type(latency) not in (int, float) or not math.isfinite(latency) or latency <= 0:
            raise ValueError("Device latency must be finite and positive")
        meta = measured.get("metadata", {})
        for key, value in case.get("params", {}).items():
            if key in meta and meta[key] != value:
                raise ValueError(f"Measured input parameter changed: {key}")
        method = measured.get("benchmark_method", meta.get("benchmark_method"))
        if method not in ("cuda_graph", "cuda_event_fallback"):
            raise ValueError("Missing or unsupported device benchmark method")
        row = case_result(case, "PASS")
        row.update(execution_time_ms=latency, benchmark_method=method, metadata=deepcopy(meta))
        for key, value in measured.items():
            if key.startswith("benchmark_") and key != "benchmark_method":
                row["metadata"][key] = value
        result.append(row)
    return result


def invoke(argv, *, harness_loader=load_harness):
    role, action = "task", "validate-task"
    if (len(argv) == 2 and argv[0] in ("baseline", "candidate")
            and argv[1] in ("compile", "correctness", "performance")):
        role, action = argv
    rows = []
    report = {"protocol": "arena-eval-v1", "role": role, "action": action,
              "status": "FAIL", "cases": []}
    try:
        if argv != ["validate-task"] and not (
            len(argv) == 2 and role in ("baseline", "candidate")
            and action in ("compile", "correctness", "performance")
        ):
            raise ValueError("Use validate-task or {baseline,candidate} {compile,correctness,performance}")
        rows = manifest()
        from task_adapter import prepare, validate_workloads, run_correctness
        from setup_task import verify_sources
        source_evidence = verify_sources()
        harness = harness_loader()
        validate_workloads(harness)
        build_evidence = prepare(harness)
        metadata = {"candidate_state": "implemented", "baseline_kind": "initial_candidate",
                    "implementation_sources": source_evidence,
                    "phase": os.environ.get("ARENA_EVAL_PHASE", "candidate_evaluation")}
        if role == "task":
            from reference_controls import check_reference
            metadata["reference_controls"] = check_reference(harness)
            report["cases"] = [dict(case, status="PASS") for case in rows]
        elif action == "compile":
            # The original compile smoke launches the actual JIT/operator path.
            harness.run_compile()
        elif action == "correctness":
            run_correctness(harness)
            report["cases"] = [case_result(c, "PASS") for c in rows if action in c["checks"]]
        else:
            expected = [c for c in rows if action in c["checks"]]
            report["cases"] = validate_performance(harness.run_performance(), expected)
        if role != "task" and build_evidence is not None:
            metadata["source_build"] = build_evidence.finish()
        report.update(status="PASS", metadata=metadata)
        # No NaN/Infinity leaks through an otherwise passing JSON envelope.
        json.dumps(report, allow_nan=False)
        return report, 0
    except (Exception, SystemExit) as exc:
        traceback.print_exc(file=sys.stderr)
        # An assertion can indicate shape, dtype, dispatch or state corruption.
        # Do not relabel it as a diagnostic-eligible numerical mismatch.
        report = {"protocol": "arena-eval-v1", "role": role, "action": action,
                  "status": "FAIL", "reason": f"{type(exc).__name__}: {exc}",
                  "failure_kind": "evaluation_error", "cases": []}
        report["cases"] = [dict(case_result(c, "FAIL"), reason=report["reason"],
                                failure_kind="evaluation_error")
                           for c in rows if action == "validate-task" or action in c["checks"]]
        if action == "validate-task":
            for result, case in zip(report["cases"], rows):
                result["checks"] = case["checks"]
        return report, 1


def main():
    report, code = invoke(sys.argv[1:])
    print(PREFIX + json.dumps(report, allow_nan=False), flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
