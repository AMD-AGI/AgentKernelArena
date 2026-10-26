#!/usr/bin/env python3
"""Seven task-owned Arena-v2 actions for one bound whole-call workload."""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import importlib
import json
import math
from pathlib import Path
import statistics
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
IDENTITY = ("test_case_id", "shape", "dtype", "params")


def public_case(case, *, manifest=False):
    row = {key: deepcopy(case[key]) for key in IDENTITY if key in case}
    if manifest:
        row["checks"] = deepcopy(case["checks"])
    return row


def read_binding():
    data = json.loads((ROOT / "workloads.json").read_text())
    if data.get("binding_status") != "frozen":
        raise ValueError("Workload cases, frequencies and numerical contract are not yet bound")
    cases = data["cases"]
    ids = [case["test_case_id"] for case in cases]
    if not cases or len(ids) != len(set(ids)) or any(not isinstance(i, str) or not i for i in ids):
        raise ValueError("The protected manifest requires nonempty unique case IDs")
    for case in cases:
        checks = case["checks"]
        if not checks or set(checks) - {"correctness", "performance"}:
            raise ValueError("Invalid case checks")
        if "performance" in checks and "correctness" not in checks:
            raise ValueError("Every performance case needs correctness coverage")
        if case.get("params", {}).get("benchmark_method") not in ("cuda_graph", "cuda_event_fallback"):
            raise ValueError("Every case must explicitly declare params.benchmark_method")
    # The common evaluator performs weight validation/scoring. Require this
    # task's intended objective so final authoring cannot silently use defaults.
    import yaml

    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    aggregation = config["evaluation"].get("aggregation", {})
    performance_ids = {case["test_case_id"] for case in cases if "performance" in case["checks"]}
    if not performance_ids or aggregation.get("method") != "weighted_total_time" or set(aggregation.get("case_weights", {})) != performance_ids:
        raise ValueError("The fixed-workload task needs weights for exactly its performance cases")
    source = aggregation.get("source")
    if not isinstance(source, dict):
        raise ValueError("Measured frequency provenance is required")
    path = (ROOT / source["path"]).resolve(strict=True)
    if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
        raise ValueError("Frequency provenance is outside the task or has changed")
    from reference import validate_contract

    validate_contract(data["numerical_contract"])
    timing = data["timing"]
    for key in ("warmup", "repetition", "max_graph_repeats"):
        if type(timing.get(key)) is not int or timing[key] <= 0:
            raise ValueError(f"A positive frozen timing policy is required: {key}")
    if type(timing.get("target_ms")) not in (int, float) or not math.isfinite(timing["target_ms"]) or timing["target_ms"] <= 0:
        raise ValueError("A positive finite target_ms is required")
    json.dumps(data, allow_nan=False)
    return data


def candidate_callable():
    path = ROOT / "source/kernel.py"
    ast.parse(path.read_text(), filename="source/kernel.py")
    module = importlib.import_module("source.kernel")
    function = getattr(module, "fused_moe", None)
    if not callable(function):
        raise ValueError("source/kernel.py must define callable fused_moe")
    return function


def run_action(role, action, data):
    from runtime import verify_runtime
    from cases import load_case, snapshot_inputs, check_inputs_unchanged
    from reference import reference, check_output, check_reference_controls

    metadata = verify_runtime()
    if role == "task":
        tree = ast.parse((ROOT / "source/kernel.py").read_text())
        entry = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "fused_moe"), None)
        if entry is None or not any(isinstance(n, ast.Return) and isinstance(n.value, ast.Call) for n in ast.walk(entry)):
            raise ValueError("Starting fused_moe delegation is absent or unimplemented")
        metadata["candidate_state"] = "implemented"
        # Check the reference's actual input/load/math path, not just its name.
        for case in data["cases"]:
            payload = load_case(ROOT, case, data["numerical_contract"])
            metadata.setdefault("reference_controls", {})[case["test_case_id"]] = check_reference_controls(payload, data["numerical_contract"])
        return [public_case(c, manifest=True) | {"status": "PASS"} for c in data["cases"]], metadata
    if role == "baseline":
        from aiter.fused_moe import fused_moe
    else:
        fused_moe = candidate_callable()
    rows = []
    for case in data["cases"]:
        if action != "compile" and action not in case["checks"]:
            continue
        payload = load_case(ROOT, case, data["numerical_contract"])
        original_inputs = snapshot_inputs(payload)
        launch = lambda: fused_moe(*payload["args"], **payload["kwargs"])
        if action == "compile":
            # Executes the actual entrypoint so its JIT/native dependencies must
            # build successfully; Python syntax alone is not GPU compilation.
            launch()
            import torch
            torch.cuda.synchronize()
            check_inputs_unchanged(original_inputs)
            continue
        expected = reference(payload, data["numerical_contract"])
        row = public_case(case)
        if action == "correctness":
            row["metrics"] = check_output(launch(), expected, data["numerical_contract"])
        else:
            from _aka_benchmark import TimedRun, benchmark_cuda_graph_or_events_samples

            timed = TimedRun()
            timed.after_sample = lambda output: check_output(output, expected, data["numerical_contract"])
            policy = data["timing"]
            declared_method = case["params"]["benchmark_method"]
            samples, timing = benchmark_cuda_graph_or_events_samples(
                launch, warmup=policy["warmup"], repetition=policy["repetition"],
                target_ms=policy["target_ms"], max_graph_repeats=policy["max_graph_repeats"],
                use_cuda_graph=declared_method == "cuda_graph", timed_run=timed,
            )
            if timing.get("benchmark_method") != declared_method:
                raise ValueError("Measured timing method differs from the protected case declaration")
            check_output(timed.outputs, expected, data["numerical_contract"])
            check_output(timed.rerun(), expected, data["numerical_contract"])
            elapsed = statistics.median(samples)
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError("Invalid device timing")
            row.update(execution_time_ms=elapsed, benchmark_method=timing["benchmark_method"], metadata=timing)
        check_inputs_unchanged(original_inputs)
        row["status"] = "PASS"
        rows.append(row)
    return rows, metadata


def invoke(argv):
    role, action = "task", "validate-task"
    if len(argv) == 2 and argv[0] in ("baseline", "candidate") and argv[1] in ("compile", "correctness", "performance"):
        role, action = argv
    report = {"protocol": "arena-eval-v1", "role": role, "action": action, "status": "FAIL", "cases": []}
    try:
        if argv != ["validate-task"] and not (len(argv) == 2 and role in ("baseline", "candidate") and argv == [role, action]):
            raise ValueError("Use validate-task or {baseline,candidate} {compile,correctness,performance}")
        data = read_binding()
        rows, metadata = run_action(role, action, data)
        report.update(status="PASS", cases=rows, metadata=metadata)
        json.dumps(report, allow_nan=False)
        return report, 0
    except (Exception, SystemExit) as error:
        traceback.print_exc(file=sys.stderr)
        report.update(reason=f"{type(error).__name__}: {error}", failure_kind="evaluation_error")
        return report, 1


if __name__ == "__main__":
    result, code = invoke(sys.argv[1:])
    print("ARENA_EVAL_RESULT=" + json.dumps(result, allow_nan=False), flush=True)
    raise SystemExit(code)
