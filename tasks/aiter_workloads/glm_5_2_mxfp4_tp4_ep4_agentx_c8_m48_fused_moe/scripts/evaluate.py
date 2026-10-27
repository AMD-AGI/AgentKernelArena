#!/usr/bin/env python3
"""Standard Arena actions for one explicit configuration-derived MoE call."""
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


def public_case(case, *, manifest=False):
    row = {key: deepcopy(case[key]) for key in ("test_case_id", "shape", "dtype", "params")}
    if manifest:
        row["checks"] = list(case["checks"])
    return row


def read_binding():
    import yaml

    data = json.loads((ROOT / "workloads.json").read_text())
    cases = data.get("cases")
    if data.get("binding_status") != "frozen_config_derived_per_call" or not isinstance(cases, list) or len(cases) != 1:
        raise ValueError("Expected one frozen configuration-derived per-call scenario")
    case = cases[0]
    if (case.get("test_case_id") != "local_ep4_m48" or case.get("shape") != [48, 6144, 2048, 65, 9]
            or case.get("dtype") != "bfloat16" or case.get("checks") != ["correctness", "performance"]):
        raise ValueError("Case identity differs from the declared MoE scenario")
    expected = {"ep_rank": 0, "input_seed": 17, "fresh_seed": 29, "weight_seed": 0,
                "benchmark_method": "cuda_graph", "input_policy": "upstream_style_normal_mxfp4_with_global_top8_plus_shared"}
    if case.get("params") != expected:
        raise ValueError("Input distribution, rank or timing policy differs")
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    aggregation = config["evaluation"].get("aggregation", {})
    if (aggregation.get("method") != "weighted_total_time"
            or aggregation.get("case_weights") != {"local_ep4_m48": 1}):
        raise ValueError("The objective must remain one local invocation")
    source = aggregation.get("source", {})
    if source.get("path") != "frequency_provenance.json":
        raise ValueError("Per-call objective provenance is required")
    body = (ROOT / source["path"]).read_bytes()
    if hashlib.sha256(body).hexdigest() != source.get("sha256"):
        raise ValueError("Objective provenance digest differs")
    provenance = json.loads(body)
    if (provenance.get("status") != "config_derived_per_call"
            or provenance.get("case_weights") != aggregation["case_weights"]
            or provenance.get("observed_workload_frequency") is not False):
        raise ValueError("Per-call weights cannot be relabeled as observed frequencies")
    source_record = data["workload_source"]
    if source_record.get("path") != "workload_source.json":
        raise ValueError("Workload source record is required")
    if hashlib.sha256((ROOT / source_record["path"]).read_bytes()).hexdigest() != source_record.get("sha256"):
        raise ValueError("Workload source digest differs")
    if provenance.get("source") != source_record:
        raise ValueError("Objective and workload source bindings disagree")
    if data.get("timing") != {"warmup": 10, "repetition": 100, "target_ms": 1.0, "max_graph_repeats": 1000}:
        raise ValueError("Frozen timing policy differs")
    json.dumps(data, allow_nan=False)
    return data


def candidate_callable():
    for path in sorted((ROOT / "source").rglob("*.py")):
        compile(path.read_text(), str(path.relative_to(ROOT)), "exec")
    function = getattr(importlib.import_module("source.kernel"), "fused_moe", None)
    if not callable(function):
        raise ValueError("Candidate must define callable source.kernel.fused_moe")
    return function


def observed_native_modules():
    result = {}
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("module_") and isinstance(path, str) and Path(path).suffix == ".so":
            file = Path(path)
            if file.is_file():
                result[name] = {"sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                                "bytes": file.stat().st_size}
    return result


def run_action(role, action, data):
    from runtime import verify_runtime
    from cases import (load_case, dynamic_inputs, replace_dynamic,
                       snapshot_inputs, check_inputs_unchanged)
    from reference import reference, check_output, compare_output, reference_controls

    metadata = verify_runtime()
    case = data["cases"][0]
    payload = load_case(case)
    fresh = dynamic_inputs(case, case["params"]["fresh_seed"])
    if role == "task":
        candidate_callable()
        tree = ast.parse((ROOT / "source/kernel.py").read_text())
        entry = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "fused_moe"), None)
        if entry is None or not any(isinstance(node, ast.Return) for node in ast.walk(entry)):
            raise ValueError("Starting candidate is absent or unimplemented")
        metadata["candidate_state"] = "implemented"
        before = snapshot_inputs(payload)
        metadata["reference_controls"] = reference_controls(payload)
        initial = reference(payload)
        check_inputs_unchanged(before)
        del before
        replace_dynamic(payload, fresh)
        second = reference(payload)
        if compare_output(initial, second)["accepted"]:
            raise ValueError("Fresh input state cannot reject a stale output")
        metadata["fresh_state_rejects_stale_output"] = True
        return [public_case(case, manifest=True) | {"status": "PASS"}], metadata

    if role == "baseline":
        from aiter.fused_moe import fused_moe
    else:
        fused_moe = candidate_callable()
    launch = lambda: fused_moe(*payload["args"], **payload["kwargs"])
    before = snapshot_inputs(payload)
    if action == "compile":
        launch()
        import torch
        torch.cuda.synchronize()
        check_inputs_unchanged(before)
        metadata["loaded_native_modules"] = observed_native_modules()
        return [], metadata
    expected = reference(payload)
    row = public_case(case)
    if action == "correctness":
        row["metrics"] = check_output(launch(), expected)
        replay = launch
    else:
        from _aka_benchmark import TimedRun, benchmark_cuda_graph_or_events_samples

        timed = TimedRun()
        timed.after_sample = lambda output: check_output(output, expected)
        samples, timing = benchmark_cuda_graph_or_events_samples(
            launch, **data["timing"], use_cuda_graph=True, timed_run=timed)
        if timing.get("benchmark_method") != "cuda_graph":
            raise ValueError("Graph timing cannot silently fall back")
        check_output(timed.outputs, expected)
        check_output(timed.rerun(), expected)
        elapsed = statistics.median(samples)
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("Invalid device timing")
        row.update(execution_time_ms=elapsed, benchmark_method="cuda_graph", metadata=timing)
        replay = timed.rerun
    check_inputs_unchanged(before)
    del before
    replace_dynamic(payload, fresh)
    before = snapshot_inputs(payload)
    second = reference(payload)
    if compare_output(expected, second)["accepted"]:
        raise ValueError("Fresh input state cannot reject a stale output")
    row.setdefault("metadata", {})["fresh_value_replay"] = check_output(replay(), second)
    check_inputs_unchanged(before)
    row["status"] = "PASS"
    metadata["loaded_native_modules"] = observed_native_modules()
    return [row], metadata


def invoke(argv):
    role, action = "task", "validate-task"
    if len(argv) == 2 and argv[0] in ("baseline", "candidate"):
        role, action = argv
    report = {"protocol": "arena-eval-v1", "role": role, "action": action,
              "status": "FAIL", "cases": []}
    try:
        if argv != ["validate-task"] and not (role in ("baseline", "candidate")
                and action in ("compile", "correctness", "performance") and argv == [role, action]):
            raise ValueError("Use validate-task or {baseline,candidate} {compile,correctness,performance}")
        rows, metadata = run_action(role, action, read_binding())
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
