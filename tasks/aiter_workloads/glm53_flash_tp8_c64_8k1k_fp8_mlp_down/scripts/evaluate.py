#!/usr/bin/env python3
"""Standard Arena actions for one config-derived GLM5.3-Flash GEMM site."""
from copy import deepcopy
import ast
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_contract():
    import yaml

    data = json.loads((ROOT / "workloads.json").read_text())
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    aggregation = config["evaluation"]["aggregation"]
    source = aggregation["source"]
    body = (ROOT / source["path"]).read_bytes()
    if hashlib.sha256(body).hexdigest() != source["sha256"]:
        raise ValueError("Frequency provenance hash mismatch")
    provenance = json.loads(body)
    cases = data["cases"]
    ids = [case["test_case_id"] for case in cases]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Nonempty unique cases are required")
    if aggregation["method"] != "weighted_total_time" or aggregation["case_weights"] != provenance["case_weights"]:
        raise ValueError("Configured objective differs from frequency provenance")
    if set(ids) != set(aggregation["case_weights"]):
        raise ValueError("Weights must cover exactly the performance cases")
    if provenance["status"] != "configuration_derived_scenario":
        raise ValueError("This task does not declare measured serving frequencies")
    model = json.loads((ROOT / "model_config.json").read_text())
    if hashlib.sha256((ROOT / "model_config.json").read_bytes()).hexdigest() != data["model"]["config_sha256"]:
        raise ValueError("Pinned model config differs")
    text = model["text_config"]
    n, k = text["hidden_size"], text["intermediate_size"] // data["workload"]["tp"]
    if (n, k) != (4096, 1536) or any(c["shape"][1:] != [n, k] for c in cases):
        raise ValueError("Cases do not match the selected model/TP projection")
    for case in cases:
        if case["checks"] != ["correctness", "performance"]:
            raise ValueError("Every scored case requires correctness")
    return data


def case_row(case, manifest=False):
    keys = ("test_case_id", "shape", "dtype", "params")
    row = {key: deepcopy(case[key]) for key in keys}
    if manifest:
        row["checks"] = deepcopy(case["checks"])
    return row


def candidate():
    ast.parse((ROOT / "source/kernel.py").read_text())
    fn = getattr(importlib.import_module("source.kernel"), "mlp_down_gemm", None)
    if not callable(fn):
        raise ValueError("source.kernel.mlp_down_gemm must be callable")
    return fn


def baseline(x, weight, x_scale, weight_scale):
    import aiter
    import torch

    return aiter.gemm_a8w8_blockscale_bpreshuffle(x, weight, x_scale, weight_scale, dtype=torch.bfloat16)


def execute(role, action, data):
    import torch
    from cases import make_case, snapshot, check_unchanged
    from reference import reference, check_output
    from runtime import verify_runtime

    metadata = verify_runtime()
    torch.backends.cuda.matmul.allow_tf32 = False
    fn = baseline if role == "baseline" else candidate()
    rows = []
    for case in data["cases"]:
        args, canonical = make_case(case, data["input_policy"]["seed"])
        before = snapshot(args)
        expected = reference(args[0], canonical, args[2], args[3])
        actual = fn(*args)  # Includes real JIT compilation for compile actions.
        metrics = check_output(actual, expected)
        check_unchanged(args, before)
        row = case_row(case, manifest=role == "task")
        if role == "task":
            try:
                check_output(torch.zeros_like(expected), expected)
            except AssertionError:
                pass
            else:
                raise AssertionError("The oracle accepted a zero-output negative control")
        elif action == "performance":
            from _aka_benchmark import TimedRun, benchmark_cuda_graph_or_events_samples

            timed = TimedRun()
            timed.after_sample = lambda output: check_output(output, expected)
            values, timing = benchmark_cuda_graph_or_events_samples(
                lambda: fn(*args), timed_run=timed,
                use_cuda_graph=case["params"]["benchmark_method"] == "cuda_graph",
                fallback_reason="declared_eager_prefill", **data["timing"]
            )
            if timing["benchmark_method"] != case["params"]["benchmark_method"]:
                raise RuntimeError("Observed timing method differs from the declared case")
            check_unchanged(args, before)
            row.update(execution_time_ms=statistics.median(values),
                       benchmark_method=timing["benchmark_method"], metadata=timing)
            # Same public addresses and retained graph, with new dynamic values.
            fresh, _ = make_case(case, data["input_policy"]["replay_seed"])
            args[0].copy_(fresh[0])
            args[2].copy_(fresh[2])
            replay_before = snapshot(args)
            replay_expected = reference(args[0], canonical, args[2], args[3])
            check_output(timed.rerun(), replay_expected)
            check_unchanged(args, replay_before)
        if role == "task" or action == "correctness":
            fresh, fresh_canonical = make_case(case, data["input_policy"]["replay_seed"])
            fresh_before = snapshot(fresh)
            fresh_expected = reference(fresh[0], fresh_canonical, fresh[2], fresh[3])
            check_output(fn(*fresh), fresh_expected)
            check_unchanged(fresh, fresh_before)
        row.update(status="PASS", metrics=metrics)
        rows.append(row)
    if role == "task":
        metadata["candidate_state"] = "implemented"
    metadata["case_policy"] = "configuration_derived_scenario"
    metadata["serving_integration_verified"] = False
    return {"protocol": "arena-eval-v1", "role": role, "action": action,
            "status": "PASS", "cases": rows, "metadata": metadata}


def main():
    argv = sys.argv[1:]
    role, action = ("task", "validate-task") if argv == ["validate-task"] else (argv + ["", ""])[:2]
    try:
        if (role, action) != ("task", "validate-task") and not (
            role in ("baseline", "candidate") and action in ("compile", "correctness", "performance")
        ):
            raise ValueError("Expected validate-task or baseline|candidate compile|correctness|performance")
        result = execute(role, action, load_contract())
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        result = {"protocol": "arena-eval-v1", "role": role, "action": action,
                  "status": "FAIL", "reason": str(error), "cases": []}
    print("ARENA_EVAL_RESULT=" + json.dumps(result, allow_nan=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
