#!/usr/bin/env python3
"""Seven Arena-v2 actions for one fixed Kimi KDA decode scenario."""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import importlib
import json
import math
from pathlib import Path
import secrets
import statistics
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ENTRY = "fused_recurrent_kda"
IDENTITY = ("test_case_id", "shape", "dtype", "params")


def public_case(case, *, manifest=False):
    row = {key: deepcopy(case[key]) for key in IDENTITY}
    if manifest:
        row["checks"] = list(case["checks"])
    return row


def read_contract():
    import yaml

    data = json.loads((ROOT / "workloads.json").read_text())
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    lock = json.loads((ROOT / "source_identity.json").read_text())
    model_bytes = (ROOT / "model_config.json").read_bytes()
    if hashlib.sha256(model_bytes).hexdigest() != lock["model_config_sha256"]:
        raise ValueError("The checkpoint model config changed")
    model = json.loads(model_bytes)["text_config"]["linear_attn_config"]
    workload = data["workload"]
    if (workload["tp"], workload["concurrency"], workload["input_tokens"],
            workload["output_tokens"]) != (8, 64, 8192, 1024):
        raise ValueError("The selected workload settings changed")
    if workload["ep"] is not None or workload["dcp"] is not None:
        raise ValueError("EP/DCP have no recorded binding in this task")
    if model["num_heads"] % workload["tp"]:
        raise ValueError("KDA heads must divide the declared TP size")
    expected_shape = [64, 1, model["num_heads"] // workload["tp"], model["head_dim"]]
    if expected_shape != [64, 1, 12, 128] or model["gate_lower_bound"] != -5.0:
        raise ValueError("Unexpected checkpoint KDA geometry/gate")
    cases = data["cases"]
    ids = [case["test_case_id"] for case in cases]
    if ids != ["decode_b64_t1", "decode_b64_t1_padded"]:
        raise ValueError("Incomplete or changed case manifest")
    for case, padding in zip(cases, (0, 128)):
        if case["shape"] != expected_shape or case["dtype"] != "bfloat16":
            raise ValueError("Case geometry differs from the model/TP derivation")
        expected_checks = ["correctness", "performance"] if padding == 0 else ["correctness"]
        if case["checks"] != expected_checks:
            raise ValueError("Every performance case needs its declared correctness coverage")
        expected_params = {
            "phase": "decode", "tokens_per_sequence": 1, "state_slots": 192,
            "state_dtype": "float32", "state_layout": "V-first",
            "state_padding_floats": padding, "state_indices": "64 unique permuted slots",
            "qkv_dtype": "bfloat16", "g_beta_parameter_dtype": "float32",
            "lower_bound": -5.0, "scale": 128 ** -0.5, "qk_l2norm": True,
            "gate_in_kernel": True, "beta_sigmoid_in_kernel": True,
            "benchmark_method": "cuda_graph" if padding == 0 else None,
        }
        if case["params"] != expected_params:
            raise ValueError("The declared case parameters differ from the generated public call")
    aggregation = config["evaluation"]["aggregation"]
    if aggregation["method"] != "weighted_total_time" or aggregation["case_weights"] != {ids[0]: 1024}:
        raise ValueError("The derived one-site decode objective changed")
    reference = aggregation["source"]
    if reference["path"] != "frequency_provenance.json":
        raise ValueError("Unexpected frequency provenance path")
    body = (ROOT / reference["path"]).read_bytes()
    if hashlib.sha256(body).hexdigest() != reference["sha256"]:
        raise ValueError("Frequency provenance content differs")
    provenance = json.loads(body)
    if provenance["status"] != "config_derived_scenario" or provenance["case_weights"] != aggregation["case_weights"]:
        raise ValueError("Weights are not bound to the declared scenario")
    if data["numerical_contract"]["max_sequence_rms_ratio"] != 0.005:
        raise ValueError("The upstream-derived error limit changed")
    if data["timing"] != {"warmup": 10, "repetition": 100, "target_ms": 1.0,
                          "max_graph_repeats": 1, "method": "cuda_graph"}:
        raise ValueError("The frozen timing policy changed")
    json.dumps(data, allow_nan=False)
    return data


def check_candidate_source():
    path = ROOT / "source/kernel.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    if not any(isinstance(node, ast.FunctionDef) and node.name == ENTRY for node in tree.body):
        raise ValueError(f"The candidate must implement {ENTRY}")
    for source in (ROOT / "source").rglob("*.py"):
        compile(source.read_text(), str(source), "exec")


def selected_callable(role):
    if role == "baseline":
        module = importlib.import_module("aiter.ops.triton.attention.kda")
    else:
        module = importlib.import_module("source.kernel")
    call = getattr(module, ENTRY)
    if not callable(call):
        raise ValueError("The selected entrypoint is not callable")
    return call


def check_case(call, case, *, seed, limit):
    import torch
    from cases import make_case, refresh_operands, snapshot
    from reference import check_result, expected

    payload = make_case(case, seed)
    measurements = []
    # The second call consumes the first call's state, with changed operands.
    # A constant output or a candidate that silently resets state cannot pass.
    for step in range(2):
        if step:
            refresh_operands(payload, seed + 1)
        payload["kwargs"]["out"].fill_(float("nan"))
        before, gold = snapshot(payload), expected(payload)
        result = call(**payload["kwargs"])
        torch.cuda.synchronize()
        measurements.append(check_result(payload, result, gold, before, limit=limit))
    # Zero is a valid state. Exercise it independently of the random prior.
    active = payload["kwargs"]["ssm_state_indices"].long()
    payload["kwargs"]["initial_state"][active] = 0
    payload["kwargs"]["out"].fill_(float("nan"))
    before, gold = snapshot(payload), expected(payload)
    result = call(**payload["kwargs"])
    torch.cuda.synchronize()
    measurements.append(check_result(payload, result, gold, before, limit=limit))
    return {"seed": seed, "state_transitions": measurements}


def measure_case(call, case, timing, limit):
    from _aka_benchmark import TimedRun, benchmark_cuda_graph_or_events_samples
    from cases import make_case, prepare, refresh_operands, snapshot
    from reference import check_result, expected

    payload = make_case(case, 42)
    prepare(payload)
    before, gold = snapshot(payload), expected(payload)
    timed = TimedRun()
    checked = []

    def observe(result):
        # The helper invokes this after the timed end event, before any reset.
        checked.append(check_result(payload, result, gold, before, limit=limit))

    timed.after_sample = observe
    samples, metadata = benchmark_cuda_graph_or_events_samples(
        lambda: call(**payload["kwargs"]),
        warmup=timing["warmup"], repetition=timing["repetition"],
        target_ms=timing["target_ms"], max_graph_repeats=1,
        use_cuda_graph=True, prepare_fn=lambda: prepare(payload), timed_run=timed,
    )
    if metadata["benchmark_method"] != "cuda_graph" or metadata["benchmark_effective_repeats"] != 1:
        raise AssertionError("The fixed state transition requires one call per graph replay")
    if len(samples) != timing["repetition"] or len(checked) != len(samples):
        raise AssertionError("Every scored sample must have an output/state check")
    if any(not math.isfinite(value) or value <= 0 for value in samples):
        raise AssertionError("Invalid device timing")
    # Preserve the timed graph and buffer addresses. Change both operands and
    # the incoming state, then validate replay of that exact captured unit.
    refresh_operands(payload, 43)
    payload["pristine"].mul_(0.25)
    prepare(payload)
    current_before, current_gold = snapshot(payload), expected(payload)
    current = check_result(payload, timed.rerun(), current_gold, current_before, limit=limit)
    return statistics.median(samples), metadata, {
        "checked_timed_samples": len(checked), "fresh_value_graph_replay": current,
        "output_max_sequence_rms_ratio": max(x["output_max_sequence_rms_ratio"] for x in checked),
        "state_max_sequence_rms_ratio": max(x["state_max_sequence_rms_ratio"] for x in checked),
    }


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    role, action = ("task", "validate-task") if argv == ["validate-task"] else (argv + ["", ""])[:2]
    report = {"protocol": "arena-eval-v1", "role": role, "action": action,
              "status": "FAIL", "cases": []}
    try:
        if not (argv == ["validate-task"] or len(argv) == 2 and role in ("baseline", "candidate")
                and action in ("compile", "correctness", "performance")):
            raise ValueError("Expected validate-task or {baseline,candidate} {compile,correctness,performance}")
        data = read_contract()
        check_candidate_source()
        from runtime import verify_runtime

        report["metadata"] = {"runtime": verify_runtime(),
                              "workload_binding": data["binding_status"]}
        if action == "validate-task":
            for path in (ROOT / "scripts").glob("*.py"):
                compile(path.read_text(), str(path), "exec")
            from reference import expected  # Ensure the protected oracle imports.

            report["cases"] = [public_case(case, manifest=True) | {"status": "PASS"}
                               for case in data["cases"]]
            report["metadata"]["candidate_state"] = "implemented"
        else:
            import torch
            from cases import make_case

            call = selected_callable(role)
            limit = data["numerical_contract"]["max_sequence_rms_ratio"]
            for case in data["cases"]:
                if action != "compile" and action not in case["checks"]:
                    continue
                row = public_case(case)
                row["status"] = "PASS"
                if action == "compile":
                    payload = make_case(case)
                    call(**payload["kwargs"])
                    torch.cuda.synchronize()
                elif action == "correctness":
                    row["metrics"] = {"draws": [
                        check_case(call, case, seed=seed, limit=limit)
                        for seed in (42, secrets.randbits(31))
                    ]}
                else:
                    elapsed, metadata, metrics = measure_case(call, case, data["timing"], limit)
                    row.update(execution_time_ms=elapsed, benchmark_method="cuda_graph",
                               metadata=metadata, metrics=metrics)
                report["cases"].append(row)
        report["status"] = "PASS"
    except Exception as exc:
        report["reason"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc(file=sys.stderr)
    print("ARENA_EVAL_RESULT=" + json.dumps(report, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
