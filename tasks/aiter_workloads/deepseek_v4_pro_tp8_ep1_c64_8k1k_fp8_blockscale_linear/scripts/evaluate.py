#!/usr/bin/env python3
"""Seven task-owned Arena-v2 actions for the bound FP8 whole-call workload.

The two enrolled cases use four fixed authentic carriers (original and replay
per case). Every enrolled call receives x_scale; activation quantization runs
in the upstream producer, outside the call and its timing.
"""
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
ENTRY = "fp8_blockscale_linear"


def public_case(case, *, manifest=False):
    row = {key: deepcopy(case[key]) for key in IDENTITY if key in case}
    if manifest:
        row["checks"] = deepcopy(case["checks"])
    return row


def read_binding():
    """Check the whole manifest structure, then fail closed until it is bound."""
    data = json.loads((ROOT / "workloads.json").read_text())
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
    import yaml

    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    aggregation = config["evaluation"].get("aggregation", {})
    performance_ids = {case["test_case_id"] for case in cases if "performance" in case["checks"]}
    if not performance_ids or aggregation.get("method") != "weighted_total_time" or set(aggregation.get("case_weights", {})) != performance_ids:
        raise ValueError("The fixed-workload task needs weights for exactly its performance cases")
    source = aggregation.get("source")
    if not isinstance(source, dict):
        raise ValueError("Frequency provenance is required")
    provenance = read_protected_json(source.get("path"), source.get("sha256"), "Frequency provenance")
    from reference import validate_contract

    validate_contract(data["numerical_contract"])
    timing = data["timing"]
    for key in ("warmup", "repetition", "max_graph_repeats"):
        if type(timing.get(key)) is not int or timing[key] <= 0:
            raise ValueError(f"A positive frozen timing policy is required: {key}")
    if type(timing.get("target_ms")) not in (int, float) or not math.isfinite(timing["target_ms"]) or timing["target_ms"] <= 0:
        raise ValueError("A positive finite target_ms is required")
    json.dumps(data, allow_nan=False)
    if data.get("binding_status") != "frozen":
        raise ValueError(
            "Task structure is valid, but its input and runtime binding is not frozen; "
            "captured inputs, measured frequencies and a bound scoring runtime are required"
        )
    check_enrollment(provenance, aggregation["case_weights"], cases)
    return data


def read_protected_json(reference, digest, label):
    if not isinstance(reference, str) or not reference or Path(reference).is_absolute():
        raise ValueError(f"{label} must use a task-relative path")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"{label} requires a lowercase SHA256")
    path = (ROOT / reference).resolve(strict=True)
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
        raise ValueError(f"{label} escapes the task or is not a file")
    body = path.read_bytes()
    if hashlib.sha256(body).hexdigest() != digest:
        raise ValueError(f"{label} content differs from its protected binding")
    record = json.loads(body)
    if not isinstance(record, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return record


def check_enrollment(provenance, weights, cases):
    """A frozen status alone enrolls nothing: every scored case, count and input
    carrier must join a row the protected capture record exported. The capture's
    scientific review is separate."""
    if provenance.get("status") != "measured" or provenance.get("enrolled_for_scoring") is not True:
        raise ValueError("Frequency provenance is not an enrolled measured capture")
    capture = provenance.get("capture")
    if not isinstance(capture, dict):
        raise ValueError("Measured provenance must bind its protected capture record")
    receipt = read_protected_json(capture.get("reference"), capture.get("sha256"), "Capture record")
    rows = provenance.get("case_counts")
    if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Measured case_counts must contain explicit count rows")
    counts = {}
    for row in rows:
        key, count = row.get("test_case_id"), row.get("count")
        if not isinstance(key, str) or not key or key in counts or type(count) is not int or count <= 0:
            raise ValueError("Measured case_counts require unique IDs and positive integer counts")
        counts[key] = count
    if counts != weights or any(type(value) is not int for value in weights.values()):
        raise ValueError("Measured case_counts must equal the configured weights exactly")
    exported = receipt.get("exported_cases")
    if not isinstance(exported, list) or any(not isinstance(row, dict) for row in exported):
        raise ValueError("Capture record must enumerate exported_cases")
    by_id = {}
    for row in exported:
        key = row.get("test_case_id")
        if not isinstance(key, str) or not key or key in by_id:
            raise ValueError("Capture exported_cases require unique case IDs")
        by_id[key] = row
    if set(by_id) != set(weights):
        raise ValueError("Capture exported_cases must cover exactly the scored cases")
    for case in cases:
        if case["test_case_id"] not in weights:
            continue
        row = by_id[case["test_case_id"]]
        if type(row.get("count")) is not int or row["count"] != weights[case["test_case_id"]] \
                or public_case(row) != public_case(case) or row.get("inputs") != case.get("inputs"):
            raise ValueError("Scored case identity, count or input binding differs from the capture record")
        carrier = case.get("inputs")
        path, digest = (carrier.get("path"), carrier.get("sha256")) if isinstance(carrier, dict) else (None, None)
        if not isinstance(path, str) or not path or Path(path).is_absolute() or not isinstance(digest, str) \
                or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Every scored case needs a protected captured input carrier; synthetic cases are unscored")
        replay = carrier.get("current_value_replay")
        if not isinstance(replay, dict) or set(replay) != {"path", "sha256"}:
            raise ValueError("Every scored case needs a protected current_value_replay carrier")
        path, digest = replay["path"], replay["sha256"]
        if not isinstance(path, str) or not path or Path(path).is_absolute() or not isinstance(digest, str) \
                or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Current-value replay requires a task-relative path and lowercase SHA256")


def candidate_callable():
    path = ROOT / "source/kernel.py"
    ast.parse(path.read_text(), filename="source/kernel.py")
    module = importlib.import_module("source.kernel")
    function = getattr(module, ENTRY, None)
    if not callable(function):
        raise ValueError(f"source/kernel.py must define callable {ENTRY}")
    return function


def producer_quant(x):
    """Retained BF16 helper; enrolled FP8 cases already supply x and x_scale.

The pinned per-1x128 branch is described in
sglang python/sglang/srt/layers/quantization/fp8_utils.py:113-139 and 1195-1271
at 94602c9c2b7cbdb8efd5c52802dac6a1c180089e: M >= 2 emits the transposed scale
and views it column-major; M == 1 quantizes row-major and materializes.
"""
    import torch
    import aiter

    emit = x.shape[0] >= 2
    q, scale = aiter.get_hip_quant(aiter.QuantType.per_1x128)(
        x, quant_dtype=aiter.dtypes.fp8, transpose_scale=emit
    )
    scale = torch.as_strided(scale, scale.shape, (1, scale.shape[0])) if emit else scale.t().contiguous().t()
    return q, scale


def baseline_linear(x, weight, weight_scale, x_scale=None):
    """Pinned AITER gfx95 bpreshuffle GEMM for the enrolled FP8 inputs.

Every enrolled call supplies x_scale and skips the retained BF16 helper.
The materialization SGLang applies to an already column-major producer scale is
a no-op view and omitted.
"""
    import torch
    import aiter

    if x_scale is None:
        x, x_scale = producer_quant(x)
    return aiter.gemm_a8w8_blockscale_bpreshuffle(x, weight, x_scale, weight_scale, dtype=torch.bfloat16)


def check_quant_agreement(payload, contract):
    """Reference fidelity before any native comparison: on a bf16-input case the
    reference quantizer must reproduce the scales and fp8 bytes of the producer
    branch the baseline actually takes for this M."""
    import torch
    from reference import quantize_1x128

    x = payload["args"][0]
    if "x_scale" in payload["kwargs"]:
        return "not_applicable_fp8_input"
    expected_q, expected_scale = quantize_1x128(x, contract)
    q, scale = producer_quant(x)
    if scale.shape != expected_scale.shape:
        raise ValueError("Pinned producer scale geometry differs from the reference quantizer")
    scale_mismatch = (scale.float() != expected_scale).sum().item()
    byte_mismatch = (q.view(torch.uint8) != expected_q.view(torch.uint8)).sum().item()
    if scale_mismatch or byte_mismatch:
        raise ValueError(f"Reference quantizer differs from the pinned producer: "
                         f"{scale_mismatch} scales, {byte_mismatch} fp8 bytes")
    return "PASS"


def run_action(role, action, data):
    from runtime import verify_runtime
    from cases import (load_case, snapshot_inputs, check_inputs_unchanged,
                       load_current_value_replay, copy_current_value_replay)
    from reference import reference, check_output, check_reference_controls, check_separated

    contract = data["numerical_contract"]
    metadata = verify_runtime()
    if role == "task":
        tree = ast.parse((ROOT / "source/kernel.py").read_text())
        entry = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == ENTRY), None)
        if entry is None or not any(isinstance(n, ast.Return) and isinstance(n.value, ast.Call) for n in ast.walk(entry)):
            raise ValueError(f"Starting {ENTRY} delegation is absent or unimplemented")
        metadata["candidate_state"] = "implemented"
        # Check the reference's actual input/load/math path, not just its name.
        for case in data["cases"]:
            payload = load_case(ROOT, case, contract)
            original_inputs = snapshot_inputs(payload)
            controls = check_reference_controls(payload, contract)
            controls["producer_quant_agreement"] = check_quant_agreement(payload, contract)
            if payload["captured"]:
                replay_inputs = load_current_value_replay(ROOT, case, payload)
                expected = reference(payload, contract)
            check_inputs_unchanged(original_inputs)
            del original_inputs
            if payload["captured"]:
                copy_current_value_replay(payload, replay_inputs)
                del replay_inputs
                current_inputs = snapshot_inputs(payload)
                controls["current_value_replay"] = check_separated(expected, reference(payload, contract), contract)
                check_inputs_unchanged(current_inputs)
                del current_inputs
            metadata.setdefault("reference_controls", {})[case["test_case_id"]] = controls
        return [public_case(c, manifest=True) | {"status": "PASS"} for c in data["cases"]], metadata
    entry_fn = baseline_linear if role == "baseline" else candidate_callable()
    rows = []
    for case in data["cases"]:
        if action != "compile" and action not in case["checks"]:
            continue
        payload = load_case(ROOT, case, contract)
        original_inputs = snapshot_inputs(payload)
        launch = lambda: entry_fn(*payload["args"], **payload["kwargs"])
        if action == "compile":
            # Executes the actual entrypoint so its JIT/native dependencies must
            # build successfully; Python syntax alone is not GPU compilation.
            launch()
            import torch
            torch.cuda.synchronize()
            check_inputs_unchanged(original_inputs)
            continue
        replay_inputs = load_current_value_replay(ROOT, case, payload) if payload["captured"] else None
        expected = reference(payload, contract)
        row = public_case(case)
        if action == "correctness":
            row["metrics"] = check_output(launch(), expected, contract)
            replay = launch
        else:
            if not payload["captured"]:
                raise ValueError("Synthetic inputs are never timed for the workload score")
            from _aka_benchmark import TimedRun, benchmark_cuda_graph_or_events_samples

            timed = TimedRun()
            timed.after_sample = lambda output: check_output(output, expected, contract)
            policy = data["timing"]
            declared_method = case["params"]["benchmark_method"]
            samples, timing = benchmark_cuda_graph_or_events_samples(
                launch, warmup=policy["warmup"], repetition=policy["repetition"],
                target_ms=policy["target_ms"], max_graph_repeats=policy["max_graph_repeats"],
                use_cuda_graph=declared_method == "cuda_graph", timed_run=timed,
            )
            if timing.get("benchmark_method") != declared_method:
                raise ValueError("Measured timing method differs from the protected case declaration")
            check_output(timed.outputs, expected, contract)
            check_output(timed.rerun(), expected, contract)
            elapsed = statistics.median(samples)
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError("Invalid device timing")
            row.update(execution_time_ms=elapsed, benchmark_method=timing["benchmark_method"], metadata=timing)
            replay = timed.rerun
        check_inputs_unchanged(original_inputs)
        del original_inputs
        if payload["captured"]:
            # Keep the public objects and prepared weights held by the timed
            # unit. Update dynamic storage only, after all scored samples.
            copy_current_value_replay(payload, replay_inputs)
            del replay_inputs
            current_inputs = snapshot_inputs(payload)
            current_expected = reference(payload, contract)
            row.setdefault("metadata", {})["current_value_replay"] = check_output(replay(), current_expected, contract)
            check_inputs_unchanged(current_inputs)
            del current_inputs
        row["status"] = "PASS"
        rows.append(row)
    if role == "baseline":
        from runtime import verify_baseline_native
        metadata["baseline_native_identity"] = verify_baseline_native()
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
