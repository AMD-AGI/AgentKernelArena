"""Portable expected-case and replay checks for isolated GPU tasks.

Copy this file unchanged into a task's ``ut/evaluation_contract.py``. It uses
only the standard library; the frozen task supplies its oracle, GPU timer and
input/output reset callbacks. Metadata checks do not replace those operations.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strict_json(text):
    """Reject duplicate keys and nonfinite numbers instead of normalizing them."""
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError("nonfinite JSON value: " + value)

    return json.loads(text, object_pairs_hook=pairs, parse_constant=nonfinite)


def validate_manifest(manifest):
    require(isinstance(manifest, dict) and type(manifest.get("schema_version")) is int
            and manifest["schema_version"] == 1,
            "expected-case manifest requires schema_version 1")
    require(re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", str(manifest.get("runtime_image", ""))),
            "expected cases require a digest-pinned runtime image")
    cases = manifest.get("cases")
    require(isinstance(cases, list) and bool(cases), "expected cases must be a nonempty list")
    seen = set()
    for case in cases:
        require(isinstance(case, dict), "each expected case must be an object")
        case_id = case.get("case_id")
        require(isinstance(case_id, str) and case_id and case_id not in seen,
                "case IDs must be nonempty and unique")
        seen.add(case_id)
        for key in ("occurrences", "calls_per_sample"):
            require(type(case.get(key)) is int and case[key] > 0, f"{case_id}: invalid {key}")
        tensors = case.get("tensors")
        require(isinstance(tensors, dict) and bool(tensors), f"{case_id}: tensor ABI is missing")
        for name, tensor in tensors.items():
            require(isinstance(name, str) and name and isinstance(tensor, dict), "invalid tensor ABI entry")
            require(tensor.get("role") in ("input", "output", "inout"), f"{case_id}/{name}: missing tensor role")
            shape, strides = tensor.get("shape"), tensor.get("strides")
            require(isinstance(shape, list) and all(type(v) is int and v >= 0 for v in shape), "invalid tensor shape")
            require(isinstance(strides, list) and len(strides) == len(shape)
                    and all(type(v) is int and v >= 0 for v in strides), "invalid tensor strides")
            require(type(tensor.get("storage_offset")) is int and tensor["storage_offset"] >= 0,
                    "explicit nonnegative storage_offset is required")
            require(isinstance(tensor.get("dtype"), str) and bool(tensor["dtype"]), "explicit tensor dtype is required")
            require(tensor.get("device_type") == "cuda", "device timing requires CUDA/HIP tensors")
        require(any(t["role"] in ("output", "inout") for t in tensors.values()), "case has no observable output")
        require(isinstance(case.get("scalars"), dict), f"{case_id}: explicit scalar arguments are required")
    policy = manifest.get("measurement", {})
    require(isinstance(policy, dict) and policy.get("method") == "cuda_graph", "unsupported timing method")
    require(type(policy.get("warmup_iterations")) is int and policy["warmup_iterations"] > 0,
            "warmup_iterations must be positive")
    require(type(policy.get("benchmark_iterations")) is int and policy["benchmark_iterations"] > 0,
            "benchmark_iterations must be positive")
    seeds = policy.get("correctness_seeds")
    require(isinstance(seeds, list) and len(seeds) >= 2
            and all(type(seed) is int for seed in seeds) and len(set(seeds)) == len(seeds),
            "at least two unique correctness seeds are required")
    require(policy.get("refresh_inputs") == "each_replay"
            and policy.get("initialize_outputs") == "each_replay"
            and policy.get("validate_outputs") == "each_replay", "every replay must reset inputs and validate outputs")
    controls = policy.get("negative_controls")
    require(isinstance(controls, list) and all(isinstance(name, str) and name for name in controls)
            and len(controls) == len(set(controls))
            and {"no_op", "wrong_output"}.issubset(controls), "no-op and wrong-output controls are mandatory")
    canonical(manifest)  # Also reject nonfinite scalars/metadata.
    return manifest


def describe_tensor(tensor, role):
    """Observe a tensor without importing a framework on the host."""
    return {"role": role, "shape": list(tensor.shape), "strides": list(tensor.stride()),
            "storage_offset": tensor.storage_offset(), "dtype": str(tensor.dtype).removeprefix("torch."),
            "device_type": tensor.device.type}


def observe_case(expected, tensors, scalars):
    require(set(tensors) == set(expected["tensors"]), "runtime tensor names differ from the expected ABI")
    observed = strict_json(canonical(expected))
    observed["tensors"] = {
        name: describe_tensor(tensor, expected["tensors"][name]["role"])
        for name, tensor in tensors.items()
    }
    observed["scalars"] = scalars
    require(canonical(observed) == canonical(expected), f"runtime ABI differs for {expected['case_id']}")
    return observed


def checked_replays(case, policy, *, reset_inputs, initialize_outputs, replay, verify, measure, observe, seed):
    """Run resets and oracle checks outside each single graph replay timing.

    ``reset_inputs(seed)`` must copy fresh inputs and return CPU-owned truth:
    either CPU reference outputs or an input snapshot for a deferred reference.
    It must not leave the current candidate's golden outputs on the GPU.
    ``verify(truth)`` must snapshot candidate outputs and immutable inputs to
    CPU before computing any GPU reference, then compare those observations
    against the CPU truth/reference and raise on failure. ``measure(call)`` must
    invoke ``call`` exactly once and return synchronized device milliseconds.
    The task's negative controls must demonstrate that its callbacks reject a
    no-op replay and corrupted outputs. All callbacks are protected harness.
    """
    samples = []
    warmup, iterations = policy["warmup_iterations"], policy["benchmark_iterations"]
    for iteration in range(warmup + iterations):
        reference = reset_inputs(seed + iteration)
        initialize_outputs()
        require(canonical(observe()) == canonical(case), "runtime ABI changed during replay preparation")
        launches = 0

        def launch():
            nonlocal launches
            launches += 1
            replay()

        if iteration < warmup:
            launch()
        else:
            elapsed = measure(launch)
            require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed > 0,
                    "device samples must be finite and positive")
            samples.append(elapsed)
        require(launches == 1, "timer must execute exactly one graph replay")
        verified = verify(reference)
        require(verified is None or verified is True, "oracle rejected graph replay")
    return {"case": case, "correct": True, "samples_ms": samples,
            "warmup_iterations": warmup, "fresh_input_resets": iterations,
            "output_initializations": iterations, "oracle_checks": iterations,
            "benchmark_method": policy["method"]}


def validate_report(report, manifest, request):
    """Check fresh request identity, exact ABI/multiplicity and full coverage."""
    validate_manifest(manifest)
    require(isinstance(report, dict) and type(report.get("schema_version")) is int
            and report["schema_version"] == 1
            and report.get("status") == "ok", "unsuccessful evaluation report")
    require(report.get("request") == request and canonical(report.get("request")) == canonical(request),
            "stale or foreign evaluation request")
    require(request.get("manifest_sha256") == fingerprint(manifest), "request does not match expected cases")
    phase = request.get("phase")
    require(phase in ("compile", "correctness", "performance"), "unknown evaluation phase")
    rows = report.get("cases")
    require(isinstance(rows, list), "report cases must be a list")
    if phase == "compile":
        require(rows == [] and report.get("compiled") is True, "compilation did not complete")
        return []
    expected = {case["case_id"]: case for case in manifest["cases"]}
    require(all(isinstance(row, dict) and isinstance(row.get("case"), dict) for row in rows), "malformed case result")
    ids = [row["case"].get("case_id") for row in rows]
    require(all(isinstance(case_id, str) for case_id in ids), "malformed case ID")
    require(Counter(ids) == Counter(expected.keys()), "missing, duplicate or unexpected case IDs")
    ordered = {row["case"]["case_id"]: row for row in rows}
    policy = manifest["measurement"]
    measurements = []
    for case_id, case in expected.items():
        row = ordered[case_id]
        require(canonical(row["case"]) == canonical(case), f"workload ABI or multiplicity changed: {case_id}")
        require(row.get("correct") is True, f"oracle did not pass: {case_id}")
        if phase == "correctness":
            require(canonical(row.get("seeds")) == canonical(policy["correctness_seeds"]), "correctness seed coverage changed")
            controls = row.get("negative_controls", {})
            require(isinstance(controls, dict) and set(controls) == set(policy["negative_controls"])
                    and all(value is True for value in controls.values()), "negative controls did not reject invalid work")
            continue
        count = policy["benchmark_iterations"]
        for key in ("fresh_input_resets", "output_initializations", "oracle_checks"):
            require(type(row.get(key)) is int and row[key] == count, "incomplete replay validation: " + key)
        require(row.get("benchmark_method") == policy["method"]
                and type(row.get("warmup_iterations")) is int
                and row["warmup_iterations"] == policy["warmup_iterations"], "timing policy changed")
        samples = row.get("samples_ms")
        require(isinstance(samples, list) and len(samples) == count
                and all(type(value) in (float, int) and math.isfinite(value) and value > 0 for value in samples),
                "incomplete or invalid device samples")
        # Ignore candidate-supplied means or speedup fields; derive timings here.
        measurements.append({"test_case_id": case_id, "execution_time_ms": math.fsum(samples) / count,
                             "case_sha256": fingerprint(case), "params": case,
                             "metadata": {"benchmark_method": policy["method"]}})
    if "test_cases" in report:
        require(canonical(report["test_cases"]) == canonical(measurements), "scoreable cases differ from validated device samples")
    return measurements


def finalize_report(report, manifest, request):
    """Add Arena-compatible case timings derived from validated raw samples."""
    measured = validate_report(report, manifest, request)
    completed = strict_json(canonical(report))
    if request["phase"] == "performance":
        completed["test_cases"] = measured
    return completed
