"""Expected ABI, multiplicity and replay checks reject concrete score shortcuts."""

import copy
from types import SimpleNamespace

import pytest

from src import task_contract as contract


@pytest.fixture
def manifest():
    tensor = {"role": "input", "shape": [8, 16], "strides": [16, 1],
              "dtype": "bfloat16", "storage_offset": 0, "device_type": "cuda"}
    return {"schema_version": 1, "runtime_image": "registry/image@sha256:" + "a" * 64,
            "cases": [{"case_id": "matrix", "occurrences": 17, "calls_per_sample": 1,
                       "tensors": {"x": tensor, "out": {**tensor, "role": "output"}},
                       "scalars": {"alpha": 1.0, "transpose": False, "padding": None}}],
            "measurement": {"method": "cuda_graph", "warmup_iterations": 2, "benchmark_iterations": 4,
                            "correctness_seeds": [0, 1], "refresh_inputs": "each_replay",
                            "initialize_outputs": "each_replay", "validate_outputs": "each_replay",
                            "negative_controls": ["no_op", "wrong_output"]}}


def performance(manifest):
    request = {"request_id": "fresh", "phase": "performance", "manifest_sha256": contract.fingerprint(manifest)}
    row = {"case": copy.deepcopy(manifest["cases"][0]), "correct": True,
           "samples_ms": [2, 2, 2, 2], "fresh_input_resets": 4, "output_initializations": 4,
           "oracle_checks": 4, "warmup_iterations": 2, "benchmark_method": "cuda_graph"}
    return request, {"schema_version": 1, "status": "ok", "request": copy.deepcopy(request), "cases": [row]}


@pytest.mark.parametrize("attack", ["omit", "duplicate", "occurrences", "calls_per_sample", "shape", "strides", "dtype",
                                     "offset", "scalar", "scalar_type", "reset", "oracle", "samples", "nan", "stale", "correct"])
def test_invalid_work_or_report_cannot_be_scored(manifest, attack):
    request, report = performance(manifest)
    row = report["cases"][0]
    case = row["case"]
    if attack == "omit":
        report["cases"] = []
    elif attack == "duplicate":
        report["cases"].append(copy.deepcopy(row))
    elif attack in ("occurrences", "calls_per_sample"):
        case[attack] += 1
    elif attack in ("shape", "strides"):
        case["tensors"]["x"][attack] = [1, 1]
    elif attack == "dtype":
        case["tensors"]["x"]["dtype"] = "float32"
    elif attack == "offset":
        case["tensors"]["x"]["storage_offset"] = 1
    elif attack == "scalar":
        case["scalars"]["alpha"] = 0.0
    elif attack == "scalar_type":
        case["scalars"]["alpha"] = True
    elif attack == "reset":
        row["fresh_input_resets"] = 0
    elif attack == "oracle":
        row["oracle_checks"] = 1
    elif attack == "samples":
        row["samples_ms"].pop()
    elif attack == "nan":
        row["samples_ms"][0] = float("nan")
    elif attack == "stale":
        report["request"]["request_id"] = "old"
    else:
        row["correct"] = False
    with pytest.raises(ValueError):
        contract.validate_report(report, manifest, request)


def test_device_samples_are_authoritative_over_claimed_means(manifest):
    request, report = performance(manifest)
    report["cases"][0].update(mean_ms=0.000001, speedup=999999)
    measured = contract.validate_report(report, manifest, request)
    assert measured[0]["execution_time_ms"] == 2
    assert measured[0]["params"]["occurrences"] == 17


def test_finalized_report_exposes_only_validated_arena_timings(manifest, tmp_path):
    from src.testcases import parse_test_cases_from_json

    request, report = performance(manifest)
    completed = contract.finalize_report(report, manifest, request)
    path = tmp_path / "performance_report.json"
    path.write_text(contract.canonical(completed))
    cases = parse_test_cases_from_json(path)
    assert len(cases) == 1 and cases[0].execution_time_ms == 2
    completed["test_cases"][0]["execution_time_ms"] = 0.000001
    with pytest.raises(ValueError, match="scoreable cases differ"):
        contract.validate_report(completed, manifest, request)


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'])
def test_malformed_json_is_not_normalized(text):
    with pytest.raises(ValueError):
        contract.strict_json(text)


def test_observed_tensor_abi_comes_from_runtime_objects(manifest):
    tensor = SimpleNamespace(shape=(8, 16), stride=lambda: (16, 1), storage_offset=lambda: 0,
                             dtype="torch.bfloat16", device=SimpleNamespace(type="cuda"))
    case = manifest["cases"][0]
    assert contract.observe_case(case, {"x": tensor, "out": tensor}, case["scalars"]) == case
    tensor.stride = lambda: (1, 8)
    with pytest.raises(ValueError, match="runtime ABI differs"):
        contract.observe_case(case, {"x": tensor, "out": tensor}, case["scalars"])


@pytest.mark.parametrize("behavior", ["correct", "noop", "stale", "timer_skips", "timer_twice"])
def test_replay_protocol_detects_skipped_or_stale_work(manifest, behavior):
    state = {"x": 0, "out": None, "launches": 0}
    resets = []

    def reset(seed):
        state["x"] = seed
        resets.append(seed)
        return seed * 2

    def initialize():
        state["out"] = None

    def replay():
        state["launches"] += 1
        if behavior == "noop":
            return
        state["out"] = 10 if behavior == "stale" else state["x"] * 2

    def verify(reference):
        if state["out"] != reference:
            raise ValueError("wrong replay output")

    def measure(launch):
        for _ in range(0 if behavior == "timer_skips" else 2 if behavior == "timer_twice" else 1):
            launch()
        return 0.5

    def run():
        return contract.checked_replays(manifest["cases"][0], manifest["measurement"], seed=5,
                                        reset_inputs=reset, initialize_outputs=initialize, replay=replay,
                                        verify=verify, measure=measure, observe=lambda: manifest["cases"][0])

    if behavior == "correct":
        result = run()
        assert result["samples_ms"] == [0.5] * 4
        assert resets == [5, 6, 7, 8, 9, 10]
        assert state["launches"] == 6
    else:
        with pytest.raises(ValueError):
            run()


def test_correctness_requires_every_control_and_seed(manifest):
    request = {"phase": "correctness", "manifest_sha256": contract.fingerprint(manifest)}
    report = {"schema_version": 1, "status": "ok", "request": request,
              "cases": [{"case": manifest["cases"][0], "correct": True, "seeds": [0, 1],
                         "negative_controls": {"no_op": True, "wrong_output": True}}]}
    assert contract.validate_report(report, manifest, request) == []
    report["cases"][0]["negative_controls"]["no_op"] = False
    with pytest.raises(ValueError, match="negative controls"):
        contract.validate_report(report, manifest, request)


@pytest.mark.parametrize("wrong_output", [False, True])
def test_replay_token_can_be_cpu_input_truth_for_a_deferred_reference(manifest, wrong_output):
    events = []
    state = {}

    def reset(seed):
        events.append("reset")
        state["input"] = seed
        return {"input_cpu": seed}

    def initialize():
        state["output"] = None

    def replay():
        events.append("candidate")
        state["output"] = -1 if wrong_output else state["input"] * 2

    def reference(truth):
        events.append("reference")
        value = truth["input_cpu"] * 2
        state["output"] = value  # A late reference cannot repair the saved observation.
        return value

    def verify(truth):
        events.append("snapshot")
        observed = dict(state)
        expected = reference(truth)
        if observed["input"] != truth["input_cpu"] or observed["output"] != expected:
            raise ValueError("saved candidate observation differs")

    def measure(call):
        call()
        return 0.5

    def run():
        return contract.checked_replays(manifest["cases"][0], manifest["measurement"], seed=3,
                                        reset_inputs=reset, initialize_outputs=initialize, replay=replay,
                                        verify=verify, measure=measure, observe=lambda: manifest["cases"][0])

    if wrong_output:
        with pytest.raises(ValueError, match="saved candidate observation"):
            run()
        assert events == ["reset", "candidate", "snapshot", "reference"]
    else:
        result = run()
        assert result["samples_ms"] == [0.5] * 4
        assert events == ["reset", "candidate", "snapshot", "reference"] * 6
