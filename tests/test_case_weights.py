"""Optional per-case weights of schema-v3 performance cases.

A task either weights every performance case or none. A weight is how often
the case runs in a recorded workload; weighted tasks declare where the weights
come from and score the serving-weighted speedup, sum(w*t_baseline) /
sum(w*t_candidate); unweighted tasks keep the unweighted geometric mean and
their digests.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace

import pytest
import yaml

from src.evaluator import evaluate_kernel, write_task_result
from src.measurement import geometric_mean, paired_throughput
from src.task_contract import (
    MIN_WEIGHT_COVERAGE, TARGET_LANGUAGES, load_task_contract, package_files, validate_task_contract,
)
from src.task_protocol import ActionResult
from src.task_spec import TaskConfigError, TaskSpec, load_task_spec
from src.testcases import (TestCaseResult as CaseResult, calculate_average_speedup, load_performance_results,
                           serving_weighted_speedup)
from tests.test_held_out import PILOT, PILOT_ID, WEIGHTS, weighted_pilot

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_KEYS = {"schema_version", "task_file", "target_language", "baseline_language", "operator"}
CASE_KEYS = {"id", "axes", "scalars", "checks", "seeds"}


def _raw():
    raw = yaml.safe_load((PILOT / "task.yaml").read_text())
    workloads = json.loads((PILOT / "workloads.json").read_text())
    return raw, workloads


def _weighted(weights=WEIGHTS, **provenance):
    raw, workloads = _raw()
    for case in workloads["cases"]:
        if case["id"] in weights:
            case["weight"] = weights[case["id"]]
    workloads["weight_provenance"] = {"kind": "agentx", "source": "corpus@rev", "capture": "capture-1",
                                      "coverage": 0.85, **provenance}
    return raw, workloads


def _validate(raw, workloads):
    return validate_task_contract(raw, workloads, task_id=PILOT_ID)


# --- Schema -------------------------------------------------------------------------------------

def test_weighted_workloads_load_and_resolve_into_the_execution_spec():
    contract = _validate(*_weighted())
    assert contract.case_weights == WEIGHTS
    config = contract.execution_config("triton", package_files(PILOT))
    assert config["contract"]["case_weights"] == WEIGHTS
    spec = TaskSpec.from_mapping(config, task_id=PILOT_ID)
    assert spec.case_weights == WEIGHTS
    assert TaskSpec.from_mapping(spec.to_mapping(), task_id=PILOT_ID).case_weights == WEIGHTS
    # The evaluator's contract.json is unchanged: weights are scoring, not evaluation, inputs.
    assert all(set(case) == CASE_KEYS for case in contract.resolved_contract("triton")["cases"])


def test_weights_need_not_sum_to_one():
    contract = _validate(*_weighted({key: value * 40 for key, value in WEIGHTS.items()}))
    assert contract.case_weights["t2048"] == 16.0


def test_every_performance_case_or_none_has_a_weight():
    partial = dict(WEIGHTS)
    del partial["t204800"]
    with pytest.raises(TaskConfigError, match="every performance case"):
        _validate(*_weighted(partial))


@pytest.mark.parametrize("weight", [0, -1.0, "0.4", True, None, float("nan"), float("inf"), [1]])
def test_weights_are_positive_finite_numbers(weight):
    with pytest.raises(TaskConfigError, match="weight"):
        _validate(*_weighted({**WEIGHTS, "t2048": weight}))


def test_correctness_only_cases_have_no_weight():
    with pytest.raises(TaskConfigError, match="only performance cases"):
        _validate(*_weighted({**WEIGHTS, "t1": 0.1}))


def test_weights_need_their_provenance():
    raw, workloads = _weighted()
    del workloads["weight_provenance"]
    with pytest.raises(TaskConfigError, match="weight_provenance"):
        _validate(raw, workloads)
    raw, workloads = _raw()
    workloads["weight_provenance"] = {"kind": "agentx", "source": "s", "capture": "c", "coverage": 0.9}
    with pytest.raises(TaskConfigError, match="only for weighted"):
        _validate(raw, workloads)


@pytest.mark.parametrize("change, match", [
    ({"coverage": None}, "finite number"),
    ({"coverage": 0.79}, "at least 80%"),
    ({"coverage": 1.01}, "between"),
    ({"source": ""}, "nonempty"),
    ({"model": 3}, "nonempty"),
    ({"revision": "x"}, "unknown fields"),
])
def test_weight_provenance_is_validated(change, match):
    raw, workloads = _weighted(**change)
    with pytest.raises(TaskConfigError, match=match):
        _validate(raw, workloads)
    raw, workloads = _weighted()
    del workloads["weight_provenance"]["capture"]
    with pytest.raises(TaskConfigError, match="missing fields"):
        _validate(raw, workloads)


def test_weighted_cases_may_record_what_they_observed():
    raw, workloads = _weighted(weight_unit="calls per second of the steady state, TP rank 0",
                               deployment={"image": "img", "page_size": 16, "kv_cache_dtype": "fp8_e4m3",
                                           "speculative_algorithm": None, "mem_fraction_static": 0.68},
                               corpus={"context_cap_tokens": 262144, "dropped_share": 0.12})
    for case in workloads["cases"]:
        if case["id"] == "t2048":
            case["observed"] = {"phases": ["decode"], "time_share": 0.5, "members": [{"n_tokens": 2048}]}
    assert _validate(raw, workloads).case_weights == WEIGHTS


@pytest.mark.parametrize("observed, match", [
    ({"time_share": 0}, "time_share"),
    ({"time_share": 1.5}, "time_share"),
    ({"phases": []}, "phases"),
    ({"members": [{"m": 3}]}, "members"),
    ({"members": [{"n_tokens": -1}]}, "members"),
    ({"calls": 3}, "unknown fields"),
])
def test_observed_is_validated(observed, match):
    raw, workloads = _weighted()
    next(c for c in workloads["cases"] if c["id"] == "t2048")["observed"] = observed
    with pytest.raises(TaskConfigError, match=match):
        _validate(raw, workloads)


def test_only_weighted_performance_cases_carry_observed():
    raw, workloads = _raw()
    next(c for c in workloads["cases"] if "performance" in c["checks"])["observed"] = {"time_share": 0.5}
    with pytest.raises(TaskConfigError, match="observed"):
        _validate(raw, workloads)


@pytest.mark.parametrize("change", [{"deployment": {}}, {"deployment": {"page_size": [16]}},
                                    {"corpus": "all"}, {"deployment": {"x": float("inf")}}])
def test_provenance_records_are_flat_scalar_mappings(change):
    raw, workloads = _weighted(**change)
    with pytest.raises(TaskConfigError):
        _validate(raw, workloads)


def test_coverage_replaces_the_size_span_rule_for_weighted_tasks():
    assert MIN_WEIGHT_COVERAGE == 0.8
    narrow = {"t2048": 0.5, "t8192": 0.3, "t32768": 0.2, "t1000": 0.1}
    raw, workloads = _weighted(narrow, coverage=MIN_WEIGHT_COVERAGE)
    for case in workloads["cases"]:
        case["checks"] = ["correctness", "performance"] if case["id"] in narrow else ["correctness"]
    assert _validate(raw, workloads).case_weights == narrow          # 1000..32768 spans under 100x
    unweighted = copy.deepcopy(workloads)
    del unweighted["weight_provenance"]
    for case in unweighted["cases"]:
        case.pop("weight", None)
    with pytest.raises(TaskConfigError, match="span"):
        _validate(raw, unweighted)


def test_weighted_tasks_keep_the_performance_case_count():
    raw, workloads = _weighted({"t2048": 0.5, "t8192": 0.3, "t32768": 0.2})
    for case in workloads["cases"]:
        if case["id"] in ("t131072", "t204800"):
            case["checks"] = ["correctness"]
    with pytest.raises(TaskConfigError, match="performance cases"):
        _validate(raw, workloads)


@pytest.mark.parametrize("weights", [{}, {"t1": 0}, {"t1": True}, {"t1": float("inf")}, [1]])
def test_task_spec_rejects_invalid_case_weights(weights):
    mapping = load_task_spec(PILOT / "task.yaml", task_id=PILOT_ID, target_language="triton").to_mapping()
    mapping["contract"]["case_weights"] = weights
    with pytest.raises(TaskConfigError, match="case_weights"):
        TaskSpec.from_mapping(mapping, task_id=PILOT_ID)


# --- Digest compatibility -----------------------------------------------------------------------

def test_unweighted_tasks_serialize_as_before():
    """An unweighted task's execution spec and contract.json keep their keys; only tasks whose
    workloads declare a weight provenance carry weights."""
    for task_file in sorted((ROOT / "tasks").rglob("task.yaml")):
        task_id = task_file.parent.relative_to(ROOT / "tasks").as_posix()
        contract = load_task_contract(task_file, task_id=task_id)
        language = contract.baseline_language if contract.baseline_language in TARGET_LANGUAGES else "triton"
        if contract.case_weights is not None:
            workloads = json.loads((task_file.parent / "workloads.json").read_text())
            assert workloads["weight_provenance"]["kind"] == "agentx", task_id
            config = contract.execution_config(language, package_files(task_file.parent))
            assert config["contract"]["case_weights"] == contract.case_weights, task_id
            continue
        config = contract.execution_config(language, package_files(task_file.parent))
        assert set(config["contract"]) == CONTRACT_KEYS, task_id
        assert all(set(case) == CASE_KEYS for case in contract.resolved_contract(language)["cases"]), task_id
        assert TaskSpec.from_mapping(config, task_id=task_id).case_weights is None


# --- Aggregation --------------------------------------------------------------------------------

def _case(case_id, ms):
    return CaseResult(case_id, [1], ms, {"benchmark_method": "cuda_graph"})


def _pair(speedups):
    baseline = [_case(case_id, speedup) for case_id, speedup in speedups.items()]
    candidate = [_case(case_id, 1.0) for case_id in speedups]
    return baseline, candidate


def test_serving_weighted_speedup_of_known_values():
    # _pair: baseline time = the case's speedup, candidate time = 1 ms.
    baseline, candidate = _pair({"a": 2.0, "b": 0.5})
    assert calculate_average_speedup(baseline, candidate) == 1.0
    assert calculate_average_speedup(baseline, candidate, weights={"a": 3, "b": 1}) == pytest.approx(6.5 / 4)
    # only the ratios of the weights matter
    assert calculate_average_speedup(baseline, candidate, weights={"a": 0.75, "b": 0.25}) == pytest.approx(6.5 / 4)
    baseline, candidate = _pair({"m32": 1.5, "m16384": 1.1, "other": 0.9})
    weights = {"m32": 0.4, "m16384": 0.25, "other": 0.35}
    expected = (0.4 * 1.5 + 0.25 * 1.1 + 0.35 * 0.9) / (0.4 + 0.25 + 0.35)
    assert calculate_average_speedup(baseline, candidate, weights=weights) == pytest.approx(expected, rel=1e-15)


def test_a_win_on_one_case_cannot_hide_an_equal_loss_on_another():
    """Total time, not a geometric mean: 2x faster on one case and 2x slower on an equally busy one is a loss."""
    baseline = [_case("win", 1.0), _case("loss", 1.0)]
    candidate = [_case("win", 0.5), _case("loss", 2.0)]
    assert calculate_average_speedup(baseline, candidate) == pytest.approx(1.0)         # unweighted geomean
    assert calculate_average_speedup(baseline, candidate, weights={"win": 1, "loss": 1}) == pytest.approx(2 / 2.5)
    # the per-case weights are call rates: a rarely called shape barely moves the total
    assert calculate_average_speedup(baseline, candidate, weights={"win": 100, "loss": 1}) == pytest.approx(
        101 / 52)


@pytest.mark.parametrize("weights", [[0.0, 1.0], [-1.0, 1.0], [float("nan"), 1.0], [True, 1.0], [1.0]])
def test_serving_weighted_speedup_rejects_bad_weights(weights):
    with pytest.raises(ValueError):
        serving_weighted_speedup([(1.0, 1.0), (2.0, 1.0)], weights)


def test_unweighted_speedup_is_bit_identical_to_the_plain_geometric_mean():
    rng = random.Random(7)
    for _ in range(200):
        speedups = {f"c{i}": rng.uniform(0.2, 5.0) for i in range(rng.randint(1, 16))}
        baseline, candidate = _pair(speedups)
        values = [b.execution_time_ms / c.execution_time_ms for b, c in zip(baseline, candidate)]
        # The formula every unweighted task used before weights existed.
        assert calculate_average_speedup(baseline, candidate) == math.exp(
            sum(math.log(value) for value in values) / len(values))
        assert geometric_mean(values) == math.exp(sum(math.log(value) for value in values) / len(values))


@pytest.mark.parametrize("weights", [{"a": 1.0}, {"a": 1.0, "b": 1.0, "c": 1.0}, {"a": 1.0, "c": 1.0},
                                     {"a": 1.0, "b": 0.0}, {"a": 1.0, "b": float("nan")}])
def test_weights_must_name_exactly_the_scored_cases(weights):
    baseline, candidate = _pair({"a": 2.0, "b": 0.5})
    assert calculate_average_speedup(baseline, candidate, weights=weights) == 0.0


def test_paired_throughput_keeps_the_unweighted_geometric_mean():
    def row(role, case_id, throughput):
        return CaseResult(case_id, None, 1000.0, {
            "params": {"case": case_id}, "benchmark_method": "serving_wall_clock", "runtime_fingerprint": "r",
            "kernel_sha256": ("a" if role == "baseline" else "b") * 64,
            "metrics": {"output_tokens_per_s": throughput}})

    ratios = {"x": 1.3, "y": 0.8, "z": 1.7}
    pairs = [([row("baseline", k, 100.0) for k in ratios], [row("candidate", k, 100.0 * v) for k, v in ratios.items()])]
    result = paired_throughput(pairs)
    values = list(result["case_ratios"].values())
    assert result["raw_speedup_ratio"] == math.exp(sum(math.log(v) for v in values) / len(values))


# --- Evaluation with fake measurements ----------------------------------------------------------

class _FakeSession:
    """A schema-v3 session whose actions report fixed per-case times."""

    def __init__(self, spec, times):
        self.spec = spec
        self.times = times

    def _result(self, role, action):
        rows = ()
        if action == "performance":
            rows = tuple({"test_case_id": case_id, "status": "PASS", "execution_time_ms": self.times[role][case_id],
                          "benchmark_method": "cuda_graph", "params": {"case": case_id}}
                         for case_id in self.times[role])
        return SimpleNamespace(result=ActionResult(role, action, "PASS", rows))

    def candidate_action(self, action):
        return self._result("candidate", action)

    def baseline_action(self, action):
        return self._result("baseline", action)


def test_evaluation_scores_and_records_the_weighted_speedup(tmp_path):
    task = weighted_pilot(tmp_path / "task")
    spec = load_task_spec(task / "task.yaml", task_id=PILOT_ID, target_language="triton")
    baseline_ms = {case_id: 1.0 for case_id in WEIGHTS}
    baseline_ms["t2048"] = 4.0                      # 4x on the case that carries 40% of the workload
    session = _FakeSession(spec, {"baseline": baseline_ms, "candidate": {case_id: 1.0 for case_id in WEIGHTS}})
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    baseline = []
    result = evaluate_kernel(workspace, spec.to_mapping(), baseline, task_session=session)
    assert result["average_speedup"] == pytest.approx(0.4 * 4 + 0.6, rel=1e-12)
    assert result["benchmark_method_consistent"] and result["workload_consistent"]
    saved = load_performance_results(workspace, "optimized_perf.yaml")
    assert {case.test_case_id: case.metadata["weight"] for case in saved} == WEIGHTS
    result["case_weights"] = spec.case_weights
    write_task_result(workspace, result, baseline, PILOT_ID, "test", create_plots=False)
    report = yaml.safe_load((workspace / "task_result.yaml").read_text())
    assert report["speedup_ratio"] == pytest.approx(0.4 * 4 + 0.6, rel=1e-12)
    assert report["case_weights"] == WEIGHTS
    # Reproduce the task value from the per-case data a downstream consumer sees.
    by_id = {case.test_case_id: case for case in saved}
    reproduced = serving_weighted_speedup([(baseline_ms[k], by_id[k].execution_time_ms) for k in WEIGHTS],
                                          [by_id[k].metadata["weight"] for k in WEIGHTS])
    assert reproduced == pytest.approx(report["speedup_ratio"], rel=1e-12)


def test_unweighted_evaluation_writes_no_weights(tmp_path):
    spec = load_task_spec(PILOT / "task.yaml", task_id=PILOT_ID, target_language="triton")
    times = {case_id: 2.0 for case_id in WEIGHTS}
    session = _FakeSession(spec, {"baseline": times, "candidate": {k: 1.0 for k in times}})
    result = evaluate_kernel(tmp_path, spec.to_mapping(), [], task_session=session)
    assert result["average_speedup"] == 2.0
    assert "weight" not in (tmp_path / "optimized_perf.yaml").read_text()
