"""Workload scoring and CPU action fixtures; these are not GPU measurements."""
import hashlib
import json
import sys

import pytest
import yaml

from src.evaluator import evaluate_kernel, write_task_result
from src.score import score
from src.task_protocol import performance_cases
from src.task_session import TaskSession
from src.task_spec import TaskSpec
from src.testcases import TestCaseResult as Case, calculate_average_speedup, weighted_total_time
from tests.test_task_session_v2 import RUNNER


def cases(times):
    return [Case(key, [i + 1], time, {"benchmark_method": "cuda_graph"})
            for i, (key, time) in enumerate(zip(("rare", "frequent"), times))]


def test_workload_regression_is_not_rewarded_by_a_large_rare_case_speedup():
    baseline, candidate = cases((1, 100)), cases((0.1, 120))
    assert calculate_average_speedup(baseline, candidate) > 5
    weights = {"rare": 1, "frequent": 10}
    assert calculate_average_speedup(baseline, candidate, case_weights=weights) == pytest.approx(1001 / 1200.1)
    # Proportional frequencies produce the same objective, even if output is reordered.
    assert calculate_average_speedup(baseline, candidate[::-1], case_weights={
        "rare": 0.1, "frequent": 1,
    }) == pytest.approx(1001 / 1200.1)


@pytest.mark.parametrize("fault", ["missing", "extra", "duplicate", "renamed", "synthetic",
                                  "shape", "method", "nan", "zero", "overflow", "bool"])
def test_weighting_preserves_complete_case_and_timing_contract(fault):
    baseline, candidate = cases((1, 100)), cases((0.1, 80))
    if fault == "missing":
        candidate.pop()
    elif fault == "extra":
        candidate.append(Case("extra", [3], 1, {"benchmark_method": "cuda_graph"}))
    elif fault == "duplicate":
        candidate[1].test_case_id = "rare"
    elif fault == "renamed":
        candidate[0].test_case_id = "another"
    elif fault == "synthetic":
        candidate[0].metadata["_synthetic_test_case_id"] = True
    elif fault == "shape":
        candidate[0].shape = [99]
    elif fault == "method":
        candidate[0].metadata["benchmark_method"] = "cuda_event_fallback"
    else:
        candidate[1].execution_time_ms = {"nan": float("nan"), "zero": 0, "overflow": 1e308, "bool": True}[fault]
    assert calculate_average_speedup(baseline, candidate, require_complete_match=False,
                                     case_weights={"rare": 1, "frequent": 10}) == 0


@pytest.mark.parametrize("weights", [{}, {"rare": 1}, {"rare": 1, "frequent": 2, "extra": 1},
                                     {"rare": True, "frequent": 1}, {"rare": -1, "frequent": 1},
                                     {"rare": float("inf"), "frequent": 1},
                                     {"rare": 10**1000, "frequent": 1}])
def test_invalid_weighted_totals_fail_closed(weights):
    with pytest.raises(ValueError):
        weighted_total_time(cases((1, 2)), weights)


def test_session_evaluation_and_report_use_the_declared_workload_objective(tmp_path):
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    # Execute the standard CPU fixture protocol with two declared cases.
    text = RUNNER.replace('print("ARENA_EVAL_RESULT="', '''
if action != "compile":
    report["cases"] = [dict(case, test_case_id=key, shape=[i + 1])
                       for i, key in enumerate(("rare", "frequent"))]
    if action == "performance":
        for row, time in zip(report["cases"], (1, 100) if role == "baseline" else (0.1, 120)):
            row["execution_time_ms"] = time
print("ARENA_EVAL_RESULT="''')
    (workspace / "evaluate.py").write_text(text)
    (workspace / "kernel.py").write_text("def compute(): return 3\n")
    (workspace / "workload.json").write_text('{"workload": "CPU scoring fixture"}\n')
    config = {
        "schema_version": 2,
        "candidate": {"language": "hip", "initial_state": "implemented", "editable": ["kernel.py"],
                      "entrypoints": [{"file": "kernel.py", "kind": "function", "symbol": "compute"}]},
        "evaluation": {"runner": [sys.executable, "evaluate.py"], "workloads": "workload.json", "aggregation": {
            "method": "weighted_total_time", "case_weights": {"rare": 1, "frequent": 10},
        }},
    }
    session = TaskSession.create(TaskSpec.from_mapping(config, task_id="fixture/workload"),
                                 workspace, tmp_path / "state")
    assert session.validate_initial().accepted
    baseline = performance_cases(session.results[("task_validation", "baseline", "performance")].result)
    result = evaluate_kernel(workspace, config, baseline, task_session=session)
    assert result["pass_compilation"] and result["pass_correctness"]
    assert result["average_speedup"] == pytest.approx(1001 / 1200.1)
    write_task_result(workspace, result, baseline, session.spec.task_id, "any-agent", create_plots=False)
    report = yaml.safe_load((workspace / "task_result.yaml").read_text())
    assert report["performance_aggregation"] == {
        "method": "weighted_total_time", "case_weights": {"rare": 1, "frequent": 10},
        "baseline_total_time_ms": 1001, "candidate_total_time_ms": 1200.1,
        "task_spec_sha256": hashlib.sha256(json.dumps(
            session.spec.to_mapping(), sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest(),
        "workloads": {"path": "workload.json", "sha256": hashlib.sha256(
            (session.baseline_workspace / "workload.json").read_bytes()).hexdigest()},
    }
    assert report["base_execution_time"] == 50.5  # Legacy diagnostic averages stay averages.
    assert report["best_optimized_execution_time"] == 60.05
    assert score(True, True, 50.5, 60.05, report["speedup_ratio"],
                 benchmark_method_consistent=True) == 0


def test_declared_aggregation_cannot_fall_back_to_legacy_evaluation(tmp_path):
    with pytest.raises(ValueError, match="TaskSession"):
        evaluate_kernel(tmp_path, {"evaluation": {"aggregation": {
            "method": "weighted_total_time", "case_weights": {"one": 1},
        }}}, [])
