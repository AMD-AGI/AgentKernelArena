from unittest.mock import patch

import pytest

from src.evaluator import evaluate_kernel, write_task_result
from src.task_protocol import performance_cases
from tests.test_task_session_v2 import create


def test_v2_runs_through_existing_scoring_without_legacy_commands(tmp_path):
    session = create(tmp_path)
    assert session.validate_initial().accepted
    baseline = performance_cases(session.results[("task_validation", "baseline", "performance")].result)
    with patch("src.evaluator.evaluate_compilation", side_effect=AssertionError("legacy compilation")), \
         patch("src.evaluator.evaluate_correctness", side_effect=AssertionError("legacy correctness")), \
         patch("src.evaluator.measure_performance", side_effect=AssertionError("legacy timing")):
        result = evaluate_kernel(session.workspace, session.spec.to_mapping(), baseline, task_session=session)
    assert result["pass_compilation"]
    assert result["pass_correctness"]
    assert result["average_speedup"] == 1.0
    assert result["workload_consistent"]
    assert result["benchmark_method_consistent"]


def test_v2_missing_context_cannot_revert_to_legacy_evaluation(tmp_path):
    session = create(tmp_path)
    with pytest.raises(ValueError, match="TaskSession"):
        evaluate_kernel(session.workspace, session.spec.to_mapping(), [])


def test_v2_missing_submission_fails_and_keeps_baseline_diagnostic(tmp_path):
    import yaml

    session = create(tmp_path, empty=True, provided=4, diagnostic=True)
    initial = session.validate_initial()
    assert initial.accepted
    baseline = performance_cases(session.results[("task_validation", "baseline", "performance")].result)
    result = evaluate_kernel(session.workspace, session.spec.to_mapping(), baseline, task_session=session)
    assert not result["pass_compilation"]
    assert not result["pass_correctness"]
    assert result["average_speedup"] == 0
    result["baseline_correctness"] = session.results[("task_validation", "baseline", "correctness")].result.to_mapping()
    write_task_result(session.workspace, result, baseline, session.spec.task_id, "test", create_plots=False)
    report = yaml.safe_load((session.workspace / "task_result.yaml").read_text())
    assert report["baseline_correctness"]["status"] == "FAIL"
    assert report["speedup_ratio"] == 0
