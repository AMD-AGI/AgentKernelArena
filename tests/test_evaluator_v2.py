from unittest.mock import patch

import pytest

from src.evaluator import evaluate_kernel, write_task_result
from src.task_protocol import performance_cases
from tests.test_task_session_v2 import create


def test_v2_runs_every_step_through_the_session_actions(tmp_path):
    session = create(tmp_path)
    assert session.validate_initial().accepted
    baseline = performance_cases(session.results[("task_validation", "baseline", "performance")].result)
    with patch.object(session, "candidate_action", wraps=session.candidate_action) as action:
        result = evaluate_kernel(session.workspace, session.spec.to_mapping(), baseline, task_session=session)
    assert [call.args[0] for call in action.call_args_list] == ["compile", "correctness", "performance"]
    assert result["pass_compilation"]
    assert result["pass_correctness"]
    assert result["average_speedup"] == 1.0
    assert result["workload_consistent"]
    assert result["benchmark_method_consistent"]


def test_evaluation_requires_the_framework_session(tmp_path):
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
