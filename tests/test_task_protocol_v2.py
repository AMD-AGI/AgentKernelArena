import json

import pytest

from src.task_protocol import (
    CaseManifest, RESULT_PREFIX, TaskProtocolError, baseline_correctness_accepted,
    merge_command_results, parse_command_result,
)
from src.task_spec import BaselineSpec


def case(case_id="one", **kwargs):
    return dict(test_case_id=case_id, shape=[2, 3], dtype="float32", params={"transpose": False},
                status="PASS", **kwargs)


def envelope(role="candidate", action="correctness", cases=None, **kwargs):
    return dict(protocol="arena-eval-v1", role=role, action=action, status="PASS",
                cases=[case()] if cases is None else cases, **kwargs)


def parse(obj, returncode=0):
    return parse_command_result("debug log: fail/pass are not verdicts\n" + RESULT_PREFIX + json.dumps(obj),
                                role=obj["role"], action=obj["action"], returncode=returncode)


def manifest():
    return CaseManifest.from_result(parse(envelope("task", "validate-task", [
        case("one", checks=["correctness", "performance"]), case("two", checks=["correctness", "performance"])])))


def test_real_verdict_is_structured_not_a_search_for_pass_in_logs():
    assert parse(envelope()).passed
    obj = envelope()
    obj.update(status="FAIL", reason="Wrong output")
    obj["cases"][0]["status"] = "FAIL"
    assert not parse(obj, returncode=1).passed


@pytest.mark.parametrize("text", ["PASS", "", RESULT_PREFIX + "{}", RESULT_PREFIX + "[]",
                                  RESULT_PREFIX + '{"status":"PASS","status":"FAIL"}'])
def test_invalid_reports_fail(text):
    with pytest.raises(TaskProtocolError):
        parse_command_result(text, role="candidate", action="correctness", returncode=0)


def test_a_second_report_or_wrong_role_is_not_accepted():
    text = RESULT_PREFIX + json.dumps(envelope())
    with pytest.raises(TaskProtocolError, match="exactly one"):
        parse_command_result(text + "\n" + text, role="candidate", action="correctness", returncode=0)
    with pytest.raises(TaskProtocolError, match="match invocation"):
        parse_command_result(text, role="baseline", action="correctness", returncode=0)


@pytest.mark.parametrize("returncode", [1, 3, -9])
def test_zero_exit_and_pass_must_agree(returncode):
    with pytest.raises(TaskProtocolError, match="exit code"):
        parse(envelope(), returncode=returncode)


def test_outer_pass_cannot_override_failed_case():
    obj = envelope()
    obj["cases"][0]["status"] = "FAIL"
    with pytest.raises(TaskProtocolError, match="failed case"):
        parse(obj)


@pytest.mark.parametrize("latency", [0, -1, True, "0.1", float("nan"), float("inf")])
def test_timing_requires_real_finite_positive_number(latency):
    with pytest.raises(TaskProtocolError):
        parse(envelope(action="performance", cases=[case(execution_time_ms=latency, benchmark_method="cuda_graph")]))


def test_unknown_host_timing_is_not_a_device_measurement():
    with pytest.raises(TaskProtocolError, match="device benchmark_method"):
        parse(envelope(action="performance", cases=[case(execution_time_ms=1, benchmark_method="time.time")]))


def test_manifest_prevents_both_sides_omitting_the_same_case():
    expected = manifest()
    for role in ("baseline", "candidate"):
        result = parse(envelope(role, "performance", [case(execution_time_ms=0.1, benchmark_method="cuda_graph")]))
        with pytest.raises(TaskProtocolError, match="missing=.*two"):
            expected.validate(result)


@pytest.mark.parametrize("identity", [{"shape": [3, 2]}, {"dtype": "float16"}, {"params": {"transpose": True}}])
def test_case_id_cannot_hide_a_changed_workload(identity):
    rows = [case("one"), case("two")]
    rows[0].update(identity)
    with pytest.raises(TaskProtocolError, match="identity changed"):
        manifest().validate(parse(envelope(cases=rows)))


def test_split_command_results_require_complete_nonduplicate_coverage():
    one = parse(envelope(cases=[case("one")]))
    two = parse(envelope(cases=[case("two")]))
    manifest().validate(merge_command_results([one, two]))
    with pytest.raises(TaskProtocolError, match="Duplicate case across"):
        merge_command_results([one, one])


def test_performance_cases_need_correctness_coverage_in_task_manifest():
    with pytest.raises(TaskProtocolError, match="lacks correctness"):
        parse(envelope("task", "validate-task", [case(checks=["performance"])]))


def test_baseline_exception_requires_complete_numerical_evidence_and_keeps_fail():
    obj = envelope("baseline", "correctness", [case("one"), case("two")])
    obj.update(status="FAIL", reason="Finite output exceeds task tolerance", failure_kind="numerical_mismatch")
    obj["cases"][0].update(status="FAIL", failure_kind="numerical_mismatch")
    result = parse(obj, returncode=1)
    policy = BaselineSpec("provided", None, (), "diagnostic", "Production accuracy differs")
    assert baseline_correctness_accepted(result, baseline=policy, phase="task_validation", manifest=manifest())
    assert result.status == "FAIL"
    assert not baseline_correctness_accepted(result, baseline=policy, phase="candidate_evaluation", manifest=manifest())
    obj["role"] = "candidate"
    with pytest.raises(TaskProtocolError, match="another role"):
        baseline_correctness_accepted(parse(obj, 1), baseline=policy, phase="task_validation", manifest=manifest())


@pytest.mark.parametrize("failure_kind", [None, "crash", "nonfinite", "shape_mismatch", "missing_case"])
def test_diagnostic_cannot_excuse_non_numerical_failures(failure_kind):
    obj = envelope("baseline", "correctness", [case("one"), case("two")])
    obj.update(status="FAIL", reason="Failed", failure_kind="numerical_mismatch")
    obj["cases"][0].update(status="FAIL", failure_kind=failure_kind)
    policy = BaselineSpec("provided", None, (), "diagnostic", "Known numerical deviation")
    assert not baseline_correctness_accepted(parse(obj, 1), baseline=policy, phase="task_validation", manifest=manifest())
