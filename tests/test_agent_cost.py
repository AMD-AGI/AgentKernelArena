"""Cost normalization, aggregation, and record writing; no provider is called."""

import json
import logging
from types import SimpleNamespace

import pytest

from src.tools.agent_cost import (
    AgentCostCollector, normalize_report, normalize_usage, report_agent_usage, summarize_invocations,
)

LOGGER = logging.getLogger(__name__)


def reports(*rows):
    return [normalize_report(**row) for row in rows]


def collect():
    return AgentCostCollector(agent="fake", task_name="suite/task",
                              run_settings={"model": "m", "budget_mode": "time", "template": "x"})


def test_usage_normalization_keeps_only_numeric_token_counters():
    usage = normalize_usage({
        "inputTokens": 5749, "outputTokens": 25, "cacheReadTokens": 9728, "cacheWriteTokens": 0,
        "input_tokens_bad": "12", "flag": True, "negative": -1, "nan": float("nan"),
        "server_tool_use": {"web_search_requests": 1}, "total_cost_usd": 1.5, "cost_available": True,
    })
    assert usage == {"input_tokens": 5749, "output_tokens": 25,
                     "cache_read_input_tokens": 9728, "cache_creation_input_tokens": 0}
    assert normalize_usage([{"input_tokens": 1}, {"input_tokens": 2}, None]) == {"input_tokens": 3}
    assert normalize_usage("not usage") == {}


@pytest.mark.parametrize("cost", ["0.25", True, float("nan"), float("inf"), -1, {"usd": 1}])
def test_malformed_cost_is_recorded_as_unknown(cost):
    assert normalize_report(cost_usd=cost)["cost_usd"] is None


def test_session_meter_counts_resumed_running_totals_once():
    # Claude Code >= 2.1.277: total_cost_usd keeps counting across --resume.
    summary = summarize_invocations(reports(
        dict(cost_usd=0.25, cost_scope="session", session_id="s"),
        dict(cost_usd=0.5, cost_scope="session", session_id="s"),
        dict(cost_usd=0.75, cost_scope="session", session_id="s"),
    ))
    assert summary["total_cost_usd"] == 0.75 and summary["cost_status"] == "complete"
    # A drop is a counter reset (the running total was not restored).
    reset = summarize_invocations(reports(
        dict(cost_usd=0.5, cost_scope="session", session_id="s"),
        dict(cost_usd=0.3, cost_scope="session", session_id="s"),
    ))
    assert reset["total_cost_usd"] == 0.8


def test_codex_thread_token_totals_are_not_double_counted():
    summary = summarize_invocations(reports(
        dict(usage={"input_tokens": 15363, "output_tokens": 5}, usage_scope="session", session_id="t"),
        dict(usage={"input_tokens": 31315, "output_tokens": 10}, usage_scope="session", session_id="t"),
    ))
    assert summary["usage"] == {"input_tokens": 31315, "output_tokens": 10}
    assert summary["usage_status"] == "complete"
    assert summary["total_cost_usd"] is None and summary["cost_status"] == "not_reported"


def test_invocation_scope_sums_and_missing_values_make_a_lower_bound():
    summary = summarize_invocations(reports(
        dict(cost_usd=0.25, cost_scope="invocation", usage={"input_tokens": 100}, usage_scope="invocation"),
        dict(cost_usd=0.5, cost_scope="invocation", usage={"input_tokens": 50}, usage_scope="invocation"),
        dict(cost_usd=None, cost_scope="invocation", timed_out=True),
    ))
    assert summary["total_cost_usd"] == 0.75 and summary["cost_status"] == "partial"
    assert summary["usage"] == {"input_tokens": 150} and summary["usage_status"] == "partial"
    flagged = summarize_invocations(reports(dict(cost_usd=1.0, cost_scope="invocation", complete=False)))
    assert flagged["total_cost_usd"] == 1.0 and flagged["cost_status"] == "partial"


def test_unknown_scope_with_a_resumed_session_is_null_not_guessed():
    resumed = summarize_invocations(reports(
        dict(cost_usd=0.25, session_id="s"), dict(cost_usd=0.5, session_id="s"),
    ))
    assert resumed["total_cost_usd"] is None and resumed["cost_status"] == "ambiguous"
    single = summarize_invocations(reports(dict(cost_usd=0.25, session_id="s")))
    assert single["total_cost_usd"] == 0.25 and single["cost_status"] == "complete"


def test_nothing_reported_is_null():
    for summary in (summarize_invocations([]), summarize_invocations(reports({}))):
        assert summary == {"total_cost_usd": None, "cost_status": "not_reported",
                           "usage": None, "usage_status": "not_reported"}


def test_reporting_without_a_collector_is_a_noop_and_never_raises(tmp_path):
    report_agent_usage(cost_usd=1.0, cost_scope="invocation")
    collector = collect()
    launcher = collector.wrap(lambda **kwargs: report_agent_usage(cost_usd=1.0, cost_scope="bogus"))
    launcher(eval_config={}, task_config_dir="c", workspace=str(tmp_path / "ws"))
    assert collector.invocations == [] and collector.report_errors


def test_collector_writes_one_record_and_appends_resumed_attempts(tmp_path):
    run_dir = tmp_path / "run"
    workspace = run_dir / "task_20260101_000000"

    def launcher(*, eval_config, task_config_dir, workspace):
        report_agent_usage(cost_usd=0.5, cost_scope="invocation", usage={"input_tokens": 10},
                           usage_scope="invocation", session_id="a", source="fake")
        return "ok"

    for _ in range(2):
        collector = collect()
        assert collector.wrap(launcher)(eval_config={}, task_config_dir="c", workspace=str(workspace)) == "ok"
        path = collector.write(run_dir, LOGGER)
    assert path == run_dir / "agent_costs" / f"{workspace.name}.json"
    record = json.loads(path.read_text())
    assert record["version"] == 1 and record["agent"] == "fake" and record["task"] == "suite/task"
    assert record["total_cost_usd"] == 1.0 and record["cost_status"] == "complete"
    assert record["usage"] == {"input_tokens": 20}
    assert len(record["attempts"]) == 2
    assert record["attempts"][0]["run_settings"] == {"model": "m", "budget_mode": "time"}
    assert not list(path.parent.glob(".*.tmp"))


def test_collector_records_launcher_failure_and_preserves_unreadable_record(tmp_path):
    run_dir = tmp_path / "run"
    workspace = run_dir / "task_ws"
    (run_dir / "agent_costs").mkdir(parents=True)
    (run_dir / "agent_costs" / "task_ws.json").write_text("{not json")

    def launcher(**kwargs):
        raise TimeoutError("agent timed out")

    collector = collect()
    with pytest.raises(TimeoutError):
        collector.wrap(launcher)(eval_config={}, task_config_dir="c", workspace=str(workspace))
    path = collector.write(run_dir, LOGGER)
    record = json.loads(path.read_text())
    assert record["attempts"][0]["launcher_error"] == "TimeoutError"
    assert record["total_cost_usd"] is None and record["cost_status"] == "not_reported"
    assert len(list((run_dir / "agent_costs").glob("task_ws.json.unreadable-*"))) == 1


def test_write_failure_returns_none_and_leaves_no_temporary_file(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    collector = collect()
    collector.wrap(lambda **kwargs: "ok")(eval_config={}, task_config_dir="c",
                                         workspace=str(run_dir / "task_ws"))

    def fail_replace(self, target):
        raise OSError("disk full")

    monkeypatch.setattr(type(run_dir), "replace", fail_replace)
    assert collector.write(run_dir, LOGGER) is None
    assert list((run_dir / "agent_costs").iterdir()) == []


def test_no_record_when_the_launcher_never_ran(tmp_path):
    assert collect().write(tmp_path, LOGGER) is None
    assert not (tmp_path / "agent_costs").exists()


def _collect_from(function, tmp_path):
    collector = collect()
    collector.wrap(lambda **kwargs: function())(eval_config={}, task_config_dir="c",
                                               workspace=str(tmp_path / "ws"))
    return collector.attempt()


def test_geak_sdk_cost_counts_running_total_once_and_dedupes_replayed_tasks(tmp_path):
    from agents.geak.launch_agent import report_sdk_cost
    from agents.geak.workflow_runner import _record_runtime_identity

    def message(name, **kwargs):
        return type(name, (SimpleNamespace,), {})(**kwargs)

    identity = {}
    notification = message("TaskNotificationMessage", task_id="t1", status="failed",
                           usage={"total_tokens": 900, "tool_uses": 3})
    _record_runtime_identity(message("ResultMessage", total_cost_usd=0.4, session_id="s",
                                     usage={"input_tokens": 10, "output_tokens": 2}, num_turns=2), identity)
    _record_runtime_identity(message("ResultMessage", total_cost_usd=1.0, session_id="s",
                                     usage={"input_tokens": 5, "output_tokens": 1}, num_turns=3), identity)
    _record_runtime_identity(notification, identity)
    _record_runtime_identity(notification, identity)  # replayed after the lifecycle
    assert identity["sdk_cost"]["task_usage"] == {"t1": {"total_tokens": 900, "tool_uses": 3}}
    assert identity["sdk_cost"]["background_after_result"] is True
    attempt = _collect_from(lambda: report_sdk_cost(identity, workflow_completed=True, seconds=5,
                                                    model="m"), tmp_path)
    assert attempt["total_cost_usd"] == 1.0 and attempt["cost_status"] == "partial"
    assert attempt["usage"] == {"input_tokens": 15, "output_tokens": 3}
    _record_runtime_identity(message("ResultMessage", total_cost_usd=1.2, session_id="s", usage={}), identity)
    attempt = _collect_from(lambda: report_sdk_cost(identity, workflow_completed=True, seconds=5,
                                                    model="m"), tmp_path)
    assert attempt["total_cost_usd"] == 1.2 and attempt["cost_status"] == "complete"
    missing = _collect_from(lambda: report_sdk_cost(None, workflow_completed=False, seconds=5,
                                                    model="m"), tmp_path)
    assert missing["total_cost_usd"] is None and missing["cost_status"] == "not_reported"


def test_forge_reports_engine_usage_and_respects_cost_availability(tmp_path):
    from agents.forge.adapter import report_engine_usage

    result = {"llm_usage": {"calls": 2, "input_tokens": 29, "output_tokens": 5, "cost_available": False,
                            "cost_source": "unavailable", "total_cost_usd": 0}}
    attempt = _collect_from(lambda: report_engine_usage(result, None, timed_out=False, seconds=3,
                                                        model="m"), tmp_path)
    assert attempt["total_cost_usd"] is None and attempt["cost_status"] == "not_reported"
    assert attempt["usage"] == {"calls": 2, "input_tokens": 29, "output_tokens": 5}
    priced = {"llm_usage": {**result["llm_usage"], "cost_available": True, "total_cost_usd": 0.3}}
    attempt = _collect_from(lambda: report_engine_usage(priced, None, timed_out=True, seconds=3,
                                                        model="m"), tmp_path)
    assert attempt["total_cost_usd"] == 0.3 and attempt["cost_status"] == "partial"
    initialization = {"llm_usage": {"input_tokens": 7, "cost_available": True, "total_cost_usd": 0.1}}
    attempt = _collect_from(lambda: report_engine_usage(None, initialization, timed_out=False, seconds=3,
                                                        model="m"), tmp_path)
    assert attempt["total_cost_usd"] == 0.1 and attempt["cost_status"] == "partial"


def test_deepseek_final_usage_is_read_only_when_emitted():
    from agents.deepseek_harness.launch_agent import _final_usage

    assert _final_usage(['{"type":"final","text":"done"}\n', "not json\n"]) is None
    assert _final_usage(['{"type":"final","usage":{"input_tokens":3}}\n']) == {"input_tokens": 3}
