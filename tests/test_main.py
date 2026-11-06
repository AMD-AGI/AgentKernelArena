"""Run orchestration in main.py: discovery, run-start validation and task outcomes.

CPU fixtures only; the task lifecycle (``src.task_run.run_task_v2``) is replaced
where a test concerns main's own handling around it.
"""
import argparse
from contextlib import redirect_stdout
import io
import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from agents.qa_auditor.report import load_report
from main import _build_context, _discover_tasks, run_task, should_run_task_for_platform
from src.module_registration import AgentType
from src.preprocessing import get_task_workspace_path, is_task_complete
from src.tools.agent_cost import report_agent_usage

LOGGER = logging.getLogger(__name__)
TIMESTAMP = "20261105_000000"


def write_tasks(root: Path) -> None:
    """Two schema-v3 scenarios of one operator and one schema-v2 serving task."""
    for scenario in ("bf16_a", "bf16_b"):
        task = root / "tasks/normalization/rms_norm" / scenario
        task.mkdir(parents=True)
        (task / "task.yaml").write_text("schema_version: 3\n")
    serving = root / "tasks/e2e/serving_fixture"
    serving.mkdir(parents=True)
    (serving / "config.yaml").write_text("schema_version: 2\n")


@pytest.fixture
def task_root(tmp_path, monkeypatch):
    write_tasks(tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def run(task_config: Path, run_directory: Path, *, agent=AgentType.QA_AUDITOR, task_name="e2e/serving_fixture",
        launcher=lambda **_: None, eval_config=None):
    return run_task(eval_config={} if eval_config is None else eval_config, agent=agent,
                    agent_launcher=launcher, task_name=task_name, task_config_dir=str(task_config),
                    run_directory=run_directory, timestamp=TIMESTAMP, logger=LOGGER,
                    task_index=1, total_tasks=1)


# --- task discovery and target languages ---------------------------------------------------------

def test_selectors_discover_v3_and_v2_tasks_without_languages(task_root):
    overrides = {}
    found = _discover_tasks(["normalization", "e2e"], overrides)
    assert found == {
        "e2e/serving_fixture": "tasks/e2e/serving_fixture/config.yaml",
        "normalization/rms_norm/bf16_a": "tasks/normalization/rms_norm/bf16_a/task.yaml",
        "normalization/rms_norm/bf16_b": "tasks/normalization/rms_norm/bf16_b/task.yaml",
    }
    assert overrides == {}
    assert _discover_tasks(["all"]) == found


def test_per_task_target_language_override_applies_to_every_selected_task(task_root):
    overrides = {}
    found = _discover_tasks([{"path": "normalization/rms_norm", "target_language": "hip"},
                             "normalization/rms_norm/bf16_a"], overrides)
    assert sorted(found) == ["normalization/rms_norm/bf16_a", "normalization/rms_norm/bf16_b"]
    assert overrides == {"normalization/rms_norm/bf16_a": "hip", "normalization/rms_norm/bf16_b": "hip"}


def test_same_language_selected_twice_is_accepted(task_root):
    overrides = {}
    _discover_tasks([{"path": "normalization", "target_language": "gluon"},
                     {"path": "normalization/rms_norm/bf16_a", "target_language": "gluon"}], overrides)
    assert set(overrides.values()) == {"gluon"}


def test_task_selected_with_two_target_languages_is_rejected(task_root):
    with pytest.raises(ValueError, match="normalization/rms_norm/bf16_a is selected with two target languages"):
        _discover_tasks([{"path": "normalization", "target_language": "triton"},
                         {"path": "normalization/rms_norm/bf16_a", "target_language": "hip"}], {})


@pytest.mark.parametrize("selectors,message", [
    ([], "No task selectors"),
    (["missing"], "matched no task configs"),
    ([""], "nonempty path prefixes"),
    ([{"path": "normalization", "target_language": "cuda"}], "target_language must be one of"),
    ([{"path": "normalization", "language": "hip"}], "takes path and optional target_language"),
    ([{"target_language": "hip"}], "takes path and optional target_language"),
])
def test_invalid_selectors_are_rejected(task_root, selectors, message):
    with pytest.raises(ValueError, match=message):
        _discover_tasks(selectors, {})


def build(task_root: Path, config: dict, agent_template: str = "claude_code"):
    config = {"agent": {"template": agent_template}, "workspace_directory_prefix": "workspace",
              "target_gpu_model": "MI355X", "log_directory": str(task_root / "logs"), **config}
    args = argparse.Namespace(run_suffix=None, config_name="unused.yaml", worker_id=None)
    run_directory = task_root / "run"
    run_directory.mkdir(exist_ok=True)
    with patch("main._load_config", return_value=config), \
            patch("main._resolve_run", return_value=(run_directory, "run_" + TIMESTAMP, TIMESTAMP, False)), \
            patch("main._configure_logging", return_value=LOGGER), \
            patch("main.apply_subprocess_python_path", return_value="python"), \
            patch("main.setup_rocm_env"):
        return _build_context(args, need_agent_launcher=False)


def test_v3_task_without_any_target_language_fails_at_run_start(task_root):
    assert build(task_root, {"tasks": ["normalization"]}) is None


def test_run_and_per_task_target_languages_reach_the_context(task_root):
    context = build(task_root, {"tasks": ["normalization", "e2e"], "target_language": "triton"})
    assert sorted(context["task_config_dict"]) == [
        "e2e/serving_fixture", "normalization/rms_norm/bf16_a", "normalization/rms_norm/bf16_b"]
    assert context["config"]["_task_target_languages"] == {}
    context = build(task_root, {"tasks": [{"path": "normalization", "target_language": "flydsl"}]})
    assert context["config"]["_task_target_languages"] == {
        "normalization/rms_norm/bf16_a": "flydsl", "normalization/rms_norm/bf16_b": "flydsl"}


def test_serving_tasks_and_audits_need_no_target_language(task_root):
    assert build(task_root, {"tasks": ["e2e"]}) is not None
    assert build(task_root, {"tasks": ["all"]}, agent_template="qa_auditor") is not None


@pytest.mark.parametrize("config", [
    {"tasks": ["normalization"], "target_language": "cuda"},
    {"tasks": [{"path": "normalization", "target_language": "triton"},
               {"path": "normalization/rms_norm/bf16_b", "target_language": "hip"}]},
])
def test_invalid_target_language_configuration_fails_at_run_start(task_root, config):
    assert build(task_root, config) is None


# --- task outcomes -------------------------------------------------------------------------------

def test_audit_setup_failure_is_recorded_as_a_complete_blocking_report(tmp_path):
    write_tasks(tmp_path)
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    task_name = "e2e/serving_fixture"
    expected = get_task_workspace_path(run_directory, task_name, TIMESTAMP)

    def fail_after_workspace_creation(**kwargs):
        expected.mkdir(parents=True)
        raise FileNotFoundError("serving runtime is unavailable")

    with patch("src.task_run.run_task_v2", side_effect=fail_after_workspace_creation):
        completed, workspace = run(tmp_path / "tasks/e2e/serving_fixture/config.yaml", run_directory)
    assert completed and workspace == expected
    report = load_report(expected)
    assert report["status"] == "FAIL" and report["task_id"] == task_name
    assert [finding["id"] for finding in report["findings"]] == ["QA-AUDIT"]
    assert "serving runtime is unavailable" in report["findings"][0]["detail"]
    assert is_task_complete(run_directory, task_name, TIMESTAMP, "qa_auditor")


def test_audit_failure_before_a_workspace_exists_writes_no_report(tmp_path):
    write_tasks(tmp_path)
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    with patch("src.task_run.run_task_v2", side_effect=ValueError("bad task")):
        completed, workspace = run(tmp_path / "tasks/e2e/serving_fixture/config.yaml", run_directory)
    assert (completed, workspace) == (False, None)
    assert not any(run_directory.iterdir())


def test_optimization_failure_never_writes_a_qa_report(tmp_path):
    write_tasks(tmp_path)
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    expected = get_task_workspace_path(run_directory, "e2e/serving_fixture", TIMESTAMP)

    def fail_after_workspace_creation(**kwargs):
        expected.mkdir(parents=True)
        raise RuntimeError("setup failed")

    with patch("src.task_run.run_task_v2", side_effect=fail_after_workspace_creation):
        completed, _ = run(tmp_path / "tasks/e2e/serving_fixture/config.yaml", run_directory,
                           agent=AgentType.CLAUDE_CODE)
    assert not completed
    assert not (expected / "qa_report.yaml").exists()


@pytest.mark.parametrize("name,text", [
    ("config.yaml", "schema_version: 1\n"),
    ("config.yaml", "schema_version: 3\n"),
    ("task.yaml", "schema_version: 2\n"),
    ("task.yaml", "schema_version: '3'\n"),
    ("task.yaml", "task_type: triton2triton\n"),
])
def test_unsupported_schema_versions_never_reach_the_task_lifecycle(tmp_path, name, text):
    (tmp_path / "task").mkdir()
    task_config = tmp_path / "task" / name
    task_config.write_text(text)
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    with patch("src.task_run.run_task_v2", side_effect=AssertionError("lifecycle started")):
        assert run(task_config, run_directory, task_name="suite/task") == (False, None)


# --- agent cost records ---------------------------------------------------------------------------

def run_with_launcher(tmp_path: Path, launcher, *, setup_error: Exception | None = None):
    write_tasks(tmp_path)
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    task_name = "normalization/rms_norm/bf16_a"

    def lifecycle(*, eval_config, agent_launcher, task_name, task_config_dir, run_directory, timestamp, **_):
        if setup_error is not None:
            raise setup_error
        workspace = get_task_workspace_path(run_directory, task_name, timestamp)
        workspace.mkdir(parents=True)
        try:
            agent_launcher(eval_config=eval_config, task_config_dir=task_config_dir, workspace=str(workspace))
        except Exception:
            return False, workspace
        return True, workspace

    with patch("src.task_run.run_task_v2", side_effect=lifecycle):
        completed, workspace = run(
            tmp_path / "tasks/normalization/rms_norm/bf16_a/task.yaml", run_directory,
            agent=AgentType.CLAUDE_CODE, task_name=task_name, launcher=launcher,
            eval_config={"agent": {"template": "claude_code", "timeout_seconds": 60},
                         "target_language": "triton"})
    return completed, workspace, run_directory


def cost_record(run_directory: Path, workspace: Path) -> dict:
    return json.loads((run_directory / "agent_costs" / f"{workspace.name}.json").read_text())


def test_agent_that_reports_nothing_gets_a_null_cost_record(tmp_path):
    completed, workspace, run_directory = run_with_launcher(tmp_path, lambda **_: "done")
    assert completed
    record = cost_record(run_directory, workspace)
    assert record["agent"] == "claude_code"
    assert record["task"] == "normalization/rms_norm/bf16_a"
    assert record["total_cost_usd"] is None
    assert record["cost_status"] == "not_reported"
    assert record["usage"] is None
    assert record["attempts"][0]["invocations"] == []
    assert record["attempts"][0]["run_settings"] == {"timeout_seconds": 60}


def test_reported_usage_and_launcher_failure_are_recorded(tmp_path):
    def launcher(**_):
        report_agent_usage(cost_usd=0.2, cost_scope="invocation",
                           usage={"input_tokens": 7}, usage_scope="invocation")
        raise RuntimeError("agent exited with code 1")

    completed, workspace, run_directory = run_with_launcher(tmp_path, launcher)
    assert not completed
    record = cost_record(run_directory, workspace)
    assert record["total_cost_usd"] == 0.2
    assert record["usage"] == {"input_tokens": 7}
    assert record["attempts"][0]["launcher_error"] == "RuntimeError"


def test_no_cost_record_when_the_agent_never_launched(tmp_path):
    completed, _, run_directory = run_with_launcher(
        tmp_path, lambda **_: "unused", setup_error=FileNotFoundError("missing source"))
    assert not completed
    assert not (run_directory / "agent_costs").exists()


# --- run-start budget validation ------------------------------------------------------------------

def build_budget(agent_settings: dict):
    config = {"tasks": [], "agent": agent_settings,
              "workspace_directory_prefix": "workspace", "target_gpu_model": "MI355X"}
    args = argparse.Namespace(run_suffix=None, config_name="unused.yaml")
    output = io.StringIO()
    with patch("main._load_config", return_value=config), \
            patch("main._resolve_run", side_effect=AssertionError("run directory resolved")), \
            redirect_stdout(output):
        context = _build_context(args, need_agent_launcher=True)
    return context, output.getvalue()


@pytest.mark.parametrize("template", ["geak", "forge", "qa_auditor"])
def test_time_mode_rejected_for_agents_without_session_resume(template):
    context, output = build_budget({"template": template, "budget_mode": "time"})
    assert context is None
    assert "not supported" in output


@pytest.mark.parametrize("template", ["claude_code", "geak"])
def test_unknown_budget_mode_rejected_for_every_agent(template):
    context, output = build_budget({"template": template, "budget_mode": "tokens"})
    assert context is None
    assert "budget_mode" in output


def test_removed_task_validator_template_is_rejected_at_run_start():
    context, output = build_budget({"template": "task_validator"})
    assert context is None
    assert "Invalid agent type" in output


@pytest.mark.parametrize("settings", [
    {"template": "claude_code", "budget_mode": "time"},
    {"template": "deepseek_harness", "budget_mode": "time"},
    {"template": "geak", "budget_mode": "iterations"},
    {"template": "geak"},
    {"template": "qa_auditor"},
])
def test_supported_modes_pass_validation(settings):
    with pytest.raises(AssertionError, match="run directory resolved"):
        build_budget(settings)


# --- platform support -----------------------------------------------------------------------------

def test_status_skip_prevents_task_from_running():
    assert not should_run_task_for_platform("example", {"platform_support": {"status": "skip"}}, "gfx950", LOGGER)


def test_required_arch_must_match_exactly():
    config = {"platform_support": {"status": "active", "required_arch": "gfx942"}}
    assert should_run_task_for_platform("example", config, "gfx942", LOGGER)
    assert not should_run_task_for_platform("example", config, "gfx950", LOGGER)


def test_required_arch_list_from_a_v3_task_and_unknown_arch():
    config = yaml.safe_load("platform_support: {required_arch: [gfx950, gfx942]}\n")
    assert should_run_task_for_platform("example", config, "gfx950", LOGGER)
    assert not should_run_task_for_platform("example", config, "gfx1201", LOGGER)
    assert not should_run_task_for_platform("example", config, None, LOGGER)


def test_active_without_required_arch_runs_on_current_arch():
    assert should_run_task_for_platform("example", {"platform_support": {"status": "active"}}, "gfx950", LOGGER)
