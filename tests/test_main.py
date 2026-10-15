import argparse
from contextlib import redirect_stdout
import io
import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from main import _build_context, run_task, should_run_task_for_platform
from src.tools.agent_cost import report_agent_usage
from agents.task_validator.report_schema import (
    CHECK_NAMES,
    REPORT_SCHEMA_VERSION,
    finalize_report,
    validation_report_is_complete,
)
from src.module_registration import AgentType
from src.preprocessing import get_task_workspace_path


class TaskValidatorWorkspaceTests(unittest.TestCase):
    def test_validator_records_workspace_setup_failure_as_complete_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_directory = root / "run"
            run_directory.mkdir()
            config_path = root / "config.yaml"
            config_path.write_text("task_type: image_kernel\n")
            task_name = "image_kernel/missing_repo"
            timestamp = "20260721_000000"
            expected_workspace = get_task_workspace_path(
                run_directory, task_name, timestamp
            )

            def fail_after_workspace_creation(*args, **kwargs):
                expected_workspace.mkdir(parents=True)
                raise FileNotFoundError("image_repo_path is unavailable")

            with patch("main.setup_workspace", side_effect=fail_after_workspace_creation):
                completed, workspace = run_task(
                    eval_config={},
                    agent=AgentType.TASK_VALIDATOR,
                    agent_launcher=lambda **kwargs: None,
                    task_name=task_name,
                    task_config_dir=str(config_path),
                    run_directory=run_directory,
                    timestamp=timestamp,
                    logger=logging.getLogger(__name__),
                    task_index=1,
                    total_tasks=1,
                )

            self.assertTrue(completed)
            self.assertEqual(workspace, expected_workspace)
            self.assertTrue(validation_report_is_complete(expected_workspace))
            report = yaml.safe_load(
                (expected_workspace / "validation_report.yaml").read_text()
            )
            self.assertEqual(report["framework_status"], "FAIL")
            self.assertTrue(
                any(
                    "image_repo_path is unavailable" in error
                    for error in report["validation_errors"]
                )
            )

    def test_validator_does_not_treat_copied_report_as_current_run_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            task_dir = root / "task"
            run_directory = root / "run"
            task_dir.mkdir()
            run_directory.mkdir()
            config_path = task_dir / "config.yaml"
            config_path.write_text("task_type: flydsl2flydsl\n")
            (task_dir / "validation_report.yaml").write_text("overall_status: WARN\n")
            (task_dir / ".validation_complete").write_text("copied stale marker\n")

            launcher_called = False

            def launcher(*, eval_config, task_config_dir, workspace):
                nonlocal launcher_called
                launcher_called = True
                report_path = Path(workspace) / "validation_report.yaml"
                self.assertFalse(report_path.exists())
                self.assertFalse((Path(workspace) / ".validation_complete").exists())
                checks = {
                    name: {"status": "PASS", "details": "checked"}
                    for name in CHECK_NAMES
                }
                attempt = {
                    "command": "true",
                    "exit_code": 0,
                    "timed_out": False,
                }
                for name in ("compilation", "correctness", "performance"):
                    checks[name]["attempts"] = [dict(attempt)]
                checks["benchmark_integrity"].update(
                    {
                        "case_count": 1,
                        "valid_case_count": 1,
                        "benchmark_methods": ["cuda_graph"],
                        "method_metadata_complete": True,
                        "method_policy_valid": True,
                        "case_identity_complete": True,
                        "baseline_policy_immutable": True,
                        "state_restore_valid": True,
                        "workload_symmetric": True,
                        "replay_validation_valid": True,
                        "representative_inputs_valid": True,
                        "timing_boundaries_valid": True,
                    }
                )
                checks["harness_integrity"].update(
                    {
                        "guard_coverage_reviewed": True,
                        "editable_targets_preserved": True,
                    }
                )
                report_path.write_text(
                    yaml.safe_dump(
                        {
                            "validation_schema_version": REPORT_SCHEMA_VERSION,
                            "task_name": "flydsl2flydsl/example",
                            "validation_timestamp": "2026-07-21T00:00:00+00:00",
                            "overall_status": "PASS",
                            "checks": checks,
                        }
                    )
                )
                finalize_report(
                    workspace, expected_task_name="flydsl2flydsl/example"
                )

            completed, workspace = run_task(
                eval_config={},
                agent=AgentType.TASK_VALIDATOR,
                agent_launcher=launcher,
                task_name="flydsl2flydsl/example",
                task_config_dir=str(config_path),
                run_directory=run_directory,
                timestamp="20260721_000000",
                logger=logging.getLogger(__name__),
                task_index=1,
                total_tasks=1,
            )

            self.assertTrue(launcher_called)
            self.assertTrue(completed)
            self.assertIsNotNone(workspace)
            self.assertTrue(validation_report_is_complete(workspace))


class AgentCostRecordTests(unittest.TestCase):
    """main.run_task writes one cost record per launched task, for every agent."""

    task_name = "hip2hip/example"
    timestamp = "20260721_000000"

    def run_with(self, launcher, root: Path, *, setup_error: Exception | None = None):
        run_directory = root / "run"
        run_directory.mkdir()
        config_path = root / "config.yaml"
        config_path.write_text("task_type: hip2hip\n")

        def make_workspace(task_config_dir, run_dir, timestamp, logger, task_name):
            if setup_error is not None:
                raise setup_error
            workspace = get_task_workspace_path(run_dir, task_name, timestamp)
            workspace.mkdir(parents=True)
            return workspace

        with patch("main.setup_workspace", side_effect=make_workspace), \
                patch("main.evaluate_compilation", return_value=(False, "skipped")), \
                patch("main.snapshot_workspace_harness", return_value=object()), \
                patch("main.verify_workspace_harness"), \
                patch("main.materialize_perf_helpers_in_workspace"), \
                patch("main.evaluate_kernel", return_value={}), \
                patch("main.write_task_result"), \
                patch("main.is_task_complete", return_value=True):
            completed, workspace = run_task(
                eval_config={"agent": {"template": "claude_code", "timeout_seconds": 60}},
                agent=AgentType.CLAUDE_CODE,
                agent_launcher=launcher,
                task_name=self.task_name,
                task_config_dir=str(config_path),
                run_directory=run_directory,
                timestamp=self.timestamp,
                logger=logging.getLogger(__name__),
                task_index=1,
                total_tasks=1,
            )
        return completed, workspace, run_directory

    def record(self, run_directory: Path, workspace: Path) -> dict:
        return json.loads((run_directory / "agent_costs" / f"{workspace.name}.json").read_text())

    def test_agent_that_reports_nothing_gets_a_null_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            completed, workspace, run_directory = self.run_with(
                lambda **kwargs: "done", Path(temporary_directory))
            self.assertTrue(completed)
            record = self.record(run_directory, workspace)
            self.assertEqual(record["agent"], "claude_code")
            self.assertEqual(record["task"], self.task_name)
            self.assertIsNone(record["total_cost_usd"])
            self.assertEqual(record["cost_status"], "not_reported")
            self.assertIsNone(record["usage"])
            self.assertEqual(record["attempts"][0]["invocations"], [])
            self.assertEqual(record["attempts"][0]["run_settings"], {"timeout_seconds": 60})

    def test_reported_usage_and_launcher_failure_are_recorded(self) -> None:
        def launcher(**kwargs):
            report_agent_usage(cost_usd=0.2, cost_scope="invocation",
                               usage={"input_tokens": 7}, usage_scope="invocation")
            raise RuntimeError("agent exited with code 1")

        with tempfile.TemporaryDirectory() as temporary_directory:
            completed, workspace, run_directory = self.run_with(launcher, Path(temporary_directory))
            self.assertFalse(completed)
            record = self.record(run_directory, workspace)
            self.assertEqual(record["total_cost_usd"], 0.2)
            self.assertEqual(record["usage"], {"input_tokens": 7})
            self.assertEqual(record["attempts"][0]["launcher_error"], "RuntimeError")

    def test_no_record_when_the_agent_never_launched(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            completed, _, run_directory = self.run_with(
                lambda **kwargs: "unused", Path(temporary_directory),
                setup_error=FileNotFoundError("missing source"))
            self.assertFalse(completed)
            self.assertFalse((run_directory / "agent_costs").exists())


class BudgetModeRunValidationTests(unittest.TestCase):
    """An unsupported budget_mode fails at run start, before any run directory exists."""

    def build(self, agent_settings: dict):
        config = {"tasks": [], "agent": agent_settings,
                  "workspace_directory_prefix": "workspace", "target_gpu_model": "MI355X"}
        args = argparse.Namespace(run_suffix=None, config_name="unused.yaml")
        output = io.StringIO()
        with patch("main._load_config", return_value=config), \
                patch("main._resolve_run", side_effect=AssertionError("run directory resolved")), \
                redirect_stdout(output):
            context = _build_context(args, need_agent_launcher=True)
        return context, output.getvalue()

    def test_time_mode_rejected_for_agents_without_session_resume(self) -> None:
        for template in ("geak", "forge", "deepseek_harness", "task_validator"):
            with self.subTest(template=template):
                context, output = self.build({"template": template, "budget_mode": "time"})
                self.assertIsNone(context)
                self.assertIn("not supported", output)

    def test_unknown_budget_mode_rejected_for_every_agent(self) -> None:
        for template in ("claude_code", "geak"):
            with self.subTest(template=template):
                context, output = self.build({"template": template, "budget_mode": "tokens"})
                self.assertIsNone(context)
                self.assertIn("budget_mode", output)

    def test_supported_modes_pass_validation(self) -> None:
        for settings in ({"template": "claude_code", "budget_mode": "time"},
                         {"template": "geak", "budget_mode": "iterations"},
                         {"template": "geak"}):
            with self.subTest(settings=settings):
                with self.assertRaisesRegex(AssertionError, "run directory resolved"):
                    self.build(settings)


class PlatformSupportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.platform")
        self.logger.disabled = True

    def test_status_skip_prevents_task_from_running(self) -> None:
        self.assertFalse(
            should_run_task_for_platform(
                "example",
                {"platform_support": {"status": "skip"}},
                "gfx950",
                self.logger,
            )
        )

    def test_required_arch_must_match_exactly(self) -> None:
        config = {
            "platform_support": {
                "status": "active",
                "required_arch": "gfx942",
            }
        }
        self.assertTrue(
            should_run_task_for_platform("example", config, "gfx942", self.logger)
        )
        self.assertFalse(
            should_run_task_for_platform("example", config, "gfx950", self.logger)
        )

    def test_active_without_required_arch_runs_on_current_arch(self) -> None:
        self.assertTrue(
            should_run_task_for_platform(
                "example",
                {"platform_support": {"status": "active"}},
                "gfx950",
                self.logger,
            )
        )


if __name__ == "__main__":
    unittest.main()
