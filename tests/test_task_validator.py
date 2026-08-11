import logging
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import yaml

from agents.task_validator.launch_agent import _resolve_validation_timeouts
from agents.task_validator.report_schema import (
    CHECK_NAMES,
    finalize_report,
    normalize_report,
    validation_report_is_complete,
)
from agents.task_validator.validation_postprocessing import validation_post_processing
from agents.task_validator.validation_prompt import build_validation_prompt


def _valid_raw_report(task_name: str = "hip2hip/example") -> dict:
    checks = {
        name: {"status": "PASS", "details": f"{name} checked"}
        for name in CHECK_NAMES
    }
    attempt = {
        "command": "python3 check.py",
        "exit_code": 0,
        "timed_out": False,
        "duration_seconds": 1.0,
        "stdout_snippet": "ok",
        "stderr_snippet": "",
    }
    for name in ("compilation", "correctness", "performance"):
        checks[name]["attempts"] = [dict(attempt)]
    checks["benchmark_integrity"].update(
        {
            "case_count": 2,
            "valid_case_count": 2,
            "benchmark_methods": ["cuda_graph"],
            "event_fallback_reasons": [],
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
        {"guard_coverage_reviewed": True, "editable_targets_preserved": True}
    )
    return {
        "validation_schema_version": 2,
        "task_name": task_name,
        "validation_timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_status": "PASS",
        "checks": checks,
        "summary": "valid",
    }


class ValidationReportSchemaTests(unittest.TestCase):
    def test_framework_recomputes_lying_overall_status(self) -> None:
        raw = _valid_raw_report()
        raw["checks"]["correctness"]["status"] = "FAIL"
        report = normalize_report(raw, expected_task_name="hip2hip/example")
        self.assertEqual(report["agent_reported_overall_status"], "PASS")
        self.assertEqual(report["overall_status"], "FAIL")

    def test_timeout_is_an_overall_failure(self) -> None:
        raw = _valid_raw_report()
        raw["checks"]["performance"]["attempts"][0].update(
            {"timed_out": True, "exit_code": 124}
        )
        report = normalize_report(raw, expected_task_name="hip2hip/example")
        self.assertEqual(report["checks"]["performance"]["status"], "TIMEOUT")
        self.assertEqual(report["overall_status"], "FAIL")

    def test_missing_check_fails_closed(self) -> None:
        raw = _valid_raw_report()
        del raw["checks"]["self_contained"]
        report = normalize_report(raw, expected_task_name="hip2hip/example")
        self.assertEqual(report["checks"]["self_contained"]["status"], "FAIL")
        self.assertEqual(report["overall_status"], "FAIL")

    def test_task_name_mismatch_is_a_contract_failure(self) -> None:
        raw = _valid_raw_report(task_name="hip2hip/wrong")
        report = normalize_report(raw, expected_task_name="hip2hip/example")
        self.assertEqual(report["framework_status"], "FAIL")
        self.assertEqual(report["overall_status"], "FAIL")

    def test_warn_is_recomputed_without_becoming_failure(self) -> None:
        raw = _valid_raw_report()
        raw["checks"]["performance"]["status"] = "WARN"
        report = normalize_report(raw, expected_task_name="hip2hip/example")
        self.assertEqual(report["framework_status"], "PASS")
        self.assertEqual(report["overall_status"], "WARN")

    def test_skip_without_reason_fails_closed(self) -> None:
        raw = _valid_raw_report()
        raw["checks"]["performance"] = {"status": "SKIP", "details": "skipped"}
        raw["checks"]["benchmark_integrity"] = {
            "status": "SKIP",
            "skip_reason_code": "dependency_failed",
            "details": "skipped",
        }
        report = normalize_report(raw, expected_task_name="hip2hip/example")
        self.assertEqual(report["checks"]["performance"]["status"], "FAIL")
        self.assertEqual(report["overall_status"], "FAIL")

    def test_cpu_timer_is_not_a_scoreable_method(self) -> None:
        raw = _valid_raw_report()
        raw["checks"]["benchmark_integrity"]["benchmark_methods"] = [
            "cpu_timer_fallback"
        ]
        report = normalize_report(raw, expected_task_name="hip2hip/example")
        self.assertEqual(report["checks"]["benchmark_integrity"]["status"], "FAIL")
        self.assertEqual(report["overall_status"], "FAIL")

    def test_event_fallback_requires_reason(self) -> None:
        raw = _valid_raw_report()
        raw["checks"]["benchmark_integrity"]["benchmark_methods"] = [
            "cuda_event_fallback"
        ]
        report = normalize_report(raw, expected_task_name="hip2hip/example")
        self.assertEqual(report["checks"]["benchmark_integrity"]["status"], "FAIL")

    def test_unrestored_state_fails_benchmark_integrity(self) -> None:
        raw = _valid_raw_report()
        raw["checks"]["benchmark_integrity"]["state_restore_valid"] = False
        report = normalize_report(raw, expected_task_name="hip2hip/example")
        self.assertEqual(report["checks"]["benchmark_integrity"]["status"], "FAIL")

    def test_command_report_cannot_override_nonzero_exit(self) -> None:
        raw = _valid_raw_report()
        raw["checks"]["compilation"]["attempts"][0]["exit_code"] = 1
        raw["checks"]["compilation"]["report_file_valid"] = True
        report = normalize_report(raw, expected_task_name="hip2hip/example")
        self.assertEqual(report["checks"]["compilation"]["status"], "FAIL")

    def test_framework_finalization_is_required_for_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            report_path = workspace / "validation_report.yaml"
            report_path.write_text(yaml.safe_dump(_valid_raw_report()))
            self.assertFalse(validation_report_is_complete(workspace))

            finalized = finalize_report(
                workspace, expected_task_name="hip2hip/example"
            )
            self.assertEqual(finalized["overall_status"], "PASS")
            self.assertTrue(validation_report_is_complete(workspace))

            report_path.write_text(report_path.read_text() + "\n# tampered\n")
            self.assertFalse(validation_report_is_complete(workspace))

    def test_backend_failure_creates_complete_failing_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            report = finalize_report(
                workspace,
                expected_task_name="hip2hip/example",
                framework_error="backend timed out",
            )
            self.assertEqual(report["framework_status"], "FAIL")
            self.assertEqual(report["overall_status"], "FAIL")
            self.assertTrue(validation_report_is_complete(workspace))


class ValidationAggregationTests(unittest.TestCase):
    def test_postprocessor_uses_computed_report_status(self) -> None:
        logger = logging.getLogger(f"{__name__}.postprocess")
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory) / "task"
            workspace.mkdir()
            raw = _valid_raw_report()
            raw["checks"]["correctness"]["status"] = "FAIL"
            (workspace / "validation_report.yaml").write_text(yaml.safe_dump(raw))
            finalize_report(workspace, expected_task_name="hip2hip/example")

            self.assertFalse(validation_post_processing([str(workspace)], logger))
            summary = yaml.safe_load(
                (workspace.parent / "validation_summary.yaml").read_text()
            )
            self.assertFalse(summary["validation_passed"])
            self.assertEqual(summary["overall_counts"]["FAIL"], 1)


class ValidationLauncherTests(unittest.TestCase):
    def test_task_timeouts_override_validator_defaults(self) -> None:
        resolved = _resolve_validation_timeouts(
            {
                "compile_timeout": 1800,
                "correctness_timeout": 3600,
                "performance_timeout": 1800,
            },
            {
                "timeout_seconds": 1200,
                "compile_timeout": 600,
                "correctness_timeout": 600,
                "performance_timeout": 600,
            },
        )
        self.assertEqual(resolved[:3], (1800, 3600, 1800))
        self.assertGreaterEqual(resolved[3], 7500)

    def test_prompt_builder_handles_every_current_task_config(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        configs = sorted((repo_root / "tasks").rglob("config.yaml"))
        self.assertGreaterEqual(len(configs), 400)
        for config in configs:
            prompt = build_validation_prompt(
                str(config), "/tmp/validator-workspace", {"agent": {}}
            )
            self.assertIn("Perform all 12 checks", prompt, str(config))
            self.assertNotIn("cpu_timer_fallback` path", prompt, str(config))


if __name__ == "__main__":
    unittest.main()
