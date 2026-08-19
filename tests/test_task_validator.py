import logging
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import yaml

from agents.task_validator.launch_agent import (
    _resolve_backend_settings,
    _resolve_validation_timeouts,
)
from agents.task_validator.report_schema import (
    CHECK_NAMES,
    REPORT_SCHEMA_VERSION,
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
        "validation_schema_version": REPORT_SCHEMA_VERSION,
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

    def test_task_name_mismatch_is_normalized_as_report_quality_warning(self) -> None:
        raw = _valid_raw_report(task_name="hip2hip/wrong")
        report = normalize_report(raw, expected_task_name="hip2hip/example")
        self.assertEqual(report["task_name"], "hip2hip/example")
        self.assertEqual(report["framework_status"], "PASS")
        self.assertEqual(report["overall_status"], "PASS")
        self.assertIn(
            "task_name mismatch: expected 'hip2hip/example', got 'hip2hip/wrong'",
            report["validation_warnings"],
        )

    def test_warn_is_recomputed_without_becoming_failure(self) -> None:
        raw = _valid_raw_report()
        raw["checks"]["performance"]["status"] = "WARN"
        report = normalize_report(raw, expected_task_name="hip2hip/example")
        self.assertEqual(report["framework_status"], "PASS")
        self.assertEqual(report["overall_status"], "WARN")

    def test_parseable_but_unscoreable_template_can_warn(self) -> None:
        raw = _valid_raw_report()
        raw["checks"]["result_template_compatibility"]["status"] = "WARN"
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

    def test_downstream_skip_inherits_valid_upstream_reason(self) -> None:
        raw = _valid_raw_report()
        raw["checks"]["performance"].update(
            {"status": "SKIP", "skip_reason_code": "starter_stub"}
        )
        raw["checks"]["correctness_implementation_review"] = {
            "status": "SKIP",
            "details": "No candidate implementation exists for the starter task.",
        }
        raw["checks"]["benchmark_integrity"].update(
            {"status": "SKIP", "skip_reason_code": "starter_stub"}
        )
        report = normalize_report(raw, expected_task_name="hip2hip/example")
        review = report["checks"]["correctness_implementation_review"]
        self.assertEqual(review["status"], "SKIP")
        self.assertEqual(review["skip_reason_code"], "starter_stub")
        self.assertEqual(report["framework_status"], "PASS")
        self.assertEqual(report["overall_status"], "PASS")
        self.assertIn(
            "correctness_implementation_review: inherited SKIP reason 'starter_stub' "
            "from an upstream check",
            report["validation_warnings"],
        )

    def test_benchmark_review_is_skipped_when_performance_is_skipped(self) -> None:
        raw = _valid_raw_report()
        raw["checks"]["performance"].update(
            {"status": "SKIP", "skip_reason_code": "starter_stub"}
        )
        raw["checks"]["benchmark_integrity"].update(
            {"status": "WARN", "replay_validation_valid": False}
        )
        report = normalize_report(raw, expected_task_name="hip2hip/example")
        benchmark = report["checks"]["benchmark_integrity"]
        self.assertEqual(benchmark["status"], "SKIP")
        self.assertEqual(benchmark["skip_reason_code"], "starter_stub")
        self.assertEqual(report["framework_status"], "PASS")
        self.assertEqual(report["overall_status"], "PASS")
        self.assertIn(
            "benchmark_integrity: normalized to SKIP because performance was "
            "SKIP/starter_stub",
            report["validation_warnings"],
        )

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
        self.assertEqual(report["framework_status"], "PASS")

    def test_missing_exact_replay_validation_is_warning(self) -> None:
        raw = _valid_raw_report()
        raw["checks"]["benchmark_integrity"].update(
            {
                "status": "FAIL",
                "replay_validation_valid": False,
                "evidence": [
                    {
                        "path": "scripts/task_runner.py",
                        "line_start": 20,
                        "line_end": 24,
                        "finding": "Graph timing does not request a timed replay handle.",
                    }
                ],
            }
        )
        report = normalize_report(raw, expected_task_name="hip2hip/example")
        self.assertEqual(report["framework_status"], "PASS")
        self.assertEqual(report["checks"]["benchmark_integrity"]["status"], "WARN")
        self.assertEqual(report["overall_status"], "WARN")
        self.assertIn("replay output is not validated", report["policy_findings"][0])

    def test_undetermined_review_field_is_warning(self) -> None:
        raw = _valid_raw_report()
        raw["checks"]["benchmark_integrity"].update(
            {
                "baseline_policy_immutable": None,
                "evidence": [
                    {
                        "path": "scripts/task_runner.py",
                        "finding": "Available task evidence cannot establish this field.",
                    }
                ],
            }
        )
        report = normalize_report(raw, expected_task_name="hip2hip/example")
        self.assertEqual(report["checks"]["benchmark_integrity"]["status"], "WARN")
        self.assertEqual(report["overall_status"], "WARN")

    def test_judgment_warning_without_evidence_is_report_quality_warning(self) -> None:
        raw = _valid_raw_report()
        raw["checks"]["correctness_implementation_review"]["status"] = "WARN"
        report = normalize_report(raw, expected_task_name="hip2hip/example")
        self.assertEqual(report["overall_status"], "WARN")
        self.assertIn(
            "correctness_implementation_review: FAIL/WARN should include a "
            "non-empty evidence[] list",
            report["validation_warnings"],
        )

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

    def test_finalizer_injects_authoritative_harness_guard_facts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            (workspace / "scripts").mkdir()
            (workspace / "scripts" / "task_runner.py").write_text("print('ok')\n")
            (workspace / "config.yaml").write_text(
                "task_type: hip2hip\nperformance_command:\n  - python3 scripts/task_runner.py\n"
            )
            raw = _valid_raw_report()
            raw["checks"]["harness_integrity"].update(
                {"guard_coverage_reviewed": False, "protected_paths": []}
            )
            (workspace / "validation_report.yaml").write_text(yaml.safe_dump(raw))

            report = finalize_report(workspace, expected_task_name="hip2hip/example")

            harness = report["checks"]["harness_integrity"]
            self.assertTrue(harness["framework_guard_enforced"])
            self.assertTrue(harness["guard_coverage_reviewed"])
            self.assertIn("config.yaml", harness["protected_paths"])
            self.assertIn("scripts/task_runner.py", harness["protected_paths"])


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
    def test_run_config_overrides_validator_backend_model_and_effort(self) -> None:
        resolved = _resolve_backend_settings(
            {
                "agent": {
                    "template": "task_validator",
                    "backend": "codex",
                    "model": "gpt-5.6-terra",
                    "effort": "high",
                }
            },
            {
                "backend": "claude_code",
                "model": "claude-sonnet-5",
                "effort": "max",
            },
        )

        self.assertEqual(resolved, ("codex", "gpt-5.6-terra", "high"))

    def test_validator_backend_settings_fall_back_to_agent_defaults(self) -> None:
        resolved = _resolve_backend_settings(
            {"agent": {"template": "task_validator"}},
            {
                "backend": "claude_code",
                "model": "claude-sonnet-5",
                "effort": "max",
            },
        )

        self.assertEqual(resolved, ("claude_code", "claude-sonnet-5", "max"))

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
            self.assertIn(
                "An asynchronous shell yield/session identifier is not a command result",
                prompt,
                str(config),
            )
            self.assertIn(
                "complete top-level `@triton.jit`/`@jit` helper nodes",
                prompt,
                str(config),
            )
            self.assertIn(
                "not require it to time a separate reference/candidate pair",
                prompt,
                str(config),
            )
            self.assertIn(
                "Required per-invocation intermediates that implement the documented public",
                prompt,
                str(config),
            )
            self.assertIn(
                "allocation of a reusable scratch/intermediate buffer may be hoisted",
                prompt,
                str(config),
            )
            self.assertIn(
                "distinguish Python executed once while constructing/capturing",
                prompt,
                str(config),
            )
            self.assertIn(
                "explicitly unsupported by the active backend/runtime",
                prompt,
                str(config),
            )
            self.assertIn("Missing replay\nvalidation alone is WARN", prompt, str(config))

    def test_prompt_includes_only_relevant_task_family_exception(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        torch2hip_prompt = build_validation_prompt(
            str(
                repo_root
                / "tasks/torch2hip/kernelbench/level3/l3n31_VisionAttention/config.yaml"
            ),
            "/tmp/validator-workspace",
            {"agent": {}},
        )
        hip2hip_prompt = build_validation_prompt(
            str(repo_root / "tasks/hip2hip/gpumode/SoftmaxModule/config.yaml"),
            "/tmp/validator-workspace",
            {"agent": {}},
        )

        self.assertIn("torch2hip generation placeholder policy", torch2hip_prompt)
        self.assertNotIn("torch2flydsl starter policy", torch2hip_prompt)
        self.assertNotIn("torch2hip generation placeholder policy", hip2hip_prompt)
        self.assertNotIn("torch2flydsl starter policy", hip2hip_prompt)


if __name__ == "__main__":
    unittest.main()
