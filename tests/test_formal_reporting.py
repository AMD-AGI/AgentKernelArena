import csv
import hashlib
import io
import json
import logging
import os
import shutil
import stat
from pathlib import Path

import pytest
import yaml

import main as aka_main
from src import campaign, postprocessing
from src.tools import compare_runs
from src.score import task_result_scoring


def _write_read_only_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    path.chmod(0o444)


def _formal_manifest(task_names: list[str], arm: str = "codex") -> dict:
    tasks = [
        {"task_index": index, "task_name": task_name}
        for index, task_name in enumerate(task_names, 1)
    ]
    codex = {
        "attempt_timeout_seconds": 3600,
        "backend": "codex",
        "codex_binary_sha256": "a" * 64,
        "codex_version": "codex-test",
        "effort": "xhigh",
        "inner_max_iterations": 1,
        "isolation": {"sandbox": "workspace-write"},
        "max_turns": 50,
        "model": "gpt-test",
        "permission_mode": "workspace_write_isolated",
        "structured_stream_output_limit_bytes": 1024,
        "turn_policy": "structured_agent_turn_v1",
    }
    comparison = {
        "schema": "aka.apex-vs-codex-comparison-contract/v1",
        "objective_policy_id": "aka.task-package-objective-and-protected-harness/v1",
        "prompt_policy_id": "aka.shared-objective-backend-native-context-receipted/v1",
        "codex": codex,
        "tasks": tasks,
    }
    return {
        "schema": "aka.matched-campaign/v1",
        "agent": {"template": arm, **codex},
        "comparison_contract": comparison,
        "comparison_contract_sha256": hashlib.sha256(
            json.dumps(comparison, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "configuration": {
            "tasks": tasks,
        },
    }


def _task_result(task_name: str, speedup: float) -> dict:
    return {
        "task_name": task_name,
        "pass_compilation": True,
        "pass_correctness": True,
        "base_execution_time": speedup,
        "best_optimized_execution_time": 1.0,
        "speedup_ratio": speedup,
        "optimization_summary": "formal canonical result",
    }


def _write_canonical_workspace(
    run: Path, task_name: str, speedup: float
) -> Path:
    timestamp = "20260807_000000"
    safe_name = task_name.replace("/", "_")
    attempt_root = run / ".campaign_attempts" / safe_name
    selected = attempt_root / "attempt_01" / f"{safe_name}_{timestamp}"
    selected.mkdir(parents=True)
    result = _task_result(task_name, speedup)
    (selected / "task_result.yaml").write_text(yaml.safe_dump(result), encoding="utf-8")
    for evidence_name in ("baseline_perf.yaml", "optimized_perf.yaml"):
        (selected / evidence_name).write_text("test_cases: []\n", encoding="utf-8")
    selected_manifest = {
        path.relative_to(selected).as_posix(): postprocessing._sha256_file(path)
        for path in sorted(selected.rglob("*"))
        if path.is_file()
    }
    selected_manifest_sha256 = hashlib.sha256(
        json.dumps(selected_manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest_path = run / "campaign_manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    task_campaign_path = attempt_root / "task_campaign.yaml"
    task_campaign = {
        "schema": "aka.matched-task-attempts/v1",
        "task_name": task_name,
        "campaign_manifest_sha256": postprocessing._sha256_file(manifest_path),
        "comparison_contract_sha256": manifest["comparison_contract_sha256"],
        "campaign_manifest_unchanged": True,
        "policy": {"attempts": 1, "selection_policy": "test-selection"},
        "measurement_contract": "test-measurement",
        "attempts": [{
            "attempt": 1,
            "workspace": str(selected.relative_to(run)),
            "central_evaluator_report": str((selected / "task_result.yaml").relative_to(run)),
            "central_evaluator_report_sha256": selected_manifest["task_result.yaml"],
            "workspace_manifest_sha256": selected_manifest_sha256,
            "selection_eligible": True,
            "measured_rate_per_ms": 1.0,
        }],
        "selected_attempt": 1,
        "all_attempts_centrally_evaluated": True,
        "all_agent_sessions_succeeded": True,
        "within_evaluator_allowance": True,
        "within_task_timeout": True,
        "failure_reasons": [],
    }
    _write_read_only_yaml(task_campaign_path, task_campaign)
    campaign_evidence = {
        "schema": "aka.matched-task-attempts/v1",
        "campaign_manifest_sha256": postprocessing._sha256_file(manifest_path),
        "comparison_contract_sha256": manifest["comparison_contract_sha256"],
        "task_campaign_sha256": postprocessing._sha256_file(task_campaign_path),
        "attempt_count": 1,
        "selected_attempt": 1,
        "selection_policy": "test-selection",
        "selected_measured_rate_per_ms": 1.0,
        "attempt_manifest": str(task_campaign_path.relative_to(run)),
        "measurement_contract": "test-measurement",
        "is_apex_canonical_300_sample_grade": False,
        "selected_central_evaluator_report_sha256": selected_manifest["task_result.yaml"],
        "selected_performance_evidence_sha256": {
            name: selected_manifest[name]
            for name in ("baseline_perf.yaml", "optimized_perf.yaml")
        },
        "selected_workspace_manifest_sha256": selected_manifest_sha256,
    }
    canonical = run / f"{safe_name}_{timestamp}"
    canonical.mkdir()
    result["campaign_evidence"] = campaign_evidence
    _write_read_only_yaml(canonical / "task_result.yaml", result)
    for evidence_name in ("baseline_perf.yaml", "optimized_perf.yaml"):
        _write_read_only_yaml(canonical / evidence_name, {"test_cases": []})
    return canonical


def _write_failed_task(
    run: Path,
    task_name: str,
    *,
    index: int,
    total_tasks: int,
    eligibility_error: str = "agent_session_or_attempt_failed",
) -> Path:
    manifest_path = run / "campaign_manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    evidence_path = (
        run / ".campaign_attempts" / task_name.replace("/", "_") / "task_campaign.yaml"
    )
    evidence = {
        "schema": "aka.matched-task-attempts/v1",
        "task_name": task_name,
        "campaign_manifest_sha256": postprocessing._sha256_file(manifest_path),
        "comparison_contract_sha256": manifest["comparison_contract_sha256"],
        "campaign_manifest_unchanged": True,
        "all_attempts_centrally_evaluated": False,
        "all_agent_sessions_succeeded": False,
        "within_evaluator_allowance": True,
        "within_task_timeout": True,
        "selected_attempt": None,
        "attempts": [
            {
                "attempt": 1,
                "attempt_completed": False,
                "central_evaluator_report": None,
                "selection_eligible": False,
                "eligibility_errors": [eligibility_error],
            }
        ],
    }
    evidence["failure_reasons"] = campaign._campaign_failure_reasons(evidence)
    _write_read_only_yaml(evidence_path, evidence)

    descriptor = (
        run
        / ".parallel/running"
        / f"worker_{index}__{index:06d}_{task_name.replace('/', '_')}.yaml"
    )
    descriptor.parent.mkdir(parents=True, exist_ok=True)
    aka_main._write_descriptor(
        descriptor,
        {
            "index": index,
            "total_tasks": total_tasks,
            "task_name": task_name,
            "status": "running",
            "workspace_path": str(
                run / f"{task_name.replace('/', '_')}_20260807_000000"
            ),
        },
    )
    descriptor.chmod(0o444)
    aka_main.finish_descriptor(
        descriptor,
        "failed",
        workspace_path=None,
        worker_id=str(index),
        failure_reason="formal_task_not_canonical",
    )
    return evidence_path


def test_observed_percentiles_never_extrapolate_small_samples() -> None:
    for samples in ([1.0], [1.0, 4.0], [1.0, 2.0, 4.0]):
        stats = postprocessing._compute_speedup_stats(list(samples))
        for key in ("p25_speedup", "p75_speedup", "p90_speedup"):
            assert min(samples) <= stats[key] <= max(samples)
            assert stats[key] in samples
    assert postprocessing._compute_speedup_stats([1.0, 4.0])["p90_speedup"] == 4.0


def test_read_only_task_result_is_scored_without_mutation(tmp_path: Path) -> None:
    workspace = tmp_path / "canonical"
    result_path = workspace / "task_result.yaml"
    _write_read_only_yaml(result_path, _task_result("triton2triton/example", 2.0))
    before = result_path.read_bytes()

    assert task_result_scoring(str(workspace)) == 320.0
    assert result_path.read_bytes() == before
    assert stat.S_IMODE(result_path.stat().st_mode) == 0o444


def test_formal_report_uses_manifest_cohort_and_seals_outputs(tmp_path: Path) -> None:
    run = tmp_path / "workspace_MI355X_codex" / "run_20260807_000000_formal"
    run.mkdir(parents=True)
    task_names = [f"triton2triton/task_{index}" for index in range(1, 11)]
    _write_read_only_yaml(run / "campaign_manifest.yaml", _formal_manifest(task_names))

    canonical_paths: list[str] = []
    for index, task_name in enumerate(task_names[:6], 1):
        workspace = _write_canonical_workspace(run, task_name, float(index))
        canonical_paths.append(str(workspace))

    for index, task_name in enumerate(task_names[6:], 7):
        _write_failed_task(
            run,
            task_name,
            index=index,
            total_tasks=len(task_names),
            eligibility_error=f"failure_{index}",
        )

    aggregate = postprocessing.general_post_processing(
        canonical_paths,
        logging.getLogger(__name__),
        run_directory=run,
    )

    assert aggregate["total_tasks"] == 10
    assert aggregate["canonical_success_count"] == 6
    assert aggregate["failed_task_count"] == 4
    assert aggregate["correctness_pass_count"] == 6
    assert aggregate["correctness_pass_rate"] == 60.0
    report = (run / "reports/overall_report.txt").read_text(encoding="utf-8")
    assert "Manifest Tasks:        10" in report
    assert "Canonical Successes:   6/10" in report
    assert "Failed Tasks:          4/10" in report
    assert "Canonical-success Average Speedup" in report
    assert "Canonical-success Speedup Count:   6" in report
    assert "attempt_1:failure_7" in report
    assert ".campaign_attempts/triton2triton_task_7/task_campaign.yaml" in report

    csv_rows = list(
        csv.DictReader(
            io.StringIO(
                (run / "reports/overall_summary.csv").read_text(encoding="utf-8")
            )
        )
    )
    assert len(csv_rows) == 10
    assert sum(row["Campaign Status"] == "failed" for row in csv_rows) == 4

    report_paths = [
        run / "reports/overall_report.txt",
        run / "reports/task_type_breakdown.json",
        run / "reports/overall_summary.csv",
    ]
    before = {path: path.read_bytes() for path in report_paths}
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o444 for path in report_paths)
    assert aggregate["formal_completion_verified"] is True

    # Re-reading/re-projecting sealed evidence is idempotent and does not need a
    # temporary chmod or a score write-back.
    second = postprocessing.general_post_processing(
        canonical_paths,
        logging.getLogger(__name__),
        run_directory=run,
    )
    assert second == aggregate
    assert {path: path.read_bytes() for path in report_paths} == before


def test_invalid_canonical_result_contributes_no_passes_or_score(
    tmp_path: Path,
) -> None:
    run = tmp_path / "workspace_MI355X_apex" / "run_20260807_000000_formal"
    run.mkdir(parents=True)
    task_names = ["triton2triton/good", "triton2triton/invalid"]
    _write_read_only_yaml(run / "campaign_manifest.yaml", _formal_manifest(task_names))
    good = _write_canonical_workspace(run, task_names[0], 2.0)
    invalid = _write_canonical_workspace(run, task_names[1], 100.0)
    invalid_result = invalid / "task_result.yaml"
    payload = yaml.safe_load(invalid_result.read_text(encoding="utf-8"))
    payload["optimization_summary"] = 123
    invalid_result.chmod(0o644)
    invalid_result.write_text(yaml.safe_dump(payload), encoding="utf-8")
    invalid_result.chmod(0o444)

    with pytest.raises(ValueError, match="not terminal"):
        postprocessing.general_post_processing(
            [str(good), str(invalid)],
            logging.getLogger(__name__),
            run_directory=run,
        )
    assert not (run / "reports").exists()


def test_failed_marker_binds_read_only_campaign_evidence(tmp_path: Path) -> None:
    run = tmp_path / "run_20260807_000000_formal"
    run.mkdir()
    task_name = "triton2triton/example"
    _write_read_only_yaml(run / "campaign_manifest.yaml", _formal_manifest([task_name]))
    evidence_path = _write_failed_task(
        run,
        task_name,
        index=1,
        total_tasks=1,
    )

    marker = (
        run
        / ".parallel/failed/worker_1__000001_triton2triton_example.yaml"
    )
    payload = yaml.safe_load(marker.read_text(encoding="utf-8"))
    assert payload["failure"]["campaign_evidence_path"] == (
        ".campaign_attempts/triton2triton_example/task_campaign.yaml"
    )
    assert payload["failure"]["campaign_evidence_sha256"] == (
        postprocessing._sha256_file(evidence_path)
    )
    assert payload["failure"]["campaign_manifest_sha256"] == (
        postprocessing._sha256_file(run / "campaign_manifest.yaml")
    )
    assert payload["failure"]["comparison_contract_sha256"] == (
        _formal_manifest([task_name])["comparison_contract_sha256"]
    )
    assert "formal_task_not_canonical" in payload["failure"]["reason_codes"]
    assert stat.S_IMODE(marker.stat().st_mode) == 0o444


def test_task_campaign_failure_reasons_and_sealing_are_stable(tmp_path: Path) -> None:
    evidence = {
        "campaign_manifest_unchanged": True,
        "all_attempts_centrally_evaluated": True,
        "all_agent_sessions_succeeded": False,
        "within_evaluator_allowance": True,
        "within_task_timeout": True,
        "selected_attempt": 1,
        "attempts": [
            {
                "attempt": 1,
                "attempt_completed": False,
                "central_evaluator_report": "attempt/task_result.yaml",
                "selection_eligible": True,
                "eligibility_errors": ["agent_session_or_attempt_failed"],
            }
        ],
    }
    assert campaign._campaign_failure_reasons(evidence) == [
        "agent_session_failed",
        "attempt_1:agent_session_or_attempt_failed",
        "attempt_1:session_incomplete",
    ]

    path = tmp_path / "task_campaign.yaml"
    path.write_text("schema: test\n", encoding="utf-8")
    campaign._seal_evidence_file(path, "test evidence")
    assert stat.S_IMODE(path.stat().st_mode) == 0o444


def test_formal_reporting_never_scans_unexpected_attacker_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run_20260807_000000_formal"
    run.mkdir()
    task_name = "triton2triton/expected"
    _write_read_only_yaml(run / "campaign_manifest.yaml", _formal_manifest([task_name]))
    canonical = _write_canonical_workspace(run, task_name, 2.0)
    attacker = run / "attacker_controlled_20260807_000000"
    attacker.mkdir()
    (attacker / "task_result.yaml").symlink_to(tmp_path / "must_not_be_opened")
    monkeypatch.setattr(
        postprocessing,
        "_collect_all_tasks_from_run",
        lambda *_args: (_ for _ in ()).throw(AssertionError("attacker scan")),
    )

    aggregate = postprocessing.general_post_processing(
        [str(attacker), str(canonical)],
        logging.getLogger(__name__),
        run_directory=run,
    )

    assert aggregate["canonical_success_count"] == 1
    assert aggregate["total_tasks"] == 1


def test_canonical_full_tree_mutation_cannot_count_as_success(tmp_path: Path) -> None:
    run = tmp_path / "run_20260807_000000_formal"
    run.mkdir()
    task_name = "triton2triton/full_tree"
    _write_read_only_yaml(run / "campaign_manifest.yaml", _formal_manifest([task_name]))
    canonical = _write_canonical_workspace(run, task_name, 2.0)
    (canonical / "attacker_kernel.py").write_text("return_forged_result = True\n")

    with pytest.raises(ValueError, match="not terminal"):
        postprocessing.general_post_processing(
            [str(canonical)], logging.getLogger(__name__), run_directory=run
        )
    assert not (run / "reports").exists()


def test_formal_report_publish_rejects_symlink_escape_and_final_symlink(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (run / "reports").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="reports directory is unsafe"):
        postprocessing._prepare_reports_directory(run)

    (run / "reports").unlink()
    reports = postprocessing._prepare_reports_directory(run)
    victim = outside / "victim.txt"
    victim.write_text("unchanged", encoding="utf-8")
    final = reports / "overall_report.txt"
    final.symlink_to(victim)
    with pytest.raises(ValueError, match="unsafe immutable evidence"):
        postprocessing._publish_report(final, "forged", immutable=True)
    assert victim.read_text(encoding="utf-8") == "unchanged"

    final.unlink()
    predictable = reports / f".{final.name}.tmp.{os.getpid()}"
    predictable.symlink_to(victim)
    postprocessing._publish_report(final, "safe\n", immutable=True)
    assert final.read_text(encoding="utf-8") == "safe\n"
    assert victim.read_text(encoding="utf-8") == "unchanged"


def test_formal_postprocess_exception_propagates_and_postprocess_only_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run_20260807_000000_formal"
    run.mkdir()
    (run / "campaign_manifest.yaml").write_text("schema: test\n")

    def fail_handler(*_args, **_kwargs):
        raise RuntimeError("formal mismatch")

    monkeypatch.setattr(
        aka_main, "load_post_processing_handler", lambda *_args: fail_handler
    )
    with pytest.raises(RuntimeError, match="formal mismatch"):
        aka_main.run_post_processing(
            aka_main.AgentType.CODEX,
            [],
            logging.getLogger(__name__),
            run_directory=run,
        )

    context = {
        "agent": aka_main.AgentType.CODEX,
        "run_directory": run,
        "task_config_dict": {},
        "timestamp": "20260807_000000",
        "logger": logging.getLogger(__name__),
    }
    monkeypatch.setattr(aka_main, "_build_context", lambda *_args, **_kwargs: context)
    monkeypatch.setattr(
        aka_main,
        "run_post_processing",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("failed")),
    )
    assert aka_main.run_postprocess_only(object()) == 1


def test_duplicate_failed_marker_rejects_primary_reason(tmp_path: Path) -> None:
    run = tmp_path / "run_20260807_000000_formal"
    run.mkdir()
    task_name = "triton2triton/failure"
    _write_read_only_yaml(run / "campaign_manifest.yaml", _formal_manifest([task_name]))
    _write_failed_task(run, task_name, index=1, total_tasks=1)
    marker = next((run / ".parallel/failed").iterdir())
    duplicate = marker.with_name(marker.name.replace("worker_1__", "worker_evil__"))
    shutil.copy2(marker, duplicate)

    formal = postprocessing._load_formal_cohort(run)
    failure = postprocessing._validated_failure_binding(run, task_name, formal)

    assert failure["terminal_binding_verified"] is False
    assert "duplicate_failed_markers" in failure["reason_codes"]
    assert "formal_task_not_canonical" not in failure["reason_codes"]


def _write_complete_formal_run(
    root: Path, task_names: list[str], arm: str
) -> Path:
    root.mkdir(parents=True)
    _write_read_only_yaml(
        root / "campaign_manifest.yaml", _formal_manifest(task_names, arm)
    )
    canonical = [
        str(_write_canonical_workspace(root, task_name, float(index + 2)))
        for index, task_name in enumerate(task_names)
    ]
    postprocessing.general_post_processing(
        canonical, logging.getLogger(__name__), run_directory=root
    )
    return root


def test_compare_runs_requires_matching_completed_formal_contracts(tmp_path: Path) -> None:
    tasks = ["triton2triton/one", "triton2triton/two"]
    run1 = _write_complete_formal_run(
        tmp_path
        / "baseline/workspace_MI355X_codex/run_20260807_000000_formal",
        tasks,
        "codex",
    )
    run2 = _write_complete_formal_run(
        tmp_path
        / "treatment/workspace_MI355X_apex/run_20260807_000000_formal",
        tasks,
        "apex",
    )
    report = compare_runs.generate_comparison_report(
        run1, run2, tmp_path / "comparison.txt"
    )
    assert "Run Comparison Report" in report

    run3 = _write_complete_formal_run(
        tmp_path
        / "other/workspace_MI355X_apex/run_20260807_000000_formal",
        tasks[:1],
        "apex",
    )
    with pytest.raises(ValueError, match="contracts differ|cohorts differ"):
        compare_runs.generate_comparison_report(
            run1, run3, tmp_path / "must_not_publish.txt"
        )


def test_compare_runs_rejects_incomplete_formal_report(tmp_path: Path) -> None:
    run = tmp_path / "run_20260807_000000_formal"
    run.mkdir()
    task_name = "triton2triton/incomplete"
    _write_read_only_yaml(run / "campaign_manifest.yaml", _formal_manifest([task_name]))
    with pytest.raises(ValueError, match="not terminal"):
        postprocessing.general_post_processing(
            [], logging.getLogger(__name__), run_directory=run
        )
    assert not (run / "reports").exists()
