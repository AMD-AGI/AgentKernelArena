import hashlib
import json
import logging
import shutil
from pathlib import Path

import pytest
import yaml

import main as aka_main
from src import campaign, postprocessing
from src.tools import compare_runs


TASK = "triton2triton/formal_compare"
TIMESTAMP = "20260807_000000"


def _write_read_only_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    path.chmod(0o444)


@pytest.fixture(autouse=True)
def _use_sealed_v5_runtime_test_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep reporting tests independent of host-only mount/runtime probes."""

    monkeypatch.setattr(campaign, "_revalidate_aka_runtime", lambda _manifest: True)
    monkeypatch.setattr(
        campaign,
        "verify_backend_closure",
        lambda closure, _expected_digest: closure,
    )


def _manifest(task_names: list[str], arm: str) -> dict:
    tasks = []
    for index, task_name in enumerate(task_names, 1):
        config_sha256 = hashlib.sha256(
            f"config:{index}:{task_name}".encode()
        ).hexdigest()
        package_files = {"config.yaml": config_sha256}
        tasks.append(
            {
                "task_index": index,
                "task_name": task_name,
                "config_path": f"/test/task_packages/task_{index:02d}/config.yaml",
                "config_sha256": config_sha256,
                "package_files_sha256": package_files,
                "package_manifest_sha256": hashlib.sha256(
                    json.dumps(
                        package_files, sort_keys=True, separators=(",", ":")
                    ).encode()
                ).hexdigest(),
            }
        )
    closure_material = {
        "schema": campaign.BACKEND_CLOSURE_SCHEMA,
        "backend": "codex",
        "launcher": {
            "requested_path": "/opt/node/bin/codex",
            "symlink_chain": [],
            "resolved_path": "/opt/node/bin/codex",
            "mode": 0o555,
            "size": 1,
            "sha256": "a" * 64,
        },
        "interpreter": {
            "resolved_path": "/opt/node/bin/node",
            "mode": 0o555,
            "size": 1,
            "sha256": "b" * 64,
        },
        "components": [],
    }
    closure = {
        **closure_material,
        "closure_sha256": hashlib.sha256(
            json.dumps(
                closure_material, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest(),
    }
    codex = {
        "attempt_timeout_seconds": 3600,
        "backend": "codex",
        "codex_binary_sha256": "a" * 64,
        "codex_version": "codex-test",
        "effort": "xhigh",
        "inner_max_iterations": 1,
        "isolation": {
            "approval": "never_via_strict_config",
            "execpolicy_rules": "ignored",
            "project_instructions": "backend_default_may_load",
            "sandbox": "workspace-write",
            "session": "ephemeral",
            "user_config": "ignored",
            "mount_scope": "attempt_only_bubblewrap",
            "attempt_containment_policy_id": (
                campaign.ATTEMPT_CONTAINMENT_POLICY
            ),
        },
        "max_turns": 50,
        "model": "gpt-test",
        "permission_mode": "workspace_write_isolated",
        "structured_stream_output_limit_bytes": 16 * 1024 * 1024,
        "turn_policy": campaign.CANDIDATE_PERSISTENCE_POLICY,
        "agent_process_containment_policy_id": (
            campaign.AGENT_PROCESS_CONTAINMENT_POLICY
        ),
        "attempt_containment_policy_id": campaign.ATTEMPT_CONTAINMENT_POLICY,
        "backend_runtime_closure_schema": campaign.BACKEND_CLOSURE_SCHEMA,
        "backend_runtime_closure_sha256": closure["closure_sha256"],
        "backend_runtime_closure": closure,
    }
    repositories = {
        "agent_kernel_arena": {
            "commit": "1" * 40,
            "tree": "2" * 40,
            "dirty": False,
            "status_sha256": "3" * 64,
            "execution_manifest_schema": campaign.EXECUTION_MANIFEST_SCHEMA,
            "execution_manifest_sha256": "4" * 64,
            "git_evidence_policy_id": "head_tree_direct_bytes_no_filters_v1",
        },
        "apex": {
            "commit": "5" * 40,
            "dirty": False,
            "status_sha256": "6" * 64,
            "runtime_manifest_sha256": "7" * 64,
        },
    }
    mount_receipt = {
        "schema": campaign.IMMUTABLE_MOUNT_RECEIPT_SCHEMA,
        "manifest_sha256": "4" * 64,
        "mount_point": "/test/aka-runtime",
        "sha256": "a" * 64,
    }
    aka_runtime = {
        "schema": "aka.execution-snapshot-runtime/v1",
        "root": "/test/aka-runtime",
        "manifest_path": "/test/aka-runtime-manifest.json",
        "manifest_file_sha256": "8" * 64,
        "manifest_sha256": "4" * 64,
        "mount_receipt_path": "/test/aka-runtime-mount-receipt.json",
        "mount_receipt_file_sha256": "9" * 64,
        "mount_receipt_sha256": mount_receipt["sha256"],
        "mount_receipt_schema": campaign.IMMUTABLE_MOUNT_RECEIPT_SCHEMA,
        "mount_receipt": mount_receipt,
    }
    gpu = {
        "gpu_boundary_plan_sha256": "d" * 64,
        "exclusivity": {
            "sha256": "e" * 64,
            "exclusivity_verified": True,
        },
        "devices": [
            {
                "host_device_id": "0",
                "unique_id": "0x0000000000000001",
                "render_nodes": ["/dev/dri/renderD128"],
            }
        ],
        "task_mapping": [
            {
                "task_index": index,
                "task_name": task_name,
                "assigned_host_gpu_id": "0",
            }
            for index, task_name in enumerate(task_names, 1)
        ],
    }
    runtime = {"gpu": gpu, "aka_execution_snapshot": aka_runtime}
    evaluator = {
        "schema": "aka.evaluator-source-binding/v2",
        "coverage": "all_committed_files",
        "execution_manifest_schema": campaign.EXECUTION_MANIFEST_SCHEMA,
        "execution_manifest_sha256": "4" * 64,
        "commit": "1" * 40,
        "tree": "2" * 40,
    }
    apex_treatment = {
        "template": "apex",
        "session_receipt_schema": "agentkernelarena.apex-attempt-receipt/v5",
        "apex_runtime_mount_policy_id": campaign.APEX_RUNTIME_MOUNT_POLICY,
        "attempt_mount_receipt_schema": campaign.ATTEMPT_MOUNT_RECEIPT_SCHEMA,
        "apex_runtime_mount_schema": campaign.APEX_RUNTIME_MOUNT_SCHEMA,
        "runtime_manifest_sha256": repositories["apex"][
            "runtime_manifest_sha256"
        ],
    }
    comparison = {
        "schema": "aka.apex-vs-codex-comparison-contract/v5",
        "formal_execution": dict(campaign._FORMAL_LIVE_COMMITMENT),
        "formal_execution_sha256": campaign.FORMAL_LIVE_EXECUTION_SHA256,
        "objective_policy_id": "aka.task-package-objective-and-protected-harness/v1",
        "prompt_policy_id": "aka.shared-objective-backend-native-context-receipted/v1",
        "candidate_persistence_policy_id": campaign.CANDIDATE_PERSISTENCE_POLICY,
        "boundary_quiescence_policy_id": campaign.BOUNDARY_QUIESCENCE_POLICY,
        "agent_process_containment_policy_id": (
            campaign.AGENT_PROCESS_CONTAINMENT_POLICY
        ),
        "attempt_containment_policy_id": campaign.ATTEMPT_CONTAINMENT_POLICY,
        "repositories": repositories,
        "apex_treatment": apex_treatment,
        "codex": codex,
        "runtime": runtime,
        "evaluator_files_sha256": evaluator,
        "tasks": tasks,
    }
    agent = {
        **codex,
        "template": arm,
        "session_receipt_schema": (
            "agentkernelarena.apex-attempt-receipt/v5"
            if arm == "apex"
            else "agentkernelarena.codex-attempt-receipt/v4"
        ),
    }
    if arm == "apex":
        agent |= apex_treatment
    return {
        "schema": "aka.matched-campaign/v1",
        "formal_execution": dict(campaign._FORMAL_LIVE_COMMITMENT),
        "formal_execution_sha256": campaign.FORMAL_LIVE_EXECUTION_SHA256,
        "agent": agent,
        "comparison_contract": comparison,
        "comparison_contract_sha256": hashlib.sha256(
            json.dumps(comparison, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "repositories": repositories,
        "runtime": runtime,
        "evaluator_files_sha256": evaluator,
        "configuration": {"tasks": tasks},
    }


def _result(task_name: str, speedup: float) -> dict:
    return {
        "task_name": task_name,
        "pass_compilation": True,
        "pass_correctness": True,
        "base_execution_time": speedup,
        "best_optimized_execution_time": 1.0,
        "speedup_ratio": speedup,
        "optimization_summary": "sealed formal result",
    }


def _write_canonical_workspace(
    run: Path, task_name: str, speedup: float = 2.0
) -> Path:
    safe_name = task_name.replace("/", "_")
    attempt_root = run / ".campaign_attempts" / safe_name
    selected = attempt_root / "attempt_01" / f"{safe_name}_{TIMESTAMP}"
    selected.mkdir(parents=True)
    result = _result(task_name, speedup)
    (selected / "task_result.yaml").write_text(
        yaml.safe_dump(result), encoding="utf-8"
    )
    for name in ("baseline_perf.yaml", "optimized_perf.yaml"):
        (selected / name).write_text("test_cases: []\n", encoding="utf-8")
    selected_manifest = {
        path.relative_to(selected).as_posix(): postprocessing._sha256_file(path)
        for path in sorted(selected.rglob("*"))
        if path.is_file()
    }
    selected_manifest_sha256 = hashlib.sha256(
        json.dumps(
            selected_manifest, sort_keys=True, separators=(",", ":")
        ).encode()
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
            "central_evaluator_report": str(
                (selected / "task_result.yaml").relative_to(run)
            ),
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
        "selected_central_evaluator_report_sha256": selected_manifest[
            "task_result.yaml"
        ],
        "selected_performance_evidence_sha256": {
            name: selected_manifest[name]
            for name in ("baseline_perf.yaml", "optimized_perf.yaml")
        },
        "selected_workspace_manifest_sha256": selected_manifest_sha256,
    }
    canonical = run / f"{safe_name}_{TIMESTAMP}"
    canonical.mkdir()
    result["campaign_evidence"] = campaign_evidence
    _write_read_only_yaml(canonical / "task_result.yaml", result)
    for name in ("baseline_perf.yaml", "optimized_perf.yaml"):
        _write_read_only_yaml(canonical / name, {"test_cases": []})
    return canonical


def _write_failed_task(run: Path, task_name: str) -> None:
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
        "attempts": [{
            "attempt": 1,
            "attempt_completed": False,
            "central_evaluator_report": None,
            "selection_eligible": False,
            "eligibility_errors": ["agent_session_or_attempt_failed"],
        }],
    }
    evidence["failure_reasons"] = campaign._campaign_failure_reasons(evidence)
    _write_read_only_yaml(evidence_path, evidence)

    safe_name = task_name.replace("/", "_")
    descriptor = run / ".parallel/running" / f"worker_1__000001_{safe_name}.yaml"
    descriptor.parent.mkdir(parents=True)
    aka_main._write_descriptor(
        descriptor,
        {
            "index": 1,
            "total_tasks": 1,
            "task_name": task_name,
            "status": "running",
            "workspace_path": str(run / f"{safe_name}_{TIMESTAMP}"),
        },
    )
    descriptor.chmod(0o444)
    aka_main.finish_descriptor(
        descriptor,
        "failed",
        workspace_path=None,
        worker_id="1",
        failure_reason="formal_task_not_canonical",
    )


def _make_run(
    root: Path, arm: str, *, failed: bool = False, label: str = "arm"
) -> Path:
    run = (
        root
        / label
        / f"workspace_MI355X_{arm}"
        / f"run_{TIMESTAMP}_formal"
    )
    run.mkdir(parents=True)
    _write_read_only_yaml(run / "campaign_manifest.yaml", _manifest([TASK], arm))
    if failed:
        _write_failed_task(run, TASK)
        workspaces = []
    else:
        workspaces = [str(_write_canonical_workspace(run, TASK))]
    postprocessing.general_post_processing(
        workspaces, logging.getLogger(__name__), run_directory=run
    )
    return run


def _rewrite_report(run: Path, mutate) -> None:
    path = run / "reports/task_type_breakdown.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.chmod(0o644)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o444)


def test_formal_compare_recomputes_score_and_emits_population_labels(
    tmp_path: Path,
) -> None:
    codex = _make_run(tmp_path, "codex", label="baseline")
    apex = _make_run(tmp_path, "apex", label="treatment")
    report = compare_runs.generate_comparison_report(
        codex, apex, tmp_path / "comparison.txt"
    )
    assert "Canonical Success Tasks" in report
    assert "Failed Tasks" in report
    assert "Canonical-success-only Speedup Count" in report
    assert "Canonical-success-only Average Speedup" in report

    _rewrite_report(
        codex,
        lambda payload: payload["overall"].__setitem__("total_score", 999999.0),
    )
    with pytest.raises(ValueError, match="recomputed sealed evidence"):
        compare_runs.load_run_data(codex)


def test_formal_compare_rejects_forged_completion_after_evidence_loss(
    tmp_path: Path,
) -> None:
    run = _make_run(tmp_path, "codex")
    canonical = run / f"{TASK.replace('/', '_')}_{TIMESTAMP}"
    shutil.rmtree(canonical)

    with pytest.raises(ValueError, match="recomputed sealed evidence"):
        compare_runs.load_run_data(run)


def test_formal_compare_recomputes_failed_task_evidence(tmp_path: Path) -> None:
    run = _make_run(tmp_path, "apex", failed=True)
    _rewrite_report(
        run,
        lambda payload: payload["failed_tasks"][0].__setitem__(
            "reason_codes", ["forged_failure"]
        ),
    )

    with pytest.raises(ValueError, match="recomputed sealed evidence"):
        compare_runs.load_run_data(run)


def test_formal_compare_requires_distinct_apex_and_codex_arms(tmp_path: Path) -> None:
    codex_one = _make_run(tmp_path, "codex", label="one")
    codex_two = _make_run(tmp_path, "codex", label="two")
    with pytest.raises(ValueError, match="exactly one apex and one codex"):
        compare_runs.generate_comparison_report(
            codex_one, codex_two, tmp_path / "same_arm.txt"
        )

    with pytest.raises(ValueError, match="cannot compare a run with itself"):
        compare_runs.generate_comparison_report(
            codex_one, codex_one, tmp_path / "self.txt"
        )
