import copy
import hashlib
import json
import logging
import os
import re
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import main as aka_main
from src import campaign


def _policy() -> dict[str, object]:
    return {
        "comparison": "apex_vs_codex",
        "attempts": 3,
        "attempt_timeout_seconds": 3600,
        "apex_internal_allowance_seconds": 3600,
        "task_timeout_seconds": 25200,
        "evaluator_allowance_seconds": 3600,
        "selection_policy": "correctness_then_measured_rate_v1",
        "workspace_policy": "fresh_per_attempt",
        "gpu_policy": "deterministic_task_gpu_v1",
        "require_clean_checkouts": True,
    }


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@pytest.fixture(autouse=True)
def _use_sealed_runtime_test_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use complete gen6 contracts while isolating host-only runtime probes.

    Mount topology and installed-backend revalidation have dedicated tests.  The
    descriptor tests need the same sealed contract fields, but do not need a
    live SquashFS mount or Codex installation merely to exercise queue races.
    """

    monkeypatch.setattr(campaign, "_revalidate_aka_runtime", lambda _manifest: True)
    monkeypatch.setattr(
        campaign,
        "verify_backend_closure",
        lambda closure, _expected_digest: closure,
    )


def _backend_closure() -> dict[str, object]:
    return {
        "schema": campaign.BACKEND_CLOSURE_SCHEMA,
        "backend": "codex",
        "launcher": {"path": "/opt/node/bin/codex", "sha256": "c" * 64},
        "interpreter": {"path": "/opt/node/bin/node", "sha256": "d" * 64},
        "components": [],
        "closure_sha256": "e" * 64,
    }


def _formal_v6_contract(
    *,
    tasks: list[dict[str, object]],
    mappings: list[dict[str, object]],
    devices: list[dict[str, object]],
    run_config_path: Path,
    run_config: dict[str, object],
) -> dict[str, object]:
    runtime_manifest_sha256 = "7" * 64
    aka_manifest_sha256 = "4" * 64
    closure = _backend_closure()
    repositories = {
        "agent_kernel_arena": {
            "commit": "1" * 40,
            "tree": "2" * 40,
            "dirty": False,
            "status_sha256": "3" * 64,
            "execution_manifest_schema": campaign.EXECUTION_MANIFEST_SCHEMA,
            "execution_manifest_sha256": aka_manifest_sha256,
            "git_evidence_policy_id": "head_tree_direct_bytes_no_filters_v1",
        },
        "apex": {
            "commit": "5" * 40,
            "dirty": False,
            "status_sha256": "6" * 64,
            "runtime_manifest_sha256": runtime_manifest_sha256,
        },
    }
    aka_runtime = {
        "schema": "aka.execution-snapshot-runtime/v1",
        "root": "/test/aka-runtime",
        "manifest_path": "/test/aka-runtime-manifest.json",
        "manifest_file_sha256": "8" * 64,
        "manifest_sha256": aka_manifest_sha256,
        "mount_receipt_path": "/test/aka-runtime-mount-receipt.json",
        "mount_receipt_file_sha256": "9" * 64,
        "mount_receipt_sha256": "a" * 64,
        "mount_receipt_schema": campaign.IMMUTABLE_MOUNT_RECEIPT_SCHEMA,
        "mount_receipt": {
            "schema": campaign.IMMUTABLE_MOUNT_RECEIPT_SCHEMA,
            "manifest_sha256": aka_manifest_sha256,
            "mount_point": "/test/aka-runtime",
            "sha256": "a" * 64,
        },
    }
    gpu = {
        "gpu_boundary_plan_sha256": "a" * 64,
        "exclusivity": {
            "sha256": "b" * 64,
            "exclusivity_verified": True,
        },
        "devices": devices,
        "task_mapping": mappings,
    }
    runtime = {"gpu": gpu, "aka_execution_snapshot": aka_runtime}
    comparison_runtime = campaign.comparison_runtime_projection(runtime)
    assert comparison_runtime is not None
    evaluator = {
        "schema": "aka.evaluator-source-binding/v2",
        "coverage": "all_committed_files",
        "execution_manifest_schema": campaign.EXECUTION_MANIFEST_SCHEMA,
        "execution_manifest_sha256": aka_manifest_sha256,
        "commit": "1" * 40,
        "tree": "2" * 40,
    }
    apex_treatment = {
        "template": "apex",
        "session_receipt_schema": "agentkernelarena.apex-attempt-receipt/v6",
        "apex_runtime_mount_policy_id": campaign.APEX_RUNTIME_MOUNT_POLICY,
        "attempt_mount_receipt_schema": campaign.ATTEMPT_MOUNT_RECEIPT_SCHEMA,
        "apex_runtime_mount_schema": campaign.APEX_RUNTIME_MOUNT_SCHEMA,
        "runtime_manifest_sha256": runtime_manifest_sha256,
    }
    codex = {
        "backend": "codex",
        "model": "gpt-5.5",
        "effort": "xhigh",
        "permission_mode": "workspace_write_isolated",
        "inner_max_iterations": 1,
        "attempt_timeout_seconds": 3600,
        "max_turns": 50,
        "turn_policy": campaign.CANDIDATE_PERSISTENCE_POLICY,
        "agent_process_containment_policy_id": (
            campaign.AGENT_PROCESS_CONTAINMENT_POLICY
        ),
        "attempt_containment_policy_id": campaign.ATTEMPT_CONTAINMENT_POLICY,
        "structured_stream_output_limit_bytes": 16 * 1024 * 1024,
        "codex_version": "codex-cli test",
        "codex_binary_sha256": "f" * 64,
        "backend_runtime_closure_schema": campaign.BACKEND_CLOSURE_SCHEMA,
        "backend_runtime_closure_sha256": closure["closure_sha256"],
        "backend_runtime_closure": closure,
        "isolation": {
            "approval": "never_via_strict_config",
            "execpolicy_rules": "ignored",
            "project_instructions": "backend_default_may_load",
            "sandbox": "workspace-write",
            "session": "ephemeral",
            "user_config": "ignored",
            "mount_scope": "attempt_only_bubblewrap",
            "attempt_containment_policy_id": campaign.ATTEMPT_CONTAINMENT_POLICY,
        },
    }
    apex_transport = campaign.FORMAL_AGENT_TRANSPORT_TREATMENTS["apex"]
    agent = codex | apex_treatment | {
        "max_process_output_bytes": apex_transport["max_process_output_bytes"],
        "structured_stream_overflow_policy": apex_transport["overflow_policy"],
    }
    formal_execution = dict(campaign._FORMAL_LIVE_COMMITMENT)
    comparison = {
        "schema": "aka.apex-vs-codex-comparison-contract/v6",
        "formal_execution": formal_execution,
        "formal_execution_sha256": campaign.FORMAL_LIVE_EXECUTION_SHA256,
        "objective_policy_id": "aka.task-package-objective-and-protected-harness/v1",
        "prompt_policy_id": (
            "aka.shared-objective-backend-native-context-receipted/v1"
        ),
        "candidate_persistence_policy_id": campaign.CANDIDATE_PERSISTENCE_POLICY,
        "boundary_quiescence_policy_id": campaign.BOUNDARY_QUIESCENCE_POLICY,
        "agent_process_containment_policy_id": (
            campaign.AGENT_PROCESS_CONTAINMENT_POLICY
        ),
        "attempt_containment_policy_id": campaign.ATTEMPT_CONTAINMENT_POLICY,
        "repositories": repositories,
        "apex_treatment": apex_treatment,
        "agent_transport_treatments": copy.deepcopy(
            campaign.FORMAL_AGENT_TRANSPORT_TREATMENTS
        ),
        "codex": codex,
        "runtime": comparison_runtime,
        "evaluator_files_sha256": evaluator,
        "tasks": tasks,
        "run_config": run_config,
    }
    return {
        "schema": "aka.matched-campaign/v1",
        "formal_execution": formal_execution,
        "formal_execution_sha256": campaign.FORMAL_LIVE_EXECUTION_SHA256,
        "comparison_contract_sha256": _digest(comparison),
        "comparison_contract": comparison,
        "repositories": repositories,
        "agent": agent,
        "runtime": runtime,
        "evaluator_files_sha256": evaluator,
        "configuration": {
            "run_config_path": str(run_config_path.resolve()),
            "run_config_sha256": campaign._sha256_file(run_config_path),
            "run_config_size_bytes": run_config_path.stat().st_size,
            "run_config_contract": run_config,
            "tasks": tasks,
        },
    }


def _make_formal_run(
    tmp_path: Path, task_names: tuple[str, ...] = ("task_a", "task_b")
) -> tuple[Path, list[dict[str, object]], list[dict[str, object]]]:
    run = tmp_path / "run"
    run.mkdir()
    tasks: list[dict[str, object]] = []
    mappings: list[dict[str, object]] = []
    devices: list[dict[str, object]] = []
    for index, task_name in enumerate(task_names, 1):
        package = tmp_path / f"package_{index}"
        package.mkdir()
        config = package / "config.yaml"
        config.write_text(f"task_name: {task_name}\n", encoding="utf-8")
        (package / "kernel.py").write_text(
            f"VALUE = {index}\n", encoding="utf-8"
        )
        files = campaign._regular_tree_manifest(package)
        tasks.append(
            {
                "task_index": index,
                "task_name": task_name,
                "config_path": str(config.resolve()),
                "config_sha256": campaign._sha256_file(config),
                "package_files_sha256": files,
                "package_manifest_sha256": _digest(files),
            }
        )
        host_gpu = str(index - 1)
        mappings.append(
            {
                "task_index": index,
                "task_name": task_name,
                "assigned_host_gpu_id": host_gpu,
            }
        )
        devices.append(
            {
                "host_device_id": host_gpu,
                "unique_id": f"gpu-{host_gpu}",
                "render_nodes": [f"/dev/dri/renderD{128 + index - 1}"],
            }
        )
    run_config_path = tmp_path / "formal_run_config.yaml"
    run_config_path.write_text(
        yaml.safe_dump(
            {
                "agent": {"template": "apex"},
                "campaign": _policy(),
                "tasks": list(task_names),
                "target_gpu_model": "MI355X",
                "log_directory": "/test/logs",
                "workspace_directory_prefix": "/test/workspace",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    durable_run_config_path = campaign._materialize_durable_run_config(
        run, run_config_path
    )
    run_config = campaign._run_config_contract(
        durable_run_config_path, agent_name="apex"
    )
    manifest = _formal_v6_contract(
        tasks=tasks,
        mappings=mappings,
        devices=devices,
        run_config_path=durable_run_config_path,
        run_config=run_config,
    )
    manifest_path = run / "campaign_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    manifest_path.chmod(0o444)
    for state in aka_main.QUEUE_STATES:
        (run / aka_main.QUEUE_DIR_NAME / state).mkdir(parents=True)
    return run, tasks, mappings


def _pending_descriptor(
    run: Path,
    task: dict[str, object],
    mapping: dict[str, object],
    *,
    total_tasks: int,
) -> Path:
    index = int(task["task_index"])
    task_name = str(task["task_name"])
    path = (
        run
        / aka_main.QUEUE_DIR_NAME
        / "pending"
        / aka_main._descriptor_name(index, task_name)
    )
    aka_main._write_descriptor(
        path,
        {
            "index": index,
            "total_tasks": total_tasks,
            "task_name": task_name,
            "task_config_dir": task["config_path"],
            "workspace_path": str(run / f"workspace_{index}"),
            "assigned_host_gpu_id": mapping["assigned_host_gpu_id"],
            "status": "pending",
        },
        no_clobber=True,
    )
    path.chmod(0o444)
    return path


def test_formal_claim_rejects_descriptor_payload_swapped_between_task_names(
    tmp_path: Path,
) -> None:
    run, tasks, mappings = _make_formal_run(tmp_path)
    first = _pending_descriptor(run, tasks[0], mappings[0], total_tasks=2)
    second = _pending_descriptor(run, tasks[1], mappings[1], total_tasks=2)
    first_bytes = first.read_bytes()
    second_bytes = second.read_bytes()
    first.chmod(0o600)
    second.chmod(0o600)
    first.write_bytes(second_bytes)
    second.write_bytes(first_bytes)
    first.chmod(0o444)
    second.chmod(0o444)

    with pytest.raises(campaign.CampaignError, match="filename differs"):
        aka_main.claim_next_descriptor(
            run, "0", logging.getLogger(__name__), host_gpu_id="0"
        )

    assert len(list((run / aka_main.QUEUE_DIR_NAME / "pending").glob("*.yaml"))) == 2
    assert not list((run / aka_main.QUEUE_DIR_NAME / "running").glob("*.yaml"))


def test_task_package_mutation_is_detected_immediately_after_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, tasks, mappings = _make_formal_run(tmp_path, ("task_a",))
    monkeypatch.setenv("AGENT_KERNEL_ARENA_HOST_GPU_ID", "0")
    calls = 0

    def mutate_package(**_kwargs):
        nonlocal calls
        calls += 1
        config = Path(str(tasks[0]["config_path"]))
        (config.parent / "kernel.py").write_text("VALUE = 999\n", encoding="utf-8")
        return False, None

    with pytest.raises(campaign.CampaignError, match="package bytes differ"):
        campaign.run_matched_task_campaign(
            eval_config={
                "campaign": _policy(),
                "assigned_host_gpu_id": mappings[0]["assigned_host_gpu_id"],
            },
            agent=SimpleNamespace(value="test"),
            agent_launcher=None,
            task_name="task_a",
            task_config_dir=str(tasks[0]["config_path"]),
            run_directory=run,
            timestamp="20260101_000000",
            logger=logging.getLogger(__name__),
            task_index=1,
            total_tasks=1,
            single_attempt=mutate_package,
            clock=lambda: 0.0,
        )

    assert calls == 1


def _gpu_receipt(host_gpu: str) -> dict[str, object]:
    index = int(host_gpu)
    return {
        "gpu": {
            "policy": "physical_device_boundary_with_host_exclusivity_v1",
            "plan_sha256": "a" * 64,
            "boundary_receipt_sha256": "c" * 64,
            "exclusivity_receipt_sha256": "b" * 64,
            "exclusivity_verified": True,
            "host_gpu_id": host_gpu,
            "unique_id": f"gpu-{host_gpu}",
            "allowed_render_nodes": [f"/dev/dri/renderD{128 + index}"],
            "runtime_identity": {
                "visible_physical_gpu_count": 1,
                "rocm_smi_identity": {"unique_id": f"gpu-{host_gpu}"},
                "torch": {"device_count": 1},
            },
        }
    }


def test_gpu_receipt_must_match_the_current_tasks_assigned_gpu(tmp_path: Path) -> None:
    run, _, _ = _make_formal_run(tmp_path)
    receipt = _gpu_receipt("1")

    assert campaign._gpu_receipt_errors(
        receipt, run, expected_task_name="task_b"
    ) == []
    assert campaign._gpu_receipt_errors(
        receipt, run, expected_task_name="task_a"
    ) == ["attempt_gpu_boundary_or_exclusivity_mismatch"]


def test_concurrent_formal_claim_has_exactly_one_consumer_and_no_hardlink(
    tmp_path: Path,
) -> None:
    run, tasks, mappings = _make_formal_run(tmp_path, ("task_a",))
    _pending_descriptor(run, tasks[0], mappings[0], total_tasks=1)
    barrier = threading.Barrier(2)
    results: list[Path | None] = []
    errors: list[BaseException] = []

    def claim(worker_id: str) -> None:
        try:
            barrier.wait()
            results.append(
                aka_main.claim_next_descriptor(
                    run,
                    worker_id,
                    logging.getLogger(__name__),
                    host_gpu_id="0",
                )
            )
        except BaseException as error:  # recorded and asserted in the main thread
            errors.append(error)

    threads = [threading.Thread(target=claim, args=(worker,)) for worker in ("0", "1")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert errors == []
    claimed = [path for path in results if path is not None]
    assert len(claimed) == 1
    assert results.count(None) == 1
    assert not list((run / aka_main.QUEUE_DIR_NAME / "pending").glob("*.yaml"))
    running = list((run / aka_main.QUEUE_DIR_NAME / "running").glob("*.yaml"))
    assert running == claimed
    assert os.lstat(running[0]).st_nlink == 1


def test_formal_claim_tolerates_descriptor_disappearing_before_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, tasks, mappings = _make_formal_run(tmp_path, ("task_a",))
    pending = _pending_descriptor(run, tasks[0], mappings[0], total_tasks=1)
    running = run / aka_main.QUEUE_DIR_NAME / "running"
    claimed_by_other = running / f"worker_other__{pending.name}"
    original_snapshot = aka_main._formal_descriptor_snapshot
    simulated = False

    def simulate_competing_claim(path: Path):
        nonlocal simulated
        if not simulated and path == pending:
            simulated = True
            os.rename(path, claimed_by_other)
        return original_snapshot(path)

    monkeypatch.setattr(
        aka_main, "_formal_descriptor_snapshot", simulate_competing_claim
    )

    assert (
        aka_main.claim_next_descriptor(
            run, "0", logging.getLogger(__name__), host_gpu_id="0"
        )
        is None
    )
    assert claimed_by_other.exists()
    assert not pending.exists()


def test_formal_claim_tolerates_descriptor_disappearing_before_atomic_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, tasks, mappings = _make_formal_run(tmp_path, ("task_a",))
    pending = _pending_descriptor(run, tasks[0], mappings[0], total_tasks=1)
    running = run / aka_main.QUEUE_DIR_NAME / "running"
    claimed_by_other = running / f"worker_other__{pending.name}"
    original_rename = aka_main.os.rename
    simulated = False

    def simulate_competing_rename(source: Path, destination: Path):
        nonlocal simulated
        if not simulated and Path(source) == pending:
            simulated = True
            original_rename(source, claimed_by_other)
        return original_rename(source, destination)

    monkeypatch.setattr(aka_main.os, "rename", simulate_competing_rename)

    assert (
        aka_main.claim_next_descriptor(
            run, "0", logging.getLogger(__name__), host_gpu_id="0"
        )
        is None
    )
    assert claimed_by_other.exists()
    assert not pending.exists()


def test_formal_claim_fails_closed_if_claim_disappears_after_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, tasks, mappings = _make_formal_run(tmp_path, ("task_a",))
    pending = _pending_descriptor(run, tasks[0], mappings[0], total_tasks=1)
    original_snapshot = aka_main._formal_descriptor_snapshot

    def simulate_post_claim_removal(path: Path):
        if path.parent.name == "running":
            path.unlink()
        return original_snapshot(path)

    monkeypatch.setattr(
        aka_main, "_formal_descriptor_snapshot", simulate_post_claim_removal
    )

    with pytest.raises(
        campaign.CampaignError,
        match="claimed descriptor disappeared before verification",
    ):
        aka_main.claim_next_descriptor(
            run, "0", logging.getLogger(__name__), host_gpu_id="0"
        )

    assert not pending.exists()


def test_failed_descriptor_roundtrip_preserves_parser_worker_identity(
    tmp_path: Path,
) -> None:
    run, tasks, mappings = _make_formal_run(tmp_path, ("task_a",))
    _pending_descriptor(run, tasks[0], mappings[0], total_tasks=1)
    claimed = aka_main.claim_next_descriptor(
        run, "0", logging.getLogger(__name__), host_gpu_id="0"
    )
    assert claimed is not None

    manifest_path = run / "campaign_manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    task_evidence = {
        "schema": "aka.matched-task-attempts/v1",
        "task_name": "task_a",
        "campaign_manifest_sha256": aka_main._sha256_file(manifest_path),
        "comparison_contract_sha256": manifest["comparison_contract_sha256"],
        "campaign_manifest_unchanged": True,
        "all_attempts_centrally_evaluated": False,
        "all_agent_sessions_succeeded": False,
        "within_evaluator_allowance": True,
        "within_task_timeout": True,
        "attempts": [],
        "selected_attempt": None,
    }
    task_evidence["failure_reasons"] = campaign._campaign_failure_reasons(
        task_evidence
    )
    evidence_path = (
        run
        / ".campaign_attempts"
        / campaign.campaign_task_path_component("task_a")
        / "task_campaign.yaml"
    )
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text(yaml.safe_dump(task_evidence), encoding="utf-8")
    evidence_path.chmod(0o444)

    aka_main.finish_descriptor(
        claimed,
        "failed",
        workspace_path=None,
        worker_id="0",
        failure_reason="formal_task_not_canonical",
    )

    marker = next((run / aka_main.QUEUE_DIR_NAME / "failed").glob("*.yaml"))
    expected_suffix = aka_main._descriptor_name(1, "task_a")
    parsed = re.fullmatch(
        rf"worker_([A-Za-z0-9._-]+)__{re.escape(expected_suffix)}",
        marker.name,
    )
    payload = yaml.safe_load(marker.read_text(encoding="utf-8"))
    assert parsed is not None
    assert parsed.group(1) == payload["worker_id"] == "0"
    assert payload["failure"]["primary_reason"] == "formal_task_not_canonical"
    assert os.lstat(marker).st_nlink == 1
