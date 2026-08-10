import hashlib
import json
from pathlib import Path

import pytest
import yaml

from src import campaign, postprocessing
from src.tools import compare_runs


POLICY = {
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


def _source_config(path: Path) -> bytes:
    payload = (
        "agent:\n"
        "  template: codex\n"
        "campaign:\n"
        "  comparison: apex_vs_codex\n"
        "tasks:\n"
        "  - triton2triton/vllm/example\n"
    ).encode()
    path.write_bytes(payload)
    return payload


def test_manifest_uses_an_exact_durable_read_only_run_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "ephemeral-runtime-config.yaml"
    payload = _source_config(source)
    run = tmp_path / "run_20260808_000000_test"
    run.mkdir()

    monkeypatch.setattr(
        campaign,
        "_aka_state_from_environment",
        lambda _root: ({}, {}),
    )
    monkeypatch.setattr(
        campaign,
        "_apex_state_from_environment",
        lambda: {"runtime_manifest_sha256": "a" * 64},
    )
    monkeypatch.setattr(
        campaign,
        "_agent_manifest",
        lambda *_args, **_kwargs: {"template": "codex"},
    )
    monkeypatch.setattr(campaign, "_task_manifests", lambda _tasks: [])
    monkeypatch.setattr(campaign, "runtime_isolation_receipt", lambda: {})
    monkeypatch.setattr(campaign, "_image_manifest", lambda: {})
    monkeypatch.setattr(campaign, "_gpu_inventory", lambda *_args: {})
    monkeypatch.setattr(campaign, "_evaluator_manifest", lambda _state: {})
    monkeypatch.setattr(
        campaign,
        "_comparison_contract",
        lambda **kwargs: {"run_config": kwargs["run_config"]},
    )

    manifest_path = campaign.ensure_campaign_manifest(
        run_directory=run,
        eval_config={"campaign": POLICY},
        run_config_path=source,
        task_config_paths={},
        agent_name="codex",
    )

    assert manifest_path is not None
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    configuration = manifest["configuration"]
    durable = Path(configuration["run_config_path"])
    assert durable == run / ".formal-run-config/run_config.yaml"
    assert durable.read_bytes() == payload
    assert durable.stat().st_nlink == 1
    assert durable.stat().st_mode & 0o222 == 0
    assert durable.parent.stat().st_mode & 0o777 == 0o555
    assert configuration["run_config_size_bytes"] == len(payload)
    assert configuration["run_config_sha256"] == hashlib.sha256(payload).hexdigest()

    source.unlink()
    assert campaign._run_config_contract(durable, agent_name="codex") == (
        configuration["run_config_contract"]
    )


def _offline_manifest(run_config: Path) -> dict:
    run_config_contract = campaign._run_config_contract(
        run_config, agent_name="codex"
    )
    tasks = [{"task_index": 1, "task_name": "triton2triton/vllm/example"}]
    identity = {
        field: f"bound-{field}" for field in compare_runs._CODEX_IDENTITY_FIELDS
    }
    identity.update(
        {
            "cloud_config_bootstrap_schema": (
                campaign.CODEX_CLOUD_CONFIG_BOOTSTRAP_SCHEMA
            ),
            "cloud_config_bootstrap_policy": (
                campaign.CODEX_CLOUD_CONFIG_BOOTSTRAP_POLICY
            ),
            "cloud_config_bundle_sha256": "c" * 64,
            "cloud_config_host_runtime_closure_sha256": "d" * 64,
        }
    )
    agent = {
        "template": "codex",
        **identity,
        "cloud_config_initial_refresh_receipt_sha256": "e" * 64,
        "session_receipt_schema": "agentkernelarena.codex-attempt-receipt/v6",
    }
    repositories = {
        "apex": {"runtime_manifest_sha256": "a" * 64},
    }
    apex_treatment = {
        "template": "apex",
        "session_receipt_schema": "agentkernelarena.apex-attempt-receipt/v7",
        "apex_runtime_mount_policy_id": campaign.APEX_RUNTIME_MOUNT_POLICY,
        "attempt_mount_receipt_schema": campaign.ATTEMPT_MOUNT_RECEIPT_SCHEMA,
        "apex_runtime_mount_schema": campaign.APEX_RUNTIME_MOUNT_SCHEMA,
        "runtime_manifest_sha256": repositories["apex"][
            "runtime_manifest_sha256"
        ],
    }
    comparison = {
        "schema": compare_runs._COMPARISON_SCHEMA,
        "formal_execution": dict(campaign._FORMAL_LIVE_COMMITMENT),
        "formal_execution_sha256": campaign.FORMAL_LIVE_EXECUTION_SHA256,
        "objective_policy_id": compare_runs._OBJECTIVE_POLICY,
        "prompt_policy_id": compare_runs._PROMPT_POLICY,
        "repositories": repositories,
        "apex_treatment": apex_treatment,
        "tasks": tasks,
        "codex": identity,
        "run_config": run_config_contract,
    }
    comparison_sha256 = hashlib.sha256(
        json.dumps(comparison, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    payload = run_config.read_bytes()
    return {
        "schema": "aka.matched-campaign/v1",
        "formal_execution": dict(campaign._FORMAL_LIVE_COMMITMENT),
        "formal_execution_sha256": campaign.FORMAL_LIVE_EXECUTION_SHA256,
        "agent": agent,
        "repositories": repositories,
        "comparison_contract": comparison,
        "comparison_contract_sha256": comparison_sha256,
        "configuration": {
            "run_config_path": str(run_config),
            "run_config_sha256": hashlib.sha256(payload).hexdigest(),
            "run_config_size_bytes": len(payload),
            "run_config_contract": run_config_contract,
            "tasks": tasks,
        },
    }


def test_postprocessing_and_compare_reopen_config_after_source_unmount_and_reject_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "ephemeral-runtime-config.yaml"
    _source_config(source)
    run = tmp_path / "run_20260808_000000_test"
    run.mkdir()
    durable = campaign._materialize_durable_run_config(run, source)
    manifest = _offline_manifest(durable)
    manifest_path = run / "campaign_manifest.yaml"
    manifest_bytes = yaml.safe_dump(manifest, sort_keys=True).encode()
    manifest_path.write_bytes(manifest_bytes)
    manifest_path.chmod(0o444)
    source.unlink()

    monkeypatch.setattr(
        compare_runs, "_v7_manifest_bindings_valid", lambda *_args: True
    )
    monkeypatch.setattr(
        compare_runs.postprocessing,
        "_extract_run_metadata",
        lambda _run: {
            "timestamp": "20260808_000000",
            "agent": "codex",
            "target_gpu": "MI355X",
        },
    )

    cohort = postprocessing._load_formal_cohort(run)
    assert cohort is not None
    context = compare_runs._formal_manifest_context(run, manifest, manifest_bytes)
    assert context["task_names"] == ["triton2triton/vllm/example"]

    durable.chmod(0o644)
    durable.write_bytes(durable.read_bytes() + b"# tampered\n")
    with pytest.raises(ValueError, match="not digest-bound"):
        postprocessing._load_formal_cohort(run)
    with pytest.raises(ValueError, match="binding is invalid"):
        compare_runs._formal_manifest_context(run, manifest, manifest_bytes)


def test_durable_run_config_rejects_a_new_hard_link(
    tmp_path: Path,
) -> None:
    source = tmp_path / "ephemeral-runtime-config.yaml"
    _source_config(source)
    run = tmp_path / "run_20260808_000000_test"
    run.mkdir()
    durable = campaign._materialize_durable_run_config(run, source)
    (tmp_path / "second-name.yaml").hardlink_to(durable)

    with pytest.raises(campaign.CampaignError, match="safe bounded regular file"):
        campaign._materialize_durable_run_config(run, source)
