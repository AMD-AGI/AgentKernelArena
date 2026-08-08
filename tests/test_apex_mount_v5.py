import hashlib
import json
import os
import stat
from pathlib import Path

import pytest
import yaml

from src import campaign
from src import campaign_isolation
from src import apex_runtime


def _digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _repository(runtime_digest: str = "c" * 64) -> dict:
    return {
        "commit": "a" * 40,
        "dirty": False,
        "status_sha256": "b" * 64,
        "runtime_manifest_sha256": runtime_digest,
    }


def _v5_manifest(runtime_digest: str = "c" * 64) -> dict:
    repositories = {
        "agent_kernel_arena": {
            "commit": "d" * 40,
            "dirty": False,
            "status_sha256": "e" * 64,
        },
        "apex": _repository(runtime_digest),
    }
    agent = {
        "template": "apex",
        "session_receipt_schema": "agentkernelarena.apex-attempt-receipt/v5",
        "apex_runtime_mount_policy_id": (
            campaign_isolation.APEX_RUNTIME_MOUNT_POLICY
        ),
        "attempt_mount_receipt_schema": (
            campaign_isolation.ATTEMPT_MOUNT_RECEIPT_SCHEMA
        ),
        "apex_runtime_mount_schema": campaign_isolation.APEX_RUNTIME_MOUNT_SCHEMA,
        "runtime_manifest_sha256": runtime_digest,
    }
    comparison = {
        "schema": "aka.apex-vs-codex-comparison-contract/v5",
        "objective_policy_id": (
            "aka.task-package-objective-and-protected-harness/v1"
        ),
        "prompt_policy_id": (
            "aka.shared-objective-backend-native-context-receipted/v1"
        ),
        "candidate_persistence_policy_id": (
            campaign.CANDIDATE_PERSISTENCE_POLICY
        ),
        "boundary_quiescence_policy_id": campaign.BOUNDARY_QUIESCENCE_POLICY,
        "agent_process_containment_policy_id": (
            campaign.AGENT_PROCESS_CONTAINMENT_POLICY
        ),
        "attempt_containment_policy_id": (
            campaign_isolation.ATTEMPT_CONTAINMENT_POLICY
        ),
        "repositories": repositories,
        "apex_treatment": dict(agent),
    }
    return {
        "schema": "aka.matched-campaign/v1",
        "repositories": repositories,
        "agent": agent,
        "comparison_contract": comparison,
        "comparison_contract_sha256": _digest(comparison),
    }


def _write_manifest(run: Path, manifest: dict) -> Path:
    path = run / "campaign_manifest.yaml"
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    path.chmod(0o444)
    return path


def test_v5_comparison_binds_repository_mount_policy_and_runtime_digest(
    tmp_path: Path,
) -> None:
    manifest = _v5_manifest()
    _write_manifest(tmp_path, manifest)

    loaded = campaign._load_verified_campaign_manifest(tmp_path)

    comparison = loaded["comparison_contract"]
    assert campaign._expected_session_receipt_schema(tmp_path) == (
        "agentkernelarena.apex-attempt-receipt/v5"
    )
    assert comparison["repositories"] == loaded["repositories"]
    assert comparison["apex_treatment"] == {
        "template": "apex",
        "session_receipt_schema": "agentkernelarena.apex-attempt-receipt/v5",
        "apex_runtime_mount_policy_id": (
            campaign_isolation.APEX_RUNTIME_MOUNT_POLICY
        ),
        "attempt_mount_receipt_schema": (
            campaign_isolation.ATTEMPT_MOUNT_RECEIPT_SCHEMA
        ),
        "apex_runtime_mount_schema": campaign_isolation.APEX_RUNTIME_MOUNT_SCHEMA,
        "runtime_manifest_sha256": "c" * 64,
    }

    codex_manifest = _v5_manifest()
    codex_manifest["agent"] = {
        "template": "codex",
        "session_receipt_schema": "agentkernelarena.codex-attempt-receipt/v4",
    }
    codex_run = tmp_path / "codex"
    codex_run.mkdir()
    _write_manifest(codex_run, codex_manifest)
    loaded_codex = campaign._load_verified_campaign_manifest(codex_run)
    assert campaign._expected_session_receipt_schema(codex_run) == (
        "agentkernelarena.codex-attempt-receipt/v4"
    )
    assert (
        loaded_codex["comparison_contract_sha256"]
        == loaded["comparison_contract_sha256"]
    )


def test_apex_repository_requires_runner_runtime_manifest_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_KERNEL_ARENA_APEX_COMMIT", "a" * 40)
    monkeypatch.setenv("AGENT_KERNEL_ARENA_APEX_DIRTY", "false")
    monkeypatch.setenv("AGENT_KERNEL_ARENA_APEX_STATUS_SHA256", "b" * 64)
    monkeypatch.setenv(
        "AGENT_KERNEL_ARENA_APEX_RUNTIME_MANIFEST_SHA256", "c" * 64
    )

    assert campaign._apex_state_from_environment() == _repository()

    monkeypatch.delenv("AGENT_KERNEL_ARENA_APEX_RUNTIME_MANIFEST_SHA256")
    with pytest.raises(campaign.CampaignError, match="runtime manifest digest"):
        campaign._apex_state_from_environment()


def test_comparison_v5_digest_is_identical_for_apex_and_codex_arms() -> None:
    policy = campaign.CampaignPolicy(
        comparison="apex_vs_codex",
        attempts=3,
        attempt_timeout_seconds=3600,
        apex_internal_allowance_seconds=3600,
        task_timeout_seconds=25200,
        evaluator_allowance_seconds=3600,
        selection_policy="correctness_then_measured_rate_v1",
        workspace_policy="fresh_per_attempt",
        gpu_policy="deterministic_task_gpu_v1",
        require_clean_checkouts=True,
    )
    common = {
        "backend": "codex",
        "model": "gpt-5.5",
        "effort": "xhigh",
        "permission_mode": "workspace_write_isolated",
        "inner_max_iterations": 1,
        "attempt_timeout_seconds": 3600,
        "max_turns": 50,
        "turn_policy": campaign.TURN_POLICY,
        "agent_process_containment_policy_id": (
            campaign.AGENT_PROCESS_CONTAINMENT_POLICY
        ),
        "attempt_containment_policy_id": (
            campaign_isolation.ATTEMPT_CONTAINMENT_POLICY
        ),
        "structured_stream_output_limit_bytes": 16 * 1024 * 1024,
        "codex_version": "codex test",
        "codex_binary_sha256": "f" * 64,
        "isolation": {},
    }
    apex = {
        **common,
        "template": "apex",
        "session_receipt_schema": "agentkernelarena.apex-attempt-receipt/v5",
        "apex_runtime_mount_policy_id": (
            campaign_isolation.APEX_RUNTIME_MOUNT_POLICY
        ),
        "attempt_mount_receipt_schema": (
            campaign_isolation.ATTEMPT_MOUNT_RECEIPT_SCHEMA
        ),
        "apex_runtime_mount_schema": campaign_isolation.APEX_RUNTIME_MOUNT_SCHEMA,
        "runtime_manifest_sha256": "c" * 64,
    }
    codex = {
        **common,
        "template": "codex",
        "session_receipt_schema": "agentkernelarena.codex-attempt-receipt/v4",
    }
    arguments = {
        "policy": policy,
        "measurement": {},
        "repositories": {
            "agent_kernel_arena": {
                "commit": "d" * 40,
                "dirty": False,
                "status_sha256": "e" * 64,
            },
            "apex": _repository(),
        },
        "runtime": {},
        "evaluator": {},
        "tasks": [],
    }

    apex_contract = campaign._comparison_contract(agent=apex, **arguments)
    codex_contract = campaign._comparison_contract(agent=codex, **arguments)

    assert apex_contract == codex_contract
    assert _digest(apex_contract) == _digest(codex_contract)


@pytest.mark.parametrize(
    "tamper", ["repository", "marker", "receipt", "schema", "combined"]
)
def test_v5_manifest_divergence_and_downgrade_fail_closed(
    tmp_path: Path, tamper: str
) -> None:
    manifest = _v5_manifest()
    if tamper == "repository":
        manifest["repositories"] = {
            **manifest["repositories"],
            "apex": {**manifest["repositories"]["apex"], "commit": "f" * 40},
        }
    elif tamper == "marker":
        del manifest["agent"]["apex_runtime_mount_policy_id"]
    elif tamper == "receipt":
        manifest["agent"]["session_receipt_schema"] = (
            "agentkernelarena.apex-attempt-receipt/v4"
        )
    elif tamper == "schema":
        comparison = manifest["comparison_contract"]
        comparison["schema"] = "aka.apex-vs-codex-comparison-contract/v4"
        manifest["comparison_contract_sha256"] = _digest(comparison)
    else:
        manifest["agent"] = {
            "template": "apex",
            "session_receipt_schema": (
                "agentkernelarena.apex-attempt-receipt/v4"
            ),
        }
        comparison = manifest["comparison_contract"]
        comparison["schema"] = "aka.apex-vs-codex-comparison-contract/v4"
        comparison.pop("apex_treatment")
        manifest["comparison_contract_sha256"] = _digest(comparison)
    _write_manifest(tmp_path, manifest)

    with pytest.raises(campaign.CampaignError, match="comparison contract"):
        campaign._load_verified_campaign_manifest(tmp_path)
    assert campaign._expected_session_receipt_schema(tmp_path) is None
    assert campaign._expected_comparison_contract_sha256(tmp_path) is None


@pytest.mark.parametrize("generation", [1, 2, 3, 4])
def test_v1_through_v4_history_never_acquires_mount_semantics(
    tmp_path: Path, generation: int
) -> None:
    comparison = {
        "schema": f"aka.apex-vs-codex-comparison-contract/v{generation}",
        "objective_policy_id": (
            "aka.task-package-objective-and-protected-harness/v1"
        ),
        "prompt_policy_id": (
            "aka.shared-objective-backend-native-context-receipted/v1"
        ),
    }
    if generation >= 2:
        comparison["candidate_persistence_policy_id"] = (
            campaign.CANDIDATE_PERSISTENCE_POLICY
        )
    if generation == 3:
        comparison["boundary_quiescence_policy_id"] = (
            campaign.BOUNDARY_QUIESCENCE_POLICY
        )
    if generation == 4:
        comparison.update(
            {
                "agent_process_containment_policy_id": (
                    campaign.AGENT_PROCESS_CONTAINMENT_POLICY
                ),
                "attempt_containment_policy_id": (
                    campaign_isolation.ATTEMPT_CONTAINMENT_POLICY
                ),
            }
        )
    agent = {
        "template": "apex",
        "session_receipt_schema": (
            f"agentkernelarena.apex-attempt-receipt/v{generation}"
        ),
    }
    if generation == 4:
        # This transitional marker existed briefly, but v4 never promised
        # role-complete mount evidence.
        agent["apex_runtime_mount_policy_id"] = (
            campaign_isolation.APEX_RUNTIME_MOUNT_POLICY
        )
    manifest = {
        "schema": "aka.matched-campaign/v1",
        "agent": agent,
        "repositories": {
            "apex": {
                "commit": "a" * 40,
                "dirty": False,
                "status_sha256": "b" * 64,
            }
        },
        "comparison_contract": comparison,
        "comparison_contract_sha256": _digest(comparison),
    }
    _write_manifest(tmp_path, manifest)

    assert campaign._load_verified_campaign_manifest(tmp_path) == manifest
    assert campaign._expected_apex_runtime_mount(tmp_path) is None
    assert campaign._apex_runtime_mount_errors({}, tmp_path) == []
    assert campaign.resolve_session_receipt_schema(
        "apex", "agentkernelarena.apex-attempt-receipt/v5"
    ) == "agentkernelarena.apex-attempt-receipt/v5"


def _mount_identity(path: Path) -> dict:
    metadata = path.lstat()
    return {
        "path": str(path),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": stat.S_IMODE(metadata.st_mode),
        "mount": {
            "mount_id": 10,
            "parent_id": 1,
            "major_minor": "0:1",
            "root": "/",
            "mount_point": "/",
        },
        "nested_mounts": [],
        "source": "o_path_nofollow_bind_fd",
    }


def _role_fixture(tmp_path: Path) -> tuple[dict, dict, Path, Path, Path]:
    data = tmp_path / "campaign-data"
    attempt = data / "run/.campaign_attempts/task/attempt_01"
    workspace = attempt / "workspace"
    artifacts = attempt / ".workspace_apex/run"
    contract = attempt / ".workspace_apex/run.contract"
    home = attempt / ".agent-home"
    runtime = attempt / ".workspace_apex/run.runtime"
    for path in (workspace, artifacts, contract, home, runtime):
        path.mkdir(parents=True, exist_ok=True)
    contract_path = contract / "task_spec.json"
    contract_path.write_text("{}\n", encoding="utf-8")
    receipt_path = attempt / "session_receipt.json"
    receipt_path.write_text("{}\n", encoding="utf-8")
    paths = {
        "apex_artifacts": artifacts,
        "backend_home": home,
        "scored_workspace": workspace,
        "sealed_task_contract": contract,
        "apex_runtime": runtime,
    }
    material = {
        "schema": campaign_isolation.ATTEMPT_MOUNT_RECEIPT_SCHEMA,
        "campaign_data_root": str(data),
        "campaign_data_root_hidden": True,
        "campaign_data_identity": _mount_identity(data),
        "roles": {
            "persistent_writable": {
                role: _mount_identity(paths[role])
                for role in ("apex_artifacts", "backend_home")
            },
            "read_only": {
                role: _mount_identity(paths[role])
                for role in (
                    "scored_workspace",
                    "sealed_task_contract",
                    "apex_runtime",
                )
            },
            "private_tmpfs": {
                "tmp": {"path": "/tmp", "persistence": "private"},
                "dev_shm": {"path": "/dev/shm", "persistence": "private"},
            },
        },
    }
    receipt = {"attempt_mounts": {**material, "sha256": _digest(material)}}
    task_spec = {"workspace": str(workspace), "results_dir": str(artifacts)}
    return receipt, task_spec, receipt_path, contract_path, runtime


def test_v5_mount_roles_are_closed_canonical_and_non_overlapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, task_spec, receipt_path, contract_path, runtime = _role_fixture(
        tmp_path
    )
    workspace = Path(task_spec["workspace"])
    monkeypatch.setenv(
        "AGENT_KERNEL_ARENA_CAMPAIGN_DATA_ROOT",
        str(tmp_path / "campaign-data"),
    )

    assert campaign._apex_attempt_mount_role_errors(
        receipt=receipt,
        receipt_path=receipt_path,
        workspace=workspace,
        task_spec=task_spec,
        contract_path=contract_path,
        runtime_root=runtime,
    ) == []

    roles = receipt["attempt_mounts"]["roles"]
    roles["persistent_writable"]["apex_artifacts"] = roles["read_only"][
        "scored_workspace"
    ]
    task_spec["results_dir"] = task_spec["workspace"]
    material = dict(receipt["attempt_mounts"])
    material.pop("sha256")
    receipt["attempt_mounts"]["sha256"] = _digest(material)

    assert campaign._apex_attempt_mount_role_errors(
        receipt=receipt,
        receipt_path=receipt_path,
        workspace=workspace,
        task_spec=task_spec,
        contract_path=contract_path,
        runtime_root=runtime,
    ) == ["apex_attempt_mount_role_contract_mismatch"]


def test_v5_mount_roles_reject_private_tmpfs_or_role_set_tampering(
    tmp_path: Path,
) -> None:
    receipt, task_spec, receipt_path, contract_path, runtime = _role_fixture(
        tmp_path
    )
    mounts = receipt["attempt_mounts"]
    mounts["roles"]["private_tmpfs"]["tmp"] = {"path": "/tmp"}
    material = dict(mounts)
    material.pop("sha256")
    mounts["sha256"] = _digest(material)

    assert campaign._apex_attempt_mount_role_errors(
        receipt=receipt,
        receipt_path=receipt_path,
        workspace=Path(task_spec["workspace"]),
        task_spec=task_spec,
        contract_path=contract_path,
        runtime_root=runtime,
    ) == ["apex_attempt_mount_role_contract_mismatch"]


def test_v5_runtime_snapshot_binds_manifest_and_role_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, task_spec, receipt_path, contract_path, runtime_parent = (
        _role_fixture(tmp_path)
    )
    source_root = tmp_path / "apex-source"
    source_root.mkdir()
    source_main = source_root / "main.py"
    source_main.write_text("print('apex')\n", encoding="utf-8")
    system_python = Path("/usr/bin/python3").resolve(strict=True)
    source_identity = {
        "path": str(source_root),
        "device": source_root.stat().st_dev,
        "inode": source_root.stat().st_ino,
        "mode": stat.S_IMODE(source_root.stat().st_mode),
    }
    main_bytes = source_main.read_bytes()
    apex_root_material = {
        "role": "apex",
        "source": source_identity,
        "destination": "repo",
        "files": [
            {
                "path": "main.py",
                "type": "file",
                "mode": "100644",
                "size": len(main_bytes),
                "sha256": hashlib.sha256(main_bytes).hexdigest(),
            }
        ],
    }
    python_target = str(system_python)
    venv_root_material = {
        "role": "venv",
        "source": source_identity,
        "destination": "venv",
        "files": [
            {"path": "bin", "type": "directory", "mode": 0o755},
            {
                "path": "bin/python",
                "type": "symlink",
                "mode": 0o777,
                "target": python_target,
                "sha256": hashlib.sha256(os.fsencode(python_target)).hexdigest(),
                "resolved_target_class": "system_image",
            },
        ],
    }
    runtime_material_without_digest = {
        "schema": apex_runtime.RUNTIME_MANIFEST_SCHEMA,
        "policy_id": apex_runtime.RUNTIME_POLICY_ID,
        "git": {
            "commit": "a" * 40,
            "dirty": False,
            "status_sha256": "b" * 64,
            "index_shortcuts_rejected": True,
            "git_environment_sanitized": True,
        },
        "launcher": {
            "system_python": {
                "path": str(system_python),
                "binding": "formal_docker_image_plus_attempt_receipt_v1",
            }
        },
        "roots": [
            {**apex_root_material, "sha256": _digest(apex_root_material)},
            {**venv_root_material, "sha256": _digest(venv_root_material)},
        ],
        "execution": {
            "interpreter": "venv/bin/python",
            "flags": ["-I", "-S"],
            "bootstrap": apex_runtime.RUNTIME_BOOTSTRAP_NAME,
            "bootstrap_policy_id": apex_runtime.RUNTIME_BOOTSTRAP_POLICY_ID,
            "bootstrap_sha256": apex_runtime.RUNTIME_BOOTSTRAP_SHA256,
            "entrypoint": "repo/main.py",
            "pythonpath": ["repo"],
            "pth_execution": False,
            "sitecustomize_execution": False,
        },
        "excluded_external_directories": [".git", ".hg", ".svn"],
    }
    runtime_digest = _digest(runtime_material_without_digest)
    runtime_manifest = {
        **runtime_material_without_digest,
        "sha256": runtime_digest,
    }
    manifest_bytes = (
        json.dumps(
            runtime_manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    runtime = runtime_parent / runtime_digest
    entrypoint = runtime / "repo/main.py"
    entrypoint.parent.mkdir(parents=True)
    (runtime / "venv/bin").mkdir(parents=True)
    entrypoint.write_bytes(main_bytes)
    entrypoint.chmod(0o444)
    os.symlink(python_target, runtime / "venv/bin/python")
    (runtime / "venv/bin").chmod(0o555)
    (runtime / "venv").chmod(0o555)
    (runtime / "repo").chmod(0o555)
    runtime_manifest_path = runtime / "runtime_manifest.json"
    runtime_manifest_path.write_bytes(manifest_bytes)
    runtime_manifest_path.chmod(0o444)
    bootstrap_path = runtime / apex_runtime.RUNTIME_BOOTSTRAP_NAME
    bootstrap_path.write_bytes(apex_runtime.RUNTIME_BOOTSTRAP)
    bootstrap_path.chmod(0o444)
    runtime.chmod(0o555)
    assert apex_runtime.verify_runtime_snapshot(runtime, runtime_digest) == (
        runtime_manifest
    )

    mounts = receipt["attempt_mounts"]
    mounts["roles"]["read_only"]["apex_runtime"] = _mount_identity(runtime)
    mount_material = dict(mounts)
    mount_material.pop("sha256")
    mounts["sha256"] = _digest(mount_material)

    run = tmp_path / "campaign-data/run"
    _write_manifest(run, _v5_manifest(runtime_digest))
    entrypoint_sha = hashlib.sha256(entrypoint.read_bytes()).hexdigest()
    runtime_material = {
        "schema": campaign_isolation.APEX_RUNTIME_MOUNT_SCHEMA,
        "policy_id": campaign_isolation.APEX_RUNTIME_MOUNT_POLICY,
        "mode": "read_only",
        "source_root": str(source_root),
        "root": str(runtime),
        "repository": _repository(runtime_digest),
        "runtime_manifest_sha256": runtime_digest,
        "runtime_manifest_path": str(runtime_manifest_path),
        "runtime_manifest_relative_path": "runtime_manifest.json",
        "entrypoint": {
            "path": str(entrypoint),
            "relative_path": "repo/main.py",
            "sha256": entrypoint_sha,
        },
        "python": {
            "source_launcher_relative_path": ".venv/bin/python",
            "launcher_path": str(runtime / "venv/bin/python"),
            "resolved_path": str(system_python),
            "resolved_sha256": hashlib.sha256(system_python.read_bytes()).hexdigest(),
            "flags": ["-I", "-S"],
            "pythonpath": [str(entrypoint.parent)],
            "environment": {
                "PYTHONNOUSERSITE": "1",
                "PYTHONSAFEPATH": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        },
        "attempt_mounts_sha256": mounts["sha256"],
    }
    receipt.update(
        {
            "schema": "agentkernelarena.apex-attempt-receipt/v5",
            "apex_runtime_mount": {
                **runtime_material,
                "sha256": _digest(runtime_material),
            },
            "apex": {
                "entrypoint": str(entrypoint),
                "entrypoint_sha256": entrypoint_sha,
                "python": str(system_python),
                "python_sha256": hashlib.sha256(system_python.read_bytes()).hexdigest(),
            },
        }
    )
    monkeypatch.setenv("APEX_ROOT", str(source_root))
    monkeypatch.setenv(
        "AGENT_KERNEL_ARENA_CAMPAIGN_DATA_ROOT",
        str(tmp_path / "campaign-data"),
    )

    arguments = {
        "receipt": receipt,
        "run_directory": run,
        "receipt_path": receipt_path,
        "workspace": Path(task_spec["workspace"]),
        "task_spec": task_spec,
        "contract_path": contract_path,
    }
    assert campaign._apex_attempt_mount_role_errors(
        receipt=receipt,
        receipt_path=receipt_path,
        workspace=Path(task_spec["workspace"]),
        task_spec=task_spec,
        contract_path=contract_path,
        runtime_root=runtime,
    ) == []
    assert campaign._apex_runtime_mount_errors(**arguments) == []

    receipt["apex_runtime_mount"]["python"]["flags"] = ["-S", "-P"]
    stale_policy_material = dict(receipt["apex_runtime_mount"])
    stale_policy_material.pop("sha256")
    receipt["apex_runtime_mount"]["sha256"] = _digest(stale_policy_material)
    assert campaign._apex_runtime_mount_errors(**arguments) == [
        "apex_runtime_mount_contract_mismatch"
    ]
    receipt["apex_runtime_mount"]["python"]["flags"] = ["-I", "-S"]
    current_policy_material = dict(receipt["apex_runtime_mount"])
    current_policy_material.pop("sha256")
    receipt["apex_runtime_mount"]["sha256"] = _digest(current_policy_material)
    assert campaign._apex_runtime_mount_errors(**arguments) == []

    bootstrap_path.chmod(0o644)
    bootstrap_path.write_bytes(b"raise RuntimeError('tampered bootstrap')\n")
    bootstrap_path.chmod(0o444)
    assert campaign._apex_runtime_mount_errors(**arguments) == [
        "apex_runtime_mount_contract_mismatch"
    ]
    bootstrap_path.chmod(0o644)
    bootstrap_path.write_bytes(apex_runtime.RUNTIME_BOOTSTRAP)
    bootstrap_path.chmod(0o444)
    assert campaign._apex_runtime_mount_errors(**arguments) == []

    runtime_manifest_path.chmod(0o644)
    runtime_manifest_path.write_text('{"files":["tampered"]}\n', encoding="utf-8")
    runtime_manifest_path.chmod(0o444)
    assert campaign._apex_runtime_mount_errors(**arguments) == [
        "apex_runtime_mount_contract_mismatch"
    ]
