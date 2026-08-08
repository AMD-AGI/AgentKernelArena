import hashlib
import importlib
import json
import os
import stat
import copy
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src import campaign
from src import campaign_isolation
from src import apex_runtime
from src import aka_runtime
from src import immutable_runtime_mount


apex_launcher = importlib.import_module("agents.apex.launch_agent")


def _digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


_REQUESTED_MOUNT_OPTIONS = [
    "ro",
    "nodev",
    "nosuid",
    "default_permissions",
    "allow_other",
    "subtype=squashfuse",
]


def _host_access_policy(private_ancestor: Path) -> dict:
    owner = {"uid": os.getuid(), "gid": os.getgid()}
    material = {
        "schema": aka_runtime.HOST_ACCESS_POLICY_SCHEMA,
        "policy_id": aka_runtime.HOST_ACCESS_POLICY_ID,
        "requested_mount_options": list(_REQUESTED_MOUNT_OPTIONS),
        "private_ancestor": {
            "path": str(private_ancestor),
            "device": 7,
            "inode": 11,
            "uid": owner["uid"],
            "gid": owner["gid"],
            "mode": 0o700,
        },
        "fuse_config": {
            "path": "/etc/fuse.conf",
            "device": 8,
            "inode": 12,
            "uid": 0,
            "gid": 0,
            "mode": 0o644,
            "nlink": 1,
            "size_bytes": 17,
            "sha256": "c" * 64,
            "user_allow_other": True,
        },
        "mount_owner": owner,
        "worker": dict(owner),
        "docker_daemon": {
            "uid": 0,
            "trusted_boundary": True,
            "access_via": "fuse_allow_other_with_private_ancestor_v1",
        },
    }
    return {**material, "sha256": _digest(material)}


def _runtime_service_evidence(
    snapshot: Path,
    manifest: dict,
    mount: dict,
    image_sha256: str = "f" * 64,
) -> dict:
    host_policy = _host_access_policy(snapshot.parent)
    image_input_sha256 = apex_runtime.runtime_image_inputs(
        snapshot, manifest
    )["sha256"]
    receipt_material = {
        "schema": immutable_runtime_mount.MOUNT_RECEIPT_SCHEMA,
        "policy_id": apex_runtime.RUNTIME_IMMUTABLE_MOUNT_POLICY_ID,
        "root": str(snapshot),
        "runtime_manifest_sha256": manifest["sha256"],
        "runtime_image_input_sha256": image_input_sha256,
        "image_sha256": image_sha256,
        "backing": {
            "kind": "sealed_memfd",
            "seals": list(immutable_runtime_mount._SEAL_NAMES),
        },
        "requested_mount_options": list(_REQUESTED_MOUNT_OPTIONS),
        "host_access_policy": host_policy,
        "mount": mount,
    }
    host_receipt = {**receipt_material, "sha256": _digest(receipt_material)}
    engine_material = {
        "schema": aka_runtime.ENGINE_EVIDENCE_SCHEMA,
        "policy_id": apex_runtime.RUNTIME_IMMUTABLE_MOUNT_POLICY_ID,
        "receipt_sha256": host_receipt["sha256"],
        "runtime_image_input_sha256": image_input_sha256,
        "image": {
            "size_bytes": 4096,
            "sha256": image_sha256,
            "memfd_seals": list(immutable_runtime_mount._SEAL_NAMES),
        },
        "tools": {
            "mksquashfs": {"path": "/usr/bin/mksquashfs", "sha256": "e" * 64},
            "squashfuse": {"path": "/usr/bin/squashfuse", "sha256": "f" * 64},
        },
        "requested_mount_options": list(_REQUESTED_MOUNT_OPTIONS),
        "host_access_policy_sha256": host_policy["sha256"],
        "process": {"pid": 101, "starttime": 202, "foreground": True},
        "mountpoint_source": {
            "path": str(snapshot),
            "device": 7,
            "inode": 13,
            "uid": os.getuid(),
            "gid": os.getgid(),
            "mode": 0o555,
        },
        "mount": mount,
        "inventory_verification": {
            "entry_count": 2,
            "inventory_sha256": image_input_sha256,
        },
        "write_probe_errno": 30,
    }
    engine = {**engine_material, "sha256": _digest(engine_material)}
    service_material = {
        "schema": aka_runtime.ENGINE_SERVICE_SCHEMA,
        "policy_id": aka_runtime.ENGINE_SERVICE_POLICY,
        "ready_path": str(snapshot.parent / "runtime-service-ready.json"),
        "service": {
            "pid": 303,
            "starttime": 404,
            "owner": dict(host_policy["mount_owner"]),
            "accepted_signals": ["SIGINT", "SIGTERM"],
            "engine_process": {"pid": 101, "starttime": 202},
        },
        "mount_receipt": host_receipt,
        "engine_evidence": engine,
    }
    return {**service_material, "sha256": _digest(service_material)}


def _repository(runtime_digest: str = "c" * 64) -> dict:
    return {
        "commit": "a" * 40,
        "dirty": False,
        "status_sha256": "b" * 64,
        "runtime_manifest_sha256": runtime_digest,
    }


def _backend_closure() -> dict:
    return {
        "schema": "aka.backend-runtime-closure/v1",
        "backend": "codex",
        "launcher": {},
        "interpreter": None,
        "components": [],
        "closure_sha256": "9" * 64,
    }


def _run_config_contract() -> dict:
    effective_config = {
        "campaign": {"comparison": "apex_vs_codex", "attempts": 3},
        "tasks": ["task"],
        "target_gpu_model": "MI355X",
        "log_directory": "/test/logs",
        "workspace_directory_prefix": "/test/workspace",
    }
    return {
        "schema": "aka.formal-run-config/v1",
        "effective_config": effective_config,
        "effective_config_sha256": _digest(effective_config),
    }


def _v6_manifest(runtime_digest: str = "c" * 64) -> dict:
    tasks = [{"task_index": 1, "task_name": "task"}]
    repositories = {
        "agent_kernel_arena": {
            "commit": "d" * 40,
            "tree": "1" * 40,
            "dirty": False,
            "status_sha256": "e" * 64,
            "execution_manifest_schema": "aka.execution-snapshot-manifest/v1",
            "execution_manifest_sha256": "2" * 64,
            "git_evidence_policy_id": "head_tree_direct_bytes_no_filters_v1",
        },
        "apex": _repository(runtime_digest),
    }
    agent = {
        "template": "apex",
        "session_receipt_schema": "agentkernelarena.apex-attempt-receipt/v6",
        "apex_runtime_mount_policy_id": (
            campaign_isolation.APEX_RUNTIME_MOUNT_POLICY
        ),
        "attempt_mount_receipt_schema": (
            campaign_isolation.ATTEMPT_MOUNT_RECEIPT_SCHEMA
        ),
        "apex_runtime_mount_schema": campaign_isolation.APEX_RUNTIME_MOUNT_SCHEMA,
        "runtime_manifest_sha256": runtime_digest,
        "backend_runtime_closure_schema": "aka.backend-runtime-closure/v1",
        "backend_runtime_closure_sha256": "9" * 64,
        "backend_runtime_closure": _backend_closure(),
        "max_process_output_bytes": campaign.FORMAL_AGENT_TRANSPORT_TREATMENTS[
            "apex"
        ]["max_process_output_bytes"],
        "structured_stream_output_limit_bytes": (
            campaign.FORMAL_AGENT_TRANSPORT_TREATMENTS["apex"][
                "structured_stream_output_limit_bytes"
            ]
        ),
        "structured_stream_overflow_policy": (
            campaign.FORMAL_AGENT_TRANSPORT_TREATMENTS["apex"][
                "overflow_policy"
            ]
        ),
    }
    aka_runtime = {"fixture": True}
    runtime = {"aka_execution_snapshot": aka_runtime}
    comparison_runtime = campaign.comparison_runtime_projection(runtime)
    assert comparison_runtime is not None
    evaluator = {"execution_manifest_sha256": "2" * 64}
    run_config = _run_config_contract()
    comparison = {
        "schema": "aka.apex-vs-codex-comparison-contract/v6",
        "formal_execution": dict(campaign._FORMAL_LIVE_COMMITMENT),
        "formal_execution_sha256": campaign._FORMAL_LIVE_COMMITMENT_SHA256,
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
        "agent_transport_treatments": copy.deepcopy(
            campaign.FORMAL_AGENT_TRANSPORT_TREATMENTS
        ),
        "apex_treatment": {
            key: agent[key]
            for key in (
                "template",
                "session_receipt_schema",
                "apex_runtime_mount_policy_id",
                "attempt_mount_receipt_schema",
                "apex_runtime_mount_schema",
                "runtime_manifest_sha256",
            )
        },
        "codex": {
            "backend_runtime_closure_sha256": "9" * 64,
            "backend_runtime_closure": _backend_closure(),
        },
        "runtime": comparison_runtime,
        "evaluator_files_sha256": evaluator,
        "tasks": tasks,
        "run_config": run_config,
    }
    return {
        "schema": "aka.matched-campaign/v1",
        "formal_execution": dict(campaign._FORMAL_LIVE_COMMITMENT),
        "formal_execution_sha256": campaign._FORMAL_LIVE_COMMITMENT_SHA256,
        "repositories": repositories,
        "agent": agent,
        "runtime": runtime,
        "evaluator_files_sha256": evaluator,
        "configuration": {
            "run_config_contract": run_config,
            "tasks": tasks,
        },
        "comparison_contract": comparison,
        "comparison_contract_sha256": _digest(comparison),
    }


def _write_manifest(run: Path, manifest: dict) -> Path:
    run_config_contract = manifest["comparison_contract"].get("run_config")
    if isinstance(run_config_contract, dict):
        run_config_directory = run / ".formal-run-config"
        run_config_directory.mkdir(parents=True, exist_ok=True)
        run_config_path = run_config_directory / "run_config.yaml"
        run_config_path.write_text(
            yaml.safe_dump(
                {
                    "agent": {"template": manifest["agent"]["template"]},
                    **run_config_contract["effective_config"],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        run_config_path.chmod(0o444)
        run_config_directory.chmod(0o555)
        manifest["configuration"].update(
            {
                "run_config_path": str(run_config_path.resolve()),
                "run_config_sha256": campaign._sha256_file(run_config_path),
                "run_config_size_bytes": run_config_path.stat().st_size,
                "run_config_contract": run_config_contract,
            }
        )
    path = run / "campaign_manifest.yaml"
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    path.chmod(0o444)
    return path


def test_v6_comparison_binds_repository_mount_policy_and_runtime_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(campaign, "_revalidate_aka_runtime", lambda _manifest: True)
    monkeypatch.setattr(
        campaign, "verify_backend_closure", lambda closure, _digest: closure
    )
    manifest = _v6_manifest()
    _write_manifest(tmp_path, manifest)

    loaded = campaign._load_verified_campaign_manifest(tmp_path)

    comparison = loaded["comparison_contract"]
    assert campaign._expected_session_receipt_schema(tmp_path) == (
        "agentkernelarena.apex-attempt-receipt/v6"
    )
    assert comparison["repositories"] == loaded["repositories"]
    assert comparison["apex_treatment"] == {
        "template": "apex",
        "session_receipt_schema": "agentkernelarena.apex-attempt-receipt/v6",
        "apex_runtime_mount_policy_id": (
            campaign_isolation.APEX_RUNTIME_MOUNT_POLICY
        ),
        "attempt_mount_receipt_schema": (
            campaign_isolation.ATTEMPT_MOUNT_RECEIPT_SCHEMA
        ),
        "apex_runtime_mount_schema": campaign_isolation.APEX_RUNTIME_MOUNT_SCHEMA,
        "runtime_manifest_sha256": "c" * 64,
    }

    codex_manifest = _v6_manifest()
    codex_manifest["agent"] = {
        "template": "codex",
        "session_receipt_schema": "agentkernelarena.codex-attempt-receipt/v6",
        "backend_runtime_closure_schema": "aka.backend-runtime-closure/v1",
        "backend_runtime_closure_sha256": "9" * 64,
        "backend_runtime_closure": _backend_closure(),
        "max_process_output_bytes": campaign.FORMAL_AGENT_TRANSPORT_TREATMENTS[
            "codex"
        ]["max_process_output_bytes"],
        "structured_stream_output_limit_bytes": (
            campaign.FORMAL_AGENT_TRANSPORT_TREATMENTS["codex"][
                "structured_stream_output_limit_bytes"
            ]
        ),
        "structured_stream_overflow_policy": (
            campaign.FORMAL_AGENT_TRANSPORT_TREATMENTS["codex"][
                "overflow_policy"
            ]
        ),
    }
    codex_run = tmp_path / "codex"
    codex_run.mkdir()
    _write_manifest(codex_run, codex_manifest)
    loaded_codex = campaign._load_verified_campaign_manifest(codex_run)
    assert campaign._expected_session_receipt_schema(codex_run) == (
        "agentkernelarena.codex-attempt-receipt/v6"
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


def test_comparison_v6_digest_is_identical_for_apex_and_codex_arms() -> None:
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
        "backend_runtime_closure_schema": "aka.backend-runtime-closure/v1",
        "backend_runtime_closure_sha256": "9" * 64,
        "backend_runtime_closure": _backend_closure(),
        "isolation": {},
    }
    apex = {
        **common,
        "template": "apex",
        "session_receipt_schema": "agentkernelarena.apex-attempt-receipt/v6",
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
        "session_receipt_schema": "agentkernelarena.codex-attempt-receipt/v6",
    }
    arguments = {
        "policy": policy,
        "measurement": {},
        "repositories": {
            "agent_kernel_arena": {
                "commit": "d" * 40,
                "tree": "1" * 40,
                "dirty": False,
                "status_sha256": "e" * 64,
                "execution_manifest_schema": "aka.execution-snapshot-manifest/v1",
                "execution_manifest_sha256": "2" * 64,
            },
            "apex": _repository(),
        },
        "runtime": {},
        "evaluator": {},
        "tasks": [],
        "run_config": _run_config_contract(),
    }

    apex_contract = campaign._comparison_contract(agent=apex, **arguments)
    codex_contract = campaign._comparison_contract(agent=codex, **arguments)

    assert apex_contract == codex_contract
    assert _digest(apex_contract) == _digest(codex_contract)


@pytest.mark.parametrize(
    "tamper", ["repository", "marker", "receipt", "schema", "combined"]
)
def test_v6_manifest_divergence_and_downgrade_fail_closed(
    tmp_path: Path, tamper: str
) -> None:
    manifest = _v6_manifest()
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


def _mount_record(path: Path, mount_id: int, *, root: Path | None = None) -> dict:
    return {
        "mount_id": mount_id,
        "parent_id": 1,
        "major_minor": "0:1",
        "root": str(root or Path("/")),
        "mount_point": str(path),
    }


def _mount_identity(path: Path, mount_id: int = 10, *, bound: bool = False) -> dict:
    metadata = path.lstat()
    return {
        "path": str(path),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": stat.S_IMODE(metadata.st_mode),
        "mount": _mount_record(
            path if bound else Path("/"),
            mount_id,
            root=path if bound else Path("/"),
        ),
        "nested_mounts": [],
        "source": "o_path_nofollow_bind_fd",
    }


def _outer_bubblewrap() -> dict:
    path = Path("/usr/bin/bwrap")
    metadata = path.lstat()
    digest = campaign._sha256_file(path)
    return {
        "policy": "canonical_source_to_sealed_memfd_exec_v1",
        "canonical_path": str(path),
        "source": {
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "mode": stat.S_IMODE(metadata.st_mode),
            "uid": metadata.st_uid,
            "gid": metadata.st_gid,
            "nlink": metadata.st_nlink,
            "size_bytes": metadata.st_size,
            "sha256": digest,
        },
        "sealed_exec": {
            "transport": "sealed_memfd_proc_self_fd",
            "size_bytes": metadata.st_size,
            "sha256": digest,
            "seals": [
                "F_SEAL_WRITE",
                "F_SEAL_SHRINK",
                "F_SEAL_GROW",
                "F_SEAL_SEAL",
            ],
        },
    }


def _private_namespace_mount(path: Path, mount_id: int) -> dict:
    return {
        "path": str(path),
        "device": mount_id,
        "inode": mount_id + 100,
        "access": "read_write",
        "filesystem_type": "tmpfs",
        "mount": _mount_record(path, mount_id),
        "mount_options": ["rw"],
        "covered_mount_ids": [],
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
    source_identities = {
        role: _mount_identity(path, 20 + index)
        for index, (role, path) in enumerate(paths.items())
    }
    target_identities = {
        role: _mount_identity(path, 40 + index, bound=True)
        for index, (role, path) in enumerate(paths.items())
    }
    namespace_roles = {"persistent_writable": {}, "read_only": {}}
    for group, names in {
        "persistent_writable": ("apex_artifacts", "backend_home"),
        "read_only": ("scored_workspace", "sealed_task_contract", "apex_runtime"),
    }.items():
        for role in names:
            source = source_identities[role]
            target = target_identities[role]
            namespace_roles[group][role] = {
                "source": {
                    key: source[key] for key in ("path", "device", "inode", "mount")
                },
                "target": {
                    "path": target["path"],
                    "device": target["device"],
                    "inode": target["inode"],
                    "access": (
                        "read_write" if group == "persistent_writable" else "read_only"
                    ),
                    "mount": target["mount"],
                    "mount_options": [
                        "rw" if group == "persistent_writable" else "ro"
                    ],
                },
            }
    declared = sorted([str(data), *(str(path) for path in paths.values())])
    material = {
        "schema": campaign_isolation.ATTEMPT_MOUNT_RECEIPT_SCHEMA,
        "campaign_data_root": str(data),
        "campaign_data_root_hidden": True,
        "campaign_data_identity": _mount_identity(data),
        "outer_bubblewrap": _outer_bubblewrap(),
        "namespace_mounts": {
            "policy": "blocked_namespace_mount_attestation_v1",
            "namespace_init_pid": 123,
            "mount_namespace_id": 456,
            "root": {
                "path": "/",
                "device": 1,
                "inode": 2,
                "access": "read_only",
                "mount": _mount_record(Path("/"), 2),
                "mount_options": ["ro"],
            },
            "campaign_data_root": _private_namespace_mount(data, 3),
            "private_tmpfs": {
                "tmp": _private_namespace_mount(Path("/tmp"), 4),
                "dev_shm": _private_namespace_mount(Path("/dev/shm"), 5),
            },
            "roles": namespace_roles,
            "declared_mount_points": declared,
            "observed_mount_points_below_campaign_data": declared,
            "closed_set": True,
            "aliases_absent": True,
        },
        "roles": {
            "persistent_writable": {
                role: target_identities[role]
                for role in ("apex_artifacts", "backend_home")
            },
            "read_only": {
                role: target_identities[role]
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


def test_v6_mount_roles_are_closed_canonical_and_non_overlapping(
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


def test_v6_mount_roles_reject_private_tmpfs_or_role_set_tampering(
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


@pytest.mark.parametrize(
    "tamper",
    [
        "outer_binary",
        "missing_attestation",
        "wrong_access",
        "wrong_mount_root",
        "undeclared_nested_mount",
        "target_alias",
    ],
)
def test_v6_mount_auditor_rejects_namespace_and_sealed_exec_tampering(
    tmp_path: Path, tamper: str
) -> None:
    receipt, task_spec, receipt_path, contract_path, runtime = _role_fixture(tmp_path)
    mounts = receipt["attempt_mounts"]
    namespace = mounts["namespace_mounts"]
    if tamper == "outer_binary":
        mounts["outer_bubblewrap"]["source"]["sha256"] = "0" * 64
    elif tamper == "missing_attestation":
        mounts["namespace_mounts"] = None
    elif tamper == "wrong_access":
        namespace["roles"]["read_only"]["scored_workspace"]["target"][
            "access"
        ] = "read_write"
    elif tamper == "wrong_mount_root":
        namespace["roles"]["read_only"]["scored_workspace"]["target"][
            "mount"
        ]["root"] = "/"
    elif tamper == "undeclared_nested_mount":
        namespace["observed_mount_points_below_campaign_data"].append(
            str(Path(task_spec["workspace"]) / "nested")
        )
    else:
        source = namespace["roles"]["read_only"]["scored_workspace"]["target"]
        alias = namespace["roles"]["persistent_writable"]["apex_artifacts"][
            "target"
        ]
        alias["device"] = source["device"]
        alias["inode"] = source["inode"]
    material = copy.deepcopy(mounts)
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


def test_v6_runtime_snapshot_binds_manifest_and_role_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, task_spec, receipt_path, contract_path, runtime_parent = (
        _role_fixture(tmp_path)
    )
    source_root = tmp_path / "apex-source"
    source_root.mkdir()
    source_main = source_root / "main.py"
    source_main.write_text("print('apex')\n", encoding="utf-8")
    source_module = source_root / "src/apex_probe.py"
    source_module.parent.mkdir()
    source_module.write_text("VALUE = 'sealed'\n", encoding="utf-8")
    (source_root / ".gitignore").write_text(".venv\n", encoding="utf-8")
    for arguments in (
        ("init", "-q"),
        ("config", "user.name", "test"),
        ("config", "user.email", "test@example.invalid"),
        ("add", "."),
        ("commit", "-qm", "fixture"),
    ):
        subprocess.run(
            ["/usr/bin/git", *arguments],
            cwd=source_root,
            check=True,
            capture_output=True,
        )
    source_venv = source_root / ".venv"
    source_venv_bin = source_venv / "bin"
    source_venv_bin.mkdir(parents=True)
    source_python = source_venv_bin / "python"
    os.symlink("/usr/bin/python3", source_python)
    (source_venv / "pyvenv.cfg").write_text(
        "include-system-site-packages = false\n", encoding="utf-8"
    )
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    (source_venv / "lib" / version / "site-packages").mkdir(parents=True)

    plan = apex_runtime.plan_runtime(
        source_root, source_python, declared_roots=[]
    )
    runtime = apex_runtime.materialize_runtime(plan, runtime_parent)
    runtime_digest = plan.sha256
    runtime_manifest = apex_runtime.verify_runtime_snapshot(
        runtime, runtime_digest
    )
    system_python = plan.system_python
    entrypoint = runtime / "repo/main.py"
    underlying = runtime / "venv/bin/python"
    runtime_manifest_path = runtime / "runtime_manifest.json"
    bootstrap_path = runtime / apex_runtime.RUNTIME_BOOTSTRAP_NAME
    assert runtime_manifest == plan.manifest

    immutable_mount = {
        "mount_id": 91,
        "device": "0:91",
        "root": "/",
        "mount_point": str(runtime),
        "filesystem": "fuse.squashfuse",
        "mount_options": ["nodev", "nosuid", "ro"],
        "super_options": [
            "allow_other",
            "default_permissions",
            f"group_id={os.getgid()}",
            "ro",
            f"user_id={os.getuid()}",
        ],
        "read_only": True,
    }
    monkeypatch.setattr(
        apex_runtime, "_observed_immutable_mount", lambda _root: immutable_mount
    )
    runtime_service_evidence = _runtime_service_evidence(
        runtime, runtime_manifest, immutable_mount
    )
    immutable_receipt = apex_runtime.create_immutable_mount_receipt(
        runtime,
        runtime_manifest,
        "f" * 64,
        runtime_service_evidence,
    )

    mounts = receipt["attempt_mounts"]
    runtime_source_mount = _mount_identity(runtime, 24)
    runtime_target_mount = _mount_identity(runtime, 44, bound=True)
    mounts["roles"]["read_only"]["apex_runtime"] = runtime_target_mount
    mounts["namespace_mounts"]["roles"]["read_only"]["apex_runtime"] = {
        "source": {
            key: runtime_source_mount[key]
            for key in ("path", "device", "inode", "mount")
        },
        "target": {
            "path": runtime_target_mount["path"],
            "device": runtime_target_mount["device"],
            "inode": runtime_target_mount["inode"],
            "access": "read_only",
            "mount": runtime_target_mount["mount"],
            "mount_options": ["ro"],
        },
    }
    for key in ("declared_mount_points", "observed_mount_points_below_campaign_data"):
        mounts["namespace_mounts"][key] = sorted(
            str(runtime) if value == str(runtime_parent) else value
            for value in mounts["namespace_mounts"][key]
        )
    mount_material = dict(mounts)
    mount_material.pop("sha256")
    mounts["sha256"] = _digest(mount_material)

    run = tmp_path / "campaign-data/run"
    monkeypatch.setattr(campaign, "_revalidate_aka_runtime", lambda _manifest: True)
    monkeypatch.setattr(
        campaign, "verify_backend_closure", lambda closure, _digest: closure
    )
    campaign_manifest = _v6_manifest(runtime_digest)
    runtime_repository = {
        "commit": runtime_manifest["git"]["commit"],
        "dirty": runtime_manifest["git"]["dirty"],
        "status_sha256": runtime_manifest["git"]["status_sha256"],
        "runtime_manifest_sha256": runtime_digest,
    }
    campaign_manifest["repositories"]["apex"] = runtime_repository
    campaign_manifest["comparison_contract"]["repositories"][
        "apex"
    ] = runtime_repository
    campaign_manifest["comparison_contract_sha256"] = _digest(
        campaign_manifest["comparison_contract"]
    )
    _write_manifest(run, campaign_manifest)
    entrypoint_sha = hashlib.sha256(entrypoint.read_bytes()).hexdigest()
    monkeypatch.setenv("AGENT_KERNEL_ARENA_APEX_SOURCE_ROOT", str(source_root))
    runtime_receipt = apex_launcher._runtime_snapshot_receipt(
        plan=apex_runtime.RuntimePlan(
            manifest=runtime_manifest,
            roots=(),
            system_python=system_python,
        ),
        snapshot=runtime,
        immutable_mount=immutable_receipt,
        attempt_mounts_sha256=mounts["sha256"],
    )
    assert set(runtime_receipt["python"]) == {
        "source_launcher_relative_path",
        "launcher_path",
        "launcher_sha256",
        "underlying_path",
        "underlying_sha256",
        "flags",
        "pythonpath",
        "environment",
    }
    assert runtime_receipt["python"]["launcher_path"] == str(
        runtime / apex_runtime.RUNTIME_WRAPPER_NAME
    )
    assert runtime_receipt["python"]["underlying_path"] == str(underlying)
    assert runtime_receipt["immutability"]["receipt_sha256"] == (
        immutable_receipt["sha256"]
    )
    receipt.update(
        {
            "schema": "agentkernelarena.apex-attempt-receipt/v6",
            "apex_runtime_mount": runtime_receipt,
            "apex": {
                "entrypoint": str(entrypoint),
                "entrypoint_sha256": entrypoint_sha,
                "python": runtime_receipt["python"]["launcher_path"],
                "python_sha256": runtime_receipt["python"]["launcher_sha256"],
            },
        }
    )
    monkeypatch.setenv("APEX_ROOT", str(runtime / "repo"))
    monkeypatch.setenv(
        "APEX_PYTHON", str(runtime / apex_runtime.RUNTIME_WRAPPER_NAME)
    )
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
    receipt["apex_runtime_mount"]["python"]["flags"] = ["-I", "-S", "-u"]
    current_policy_material = dict(receipt["apex_runtime_mount"])
    current_policy_material.pop("sha256")
    receipt["apex_runtime_mount"]["sha256"] = _digest(current_policy_material)
    assert campaign._apex_runtime_mount_errors(**arguments) == []

    receipt["apex_runtime_mount"]["immutability"]["receipt_sha256"] = "0" * 64
    stale_mount_material = dict(receipt["apex_runtime_mount"])
    stale_mount_material.pop("sha256")
    receipt["apex_runtime_mount"]["sha256"] = _digest(stale_mount_material)
    assert campaign._apex_runtime_mount_errors(**arguments) == [
        "apex_runtime_mount_contract_mismatch"
    ]
    receipt["apex_runtime_mount"]["immutability"]["receipt_sha256"] = (
        immutable_receipt["sha256"]
    )
    current_mount_material = dict(receipt["apex_runtime_mount"])
    current_mount_material.pop("sha256")
    receipt["apex_runtime_mount"]["sha256"] = _digest(current_mount_material)
    assert campaign._apex_runtime_mount_errors(**arguments) == []

    wrapper_path = runtime / apex_runtime.RUNTIME_WRAPPER_NAME
    wrapper_path.chmod(0o755)
    wrapper_path.write_bytes(b"#!/bin/sh\nexit 1\n")
    wrapper_path.chmod(0o555)
    assert campaign._apex_runtime_mount_errors(**arguments) == [
        "apex_runtime_mount_contract_mismatch"
    ]
    wrapper_path.chmod(0o755)
    wrapper_path.write_bytes(apex_runtime.RUNTIME_WRAPPER)
    wrapper_path.chmod(0o555)
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
