import base64
import copy
import fcntl
import logging
import importlib
import hashlib
import json
import os
import re
import runpy
import signal
import shutil
import sqlite3
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import main as aka_main
from src import campaign
from src import campaign_isolation
from src import evaluator
from src.evaluator_utils import inspect_formal_source_anti_tamper
from main import claim_next_descriptor, initialize_parallel_queue


codex_launcher = importlib.import_module("agents.codex.launch_agent")
_DEFAULT_TASK_COMPONENT = campaign.campaign_task_path_component(
    "triton2triton/vllm/example"
)


@pytest.fixture(autouse=True)
def _use_sealed_v5_runtime_test_evidence(monkeypatch) -> None:
    """Keep behavior tests focused on receipts, not host mount topology.

    The concrete AKA snapshot, backend-closure, and Apex mount revalidators have
    dedicated security tests.  Campaign behavior tests still carry and compare
    the independent runtime-isolation v5 evidence, while these test-only seams
    stand in for a real SquashFS mount and installed backend closure.
    """

    monkeypatch.setattr(campaign, "_revalidate_aka_runtime", lambda _manifest: True)
    monkeypatch.setattr(
        campaign,
        "verify_backend_closure",
        lambda closure, _expected_digest: closure,
    )
    monkeypatch.setattr(
        campaign,
        "_apex_runtime_mount_errors",
        lambda *_args, **_kwargs: [],
    )


def _policy() -> dict:
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


def test_campaign_task_path_component_blocks_escape_and_legacy_aliases(
    tmp_path: Path,
) -> None:
    attempt_root = (tmp_path / ".campaign_attempts").resolve()
    names = ["..", ".", "/", "a/b", "a_b", "a\\b", "任务/一"]
    components = [campaign.campaign_task_path_component(name) for name in names]

    assert len(components) == len(set(components))
    assert all(component not in {".", ".."} for component in components)
    assert all("/" not in component and "\\" not in component for component in components)
    assert all((attempt_root / component).parent == attempt_root for component in components)


def test_campaign_manifest_task_path_mapping_must_be_injective(monkeypatch) -> None:
    manifest = {
        "configuration": {
            "tasks": [
                {"task_name": "a/b"},
                {"task_name": "a_b"},
            ]
        }
    }
    assert campaign._campaign_task_paths_valid(manifest) is True

    monkeypatch.setattr(
        campaign,
        "campaign_task_path_component",
        lambda _name: "colliding-component",
    )
    assert campaign._campaign_task_paths_valid(manifest) is False


def _runtime_isolation_receipt(*, yama_ptrace_scope: int = 1) -> dict:
    return {
        "schema": "aka.runtime-isolation-receipt/v5",
        "policy": {
            "docker_user": "non_root",
            "docker_capabilities": "drop_all",
            "docker_no_new_privileges": True,
            "docker_apparmor": "unconfined_for_rootless_userns",
            "docker_seccomp": "unconfined_for_rootless_userns",
            "docker_systempaths": "unconfined_for_private_attempt_procfs",
            "docker_masked_paths_rebuilt": [
                "/proc/acpi", "/proc/asound", "/proc/scsi",
                "/sys/devices/virtual/powercap", "/sys/firmware",
                "/proc/interrupts", "/proc/kcore", "/proc/keys",
                "/proc/latency_stats", "/proc/sched_debug",
                "/proc/timer_list", "/proc/timer_stats",
            ],
            "docker_readonly_paths_rebuilt": [
                "/proc/bus", "/proc/fs", "/proc/irq", "/proc/sys",
                "/proc/sysrq-trigger",
            ],
            "docker_pid_namespace": "private_default",
            "attempt_mount_namespace": "bubblewrap",
            "attempt_pid_namespace": "private_per_attempt_with_bwrap_reaper_pid1",
            "attempt_ipc_namespace": "unshared",
            "attempt_proc": "private_procfs_for_attempt_pid_namespace",
            "direct_agent_proc": "aka_outer_private_attempt_procfs",
            "apex_outer_proc": (
                "trusted_orchestrator_inherited_worker_procfs_nested_userns_writable"
            ),
            "apex_backend_proc": "apex_inner_private_attempt_procfs_required",
            "process_lifetime_boundary": "namespace_init_pidfd_v1",
            "proc_escape_guard": "outer_process_absent_from_private_procfs_v1",
            "command_sandbox": "codex_managed_permission_profile_bwrap",
            "command_pid_namespace": "nested_codex_unshared_inside_private_attempt_pidns_v1",
            "command_network": "managed_profile_denied_live_probe_v1",
            "command_gpu_access": "sealed_memfd_immutable_path_bwrap_and_single_gpu_probe_v1",
            "credential_read": "denied_by_managed_permission_profile",
        },
        "outer_runtime": {
            "effective_uid": 1000,
            "effective_gid": 1000,
            "supplementary_gids": [44, 109],
            "capabilities": {
                "CapInh": 0,
                "CapPrm": 0,
                "CapEff": 0,
                "CapBnd": 0,
                "CapAmb": 0,
            },
            "no_new_privileges": True,
            "seccomp_mode": 0,
            "seccomp_filters": 0,
            "apparmor_profile": "unconfined",
            "yama_ptrace_scope": yama_ptrace_scope,
        },
        "bubblewrap": {
            "resolved_path": "/usr/bin/bwrap",
            "sha256": "8" * 64,
            "version": "bubblewrap 1.0",
        },
        "codex_cli": {
            "resolved_path": "/opt/node/bin/codex",
            "sha256": "7" * 64,
            "version": "codex-cli test",
        },
        "codex_requirements": {
            "resolved_path": "/etc/codex/requirements.toml",
            "sha256": "6" * 64,
            "permission_profile": "aka_formal_kernel_v1",
            "agent_requested_sandbox": "workspace-write_legacy_cli",
            "effective_profile_probe": "explicit_named_profile_live",
            "normalization_evidence": "managed_allowlist_plus_pinned_cli_identity",
            "workspace_write": True,
            "credential_path": "~/.codex/auth.json",
            "credential_read": "deny",
            "command_network": "deny",
            "device_access": (
                "sealed_pinned_immutable_path_bwrap_with_docker_device_boundary"
            ),
            "hooks": "disabled",
        },
        "codex_gpu_bubblewrap": {
            "resolved_path": "/workspace/agents/codex/bin/bwrap",
            "sha256": "9" * 64,
            "size_bytes": 2381,
            "interpreter": "/usr/bin/python3 -I",
            "real_bwrap": "/usr/bin/bwrap",
            "real_bwrap_sha256": "8" * 64,
            "sandbox_mounted_path": "/tmp/aka-codex-gpu-bwrap/bwrap",
            "mount_transport": (
                "sealed_memfd_ro_bind_data_under_remounted_ro_tmpfs"
            ),
            "device_policy": "docker_visible_kfd_and_render_nodes_only",
        },
        "attempt_probe": {
            "campaign_data_hidden": True,
            "outer_pid_namespace_absent_from_private_proc": True,
            "parent_root_sentinel_unreachable": True,
            "parent_fd_sentinel_unreachable": True,
            "proc_mount_read_write": True,
            "pid_namespace_unshared": True,
            "ipc_namespace_unshared": True,
            "private_shm": True,
            "docker_system_paths_remasked": True,
            "private_proc_control_writes_blocked": True,
            "no_new_privileges": True,
            "effective_capabilities_zero": True,
            "bounding_capabilities_zero": True,
            "all_capability_sets_zero": True,
            "seccomp_disabled": True,
        },
        "codex_sandbox_probe": {
            "workspace_write_enforced": True,
            "credential_read_denied": True,
            "command_network_denied": True,
            "command_not_in_worker_pid_namespace": True,
            "pid1_root_alias_credential_blocked": True,
            "pid1_environ_blocked": True,
            "pid1_mem_blocked": True,
            "pinned_gpu_bwrap_active": True,
            "gpu_bwrap_directory_immutable": True,
            "gpu_bwrap_path_immutable": True,
            "assigned_gpu_devices_visible": True,
            "assigned_gpu_devices_writable": True,
            "single_gpu_runtime_visible": True,
            "gpu_compute_probe_passed": True,
        },
    }


def _test_backend_closure() -> dict:
    return {
        "schema": campaign.BACKEND_CLOSURE_SCHEMA,
        "backend": "codex",
        "launcher": {"path": "/opt/node/bin/codex", "sha256": "a" * 64},
        "interpreter": {"path": "/opt/node/bin/node", "sha256": "b" * 64},
        "components": [],
        "closure_sha256": "c" * 64,
    }


def _test_aka_binding() -> tuple[dict, dict]:
    state = {
        "commit": "1" * 40,
        "tree": "2" * 40,
        "dirty": False,
        "status_sha256": "3" * 64,
        "execution_manifest_schema": campaign.EXECUTION_MANIFEST_SCHEMA,
        "execution_manifest_sha256": "4" * 64,
        "git_evidence_policy_id": "head_tree_direct_bytes_no_filters_v1",
    }
    requested_mount_options = [
        "ro",
        "nodev",
        "nosuid",
        "default_permissions",
        "allow_other",
        "subtype=squashfuse",
    ]
    owner = {"uid": os.getuid(), "gid": os.getgid()}
    host_access_policy = {
        "schema": "aka.immutable-runtime-host-access-policy/v1",
        "policy_id": "private_ancestor_docker_daemon_fuse_v1",
        "requested_mount_options": requested_mount_options,
        "private_ancestor": {
            "path": "/test",
            "device": 7,
            "inode": 11,
            **owner,
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
            "sha256": "b" * 64,
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
    host_access_policy["sha256"] = campaign._canonical_json_digest(
        host_access_policy
    )
    observed_mount = {
        "filesystem_type": "fuse.squashfuse",
        "read_only": True,
        "root": "/",
        "nested_mounts": [],
    }
    engine_evidence = {
        "schema": "aka.immutable-runtime-mount-engine/v2",
        "policy_id": "sealed_memfd_squashfs_docker_bindable_read_only_v2",
        "requested_mount_options": requested_mount_options,
        "host_access_policy_sha256": host_access_policy["sha256"],
        "tools": {
            "mksquashfs": {"path": "/usr/bin/mksquashfs", "sha256": "c" * 64},
            "squashfuse": {"path": "/usr/bin/squashfuse", "sha256": "d" * 64},
        },
    }
    engine_evidence["sha256"] = campaign._canonical_json_digest(engine_evidence)
    service_mount_receipt = {"host_access_policy": host_access_policy}
    runtime_service_evidence = {
        "schema": "aka.immutable-runtime-mount-service-ready/v2",
        "policy_id": "single_docker_bindable_snapshot_signal_lifetime_v2",
        "mount_receipt": service_mount_receipt,
        "engine_evidence": engine_evidence,
    }
    runtime_service_evidence["sha256"] = campaign._canonical_json_digest(
        runtime_service_evidence
    )
    mount_receipt = {
        "schema": campaign.IMMUTABLE_MOUNT_RECEIPT_SCHEMA,
        "policy_id": "sealed_memfd_squashfs_docker_bindable_read_only_v2",
        "manifest_sha256": state["execution_manifest_sha256"],
        "image_sha256": "e" * 64,
        "memfd_seals": [
            "F_SEAL_WRITE",
            "F_SEAL_SHRINK",
            "F_SEAL_GROW",
            "F_SEAL_SEAL",
        ],
        "requested_mount_options": requested_mount_options,
        "mount": observed_mount,
        "host_access_policy": host_access_policy,
        "runtime_service_evidence_sha256": runtime_service_evidence["sha256"],
        "runtime_engine_evidence_sha256": engine_evidence["sha256"],
    }
    mount_receipt["sha256"] = campaign._canonical_json_digest(mount_receipt)
    runtime = {
        "schema": "aka.execution-snapshot-runtime/v2",
        "root": "/test/aka-runtime",
        "manifest_path": "/test/aka-runtime-manifest.json",
        "manifest_file_sha256": "8" * 64,
        "manifest_sha256": state["execution_manifest_sha256"],
        "mount_receipt_path": "/test/aka-runtime-mount-receipt.json",
        "mount_receipt_file_sha256": "9" * 64,
        "mount_receipt_sha256": mount_receipt["sha256"],
        "mount_receipt_schema": campaign.IMMUTABLE_MOUNT_RECEIPT_SCHEMA,
        "mount_receipt": mount_receipt,
        "runtime_service_evidence_path": "/test/runtime-service-ready.json",
        "runtime_service_evidence_file_sha256": "f" * 64,
        "runtime_service_evidence_content_sha256": runtime_service_evidence[
            "sha256"
        ],
        "runtime_engine_evidence_sha256": engine_evidence["sha256"],
        "runtime_service_evidence": runtime_service_evidence,
    }
    return state, runtime


def _test_agent_manifest(name: str) -> dict:
    closure = _test_backend_closure()
    transport = campaign.FORMAL_AGENT_TRANSPORT_TREATMENTS[name]
    manifest = {
        "template": name,
        "session_receipt_schema": (
            "agentkernelarena.apex-attempt-receipt/v6"
            if name == "apex"
            else "agentkernelarena.codex-attempt-receipt/v6"
        ),
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
        "max_process_output_bytes": transport["max_process_output_bytes"],
        "structured_stream_output_limit_bytes": 16 * 1024 * 1024,
        "structured_stream_overflow_policy": transport["overflow_policy"],
        "codex_version": "codex 1.0",
        "codex_binary_sha256": "e" * 64,
        "backend_runtime_closure_schema": campaign.BACKEND_CLOSURE_SCHEMA,
        "backend_runtime_closure_sha256": closure["closure_sha256"],
        "backend_runtime_closure": closure,
        "cloud_config_bootstrap_schema": (
            campaign.CODEX_CLOUD_CONFIG_BOOTSTRAP_SCHEMA
        ),
        "cloud_config_bootstrap_policy": (
            campaign.CODEX_CLOUD_CONFIG_BOOTSTRAP_POLICY
        ),
        "cloud_config_bundle_sha256": "c" * 64,
        "cloud_config_host_runtime_closure_sha256": "d" * 64,
        "cloud_config_initial_refresh_receipt_sha256": "e" * 64,
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
        "agent_config_sha256": ("1" if name == "apex" else "2") * 64,
    }
    if name == "apex":
        manifest |= {
            "apex_runtime_mount_policy_id": campaign.APEX_RUNTIME_MOUNT_POLICY,
            "attempt_mount_receipt_schema": campaign.ATTEMPT_MOUNT_RECEIPT_SCHEMA,
            "apex_runtime_mount_schema": campaign.APEX_RUNTIME_MOUNT_SCHEMA,
            "runtime_manifest_sha256": "9" * 64,
        }
    return manifest


def test_runtime_isolation_receipt_records_only_stable_namespace_evidence(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("AGENT_KERNEL_ARENA_CAMPAIGN_DATA_ROOT", str(tmp_path))
    outer = {
        **_runtime_isolation_receipt()["outer_runtime"],
        "pid_namespace": "pid:[101]",
        "ipc_namespace": "ipc:[202]",
    }
    probe = _runtime_isolation_receipt()["attempt_probe"]
    monkeypatch.setattr(
        campaign_isolation, "_outer_runtime_observation", lambda: outer
    )
    monkeypatch.setattr(
        campaign_isolation,
        "_bubblewrap_identity",
        lambda: (
            Path("/usr/bin/bwrap"),
            _runtime_isolation_receipt()["bubblewrap"],
        ),
    )
    monkeypatch.setattr(
        campaign_isolation,
        "_codex_requirements_identity",
        lambda: (
            Path("/etc/codex/requirements.toml"),
            _runtime_isolation_receipt()["codex_requirements"],
        ),
    )
    monkeypatch.setattr(
        campaign_isolation,
        "_codex_cli_identity",
        lambda: (
            Path("/opt/node/bin/codex"),
            _runtime_isolation_receipt()["codex_cli"],
        ),
    )
    monkeypatch.setattr(
        campaign_isolation,
        "_codex_gpu_bwrap_identity",
        lambda: (
            Path("/workspace/agents/codex/bin/bwrap"),
            _runtime_isolation_receipt()["codex_gpu_bubblewrap"],
        ),
    )
    observed: dict[str, object] = {}

    def fake_probe(*, binary, data_root, outer):
        observed.update(binary=binary, data_root=data_root, outer=outer)
        return probe

    monkeypatch.setattr(campaign_isolation, "_attempt_escape_probe", fake_probe)
    monkeypatch.setattr(
        campaign_isolation,
        "_codex_sandbox_probe",
        lambda **_kwargs: _runtime_isolation_receipt()["codex_sandbox_probe"],
    )

    receipt = campaign_isolation.runtime_isolation_receipt()

    assert receipt["outer_runtime"] == _runtime_isolation_receipt()["outer_runtime"]
    assert "pid_namespace" not in receipt["outer_runtime"]
    assert "ipc_namespace" not in receipt["outer_runtime"]
    assert observed["outer"] is outer
    assert receipt["attempt_probe"]["parent_root_sentinel_unreachable"] is True
    assert receipt["policy"]["command_gpu_access"] == (
        "sealed_memfd_immutable_path_bwrap_and_single_gpu_probe_v1"
    )


def test_attempt_escape_probe_rejects_non_policy_proc_errors(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_KERNEL_ARENA_CAMPAIGN_DATA_ROOT", str(tmp_path))
    outer = {
        "pid_namespace": "pid:[101]",
        "ipc_namespace": "ipc:[202]",
    }

    class Completed:
        returncode = 0
        stderr = ""
        stdout = json.dumps(
            {
                **_runtime_isolation_receipt()["attempt_probe"],
                "parent_fd_sentinel_unreachable": False,
            }
        )

    monkeypatch.setattr(
        campaign_isolation.subprocess, "run", lambda *_args, **_kwargs: Completed()
    )
    with pytest.raises(
        campaign_isolation.CampaignIsolationError,
        match="isolation proof is incomplete",
    ):
        campaign_isolation._attempt_escape_probe(
            binary=Path("/usr/bin/bwrap"), data_root=tmp_path, outer=outer
        )


def test_codex_requirements_are_content_pinned(tmp_path, monkeypatch) -> None:
    requirements = (
        Path(__file__).resolve().parents[1]
        / "agents"
        / "codex"
        / "formal_requirements.toml"
    )
    monkeypatch.setattr(
        campaign_isolation, "_CODEX_REQUIREMENTS_PATH", requirements
    )

    resolved, identity = campaign_isolation._codex_requirements_identity()

    assert resolved == requirements.resolve()
    assert identity["permission_profile"] == "aka_formal_kernel_v1"
    assert identity["credential_read"] == "deny"
    assert identity["command_network"] == "deny"
    assert identity["device_access"] == (
        "sealed_pinned_immutable_path_bwrap_with_docker_device_boundary"
    )

    changed = tmp_path / "requirements.toml"
    changed.write_bytes(requirements.read_bytes() + b"\n# changed\n")
    monkeypatch.setattr(campaign_isolation, "_CODEX_REQUIREMENTS_PATH", changed)
    with pytest.raises(
        campaign_isolation.CampaignIsolationError, match="pinned policy"
    ):
        campaign_isolation._codex_requirements_identity()


def test_codex_sandbox_probe_uses_managed_profile(tmp_path, monkeypatch) -> None:
    observed: dict[str, object] = {}

    class Completed:
        returncode = 0
        stderr = ""
        stdout = json.dumps(
            {
                "workspace_write_enforced": True,
                "credential_read_denied": True,
                "command_network_denied": True,
                "command_not_in_worker_pid_namespace": True,
                "pid1_root_alias_credential_blocked": True,
                "pid1_environ_blocked": True,
                "pid1_mem_blocked": True,
                "pinned_gpu_bwrap_active": True,
                "gpu_bwrap_directory_immutable": True,
                "gpu_bwrap_path_immutable": True,
                "assigned_gpu_devices_visible": True,
                "assigned_gpu_devices_writable": True,
                "single_gpu_runtime_visible": True,
                "gpu_compute_probe_passed": True,
            }
        )

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["environment"] = kwargs["env"]
        observed["pass_fds"] = kwargs["pass_fds"]
        return Completed()

    wrapper = (
        Path(__file__).resolve().parents[1] / "agents" / "codex" / "bin" / "bwrap"
    )
    monkeypatch.setattr(campaign_isolation.subprocess, "run", fake_run)

    result = campaign_isolation._codex_sandbox_probe(
        codex_binary=Path("/opt/node/bin/codex"),
        bubblewrap_binary=Path("/usr/bin/bwrap"),
        gpu_bubblewrap_binary=wrapper,
        data_root=tmp_path,
    )

    command = observed["command"]
    assert result["credential_read_denied"] is True
    assert result["command_not_in_worker_pid_namespace"] is True
    assert result["pinned_gpu_bwrap_active"] is True
    assert result["gpu_bwrap_directory_immutable"] is True
    assert result["gpu_bwrap_path_immutable"] is True
    assert result["assigned_gpu_devices_visible"] is True
    assert result["assigned_gpu_devices_writable"] is True
    assert result["single_gpu_runtime_visible"] is True
    assert result["gpu_compute_probe_passed"] is True
    assert "--include-managed-config" in command
    profile_index = command.index("--permission-profile")
    assert command[profile_index + 1] == "aka_formal_kernel_v1"
    assert "--unshare-pid" in command[: command.index("--")]
    environment = observed["environment"]
    assert environment["CODEX_HOME"].endswith("/home/.codex")
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert environment["PATH"].startswith("/tmp/aka-codex-gpu-bwrap:")
    trusted_path = str(campaign_isolation._CODEX_GPU_BWRAP_TRUSTED_PATH)
    trusted_dir = str(campaign_isolation._CODEX_GPU_BWRAP_TRUSTED_DIR)
    bind_index = command.index(trusted_path)
    assert command[bind_index - 2] == "--ro-bind-data"
    assert int(command[bind_index - 1]) in observed["pass_fds"]
    assert ["--tmpfs", trusted_dir] == command[
        command.index("--tmpfs", command.index(trusted_dir) + 1) :
        command.index("--tmpfs", command.index(trusted_dir) + 1) + 2
    ]
    assert ["--remount-ro", trusted_dir] == command[bind_index + 1 : bind_index + 3]


def test_codex_sandbox_probe_fails_closed_when_gpu_compute_is_unavailable(
    tmp_path, monkeypatch
) -> None:
    class Completed:
        returncode = 74
        stderr = ""
        stdout = json.dumps(
            _runtime_isolation_receipt()["codex_sandbox_probe"]
            | {"gpu_compute_probe_passed": False}
        )

    monkeypatch.setattr(
        campaign_isolation.subprocess,
        "run",
        lambda *_args, **_kwargs: Completed(),
    )
    wrapper = (
        Path(__file__).resolve().parents[1] / "agents" / "codex" / "bin" / "bwrap"
    )

    with pytest.raises(
        campaign_isolation.CampaignIsolationError,
        match="managed sandbox probe failed closed",
    ):
        campaign_isolation._codex_sandbox_probe(
            codex_binary=Path("/opt/node/bin/codex"),
            bubblewrap_binary=Path("/usr/bin/bwrap"),
            gpu_bubblewrap_binary=wrapper,
            data_root=tmp_path,
        )


def test_codex_gpu_bwrap_injects_only_docker_visible_gpu_devices() -> None:
    wrapper = (
        Path(__file__).resolve().parents[1] / "agents" / "codex" / "bin" / "bwrap"
    )
    namespace = runpy.run_path(str(wrapper), run_name="aka_gpu_bwrap_test")
    inject = namespace["inject_gpu_devices"]
    original = ["--ro-bind", "/", "/", "--dev", "/dev", "--unshare-pid"]

    transformed = inject(
        original,
        [Path("/dev/kfd"), Path("/dev/dri/renderD136")],
    )

    assert transformed == [
        "--ro-bind",
        "/",
        "/",
        "--dev",
        "/dev",
        "--dir",
        "/dev/dri",
        "--dev-bind",
        "/dev/kfd",
        "/dev/kfd",
        "--dev-bind",
        "/dev/dri/renderD136",
        "/dev/dri/renderD136",
        "--unshare-pid",
    ]
    assert original == ["--ro-bind", "/", "/", "--dev", "/dev", "--unshare-pid"]
    with pytest.raises(ValueError, match="exactly one"):
        inject(["--ro-bind", "/", "/"], [Path("/dev/kfd")])


def test_codex_gpu_bwrap_delegates_capability_probes_unchanged(monkeypatch) -> None:
    wrapper = (
        Path(__file__).resolve().parents[1] / "agents" / "codex" / "bin" / "bwrap"
    )
    namespace = runpy.run_path(str(wrapper), run_name="aka_gpu_bwrap_probe_test")
    observed: list[tuple[Path, list[str]]] = []

    class ExecCalled(Exception):
        pass

    def fake_execv(path, argv):
        observed.append((path, argv))
        raise ExecCalled

    monkeypatch.setattr(namespace["os"], "execv", fake_execv)
    for argument in ("--help", "--version"):
        with pytest.raises(ExecCalled):
            namespace["main"]([argument])

    assert observed == [
        (Path("/usr/bin/bwrap"), ["/usr/bin/bwrap", "--help"]),
        (Path("/usr/bin/bwrap"), ["/usr/bin/bwrap", "--version"]),
    ]


def test_codex_gpu_bwrap_python_startup_ignores_user_and_environment_code(
    tmp_path,
) -> None:
    wrapper = (
        Path(__file__).resolve().parents[1] / "agents" / "codex" / "bin" / "bwrap"
    )
    home = tmp_path / "home"
    poison = tmp_path / "python-path"
    marker = tmp_path / "python-startup-marker"
    user_site = home / ".local/lib/python3.10/site-packages"
    user_site.mkdir(parents=True)
    poison.mkdir()
    payload = (
        "import os\n"
        "from pathlib import Path\n"
        "Path(os.environ['AKA_PYTHON_STARTUP_MARKER']).write_text('loaded')\n"
    )
    (user_site / "usercustomize.py").write_text(payload, encoding="utf-8")
    (poison / "sitecustomize.py").write_text(payload, encoding="utf-8")
    environment = dict(os.environ)
    environment.update(
        {
            "HOME": str(home),
            "PYTHONPATH": str(poison),
            "AKA_PYTHON_STARTUP_MARKER": str(marker),
        }
    )
    environment.pop("PYTHONNOUSERSITE", None)

    control = subprocess.run(
        ["/usr/bin/python3", "-c", "pass"],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert control.returncode == 0, control.stderr
    assert marker.read_text(encoding="utf-8") == "loaded"
    marker.unlink()

    for argument in ("--help", "--version"):
        completed = subprocess.run(
            [str(wrapper), argument],
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
        assert completed.returncode == 0, completed.stderr
        assert not marker.exists()


def test_codex_gpu_bwrap_transport_is_content_pinned_and_sealed() -> None:
    wrapper = (
        Path(__file__).resolve().parents[1] / "agents" / "codex" / "bin" / "bwrap"
    )
    descriptor = campaign_isolation._sealed_codex_gpu_bwrap(wrapper)
    try:
        seals = (
            fcntl.F_SEAL_WRITE
            | fcntl.F_SEAL_SHRINK
            | fcntl.F_SEAL_GROW
            | fcntl.F_SEAL_SEAL
        )
        assert fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) == seals
        payload = os.read(descriptor, 1024 * 1024)
        assert hashlib.sha256(payload).hexdigest() == (
            campaign_isolation._CODEX_GPU_BWRAP_SHA256
        )
        with pytest.raises(OSError):
            os.write(descriptor, b"changed")
    finally:
        os.close(descriptor)


def test_wrapped_attempt_command_releases_owned_descriptors_idempotently() -> None:
    descriptor = os.memfd_create("aka-test-command")
    command = campaign_isolation.WrappedAttemptCommand(
        ["/bin/true"], pass_fds=(descriptor,)
    )

    assert campaign_isolation.attempt_command_pass_fds(command) == (descriptor,)
    campaign_isolation.release_attempt_command_fds(command)
    campaign_isolation.release_attempt_command_fds(command)

    assert campaign_isolation.attempt_command_pass_fds(command) == ()
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_pid_namespace_scan_fails_closed_for_unreadable_same_uid_entry(
    tmp_path, monkeypatch
) -> None:
    entry = tmp_path / "424242"
    entry.mkdir()
    real_iterdir = Path.iterdir
    real_stat = Path.stat

    def fake_iterdir(path):
        if path == Path("/proc"):
            return iter((entry,))
        return real_iterdir(path)

    def fake_stat(path, *args, **kwargs):
        if path == entry / "ns/pid":
            raise PermissionError("hidden namespace identity")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "iterdir", fake_iterdir)
    monkeypatch.setattr(Path, "stat", fake_stat)

    scan = campaign_isolation._pid_namespace_membership_scan(12345)

    assert scan.enumeration_completed is True
    assert scan.complete is False
    assert scan.inaccessible_entries_count == 1
    assert scan.live_visible_members == ()


def test_pid_namespace_scan_fails_closed_for_malformed_visible_member(
    tmp_path, monkeypatch
) -> None:
    entry = tmp_path / "424243"
    entry.mkdir()
    real_iterdir = Path.iterdir
    real_stat = Path.stat
    real_read_text = Path.read_text

    def fake_iterdir(path):
        if path == Path("/proc"):
            return iter((entry,))
        return real_iterdir(path)

    def fake_stat(path, *args, **kwargs):
        if path == entry / "ns/pid":
            return SimpleNamespace(st_ino=12345)
        return real_stat(path, *args, **kwargs)

    def fake_read_text(path, *args, **kwargs):
        if path == entry / "stat":
            return "malformed"
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "iterdir", fake_iterdir)
    monkeypatch.setattr(Path, "stat", fake_stat)
    monkeypatch.setattr(Path, "read_text", fake_read_text)

    scan = campaign_isolation._pid_namespace_membership_scan(12345)

    assert scan.enumeration_completed is False
    assert scan.complete is False
    assert scan.inaccessible_entries_count == 0
    assert scan.live_visible_members == ()


def test_codex_gpu_bwrap_is_content_pinned(tmp_path, monkeypatch) -> None:
    wrapper, identity = campaign_isolation._codex_gpu_bwrap_identity()

    assert wrapper.name == "bwrap"
    assert identity["device_policy"] == "docker_visible_kfd_and_render_nodes_only"
    assert identity["size_bytes"] == campaign_isolation._CODEX_GPU_BWRAP_SIZE_BYTES
    assert identity["real_bwrap_sha256"] == (
        campaign_isolation._CODEX_GPU_REAL_BWRAP_SHA256
    )

    changed = tmp_path / "bwrap"
    changed.write_bytes(wrapper.read_bytes() + b"\n# changed\n")
    changed.chmod(0o755)
    monkeypatch.setattr(campaign_isolation, "_CODEX_GPU_BWRAP_PATH", changed)
    with pytest.raises(
        campaign_isolation.CampaignIsolationError,
        match="violates its pin",
    ):
        campaign_isolation._codex_gpu_bwrap_identity()

    monkeypatch.setattr(campaign_isolation, "_CODEX_GPU_BWRAP_PATH", wrapper)
    changed_real = tmp_path / "real-bwrap"
    changed_real.write_bytes(Path("/usr/bin/bwrap").read_bytes() + b"changed")
    changed_real.chmod(0o755)
    monkeypatch.setattr(campaign_isolation, "_CODEX_GPU_REAL_BWRAP_PATH", changed_real)
    with pytest.raises(
        campaign_isolation.CampaignIsolationError,
        match="real bwrap violates its pin",
    ):
        campaign_isolation._codex_gpu_bwrap_identity()


def test_outer_runtime_isolation_fails_closed_on_yama_or_capabilities(monkeypatch) -> None:
    status = {
        "CapInh": "0",
        "CapPrm": "0",
        "CapEff": "0",
        "CapBnd": "0",
        "CapAmb": "0",
        "NoNewPrivs": "1",
        "Seccomp": "0",
        "Seccomp_filters": "0",
    }
    monkeypatch.setattr(campaign_isolation, "_proc_status", lambda: dict(status))
    monkeypatch.setattr(campaign_isolation.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(campaign_isolation.os, "getegid", lambda: 1000)
    monkeypatch.setattr(campaign_isolation.os, "getgroups", lambda: [109, 44, 109])
    monkeypatch.setattr(
        campaign_isolation.os, "readlink", lambda path: f"namespace:{path}"
    )

    real_read_text = Path.read_text

    def unsafe_yama(path, *args, **kwargs):
        if str(path) == "/proc/self/attr/current":
            return "unconfined\n"
        if str(path) == "/proc/sys/kernel/yama/ptrace_scope":
            return "0\n"
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", unsafe_yama)
    with pytest.raises(campaign_isolation.CampaignIsolationError, match="ptrace_scope"):
        campaign_isolation._outer_runtime_observation()

    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *args, **kwargs: (
            "unconfined\n"
            if str(path) == "/proc/self/attr/current"
            else "1\n"
            if str(path) == "/proc/sys/kernel/yama/ptrace_scope"
            else real_read_text(path, *args, **kwargs)
        ),
    )
    status["CapBnd"] = "1"
    with pytest.raises(campaign_isolation.CampaignIsolationError, match="CapBnd"):
        campaign_isolation._outer_runtime_observation()

    status["CapBnd"] = "0"
    status["NoNewPrivs"] = "0"
    with pytest.raises(
        campaign_isolation.CampaignIsolationError, match="no-new-privileges"
    ):
        campaign_isolation._outer_runtime_observation()

    status["NoNewPrivs"] = "1"
    status["Seccomp_filters"] = "1"
    with pytest.raises(campaign_isolation.CampaignIsolationError, match="seccomp"):
        campaign_isolation._outer_runtime_observation()

    status["Seccomp_filters"] = "0"
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *args, **kwargs: (
            "docker-default (enforce)\n"
            if str(path) == "/proc/self/attr/current"
            else "1\n"
            if str(path) == "/proc/sys/kernel/yama/ptrace_scope"
            else real_read_text(path, *args, **kwargs)
        ),
    )
    with pytest.raises(campaign_isolation.CampaignIsolationError, match="AppArmor"):
        campaign_isolation._outer_runtime_observation()


def _write_result(workspace: Path, optimized_ms: float, *, correct: bool = True) -> None:
    source_path = workspace / "source/candidate.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    if not source_path.exists():
        source_path.write_text("import torch\n", encoding="utf-8")
    config_path = workspace / "config.yaml"
    if not config_path.exists():
        config_path.write_text(
            yaml.safe_dump({"source_file_path": ["source/candidate.py"]}),
            encoding="utf-8",
        )
    task_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    case = {
        "test_case_id": "case-1",
        "params": {"n": 64},
        "benchmark_method": "cuda_event",
        "benchmark_samples": 100,
        "benchmark_effective_repeats": 100,
    }
    (workspace / "baseline_perf.yaml").write_text(
        yaml.safe_dump({"test_cases": [{**case, "execution_time_ms": 2.0}]}),
        encoding="utf-8",
    )
    (workspace / "optimized_perf.yaml").write_text(
        yaml.safe_dump({"test_cases": [{**case, "execution_time_ms": optimized_ms}]}),
        encoding="utf-8",
    )
    payload = {
        "task_name": "triton2triton/vllm/example",
        "pass_compilation": True,
        "pass_correctness": correct,
        "base_execution_time": 2.0,
        "best_optimized_execution_time": optimized_ms,
        "speedup_ratio": 2.0 / optimized_ms if optimized_ms else 0.0,
        "valid_optimized_cases": 1 if optimized_ms else 0,
        "valid_baseline_cases": 1,
        "baseline_benchmark_methods": ["cuda_event"],
        "optimized_benchmark_methods": ["cuda_event"],
        "benchmark_method_consistent": True,
        "speedup_calculation_error_message": None,
        "evaluation_mode": "candidate_scoring_v1",
        "agent_session_score_eligible": True,
        "agent_session_succeeded": True,
        "agent_session_error_type": None,
        "agent_session_terminal_status": "candidate_ready",
    }
    initial_guard = inspect_formal_source_anti_tamper(workspace, task_config)
    payload["source_anti_tamper"] = inspect_formal_source_anti_tamper(
        workspace,
        task_config,
        expected_source_manifest_sha256=initial_guard["source_manifest_sha256"],
    )
    (workspace / "task_result.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )


def test_diagnostic_replay_is_never_selection_eligible(tmp_path) -> None:
    run = tmp_path / "run"
    workspace = run / "attempt_01" / "workspace"
    workspace.mkdir(parents=True)
    _write_result(workspace, 1.0)
    report_path = workspace / "task_result.yaml"
    report = yaml.safe_load(report_path.read_text(encoding="utf-8"))
    report["evaluation_mode"] = "diagnostic_baseline_replay_v1"
    report["agent_session_score_eligible"] = False
    report["agent_session_succeeded"] = False
    report["agent_session_error_type"] = "ApexAdapterError"
    report_path.write_text(
        yaml.safe_dump(report, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )

    record = campaign._attempt_record(
        attempt=1,
        workspace=workspace,
        run_directory=run,
        success=True,
        receipt_path=run / "attempt_01/session_receipt.json",
        require_session_receipt=False,
    )

    assert record["selection_eligible"] is False
    assert record["evaluation_mode"] == "diagnostic_baseline_replay_v1"
    assert record["agent_session_score_eligible"] is False
    assert "diagnostic_evaluation_not_scoreable" in record["eligibility_errors"]
    assert "agent_session_not_score_eligible" in record["eligibility_errors"]


def test_formal_eligibility_rejects_deleted_or_forged_guard_report(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_result(workspace, 1.0)
    report_path = workspace / "task_result.yaml"
    original = yaml.safe_load(report_path.read_text(encoding="utf-8"))

    deleted = copy.deepcopy(original)
    deleted.pop("source_anti_tamper")
    assert "missing_formal_source_anti_tamper_report" in (
        campaign._evaluation_eligibility_errors(workspace, deleted)
    )

    forged = copy.deepcopy(original)
    forged["source_anti_tamper"]["verdict"] = "PASS"
    forged["source_anti_tamper"]["rules_sha256"] = "f" * 64
    errors = campaign._evaluation_eligibility_errors(workspace, forged)
    assert "formal_source_anti_tamper_rules_mismatch" in errors
    assert "formal_source_anti_tamper_report_mismatch" in errors


def test_formal_eligibility_rejects_source_changed_after_guard_receipt(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_result(workspace, 1.0)
    report = yaml.safe_load(
        (workspace / "task_result.yaml").read_text(encoding="utf-8")
    )
    (workspace / "source/candidate.py").write_text(
        "import torch\nVALUE = 2\n", encoding="utf-8"
    )

    errors = campaign._evaluation_eligibility_errors(workspace, report)

    assert "formal_source_anti_tamper_report_mismatch" in errors


def test_formal_eligibility_rejects_cross_attempt_guard_substitution(tmp_path) -> None:
    first = tmp_path / "attempt_01"
    second = tmp_path / "attempt_02"
    for workspace, value in ((first, 1), (second, 2)):
        (workspace / "source").mkdir(parents=True)
        (workspace / "source/candidate.py").write_text(
            f"import torch\nVALUE = {value}\n", encoding="utf-8"
        )
        _write_result(workspace, 1.0)
    first_report = yaml.safe_load(
        (first / "task_result.yaml").read_text(encoding="utf-8")
    )
    second_report = yaml.safe_load(
        (second / "task_result.yaml").read_text(encoding="utf-8")
    )
    first_report["source_anti_tamper"] = second_report["source_anti_tamper"]

    errors = campaign._evaluation_eligibility_errors(first, first_report)

    assert "formal_source_anti_tamper_report_mismatch" in errors


def test_formal_attempt_mount_keeps_workspace_read_only_and_artifacts_private(
    tmp_path, monkeypatch
) -> None:
    data_root = tmp_path / "campaign-data"
    attempt = data_root / "run/task/attempt_01"
    sibling = data_root / "run/task/attempt_02"
    workspace = attempt / "workspace"
    artifact_root = attempt / ".workspace_apex/run-id"
    attempt_home = attempt / ".agent-home"
    workspace.mkdir(parents=True)
    artifact_root.mkdir(parents=True)
    attempt_home.mkdir(parents=True)
    sibling.mkdir(parents=True)
    source = workspace / "kernel.py"
    source.write_text("baseline\n", encoding="utf-8")
    (sibling / "secret.txt").write_text("prior attempt\n", encoding="utf-8")
    state_root = tmp_path / "agent-state"
    state_root.mkdir()
    (state_root / "history.jsonl").write_text("prior context\n", encoding="utf-8")
    shm_sentinel = Path("/dev/shm") / f"aka-ipc-sentinel-{tmp_path.name}"
    shm_sentinel.write_text("host-visible\n", encoding="utf-8")
    monkeypatch.setenv("AGENT_KERNEL_ARENA_CAMPAIGN_DATA_ROOT", str(data_root))
    monkeypatch.setenv("AGENT_STATE_MOUNT_ROOT", str(state_root))
    monkeypatch.setattr(
        campaign_isolation,
        "_codex_requirements_identity",
        lambda: (
            Path("/etc/codex/requirements.toml"),
            _runtime_isolation_receipt()["codex_requirements"],
        ),
    )
    command = campaign_isolation.wrap_attempt_command(
        [
            "/bin/sh",
            "-c",
            (
                f"test ! -e {sibling} && "
                f"test ! -e {state_root / 'history.jsonl'} && "
                f"test ! -e {shm_sentinel} && "
                f"! mv {campaign_isolation._CODEX_GPU_BWRAP_TRUSTED_DIR} "
                f"{campaign_isolation._CODEX_GPU_BWRAP_TRUSTED_DIR}-moved && "
                f"! rm {campaign_isolation._CODEX_GPU_BWRAP_TRUSTED_PATH} && "
                f"! printf malicious > "
                f"{campaign_isolation._CODEX_GPU_BWRAP_TRUSTED_PATH} && "
                f"test \"$(cat {source})\" = baseline && "
                f"! printf changed > {source} && "
                f"printf artifact > {artifact_root / 'probe.txt'} && "
                f"printf home > {attempt_home / 'probe.txt'}"
            ),
        ],
        eval_config={
            "campaign": {"comparison": "apex_vs_codex"},
            "campaign_attempt": {"fresh_session": True},
        },
        writable_roots=(artifact_root, attempt_home),
        read_only_roots=(workspace,),
    )
    proc_index = command.index("/proc")
    assert command[proc_index - 1] == "--proc"
    assert "--unshare-pid" in command
    assert "--json-status-fd" in command
    assert "--block-fd" in command

    try:
        process = campaign.subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            pass_fds=campaign_isolation.attempt_command_pass_fds(command),
        )
        campaign_isolation.release_attempt_command_fds(command)
        boundary = campaign_isolation.establish_attempt_boundary(command, process)
        assert boundary is not None
        stdout, stderr = process.communicate(timeout=10)
        cleanup = campaign_isolation.finalize_attempt_boundary(
            process,
            boundary,
            reason="normal_exit",
            terminate=False,
        )

        assert process.returncode == 0, stderr
        assert stdout == ""
        assert cleanup["verified_absent"] is True
        assert cleanup["teardown_mode"] == "natural_exit"
        assert cleanup["namespace_membership_enumeration_completed"] is True
        assert cleanup["namespace_membership_inaccessible_entries_count"] >= 0
        assert cleanup["namespace_membership_scan_complete"] is (
            cleanup["namespace_membership_inaccessible_entries_count"] == 0
        )
        assert cleanup["live_visible_namespace_members_after"] == []
        assert source.read_text(encoding="utf-8") == "baseline\n"
        assert (artifact_root / "probe.txt").read_text(encoding="utf-8") == "artifact"
        assert (attempt_home / "probe.txt").read_text(encoding="utf-8") == "home"
        assert (sibling / "secret.txt").read_text(encoding="utf-8") == "prior attempt\n"
        assert shm_sentinel.read_text(encoding="utf-8") == "host-visible\n"
    finally:
        campaign_isolation.release_attempt_command_fds(command)
        shm_sentinel.unlink(missing_ok=True)


def test_formal_attempt_rejects_broad_and_overlapping_trusted_roots(
    tmp_path, monkeypatch
) -> None:
    data_root = tmp_path / "campaign-data"
    writable = data_root / "run/task/attempt_01/artifacts"
    trusted = tmp_path / "runtime/apex"
    nested = trusted / "dependency"
    writable.mkdir(parents=True)
    nested.mkdir(parents=True)
    monkeypatch.setenv("AGENT_KERNEL_ARENA_CAMPAIGN_DATA_ROOT", str(data_root))
    monkeypatch.setattr(
        campaign_isolation,
        "_codex_requirements_identity",
        lambda: (
            Path("/etc/codex/requirements.toml"),
            _runtime_isolation_receipt()["codex_requirements"],
        ),
    )
    eval_config = {
        "campaign": {"comparison": "apex_vs_codex"},
        "campaign_attempt": {"fresh_session": True},
    }

    with pytest.raises(
        campaign_isolation.CampaignIsolationError,
        match="specific canonical directory",
    ):
        campaign_isolation.wrap_attempt_command(
            ["/bin/true"],
            eval_config=eval_config,
            writable_roots=(writable,),
            trusted_read_only_roots=(Path("/tmp"),),
        )

    with pytest.raises(
        campaign_isolation.CampaignIsolationError,
        match="overlap",
    ):
        campaign_isolation.wrap_attempt_command(
            ["/bin/true"],
            eval_config=eval_config,
            writable_roots=(writable,),
            trusted_read_only_roots=(trusted, nested),
        )


def test_formal_codex_home_copies_only_auth_and_signed_cloud_config(
    tmp_path, monkeypatch
) -> None:
    state = tmp_path / "agent-state/.codex"
    state.mkdir(parents=True)
    account_id = "00000000-0000-0000-0000-000000000001"
    (state / "auth.json").write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {"account_id": account_id},
            }
        ),
        encoding="utf-8",
    )
    now = datetime.now(timezone.utc)
    cache_bytes = json.dumps(
        {
            "signature": base64.b64encode(b"s" * 32).decode("ascii"),
            "signed_payload": {
                "version": 1,
                "cached_at": now.isoformat().replace("+00:00", "Z"),
                "expires_at": (now + timedelta(hours=1)).isoformat().replace(
                    "+00:00", "Z"
                ),
                "chatgpt_user_id": "fixture-user",
                "account_id": account_id,
                "bundle": {
                    "config_toml": {"enterprise_managed": []},
                    "requirements_toml": {"enterprise_managed": []},
                },
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    (state / "cloud-config-bundle-cache.json").write_bytes(cache_bytes)
    (state / "history.jsonl").write_text("prior context\n", encoding="utf-8")
    (state / "config.toml").write_text("model = 'wrong'\n", encoding="utf-8")
    attempt = tmp_path / "campaign/run/attempt_01"
    attempt.mkdir(parents=True)
    monkeypatch.setenv("AGENT_STATE_MOUNT_ROOT", str(state.parent))
    home = campaign_isolation.prepare_attempt_home(
        {
            "campaign": {"comparison": "apex_vs_codex"},
            "campaign_attempt": {
                "fresh_session": True,
                "receipt_path": str(attempt / "session_receipt.json"),
            },
        },
        backend="codex",
    )

    assert home is not None
    assert (home / ".codex/auth.json").is_file()
    assert (home / ".codex/cloud-config-bundle-cache.json").read_bytes() == (
        cache_bytes
    )
    assert not (home / ".codex/history.jsonl").exists()
    assert not (home / ".codex/config.toml").exists()


def _write_campaign_codex_contract(
    run_directory: Path,
    task_names: tuple[str, ...] = ("triton2triton/vllm/example",),
    agent_template: str = "apex",
) -> dict:
    turn_policy = campaign.CANDIDATE_PERSISTENCE_POLICY
    policy = {"attempts": 3}
    run_config_path = run_directory.parent / f"run-config-{agent_template}.yaml"
    run_config_path.write_text(
        yaml.safe_dump(
            {
                "agent": {"template": agent_template},
                "campaign": policy,
                "tasks": list(task_names),
                "target_gpu_model": "MI355X",
                "log_directory": "/test/logs",
                "workspace_directory_prefix": "/test/workspace",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    run_config_path.chmod(0o444)
    run_config = campaign._run_config_contract(
        run_config_path, agent_name=agent_template
    )
    closure = _test_backend_closure()
    codex = {
        "backend": "codex",
        "model": "gpt-5.5",
        "effort": "xhigh",
        "permission_mode": "workspace_write_isolated",
        "inner_max_iterations": 1,
        "attempt_timeout_seconds": 3600,
        "codex_version": "codex-cli test",
        "codex_binary_sha256": "a" * 64,
        "max_turns": 50,
        "turn_policy": turn_policy,
        "agent_process_containment_policy_id": (
            campaign.AGENT_PROCESS_CONTAINMENT_POLICY
        ),
        "attempt_containment_policy_id": campaign.ATTEMPT_CONTAINMENT_POLICY,
        "structured_stream_output_limit_bytes": 16 * 1024 * 1024,
        "backend_runtime_closure_schema": campaign.BACKEND_CLOSURE_SCHEMA,
        "backend_runtime_closure_sha256": closure["closure_sha256"],
        "backend_runtime_closure": closure,
        "cloud_config_bootstrap_schema": (
            campaign.CODEX_CLOUD_CONFIG_BOOTSTRAP_SCHEMA
        ),
        "cloud_config_bootstrap_policy": (
            campaign.CODEX_CLOUD_CONFIG_BOOTSTRAP_POLICY
        ),
        "cloud_config_bundle_sha256": "c" * 64,
        "cloud_config_host_runtime_closure_sha256": "d" * 64,
        "cloud_config_initial_refresh_receipt_sha256": "e" * 64,
        "isolation": {
            "approval": "never_via_strict_config",
            "execpolicy_rules": "ignored",
            "project_instructions": "backend_default_may_load",
            "sandbox": "workspace-write",
            "session": "ephemeral",
            "user_config": "ignored",
            "mount_scope": "attempt_only_bubblewrap",
            "attempt_containment_policy_id": (
                "private_pid_namespace_init_pidfd_v1"
            ),
        },
    }
    pool = os.environ.get("AGENT_KERNEL_ARENA_GPU_POOL", "0").split(",")
    package_root = run_directory.parent / "task_packages"
    package_root.mkdir(exist_ok=True)
    tasks = []
    task_config_paths = {}
    task_mapping = []
    for index, task_name in enumerate(task_names, 1):
        package = package_root / f"task_{index:02d}"
        package.mkdir()
        config_path = package / "config.yaml"
        config_path.write_text(f"task_name: {task_name}\n", encoding="utf-8")
        files = campaign._regular_tree_manifest(package)
        tasks.append(
            {
                "task_index": index,
                "task_name": task_name,
                "config_path": str(config_path.resolve()),
                "config_sha256": campaign._sha256_file(config_path),
                "package_files_sha256": files,
                "package_manifest_sha256": hashlib.sha256(
                    json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            }
        )
        task_config_paths[task_name] = str(config_path.resolve())
        task_mapping.append(
            {
                "task_index": index,
                "task_name": task_name,
                "assigned_host_gpu_id": pool[(index - 1) % len(pool)],
            }
        )
    gpu = {
        "gpu_boundary_plan_sha256": "d" * 64,
        "devices": [
            {
                "host_device_id": host_gpu,
                "unique_id": f"0x{int(host_gpu) + 1:016x}",
                "render_nodes": [f"/dev/dri/renderD{128 + int(host_gpu)}"],
            }
            for host_gpu in pool
        ],
        "exclusivity": {
            "sha256": "e" * 64,
            "exclusivity_verified": True,
        },
        "task_mapping": task_mapping,
    }
    aka_state, aka_runtime = _test_aka_binding()
    repositories = {
        "agent_kernel_arena": aka_state,
        "apex": {
            "commit": "5" * 40,
            "dirty": False,
            "status_sha256": "6" * 64,
            "runtime_manifest_sha256": "7" * 64,
        },
    }
    evaluator_files = {
        "schema": "aka.evaluator-source-binding/v2",
        "coverage": "all_committed_files",
        "execution_manifest_schema": campaign.EXECUTION_MANIFEST_SCHEMA,
        "execution_manifest_sha256": "4" * 64,
        "commit": "1" * 40,
        "tree": "2" * 40,
    }
    apex_treatment = {
        "template": "apex",
        "session_receipt_schema": "agentkernelarena.apex-attempt-receipt/v6",
        "apex_runtime_mount_policy_id": campaign.APEX_RUNTIME_MOUNT_POLICY,
        "attempt_mount_receipt_schema": campaign.ATTEMPT_MOUNT_RECEIPT_SCHEMA,
        "apex_runtime_mount_schema": campaign.APEX_RUNTIME_MOUNT_SCHEMA,
        "runtime_manifest_sha256": repositories["apex"][
            "runtime_manifest_sha256"
        ],
    }
    transport = campaign.FORMAL_AGENT_TRANSPORT_TREATMENTS[agent_template]
    agent = codex | {
        "template": agent_template,
        "max_process_output_bytes": transport["max_process_output_bytes"],
        "structured_stream_overflow_policy": transport["overflow_policy"],
        "session_receipt_schema": (
            "agentkernelarena.codex-attempt-receipt/v6"
            if agent_template == "codex"
            else "agentkernelarena.apex-attempt-receipt/v6"
        ),
    }
    if agent_template == "apex":
        agent |= apex_treatment
    runtime = {"gpu": gpu, "aka_execution_snapshot": aka_runtime}
    comparison_runtime = campaign.comparison_runtime_projection(runtime)
    assert comparison_runtime is not None
    comparison_contract = {
        "schema": "aka.apex-vs-codex-comparison-contract/v6",
        "formal_execution": dict(campaign._FORMAL_LIVE_COMMITMENT),
        "formal_execution_sha256": campaign.FORMAL_LIVE_EXECUTION_SHA256,
        "policy": policy,
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
        "agent_transport_treatments": copy.deepcopy(
            campaign.FORMAL_AGENT_TRANSPORT_TREATMENTS
        ),
        "codex": codex,
        "runtime": comparison_runtime,
        "evaluator_files_sha256": evaluator_files,
        "tasks": tasks,
        "run_config": run_config,
    }
    comparison_digest = hashlib.sha256(
        json.dumps(
            comparison_contract, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    path = run_directory / "campaign_manifest.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema": "aka.matched-campaign/v1",
                "formal_execution": dict(campaign._FORMAL_LIVE_COMMITMENT),
                "formal_execution_sha256": campaign.FORMAL_LIVE_EXECUTION_SHA256,
                "policy": policy,
                "comparison_contract_sha256": comparison_digest,
                "comparison_contract": comparison_contract,
                "repositories": repositories,
                "agent": agent,
                "runtime": runtime,
                "evaluator_files_sha256": evaluator_files,
                "configuration": {
                    "run_config_path": str(run_config_path.resolve()),
                    "run_config_sha256": campaign._sha256_file(run_config_path),
                    "run_config_size_bytes": run_config_path.stat().st_size,
                    "run_config_contract": run_config,
                    "tasks": tasks,
                },
            }
        ),
        encoding="utf-8",
    )
    path.chmod(0o444)
    return codex | {
        "_comparison_contract_sha256": comparison_digest,
        "_task_config_paths": task_config_paths,
        "_run_directory": str(run_directory.resolve()),
        "_checkpoint_policy": True,
        "_apex_receipt_schema": "agentkernelarena.apex-attempt-receipt/v6",
    }


def _fixture_campaign_binding(
    receipt_path: Path, codex_contract: dict, *, attempt_index: int | None = None
) -> dict:
    run = Path(codex_contract["_run_directory"])
    manifest_path = (run / "campaign_manifest.yaml").resolve(strict=True)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    task = manifest["configuration"]["tasks"][0]
    mapping = manifest["runtime"]["gpu"]["task_mapping"][0]
    assert receipt_path.name == "session_receipt.json"
    if attempt_index is None:
        attempt_index = int(receipt_path.parent.name.removeprefix("attempt_"))
    return {
        "schema": "aka.attempt-campaign-binding/v1",
        "formal_execution_sha256": manifest["formal_execution_sha256"],
        "campaign_manifest_path": str(manifest_path),
        "campaign_manifest_sha256": campaign._sha256_file(manifest_path),
        "comparison_contract_sha256": manifest["comparison_contract_sha256"],
        "backend_runtime_closure_sha256": codex_contract[
            "backend_runtime_closure_sha256"
        ],
        "task_package_manifest_sha256": task["package_manifest_sha256"],
        "task_config_sha256": task["config_sha256"],
        "task_name": task["task_name"],
        "task_index": task["task_index"],
        "total_tasks": len(manifest["configuration"]["tasks"]),
        "attempt_index": attempt_index,
        "attempt_count": manifest["policy"]["attempts"],
        "assigned_host_gpu_id": mapping["assigned_host_gpu_id"],
    }


def _fixture_cloud_config_bootstrap(codex_contract: dict) -> dict:
    return {
        "schema": campaign.CODEX_CLOUD_CONFIG_BOOTSTRAP_SCHEMA,
        "policy": campaign.CODEX_CLOUD_CONFIG_BOOTSTRAP_POLICY,
        "relative_path": ".codex/cloud-config-bundle-cache.json",
        "present": True,
        "sha256": "b" * 64,
        "size_bytes": 1024,
        "bundle_sha256": codex_contract["cloud_config_bundle_sha256"],
        "signed_envelope_shape_validated": True,
        "payload_recorded": False,
    }


@pytest.mark.parametrize(
    ("template", "receipt_schema"),
    [
        ("apex", "agentkernelarena.apex-attempt-receipt/v6"),
        ("codex", "agentkernelarena.codex-attempt-receipt/v6"),
    ],
)
def test_formal_campaign_fixture_is_a_complete_live_v6_contract(
    tmp_path, template, receipt_schema
) -> None:
    run = tmp_path / template
    run.mkdir()
    contract = _write_campaign_codex_contract(run, agent_template=template)

    manifest = campaign._load_verified_campaign_manifest(run)
    comparison = manifest["comparison_contract"]

    assert comparison["schema"] == "aka.apex-vs-codex-comparison-contract/v6"
    assert manifest["formal_execution_sha256"] == (
        campaign.FORMAL_LIVE_EXECUTION_SHA256
    )
    assert comparison["formal_execution_sha256"] == (
        campaign.FORMAL_LIVE_EXECUTION_SHA256
    )
    assert manifest["agent"]["session_receipt_schema"] == receipt_schema
    assert comparison["codex"]["backend_runtime_closure_sha256"] == (
        contract["backend_runtime_closure_sha256"]
    )
    assert manifest["repositories"]["agent_kernel_arena"][
        "execution_manifest_sha256"
    ] == manifest["runtime"]["aka_execution_snapshot"]["manifest_sha256"]
    assert comparison["apex_treatment"]["attempt_mount_receipt_schema"] == (
        campaign.ATTEMPT_MOUNT_RECEIPT_SCHEMA
    )


def test_formal_run_config_agent_mapping_must_be_exact(tmp_path: Path) -> None:
    run_config = tmp_path / "run.yaml"
    run_config.write_text(
        yaml.safe_dump(
            {
                "agent": {"template": "codex", "model": "attacker-override"},
                "tasks": ["triton2triton/vllm/example"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        campaign.CampaignError,
        match="agent must contain exactly the selected template",
    ):
        campaign._run_config_contract(run_config, agent_name="codex")


def test_formal_run_config_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.yaml"
    target.write_text(
        yaml.safe_dump(
            {
                "agent": {"template": "codex"},
                "tasks": ["triton2triton/vllm/example"],
            }
        ),
        encoding="utf-8",
    )
    linked = tmp_path / "run.yaml"
    linked.symlink_to(target)

    with pytest.raises(
        campaign.CampaignError,
        match="safe regular file",
    ):
        campaign._run_config_contract(linked, agent_name="codex")


def test_formal_campaign_rejects_raw_run_config_rewrite(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _write_campaign_codex_contract(run, agent_template="codex")
    manifest = yaml.safe_load((run / "campaign_manifest.yaml").read_text())
    run_config = Path(manifest["configuration"]["run_config_path"])
    run_config.chmod(0o644)
    run_config.write_text(
        run_config.read_text(encoding="utf-8") + "unexpected_output: /tmp/forged\n",
        encoding="utf-8",
    )

    with pytest.raises(
        campaign.CampaignError,
        match="campaign manifest comparison contract is invalid",
    ):
        campaign._load_verified_campaign_manifest(run)


def test_formal_campaign_rejects_forged_run_config_projection_digest(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _write_campaign_codex_contract(run, agent_template="codex")
    manifest_path = run / "campaign_manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    forged = copy.deepcopy(manifest["configuration"]["run_config_contract"])
    forged["effective_config_sha256"] = "0" * 64
    manifest["configuration"]["run_config_contract"] = copy.deepcopy(forged)
    manifest["comparison_contract"]["run_config"] = copy.deepcopy(forged)
    manifest["comparison_contract_sha256"] = campaign._canonical_json_digest(
        manifest["comparison_contract"]
    )
    manifest_path.chmod(0o644)
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    manifest_path.chmod(0o444)

    with pytest.raises(
        campaign.CampaignError,
        match="campaign manifest comparison contract is invalid",
    ):
        campaign._load_verified_campaign_manifest(run)


@pytest.mark.parametrize("synchronize_table", [False, True])
def test_formal_campaign_rejects_transport_treatment_drift(
    tmp_path: Path, synchronize_table: bool
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _write_campaign_codex_contract(run, agent_template="apex")
    path = run / "campaign_manifest.yaml"
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    manifest["agent"]["max_process_output_bytes"] = -1
    if synchronize_table:
        manifest["comparison_contract"]["agent_transport_treatments"]["apex"][
            "max_process_output_bytes"
        ] = -1
    manifest["comparison_contract_sha256"] = hashlib.sha256(
        json.dumps(
            manifest["comparison_contract"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    path.chmod(0o644)
    path.write_text(yaml.safe_dump(manifest, sort_keys=True), encoding="utf-8")
    path.chmod(0o444)

    with pytest.raises(campaign.CampaignError, match="comparison contract is invalid"):
        campaign._load_verified_campaign_manifest(run)


def _write_valid_codex_receipt(
    receipt_path: Path,
    codex_contract: dict,
    *,
    exact_boundary: bool = False,
    source_changed: bool = True,
) -> Path:
    artifact_dir = receipt_path.parent / f".{receipt_path.stem}.artifacts"
    artifact_dir.mkdir(parents=True)
    source_metadata = {"sha256": "1" * 64, "size_bytes": 1, "mode": "0644"}
    before_manifest = {"kernel.py": source_metadata}
    after_metadata = (
        {"sha256": "2" * 64, "size_bytes": 2, "mode": "0644"}
        if exact_boundary or source_changed
        else source_metadata
    )
    after_manifest = {"kernel.py": after_metadata}
    checkpoint_policy = codex_contract.get("_checkpoint_policy") is True
    assert checkpoint_policy
    assert not exact_boundary or checkpoint_policy
    artifact_payloads = {
        "rendered_prompt": (
            artifact_dir / "rendered_prompt.txt",
            b"rendered direct Codex prompt\n",
        ),
        "raw_stdout": (artifact_dir / "raw_stdout.jsonl", b"{}\n"),
        "raw_stderr": (artifact_dir / "raw_stderr.txt", b""),
        "formatted_transcript": (
            artifact_dir / "formatted_transcript.txt",
            b"assistant: done\n",
        ),
        "workspace_before_manifest": (
            artifact_dir / "workspace_before_manifest.json",
            json.dumps(before_manifest, sort_keys=True, separators=(",", ":")).encode()
            + b"\n",
        ),
        "workspace_after_manifest": (
            artifact_dir / "workspace_after_manifest.json",
            json.dumps(after_manifest, sort_keys=True, separators=(",", ":")).encode()
            + b"\n",
        ),
    }
    artifacts = {}
    for name, (path, payload) in artifact_payloads.items():
        path.write_bytes(payload)
        path.chmod(0o444)
        artifacts[name] = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
            "mode": "0444",
        }
    artifact_dir.chmod(0o555)
    boundary_snapshot = (
        {
            "policy_id": "private_pid_namespace_init_pidfd_v1",
            "capture_mode": "post_namespace_teardown_checkpoint",
            "manifest_sha256": campaign._canonical_json_digest(after_manifest),
            "files": [
                {
                    "path": "kernel.py",
                    "sha256": after_metadata["sha256"],
                    "size_bytes": after_metadata["size_bytes"],
                }
            ],
            "errors": [],
            "complete": True,
        }
        if exact_boundary
        else None
    )
    output_tail = (
        {
            "policy": "retained_and_digested_v1",
            "stdout_character_offset": 3,
            "stdout_size_bytes": 0,
            "stdout_sha256": hashlib.sha256(b"").hexdigest(),
            "post_boundary_turns": 0,
            "capture_truncated": False,
            "readers_completed": True,
        }
        if exact_boundary
        else None
    )
    boundary = {
        "schema": "aka.attempt-process-boundary/v1",
        "policy": "private_pid_namespace_init_pidfd_v1",
        "pid_namespace_unshared": True,
        "procfs": "private_attempt_procfs",
        "namespace_init_pid": 1234,
        "namespace_init_starttime": 99,
        "namespace_init_parent_pid": 1200,
        "namespace_init_inner_pid": 1,
        "pid_namespace_id": 1001,
        "mount_namespace_id": 1002,
        "ipc_namespace_id": 1003,
        "pidfd_opened": True,
        "identity_source": "pinned_bubblewrap_json_status_fd",
    }
    process_cleanup = {
        "required": exact_boundary,
        "reason": "exact_turn_boundary" if exact_boundary else "normal_exit",
        "scope": "private_pid_namespace",
        "method": "namespace_init_pidfd_v1",
        "boundary": boundary,
        "verification_performed": True,
        "namespace_init_exit_verified": True,
        "namespace_membership_enumeration_completed": True,
        "namespace_membership_scan_complete": True,
        "namespace_membership_inaccessible_entries_count": 0,
        "live_visible_namespace_members_after": [],
        "verified_absent": True,
        "sigkill_sent": exact_boundary,
        "outer_supervisor_force_killed": False,
        "outer_supervisor_exit_code": (
            128 + int(signal.SIGKILL) if exact_boundary else 0
        ),
        "kernel_semantics": "linux_pid_namespace_init_exit_sigkill_all_members",
        "bubblewrap_terminal_status_verified": not exact_boundary,
        "bubblewrap_terminal_status": None if exact_boundary else {"exit-code": 0},
        "bubblewrap_terminal_status_absent_after_sigkill": exact_boundary,
        "bubblewrap_status_eof_verified": True,
        "teardown_mode": "pidfd_sigkill" if exact_boundary else "natural_exit",
    }
    if exact_boundary:
        process_cleanup["boundary_signal"] = {
            "attempted": True,
            "stdout_character_offset": 3,
        }
    receipt = {
        "schema": "agentkernelarena.codex-attempt-receipt/v6",
        "campaign_binding": _fixture_campaign_binding(
            receipt_path, codex_contract
        ),
        "comparison_contract_sha256": codex_contract[
            "_comparison_contract_sha256"
        ],
        "session_succeeded": True,
        "thread_id": "thread-test",
        "session_id": "session-test",
        "exit_code": 128 + int(signal.SIGKILL) if exact_boundary else 0,
        "timed_out": False,
        "effective_timeout_seconds": 3599.0,
        "attempt_process_cleanup": process_cleanup,
        "capture": {
            "readers_completed": True,
            "errors": [],
            "stdout": {
                "limit_bytes": 16 * 1024 * 1024,
                "retained_bytes": 3,
                "discarded_bytes": 0,
                "truncated": False,
            },
            "stderr": {
                "limit_bytes": 16 * 1024 * 1024,
                "retained_bytes": 0,
                "discarded_bytes": 0,
                "truncated": False,
            },
        },
        "turn_budget": {
            "policy": codex_contract["turn_policy"],
            "max_turns": 50,
            "observed_turns": 50 if exact_boundary else 1,
            "budget_exceeded": False,
            "enforcement_failed": False,
            "stop_reason": "exact_turn_boundary" if exact_boundary else None,
        },
        "workspace_integrity": {
            "policy": "declared_source_only_sanitized_v1",
            "editable_files": ["kernel.py"],
            "raw_after_manifest_sha256": hashlib.sha256(
                json.dumps(after_manifest, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "raw_manifest_error": None,
            "raw_changes": {
                "before_manifest_sha256": hashlib.sha256(
                    json.dumps(before_manifest, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "after_manifest_sha256": hashlib.sha256(
                    json.dumps(after_manifest, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "created_files": [],
                "deleted_files": [],
                "changed_files": ["kernel.py"]
                if exact_boundary or source_changed
                else [],
                "unauthorized_changed_files": [],
                "editable_mode_changes": [],
            },
            "sanitization": {
                "performed": True,
                "candidate_retained": True,
                "baseline_restored": True,
            },
            "final_changes": {
                "before_manifest_sha256": hashlib.sha256(
                    json.dumps(before_manifest, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "after_manifest_sha256": hashlib.sha256(
                    json.dumps(after_manifest, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "created_files": [],
                "deleted_files": [],
                "changed_files": ["kernel.py"]
                if exact_boundary or source_changed
                else [],
                "unauthorized_changed_files": [],
                "editable_mode_changes": [],
            },
            "errors": [],
            "passed": True,
        },
        "gpu": {
            "policy": "physical_device_boundary_with_host_exclusivity_v1",
            "plan_sha256": "d" * 64,
            "boundary_receipt_sha256": "f" * 64,
            "exclusivity_receipt_sha256": "e" * 64,
            "exclusivity_verified": True,
            "host_gpu_id": "0",
            "unique_id": "0x0000000000000001",
            "allowed_render_nodes": ["/dev/dri/renderD128"],
            "observed_devices": [],
            "runtime_identity": {
                "visible_physical_gpu_count": 1,
                "rocm_smi_identity": {"unique_id": "0x0000000000000001"},
                "torch": {"device_count": 1},
            },
        },
        "codex": {
            "binary_sha256": codex_contract["codex_binary_sha256"],
            "runtime_closure_sha256": codex_contract[
                "backend_runtime_closure_sha256"
            ],
            "version": codex_contract["codex_version"],
            "model": codex_contract["model"],
            "effort": codex_contract["effort"],
            "cloud_config_bootstrap": _fixture_cloud_config_bootstrap(
                codex_contract
            ),
        },
        "invocation": {
            "argv_without_prompt": [
                "codex",
                "exec",
                "--strict-config",
                "--ignore-user-config",
                "--ignore-rules",
                "--ephemeral",
            ],
            "prompt_sha256": hashlib.sha256(
                artifact_payloads["rendered_prompt"][1]
            ).hexdigest(),
            "workspace": str(receipt_path.parent / "workspace"),
            "editable_files": ["kernel.py"],
            "max_turns": 50,
            "turn_policy": codex_contract["turn_policy"],
            "structured_stream_output_limit_bytes": 16 * 1024 * 1024,
            "isolation": codex_contract["isolation"],
        },
        "aggregated_usage": {"events": 1, "input_tokens": 1, "output_tokens": 1},
        "artifacts": artifacts,
    }
    if checkpoint_policy:
        receipt["turn_budget"] |= {
            "exact_boundary_reached": exact_boundary,
            "post_boundary_turns": 0,
        }
        receipt["invocation"]["candidate_persistence_policy_id"] = (
            "structured_agent_turn_checkpoint_v2"
        )
        receipt["invocation"]["agent_process_containment_policy_id"] = (
            "private_pid_namespace_init_pidfd_v1"
        )
        receipt["invocation"]["attempt_containment_policy_id"] = (
            "private_pid_namespace_init_pidfd_v1"
        )
        receipt["invocation"]["attempt_process_boundary"] = boundary
        receipt["candidate_persistence"] = {
            "schema": "aka.candidate-persistence-receipt/v4",
            "policy_id": "structured_agent_turn_checkpoint_v2",
            "attempt_contained": True,
            "agent_process_containment_policy_id": (
                "private_pid_namespace_init_pidfd_v1"
            ),
            "attempt_containment_policy_id": (
                "private_pid_namespace_init_pidfd_v1"
            ),
            "termination": (
                "exact_turn_boundary" if exact_boundary else "completed"
            ),
            "checkpoint": (
                {
                    "before_manifest_sha256": hashlib.sha256(
                        json.dumps(
                            before_manifest, sort_keys=True, separators=(",", ":")
                        ).encode()
                    ).hexdigest(),
                    "after_manifest_sha256": hashlib.sha256(
                        json.dumps(
                            after_manifest, sort_keys=True, separators=(",", ":")
                        ).encode()
                    ).hexdigest(),
                    "changed_files": ["kernel.py"],
                    "editable_files": ["kernel.py"],
                    "boundary_snapshot_sha256": campaign._canonical_json_digest(
                        boundary_snapshot
                    ),
                    "output_tail_sha256": campaign._canonical_json_digest(
                        output_tail
                    ),
                    "attempt_cleanup_sha256": campaign._canonical_json_digest(
                        process_cleanup
                    ),
                }
                if exact_boundary
                else None
            ),
            "boundary_snapshot": boundary_snapshot,
            "output_tail": output_tail,
            "boundary_resolution": (
                "pid_namespace_destroyed_before_snapshot"
                if exact_boundary
                else None
            ),
            "attempt_cleanup": process_cleanup if exact_boundary else None,
        }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    receipt_path.chmod(0o444)
    return artifact_payloads["raw_stdout"][0]


def _context_packet_prompt(objective: str) -> bytes:
    identity_and_role = {
        "identity": {"context_packet_id": "context-fixture"},
        "role": {"kind": "kernel_optimizer", "objective": objective},
    }
    encoded = json.dumps(
        identity_and_role,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        "# Apex ContextPacket\n\n"
        "This packet is the complete task-local observation for this invocation.\n\n"
        "## Identity and role\n\n"
        f"> {encoded}\n\n"
        "## Objective and target\n\n"
        "> {}\n"
    ).encode("utf-8")


def _write_valid_apex_receipt(
    receipt_path: Path,
    codex_contract: dict,
    *,
    max_turns: int = 50,
    status: str = "no_gain",
    run_control_turns: int | None = None,
    omit_run_control_suffix: bool = False,
    budget_turn_count: int | None = None,
    budget_reason_override: str | None = None,
    inner_exit_code: int = 0,
    omit_run_control_from_agent_prompt: bool = False,
    successful_turn_count: int | None = None,
    containment_terminal_overlap: bool = False,
) -> None:
    assert status in {"candidate_ready", "no_gain", "budget_exhausted"}
    assert (
        codex_contract.get("_apex_receipt_schema")
        == "agentkernelarena.apex-attempt-receipt/v6"
    )
    turn_policy = "structured_agent_turn_checkpoint_v2"
    failed = status == "budget_exhausted"
    candidate_ready = status == "candidate_ready"
    artifact_dir = receipt_path.parent / f".{receipt_path.stem}.artifacts"
    artifact_dir.mkdir(parents=True)
    baseline_hashes = {"source/kernel.py": "1" * 64}
    original_prompt = "Original Arena task prompt.\n".encode()
    adapted_instructions = "Adapted Apex task instructions."
    instruction_adaptation = {
        "schema": "aka.apex-instruction-adaptation/v1",
        "strategy": "test_v1",
        "original": {
            "characters": len(original_prompt.decode()),
            "bytes": len(original_prompt),
            "sha256": hashlib.sha256(original_prompt).hexdigest(),
        },
        "adapted": {
            "characters": len(adapted_instructions),
            "bytes": len(adapted_instructions.encode()),
            "sha256": hashlib.sha256(adapted_instructions.encode()).hexdigest(),
        },
    }
    interpreter_path = "/opt/venv/bin/python"
    verifier_argv = {
        phase: [interpreter_path, "scripts/task_runner.py", phase]
        for phase in ("compile", "correctness", "performance")
    }
    caller_run_control = {
        "schema": "aka.apex-caller-run-control/v1",
        "deliverable_versions": 1,
        "structured_turn_budget": {
            "policy": turn_policy,
            "max_turns": (
                max_turns if run_control_turns is None else run_control_turns
            ),
            "counting": "assistant_message_and_tool_call_start_each_count_once",
        },
        "process_containment_policy_id": "private_pid_namespace_init_pidfd_v1",
        "python_interpreter": {
            "environment_variable": "AGENT_KERNEL_ARENA_PYTHON",
            "path": interpreter_path,
            "resolved_path": interpreter_path,
            "sha256": "a" * 64,
        },
        "verifier_argv": verifier_argv,
        "candidate_persistence_policy_id": "structured_agent_turn_checkpoint_v2",
    }
    if not omit_run_control_suffix:
        adapted_instructions = (
            f"{adapted_instructions}\n\n"
            f"{campaign.render_apex_run_control(caller_run_control)}"
        )
    adapted_bytes = adapted_instructions.encode()
    instruction_adaptation["adapted"] = {
        "characters": len(adapted_instructions),
        "bytes": len(adapted_bytes),
        "sha256": hashlib.sha256(adapted_bytes).hexdigest(),
    }
    invocation = {
        "schema": "apex.agent-invocation/v3",
        "cli_name": "codex",
        "cli_version": codex_contract["codex_version"],
        "executable_path": "/usr/bin/codex",
        "resolved_executable_path": "/usr/bin/codex",
        "entrypoint_sha256": codex_contract["codex_binary_sha256"],
        "runtime_closure_sha256": codex_contract[
            "backend_runtime_closure_sha256"
        ],
        "argv": [
            "/usr/bin/codex",
            "exec",
            "--strict-config",
            "--ignore-user-config",
            "--ignore-rules",
            "--ephemeral",
        ],
        "workspace": "/attempt/apex-workspace",
        "prompt_transport": "stdin",
        "requested_allowed_files": ["source/kernel.py"],
        "allowed_files_enforced_by_cli": False,
        "max_turns": max_turns,
        "turn_policy": turn_policy,
        "isolation": {
            key: value
            for key, value in codex_contract["isolation"].items()
            if key != "mount_scope"
        }
        | {"response_token_limit": "not_supported_context_advisory_only"},
    }
    invocation["process_containment_policy_id"] = (
        "private_pid_namespace_init_pidfd_v1"
    )
    selected_turn_count = (
        max_turns + 1 if budget_turn_count is None else budget_turn_count
    ) if failed else (
        (1 if successful_turn_count is None else successful_turn_count)
    )
    semantic_events = [
        {"kind": "agent_message", "index": index}
        for index in range(selected_turn_count)
    ]
    observed_turns = len(semantic_events)
    effective_inner_exit_code = (
        128 + signal.SIGKILL
        if failed and inner_exit_code == 0
        else inner_exit_code
    )
    bwrap = Path(shutil.which("bwrap") or "/usr/bin/bwrap").resolve()
    process_containment = {
        "schema": "apex.agent-process-containment/v1",
        "policy_id": "private_pid_namespace_init_pidfd_v1",
        "launcher_path": str(bwrap),
        "launcher_sha256": campaign._sha256_file(bwrap),
        "namespace_init_host_pid": 4321,
        "namespace_init_starttime": 100,
        "namespace_init_inner_pid": 1,
        "pid_namespace_inode": 2001,
        "mount_namespace_inode": 2002,
        "ipc_namespace_inode": 2003,
        "user_namespace_inode": 2004,
        "private_procfs_verified": True,
        "pidfd_opened": True,
        "termination_reason": "stdout_budget_boundary" if failed else "natural_exit",
        "teardown_mode": (
            "pidfd_sigkill" if failed else "natural_exit"
        ),
        "pidfd_sigkill_sent": failed,
        "namespace_init_exit_verified": True,
        "wrapper_exit_verified": True,
        "wrapper_force_killed": False,
        "terminal_status_verified": (
            not failed or containment_terminal_overlap
        ),
        "terminal_status_absent_after_sigkill": failed,
        "status_eof_verified": True,
        "namespace_membership_scan_complete": True,
        "live_namespace_members_after": [],
        "namespace_empty_verified": True,
    }
    budget_reason = None
    if failed:
        budget_reason = budget_reason_override or "max_turns_overrun"
    transcript = {
        "schema": "apex.agent-transcript/v3",
        "backend": "codex",
        "model": codex_contract["model"],
        "effort": codex_contract["effort"],
        "invocation": invocation,
        "events": [],
        "semantic_events": semantic_events,
        "usage": None,
        "cost": None,
        "termination": {
            "kind": "turn_overrun" if failed else "completed",
            "reason": budget_reason,
            "capture_status": "complete",
            "candidate_capture_allowed": not failed,
            "observer_stop_sent": failed,
            "process_containment": process_containment,
            "discarded_stdout_tail": {
                "lines": 0,
                "bytes": 0,
                "sha256": None,
            },
            "observed_turns": observed_turns,
            "max_turns": max_turns,
            "turn_policy": turn_policy,
        },
    }
    transcript_content = json.dumps(transcript).encode()
    transcript_digest = hashlib.sha256(transcript_content).hexdigest()
    agent_payload = {
        "backend": "codex",
        "model": codex_contract["model"],
        "effort": codex_contract["effort"],
        "exit_code": effective_inner_exit_code,
        "timed_out": False,
        "budget_exceeded": failed,
        "budget_enforcement_failed": False,
        "invocation": invocation,
        "artifacts": [
            {
                "role": "agent_transcript",
                "receipt": {
                    "digest": transcript_digest,
                    "size": len(transcript_content),
                },
            }
        ],
        "budget_reason": budget_reason,
        "observed_turns": observed_turns,
        "message_event_count": observed_turns,
        "tool_call_event_count": 0,
        "semantic_event_count": observed_turns,
        "attempt_id": "attempt-test",
        "termination_kind": "turn_overrun" if failed else "completed",
        "termination_reason": budget_reason,
        "capture_status": "complete",
        "candidate_capture_allowed": not failed,
        "observer_stop_sent": failed,
        "process_containment_policy_id": "private_pid_namespace_init_pidfd_v1",
        "process_containment": process_containment,
        "discarded_stdout_lines": 0,
        "discarded_stdout_bytes": 0,
        "discarded_stdout_sha256": None,
    }
    prompt_objective = (
        "Adapted Apex task instructions."
        if omit_run_control_from_agent_prompt
        else adapted_instructions
    )
    rendered_prompt = _context_packet_prompt(prompt_objective)
    rendered_prompt_digest = hashlib.sha256(rendered_prompt).hexdigest()
    prompt_payload = {
        "attempt_id": "attempt-test",
        "artifacts": [
            {
                "role": "prompt",
                "receipt": {
                    "digest": rendered_prompt_digest,
                    "size": len(rendered_prompt),
                },
            }
        ],
    }
    verdict = "reject" if failed else "keep" if candidate_ready else "revert"
    reason = (
        "agent_turn_budget_overrun"
        if failed
        else "candidate_ready" if candidate_ready else "baseline_is_best"
    )
    event_values = [
        (1, "evt-prompt", "prompt_sent", prompt_payload, None, "txn-prompt"),
        (
            2,
            "evt-agent",
            "agent_failed" if failed else "agent_completed",
            agent_payload,
            "evt-prompt",
            "txn-agent",
        ),
        (
            3,
            "evt-decision",
            "decision",
            {"verdict": verdict, "reason": reason},
            "evt-agent",
            "txn-decision",
        ),
        (
            4,
            "evt-run",
            "run.failed" if failed else "run.succeeded",
            {"reason": reason},
            "evt-decision",
            "txn-run",
        ),
    ]
    events = []
    for sequence, event_id, event_type, payload, parent, transaction_id in event_values:
        material = {
            "sequence": sequence,
            "event_id": event_id,
            "run_id": "run-test",
            "event_type": event_type,
            "payload": payload,
            "parent_event_id": parent,
            "idempotency_key": f"key-{sequence}",
            "transaction_id": transaction_id,
            "created_at_ns": sequence,
        }
        events.append(material | {"checksum": campaign._canonical_json_digest(material)})
    journal = artifact_dir / "event_journal.sqlite"
    connection = sqlite3.connect(journal)
    try:
        connection.executescript(
            """
            CREATE TABLE transactions (
              transaction_id TEXT PRIMARY KEY, first_sequence INTEGER,
              last_sequence INTEGER, event_count INTEGER, checksum TEXT
            );
            CREATE TABLE events (
              sequence INTEGER PRIMARY KEY, event_id TEXT, run_id TEXT,
              event_type TEXT, payload_json TEXT, parent_event_id TEXT,
              idempotency_key TEXT, transaction_id TEXT, created_at_ns INTEGER,
              checksum TEXT
            );
            """
        )
        for event in events:
            connection.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event["sequence"],
                    event["event_id"],
                    event["run_id"],
                    event["event_type"],
                    json.dumps(
                        event["payload"], sort_keys=True, separators=(",", ":")
                    ),
                    event["parent_event_id"],
                    event["idempotency_key"],
                    event["transaction_id"],
                    event["created_at_ns"],
                    event["checksum"],
                ),
            )
            tx_checksum = campaign._canonical_json_digest(
                {
                    "transaction_id": event["transaction_id"],
                    "event_checksums": [event["checksum"]],
                }
            )
            connection.execute(
                "INSERT INTO transactions VALUES (?, ?, ?, ?, ?)",
                (
                    event["transaction_id"],
                    event["sequence"],
                    event["sequence"],
                    1,
                    tx_checksum,
                ),
            )
        connection.commit()
    finally:
        connection.close()
    task_spec = {
        "task_id": "task-test",
        "campaign_binding": _fixture_campaign_binding(
            receipt_path, codex_contract
        ),
        "agent_backend": "codex",
        "agent_options": {
            "model": codex_contract["model"],
            "effort": codex_contract["effort"],
        },
        "budget": {
            "max_iterations": 1,
            "max_turns": max_turns,
            "timeout_seconds": 3600,
        },
        "baseline": {"file_hashes": baseline_hashes},
        "instructions": adapted_instructions,
        "instruction_adaptation": instruction_adaptation,
        "commands": {
            phase: {"argv": argv, "timeout_seconds": 3600}
            for phase, argv in verifier_argv.items()
        },
        "caller_run_control": caller_run_control,
    }
    event_artifact_digests = sorted(
        {
            binding["receipt"]["digest"]
            for event in events
            for binding in event["payload"].get("artifacts", [])
        }
    )
    declared_result_digests = [transcript_digest]
    reason_code = (
        "agent_turn_budget_overrun"
        if failed
        else (
            "candidate_verified_for_external_evaluation"
            if candidate_ready
            else "baseline_is_best"
        )
    )
    result = {
        "schema_version": 1,
        "run_id": "run-test",
        "task_id": "task-test",
        "status": status,
        "reason_code": reason_code,
        "applied": False,
        "external_verification_required": True,
        "bundle_path": "/attempt/bundle" if candidate_ready else None,
        "bundle_digest": "b" * 64 if candidate_ready else None,
        "changed_files": ["source/kernel.py"] if candidate_ready else [],
        "baseline_lock": {
            "resolution_hash": "2" * 64,
            "file_hashes": baseline_hashes,
        },
        "internal_verdict": (
            "reject" if failed else "keep" if candidate_ready else "revert"
        ),
        "internal_verdict_ref": "evt-decision",
        "event_journal_ref": {
            "path": "/attempt/events.sqlite",
            "head_event_id": events[-1]["event_id"],
            "head_checksum": events[-1]["checksum"],
        },
        "artifact_store_ref": {
            "path": "/attempt/artifacts",
            "receipt_digests": declared_result_digests,
        },
        "error": {"reason_code": reason_code} if failed else None,
    }
    source_bundle_snapshot: bytes | None = None
    bundle_summary: dict[str, object] | None = None
    if candidate_ready:
        patch_content = (
            b"--- a/source/kernel.py\n"
            b"+++ b/source/kernel.py\n"
            b"@@ -1 +1 @@\n"
            b"-baseline\n"
            b"+optimized\n"
        )
        patch_path = "patches/source_kernel.py.patch"
        patch_sha256 = hashlib.sha256(patch_content).hexdigest()
        bundle_manifest = {
            "schema_version": 1,
            "task_id": "task-test",
            "baseline": {"file_hashes": baseline_hashes},
            "changed_files": ["source/kernel.py"],
            "patches": [
                {
                    "path": patch_path,
                    "sha256": patch_sha256,
                }
            ],
            "delivery": {"mode": "bundle", "applied": False},
        }
        bundle_hasher = hashlib.sha256(
            json.dumps(
                bundle_manifest,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        bundle_hasher.update(patch_content)
        bundle_digest = bundle_hasher.hexdigest()
        source_bundle_snapshot = json.dumps(
            {
                "schema": "aka.apex-source-bundle-snapshot/v1",
                "bundle_digest": bundle_digest,
                "manifest": bundle_manifest,
                "patches": [
                    {
                        "path": patch_path,
                        "sha256": patch_sha256,
                        "size_bytes": len(patch_content),
                        "content_base64": base64.b64encode(patch_content).decode(
                            "ascii"
                        ),
                    }
                ],
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        bundle_summary = {
            "bundle_digest": bundle_digest,
            "snapshot_sha256": hashlib.sha256(source_bundle_snapshot).hexdigest(),
            "snapshot_size_bytes": len(source_bundle_snapshot),
        }
        result["bundle_digest"] = bundle_digest
    payloads = {
        "task_spec": ("task_spec.json", json.dumps(task_spec).encode()),
        "apex_stdout": ("apex_stdout.txt", b"done\n"),
        "apex_stderr": ("apex_stderr.txt", b""),
        "apex_result": ("apex_result.json", json.dumps(result).encode()),
        "event_journal": ("event_journal.sqlite", journal.read_bytes()),
        "agent_transcript": (
            "agent_transcript.json",
            transcript_content,
        ),
    }
    if source_bundle_snapshot is not None:
        payloads["source_bundle"] = (
            "source_bundle_snapshot.json",
            source_bundle_snapshot,
        )
    payloads["original_arena_prompt"] = (
        "original_arena_prompt.txt",
        original_prompt,
    )
    payloads["agent_prompt"] = (
        "agent_prompt.txt",
        rendered_prompt,
    )
    artifacts = {}
    for name, (filename, content) in payloads.items():
        path = artifact_dir / filename
        if name != "event_journal":
            path.write_bytes(content)
        path.chmod(0o444)
        artifacts[name] = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
            "mode": "0444",
        }
    task_spec_contract_root = receipt_path.parent / f".{receipt_path.stem}.contract"
    task_spec_contract_root.mkdir()
    task_spec_contract_path = task_spec_contract_root / "task_spec.json"
    task_spec_contract_path.write_bytes(payloads["task_spec"][1])
    task_spec_contract_path.chmod(0o444)
    task_spec_contract_root.chmod(0o555)
    artifact_dir.chmod(0o555)
    lineage = {
        "run_id": "run-test",
        "result_sha256": artifacts["apex_result"]["sha256"],
        "campaign_binding_sha256": campaign._canonical_json_digest(
            task_spec["campaign_binding"]
        ),
        "journal_head_event_id": events[-1]["event_id"],
        "journal_head_checksum": events[-1]["checksum"],
        "event_count": len(events),
        "transcript_sha256": artifacts["agent_transcript"]["sha256"],
        "event_artifact_digests": event_artifact_digests,
        "internal_verdict": (
            "reject" if failed else "keep" if candidate_ready else "revert"
        ),
        "internal_verdict_ref": "evt-decision",
        "prompt_event": {
            "binding": "apex.prompt_sent_event_cas/v1",
            "event_id": "evt-prompt",
            "sha256": rendered_prompt_digest,
            "size_bytes": len(rendered_prompt),
            "artifact_path": "/attempt/artifacts/rendered_prompt.txt",
            "stdin_transport_attested": False,
        },
    }
    if bundle_summary is not None:
        lineage["bundle"] = bundle_summary
    receipt = {
        "schema": "agentkernelarena.apex-attempt-receipt/v6",
        "campaign_binding": task_spec["campaign_binding"],
        "comparison_contract_sha256": codex_contract[
            "_comparison_contract_sha256"
        ],
        "session_succeeded": not failed,
        "terminal_status": status,
        "exit_code": 1 if failed else 0,
        "timed_out": False,
        "budgets": {
            "inner_agent_timeout_seconds": 3600,
            "apex_internal_allowance_seconds": 3600,
            "outer_timeout_seconds": 7200.0,
            "effective_outer_timeout_seconds": 7200.0,
        },
        "agent_process_containment_policy_id": (
            "private_pid_namespace_init_pidfd_v1"
        ),
        "attempt_containment_policy_id": "private_pid_namespace_init_pidfd_v1",
        "attempt_process_cleanup": {
            "required": False,
            "reason": "normal_exit",
            "scope": "private_pid_namespace",
            "method": "namespace_init_pidfd_v1",
            "boundary": {
                "schema": "aka.attempt-process-boundary/v1",
                "policy": "private_pid_namespace_init_pidfd_v1",
                "pid_namespace_unshared": True,
                "procfs": "trusted_orchestrator_inherited_procfs",
                "namespace_init_pid": 1234,
                "namespace_init_starttime": 99,
                "namespace_init_parent_pid": 1200,
                "namespace_init_inner_pid": 1,
                "pid_namespace_id": 1001,
                "mount_namespace_id": 1002,
                "ipc_namespace_id": 1003,
                "pidfd_opened": True,
                "identity_source": "pinned_bubblewrap_json_status_fd",
            },
            "verification_performed": True,
            "namespace_init_exit_verified": True,
            "namespace_membership_enumeration_completed": True,
            "namespace_membership_scan_complete": True,
            "namespace_membership_inaccessible_entries_count": 0,
            "live_visible_namespace_members_after": [],
            "verified_absent": True,
            "sigkill_sent": False,
            "outer_supervisor_force_killed": False,
            "outer_supervisor_exit_code": 1 if failed else 0,
            "kernel_semantics": "linux_pid_namespace_init_exit_sigkill_all_members",
            "bubblewrap_terminal_status_verified": True,
            "bubblewrap_terminal_status": {"exit-code": 1 if failed else 0},
            "bubblewrap_terminal_status_absent_after_sigkill": False,
            "bubblewrap_status_eof_verified": True,
            "teardown_mode": "natural_exit",
        },
        "capture": {"readers_completed": True, "errors": []},
        "gpu": {
            "policy": "physical_device_boundary_with_host_exclusivity_v1",
            "plan_sha256": "d" * 64,
            "boundary_receipt_sha256": "f" * 64,
            "exclusivity_receipt_sha256": "e" * 64,
            "exclusivity_verified": True,
            "host_gpu_id": "0",
            "unique_id": "0x0000000000000001",
            "allowed_render_nodes": ["/dev/dri/renderD128"],
            "observed_devices": [],
            "runtime_identity": {
                "visible_physical_gpu_count": 1,
                "rocm_smi_identity": {"unique_id": "0x0000000000000001"},
                "torch": {"device_count": 1},
            },
        },
        "apex": {},
        "task_spec_sha256": artifacts["task_spec"]["sha256"],
        "task_spec_contract": {
            "policy": "prelaunch_read_only_sibling_bind_v1",
            "path": str(task_spec_contract_path.resolve()),
            "sha256": artifacts["task_spec"]["sha256"],
            "size_bytes": len(payloads["task_spec"][1]),
            "file_mode": "0444",
            "directory_mode": "0555",
            "read_only_bind": True,
            "postlaunch_unchanged": True,
        },
        "outer_isolation": codex_contract["isolation"],
        "workspace_integrity": {
            "policy": "read_only_until_adapter_bundle_apply_v1",
            "baseline_manifest_sha256": "9" * 64,
            "pre_apply_manifest_sha256": "9" * 64,
            "pre_apply_unchanged": True,
        },
        "codex": {
            "binary_sha256": codex_contract["codex_binary_sha256"],
            "runtime_closure_sha256": codex_contract[
                "backend_runtime_closure_sha256"
            ],
            "version": codex_contract["codex_version"],
            "model": codex_contract["model"],
            "effort": codex_contract["effort"],
            "cloud_config_bootstrap": _fixture_cloud_config_bootstrap(
                codex_contract
            ),
        },
        "invocation": invocation,
        "lineage": lineage,
        "artifacts": artifacts,
        "instruction_adaptation": instruction_adaptation,
    }
    receipt["candidate_persistence"] = {
        "schema": "aka.candidate-persistence-receipt/v4",
        "policy_id": "structured_agent_turn_checkpoint_v2",
        "agent_process_containment_policy_id": (
            "private_pid_namespace_init_pidfd_v1"
        ),
        "agent_process_containment_sha256": (
            campaign._canonical_json_digest(process_containment)
        ),
        "attempt_containment_policy_id": (
            "private_pid_namespace_init_pidfd_v1"
        ),
        "attempt_process_cleanup_sha256": campaign._canonical_json_digest(
            receipt["attempt_process_cleanup"]
        ),
        "termination_kind": "turn_overrun" if failed else "completed",
        "termination_reason": budget_reason,
        "capture_status": "complete",
        "candidate_capture_allowed": not failed,
        "observer_stop_sent": failed,
        "discarded_stdout_tail": {
            "lines": 0,
            "bytes": 0,
            "sha256": None,
        },
        "observed_turns": observed_turns,
        "checkpoint": None,
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    receipt_path.chmod(0o444)


def _unlock_apex_receipt_directories(root: Path) -> None:
    for directory in root.rglob(".session_receipt.artifacts"):
        directory.chmod(0o700)
    for directory in root.rglob(".session_receipt.contract"):
        directory.chmod(0o700)


def test_three_fresh_sessions_are_centrally_ranked_with_stable_tie_break(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("AGENT_KERNEL_ARENA_HOST_GPU_ID", "0")
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    codex_contract = _write_campaign_codex_contract(run_directory)
    observed: list[Path] = []
    times = [2.0, 1.0, 1.0]

    def single_attempt(**kwargs):
        attempt_run = Path(kwargs["run_directory"])
        workspace = attempt_run / "triton2triton_vllm_example_20260807_000000"
        workspace.mkdir()
        attempt = int(kwargs["eval_config"]["campaign_attempt"]["index"])
        (workspace / "solution.py").write_text(f"attempt = {attempt}\n", encoding="utf-8")
        _write_result(workspace, times[attempt - 1])
        observed.append(workspace)
        return True, workspace

    completed, canonical = campaign.run_matched_task_campaign(
        eval_config={"campaign": _policy(), "assigned_host_gpu_id": "0"},
        agent=object(),
        agent_launcher=object(),
        task_name="triton2triton/vllm/example",
        task_config_dir=codex_contract["_task_config_paths"][
            "triton2triton/vllm/example"
        ],
        run_directory=run_directory,
        timestamp="20260807_000000",
        logger=logging.getLogger(__name__),
        task_index=1,
        total_tasks=1,
        single_attempt=single_attempt,
    )

    assert completed is True
    assert canonical is not None
    assert len(observed) == 3
    assert len({workspace.parent for workspace in observed}) == 3
    assert (canonical / "solution.py").read_text(encoding="utf-8") == "attempt = 2\n"
    selected = yaml.safe_load((canonical / "task_result.yaml").read_text())
    assert selected["campaign_evidence"]["selected_attempt"] == 2
    assert selected["campaign_evidence"]["is_apex_canonical_300_sample_grade"] is False
    for evidence_name in (
        "baseline_perf.yaml",
        "optimized_perf.yaml",
        "task_result.yaml",
    ):
        assert (canonical / evidence_name).stat().st_mode & 0o777 == 0o444
    attempts = yaml.safe_load(
        (
            run_directory
            / ".campaign_attempts"
            / _DEFAULT_TASK_COMPONENT
            / "task_campaign.yaml"
        ).read_text()
    )
    assert (
        run_directory
        / ".campaign_attempts"
        / _DEFAULT_TASK_COMPONENT
        / "task_campaign.yaml"
    ).stat().st_mode & 0o777 == 0o444
    assert attempts["all_attempts_centrally_evaluated"] is True
    assert [record["attempt"] for record in attempts["attempts"]] == [1, 2, 3]


def test_any_ineligible_attempt_prevents_canonical_campaign_projection(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("AGENT_KERNEL_ARENA_HOST_GPU_ID", "0")
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    contract = _write_campaign_codex_contract(run_directory)
    times = [0.01, 1.0, 0.5]

    def single_attempt(**kwargs):
        attempt_run = Path(kwargs["run_directory"])
        workspace = attempt_run / "triton2triton_vllm_example_20260807_000000"
        workspace.mkdir()
        attempt = int(kwargs["eval_config"]["campaign_attempt"]["index"])
        _write_result(workspace, times[attempt - 1])
        if attempt == 1:
            optimized_path = workspace / "optimized_perf.yaml"
            optimized = yaml.safe_load(optimized_path.read_text(encoding="utf-8"))
            optimized["test_cases"][0]["benchmark_method"] = "cuda_graph"
            optimized_path.write_text(
                yaml.safe_dump(optimized, sort_keys=False), encoding="utf-8"
            )
            report_path = workspace / "task_result.yaml"
            report = yaml.safe_load(report_path.read_text(encoding="utf-8"))
            report["optimized_benchmark_methods"] = ["cuda_graph"]
            report["benchmark_method_consistent"] = False
            report_path.write_text(
                yaml.safe_dump(report, sort_keys=False), encoding="utf-8"
            )
        return True, workspace

    completed, canonical = campaign.run_matched_task_campaign(
        eval_config={"campaign": _policy(), "assigned_host_gpu_id": "0"},
        agent=object(),
        agent_launcher=object(),
        task_name="triton2triton/vllm/example",
        task_config_dir=contract["_task_config_paths"][
            "triton2triton/vllm/example"
        ],
        run_directory=run_directory,
        timestamp="20260807_000000",
        logger=logging.getLogger(__name__),
        task_index=1,
        total_tasks=1,
        single_attempt=single_attempt,
    )

    assert completed is False
    assert canonical is None
    evidence = yaml.safe_load(
        (
            run_directory
            / ".campaign_attempts"
            / _DEFAULT_TASK_COMPONENT
            / "task_campaign.yaml"
        ).read_text(encoding="utf-8")
    )
    assert evidence["selected_attempt"] == 3
    assert evidence["attempts"][0]["selection_eligible"] is False
    assert "benchmark_method_inconsistent" in evidence["attempts"][0][
        "eligibility_errors"
    ]
    assert "per_testcase_benchmark_method_mismatch" in evidence["attempts"][0][
        "eligibility_errors"
    ]
    assert all(
        attempt["selection_eligible"] is True for attempt in evidence["attempts"][1:]
    )
    assert evidence["all_attempts_centrally_evaluated"] is True
    assert evidence["all_agent_sessions_succeeded"] is True
    assert evidence["failure_reasons"] == [
        "attempt_1:benchmark_method_inconsistent",
        "attempt_1:per_testcase_benchmark_method_mismatch",
    ]


def test_missing_central_report_is_retained_and_invalidates_campaign(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("AGENT_KERNEL_ARENA_HOST_GPU_ID", "0")
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    codex_contract = _write_campaign_codex_contract(run_directory)

    def single_attempt(**kwargs):
        attempt_run = Path(kwargs["run_directory"])
        workspace = attempt_run / "triton2triton_vllm_example_20260807_000000"
        workspace.mkdir()
        attempt = int(kwargs["eval_config"]["campaign_attempt"]["index"])
        (workspace / "solution.py").write_text(f"attempt = {attempt}\n", encoding="utf-8")
        if attempt != 2:
            _write_result(workspace, float(attempt))
        return attempt != 2, workspace

    completed, canonical = campaign.run_matched_task_campaign(
        eval_config={"campaign": _policy(), "assigned_host_gpu_id": "0"},
        agent=object(),
        agent_launcher=object(),
        task_name="triton2triton/vllm/example",
        task_config_dir=codex_contract["_task_config_paths"][
            "triton2triton/vllm/example"
        ],
        run_directory=run_directory,
        timestamp="20260807_000000",
        logger=logging.getLogger(__name__),
        task_index=1,
        total_tasks=1,
        single_attempt=single_attempt,
    )

    assert completed is False
    assert canonical is None
    evidence = yaml.safe_load(
        (
            run_directory
            / ".campaign_attempts"
            / _DEFAULT_TASK_COMPONENT
            / "task_campaign.yaml"
        ).read_text()
    )
    assert evidence["attempts"][1]["central_evaluator_report"] is None
    assert evidence["attempts"][1]["workspace"].endswith(
        "attempt_02/triton2triton_vllm_example_20260807_000000"
    )


def test_manifest_names_native_100_repetition_score_not_apex_grade(
    tmp_path, monkeypatch
) -> None:
    run_config = tmp_path / "run.yaml"
    task_config = tmp_path / "task.yaml"
    run_config.write_text(
        "agent:\n  template: codex\ncampaign: {}\n", encoding="utf-8"
    )
    task_config.write_text("task_type: triton2triton\n", encoding="utf-8")
    monkeypatch.setenv(
        "AGENT_KERNEL_ARENA_DOCKER_IMAGE_ID", "sha256:" + "1" * 64
    )
    monkeypatch.setenv("AGENT_KERNEL_ARENA_DOCKER_IMAGE", "example@sha256:one")
    monkeypatch.setenv(
        "AGENT_KERNEL_ARENA_DOCKER_REPO_DIGESTS",
        '["example@sha256:' + "2" * 64 + '"]',
    )
    monkeypatch.setattr(
        campaign, "_aka_state_from_environment", lambda _root: _test_aka_binding()
    )
    monkeypatch.setattr(
        campaign,
        "_apex_state_from_environment",
        lambda: {
            "commit": "c" * 40,
            "dirty": False,
            "status_sha256": "d" * 64,
            "runtime_manifest_sha256": "9" * 64,
        },
    )
    monkeypatch.setattr(
        campaign,
        "_agent_manifest",
        lambda _root, agent, _policy_value, **_kwargs: _test_agent_manifest(agent),
    )
    monkeypatch.setattr(
        campaign,
        "_gpu_inventory",
        lambda _config, _tasks: {
            "ordered_host_gpu_ids": ["0"],
            "task_mapping": [{"task_name": "example", "assigned_host_gpu_id": "0"}],
        },
    )
    monkeypatch.setattr(
        campaign,
        "_evaluator_manifest",
        lambda state: {
            "schema": "aka.evaluator-source-binding/v2",
            "execution_manifest_sha256": state["execution_manifest_sha256"],
        },
    )
    monkeypatch.setattr(
        campaign, "runtime_isolation_receipt", _runtime_isolation_receipt
    )

    manifest = campaign.build_campaign_manifest(
        eval_config={"campaign": _policy()},
        run_config_path=run_config,
        task_config_paths={"example": str(task_config)},
        agent_name="codex",
    )

    assert manifest is not None
    assert manifest["measurement"]["configured_repetitions_per_test_case"] == 100
    assert manifest["measurement"]["is_apex_canonical_300_sample_grade"] is False
    assert manifest["configuration"]["tasks"][0]["config_sha256"]
    assert len(manifest["comparison_contract_sha256"]) == 64
    assert manifest["comparison_contract"]["objective_policy_id"] == (
        "aka.task-package-objective-and-protected-harness/v1"
    )
    assert manifest["comparison_contract"]["prompt_policy_id"] == (
        "aka.shared-objective-backend-native-context-receipted/v1"
    )


def test_outer_timeout_must_cover_attempts_and_evaluator() -> None:
    policy = _policy()
    policy["task_timeout_seconds"] = 10800
    try:
        campaign.parse_campaign_policy({"campaign": policy})
    except campaign.CampaignError as error:
        assert "evaluator allowance" in str(error)
    else:
        raise AssertionError("undersized task timeout must be rejected")


def test_formal_manifest_rejects_missing_sealed_aka_execution_snapshot(
    tmp_path, monkeypatch
) -> None:
    run_config = tmp_path / "run.yaml"
    run_config.write_text("campaign: {}\n", encoding="utf-8")
    monkeypatch.setattr(
        campaign,
        "_aka_state_from_environment",
        lambda _root: (_ for _ in ()).throw(
            campaign.CampaignError("sealed AKA execution snapshot unavailable")
        ),
    )

    with pytest.raises(campaign.CampaignError, match="sealed AKA execution snapshot"):
        campaign.build_campaign_manifest(
            eval_config={"campaign": _policy()},
            run_config_path=run_config,
            task_config_paths={},
            agent_name="codex",
        )


def test_campaign_manifest_is_published_read_only(tmp_path, monkeypatch) -> None:
    run = tmp_path / "run"
    run.mkdir()
    comparison_contract = {
        "runtime": {"isolation": _runtime_isolation_receipt()}
    }

    def current_manifest(**_kwargs):
        comparison = json.loads(json.dumps(comparison_contract))
        digest = hashlib.sha256(
            json.dumps(comparison, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return {
            "comparison_contract_sha256": digest,
            "comparison_contract": comparison,
        }

    monkeypatch.setattr(
        campaign,
        "build_campaign_manifest",
        current_manifest,
    )
    config = tmp_path / "config.yaml"
    config.write_text("campaign: {}\n", encoding="utf-8")

    path = campaign.ensure_campaign_manifest(
        run_directory=run,
        eval_config={"campaign": _policy()},
        run_config_path=config,
        task_config_paths={},
        agent_name="codex",
    )

    assert path is not None
    assert path.stat().st_mode & 0o222 == 0

    comparison_contract["runtime"]["isolation"] = _runtime_isolation_receipt(
        yama_ptrace_scope=2
    )
    with pytest.raises(campaign.CampaignError, match="provenance changed"):
        campaign.ensure_campaign_manifest(
            run_directory=run,
            eval_config={"campaign": _policy()},
            run_config_path=config,
            task_config_paths={},
            agent_name="codex",
        )


def test_formal_image_manifest_requires_nonempty_daemon_repo_digest(
    monkeypatch,
) -> None:
    monkeypatch.setenv("AGENT_KERNEL_ARENA_DOCKER_IMAGE", "image:tag")
    monkeypatch.setenv("AGENT_KERNEL_ARENA_DOCKER_IMAGE_ID", "sha256:" + "1" * 64)
    monkeypatch.setenv("AGENT_KERNEL_ARENA_DOCKER_REPO_DIGESTS", "[]")

    with pytest.raises(campaign.CampaignError, match="repo digest"):
        campaign._image_manifest()


def test_codex_campaign_inner_iteration_override_does_not_change_default() -> None:
    agent_config = {"max_iterations": 3, "campaign_max_iterations": 1}

    normal = codex_launcher._prompt_agent_config(agent_config, {})
    matched = codex_launcher._prompt_agent_config(
        agent_config, {"campaign": {"comparison": "apex_vs_codex"}}
    )

    assert normal["max_iterations"] == 3
    assert matched["max_iterations"] == 1
    assert agent_config["max_iterations"] == 3


def test_deterministic_gpu_mapping_and_worker_affinity(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_KERNEL_ARENA_GPU_POOL", "0,1,2,3,4,5,6,7")
    run = tmp_path / "run"
    run.mkdir()
    task_names = tuple(f"task/{index}" for index in range(10))
    contract = _write_campaign_codex_contract(run, task_names)
    tasks = contract["_task_config_paths"]
    context = {
        "run_directory": run,
        "task_config_dict": tasks,
        "timestamp": "20260807_000000",
        "agent": SimpleNamespace(value="codex"),
        "logger": logging.getLogger(__name__),
        "config": {"campaign": _policy()},
    }

    initialize_parallel_queue(context)

    pending = sorted((run / ".parallel/pending").glob("*.yaml"))
    assignments = [yaml.safe_load(path.read_text())["assigned_host_gpu_id"] for path in pending]
    assert assignments == ["0", "1", "2", "3", "4", "5", "6", "7", "0", "1"]
    before = {path.name: path.read_bytes() for path in pending}
    with pytest.raises(campaign.CampaignError, match="queue already exists"):
        initialize_parallel_queue(context)
    assert {
        path.name: path.read_bytes()
        for path in sorted((run / ".parallel/pending").glob("*.yaml"))
    } == before
    claimed = claim_next_descriptor(run, "gpu7", logging.getLogger(__name__), "7")
    assert claimed is not None
    assert yaml.safe_load(claimed.read_text())["assigned_host_gpu_id"] == "7"


def test_gpu_inventory_binds_every_pool_device_and_task_mapping(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_KERNEL_ARENA_GPU_POOL", "0,1,2,3,4,5,6,7")
    monkeypatch.setenv("AGENT_KERNEL_ARENA_GPU_ARCH", "gfx950")
    plan_digest = "a" * 64
    plan_path = "/run/agentkernelarena/formal-gpu-boundary-plan.json"
    monkeypatch.setenv("AGENT_KERNEL_ARENA_GPU_BOUNDARY_PLAN", plan_path)
    monkeypatch.setenv("AGENT_KERNEL_ARENA_GPU_BOUNDARY_PLAN_SHA256", plan_digest)
    exclusivity_digest = "b" * 64
    monkeypatch.setenv(
        "AGENT_KERNEL_ARENA_GPU_EXCLUSIVITY_RECEIPT", "/run/aka-gpu-lease.json"
    )
    monkeypatch.setenv(
        "AGENT_KERNEL_ARENA_GPU_EXCLUSIVITY_RECEIPT_SHA256", exclusivity_digest
    )
    plan = {
        "sha256": plan_digest,
        "kfd_device": {"path": "/dev/kfd", "major": 235, "minor": 0},
        "devices": [
            {
                "host_gpu_id": str(index),
                "unique_id": f"0x{index + 1:016x}",
                "serial_number": f"SERIAL-{index}",
                "card_series": "AMD Instinct MI355X",
                "gfx_version": "gfx950",
                "render_nodes": [{"path": f"/dev/dri/renderD{128 + index}"}],
            }
            for index in range(8)
        ],
    }
    monkeypatch.setattr(campaign, "load_plan", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(
        campaign,
        "load_gpu_lease_receipt",
        lambda *_args, **_kwargs: {
            "sha256": exclusivity_digest,
            "exclusivity_verified": True,
        },
    )
    tasks = [f"task/{index}" for index in range(10)]

    inventory = campaign._gpu_inventory({"target_gpu_model": "MI355X"}, tasks)

    assert [device["host_device_id"] for device in inventory["devices"]] == [
        str(index) for index in range(8)
    ]
    assert [item["assigned_host_gpu_id"] for item in inventory["task_mapping"]] == [
        "0", "1", "2", "3", "4", "5", "6", "7", "0", "1"
    ]
    assert inventory["gpu_boundary_plan_sha256"] == plan_digest
    assert inventory["exclusivity"]["exclusivity_verified"] is True


def test_comparison_contract_hash_ignores_treatment_template(tmp_path, monkeypatch) -> None:
    task_root = tmp_path / "task"
    task_root.mkdir()
    task_config = task_root / "config.yaml"
    task_config.write_text("task_type: triton2triton\n", encoding="utf-8")
    apex_run = tmp_path / "apex-run.yaml"
    codex_run = tmp_path / "codex-run.yaml"
    common_run_config = {
        "campaign": {"comparison": "apex_vs_codex", "attempts": 3},
        "tasks": ["example"],
        "target_gpu_model": "MI355X",
        "log_directory": "/test/logs",
        "workspace_directory_prefix": "/test/workspace",
    }
    apex_run.write_text(
        yaml.safe_dump(
            {"agent": {"template": "apex"}, **common_run_config},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    codex_run.write_text(
        yaml.safe_dump(
            {"agent": {"template": "codex"}, **common_run_config},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENT_KERNEL_ARENA_DOCKER_IMAGE", "image:tag")
    monkeypatch.setenv("AGENT_KERNEL_ARENA_DOCKER_IMAGE_ID", "sha256:" + "1" * 64)
    monkeypatch.setenv(
        "AGENT_KERNEL_ARENA_DOCKER_REPO_DIGESTS",
        '["image@sha256:' + "2" * 64 + '"]',
    )
    monkeypatch.setenv("AGENT_KERNEL_ARENA_GPU_POOL", "0,1")
    monkeypatch.setattr(
        campaign, "_aka_state_from_environment", lambda _root: _test_aka_binding()
    )
    monkeypatch.setattr(
        campaign,
        "_apex_state_from_environment",
        lambda: {
            "commit": "c" * 40,
            "dirty": False,
            "status_sha256": "d" * 64,
            "runtime_manifest_sha256": "9" * 64,
        },
    )
    monkeypatch.setattr(
        campaign,
        "_agent_manifest",
        lambda _root, name, _policy_value, **_kwargs: _test_agent_manifest(name),
    )
    monkeypatch.setattr(
        campaign,
        "_evaluator_manifest",
        lambda state: {
            "schema": "aka.evaluator-source-binding/v2",
            "execution_manifest_sha256": state["execution_manifest_sha256"],
        },
    )
    monkeypatch.setattr(
        campaign, "runtime_isolation_receipt", _runtime_isolation_receipt
    )
    monkeypatch.setattr(
        campaign,
        "_gpu_inventory",
        lambda _config, names: {
            "ordered_host_gpu_ids": ["0", "1"],
            "task_mapping": campaign.deterministic_task_gpu_mapping(names),
        },
    )
    kwargs = {
        "eval_config": {"campaign": _policy()},
        "task_config_paths": {"example": str(task_config)},
    }
    apex = campaign.build_campaign_manifest(
        **kwargs, run_config_path=apex_run, agent_name="apex"
    )
    codex = campaign.build_campaign_manifest(
        **kwargs, run_config_path=codex_run, agent_name="codex"
    )
    assert apex is not None and codex is not None
    assert apex["comparison_contract_sha256"] == codex["comparison_contract_sha256"]
    assert apex["agent"]["template"] != codex["agent"]["template"]
    assert apex["comparison_contract"]["agent_transport_treatments"] == (
        campaign.FORMAL_AGENT_TRANSPORT_TREATMENTS
    )
    assert apex["comparison_contract"]["run_config"] == (
        codex["comparison_contract"]["run_config"]
    )

    non_treatment_drifts = {
        "tasks": {"tasks": ["different/task"]},
        "attempts": {
            "campaign": {"comparison": "apex_vs_codex", "attempts": 4}
        },
        "gpu": {"target_gpu_model": "MI300X"},
        "output": {"workspace_directory_prefix": "/different/workspace"},
        "log": {"log_directory": "/different/logs"},
        "future_field": {"new_formal_option": {"enabled": True}},
    }
    for field, replacement in non_treatment_drifts.items():
        drifted = copy.deepcopy(common_run_config)
        drifted.update(replacement)
        codex_run.write_text(
            yaml.safe_dump(
                {"agent": {"template": "codex"}, **drifted},
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        drifted_manifest = campaign.build_campaign_manifest(
            **kwargs, run_config_path=codex_run, agent_name="codex"
        )
        assert drifted_manifest is not None
        assert drifted_manifest["comparison_contract_sha256"] != (
            apex["comparison_contract_sha256"]
        ), field

    for manifest in (apex, codex):
        transport = campaign.FORMAL_AGENT_TRANSPORT_TREATMENTS[
            manifest["agent"]["template"]
        ]
        assert manifest["agent"]["max_process_output_bytes"] == (
            transport["max_process_output_bytes"]
        )
        assert manifest["agent"]["structured_stream_overflow_policy"] == (
            transport["overflow_policy"]
        )


def test_comparison_contract_projects_run_specific_gpu_lease_receipts() -> None:
    policy = campaign.CampaignPolicy(**_policy())
    closure = _test_backend_closure()
    agent = {
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
        "codex_version": "codex test",
        "codex_binary_sha256": "1" * 64,
        "backend_runtime_closure_schema": campaign.BACKEND_CLOSURE_SCHEMA,
            "backend_runtime_closure_sha256": closure["closure_sha256"],
            "backend_runtime_closure": closure,
            "cloud_config_bootstrap_schema": (
                campaign.CODEX_CLOUD_CONFIG_BOOTSTRAP_SCHEMA
            ),
            "cloud_config_bootstrap_policy": (
                campaign.CODEX_CLOUD_CONFIG_BOOTSTRAP_POLICY
            ),
            "cloud_config_bundle_sha256": "c" * 64,
            "cloud_config_host_runtime_closure_sha256": "d" * 64,
            "cloud_config_initial_refresh_receipt_sha256": "e" * 64,
            "isolation": {},
    }

    def runtime(run_name: str, runner_pid: int, receipt_digest: str) -> dict:
        return {
            "docker": {"image_id": "sha256:" + "2" * 64},
            "isolation": _runtime_isolation_receipt(),
            "gpu": {
                "gpu_boundary_plan_sha256": "3" * 64,
                "devices": [{"host_device_id": "0", "unique_id": "0x01"}],
                "exclusivity": {
                    "policy": "physical_unique_id_flock_plus_kfd_preflight_v1",
                    "gpu_boundary_plan_sha256": "3" * 64,
                    "run_name": run_name,
                    "runner_pid": runner_pid,
                    "observed_at_ns": runner_pid * 10,
                    "sha256": receipt_digest,
                    "protected_device_paths": ["/dev/kfd", "/dev/dri/renderD128"],
                    "leases": [
                        {"unique_id": "0x01", "lock_path": f"/tmp/{run_name}.lock"}
                    ],
                    "exclusivity_verified": True,
                },
            },
        }

    kwargs = {
        "policy": policy,
        "measurement": {},
        "repositories": {
            "apex": {
                "commit": "4" * 40,
                "dirty": False,
                "status_sha256": "5" * 64,
                "runtime_manifest_sha256": "6" * 64,
            }
        },
        "agent": agent,
        "evaluator": {},
        "tasks": [],
        "run_config": {
            "schema": "aka.formal-run-config/v1",
            "effective_config": {"campaign": _policy()},
            "effective_config_sha256": campaign._canonical_json_digest(
                {"campaign": _policy()}
            ),
        },
    }
    first = campaign._comparison_contract(
        runtime=runtime("apex-run", 101, "a" * 64), **kwargs
    )
    second = campaign._comparison_contract(
        runtime=runtime("codex-run", 202, "b" * 64), **kwargs
    )

    assert first == second
    changed = runtime("codex-run", 202, "b" * 64)
    changed["gpu"]["gpu_boundary_plan_sha256"] = "4" * 64
    assert campaign._comparison_contract(runtime=changed, **kwargs) != first
    changed_isolation = runtime("codex-run", 202, "b" * 64)
    changed_isolation["isolation"] = _runtime_isolation_receipt(
        yama_ptrace_scope=2
    )
    assert campaign._comparison_contract(runtime=changed_isolation, **kwargs) != first


def test_fake_clock_enforces_remaining_session_reservation(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_KERNEL_ARENA_HOST_GPU_ID", "0")
    now = [0.0]
    run = tmp_path / "run"
    run.mkdir()
    codex_contract = _write_campaign_codex_contract(run)

    def single_attempt(**kwargs):
        attempt_run = Path(kwargs["run_directory"])
        workspace = attempt_run / "triton2triton_vllm_example_20260807_000000"
        workspace.mkdir()
        _write_result(workspace, 1.0)
        now[0] += 12000.0
        return True, workspace

    completed, canonical = campaign.run_matched_task_campaign(
        eval_config={"campaign": _policy(), "assigned_host_gpu_id": "0"},
        agent=object(),
        agent_launcher=object(),
        task_name="triton2triton/vllm/example",
        task_config_dir=codex_contract["_task_config_paths"][
            "triton2triton/vllm/example"
        ],
        run_directory=run,
        timestamp="20260807_000000",
        logger=logging.getLogger(__name__),
        task_index=1,
        total_tasks=1,
        single_attempt=single_attempt,
        clock=lambda: now[0],
    )
    assert completed is False
    assert canonical is None
    evidence = yaml.safe_load(
        (run / ".campaign_attempts" / _DEFAULT_TASK_COMPONENT / "task_campaign.yaml").read_text()
    )
    assert evidence["attempts"][1]["eligibility_errors"] == [
        "outer_task_deadline_cannot_cover_remaining_sessions"
    ]


def test_cumulative_central_evaluator_allowance_is_enforced(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("AGENT_KERNEL_ARENA_HOST_GPU_ID", "0")
    run = tmp_path / "run"
    run.mkdir()
    codex_contract = _write_campaign_codex_contract(run)

    def single_attempt(**kwargs):
        attempt_run = Path(kwargs["run_directory"])
        workspace = attempt_run / "triton2triton_vllm_example_20260807_000000"
        workspace.mkdir()
        _write_result(workspace, 1.0)
        kwargs["eval_config"]["campaign_attempt"]["evaluation_elapsed_seconds"] = 1201.0
        return True, workspace

    completed, canonical = campaign.run_matched_task_campaign(
        eval_config={"campaign": _policy(), "assigned_host_gpu_id": "0"},
        agent=object(),
        agent_launcher=object(),
        task_name="triton2triton/vllm/example",
        task_config_dir=codex_contract["_task_config_paths"][
            "triton2triton/vllm/example"
        ],
        run_directory=run,
        timestamp="20260807_000000",
        logger=logging.getLogger(__name__),
        task_index=1,
        total_tasks=1,
        single_attempt=single_attempt,
        clock=lambda: 0.0,
    )

    assert completed is False
    assert canonical is None
    evidence = yaml.safe_load(
        (run / ".campaign_attempts" / _DEFAULT_TASK_COMPONENT / "task_campaign.yaml").read_text()
    )
    assert evidence["evaluator_elapsed_seconds"] == 3603.0
    assert evidence["within_evaluator_allowance"] is False
    assert "evaluator_allowance_exceeded" in evidence["attempts"][2][
        "eligibility_errors"
    ]


def test_evaluator_command_timeout_is_bounded_by_hard_deadline(tmp_path, monkeypatch) -> None:
    observed: list[float] = []

    def fake_run(_command, _workspace, *, timeout, **_kwargs):
        observed.append(timeout)
        return True, "", ""

    monkeypatch.setattr(evaluator, "run_command", fake_run)
    passed, error = evaluator.evaluate_compilation(
        tmp_path,
        {"compile_command": ["python compile.py"], "compile_timeout": 100},
        deadline_monotonic=15.0,
        clock=lambda: 10.0,
    )

    assert passed is True and error is None
    assert observed == [5.0]


def test_eligibility_rejects_case_mismatch_and_speedup_error(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_result(workspace, 1.0)
    optimized = yaml.safe_load((workspace / "optimized_perf.yaml").read_text())
    optimized["test_cases"][0]["test_case_id"] = "different"
    (workspace / "optimized_perf.yaml").write_text(yaml.safe_dump(optimized))
    report = yaml.safe_load((workspace / "task_result.yaml").read_text())
    report["speedup_calculation_error_message"] = "mismatch"
    report["pass_correctness"] = False

    errors = campaign._evaluation_eligibility_errors(workspace, report)

    assert "baseline_optimized_testcase_set_or_order_mismatch" in errors
    assert "speedup_calculation_error" in errors
    assert "central_correctness_failed" in errors


def test_safe_projection_rejects_symlinks(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "regular.py").write_text("value = 1\n")
    (source / "escape").symlink_to("/tmp")

    try:
        campaign._safe_copy_workspace(source, tmp_path / "destination")
    except campaign.CampaignError as error:
        assert "symlink" in str(error)
    else:
        raise AssertionError("unsafe workspace symlink must be rejected")


def test_direct_codex_receipt_artifacts_and_pinned_identity_are_recomputed(
    tmp_path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    workspace = run / ".campaign_attempts/task/attempt_01/workspace"
    workspace.mkdir(parents=True)
    codex_contract = _write_campaign_codex_contract(run, agent_template="codex")
    receipt_path = workspace.parent / "session_receipt.json"
    raw_stdout = _write_valid_codex_receipt(receipt_path, codex_contract)

    receipt, errors = campaign._validate_session_receipt(
        receipt_path=receipt_path,
        workspace=workspace,
        run_directory=run,
    )
    assert receipt is not None
    assert errors == []
    _unlock_apex_receipt_directories(run)

    raw_stdout.chmod(0o644)
    raw_stdout.write_text("tampered\n", encoding="utf-8")
    raw_stdout.chmod(0o444)
    _, errors = campaign._validate_session_receipt(
        receipt_path=receipt_path,
        workspace=workspace,
        run_directory=run,
    )
    assert "direct_codex_raw_stdout_hash_mismatch" in errors
    assert "direct_codex_raw_stdout_size_mismatch" in errors
    raw_stdout.parent.chmod(0o700)


def test_direct_codex_receipt_rejects_cloud_config_bundle_drift(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    workspace = run / ".campaign_attempts/task/attempt_01/workspace"
    workspace.mkdir(parents=True)
    contract = _write_campaign_codex_contract(run, agent_template="codex")
    receipt_path = workspace.parent / "session_receipt.json"
    _write_valid_codex_receipt(receipt_path, contract)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["codex"]["cloud_config_bootstrap"]["bundle_sha256"] = "d" * 64
    receipt_path.chmod(0o644)
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    receipt_path.chmod(0o444)

    _, errors = campaign._validate_session_receipt(
        receipt_path=receipt_path,
        workspace=workspace,
        run_directory=run,
    )

    assert "direct_codex_cloud_config_bootstrap_invalid" in errors
    _unlock_apex_receipt_directories(run)


def test_direct_codex_prompt_and_comparison_contract_are_independently_recomputed(
    tmp_path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    workspace = run / ".campaign_attempts/task/attempt_01/workspace"
    workspace.mkdir(parents=True)
    codex_contract = _write_campaign_codex_contract(run, agent_template="codex")
    receipt_path = workspace.parent / "session_receipt.json"
    _write_valid_codex_receipt(receipt_path, codex_contract)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    prompt_path = Path(receipt["artifacts"]["rendered_prompt"]["path"])

    prompt_path.parent.chmod(0o700)
    original = prompt_path.read_bytes()
    prompt_path.chmod(0o644)
    prompt_path.write_bytes(b"X" + original[1:])
    prompt_path.chmod(0o444)
    prompt_path.parent.chmod(0o555)
    _, errors = campaign._validate_session_receipt(
        receipt_path=receipt_path,
        workspace=workspace,
        run_directory=run,
    )
    assert "direct_codex_rendered_prompt_hash_mismatch" in errors
    assert "direct_codex_prompt_digest_mismatch" in errors

    receipt_path.chmod(0o644)
    receipt["comparison_contract_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    receipt_path.chmod(0o444)
    _, errors = campaign._validate_session_receipt(
        receipt_path=receipt_path,
        workspace=workspace,
        run_directory=run,
    )
    assert "direct_codex_comparison_contract_digest_mismatch" in errors
    prompt_path.parent.chmod(0o700)


def test_direct_codex_receipt_rejects_cross_task_campaign_binding_transplant(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    workspace = run / ".campaign_attempts/task/attempt_01/workspace"
    workspace.mkdir(parents=True)
    contract = _write_campaign_codex_contract(run, agent_template="codex")
    receipt_path = workspace.parent / "session_receipt.json"
    _write_valid_codex_receipt(receipt_path, contract)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["campaign_binding"]["task_name"] = "other-task"
    receipt_path.chmod(0o644)
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    receipt_path.chmod(0o444)

    _, errors = campaign._validate_session_receipt(
        receipt_path=receipt_path,
        workspace=workspace,
        run_directory=run,
    )

    assert "attempt_campaign_binding_mismatch" in errors
    _unlock_apex_receipt_directories(run)


def test_direct_codex_exact_boundary_checkpoint_is_formally_eligible(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    workspace = run / ".campaign_attempts/task/attempt_01/workspace"
    workspace.mkdir(parents=True)
    codex_contract = _write_campaign_codex_contract(
        run,
        agent_template="codex",
    )
    receipt_path = workspace.parent / "session_receipt.json"
    _write_valid_codex_receipt(
        receipt_path, codex_contract, exact_boundary=True
    )

    receipt, errors = campaign._validate_session_receipt(
        receipt_path=receipt_path,
        workspace=workspace,
        run_directory=run,
    )

    assert receipt is not None
    assert errors == []
    assert receipt["turn_budget"]["observed_turns"] == 50
    assert receipt["candidate_persistence"]["termination"] == (
        "exact_turn_boundary"
    )


def test_direct_codex_malformed_candidate_persistence_fails_closed(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    workspace = run / ".campaign_attempts/task/attempt_01/workspace"
    workspace.mkdir(parents=True)
    contract = _write_campaign_codex_contract(run, agent_template="codex")
    receipt_path = workspace.parent / "session_receipt.json"
    _write_valid_codex_receipt(receipt_path, contract, exact_boundary=True)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["candidate_persistence"] = None
    receipt_path.chmod(0o644)
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    receipt_path.chmod(0o444)

    _receipt, errors = campaign._validate_session_receipt(
        receipt_path=receipt_path,
        workspace=workspace,
        run_directory=run,
    )

    assert "direct_codex_candidate_persistence_invalid" in errors
    assert "direct_codex_turn_budget_invalid" in errors


def test_direct_codex_recomputes_delta_and_rejects_contradictory_receipt(
    tmp_path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    workspace = run / ".campaign_attempts/task/attempt_01/workspace"
    workspace.mkdir(parents=True)
    contract = _write_campaign_codex_contract(run, agent_template="codex")
    receipt_path = workspace.parent / "session_receipt.json"
    _write_valid_codex_receipt(receipt_path, contract, exact_boundary=True)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    before_path = Path(payload["artifacts"]["workspace_before_manifest"]["path"])
    after_path = Path(payload["artifacts"]["workspace_after_manifest"]["path"])
    after_path.parent.chmod(0o755)
    after_path.chmod(0o644)
    after_bytes = before_path.read_bytes()
    after_path.write_bytes(after_bytes)
    after_path.chmod(0o444)
    after_path.parent.chmod(0o555)
    payload["artifacts"]["workspace_after_manifest"] |= {
        "sha256": hashlib.sha256(after_bytes).hexdigest(),
        "size_bytes": len(after_bytes),
    }
    same_digest = campaign._canonical_json_digest(
        json.loads(after_bytes.decode("utf-8"))
    )
    final_changes = payload["workspace_integrity"]["final_changes"]
    final_changes["after_manifest_sha256"] = same_digest
    checkpoint = payload["candidate_persistence"]["checkpoint"]
    checkpoint["after_manifest_sha256"] = same_digest
    receipt_path.chmod(0o644)
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    receipt_path.chmod(0o444)

    _receipt, errors = campaign._validate_session_receipt(
        receipt_path=receipt_path,
        workspace=workspace,
        run_directory=run,
    )

    assert "direct_codex_sanitized_manifest_contract_mismatch" in errors
    assert "direct_codex_zero_source_delta" not in errors


def test_direct_codex_exact_boundary_accepts_proven_natural_exit_with_hidden_siblings(
    tmp_path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    workspace = run / ".campaign_attempts/task/attempt_01/workspace"
    workspace.mkdir(parents=True)
    codex_contract = _write_campaign_codex_contract(
        run,
        agent_template="codex",
    )
    receipt_path = workspace.parent / "session_receipt.json"
    _write_valid_codex_receipt(
        receipt_path, codex_contract, exact_boundary=True
    )
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    cleanup = payload["attempt_process_cleanup"]
    cleanup |= {
        "reason": "exact_turn_boundary",
        "required": True,
        "sigkill_sent": False,
        "outer_supervisor_exit_code": 0,
        "bubblewrap_terminal_status_verified": True,
        "bubblewrap_terminal_status": {"exit-code": 0},
        "bubblewrap_terminal_status_absent_after_sigkill": False,
        "teardown_mode": "natural_exit",
        "namespace_membership_scan_complete": False,
        "namespace_membership_inaccessible_entries_count": 2,
    }
    payload["exit_code"] = 0
    persistence = payload["candidate_persistence"]
    persistence["attempt_cleanup"] = cleanup
    checkpoint = persistence["checkpoint"]
    checkpoint["attempt_cleanup_sha256"] = campaign._canonical_json_digest(
        cleanup
    )
    receipt_path.chmod(0o644)
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    receipt_path.chmod(0o444)

    receipt, errors = campaign._validate_session_receipt(
        receipt_path=receipt_path,
        workspace=workspace,
        run_directory=run,
    )

    assert receipt is not None
    assert errors == []


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("turn_49", "direct_codex_turn_budget_invalid"),
        ("turn_51", "direct_codex_turn_budget_invalid"),
        ("timeout", "direct_codex_receipt_success_status_inconsistent"),
        ("truncation", "direct_codex_stdout_capture_bound_invalid"),
        ("cleanup", "direct_codex_attempt_namespace_not_verified_absent"),
        ("escaped_descendant", "direct_codex_attempt_namespace_not_verified_absent"),
        ("false_complete_scan", "direct_codex_attempt_namespace_not_verified_absent"),
        ("boundary_identity", "direct_codex_checkpoint_suspension_invalid"),
        ("terminal_evidence", "direct_codex_attempt_namespace_not_verified_absent"),
        ("forged_terminal_status", "direct_codex_attempt_namespace_not_verified_absent"),
        ("tail_digest", "direct_codex_boundary_output_tail_digest_mismatch"),
    ],
)
def test_direct_codex_checkpoint_rejects_incomplete_boundary_evidence(
    tmp_path, mutation, expected_error
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    workspace = run / ".campaign_attempts/task/attempt_01/workspace"
    workspace.mkdir(parents=True)
    codex_contract = _write_campaign_codex_contract(
        run,
        agent_template="codex",
    )
    receipt_path = workspace.parent / "session_receipt.json"
    _write_valid_codex_receipt(
        receipt_path, codex_contract, exact_boundary=True
    )
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if mutation == "turn_49":
        payload["turn_budget"]["observed_turns"] = 49
    elif mutation == "turn_51":
        payload["turn_budget"]["observed_turns"] = 51
    elif mutation == "timeout":
        payload["timed_out"] = True
    elif mutation == "truncation":
        payload["capture"]["stdout"]["truncated"] = True
        payload["capture"]["stdout"]["discarded_bytes"] = 1
    elif mutation == "cleanup":
        payload["attempt_process_cleanup"]["verified_absent"] = False
    elif mutation == "escaped_descendant":
        payload["attempt_process_cleanup"][
            "live_visible_namespace_members_after"
        ] = [
            {
                "pid": 4321,
                "state": "S",
                "ppid": 1,
                "pgrp": 4321,
                "session": 4321,
                "starttime": 99,
            }
        ]
    elif mutation == "false_complete_scan":
        payload["attempt_process_cleanup"][
            "namespace_membership_inaccessible_entries_count"
        ] = 1
    elif mutation == "boundary_identity":
        payload["attempt_process_cleanup"]["boundary"]["namespace_init_inner_pid"] = 2
    elif mutation == "terminal_evidence":
        payload["attempt_process_cleanup"][
            "bubblewrap_terminal_status_absent_after_sigkill"
        ] = False
    elif mutation == "forged_terminal_status":
        cleanup = payload["attempt_process_cleanup"]
        cleanup["bubblewrap_terminal_status_verified"] = True
        cleanup["bubblewrap_terminal_status_absent_after_sigkill"] = False
        cleanup["bubblewrap_terminal_status"] = {"exit-code": 0}
    else:
        payload["candidate_persistence"]["output_tail"]["stdout_sha256"] = (
            "0" * 64
        )
    receipt_path.chmod(0o644)
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    receipt_path.chmod(0o444)

    _, errors = campaign._validate_session_receipt(
        receipt_path=receipt_path,
        workspace=workspace,
        run_directory=run,
    )

    assert expected_error in errors


def test_apex_manifest_rejects_substituted_direct_codex_receipt(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    workspace = run / ".campaign_attempts/task/attempt_01/workspace"
    workspace.mkdir(parents=True)
    _write_result(workspace, 1.0)
    codex_contract = _write_campaign_codex_contract(run)
    receipt_path = workspace.parent / "session_receipt.json"
    _write_valid_codex_receipt(receipt_path, codex_contract)

    try:
        receipt, errors = campaign._validate_session_receipt(
            receipt_path=receipt_path,
            workspace=workspace,
            run_directory=run,
        )
        assert receipt is not None
        assert errors == ["apex_receipt_schema_generation_mismatch"]
        record = campaign._attempt_record(
            attempt=1,
            workspace=workspace,
            run_directory=run,
            success=True,
            receipt_path=receipt_path,
            require_session_receipt=True,
        )
        assert record["selection_eligible"] is False
        assert "apex_receipt_schema_generation_mismatch" in record[
            "eligibility_errors"
        ]
    finally:
        _unlock_apex_receipt_directories(run)


def test_direct_codex_manifest_rejects_substituted_apex_receipt(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    workspace = run / ".campaign_attempts/task/attempt_01/workspace"
    workspace.mkdir(parents=True)
    codex_contract = _write_campaign_codex_contract(
        run, agent_template="codex"
    )
    receipt_path = workspace.parent / "session_receipt.json"
    _write_valid_apex_receipt(
        receipt_path,
        codex_contract,
    )

    try:
        receipt, errors = campaign._validate_session_receipt(
            receipt_path=receipt_path,
            workspace=workspace,
            run_directory=run,
        )
        assert receipt is not None
        assert errors == ["direct_codex_receipt_schema_generation_mismatch"]
    finally:
        _unlock_apex_receipt_directories(run)


def test_apex_v6_receipt_recomputes_result_event_invocation_and_transcript_lineage(
    tmp_path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    workspace = run / ".campaign_attempts/task/attempt_01/workspace"
    workspace.mkdir(parents=True)
    codex_contract = _write_campaign_codex_contract(run)
    receipt_path = workspace.parent / "session_receipt.json"
    _write_valid_apex_receipt(
        receipt_path,
        codex_contract,
    )

    receipt, errors = campaign._validate_session_receipt(
        receipt_path=receipt_path,
        workspace=workspace,
        run_directory=run,
    )

    assert receipt is not None
    assert errors == []
    receipt_path.chmod(0o644)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["workspace_integrity"]["pre_apply_unchanged"] = False
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    receipt_path.chmod(0o444)
    _, errors = campaign._validate_session_receipt(
        receipt_path=receipt_path,
        workspace=workspace,
        run_directory=run,
    )
    assert "apex_workspace_pre_apply_integrity_invalid" in errors
    _unlock_apex_receipt_directories(run)


def test_apex_v6_receipt_rejects_campaign_binding_tamper_across_all_lineage(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    workspace = run / ".campaign_attempts/task/attempt_01/workspace"
    workspace.mkdir(parents=True)
    contract = _write_campaign_codex_contract(run)
    receipt_path = workspace.parent / "session_receipt.json"
    _write_valid_apex_receipt(receipt_path, contract)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["campaign_binding"]["assigned_host_gpu_id"] = "1"
    receipt_path.chmod(0o644)
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    receipt_path.chmod(0o444)

    try:
        _, errors = campaign._validate_session_receipt(
            receipt_path=receipt_path,
            workspace=workspace,
            run_directory=run,
        )
        assert "attempt_campaign_binding_mismatch" in errors
        assert "apex_task_spec_campaign_binding_mismatch" in errors
        assert "apex_lineage_campaign_binding_mismatch" in errors
    finally:
        _unlock_apex_receipt_directories(run)


def test_apex_receipt_must_bind_immutable_comparison_contract(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    workspace = run / ".campaign_attempts/task/attempt_01/workspace"
    workspace.mkdir(parents=True)
    codex_contract = _write_campaign_codex_contract(run)
    receipt_path = workspace.parent / "session_receipt.json"
    _write_valid_apex_receipt(receipt_path, codex_contract)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["comparison_contract_sha256"] = "0" * 64
    receipt_path.chmod(0o644)
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    receipt_path.chmod(0o444)

    _, errors = campaign._validate_session_receipt(
        receipt_path=receipt_path,
        workspace=workspace,
        run_directory=run,
    )

    assert "apex_comparison_contract_digest_mismatch" in errors
    _unlock_apex_receipt_directories(run)


def test_apex_receipt_rejects_cloud_config_bundle_drift(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    workspace = run / ".campaign_attempts/task/attempt_01/workspace"
    workspace.mkdir(parents=True)
    contract = _write_campaign_codex_contract(run)
    receipt_path = workspace.parent / "session_receipt.json"
    _write_valid_apex_receipt(receipt_path, contract)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["codex"]["cloud_config_bootstrap"]["bundle_sha256"] = "d" * 64
    receipt_path.chmod(0o644)
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    receipt_path.chmod(0o444)

    _, errors = campaign._validate_session_receipt(
        receipt_path=receipt_path,
        workspace=workspace,
        run_directory=run,
    )

    assert "apex_codex_cloud_config_bootstrap_invalid" in errors
    _unlock_apex_receipt_directories(run)


def test_apex_receipt_rejects_turn_budget_drift_from_comparison_contract(
    tmp_path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    workspace = run / ".campaign_attempts/task/attempt_01/workspace"
    workspace.mkdir(parents=True)
    codex_contract = _write_campaign_codex_contract(run)
    receipt_path = workspace.parent / "session_receipt.json"
    _write_valid_apex_receipt(receipt_path, codex_contract, max_turns=49)

    _, errors = campaign._validate_session_receipt(
        receipt_path=receipt_path,
        workspace=workspace,
        run_directory=run,
    )

    assert "apex_task_spec_budget_contract_mismatch" in errors
    assert "apex_inner_codex_invocation_contract_mismatch" in errors
    _unlock_apex_receipt_directories(run)


def test_budget_exhausted_apex_receipt_has_verified_lineage_but_is_never_eligible(
    tmp_path,
) -> None:
    run = tmp_path / "run"
    workspace = run / ".campaign_attempts/task/attempt_01/workspace"
    workspace.mkdir(parents=True)
    _write_result(workspace, 1.0)
    report_path = workspace / "task_result.yaml"
    report = yaml.safe_load(report_path.read_text(encoding="utf-8"))
    report |= {
        "evaluation_mode": "diagnostic_baseline_replay_v1",
        "agent_session_score_eligible": False,
        "agent_session_succeeded": False,
    }
    report_path.write_text(yaml.safe_dump(report), encoding="utf-8")
    codex_contract = _write_campaign_codex_contract(run)
    receipt_path = workspace.parent / "session_receipt.json"
    _write_valid_apex_receipt(
        receipt_path,
        codex_contract,
        status="budget_exhausted",
    )

    try:
        receipt, errors = campaign._validate_session_receipt(
            receipt_path=receipt_path,
            workspace=workspace,
            run_directory=run,
        )
        assert receipt is not None
        assert errors == ["apex_session_not_successful"]

        record = campaign._attempt_record(
            attempt=1,
            workspace=workspace,
            run_directory=run,
            success=False,
            receipt_path=receipt_path,
            require_session_receipt=True,
        )
        assert record["selection_eligible"] is False
        assert record["session_receipt_binding"]["lineage_verified"] is True
        prompt = record["session_receipt_binding"]["event_bound_prompt"]
        assert prompt["binding"] == "apex.prompt_sent_event_cas/v1"
        assert re.fullmatch(r"[0-9a-f]{64}", prompt["sha256"])
        assert {
            "apex_artifact_set_mismatch",
            "apex_codex_identity_contract_mismatch",
            "apex_receipt_success_status_inconsistent",
            "apex_terminal_result_contract_mismatch",
            "apex_agent_completion_event_invalid",
        }.isdisjoint(record["eligibility_errors"])
        assert "apex_session_not_successful" in record["eligibility_errors"]
        assert "diagnostic_evaluation_not_scoreable" in record["eligibility_errors"]
        assert "agent_session_not_score_eligible" in record["eligibility_errors"]
    finally:
        _unlock_apex_receipt_directories(run)


def test_v6_budget_overrun_binds_inner_pidfd_containment(tmp_path) -> None:
    run = tmp_path / "run"
    workspace = run / ".campaign_attempts/task/attempt_01/workspace"
    workspace.mkdir(parents=True)
    contract = _write_campaign_codex_contract(run)
    receipt_path = workspace.parent / "session_receipt.json"
    _write_valid_apex_receipt(
        receipt_path,
        contract,
        status="budget_exhausted",
    )

    try:
        receipt, errors = campaign._validate_session_receipt(
            receipt_path=receipt_path,
            workspace=workspace,
            run_directory=run,
        )
        assert receipt is not None
        assert errors == ["apex_session_not_successful"]
        assert receipt["candidate_persistence"][
            "agent_process_containment_policy_id"
        ] == "private_pid_namespace_init_pidfd_v1"
    finally:
        _unlock_apex_receipt_directories(run)


def test_v6_budget_overrun_rejects_overlapping_inner_terminal_evidence(
    tmp_path,
) -> None:
    run = tmp_path / "run"
    workspace = run / ".campaign_attempts/task/attempt_01/workspace"
    workspace.mkdir(parents=True)
    contract = _write_campaign_codex_contract(run)
    receipt_path = workspace.parent / "session_receipt.json"
    _write_valid_apex_receipt(
        receipt_path,
        contract,
        status="budget_exhausted",
        containment_terminal_overlap=True,
    )

    try:
        receipt, errors = campaign._validate_session_receipt(
            receipt_path=receipt_path,
            workspace=workspace,
            run_directory=run,
        )
        assert receipt is not None
        assert "apex_checkpoint_termination_evidence_invalid" in errors
    finally:
        _unlock_apex_receipt_directories(run)


def test_v6_budget_overrun_rejects_sigterm_in_place_of_pidfd_exit(tmp_path) -> None:
    run = tmp_path / "run"
    workspace = run / ".campaign_attempts/task/attempt_01/workspace"
    workspace.mkdir(parents=True)
    contract = _write_campaign_codex_contract(run)
    receipt_path = workspace.parent / "session_receipt.json"
    _write_valid_apex_receipt(
        receipt_path,
        contract,
        status="budget_exhausted",
        inner_exit_code=-signal.SIGTERM,
    )

    try:
        _receipt, errors = campaign._validate_session_receipt(
            receipt_path=receipt_path,
            workspace=workspace,
            run_directory=run,
        )
        assert "apex_checkpoint_termination_evidence_invalid" in errors
    finally:
        _unlock_apex_receipt_directories(run)


def test_no_gain_apex_receipt_is_audited_but_never_selection_eligible(
    tmp_path,
) -> None:
    run = tmp_path / "run"
    workspace = run / ".campaign_attempts/task/attempt_01/workspace"
    workspace.mkdir(parents=True)
    _write_result(workspace, 1.0)
    report_path = workspace / "task_result.yaml"
    report = yaml.safe_load(report_path.read_text(encoding="utf-8"))
    report |= {
        "evaluation_mode": "no_candidate_baseline_replay_v1",
        "agent_session_score_eligible": False,
        "agent_session_succeeded": True,
        "agent_session_terminal_status": "no_gain",
    }
    report_path.write_text(yaml.safe_dump(report), encoding="utf-8")
    codex_contract = _write_campaign_codex_contract(run)
    receipt_path = workspace.parent / "session_receipt.json"
    _write_valid_apex_receipt(
        receipt_path,
        codex_contract,
        status="no_gain",
    )

    try:
        receipt, receipt_errors = campaign._validate_session_receipt(
            receipt_path=receipt_path,
            workspace=workspace,
            run_directory=run,
        )
        assert receipt is not None
        assert receipt_errors == []
        record = campaign._attempt_record(
            attempt=1,
            workspace=workspace,
            run_directory=run,
            success=True,
            receipt_path=receipt_path,
            require_session_receipt=True,
        )
        assert record["attempt_completed"] is True
        assert record["selection_eligible"] is False
        assert record["eligibility_errors"] == []
        assert record["agent_session_terminal_status"] == "no_gain"
    finally:
        _unlock_apex_receipt_directories(run)


def test_apex_v6_receipt_rejects_run_control_drift(tmp_path) -> None:
    run = tmp_path / "run"
    workspace = run / ".campaign_attempts/task/attempt_01/workspace"
    workspace.mkdir(parents=True)
    codex_contract = _write_campaign_codex_contract(run)
    receipt_path = workspace.parent / "session_receipt.json"
    _write_valid_apex_receipt(
        receipt_path,
        codex_contract,
        run_control_turns=49,
    )

    try:
        _, errors = campaign._validate_session_receipt(
            receipt_path=receipt_path,
            workspace=workspace,
            run_directory=run,
        )
        assert "apex_caller_run_control_invalid" in errors
    finally:
        _unlock_apex_receipt_directories(run)


def test_apex_v6_receipt_rejects_run_control_missing_from_instructions(
    tmp_path,
) -> None:
    run = tmp_path / "run"
    workspace = run / ".campaign_attempts/task/attempt_01/workspace"
    workspace.mkdir(parents=True)
    codex_contract = _write_campaign_codex_contract(run)
    receipt_path = workspace.parent / "session_receipt.json"
    _write_valid_apex_receipt(
        receipt_path,
        codex_contract,
        omit_run_control_suffix=True,
    )

    try:
        _, errors = campaign._validate_session_receipt(
            receipt_path=receipt_path,
            workspace=workspace,
            run_directory=run,
        )
        assert "apex_caller_run_control_invalid" in errors
        assert "apex_adapted_prompt_digest_mismatch" not in errors
    finally:
        _unlock_apex_receipt_directories(run)


def test_apex_v6_receipt_rejects_event_prompt_missing_run_control(tmp_path) -> None:
    run = tmp_path / "run"
    workspace = run / ".campaign_attempts/task/attempt_01/workspace"
    workspace.mkdir(parents=True)
    codex_contract = _write_campaign_codex_contract(run)
    receipt_path = workspace.parent / "session_receipt.json"
    _write_valid_apex_receipt(
        receipt_path,
        codex_contract,
        omit_run_control_from_agent_prompt=True,
    )

    try:
        _, errors = campaign._validate_session_receipt(
            receipt_path=receipt_path,
            workspace=workspace,
            run_directory=run,
        )
        assert "apex_prompt_objective_binding_mismatch" in errors
        assert "apex_prompt_event_binding_mismatch" not in errors
        assert "apex_caller_run_control_invalid" not in errors
    finally:
        _unlock_apex_receipt_directories(run)


@pytest.mark.parametrize("status", ["candidate_ready", "no_gain"])
@pytest.mark.parametrize("turn_count", [0, 51])
def test_successful_apex_receipt_rejects_turn_count_outside_budget(
    tmp_path, status, turn_count
) -> None:
    run = tmp_path / "run"
    workspace = run / ".campaign_attempts/task/attempt_01/workspace"
    workspace.mkdir(parents=True)
    codex_contract = _write_campaign_codex_contract(run)
    receipt_path = workspace.parent / "session_receipt.json"
    _write_valid_apex_receipt(
        receipt_path,
        codex_contract,
        status=status,
        successful_turn_count=turn_count,
    )

    try:
        _, errors = campaign._validate_session_receipt(
            receipt_path=receipt_path,
            workspace=workspace,
            run_directory=run,
        )
        assert "apex_checkpoint_termination_evidence_invalid" in errors
    finally:
        _unlock_apex_receipt_directories(run)


def test_live_apex_receipt_cannot_downgrade_to_historical_validation(tmp_path) -> None:
    run = tmp_path / "run"
    workspace = run / ".campaign_attempts/task/attempt_01/workspace"
    workspace.mkdir(parents=True)
    codex_contract = _write_campaign_codex_contract(run)
    receipt_path = workspace.parent / "session_receipt.json"
    _write_valid_apex_receipt(
        receipt_path,
        codex_contract,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["schema"] = "agentkernelarena.apex-attempt-receipt/v1"
    receipt.pop("instruction_adaptation")
    receipt["artifacts"].pop("original_arena_prompt")
    receipt["artifacts"].pop("agent_prompt")
    receipt_path.chmod(0o644)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    receipt_path.chmod(0o444)

    try:
        _, errors = campaign._validate_session_receipt(
            receipt_path=receipt_path,
            workspace=workspace,
            run_directory=run,
        )
        assert errors == ["apex_receipt_schema_generation_mismatch"]
    finally:
        _unlock_apex_receipt_directories(run)


@pytest.mark.parametrize(
    ("turn_count", "budget_reason"),
    [
        (50, "max_turns_exceeded"),
        (51, "max_turns_exhausted_before_follow_up"),
    ],
)
def test_budget_exhausted_receipt_rejects_reason_count_mismatch(
    tmp_path, turn_count, budget_reason
) -> None:
    run = tmp_path / "run"
    workspace = run / ".campaign_attempts/task/attempt_01/workspace"
    workspace.mkdir(parents=True)
    codex_contract = _write_campaign_codex_contract(run)
    receipt_path = workspace.parent / "session_receipt.json"
    _write_valid_apex_receipt(
        receipt_path,
        codex_contract,
        status="budget_exhausted",
        budget_turn_count=turn_count,
        budget_reason_override=budget_reason,
    )

    try:
        _, errors = campaign._validate_session_receipt(
            receipt_path=receipt_path,
            workspace=workspace,
            run_directory=run,
        )
        assert "apex_checkpoint_termination_evidence_invalid" in errors
    finally:
        _unlock_apex_receipt_directories(run)


@pytest.mark.parametrize(
    ("tamper", "expected_error"),
    [
        ("journal", "apex_event_journal_hash_mismatch"),
        ("transcript", "apex_agent_transcript_hash_mismatch"),
        ("prompt_binding", "apex_prompt_event_binding_mismatch"),
    ],
)
def test_budget_exhausted_apex_receipt_detects_lineage_corruption(
    tmp_path, tamper, expected_error
) -> None:
    run = tmp_path / "run"
    workspace = run / ".campaign_attempts/task/attempt_01/workspace"
    workspace.mkdir(parents=True)
    codex_contract = _write_campaign_codex_contract(run)
    receipt_path = workspace.parent / "session_receipt.json"
    _write_valid_apex_receipt(
        receipt_path,
        codex_contract,
        status="budget_exhausted",
    )
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if tamper in {"journal", "transcript"}:
        artifact_name = (
            "event_journal" if tamper == "journal" else "agent_transcript"
        )
        artifact_path = Path(payload["artifacts"][artifact_name]["path"])
        artifact_path.parent.chmod(0o700)
        artifact_path.chmod(0o644)
        with artifact_path.open("ab") as stream:
            stream.write(b" ")
        artifact_path.chmod(0o444)
        artifact_path.parent.chmod(0o555)
    else:
        receipt_path.chmod(0o644)
        payload["lineage"]["prompt_event"]["sha256"] = "0" * 64
        receipt_path.write_text(json.dumps(payload), encoding="utf-8")
        receipt_path.chmod(0o444)

    try:
        _, errors = campaign._validate_session_receipt(
            receipt_path=receipt_path,
            workspace=workspace,
            run_directory=run,
        )
        assert expected_error in errors
    finally:
        _unlock_apex_receipt_directories(run)


@pytest.mark.parametrize("agent_name", ["apex", "codex"])
def test_formal_agents_missing_receipts_are_diagnostic_only_and_never_complete(
    tmp_path, monkeypatch, agent_name
) -> None:
    monkeypatch.setenv("AGENT_KERNEL_ARENA_HOST_GPU_ID", "0")
    run = tmp_path / "run"
    run.mkdir()
    codex_contract = _write_campaign_codex_contract(run)

    def single_attempt(**kwargs):
        attempt_run = Path(kwargs["run_directory"])
        workspace = attempt_run / "triton2triton_vllm_example_20260807_000000"
        workspace.mkdir()
        _write_result(workspace, 1.0)
        return True, workspace

    completed, canonical = campaign.run_matched_task_campaign(
        eval_config={"campaign": _policy(), "assigned_host_gpu_id": "0"},
        agent=SimpleNamespace(value=agent_name),
        agent_launcher=object(),
        task_name="triton2triton/vllm/example",
        task_config_dir=codex_contract["_task_config_paths"][
            "triton2triton/vllm/example"
        ],
        run_directory=run,
        timestamp="20260807_000000",
        logger=logging.getLogger(__name__),
        task_index=1,
        total_tasks=1,
        single_attempt=single_attempt,
    )

    assert completed is False
    assert canonical is None
    evidence = yaml.safe_load(
        (run / ".campaign_attempts" / _DEFAULT_TASK_COMPONENT / "task_campaign.yaml").read_text()
    )
    assert all(
        "missing_agent_session_receipt" in attempt["eligibility_errors"]
        for attempt in evidence["attempts"]
    )


def test_three_valid_direct_codex_receipts_allow_canonical_projection(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("AGENT_KERNEL_ARENA_HOST_GPU_ID", "0")
    run = tmp_path / "run"
    run.mkdir()
    codex_contract = _write_campaign_codex_contract(run, agent_template="codex")

    def single_attempt(**kwargs):
        attempt_run = Path(kwargs["run_directory"])
        workspace = attempt_run / "triton2triton_vllm_example_20260807_000000"
        workspace.mkdir()
        _write_result(workspace, 1.0)
        receipt_path = Path(kwargs["eval_config"]["campaign_attempt"]["receipt_path"])
        assert kwargs["eval_config"]["campaign_attempt"][
            "comparison_contract_sha256"
        ] == codex_contract["_comparison_contract_sha256"]
        _write_valid_codex_receipt(receipt_path, codex_contract)
        return True, workspace

    try:
        completed, canonical = campaign.run_matched_task_campaign(
            eval_config={"campaign": _policy(), "assigned_host_gpu_id": "0"},
            agent=SimpleNamespace(value="codex"),
            agent_launcher=object(),
            task_name="triton2triton/vllm/example",
            task_config_dir=codex_contract["_task_config_paths"][
                "triton2triton/vllm/example"
            ],
            run_directory=run,
            timestamp="20260807_000000",
            logger=logging.getLogger(__name__),
            task_index=1,
            total_tasks=1,
            single_attempt=single_attempt,
        )
        assert completed is True
        assert canonical is not None
    finally:
        _unlock_apex_receipt_directories(run)


@pytest.mark.parametrize("agent_name", ["apex", "codex"])
def test_all_no_gain_attempts_complete_without_becoming_canonical(
    tmp_path, monkeypatch, agent_name
) -> None:
    monkeypatch.setenv("AGENT_KERNEL_ARENA_HOST_GPU_ID", "0")
    run = tmp_path / "run"
    run.mkdir()
    codex_contract = _write_campaign_codex_contract(
        run,
        agent_template=agent_name,
    )

    def single_attempt(**kwargs):
        attempt_run = Path(kwargs["run_directory"])
        workspace = attempt_run / "triton2triton_vllm_example_20260807_000000"
        workspace.mkdir()
        _write_result(workspace, 0.5)
        report_path = workspace / "task_result.yaml"
        report = yaml.safe_load(report_path.read_text(encoding="utf-8"))
        report |= {
            "evaluation_mode": "no_candidate_baseline_replay_v1",
            "agent_session_score_eligible": False,
            "agent_session_succeeded": True,
            "agent_session_terminal_status": "no_gain",
        }
        report_path.write_text(yaml.safe_dump(report), encoding="utf-8")
        receipt_path = Path(kwargs["eval_config"]["campaign_attempt"]["receipt_path"])
        if agent_name == "apex":
            _write_valid_apex_receipt(
                receipt_path,
                codex_contract,
                status="no_gain",
            )
        else:
            _write_valid_codex_receipt(
                receipt_path,
                codex_contract,
                source_changed=False,
            )
        return True, workspace

    try:
        completed, canonical = campaign.run_matched_task_campaign(
            eval_config={"campaign": _policy(), "assigned_host_gpu_id": "0"},
            agent=SimpleNamespace(value=agent_name),
            agent_launcher=object(),
            task_name="triton2triton/vllm/example",
            task_config_dir=codex_contract["_task_config_paths"][
                "triton2triton/vllm/example"
            ],
            run_directory=run,
            timestamp="20260807_000000",
            logger=logging.getLogger(__name__),
            task_index=1,
            total_tasks=1,
            single_attempt=single_attempt,
        )
        assert completed is False
        assert canonical is None
        evidence = yaml.safe_load(
            (
                run
                / ".campaign_attempts"
                / _DEFAULT_TASK_COMPONENT
                / "task_campaign.yaml"
            ).read_text()
        )
        assert evidence["all_agent_sessions_succeeded"] is True
        assert all(
            attempt["attempt_completed"] is True
            and attempt["selection_eligible"] is False
            and attempt["eligibility_errors"] == []
            for attempt in evidence["attempts"]
        )
    finally:
        _unlock_apex_receipt_directories(run)


@pytest.mark.parametrize("agent_name", ["apex", "codex"])
def test_mixed_no_gain_and_candidate_attempts_select_a_candidate_symmetrically(
    tmp_path, monkeypatch, agent_name
) -> None:
    monkeypatch.setenv("AGENT_KERNEL_ARENA_HOST_GPU_ID", "0")
    run = tmp_path / "run"
    run.mkdir()
    contract = _write_campaign_codex_contract(
        run,
        agent_template=agent_name,
    )

    def single_attempt(**kwargs):
        attempt = int(kwargs["eval_config"]["campaign_attempt"]["index"])
        candidate_ready = attempt > 1
        optimized_ms = {1: 0.25, 2: 1.0, 3: 0.5}[attempt]
        attempt_run = Path(kwargs["run_directory"])
        workspace = attempt_run / "triton2triton_vllm_example_20260807_000000"
        workspace.mkdir()
        _write_result(workspace, optimized_ms)
        report_path = workspace / "task_result.yaml"
        report = yaml.safe_load(report_path.read_text(encoding="utf-8"))
        report |= {
            "evaluation_mode": (
                "candidate_scoring_v1"
                if candidate_ready
                else "no_candidate_baseline_replay_v1"
            ),
            "agent_session_score_eligible": candidate_ready,
            "agent_session_succeeded": True,
            "agent_session_terminal_status": (
                "candidate_ready" if candidate_ready else "no_gain"
            ),
        }
        report_path.write_text(yaml.safe_dump(report), encoding="utf-8")
        receipt_path = Path(kwargs["eval_config"]["campaign_attempt"]["receipt_path"])
        if agent_name == "apex":
            _write_valid_apex_receipt(
                receipt_path,
                contract,
                status="candidate_ready" if candidate_ready else "no_gain",
            )
        else:
            _write_valid_codex_receipt(
                receipt_path,
                contract,
                source_changed=candidate_ready,
            )
        return True, workspace

    try:
        completed, canonical = campaign.run_matched_task_campaign(
            eval_config={"campaign": _policy(), "assigned_host_gpu_id": "0"},
            agent=SimpleNamespace(value=agent_name),
            agent_launcher=object(),
            task_name="triton2triton/vllm/example",
            task_config_dir=contract["_task_config_paths"][
                "triton2triton/vllm/example"
            ],
            run_directory=run,
            timestamp="20260807_000000",
            logger=logging.getLogger(__name__),
            task_index=1,
            total_tasks=1,
            single_attempt=single_attempt,
        )
        assert completed is True
        assert canonical is not None
        result = yaml.safe_load((canonical / "task_result.yaml").read_text())
        assert result["campaign_evidence"]["selected_attempt"] == 3
        evidence = yaml.safe_load(
            (
                run
                / ".campaign_attempts"
                / _DEFAULT_TASK_COMPONENT
                / "task_campaign.yaml"
            ).read_text()
        )
        assert evidence["all_agent_sessions_succeeded"] is True
        assert evidence["attempts"][0]["attempt_completed"] is True
        assert evidence["attempts"][0]["selection_eligible"] is False
        assert evidence["attempts"][0]["eligibility_errors"] == []
        assert all(
            attempt["selection_eligible"] is True
            for attempt in evidence["attempts"][1:]
        )
    finally:
        _unlock_apex_receipt_directories(run)


def test_three_valid_apex_receipts_allow_canonical_projection(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("AGENT_KERNEL_ARENA_HOST_GPU_ID", "0")
    run = tmp_path / "run"
    run.mkdir()
    codex_contract = _write_campaign_codex_contract(run)

    def single_attempt(**kwargs):
        attempt_run = Path(kwargs["run_directory"])
        workspace = attempt_run / "triton2triton_vllm_example_20260807_000000"
        workspace.mkdir()
        _write_result(workspace, 1.0)
        receipt_path = Path(kwargs["eval_config"]["campaign_attempt"]["receipt_path"])
        assert kwargs["eval_config"]["campaign_attempt"][
            "comparison_contract_sha256"
        ] == codex_contract["_comparison_contract_sha256"]
        _write_valid_apex_receipt(
            receipt_path,
            codex_contract,
            status="candidate_ready",
        )
        return True, workspace

    try:
        completed, canonical = campaign.run_matched_task_campaign(
            eval_config={"campaign": _policy(), "assigned_host_gpu_id": "0"},
            agent=SimpleNamespace(value="apex"),
            agent_launcher=object(),
            task_name="triton2triton/vllm/example",
            task_config_dir=codex_contract["_task_config_paths"][
                "triton2triton/vllm/example"
            ],
            run_directory=run,
            timestamp="20260807_000000",
            logger=logging.getLogger(__name__),
            task_index=1,
            total_tasks=1,
            single_attempt=single_attempt,
        )
        assert completed is True
        assert canonical is not None
        evidence = yaml.safe_load(
            (
                run
                / ".campaign_attempts"
                / _DEFAULT_TASK_COMPONENT
                / "task_campaign.yaml"
            ).read_text()
        )
        assert all(
            attempt["session_receipt_binding"]["schema"]
            == "agentkernelarena.apex-attempt-receipt/v6"
            for attempt in evidence["attempts"]
        )
        assert all(
            attempt["session_receipt_binding"]["comparison_contract_sha256"]
            == codex_contract["_comparison_contract_sha256"]
            for attempt in evidence["attempts"]
        )
    finally:
        _unlock_apex_receipt_directories(run)


def test_failed_agent_session_still_gets_diagnostic_evaluation_but_returns_failure(
    tmp_path, monkeypatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("task_type: triton2triton\n", encoding="utf-8")
    attempt = {"task_deadline_monotonic": 1000.0}

    monkeypatch.setattr(aka_main, "setup_workspace", lambda *args, **kwargs: workspace)
    monkeypatch.setattr(
        aka_main, "evaluate_compilation", lambda *args, **kwargs: (True, None)
    )
    monkeypatch.setattr(aka_main, "measure_baseline", lambda *args, **kwargs: [])
    monkeypatch.setattr(aka_main, "snapshot_workspace_harness", lambda *_args: object())
    monkeypatch.setattr(aka_main, "verify_workspace_harness", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        aka_main, "materialize_perf_helpers_in_workspace", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        aka_main,
        "evaluate_kernel",
        lambda *args, **kwargs: {
            "pass_compilation": True,
            "pass_correctness": True,
        },
    )

    def write_result(path, *_args, **_kwargs):
        _write_result(path, 1.0)

    monkeypatch.setattr(aka_main, "write_task_result", write_result)
    monkeypatch.setattr(aka_main, "is_task_complete", lambda *_args, **_kwargs: True)

    def failed_launcher(**_kwargs):
        raise RuntimeError("session failed")

    completed, observed_workspace = aka_main._run_single_task(
        eval_config={"campaign_attempt": attempt},
        agent=aka_main.AgentType.CODEX,
        agent_launcher=failed_launcher,
        task_name="triton2triton/vllm/example",
        task_config_dir=str(config_path),
        run_directory=tmp_path / "run",
        timestamp="20260807_000000",
        logger=logging.getLogger(__name__),
        task_index=1,
        total_tasks=1,
    )

    assert completed is False
    assert observed_workspace == workspace
    report = yaml.safe_load((workspace / "task_result.yaml").read_text())
    assert report["evaluation_mode"] == "diagnostic_baseline_replay_v1"
    assert report["agent_session_score_eligible"] is False
    assert report["agent_session_succeeded"] is False
    assert report["agent_session_error_type"] == "RuntimeError"
    assert "evaluation_elapsed_seconds" in attempt


def _metadata_campaign_attempt(
    tmp_path: Path,
    *,
    receipt_path: Path,
    template: str,
    receipt_schema: str,
) -> dict[str, str]:
    run = tmp_path / f"{template}-metadata-campaign"
    run.mkdir()
    _write_campaign_codex_contract(run, agent_template=template)
    manifest_path = run / "campaign_manifest.yaml"
    live_schema = {
        "apex": "agentkernelarena.apex-attempt-receipt/v6",
        "codex": "agentkernelarena.codex-attempt-receipt/v6",
    }[template]
    if receipt_schema != live_schema:
        manifest_path.chmod(0o644)
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        manifest["agent"]["session_receipt_schema"] = receipt_schema
        manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
        manifest_path.chmod(0o444)
    return {
        "receipt_path": str(receipt_path.resolve()),
        "campaign_manifest_path": str(manifest_path.resolve()),
        "campaign_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
    }


@pytest.mark.parametrize(
    ("agent_name", "declared_schema", "expected"),
    [
        (
            "apex",
            "agentkernelarena.apex-attempt-receipt/v6",
            "agentkernelarena.apex-attempt-receipt/v6",
        ),
        (
            "codex",
            "agentkernelarena.codex-attempt-receipt/v6",
            "agentkernelarena.codex-attempt-receipt/v6",
        ),
        ("unknown", None, None),
        ("apex", 4, None),
        ("codex", "agentkernelarena.codex-attempt-receipt/v5", None),
    ],
)
def test_live_session_receipt_schema_registry_fails_closed(
    agent_name, declared_schema, expected
) -> None:
    assert (
        campaign.resolve_session_receipt_schema(agent_name, declared_schema)
        == expected
    )


def test_apex_no_gain_metadata_never_marks_baseline_as_scoreable(tmp_path) -> None:
    receipt_path = tmp_path / "session_receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema": "agentkernelarena.apex-attempt-receipt/v6",
                "session_succeeded": True,
                "terminal_status": "no_gain",
            }
        ),
        encoding="utf-8",
    )
    receipt_path.chmod(0o444)

    metadata = aka_main._campaign_evaluation_metadata(
        agent=aka_main.AgentType.APEX,
        campaign_attempt=_metadata_campaign_attempt(
            tmp_path,
            receipt_path=receipt_path,
            template="apex",
            receipt_schema="agentkernelarena.apex-attempt-receipt/v6",
        ),
        agent_error=None,
    )

    assert metadata == {
        "evaluation_mode": "no_candidate_baseline_replay_v1",
        "agent_session_score_eligible": False,
        "agent_session_succeeded": True,
        "agent_session_error_type": None,
        "agent_session_terminal_status": "no_gain",
    }


def test_apex_candidate_ready_metadata_is_scoreable(tmp_path) -> None:
    receipt_path = tmp_path / "session_receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema": "agentkernelarena.apex-attempt-receipt/v6",
                "session_succeeded": True,
                "terminal_status": "candidate_ready",
            }
        ),
        encoding="utf-8",
    )
    receipt_path.chmod(0o444)

    metadata = aka_main._campaign_evaluation_metadata(
        agent=aka_main.AgentType.APEX,
        campaign_attempt=_metadata_campaign_attempt(
            tmp_path,
            receipt_path=receipt_path,
            template="apex",
            receipt_schema="agentkernelarena.apex-attempt-receipt/v6",
        ),
        agent_error=None,
    )

    assert metadata["evaluation_mode"] == "candidate_scoring_v1"
    assert metadata["agent_session_score_eligible"] is True
    assert metadata["agent_session_succeeded"] is True
    assert metadata["agent_session_terminal_status"] == "candidate_ready"


@pytest.mark.parametrize(
    ("changed_files", "mode", "eligible", "terminal"),
    [
        (["kernel.py"], "candidate_scoring_v1", True, "candidate_ready"),
        ([], "no_candidate_baseline_replay_v1", False, "no_gain"),
    ],
)
def test_direct_codex_metadata_requires_a_source_delta(
    tmp_path, changed_files, mode, eligible, terminal
) -> None:
    receipt_path = tmp_path / "codex_session_receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema": "agentkernelarena.codex-attempt-receipt/v6",
                "session_succeeded": True,
                "workspace_integrity": {
                    "final_changes": {"changed_files": changed_files}
                },
            }
        ),
        encoding="utf-8",
    )
    receipt_path.chmod(0o444)

    metadata = aka_main._campaign_evaluation_metadata(
        agent=aka_main.AgentType.CODEX,
        campaign_attempt=_metadata_campaign_attempt(
            tmp_path,
            receipt_path=receipt_path,
            template="codex",
            receipt_schema="agentkernelarena.codex-attempt-receipt/v6",
        ),
        agent_error=None,
    )

    assert metadata["evaluation_mode"] == mode
    assert metadata["agent_session_score_eligible"] is eligible
    assert metadata["agent_session_succeeded"] is True
    assert metadata["agent_session_terminal_status"] == terminal


@pytest.mark.parametrize(
    ("agent", "template", "receipt_fields", "mode", "eligible", "terminal"),
    [
        (
            aka_main.AgentType.APEX,
            "apex",
            {"terminal_status": "candidate_ready"},
            "candidate_scoring_v1",
            True,
            "candidate_ready",
        ),
        (
            aka_main.AgentType.APEX,
            "apex",
            {"terminal_status": "no_gain"},
            "no_candidate_baseline_replay_v1",
            False,
            "no_gain",
        ),
        (
            aka_main.AgentType.CODEX,
            "codex",
            {"workspace_integrity": {"final_changes": {"changed_files": ["kernel.py"]}}},
            "candidate_scoring_v1",
            True,
            "candidate_ready",
        ),
        (
            aka_main.AgentType.CODEX,
            "codex",
            {"workspace_integrity": {"final_changes": {"changed_files": []}}},
            "no_candidate_baseline_replay_v1",
            False,
            "no_gain",
        ),
    ],
)
def test_live_v6_campaign_metadata_is_symmetric_for_apex_and_direct_codex(
    tmp_path, agent, template, receipt_fields, mode, eligible, terminal
) -> None:
    schema = {
        "apex": "agentkernelarena.apex-attempt-receipt/v6",
        "codex": "agentkernelarena.codex-attempt-receipt/v6",
    }[template]
    receipt_path = tmp_path / f"{template}_live_session_receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema": schema,
                "session_succeeded": True,
                **receipt_fields,
            }
        ),
        encoding="utf-8",
    )
    receipt_path.chmod(0o444)

    metadata = aka_main._campaign_evaluation_metadata(
        agent=agent,
        campaign_attempt=_metadata_campaign_attempt(
            tmp_path,
            receipt_path=receipt_path,
            template=template,
            receipt_schema=schema,
        ),
        agent_error=None,
    )

    assert metadata == {
        "evaluation_mode": mode,
        "agent_session_score_eligible": eligible,
        "agent_session_succeeded": True,
        "agent_session_error_type": None,
        "agent_session_terminal_status": terminal,
    }


@pytest.mark.parametrize(
    ("agent", "template"),
    [
        (aka_main.AgentType.APEX, "apex"),
        (aka_main.AgentType.CODEX, "codex"),
    ],
)
def test_campaign_metadata_rejects_old_sealed_schema(
    tmp_path, agent, template
) -> None:
    schema = f"agentkernelarena.{template}-attempt-receipt/v5"
    receipt_path = tmp_path / f"{template}_unsupported_receipt.json"
    receipt_path.write_text(
        json.dumps({"schema": schema, "session_succeeded": True}),
        encoding="utf-8",
    )
    receipt_path.chmod(0o444)

    metadata = aka_main._campaign_evaluation_metadata(
        agent=agent,
        campaign_attempt=_metadata_campaign_attempt(
            tmp_path,
            receipt_path=receipt_path,
            template=template,
            receipt_schema=schema,
        ),
        agent_error=None,
    )

    assert metadata["evaluation_mode"] == "diagnostic_unbound_session_replay_v1"
    assert metadata["agent_session_score_eligible"] is False
    assert metadata["agent_session_succeeded"] is False
    assert metadata["agent_session_error_type"] == "CampaignManifestMetadataError"


@pytest.mark.parametrize(
    ("agent", "template", "other_template"),
    [
        (aka_main.AgentType.APEX, "apex", "codex"),
        (aka_main.AgentType.CODEX, "codex", "apex"),
    ],
)
def test_campaign_metadata_rejects_cross_arm_live_receipt_substitution(
    tmp_path, agent, template, other_template
) -> None:
    live_schemas = {
        "apex": "agentkernelarena.apex-attempt-receipt/v6",
        "codex": "agentkernelarena.codex-attempt-receipt/v6",
    }
    expected_schema = live_schemas[template]
    receipt_path = tmp_path / f"{template}_substituted_receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema": live_schemas[other_template],
                "session_succeeded": True,
            }
        ),
        encoding="utf-8",
    )
    receipt_path.chmod(0o444)

    metadata = aka_main._campaign_evaluation_metadata(
        agent=agent,
        campaign_attempt=_metadata_campaign_attempt(
            tmp_path,
            receipt_path=receipt_path,
            template=template,
            receipt_schema=expected_schema,
        ),
        agent_error=None,
    )

    assert metadata["evaluation_mode"] == "diagnostic_unbound_session_replay_v1"
    assert metadata["agent_session_score_eligible"] is False
    assert metadata["agent_session_succeeded"] is False
    assert metadata["agent_session_error_type"] == "ApexReceiptMetadataError"


def test_metadata_rejects_receipt_schema_not_selected_by_sealed_manifest(
    tmp_path,
) -> None:
    receipt_path = tmp_path / "session_receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema": "agentkernelarena.apex-attempt-receipt/v4",
                "session_succeeded": True,
                "terminal_status": "candidate_ready",
            }
        ),
        encoding="utf-8",
    )
    receipt_path.chmod(0o444)

    metadata = aka_main._campaign_evaluation_metadata(
        agent=aka_main.AgentType.APEX,
        campaign_attempt=_metadata_campaign_attempt(
            tmp_path,
            receipt_path=receipt_path,
            template="apex",
            receipt_schema="agentkernelarena.apex-attempt-receipt/v6",
        ),
        agent_error=None,
    )

    assert metadata["evaluation_mode"] == "diagnostic_unbound_session_replay_v1"
    assert metadata["agent_session_score_eligible"] is False
