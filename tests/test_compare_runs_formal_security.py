import copy
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
    payload = copy.deepcopy(payload)
    if path.name == "campaign_manifest.yaml" and payload.get("schema") == "aka.matched-campaign/v1":
        run_config_contract = payload["comparison_contract"]["run_config"]
        run_config = path.parent / "formal_run_config.yaml"
        run_config.write_text(
            yaml.safe_dump(
                {
                    "agent": {"template": payload["agent"]["template"]},
                    **run_config_contract["effective_config"],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        payload["configuration"].update(
            {
                "run_config_path": str(run_config.resolve()),
                "run_config_sha256": postprocessing._sha256_file(run_config),
                "run_config_contract": run_config_contract,
            }
        )
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


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


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


def _measurement() -> dict:
    return {
        "contract": "aka_native_100_repetition_external_score",
        "owner": "AgentKernelArena centralized evaluator",
        "configured_repetitions_per_test_case": 100,
        "is_apex_kernel_measurement_v1": False,
        "is_apex_canonical_300_sample_grade": False,
    }


def _runtime_isolation_receipt() -> dict:
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
                "/proc/acpi",
                "/proc/asound",
                "/proc/scsi",
                "/sys/devices/virtual/powercap",
                "/sys/firmware",
                "/proc/interrupts",
                "/proc/kcore",
                "/proc/keys",
                "/proc/latency_stats",
                "/proc/sched_debug",
                "/proc/timer_list",
                "/proc/timer_stats",
            ],
            "docker_readonly_paths_rebuilt": [
                "/proc/bus",
                "/proc/fs",
                "/proc/irq",
                "/proc/sys",
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
            "command_pid_namespace": (
                "nested_codex_unshared_inside_private_attempt_pidns_v1"
            ),
            "command_network": "managed_profile_denied_live_probe_v1",
            "command_gpu_access": (
                "sealed_memfd_immutable_path_bwrap_and_single_gpu_probe_v1"
            ),
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
            "yama_ptrace_scope": 1,
        },
        "bubblewrap": {
            "resolved_path": "/usr/bin/bwrap",
            "sha256": "d78807229d616606e339c5988392b9e0ab4a6a6998fa51e4590837f426a12fca",
            "version": "bubblewrap 0.6.1",
            "execution_transport": "sealed_memfd_proc_self_fd",
        },
        "codex_cli": {
            "resolved_path": "/opt/node/lib/node_modules/@openai/codex/bin/codex.js",
            "sha256": "a" * 64,
            "version": "codex-cli test",
        },
        "codex_requirements": {
            "resolved_path": "/etc/codex/requirements.toml",
            "sha256": "0c68db4f0ee56b42f15af2896e51f4e667d9d6f86d9d3864dfec571278572ade",
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
            "sha256": "9271bd346d1ea5f878c8f345537e8464a56156b82f956942b66b82feb61791ef",
            "size_bytes": 2381,
            "interpreter": "/usr/bin/python3 -I",
            "real_bwrap": "/usr/bin/bwrap",
            "real_bwrap_sha256": "d78807229d616606e339c5988392b9e0ab4a6a6998fa51e4590837f426a12fca",
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


def _backend_closure() -> dict:
    files = [
        {
            "path": "bin/codex.js",
            "mode": 0o555,
            "size": 4096,
            "sha256": "a" * 64,
        },
        {
            "path": "package.json",
            "mode": 0o444,
            "size": 1024,
            "sha256": "c" * 64,
        },
    ]
    material = {
        "schema": campaign.BACKEND_CLOSURE_SCHEMA,
        "backend": "codex",
        "launcher": {
            "requested_path": "/opt/node/bin/codex",
            "symlink_chain": [
                {
                    "path": "/opt/node/bin/codex",
                    "target": "../lib/node_modules/@openai/codex/bin/codex.js",
                }
            ],
            "resolved_path": "/opt/node/lib/node_modules/@openai/codex/bin/codex.js",
            "mode": 0o555,
            "size": 4096,
            "sha256": "a" * 64,
        },
        "interpreter": {
            "resolved_path": "/opt/node/bin/node",
            "mode": 0o555,
            "size": 1024,
            "sha256": "d" * 64,
        },
        "components": [
            {
                "kind": "node_package",
                "root": "/opt/node/lib/node_modules/@openai/codex",
                "files": files,
                "files_sha256": _digest(files),
            }
        ],
    }
    return {**material, "closure_sha256": _digest(material)}


def _aka_runtime_snapshot() -> dict:
    runtime_root = "/test/aka-runtime"
    manifest_sha256 = "4" * 64
    mount = {
        "path": runtime_root,
        "mount_id": 101,
        "parent_id": 1,
        "major_minor": "0:42",
        "root": "/",
        "mount_point": runtime_root,
        "mount_options": ["ro", "nosuid", "nodev"],
        "filesystem_type": "fuse.squashfuse",
        "source": "squashfuse",
        "super_options": ["ro"],
        "read_only": True,
        "nested_mounts": [],
    }
    receipt_material = {
        "schema": campaign.IMMUTABLE_MOUNT_RECEIPT_SCHEMA,
        "policy_id": "sealed_memfd_squashfs_read_only_v1",
        "manifest_sha256": manifest_sha256,
        "image_sha256": "5" * 64,
        "memfd_seals": [
            "F_SEAL_WRITE",
            "F_SEAL_SHRINK",
            "F_SEAL_GROW",
            "F_SEAL_SEAL",
        ],
        "mount": mount,
    }
    receipt = {**receipt_material, "sha256": _digest(receipt_material)}
    receipt_file = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    return {
        "schema": "aka.execution-snapshot-runtime/v1",
        "root": runtime_root,
        "manifest_path": "/test/evidence/aka-runtime-manifest.json",
        "manifest_file_sha256": "8" * 64,
        "manifest_sha256": manifest_sha256,
        "mount_receipt_path": "/test/evidence/aka-runtime-mount-receipt.json",
        "mount_receipt_file_sha256": hashlib.sha256(
            receipt_file.encode()
        ).hexdigest(),
        "mount_receipt_sha256": receipt["sha256"],
        "mount_receipt_schema": campaign.IMMUTABLE_MOUNT_RECEIPT_SCHEMA,
        "mount_receipt": receipt,
    }


def _runtime_contract(task_names: list[str]) -> tuple[dict, dict]:
    plan_sha256 = "d" * 64
    unique_id = "0x0000000000000001"
    exclusivity_material = {
        "schema": "aka.gpu-exclusivity-receipt/v1",
        "policy": "physical_unique_id_flock_plus_kfd_preflight_v1",
        "gpu_boundary_plan_sha256": plan_sha256,
        "run_name": "formal-test-run",
        "runner_pid": 1001,
        "observed_at_ns": 123456789,
        "leases": [
            {"unique_id": unique_id, "lock_path": "/tmp/aka-gpu-lease-0.lock"}
        ],
        "protected_device_paths": ["/dev/dri/renderD128", "/dev/kfd"],
        "foreign_device_owners": [],
        "authoritative_kfd_process_inventory": {
            "source": "librocm_smi64.rsmi_compute_process_info_get",
            "artifact_sha256": "1" * 64,
            "document_sha256": "2" * 64,
            "library": {
                "path": "/opt/rocm/lib/librocm_smi64.so.7",
                "sha256": "3" * 64,
            },
            "observed_at_ns": 123456780,
            "query": {
                "init_status": 0,
                "count_status": 0,
                "count_hint": 0,
                "fetch_status": 0,
                "fetch_capacity": 64,
                "fetched_count": 0,
                "shutdown_status": 0,
            },
            "pids": [],
            "process_count": 0,
            "verified_empty": True,
            "path": "/test/evidence/kfd-process-inventory.json",
        },
        "supplementary_proc_audit": {
            "owners": [],
            "complete": True,
            "inaccessible_pid_count": 0,
            "inaccessible_pids_sha256": _digest([]),
            "inaccessible_pids_sample": [],
        },
        "proof_basis": "rocm_smi_kfd_process_api_v1",
        "exclusivity_verified": True,
    }
    exclusivity = {
        **exclusivity_material,
        "sha256": _digest(exclusivity_material),
    }
    gpu = {
        "policy": "deterministic_task_gpu_v1",
        "ordered_host_gpu_ids": ["0"],
        "target_gpu_model": "MI355X",
        "gpu_arch": "gfx950",
        "gpu_boundary_plan_sha256": plan_sha256,
        "kfd_device": {"path": "/dev/kfd", "major": 235, "minor": 0},
        "exclusivity": exclusivity,
        "devices": [
            {
                "host_device_id": "0",
                "unique_id": unique_id,
                "serial_number": "TEST-SERIAL-0",
                "card_series": "AMD Instinct MI355X",
                "observed_gfx_version": "gfx950",
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
    runtime = {
        "docker": {
            "reference": "rocm/vllm:test",
            "image_id": "sha256:" + "e" * 64,
            "repo_digests": ["rocm/vllm@sha256:" + "f" * 64],
        },
        "gpu": gpu,
        "isolation": _runtime_isolation_receipt(),
        "aka_execution_snapshot": _aka_runtime_snapshot(),
    }
    comparison_runtime = campaign.comparison_runtime_projection(runtime)
    assert comparison_runtime is not None
    return runtime, comparison_runtime


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
    closure = _backend_closure()
    codex = {
        "attempt_timeout_seconds": 3600,
        "backend": "codex",
        "codex_binary_sha256": "a" * 64,
        "codex_version": "codex-cli test",
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
        "model": "gpt-5.5",
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
            "status_sha256": hashlib.sha256(b"").hexdigest(),
            "execution_manifest_schema": campaign.EXECUTION_MANIFEST_SCHEMA,
            "execution_manifest_sha256": "4" * 64,
            "git_evidence_policy_id": "head_tree_direct_bytes_no_filters_v1",
        },
        "apex": {
            "commit": "5" * 40,
            "dirty": False,
            "status_sha256": hashlib.sha256(b"").hexdigest(),
            "runtime_manifest_sha256": "7" * 64,
        },
    }
    runtime, comparison_runtime = _runtime_contract(task_names)
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
    effective_run_config = {
        "campaign": _policy(),
        "tasks": task_names,
        "target_gpu_model": "MI355X",
        "log_directory": "/test/logs",
        "workspace_directory_prefix": "/test/workspace",
    }
    run_config = {
        "schema": "aka.formal-run-config/v1",
        "effective_config": effective_run_config,
        "effective_config_sha256": _digest(effective_run_config),
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
        "policy": _policy(),
        "measurement": _measurement(),
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
    transport = campaign.FORMAL_AGENT_TRANSPORT_TREATMENTS[arm]
    agent = {
        **codex,
        "max_process_output_bytes": transport["max_process_output_bytes"],
        "structured_stream_overflow_policy": transport["overflow_policy"],
        "codex_binary": "/opt/node/bin/codex",
        "agent_config_sha256": "1" * 64,
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
        "policy": _policy(),
        "measurement": _measurement(),
        "agent": agent,
        "comparison_contract": comparison,
        "comparison_contract_sha256": hashlib.sha256(
            json.dumps(comparison, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "repositories": repositories,
        "runtime": runtime,
        "evaluator_files_sha256": evaluator,
        "configuration": {
            "run_config_path": "/test/run_config.yaml",
            "run_config_sha256": "2" * 64,
            "run_config_contract": run_config,
            "tasks": tasks,
        },
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


def _formal_task_entry(manifest: dict, task_name: str) -> tuple[dict, dict]:
    task = next(
        task
        for task in manifest["configuration"]["tasks"]
        if task["task_name"] == task_name
    )
    mapping = next(
        mapping
        for mapping in manifest["runtime"]["gpu"]["task_mapping"]
        if mapping["task_name"] == task_name
    )
    return task, mapping


def _attempt_campaign_binding(
    run: Path, manifest: dict, task_name: str, attempt: int
) -> dict:
    task, mapping = _formal_task_entry(manifest, task_name)
    return {
        "schema": "aka.attempt-campaign-binding/v1",
        "formal_execution_sha256": manifest["formal_execution_sha256"],
        "campaign_manifest_path": str(
            (run / "campaign_manifest.yaml").resolve(strict=True)
        ),
        "campaign_manifest_sha256": postprocessing._sha256_file(
            run / "campaign_manifest.yaml"
        ),
        "comparison_contract_sha256": manifest[
            "comparison_contract_sha256"
        ],
        "backend_runtime_closure_sha256": manifest["comparison_contract"][
            "codex"
        ]["backend_runtime_closure_sha256"],
        "task_package_manifest_sha256": task["package_manifest_sha256"],
        "task_config_sha256": task["config_sha256"],
        "task_name": task_name,
        "task_index": task["task_index"],
        "total_tasks": len(manifest["configuration"]["tasks"]),
        "attempt_index": attempt,
        "attempt_count": 3,
        "assigned_host_gpu_id": mapping["assigned_host_gpu_id"],
    }


def _attempt_receipt(
    run: Path, manifest: dict, task_name: str, attempt: int
) -> dict:
    arm = manifest["agent"]["template"]
    _, mapping = _formal_task_entry(manifest, task_name)
    gpu = manifest["runtime"]["gpu"]
    device = next(
        device
        for device in gpu["devices"]
        if device["host_device_id"] == mapping["assigned_host_gpu_id"]
    )
    return {
        "schema": manifest["agent"]["session_receipt_schema"],
        "comparison_contract_sha256": manifest[
            "comparison_contract_sha256"
        ],
        "session_succeeded": True,
        "terminal_status": "candidate_ready",
        "invocation": {"attempt": attempt, "backend": "codex"},
        "attempt_process_cleanup": {"verified_absent": True},
        "gpu": {
            "policy": "physical_device_boundary_with_host_exclusivity_v1",
            "plan_sha256": gpu["gpu_boundary_plan_sha256"],
            "boundary_receipt_sha256": "b" * 64,
            "exclusivity_receipt_sha256": gpu["exclusivity"]["sha256"],
            "exclusivity_verified": True,
            "host_gpu_id": mapping["assigned_host_gpu_id"],
            "unique_id": device["unique_id"],
            "allowed_render_nodes": device["render_nodes"],
            "runtime_identity": {
                "visible_physical_gpu_count": 1,
                "rocm_smi_identity": {"unique_id": device["unique_id"]},
                "torch": {"device_count": 1},
            },
        },
        "workspace_integrity": (
            {"final_changes": {"changed_files": ["kernel.py"]}}
            if arm == "codex"
            else {"pre_apply_unchanged": True}
        ),
        "lineage": (
            {
                "prompt_event": {
                    "binding": "apex.prompt_sent_event_cas/v1",
                    "event_id": f"prompt-{attempt}",
                    "sha256": f"{attempt}" * 64,
                    "size_bytes": 128,
                    "stdin_transport_attested": False,
                }
            }
            if arm == "apex"
            else None
        ),
        "campaign_binding": _attempt_campaign_binding(
            run, manifest, task_name, attempt
        ),
    }


def _performance_case(execution_time_ms: float) -> dict:
    return {
        "test_case_id": "case-1",
        "shape": [1],
        "params": {},
        "execution_time_ms": execution_time_ms,
        "benchmark_samples": 100,
        "benchmark_method": "median",
    }


def _attempt_result(
    task_name: str, *, baseline_ms: float, optimized_ms: float
) -> dict:
    result = _result(task_name, baseline_ms / optimized_ms)
    result.update(
        {
            "base_execution_time": baseline_ms,
            "best_optimized_execution_time": optimized_ms,
            "valid_baseline_cases": 1,
            "valid_optimized_cases": 1,
            "speedup_calculation_error_message": None,
            "benchmark_method_consistent": True,
            "baseline_benchmark_methods": ["median"],
            "optimized_benchmark_methods": ["median"],
            "evaluation_mode": "candidate_scoring_v1",
            "agent_session_score_eligible": True,
            "agent_session_succeeded": True,
            "agent_session_terminal_status": "candidate_ready",
        }
    )
    return result


def _write_canonical_workspace(
    run: Path, task_name: str, speedup: float = 2.0
) -> Path:
    safe_name = task_name.replace("/", "_")
    attempt_root = (
        run
        / ".campaign_attempts"
        / campaign.campaign_task_path_component(task_name)
    )
    manifest_path = run / "campaign_manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    task, mapping = _formal_task_entry(manifest, task_name)
    attempts = []
    workspaces = {}
    optimized_times = (1.25, 1.0, 1.1)
    for attempt, optimized_ms in enumerate(optimized_times, 1):
        workspace = (
            attempt_root
            / f"attempt_{attempt:02d}"
            / f"{safe_name}_{TIMESTAMP}"
        )
        workspace.mkdir(parents=True)
        result = _attempt_result(
            task_name, baseline_ms=speedup, optimized_ms=optimized_ms
        )
        _write_read_only_yaml(workspace / "task_result.yaml", result)
        _write_read_only_yaml(
            workspace / "baseline_perf.yaml",
            {"test_cases": [_performance_case(speedup)]},
        )
        _write_read_only_yaml(
            workspace / "optimized_perf.yaml",
            {"test_cases": [_performance_case(optimized_ms)]},
        )
        workspace_manifest = {
            path.relative_to(workspace).as_posix(): postprocessing._sha256_file(path)
            for path in sorted(workspace.rglob("*"))
            if path.is_file()
        }
        receipt_path = workspace.parent / "session_receipt.json"
        receipt = _attempt_receipt(run, manifest, task_name, attempt)
        receipt_path.write_text(
            json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
        )
        receipt_path.chmod(0o444)
        receipt_binding = postprocessing._static_session_receipt_binding(
            receipt, agent_template=manifest["agent"]["template"]
        )
        attempts.append(
            {
                "attempt": attempt,
                "session": f"fresh-{attempt:02d}",
                "attempt_completed": True,
                "workspace": str(workspace.relative_to(run)),
                "central_evaluator_report": str(
                    (workspace / "task_result.yaml").relative_to(run)
                ),
                "central_evaluator_report_sha256": workspace_manifest[
                    "task_result.yaml"
                ],
                "workspace_manifest_sha256": postprocessing._canonical_json_digest(
                    workspace_manifest
                ),
                "session_receipt": str(receipt_path.relative_to(run)),
                "session_receipt_sha256": postprocessing._sha256_file(
                    receipt_path
                ),
                "session_succeeded": True,
                "session_receipt_binding": receipt_binding,
                "session_receipt_binding_sha256": (
                    postprocessing._canonical_json_digest(receipt_binding)
                ),
                "pass_compilation": True,
                "pass_correctness": True,
                "optimized_execution_time_ms": optimized_ms,
                "speedup_ratio": speedup / optimized_ms,
                "benchmark_method_consistent": True,
                "evaluation_mode": "candidate_scoring_v1",
                "agent_session_score_eligible": True,
                "agent_session_terminal_status": "candidate_ready",
                "selection_eligible": True,
                "measured_rate_per_ms": 1.0 / optimized_ms,
                "eligibility_errors": [],
            }
        )
        workspaces[attempt] = (workspace, workspace_manifest, result)
    selected_attempt = 2
    selected, selected_manifest, result = workspaces[selected_attempt]
    selected_manifest_sha256 = postprocessing._canonical_json_digest(
        selected_manifest
    )
    task_campaign_path = attempt_root / "task_campaign.yaml"
    task_campaign = {
        "schema": "aka.matched-task-attempts/v1",
        "formal_execution_sha256": manifest["formal_execution_sha256"],
        "task_name": task_name,
        "assigned_host_gpu_id": mapping["assigned_host_gpu_id"],
        "task_index": task["task_index"],
        "total_tasks": len(manifest["configuration"]["tasks"]),
        "task_config_path": task["config_path"],
        "task_config_sha256": task["config_sha256"],
        "task_package_manifest_sha256": task["package_manifest_sha256"],
        "gpu_exclusivity_verified": True,
        "campaign_manifest_sha256": postprocessing._sha256_file(manifest_path),
        "comparison_contract_sha256": manifest["comparison_contract_sha256"],
        "campaign_manifest_unchanged": True,
        "policy": copy.deepcopy(manifest["policy"]),
        "measurement_contract": manifest["measurement"]["contract"],
        "is_apex_canonical_300_sample_grade": False,
        "attempts": attempts,
        "selected_attempt": selected_attempt,
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
        "attempt_count": 3,
        "selected_attempt": selected_attempt,
        "selection_policy": manifest["policy"]["selection_policy"],
        "selected_measured_rate_per_ms": 1.0,
        "attempt_manifest": str(task_campaign_path.relative_to(run)),
        "measurement_contract": manifest["measurement"]["contract"],
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
    _write_read_only_yaml(
        canonical / "baseline_perf.yaml",
        {"test_cases": [_performance_case(speedup)]},
    )
    _write_read_only_yaml(
        canonical / "optimized_perf.yaml",
        {"test_cases": [_performance_case(1.0)]},
    )
    return canonical


def _write_failed_task(run: Path, task_name: str) -> None:
    manifest_path = run / "campaign_manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    task, mapping = _formal_task_entry(manifest, task_name)
    evidence_path = (
        run
        / ".campaign_attempts"
        / campaign.campaign_task_path_component(task_name)
        / "task_campaign.yaml"
    )
    evidence = {
        "schema": "aka.matched-task-attempts/v1",
        "formal_execution_sha256": manifest["formal_execution_sha256"],
        "task_name": task_name,
        "assigned_host_gpu_id": mapping["assigned_host_gpu_id"],
        "task_index": task["task_index"],
        "total_tasks": len(manifest["configuration"]["tasks"]),
        "task_config_path": task["config_path"],
        "task_config_sha256": task["config_sha256"],
        "task_package_manifest_sha256": task["package_manifest_sha256"],
        "gpu_exclusivity_verified": True,
        "campaign_manifest_sha256": postprocessing._sha256_file(manifest_path),
        "comparison_contract_sha256": manifest["comparison_contract_sha256"],
        "campaign_manifest_unchanged": True,
        "policy": copy.deepcopy(manifest["policy"]),
        "measurement_contract": manifest["measurement"]["contract"],
        "is_apex_canonical_300_sample_grade": False,
        "all_attempts_centrally_evaluated": False,
        "all_agent_sessions_succeeded": False,
        "within_evaluator_allowance": True,
        "within_task_timeout": True,
        "selected_attempt": None,
        "attempts": [{
            "attempt": 1,
            "session": "fresh-01",
            "attempt_completed": False,
            "workspace": None,
            "central_evaluator_report": None,
            "selection_eligible": False,
            "measured_rate_per_ms": 0.0,
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


def _rewrite_sealed_yaml(path: Path, payload: dict) -> None:
    path.chmod(0o644)
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    path.chmod(0o444)


def _rewrite_sealed_json(path: Path, payload: dict) -> None:
    path.chmod(0o644)
    path.write_text(
        json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
    )
    path.chmod(0o444)


def _refresh_receipt_record(run: Path, attempt: int, receipt: dict) -> Path:
    root = (
        run
        / ".campaign_attempts"
        / campaign.campaign_task_path_component(TASK)
    )
    receipt_path = root / f"attempt_{attempt:02d}" / "session_receipt.json"
    _rewrite_sealed_json(receipt_path, receipt)
    campaign_path = root / "task_campaign.yaml"
    task_campaign = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    record = next(
        record
        for record in task_campaign["attempts"]
        if record["attempt"] == attempt
    )
    binding = postprocessing._static_session_receipt_binding(
        receipt,
        agent_template=yaml.safe_load(
            (run / "campaign_manifest.yaml").read_text(encoding="utf-8")
        )["agent"]["template"],
    )
    record["session_receipt_sha256"] = postprocessing._sha256_file(receipt_path)
    record["session_receipt_binding"] = binding
    record["session_receipt_binding_sha256"] = (
        postprocessing._canonical_json_digest(binding)
    )
    _rewrite_sealed_yaml(campaign_path, task_campaign)
    return campaign_path


def _refresh_comparison_digest(manifest: dict) -> None:
    manifest["comparison_contract_sha256"] = _digest(
        manifest["comparison_contract"]
    )


def _assert_manifest_rejected(tmp_path: Path, manifest: dict) -> None:
    arm = manifest["agent"]["template"]
    run = (
        tmp_path
        / f"workspace_MI355X_{arm}"
        / f"run_{TIMESTAMP}_formal"
    )
    run.mkdir(parents=True)
    manifest_bytes = yaml.safe_dump(manifest, sort_keys=True).encode()
    with pytest.raises(ValueError, match="comparison/cohort/agent binding"):
        compare_runs._formal_manifest_context(run, manifest, manifest_bytes)


@pytest.mark.parametrize("tamper", ["raw_config", "projection_digest"])
def test_compare_rejects_run_config_tamper(tmp_path: Path, tamper: str) -> None:
    run = (
        tmp_path
        / "workspace_MI355X_codex"
        / f"run_{TIMESTAMP}_formal"
    )
    run.mkdir(parents=True)
    manifest = _manifest([TASK], "codex")
    if tamper == "projection_digest":
        forged = copy.deepcopy(manifest["comparison_contract"]["run_config"])
        forged["effective_config_sha256"] = "0" * 64
        manifest["comparison_contract"]["run_config"] = copy.deepcopy(forged)
        manifest["configuration"]["run_config_contract"] = copy.deepcopy(forged)
        _refresh_comparison_digest(manifest)
    manifest_path = run / "campaign_manifest.yaml"
    _write_read_only_yaml(manifest_path, manifest)
    manifest_bytes = manifest_path.read_bytes()
    sealed_manifest = yaml.safe_load(manifest_bytes)
    if tamper == "raw_config":
        run_config = Path(sealed_manifest["configuration"]["run_config_path"])
        run_config.write_text(
            run_config.read_text(encoding="utf-8")
            + "unexpected_output: /tmp/forged\n",
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="comparison/cohort/agent binding"):
        compare_runs._formal_manifest_context(
            run, sealed_manifest, manifest_bytes
        )


def test_v5_manifest_rejects_forged_static_squashfs_receipt(
    tmp_path: Path,
) -> None:
    manifest = _manifest([TASK], "codex")
    snapshot = manifest["runtime"]["aka_execution_snapshot"]
    receipt = snapshot["mount_receipt"]
    receipt["mount"]["read_only"] = False
    receipt_material = {key: value for key, value in receipt.items() if key != "sha256"}
    receipt["sha256"] = _digest(receipt_material)
    snapshot["mount_receipt_sha256"] = receipt["sha256"]
    snapshot["mount_receipt_file_sha256"] = hashlib.sha256(
        (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()
    manifest["comparison_contract"]["runtime"] = (
        compare_runs._project_comparison_runtime(manifest["runtime"])
    )
    _refresh_comparison_digest(manifest)

    _assert_manifest_rejected(tmp_path, manifest)


def test_v5_manifest_rejects_digest_valid_but_incomplete_backend_closure(
    tmp_path: Path,
) -> None:
    manifest = _manifest([TASK], "codex")
    closure_material = {
        "schema": campaign.BACKEND_CLOSURE_SCHEMA,
        "backend": "codex",
    }
    closure = {
        **closure_material,
        "closure_sha256": _digest(closure_material),
    }
    for agent in (
        manifest["agent"],
        manifest["comparison_contract"]["codex"],
    ):
        agent["backend_runtime_closure"] = closure
        agent["backend_runtime_closure_sha256"] = closure["closure_sha256"]
    _refresh_comparison_digest(manifest)

    _assert_manifest_rejected(tmp_path, manifest)


def test_v5_manifest_rejects_launcher_component_identity_drift(
    tmp_path: Path,
) -> None:
    manifest = _manifest([TASK], "codex")
    closure = copy.deepcopy(manifest["agent"]["backend_runtime_closure"])
    component = closure["components"][0]
    component["files"][0]["sha256"] = "b" * 64
    component["files_sha256"] = _digest(component["files"])
    closure_material = {
        key: value for key, value in closure.items() if key != "closure_sha256"
    }
    closure["closure_sha256"] = _digest(closure_material)
    for agent in (manifest["agent"], manifest["comparison_contract"]["codex"]):
        agent["backend_runtime_closure"] = copy.deepcopy(closure)
        agent["backend_runtime_closure_sha256"] = closure["closure_sha256"]
    _refresh_comparison_digest(manifest)

    _assert_manifest_rejected(tmp_path, manifest)


@pytest.mark.parametrize(
    "tamper",
    ["policy", "measurement", "docker", "gpu", "isolation", "agent_policy"],
)
def test_v5_manifest_rejects_drift_from_formal_contract(
    tmp_path: Path,
    tamper: str,
) -> None:
    manifest = _manifest([TASK], "codex")
    if tamper == "policy":
        manifest["policy"]["attempts"] = 2
        manifest["comparison_contract"]["policy"] = copy.deepcopy(
            manifest["policy"]
        )
    elif tamper == "measurement":
        manifest["measurement"]["configured_repetitions_per_test_case"] = 99
        manifest["comparison_contract"]["measurement"] = copy.deepcopy(
            manifest["measurement"]
        )
    elif tamper == "docker":
        manifest["runtime"]["docker"]["image_id"] = "sha256:" + "0" * 64
    elif tamper == "gpu":
        manifest["runtime"]["gpu"]["target_gpu_model"] = "attacker-gpu"
    elif tamper == "isolation":
        manifest["runtime"]["isolation"]["policy"]["command_network"] = "allowed"
    else:
        for agent in (
            manifest["agent"],
            manifest["comparison_contract"]["codex"],
        ):
            agent["turn_policy"] = "attacker-controlled-turn-policy"
    if tamper in {"docker", "gpu", "isolation"}:
        manifest["comparison_contract"]["runtime"] = (
            compare_runs._project_comparison_runtime(manifest["runtime"])
        )
    _refresh_comparison_digest(manifest)

    _assert_manifest_rejected(tmp_path, manifest)


@pytest.mark.parametrize(
    ("agent_field", "table_field", "tampered_value"),
    [
        ("max_process_output_bytes", "max_process_output_bytes", -1),
        (
            "structured_stream_overflow_policy",
            "overflow_policy",
            "continue_unbounded",
        ),
    ],
)
@pytest.mark.parametrize("synchronize_table", [False, True])
def test_v5_manifest_rejects_transport_treatment_drift(
    tmp_path: Path,
    agent_field: str,
    table_field: str,
    tampered_value: object,
    synchronize_table: bool,
) -> None:
    manifest = _manifest([TASK], "codex")
    manifest["agent"][agent_field] = tampered_value
    if synchronize_table:
        manifest["comparison_contract"]["agent_transport_treatments"][
            "codex"
        ][table_field] = tampered_value
    _refresh_comparison_digest(manifest)

    _assert_manifest_rejected(tmp_path, manifest)


@pytest.mark.parametrize("tamper", ["git_policy", "clean_status"])
def test_v5_manifest_rejects_untrusted_repository_provenance(
    tmp_path: Path,
    tamper: str,
) -> None:
    manifest = _manifest([TASK], "codex")
    aka = manifest["repositories"]["agent_kernel_arena"]
    if tamper == "git_policy":
        aka.pop("git_evidence_policy_id")
    else:
        aka["status_sha256"] = "9" * 64
    _refresh_comparison_digest(manifest)

    _assert_manifest_rejected(tmp_path, manifest)


def test_v5_manifest_rejects_placeholder_runtime_isolation(
    tmp_path: Path,
) -> None:
    manifest = _manifest([TASK], "codex")
    isolation = manifest["runtime"]["isolation"]
    for key in (
        "outer_runtime",
        "bubblewrap",
        "codex_gpu_bubblewrap",
        "codex_cli",
        "codex_requirements",
        "attempt_probe",
        "codex_sandbox_probe",
    ):
        isolation[key] = {"attacker_placeholder": True}
    manifest["comparison_contract"]["runtime"]["isolation"] = copy.deepcopy(
        isolation
    )
    _refresh_comparison_digest(manifest)

    _assert_manifest_rejected(tmp_path, manifest)


def test_v5_manifest_rejects_incomplete_task_package_binding(
    tmp_path: Path,
) -> None:
    manifest = _manifest([TASK], "codex")
    minimal = {"task_index": 1, "task_name": TASK}
    manifest["configuration"]["tasks"] = [minimal]
    manifest["comparison_contract"]["tasks"] = [minimal]
    _refresh_comparison_digest(manifest)

    _assert_manifest_rejected(tmp_path, manifest)


def test_v5_manifest_rejects_forged_mount_receipt_file_digest(
    tmp_path: Path,
) -> None:
    manifest = _manifest([TASK], "codex")
    snapshot = manifest["runtime"]["aka_execution_snapshot"]
    snapshot["mount_receipt_file_sha256"] = "0" * 64
    _refresh_comparison_digest(manifest)

    _assert_manifest_rejected(tmp_path, manifest)


def test_v5_manifest_accepts_incomplete_supplementary_proc_audit() -> None:
    manifest = _manifest([TASK], "codex")
    exclusivity = manifest["runtime"]["gpu"]["exclusivity"]
    audit = exclusivity["supplementary_proc_audit"]
    audit.update(
        {
            "complete": False,
            "inaccessible_pid_count": 3,
            "inaccessible_pids_sha256": _digest([101, 102, 103]),
            "inaccessible_pids_sample": [101, 102, 103],
        }
    )
    material = {key: value for key, value in exclusivity.items() if key != "sha256"}
    exclusivity["sha256"] = _digest(material)

    assert compare_runs._v5_manifest_bindings_valid(
        manifest,
        manifest["comparison_contract"],
        manifest["configuration"]["tasks"],
    )


@pytest.mark.parametrize(
    "tamper",
    [
        "minimal_kfd_inventory",
        "failed_kfd_query",
        "invalid_kfd_digest",
        "invalid_kfd_count",
        "invalid_kfd_timestamp",
        "noncanonical_pool_id",
        "zero_unique_id",
        "malformed_render_path",
        "duplicate_render_path",
    ],
)
def test_v5_manifest_rejects_forged_gpu_inventory_identity(
    tmp_path: Path,
    tamper: str,
) -> None:
    manifest = _manifest([TASK], "codex")
    gpu = manifest["runtime"]["gpu"]
    exclusivity = gpu["exclusivity"]
    inventory = exclusivity["authoritative_kfd_process_inventory"]

    if tamper == "minimal_kfd_inventory":
        exclusivity["authoritative_kfd_process_inventory"] = {
            "verified_empty": True,
            "pids": [],
            "process_count": 0,
            "path": "/test/evidence/kfd-process-inventory.json",
        }
    elif tamper == "failed_kfd_query":
        inventory["query"]["fetch_status"] = 1
    elif tamper == "invalid_kfd_digest":
        inventory["document_sha256"] = "not-a-sha256"
    elif tamper == "invalid_kfd_count":
        inventory["query"]["fetch_capacity"] = 63
    elif tamper == "invalid_kfd_timestamp":
        inventory["observed_at_ns"] = 0
    elif tamper == "noncanonical_pool_id":
        gpu["ordered_host_gpu_ids"] = ["00"]
        gpu["devices"][0]["host_device_id"] = "00"
        gpu["task_mapping"][0]["assigned_host_gpu_id"] = "00"
    elif tamper == "zero_unique_id":
        zero = "0x0000000000000000"
        gpu["devices"][0]["unique_id"] = zero
        exclusivity["leases"][0]["unique_id"] = zero
    elif tamper == "malformed_render_path":
        malformed = "/dev/dri/renderD128-forged"
        gpu["devices"][0]["render_nodes"] = [malformed]
        exclusivity["protected_device_paths"] = [malformed, "/dev/kfd"]
    else:
        render_path = gpu["devices"][0]["render_nodes"][0]
        gpu["devices"][0]["render_nodes"] = [render_path, render_path]

    receipt_material = {
        key: value for key, value in exclusivity.items() if key != "sha256"
    }
    exclusivity["sha256"] = _digest(receipt_material)
    comparison_runtime = campaign.comparison_runtime_projection(manifest["runtime"])
    assert comparison_runtime is not None
    manifest["comparison_contract"]["runtime"] = comparison_runtime
    _refresh_comparison_digest(manifest)

    _assert_manifest_rejected(tmp_path, manifest)


def test_comparison_runtime_ignores_per_arm_mount_namespace_ids() -> None:
    manifest = _manifest([TASK], "codex")
    first = manifest["runtime"]
    second = copy.deepcopy(first)
    snapshot = second["aka_execution_snapshot"]
    receipt = snapshot["mount_receipt"]
    receipt["mount"].update(
        {"mount_id": 9876, "parent_id": 8765, "major_minor": "0:999"}
    )
    material = {key: value for key, value in receipt.items() if key != "sha256"}
    receipt["sha256"] = _digest(material)
    snapshot["mount_receipt_sha256"] = receipt["sha256"]
    snapshot["mount_receipt_file_sha256"] = hashlib.sha256(
        (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()

    assert campaign.comparison_runtime_projection(first) == (
        campaign.comparison_runtime_projection(second)
    )


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


@pytest.mark.parametrize("arm", ["apex", "codex"])
def test_formal_compare_rejects_resigned_receipt_campaign_binding(
    tmp_path: Path, arm: str
) -> None:
    run = _make_run(tmp_path, arm)
    receipt_path = (
        run
        / ".campaign_attempts"
        / campaign.campaign_task_path_component(TASK)
        / "attempt_01/session_receipt.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["campaign_binding"]["task_config_sha256"] = "f" * 64
    _refresh_receipt_record(run, 1, receipt)

    with pytest.raises(ValueError, match="formal canonical evidence"):
        compare_runs.load_run_data(run)


def test_formal_compare_rejects_receipt_transplanted_between_attempts(
    tmp_path: Path,
) -> None:
    run = _make_run(tmp_path, "codex")
    root = (
        run
        / ".campaign_attempts"
        / campaign.campaign_task_path_component(TASK)
    )
    transplanted = json.loads(
        (root / "attempt_01/session_receipt.json").read_text(encoding="utf-8")
    )
    _refresh_receipt_record(run, 3, transplanted)

    with pytest.raises(ValueError, match="formal canonical evidence"):
        compare_runs.load_run_data(run)


def test_formal_compare_rejects_resigned_receipt_gpu_drift(tmp_path: Path) -> None:
    run = _make_run(tmp_path, "apex")
    receipt_path = (
        run
        / ".campaign_attempts"
        / campaign.campaign_task_path_component(TASK)
        / "attempt_01/session_receipt.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["gpu"]["host_gpu_id"] = "1"
    _refresh_receipt_record(run, 1, receipt)

    with pytest.raises(ValueError, match="formal canonical evidence"):
        compare_runs.load_run_data(run)


def test_formal_compare_rejects_forged_slower_selected_attempt(
    tmp_path: Path,
) -> None:
    run = _make_run(tmp_path, "apex")
    campaign_path = (
        run
        / ".campaign_attempts"
        / campaign.campaign_task_path_component(TASK)
        / "task_campaign.yaml"
    )
    task_campaign = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    assert task_campaign["attempts"][0]["measured_rate_per_ms"] < (
        task_campaign["attempts"][1]["measured_rate_per_ms"]
    )
    task_campaign["selected_attempt"] = 1
    _rewrite_sealed_yaml(campaign_path, task_campaign)

    with pytest.raises(ValueError, match="formal canonical evidence"):
        compare_runs.load_run_data(run)


def test_formal_compare_rejects_duplicate_attempt_identity(tmp_path: Path) -> None:
    run = _make_run(tmp_path, "codex")
    campaign_path = (
        run
        / ".campaign_attempts"
        / campaign.campaign_task_path_component(TASK)
        / "task_campaign.yaml"
    )
    task_campaign = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    task_campaign["attempts"][2]["attempt"] = 2
    _rewrite_sealed_yaml(campaign_path, task_campaign)

    with pytest.raises(ValueError, match="formal canonical evidence"):
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
