import copy
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


def _formal_manifest(task_names: list[str], arm: str = "codex") -> dict:
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
    receipt = {
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
    return receipt


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
    result = _task_result(task_name, baseline_ms / optimized_ms)
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
    run: Path, task_name: str, speedup: float
) -> Path:
    timestamp = "20260807_000000"
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
            / f"{safe_name}_{timestamp}"
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
    _write_read_only_yaml(
        canonical / "baseline_perf.yaml",
        {"test_cases": [_performance_case(speedup)]},
    )
    _write_read_only_yaml(
        canonical / "optimized_perf.yaml",
        {"test_cases": [_performance_case(1.0)]},
    )
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
        "attempts": [
            {
                "attempt": 1,
                "session": "fresh-01",
                "attempt_completed": False,
                "workspace": None,
                "central_evaluator_report": None,
                "selection_eligible": False,
                "measured_rate_per_ms": 0.0,
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
    assert (
        f".campaign_attempts/{campaign.campaign_task_path_component(task_names[6])}"
        "/task_campaign.yaml"
    ) in report

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
        f".campaign_attempts/{campaign.campaign_task_path_component(task_name)}"
        "/task_campaign.yaml"
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


def test_receiptless_single_attempt_fixture_cannot_count_as_formal_success(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run_20260807_000000_formal"
    run.mkdir()
    task_name = "triton2triton/receiptless"
    _write_read_only_yaml(run / "campaign_manifest.yaml", _formal_manifest([task_name]))
    canonical = _write_canonical_workspace(run, task_name, 2.0)
    campaign_path = (
        run
        / ".campaign_attempts"
        / campaign.campaign_task_path_component(task_name)
        / "task_campaign.yaml"
    )
    task_campaign = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    only = task_campaign["attempts"][0]
    for key in (
        "session_receipt",
        "session_receipt_sha256",
        "session_receipt_binding",
        "session_receipt_binding_sha256",
        "session_succeeded",
    ):
        only.pop(key, None)
    task_campaign["attempts"] = [only]
    task_campaign["policy"]["attempts"] = 1
    task_campaign["selected_attempt"] = 1
    campaign_path.chmod(0o644)
    _write_read_only_yaml(campaign_path, task_campaign)

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
