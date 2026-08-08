# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""
Standalone script to compare two AgentKernelArena runs.

Usage:
    python3 src/tools/compare_runs.py run1_path run2_path
    python3 src/tools/compare_runs.py workspace_MI300_cursor/run_20260714_120000_baseline workspace_MI300_cursor/run_20260714_140000_treatment
"""

import json
import argparse
import hashlib
import math
import os
import re
import stat
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Dict, Any, Optional

import yaml

try:
    from src import postprocessing
    from src.aka_runtime import (
        BACKEND_CLOSURE_SCHEMA,
        EXECUTION_MANIFEST_SCHEMA,
        GIT_EVIDENCE_POLICY,
        IMMUTABLE_MOUNT_POLICY,
        IMMUTABLE_MOUNT_RECEIPT_SCHEMA,
    )
    from src.campaign_isolation import (
        APEX_RUNTIME_MOUNT_POLICY,
        APEX_RUNTIME_MOUNT_SCHEMA,
        ATTEMPT_MOUNT_RECEIPT_SCHEMA,
    )
    from src.campaign import (
        FORMAL_AGENT_TRANSPORT_TREATMENTS,
        campaign_task_path_component,
        comparison_runtime_projection as _campaign_runtime_projection,
    )
    from src.gpu_exclusivity import POLICY as GPU_EXCLUSIVITY_POLICY
    from src.gpu_exclusivity import SCHEMA as GPU_EXCLUSIVITY_SCHEMA
    from src.score import resolve_speedup_ratio, task_result_scoring
except (ModuleNotFoundError, ImportError):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src import postprocessing
    from src.aka_runtime import (
        BACKEND_CLOSURE_SCHEMA,
        EXECUTION_MANIFEST_SCHEMA,
        GIT_EVIDENCE_POLICY,
        IMMUTABLE_MOUNT_POLICY,
        IMMUTABLE_MOUNT_RECEIPT_SCHEMA,
    )
    from src.campaign_isolation import (
        APEX_RUNTIME_MOUNT_POLICY,
        APEX_RUNTIME_MOUNT_SCHEMA,
        ATTEMPT_MOUNT_RECEIPT_SCHEMA,
    )
    from src.campaign import (
        FORMAL_AGENT_TRANSPORT_TREATMENTS,
        campaign_task_path_component,
        comparison_runtime_projection as _campaign_runtime_projection,
    )
    from src.gpu_exclusivity import POLICY as GPU_EXCLUSIVITY_POLICY
    from src.gpu_exclusivity import SCHEMA as GPU_EXCLUSIVITY_SCHEMA
    from src.score import resolve_speedup_ratio, task_result_scoring


_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMPARISON_SCHEMA = "aka.apex-vs-codex-comparison-contract/v5"
_CANDIDATE_PERSISTENCE_POLICY = "structured_agent_turn_checkpoint_v2"
_BOUNDARY_QUIESCENCE_POLICY = "sigstop_process_group_snapshot_v1"
_ATTEMPT_CONTAINMENT_POLICY = "private_pid_namespace_init_pidfd_v1"
_AGENT_PROCESS_CONTAINMENT_POLICY = "private_pid_namespace_init_pidfd_v1"
_OBJECTIVE_POLICY = "aka.task-package-objective-and-protected-harness/v1"
_PROMPT_POLICY = "aka.shared-objective-backend-native-context-receipted/v1"
_FORMAL_LIVE_EXECUTION = {
    "mode": "live_formal_scoring",
    "comparison_generation": 5,
    "historical_compatibility": False,
    "policy_id": "aka.live-formal-v5-only/v1",
}
_FORMAL_LIVE_EXECUTION_SHA256 = hashlib.sha256(
    json.dumps(
        _FORMAL_LIVE_EXECUTION, sort_keys=True, separators=(",", ":")
    ).encode()
).hexdigest()
_CODEX_IDENTITY_FIELDS = (
    "attempt_timeout_seconds",
    "backend",
    "codex_binary_sha256",
    "codex_version",
    "effort",
    "inner_max_iterations",
    "isolation",
    "max_turns",
    "model",
    "permission_mode",
    "structured_stream_output_limit_bytes",
    "turn_policy",
    "agent_process_containment_policy_id",
    "attempt_containment_policy_id",
    "backend_runtime_closure_schema",
    "backend_runtime_closure_sha256",
    "backend_runtime_closure",
)
_APEX_RECEIPT_SCHEMA = "agentkernelarena.apex-attempt-receipt/v5"
_CODEX_RECEIPT_SCHEMA = "agentkernelarena.codex-attempt-receipt/v4"
_CLEAN_STATUS_SHA256 = hashlib.sha256(b"").hexdigest()
_MOUNT_SEALS = ["F_SEAL_WRITE", "F_SEAL_SHRINK", "F_SEAL_GROW", "F_SEAL_SEAL"]
_RUNTIME_ISOLATION_SCHEMA = "aka.runtime-isolation-receipt/v5"
_RUNTIME_ISOLATION_POLICY = {
    "docker_user": "non_root",
    "docker_capabilities": "drop_all",
    "docker_no_new_privileges": True,
    "docker_apparmor": "unconfined_for_rootless_userns",
    "docker_seccomp": "unconfined_for_rootless_userns",
    "docker_systempaths": "unconfined_for_private_attempt_procfs",
    "docker_masked_paths_rebuilt": [
        "/proc/acpi", "/proc/asound", "/proc/scsi", "/sys/devices/virtual/powercap",
        "/sys/firmware", "/proc/interrupts", "/proc/kcore", "/proc/keys",
        "/proc/latency_stats", "/proc/sched_debug", "/proc/timer_list",
        "/proc/timer_stats",
    ],
    "docker_readonly_paths_rebuilt": [
        "/proc/bus", "/proc/fs", "/proc/irq", "/proc/sys", "/proc/sysrq-trigger",
    ],
    "docker_pid_namespace": "private_default",
    "attempt_mount_namespace": "bubblewrap",
    "attempt_pid_namespace": "private_per_attempt_with_bwrap_reaper_pid1",
    "attempt_ipc_namespace": "unshared",
    "attempt_proc": "private_procfs_for_attempt_pid_namespace",
    "direct_agent_proc": "aka_outer_private_attempt_procfs",
    "apex_outer_proc": "trusted_orchestrator_inherited_worker_procfs_nested_userns_writable",
    "apex_backend_proc": "apex_inner_private_attempt_procfs_required",
    "process_lifetime_boundary": "namespace_init_pidfd_v1",
    "proc_escape_guard": "outer_process_absent_from_private_procfs_v1",
    "command_sandbox": "codex_managed_permission_profile_bwrap",
    "command_pid_namespace": "nested_codex_unshared_inside_private_attempt_pidns_v1",
    "command_network": "managed_profile_denied_live_probe_v1",
    "command_gpu_access": "sealed_memfd_immutable_path_bwrap_and_single_gpu_probe_v1",
    "credential_read": "denied_by_managed_permission_profile",
}
_GPU_MODEL_ARCH = {
    "MI300": "gfx942",
    "MI300X": "gfx942",
    "MI325": "gfx942",
    "MI325X": "gfx942",
    "MI355X": "gfx950",
    "RDNA4": "gfx1201",
}
_CANONICAL_DECIMAL = re.compile(r"0|[1-9][0-9]*")
_GPU_UNIQUE_ID = re.compile(r"0x[0-9a-f]{16}")
_RENDER_NODE_PATH = re.compile(r"/dev/dri/renderD[1-9][0-9]*")
_KFD_PROCESS_SOURCE = "librocm_smi64.rsmi_compute_process_info_get"
_KFD_FETCH_HEADROOM = 64
_KFD_MAX_PROCESS_RECORDS = 262_144


def _read_regular_file_no_follow(path: Path, *, immutable: bool = False) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"unsafe report evidence file: {path}") from error
    try:
        opened = os.fstat(descriptor)
        lexical = path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (immutable and opened.st_mode & 0o222)
            or opened.st_dev != lexical.st_dev
            or opened.st_ino != lexical.st_ino
        ):
            raise ValueError(f"unsafe report evidence file: {path}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _gpu_receipt_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _absolute_path_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and PurePosixPath(value).is_absolute()


def _file_identity_valid(value: Any, *, launcher: bool = False) -> bool:
    expected = {"resolved_path", "mode", "size", "sha256"}
    if launcher:
        expected |= {"requested_path", "symlink_chain"}
    if not isinstance(value, dict) or set(value) != expected:
        return False
    if not (
        _absolute_path_string(value.get("resolved_path"))
        and isinstance(value.get("mode"), int)
        and 0 < value["mode"] <= 0o777
        and isinstance(value.get("size"), int)
        and value["size"] > 0
        and _SHA256.fullmatch(str(value.get("sha256") or ""))
    ):
        return False
    if not launcher:
        return True
    chain = value.get("symlink_chain")
    return bool(
        _absolute_path_string(value.get("requested_path"))
        and isinstance(chain, list)
        and all(
            isinstance(link, dict)
            and set(link) == {"path", "target"}
            and _absolute_path_string(link.get("path"))
            and isinstance(link.get("target"), str)
            and bool(link["target"])
            for link in chain
        )
    )


def _component_files_valid(files: Any) -> bool:
    if not isinstance(files, list) or not files:
        return False
    paths: set[str] = set()
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "mode", "size", "sha256"}:
            return False
        path = item.get("path")
        parts = PurePosixPath(path).parts if isinstance(path, str) else ()
        if (
            not path
            or PurePosixPath(path).is_absolute()
            or ".." in parts
            or path in paths
            or not isinstance(item.get("mode"), int)
            or not 0 < item["mode"] <= 0o777
            or not isinstance(item.get("size"), int)
            or item["size"] < 0
            or not _SHA256.fullmatch(str(item.get("sha256") or ""))
        ):
            return False
        paths.add(path)
    return True


def _backend_components_valid(components: Any) -> bool:
    if not isinstance(components, list):
        return False
    roots: set[str] = set()
    for component in components:
        if not isinstance(component, dict) or set(component) != {
            "kind", "root", "files", "files_sha256"
        }:
            return False
        root = component.get("root")
        files = component.get("files")
        if (
            component.get("kind") != "node_package"
            or not _absolute_path_string(root)
            or root in roots
            or not _component_files_valid(files)
            or component.get("files_sha256")
            != _sha256_bytes(_canonical_json(files).encode())
        ):
            return False
        roots.add(root)
    return True


def _launcher_component_binding_valid(closure: Dict[str, Any]) -> bool:
    launcher = closure.get("launcher")
    components = closure.get("components")
    if not isinstance(launcher, dict) or not isinstance(components, list):
        return False
    resolved = PurePosixPath(str(launcher.get("resolved_path") or ""))
    matches: list[dict[str, Any]] = []
    for component in components:
        if not isinstance(component, dict):
            continue
        root = PurePosixPath(str(component.get("root") or ""))
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError:
            continue
        matches.extend(
            item
            for item in component.get("files", [])
            if isinstance(item, dict) and item.get("path") == relative
        )
    return bool(
        len(matches) == 1
        and all(
            matches[0].get(key) == launcher.get(key)
            for key in ("mode", "size", "sha256")
        )
    )


def _backend_closure_binding_valid(agent: Any) -> bool:
    if not isinstance(agent, dict):
        return False
    closure = agent.get("backend_runtime_closure")
    digest = agent.get("backend_runtime_closure_sha256")
    if not isinstance(closure, dict) or set(closure) != {
        "schema", "backend", "launcher", "interpreter", "components", "closure_sha256"
    }:
        return False
    material = dict(closure)
    embedded_digest = material.pop("closure_sha256", None)
    interpreter = closure.get("interpreter")
    launcher = closure.get("launcher")
    return bool(
        agent.get("backend") == "codex"
        and closure.get("backend") == agent.get("backend")
        and agent.get("backend_runtime_closure_schema") == BACKEND_CLOSURE_SCHEMA
        and closure.get("schema") == BACKEND_CLOSURE_SCHEMA
        and isinstance(digest, str)
        and _SHA256.fullmatch(digest)
        and embedded_digest == digest
        and _sha256_bytes(_canonical_json(material).encode()) == digest
        and _file_identity_valid(launcher, launcher=True)
        and (interpreter is None or _file_identity_valid(interpreter))
        and _backend_components_valid(closure.get("components"))
        and _launcher_component_binding_valid(closure)
        and agent.get("codex_binary_sha256") == launcher.get("sha256")
    )


def _expected_apex_treatment(repositories: Any) -> Dict[str, Any] | None:
    apex = repositories.get("apex") if isinstance(repositories, dict) else None
    runtime_digest = apex.get("runtime_manifest_sha256") if isinstance(apex, dict) else None
    if not isinstance(runtime_digest, str) or not _SHA256.fullmatch(runtime_digest):
        return None
    return {
        "template": "apex",
        "session_receipt_schema": _APEX_RECEIPT_SCHEMA,
        "apex_runtime_mount_policy_id": APEX_RUNTIME_MOUNT_POLICY,
        "attempt_mount_receipt_schema": ATTEMPT_MOUNT_RECEIPT_SCHEMA,
        "apex_runtime_mount_schema": APEX_RUNTIME_MOUNT_SCHEMA,
        "runtime_manifest_sha256": runtime_digest,
    }


def _repository_bindings_valid(repositories: Any) -> bool:
    if not isinstance(repositories, dict):
        return False
    aka = repositories.get("agent_kernel_arena")
    apex = repositories.get("apex")
    return bool(
        isinstance(aka, dict)
        and re.fullmatch(r"[0-9a-f]{40}", str(aka.get("commit") or ""))
        and re.fullmatch(r"[0-9a-f]{40}", str(aka.get("tree") or ""))
        and aka.get("dirty") is False
        and aka.get("status_sha256") == _CLEAN_STATUS_SHA256
        and aka.get("execution_manifest_schema") == EXECUTION_MANIFEST_SCHEMA
        and _SHA256.fullmatch(str(aka.get("execution_manifest_sha256") or ""))
        and aka.get("git_evidence_policy_id") == GIT_EVIDENCE_POLICY
        and isinstance(apex, dict)
        and re.fullmatch(r"[0-9a-f]{40}", str(apex.get("commit") or ""))
        and apex.get("dirty") is False
        and apex.get("status_sha256") == _CLEAN_STATUS_SHA256
        and _SHA256.fullmatch(str(apex.get("runtime_manifest_sha256") or ""))
    )


def _static_mount_record_valid(mount: Any, expected_root: Any) -> bool:
    expected_keys = {
        "path", "mount_id", "parent_id", "major_minor", "root", "mount_point",
        "mount_options", "filesystem_type", "source", "super_options", "read_only",
        "nested_mounts",
    }
    return bool(
        isinstance(mount, dict)
        and set(mount) == expected_keys
        and _absolute_path_string(expected_root)
        and mount.get("path") == expected_root
        and mount.get("mount_point") == expected_root
        and mount.get("root") == "/"
        and isinstance(mount.get("mount_id"), int)
        and mount["mount_id"] > 0
        and isinstance(mount.get("parent_id"), int)
        and mount["parent_id"] > 0
        and re.fullmatch(r"[0-9]+:[0-9]+", str(mount.get("major_minor") or ""))
        and mount.get("filesystem_type")
        in {"squashfs", "fuse.squashfuse", "fuse.squashfuse_ll"}
        and isinstance(mount.get("source"), str)
        and bool(mount["source"])
        and isinstance(mount.get("mount_options"), list)
        and "ro" in mount["mount_options"]
        and isinstance(mount.get("super_options"), list)
        and mount.get("read_only") is True
        and mount.get("nested_mounts") == []
    )


def _static_mount_receipt_valid(
    receipt: Any, expected_manifest_sha256: Any, expected_root: Any
) -> bool:
    if not isinstance(receipt, dict) or set(receipt) != {
        "schema", "policy_id", "manifest_sha256", "image_sha256", "memfd_seals",
        "mount", "sha256",
    }:
        return False
    material = dict(receipt)
    observed = material.pop("sha256", None)
    return bool(
        receipt.get("schema") == IMMUTABLE_MOUNT_RECEIPT_SCHEMA
        and receipt.get("policy_id") == IMMUTABLE_MOUNT_POLICY
        and receipt.get("manifest_sha256") == expected_manifest_sha256
        and _SHA256.fullmatch(str(receipt.get("image_sha256") or ""))
        and receipt.get("memfd_seals") == _MOUNT_SEALS
        and _static_mount_record_valid(receipt.get("mount"), expected_root)
        and isinstance(observed, str)
        and _SHA256.fullmatch(observed)
        and observed == _sha256_bytes(_canonical_json(material).encode())
    )


def _runtime_snapshot_binding_valid(
    repositories: Any, manifest_runtime: Any, comparison_runtime: Any
) -> bool:
    if not all(
        isinstance(value, dict)
        for value in (repositories, manifest_runtime, comparison_runtime)
    ):
        return False
    aka = repositories.get("agent_kernel_arena")
    snapshot = manifest_runtime.get("aka_execution_snapshot")
    comparison_snapshot = comparison_runtime.get("aka_execution_snapshot")
    projected_runtime = _project_comparison_runtime(manifest_runtime)
    projected_snapshot = (
        projected_runtime.get("aka_execution_snapshot")
        if isinstance(projected_runtime, dict)
        else None
    )
    if not isinstance(aka, dict) or not isinstance(snapshot, dict) or set(snapshot) != {
        "schema", "root", "manifest_path", "manifest_file_sha256", "manifest_sha256",
        "mount_receipt_path", "mount_receipt_file_sha256", "mount_receipt_sha256",
        "mount_receipt_schema", "mount_receipt",
    }:
        return False
    receipt = snapshot.get("mount_receipt")
    expected_digest = aka.get("execution_manifest_sha256")
    receipt_file = (
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        if isinstance(receipt, dict)
        else ""
    )
    return bool(
        comparison_snapshot == projected_snapshot
        and snapshot.get("schema") == "aka.execution-snapshot-runtime/v1"
        and _absolute_path_string(snapshot.get("root"))
        and _absolute_path_string(snapshot.get("manifest_path"))
        and _absolute_path_string(snapshot.get("mount_receipt_path"))
        and _SHA256.fullmatch(str(snapshot.get("manifest_file_sha256") or ""))
        and snapshot.get("mount_receipt_file_sha256")
        == _sha256_bytes(receipt_file.encode())
        and snapshot.get("manifest_sha256") == expected_digest
        and snapshot.get("mount_receipt_schema") == IMMUTABLE_MOUNT_RECEIPT_SCHEMA
        and _static_mount_receipt_valid(receipt, expected_digest, snapshot.get("root"))
        and snapshot.get("mount_receipt_sha256") == receipt.get("sha256")
    )


def _evaluator_binding_valid(repositories: Any, evaluator: Any) -> bool:
    aka = repositories.get("agent_kernel_arena") if isinstance(repositories, dict) else None
    return bool(
        isinstance(aka, dict)
        and isinstance(evaluator, dict)
        and evaluator.get("schema") == "aka.evaluator-source-binding/v2"
        and evaluator.get("coverage") == "all_committed_files"
        and evaluator.get("execution_manifest_schema") == EXECUTION_MANIFEST_SCHEMA
        and evaluator.get("execution_manifest_sha256")
        == aka.get("execution_manifest_sha256")
        and evaluator.get("commit") == aka.get("commit")
        and evaluator.get("tree") == aka.get("tree")
    )


def _campaign_policy_valid(manifest: Dict[str, Any], comparison: Dict[str, Any]) -> bool:
    policy = manifest.get("policy")
    if not isinstance(policy, dict) or comparison.get("policy") != policy or set(policy) != {
        "comparison", "attempts", "attempt_timeout_seconds",
        "apex_internal_allowance_seconds", "task_timeout_seconds",
        "evaluator_allowance_seconds", "selection_policy", "workspace_policy",
        "gpu_policy", "require_clean_checkouts",
    }:
        return False
    evaluator_allowance = policy.get("evaluator_allowance_seconds")
    task_timeout = policy.get("task_timeout_seconds")
    return bool(
        policy.get("comparison") == "apex_vs_codex"
        and policy.get("attempts") == 3
        and policy.get("attempt_timeout_seconds") == 3600
        and policy.get("apex_internal_allowance_seconds") == 3600
        and isinstance(evaluator_allowance, int)
        and evaluator_allowance >= 0
        and isinstance(task_timeout, int)
        and task_timeout >= 3 * (3600 + 3600) + evaluator_allowance
        and policy.get("selection_policy") == "correctness_then_measured_rate_v1"
        and policy.get("workspace_policy") == "fresh_per_attempt"
        and policy.get("gpu_policy") == "deterministic_task_gpu_v1"
        and policy.get("require_clean_checkouts") is True
    )


def _measurement_binding_valid(
    manifest: Dict[str, Any], comparison: Dict[str, Any]
) -> bool:
    measurement = manifest.get("measurement")
    return bool(
        isinstance(measurement, dict)
        and comparison.get("measurement") == measurement
        and measurement == {
            "contract": "aka_native_100_repetition_external_score",
            "owner": "AgentKernelArena centralized evaluator",
            "configured_repetitions_per_test_case": 100,
            "is_apex_kernel_measurement_v1": False,
            "is_apex_canonical_300_sample_grade": False,
        }
    )


def _task_contracts_valid(tasks: Any) -> bool:
    if not isinstance(tasks, list) or not tasks:
        return False
    names: set[str] = set()
    for expected_index, task in enumerate(tasks, 1):
        if not isinstance(task, dict) or set(task) != {
            "task_index", "task_name", "config_path", "config_sha256",
            "package_files_sha256", "package_manifest_sha256",
        }:
            return False
        name = task.get("task_name")
        config_path = task.get("config_path")
        files = task.get("package_files_sha256")
        if (
            task.get("task_index") != expected_index
            or not isinstance(name, str)
            or not name
            or name in names
            or not _absolute_path_string(config_path)
            or not _SHA256.fullmatch(str(task.get("config_sha256") or ""))
            or not isinstance(files, dict)
            or not files
            or any(
                not isinstance(path, str)
                or not path
                or PurePosixPath(path).is_absolute()
                or ".." in PurePosixPath(path).parts
                or not _SHA256.fullmatch(str(digest or ""))
                for path, digest in files.items()
            )
            or files.get(PurePosixPath(config_path).name) != task.get("config_sha256")
            or task.get("package_manifest_sha256")
            != _sha256_bytes(_canonical_json(files).encode())
        ):
            return False
        names.add(name)
    return True


def _agent_isolation_valid(isolation: Any) -> bool:
    return isolation == {
        "approval": "never_via_strict_config",
        "execpolicy_rules": "ignored",
        "project_instructions": "backend_default_may_load",
        "sandbox": "workspace-write",
        "session": "ephemeral",
        "user_config": "ignored",
        "mount_scope": "attempt_only_bubblewrap",
        "attempt_containment_policy_id": _ATTEMPT_CONTAINMENT_POLICY,
    }


def _formal_agent_valid(agent: Any) -> bool:
    return bool(
        isinstance(agent, dict)
        and agent.get("backend") == "codex"
        and agent.get("model") == "gpt-5.5"
        and agent.get("effort") == "xhigh"
        and agent.get("permission_mode") == "workspace_write_isolated"
        and agent.get("inner_max_iterations") == 1
        and agent.get("attempt_timeout_seconds") == 3600
        and agent.get("max_turns") == 50
        and agent.get("turn_policy") == _CANDIDATE_PERSISTENCE_POLICY
        and agent.get("agent_process_containment_policy_id")
        == _AGENT_PROCESS_CONTAINMENT_POLICY
        and agent.get("attempt_containment_policy_id")
        == _ATTEMPT_CONTAINMENT_POLICY
        and agent.get("structured_stream_output_limit_bytes") == 16 * 1024 * 1024
        and isinstance(agent.get("codex_version"), str)
        and bool(agent["codex_version"])
        and _agent_isolation_valid(agent.get("isolation"))
        and _backend_closure_binding_valid(agent)
    )


def _agent_transport_treatments_valid(
    agent: Any, comparison: Dict[str, Any]
) -> bool:
    treatments = comparison.get("agent_transport_treatments")
    treatment = (
        treatments.get(agent.get("template"))
        if isinstance(agent, dict) and isinstance(treatments, dict)
        else None
    )
    return bool(
        treatments == FORMAL_AGENT_TRANSPORT_TREATMENTS
        and isinstance(treatment, dict)
        and agent.get("max_process_output_bytes")
        == treatment.get("max_process_output_bytes")
        and agent.get("structured_stream_output_limit_bytes")
        == treatment.get("structured_stream_output_limit_bytes")
        and agent.get("structured_stream_overflow_policy")
        == treatment.get("overflow_policy")
    )


def _docker_runtime_valid(docker: Any) -> bool:
    digests = docker.get("repo_digests") if isinstance(docker, dict) else None
    return bool(
        isinstance(docker, dict)
        and set(docker) == {"reference", "image_id", "repo_digests"}
        and isinstance(docker.get("reference"), str)
        and bool(docker["reference"])
        and re.fullmatch(r"sha256:[0-9a-f]{64}", str(docker.get("image_id") or ""))
        and docker.get("image_id") != "sha256:" + "0" * 64
        and isinstance(digests, list)
        and bool(digests)
        and digests == sorted(set(digests))
        and all(
            isinstance(value, str)
            and re.fullmatch(r"[^@\s]+@sha256:[0-9a-f]{64}", value)
            and not value.endswith("@sha256:" + "0" * 64)
            for value in digests
        )
    )


def _all_true_evidence(value: Any, expected_keys: set[str]) -> bool:
    return bool(
        isinstance(value, dict)
        and set(value) == expected_keys
        and all(value.get(key) is True for key in expected_keys)
    )


def _outer_runtime_valid(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "effective_uid", "effective_gid", "supplementary_gids", "capabilities",
        "no_new_privileges", "seccomp_mode", "seccomp_filters", "apparmor_profile",
        "yama_ptrace_scope",
    }:
        return False
    return bool(
        isinstance(value.get("effective_uid"), int)
        and value["effective_uid"] > 0
        and isinstance(value.get("effective_gid"), int)
        and value["effective_gid"] >= 0
        and isinstance(value.get("supplementary_gids"), list)
        and value["supplementary_gids"] == sorted(set(value["supplementary_gids"]))
        and all(isinstance(group, int) and group >= 0 for group in value["supplementary_gids"])
        and value.get("capabilities")
        == {"CapInh": 0, "CapPrm": 0, "CapEff": 0, "CapBnd": 0, "CapAmb": 0}
        and value.get("no_new_privileges") is True
        and value.get("seccomp_mode") == 0
        and value.get("seccomp_filters") == 0
        and value.get("apparmor_profile") == "unconfined"
        and isinstance(value.get("yama_ptrace_scope"), int)
        and value["yama_ptrace_scope"] >= 1
    )


def _runtime_isolation_valid(receipt: Any, agent: Any) -> bool:
    if not isinstance(receipt, dict) or set(receipt) != {
        "schema", "policy", "outer_runtime", "bubblewrap", "codex_gpu_bubblewrap",
        "codex_cli", "codex_requirements", "attempt_probe", "codex_sandbox_probe",
    }:
        return False
    policy = receipt.get("policy")
    bubblewrap = receipt.get("bubblewrap")
    gpu_bubblewrap = receipt.get("codex_gpu_bubblewrap")
    codex_cli = receipt.get("codex_cli")
    requirements = receipt.get("codex_requirements")
    closure = agent.get("backend_runtime_closure") if isinstance(agent, dict) else None
    launcher = closure.get("launcher") if isinstance(closure, dict) else None
    attempt_keys = {
        "campaign_data_hidden", "outer_pid_namespace_absent_from_private_proc",
        "parent_root_sentinel_unreachable", "parent_fd_sentinel_unreachable",
        "proc_mount_read_write", "pid_namespace_unshared", "ipc_namespace_unshared",
        "private_shm", "docker_system_paths_remasked",
        "private_proc_control_writes_blocked", "no_new_privileges",
        "effective_capabilities_zero", "bounding_capabilities_zero",
        "all_capability_sets_zero", "seccomp_disabled",
    }
    sandbox_keys = {
        "workspace_write_enforced", "credential_read_denied", "command_network_denied",
        "command_not_in_worker_pid_namespace", "pid1_root_alias_credential_blocked",
        "pid1_environ_blocked", "pid1_mem_blocked", "pinned_gpu_bwrap_active",
        "gpu_bwrap_directory_immutable", "gpu_bwrap_path_immutable",
        "assigned_gpu_devices_visible", "assigned_gpu_devices_writable",
        "single_gpu_runtime_visible", "gpu_compute_probe_passed",
    }
    return bool(
        receipt.get("schema") == _RUNTIME_ISOLATION_SCHEMA
        and policy == _RUNTIME_ISOLATION_POLICY
        and _outer_runtime_valid(receipt.get("outer_runtime"))
        and bubblewrap == {
            "resolved_path": "/usr/bin/bwrap",
            "sha256": "d78807229d616606e339c5988392b9e0ab4a6a6998fa51e4590837f426a12fca",
            "version": "bubblewrap 0.6.1",
            "execution_transport": "sealed_memfd_proc_self_fd",
        }
        and isinstance(gpu_bubblewrap, dict)
        and set(gpu_bubblewrap) == {
            "resolved_path", "sha256", "size_bytes", "interpreter", "real_bwrap",
            "real_bwrap_sha256", "sandbox_mounted_path", "mount_transport",
            "device_policy",
        }
        and _absolute_path_string(gpu_bubblewrap.get("resolved_path"))
        and gpu_bubblewrap.get("sha256")
        == "9271bd346d1ea5f878c8f345537e8464a56156b82f956942b66b82feb61791ef"
        and gpu_bubblewrap.get("size_bytes") == 2381
        and gpu_bubblewrap.get("interpreter") == "/usr/bin/python3 -I"
        and gpu_bubblewrap.get("real_bwrap") == "/usr/bin/bwrap"
        and gpu_bubblewrap.get("real_bwrap_sha256") == bubblewrap.get("sha256")
        and gpu_bubblewrap.get("sandbox_mounted_path")
        == "/tmp/aka-codex-gpu-bwrap/bwrap"
        and gpu_bubblewrap.get("mount_transport")
        == "sealed_memfd_ro_bind_data_under_remounted_ro_tmpfs"
        and gpu_bubblewrap.get("device_policy")
        == "docker_visible_kfd_and_render_nodes_only"
        and isinstance(codex_cli, dict)
        and set(codex_cli) == {"resolved_path", "sha256", "version"}
        and isinstance(launcher, dict)
        and codex_cli.get("resolved_path") == launcher.get("resolved_path")
        and codex_cli.get("sha256") == agent.get("codex_binary_sha256")
        and codex_cli.get("version") == agent.get("codex_version")
        and requirements == {
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
            "device_access": "sealed_pinned_immutable_path_bwrap_with_docker_device_boundary",
            "hooks": "disabled",
        }
        and _all_true_evidence(receipt.get("attempt_probe"), attempt_keys)
        and _all_true_evidence(receipt.get("codex_sandbox_probe"), sandbox_keys)
    )


def _authoritative_kfd_inventory_valid(inventory: Any) -> bool:
    expected_keys = {
        "source", "artifact_sha256", "document_sha256", "library",
        "observed_at_ns", "query", "pids", "process_count",
        "verified_empty", "path",
    }
    query_keys = {
        "init_status", "count_status", "count_hint", "fetch_status",
        "fetch_capacity", "fetched_count", "shutdown_status",
    }
    if not isinstance(inventory, dict) or set(inventory) != expected_keys:
        return False
    library = inventory.get("library")
    query = inventory.get("query")
    if (
        not isinstance(library, dict)
        or set(library) != {"path", "sha256"}
        or not _absolute_path_string(library.get("path"))
        or not isinstance(library.get("sha256"), str)
        or not _SHA256.fullmatch(library["sha256"])
        or not isinstance(query, dict)
        or set(query) != query_keys
        or any(
            isinstance(query.get(key), bool)
            or not isinstance(query.get(key), int)
            for key in query_keys
        )
    ):
        return False
    count_hint = query["count_hint"]
    return bool(
        inventory.get("source") == _KFD_PROCESS_SOURCE
        and isinstance(inventory.get("artifact_sha256"), str)
        and _SHA256.fullmatch(inventory["artifact_sha256"])
        and isinstance(inventory.get("document_sha256"), str)
        and _SHA256.fullmatch(inventory["document_sha256"])
        and isinstance(inventory.get("observed_at_ns"), int)
        and not isinstance(inventory["observed_at_ns"], bool)
        and inventory["observed_at_ns"] > 0
        and query["init_status"] == 0
        and query["count_status"] == 0
        and 0 <= count_hint <= _KFD_MAX_PROCESS_RECORDS
        and query["fetch_status"] == 0
        and query["fetch_capacity"]
        == max(count_hint + _KFD_FETCH_HEADROOM, _KFD_FETCH_HEADROOM)
        and query["fetched_count"] == 0
        and query["shutdown_status"] == 0
        and inventory.get("pids") == []
        and inventory.get("process_count") == 0
        and inventory.get("verified_empty") is True
        and _absolute_path_string(inventory.get("path"))
    )


def _gpu_exclusivity_valid(exclusivity: Any, gpu: Dict[str, Any]) -> bool:
    if not isinstance(exclusivity, dict):
        return False
    material = dict(exclusivity)
    observed = material.pop("sha256", None)
    leases = exclusivity.get("leases")
    devices = gpu.get("devices")
    unique_ids = {
        device.get("unique_id") for device in devices if isinstance(device, dict)
    } if isinstance(devices, list) else set()
    expected_paths = sorted({
        str(gpu.get("kfd_device", {}).get("path") or ""),
        *(
            str(path)
            for device in devices or []
            if isinstance(device, dict)
            for path in device.get("render_nodes", [])
        ),
    })
    inventory = exclusivity.get("authoritative_kfd_process_inventory")
    audit = exclusivity.get("supplementary_proc_audit")
    return bool(
        exclusivity.get("schema") == GPU_EXCLUSIVITY_SCHEMA
        and exclusivity.get("policy") == GPU_EXCLUSIVITY_POLICY
        and exclusivity.get("gpu_boundary_plan_sha256")
        == gpu.get("gpu_boundary_plan_sha256")
        and exclusivity.get("exclusivity_verified") is True
        and exclusivity.get("foreign_device_owners") == []
        and isinstance(leases, list)
        and bool(leases)
        and all(isinstance(lease, dict) for lease in leases)
        and len(leases) == len(unique_ids)
        and {lease.get("unique_id") for lease in leases} == unique_ids
        and all(
            set(lease) == {"unique_id", "lock_path"}
            and _absolute_path_string(lease.get("lock_path"))
            for lease in leases
        )
        and exclusivity.get("protected_device_paths") == expected_paths
        and all(path.startswith("/dev/") for path in expected_paths)
        and _authoritative_kfd_inventory_valid(inventory)
        and isinstance(audit, dict)
        and audit.get("owners") == []
        and isinstance(audit.get("complete"), bool)
        and isinstance(audit.get("inaccessible_pid_count"), int)
        and audit["inaccessible_pid_count"] >= 0
        and _SHA256.fullmatch(str(audit.get("inaccessible_pids_sha256") or ""))
        and isinstance(audit.get("inaccessible_pids_sample"), list)
        and all(isinstance(pid, int) and pid > 0 for pid in audit["inaccessible_pids_sample"])
        and len(audit["inaccessible_pids_sample"])
        <= min(64, audit["inaccessible_pid_count"])
        and (
            (audit["complete"] is True and audit["inaccessible_pid_count"] == 0)
            or (audit["complete"] is False and audit["inaccessible_pid_count"] > 0)
        )
        and isinstance(observed, str)
        and _SHA256.fullmatch(observed)
        and observed == _gpu_receipt_digest(material)
    )


def _gpu_runtime_valid(gpu: Any, tasks: Any) -> bool:
    if not isinstance(gpu, dict) or not isinstance(tasks, list):
        return False
    pool = gpu.get("ordered_host_gpu_ids")
    devices = gpu.get("devices")
    mappings = gpu.get("task_mapping")
    if not all(isinstance(value, list) and bool(value) for value in (pool, devices, mappings)):
        return False
    expected_names = [task.get("task_name") for task in tasks if isinstance(task, dict)]
    host_ids = [
        device.get("host_device_id") for device in devices if isinstance(device, dict)
    ]
    unique_ids = [
        device.get("unique_id") for device in devices if isinstance(device, dict)
    ]
    render_paths = [
        path
        for device in devices
        if isinstance(device, dict)
        and isinstance(device.get("render_nodes"), list)
        for path in device["render_nodes"]
    ]
    kfd_device = gpu.get("kfd_device")
    target = gpu.get("target_gpu_model")
    return bool(
        gpu.get("policy") == "deterministic_task_gpu_v1"
        and all(
            isinstance(value, str) and _CANONICAL_DECIMAL.fullmatch(value)
            for value in pool
        )
        and len(pool) == len(set(pool))
        and isinstance(target, str)
        and _GPU_MODEL_ARCH.get(target) == gpu.get("gpu_arch")
        and _SHA256.fullmatch(str(gpu.get("gpu_boundary_plan_sha256") or ""))
        and isinstance(kfd_device, dict)
        and set(kfd_device) == {"path", "major", "minor"}
        and kfd_device.get("path") == "/dev/kfd"
        and all(
            isinstance(kfd_device.get(key), int)
            and not isinstance(kfd_device[key], bool)
            and kfd_device[key] >= 0
            for key in ("major", "minor")
        )
        and len(devices) == len(pool)
        and host_ids == pool
        and len(set(unique_ids)) == len(devices)
        and all(
            isinstance(unique_id, str)
            and _GPU_UNIQUE_ID.fullmatch(unique_id)
            and unique_id != "0x0000000000000000"
            for unique_id in unique_ids
        )
        and len(render_paths) == len(set(render_paths))
        and all(
            isinstance(path, str) and _RENDER_NODE_PATH.fullmatch(path)
            for path in render_paths
        )
        and all(
            isinstance(device, dict)
            and device.get("host_device_id") in pool
            and device.get("observed_gfx_version") == gpu.get("gpu_arch")
            and isinstance(device.get("serial_number"), str)
            and bool(device["serial_number"])
            and isinstance(device.get("card_series"), str)
            and bool(device["card_series"])
            and isinstance(device.get("render_nodes"), list)
            and bool(device["render_nodes"])
            and all(str(path).startswith("/dev/dri/renderD") for path in device["render_nodes"])
            for device in devices
        )
        and len(mappings) == len(tasks)
        and [mapping.get("task_name") for mapping in mappings if isinstance(mapping, dict)]
        == expected_names
        and all(
            isinstance(mapping, dict)
            and mapping.get("task_index") == index
            and mapping.get("assigned_host_gpu_id") == pool[(index - 1) % len(pool)]
            for index, mapping in enumerate(mappings, 1)
        )
        and _gpu_exclusivity_valid(gpu.get("exclusivity"), gpu)
    )


def _project_comparison_runtime(runtime: Dict[str, Any]) -> Dict[str, Any] | None:
    return _campaign_runtime_projection(runtime)


def _runtime_binding_valid(
    runtime: Any, comparison_runtime: Any, repositories: Any, tasks: Any, agent: Any
) -> bool:
    return bool(
        isinstance(runtime, dict)
        and set(runtime) == {"docker", "gpu", "isolation", "aka_execution_snapshot"}
        and isinstance(comparison_runtime, dict)
        and comparison_runtime == _project_comparison_runtime(runtime)
        and _docker_runtime_valid(runtime.get("docker"))
        and _runtime_isolation_valid(runtime.get("isolation"), agent)
        and _gpu_runtime_valid(runtime.get("gpu"), tasks)
        and _runtime_snapshot_binding_valid(repositories, runtime, comparison_runtime)
    )


def _v5_manifest_bindings_valid(
    manifest: Dict[str, Any], comparison: Dict[str, Any], tasks: Any
) -> bool:
    repositories = manifest.get("repositories")
    runtime = manifest.get("runtime")
    evaluator = manifest.get("evaluator_files_sha256")
    agent = manifest.get("agent")
    treatment = _expected_apex_treatment(repositories)
    if not isinstance(agent, dict) or treatment is None:
        return False
    template = agent.get("template")
    forbidden = {
        "apex_runtime_mount_policy_id",
        "attempt_mount_receipt_schema",
        "apex_runtime_mount_schema",
        "runtime_manifest_sha256",
    }
    agent_binding_valid = (
        template == "apex"
        and all(agent.get(key) == value for key, value in treatment.items())
    ) or (
        template == "codex"
        and agent.get("session_receipt_schema") == _CODEX_RECEIPT_SCHEMA
        and not forbidden.intersection(agent)
    )
    return bool(
        manifest.get("formal_execution") == _FORMAL_LIVE_EXECUTION
        and manifest.get("formal_execution_sha256")
        == _FORMAL_LIVE_EXECUTION_SHA256
        and comparison.get("formal_execution") == _FORMAL_LIVE_EXECUTION
        and comparison.get("formal_execution_sha256")
        == _FORMAL_LIVE_EXECUTION_SHA256
        and comparison.get("candidate_persistence_policy_id")
        == _CANDIDATE_PERSISTENCE_POLICY
        and comparison.get("boundary_quiescence_policy_id")
        == _BOUNDARY_QUIESCENCE_POLICY
        and comparison.get("agent_process_containment_policy_id")
        == _AGENT_PROCESS_CONTAINMENT_POLICY
        and comparison.get("attempt_containment_policy_id")
        == _ATTEMPT_CONTAINMENT_POLICY
        and _campaign_policy_valid(manifest, comparison)
        and _measurement_binding_valid(manifest, comparison)
        and _task_contracts_valid(tasks)
        and comparison.get("repositories") == repositories
        and _repository_bindings_valid(repositories)
        and comparison.get("apex_treatment") == treatment
        and agent_binding_valid
        and _formal_agent_valid(agent)
        and _agent_transport_treatments_valid(agent, comparison)
        and _formal_agent_valid(comparison.get("codex"))
        and _runtime_binding_valid(
            runtime, comparison.get("runtime"), repositories, tasks, agent
        )
        and comparison.get("evaluator_files_sha256") == evaluator
        and _evaluator_binding_valid(repositories, evaluator)
    )


def _formal_manifest_context(
    run_path: Path, manifest: Dict[str, Any], manifest_bytes: bytes
) -> Dict[str, Any]:
    configuration = manifest.get("configuration")
    tasks = configuration.get("tasks") if isinstance(configuration, dict) else None
    comparison = manifest.get("comparison_contract")
    comparison_sha256 = manifest.get("comparison_contract_sha256")
    agent = manifest.get("agent")
    agent_template = agent.get("template") if isinstance(agent, dict) else None
    comparison_codex = comparison.get("codex") if isinstance(comparison, dict) else None
    if (
        not isinstance(tasks, list)
        or not tasks
        or not isinstance(comparison, dict)
        or comparison.get("schema") != _COMPARISON_SCHEMA
        or not _v5_manifest_bindings_valid(manifest, comparison, tasks)
        or comparison.get("objective_policy_id") != _OBJECTIVE_POLICY
        or comparison.get("prompt_policy_id") != _PROMPT_POLICY
        or comparison.get("tasks") != tasks
        or not isinstance(comparison_codex, dict)
        or any(
            field not in comparison_codex
            or not isinstance(agent, dict)
            or agent.get(field) != comparison_codex[field]
            for field in _CODEX_IDENTITY_FIELDS
        )
        or any(
            not isinstance(agent, dict) or agent.get(field) != value
            for field, value in comparison_codex.items()
        )
        or not isinstance(comparison_sha256, str)
        or not _SHA256.fullmatch(comparison_sha256)
        or _sha256_bytes(_canonical_json(comparison).encode()) != comparison_sha256
        or agent_template not in {"apex", "codex"}
    ):
        raise ValueError("formal campaign comparison/cohort/agent binding is invalid")

    task_names = []
    task_components = []
    task_entries = {}
    for expected_index, task in enumerate(tasks, 1):
        if (
            not isinstance(task, dict)
            or task.get("task_index") != expected_index
            or not isinstance(task.get("task_name"), str)
            or not task["task_name"]
        ):
            raise ValueError("formal campaign cohort is malformed")
        task_names.append(task["task_name"])
        try:
            task_components.append(
                campaign_task_path_component(task["task_name"])
            )
        except (RuntimeError, TypeError, UnicodeError) as error:
            raise ValueError("formal campaign task name is unsafe") from error
        task_entries[task["task_name"]] = task
    if (
        len(task_names) != len(set(task_names))
        or len(task_components) != len(set(task_components))
    ):
        raise ValueError("formal campaign cohort contains duplicates")

    metadata = postprocessing._extract_run_metadata(run_path)
    if metadata.get("agent") != agent_template:
        raise ValueError("formal run path agent differs from campaign manifest agent")
    return {
        "task_names": task_names,
        "task_entries": task_entries,
        "manifest": manifest,
        "campaign_manifest_sha256": _sha256_bytes(manifest_bytes),
        "comparison_contract_sha256": comparison_sha256,
        "ordered_cohort_sha256": _sha256_bytes(_canonical_json(tasks).encode()),
        "agent_template": agent_template,
        "run_metadata": metadata,
    }


def _formal_success_projection(
    run_path: Path, task_name: str, workspace: Path, formal: Dict[str, Any]
) -> tuple[Dict[str, Any], str]:
    try:
        result, lineage = postprocessing._validate_canonical_lineage(
            run_directory=run_path,
            task_name=task_name,
            canonical=workspace,
            formal=formal,
        )
        pass_compilation = result.get("pass_compilation")
        pass_correctness = result.get("pass_correctness")
        summary = result.get("optimization_summary", "") or ""
        if (
            pass_compilation is not True
            or pass_correctness is not True
            or not isinstance(summary, str)
        ):
            raise ValueError("canonical task result has invalid success fields")
        speedup = resolve_speedup_ratio(
            speedup_ratio=result.get("speedup_ratio", 0.0),
            base_execution_time=result.get("base_execution_time", 0.0),
            best_optimized_execution_time=result.get(
                "best_optimized_execution_time", 0.0
            ),
        )
        if not math.isfinite(speedup) or speedup <= 0:
            raise ValueError("canonical task result has invalid speedup")
        score = task_result_scoring(str(workspace))
    except Exception as error:
        raise ValueError(
            f"formal canonical evidence is invalid for {task_name}: {error}"
        ) from error
    return {
        "task_name": task_name,
        "score": score,
        "pass_compilation": True,
        "pass_correctness": True,
        "speedup_ratio": speedup,
    }, lineage["canonical_workspace_manifest_sha256"]


def _formal_failure_projection(
    run_path: Path, task_name: str, formal: Dict[str, Any]
) -> tuple[Dict[str, Any], Dict[str, Any], bool]:
    evidence_parent = (
        run_path / ".campaign_attempts" / campaign_task_path_component(task_name)
    )
    if evidence_parent.exists():
        postprocessing._require_regular_directory_chain(
            evidence_parent, run_path, "failed task campaign"
        )
    failed_directory = run_path / ".parallel" / "failed"
    if failed_directory.exists():
        postprocessing._require_regular_directory_chain(
            failed_directory, run_path, "failed task descriptors"
        )
    failure = postprocessing._validated_failure_binding(
        run_path, task_name, formal
    )
    task = {
        "task_name": task_name,
        "score": 0.0,
        "pass_compilation": False,
        "pass_correctness": False,
        "speedup_ratio": 0.0,
    }
    report_entry = {
        "task_name": task_name,
        "reason_codes": failure["reason_codes"],
        "campaign_evidence_path": failure["campaign_evidence_path"],
        "campaign_evidence_sha256": failure["campaign_evidence_sha256"],
    }
    return task, report_entry, failure["terminal_binding_verified"] is True


def _recompute_formal_report(
    run_path: Path, formal: Dict[str, Any]
) -> Dict[str, Any]:
    task_names = formal["task_names"]
    workspace_map = postprocessing._formal_workspace_map(run_path, task_names)
    task_details = []
    failed_tasks = []
    canonical_manifests = {}
    terminal_task_count = 0

    for task_name in task_names:
        workspace = workspace_map.get(task_name)
        if workspace is not None:
            detail, manifest_sha256 = _formal_success_projection(
                run_path, task_name, workspace, formal
            )
            canonical_manifests[task_name] = manifest_sha256
            terminal_task_count += 1
        else:
            detail, failed, terminal = _formal_failure_projection(
                run_path, task_name, formal
            )
            failed_tasks.append(failed)
            terminal_task_count += int(terminal)
        task_details.append(detail)

    total_tasks = len(task_names)
    total_score = sum(task["score"] for task in task_details)
    compilation_count = sum(task["pass_compilation"] for task in task_details)
    correctness_count = sum(task["pass_correctness"] for task in task_details)
    speedups = [
        task["speedup_ratio"]
        for task in task_details
        if task["pass_compilation"]
        and task["pass_correctness"]
        and task["speedup_ratio"] > 0
    ]
    speedup_gt_1_count = sum(value > 1.0 for value in speedups)
    reason_counts: Dict[str, int] = defaultdict(int)
    for failure in failed_tasks:
        for reason in failure["reason_codes"]:
            reason_counts[reason] += 1
    completed = bool(
        terminal_task_count == total_tasks
        and postprocessing._formal_queue_has_no_unfinished_work(run_path)
    )
    overall = {
        "total_tasks": total_tasks,
        "total_score": total_score,
        "average_score": total_score / total_tasks,
        "compilation_pass_count": compilation_count,
        "compilation_pass_rate": compilation_count / total_tasks * 100,
        "correctness_pass_count": correctness_count,
        "correctness_pass_rate": correctness_count / total_tasks * 100,
        "speedup_gt_1_count": speedup_gt_1_count,
        "speedup_gt_1_rate": speedup_gt_1_count / total_tasks * 100,
        **postprocessing._compute_speedup_stats(speedups),
        "valid_speedup_count": len(speedups),
        "speedup_population": "canonical_compilation_and_correctness_successes_only",
        "formal_campaign": True,
        "canonical_success_count": len(canonical_manifests),
        "failed_task_count": len(failed_tasks),
        "failure_reason_counts": dict(sorted(reason_counts.items())),
        "campaign_manifest_sha256": formal["campaign_manifest_sha256"],
        "comparison_contract_sha256": formal["comparison_contract_sha256"],
        "ordered_cohort_sha256": formal["ordered_cohort_sha256"],
        "terminal_task_count": terminal_task_count,
        "formal_completion_verified": completed,
    }
    evidence = {
        "schema": "aka.formal-report-evidence/v1",
        "campaign_manifest_sha256": formal["campaign_manifest_sha256"],
        "comparison_contract_sha256": formal["comparison_contract_sha256"],
        "ordered_cohort_sha256": formal["ordered_cohort_sha256"],
        "completion_verified": completed,
        "terminal_task_count": terminal_task_count,
        "canonical_workspace_manifests": canonical_manifests,
    }
    metadata = formal["run_metadata"]
    return {
        "run_timestamp": metadata["timestamp"],
        "agent": formal["agent_template"],
        "target_gpu": metadata["target_gpu"],
        "overall": overall,
        "task_types": postprocessing._aggregate_by_task_type(task_details),
        "formal_evidence": evidence,
        "failed_tasks": failed_tasks,
    }


def _formal_report_contract(run_path: Path, data: Dict[str, Any]) -> Dict[str, str]:
    run_metadata = run_path.lstat()
    if stat.S_ISLNK(run_metadata.st_mode) or not stat.S_ISDIR(run_metadata.st_mode):
        raise ValueError(f"formal run directory is unsafe: {run_path}")
    reports = run_path / "reports"
    reports_metadata = reports.lstat()
    if stat.S_ISLNK(reports_metadata.st_mode) or not stat.S_ISDIR(reports_metadata.st_mode):
        raise ValueError(f"formal reports directory is unsafe: {reports}")
    if reports.resolve(strict=True).parent != run_path.resolve(strict=True):
        raise ValueError("formal reports directory escapes its run")

    manifest_path = run_path / "campaign_manifest.yaml"
    manifest_bytes = _read_regular_file_no_follow(manifest_path, immutable=True)
    manifest = yaml.safe_load(manifest_bytes.decode("utf-8")) or {}
    if not isinstance(manifest, dict) or manifest.get("schema") != "aka.matched-campaign/v1":
        raise ValueError("formal campaign manifest schema is invalid")
    formal = _formal_manifest_context(run_path, manifest, manifest_bytes)
    expected = _recompute_formal_report(run_path, formal)
    if _canonical_json(data) != _canonical_json(expected):
        raise ValueError("formal report does not match recomputed sealed evidence")
    if expected["overall"]["formal_completion_verified"] is not True:
        raise ValueError("formal campaign sealed evidence is not terminal")
    return {
        "campaign_manifest_sha256": formal["campaign_manifest_sha256"],
        "comparison_contract_sha256": formal["comparison_contract_sha256"],
        "ordered_cohort_sha256": formal["ordered_cohort_sha256"],
        "agent_template": formal["agent_template"],
        "resolved_run_path": str(run_path.resolve(strict=True)),
    }


def load_run_data(run_path: Path) -> Dict[str, Any]:
    """
    Load task_type_breakdown.json from a run directory.
    
    Args:
        run_path: Path to run directory (e.g., workspace_MI300_cursor/run_20260714_120000_baseline)
    
    Returns:
        Dictionary containing run data from JSON file
    
    Raises:
        FileNotFoundError: If JSON file doesn't exist
        json.JSONDecodeError: If JSON file is invalid
    """
    json_path = run_path / "reports" / "task_type_breakdown.json"
    
    raw = _read_regular_file_no_follow(json_path)
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"run report is not a JSON object: {json_path}")
    manifest_path = run_path / "campaign_manifest.yaml"
    try:
        manifest_path.lstat()
    except FileNotFoundError:
        manifest_present = False
    else:
        manifest_present = True
    report_declares_formal = data.get("overall", {}).get("formal_campaign") is True
    if manifest_present != report_declares_formal:
        raise ValueError("run report formal status differs from campaign manifest")
    if report_declares_formal:
        # Re-open under the immutable policy so a writable or swapped report
        # cannot opt itself into formal comparison.
        immutable_raw = _read_regular_file_no_follow(json_path, immutable=True)
        if immutable_raw != raw:
            raise ValueError("formal report changed while being loaded")
        data["_formal_contract"] = _formal_report_contract(run_path, data)
        if _read_regular_file_no_follow(json_path, immutable=True) != immutable_raw:
            raise ValueError("formal report changed during evidence validation")
    return data


def _validate_comparable_formal_runs(
    run1_data: Dict[str, Any], run2_data: Dict[str, Any]
) -> None:
    formal1 = run1_data.get("overall", {}).get("formal_campaign") is True
    formal2 = run2_data.get("overall", {}).get("formal_campaign") is True
    if not (formal1 or formal2):
        return
    if not (formal1 and formal2):
        raise ValueError("formal comparison requires two formal completed runs")
    contract1 = run1_data.get("_formal_contract")
    contract2 = run2_data.get("_formal_contract")
    if not isinstance(contract1, dict) or not isinstance(contract2, dict):
        raise ValueError("formal comparison lacks validated run contracts")
    if (
        contract1.get("comparison_contract_sha256")
        != contract2.get("comparison_contract_sha256")
    ):
        raise ValueError("formal run comparison contracts differ")
    if contract1.get("ordered_cohort_sha256") != contract2.get("ordered_cohort_sha256"):
        raise ValueError("formal run ordered cohorts differ")
    if contract1.get("resolved_run_path") == contract2.get("resolved_run_path"):
        raise ValueError("formal comparison cannot compare a run with itself")
    arms = {contract1.get("agent_template"), contract2.get("agent_template")}
    if arms != {"apex", "codex"}:
        raise ValueError("formal comparison requires exactly one apex and one codex arm")


def format_difference(value1: float, value2: float, is_percentage: bool = False) -> str:
    """
    Format the difference between two values.
    
    Args:
        value1: First value (baseline)
        value2: Second value (comparison)
        is_percentage: If True, format as percentage change
    
    Returns:
        Formatted string showing difference
    """
    diff = value2 - value1
    if is_percentage:
        if value1 == 0:
            return f"{diff:+.1f}pp" if diff != 0 else "0.0pp"
        pct_change = (diff / value1 * 100) if value1 != 0 else 0
        return f"{diff:+.1f}pp ({pct_change:+.1f}%)"
    else:
        pct_change = (diff / value1 * 100) if value1 != 0 else 0
        return f"{diff:+.3f} ({pct_change:+.1f}%)"


def compare_overall(run1_data: Dict[str, Any], run2_data: Dict[str, Any]) -> list:
    """
    Compare overall statistics between two runs.
    
    Returns:
        List of formatted comparison lines
    """
    overall1 = run1_data.get('overall', {})
    overall2 = run2_data.get('overall', {})
    formal = (
        overall1.get("formal_campaign") is True
        and overall2.get("formal_campaign") is True
    )
    speedup_population = "Canonical-success-only " if formal else ""
    
    lines = [
        "=" * 80,
        "OVERALL STATISTICS COMPARISON",
        "=" * 80,
        f"Run 1: {run1_data.get('run_timestamp', 'unknown')} ({run1_data.get('agent', 'unknown')})",
        f"Run 2: {run2_data.get('run_timestamp', 'unknown')} ({run2_data.get('agent', 'unknown')})",
        "=" * 80,
        "",
        f"{'Metric':<40} {'Run 1':<15} {'Run 2':<15} {'Difference':<20}",
        "-" * 80,
    ]
    
    metrics = []
    if formal:
        metrics.extend([
            ('Canonical Success Tasks', 'canonical_success_count', False),
            ('Failed Tasks', 'failed_task_count', False),
            ('Canonical-success-only Speedup Count', 'valid_speedup_count', False),
        ])
    metrics.extend([
        ('Total Tasks', 'total_tasks', False),
        ('Total Score', 'total_score', False),
        ('Average Score', 'average_score', False),
        ('Compilation Pass Rate', 'compilation_pass_rate', True),
        ('Correctness Pass Rate', 'correctness_pass_rate', True),
        ('Speedup > 1.0 Rate', 'speedup_gt_1_rate', True),
        (f'{speedup_population}Average Speedup', 'average_speedup', False),
        (f'{speedup_population}Median Speedup', 'median_speedup', False),
        (f'{speedup_population}Std Dev Speedup', 'std_dev_speedup', False),
        (f'{speedup_population}P25 Speedup', 'p25_speedup', False),
        (f'{speedup_population}P75 Speedup', 'p75_speedup', False),
        (f'{speedup_population}P90 Speedup', 'p90_speedup', False),
    ])
    
    for label, key, is_percentage in metrics:
        val1 = overall1.get(key, 0.0)
        val2 = overall2.get(key, 0.0)
        
        if is_percentage:
            fmt1 = f"{val1:.1f}%"
            fmt2 = f"{val2:.1f}%"
        elif key in {
            'total_tasks',
            'canonical_success_count',
            'failed_task_count',
            'valid_speedup_count',
        }:
            fmt1 = f"{int(val1)}"
            fmt2 = f"{int(val2)}"
        elif key == 'total_score':
            fmt1 = f"{val1:.2f}"
            fmt2 = f"{val2:.2f}"
        else:
            fmt1 = f"{val1:.3f}"
            fmt2 = f"{val2:.3f}"
        
        diff_str = format_difference(val1, val2, is_percentage)
        
        # Determine if improvement (green) or regression (red) - for display purposes
        if key in ['average_score', 'compilation_pass_rate', 'correctness_pass_rate',
                   'canonical_success_count', 'valid_speedup_count',
                   'speedup_gt_1_rate', 'average_speedup', 'median_speedup', 
                   'p25_speedup', 'p75_speedup', 'p90_speedup']:
            if val2 > val1:
                indicator = "↑"
            elif val2 < val1:
                indicator = "↓"
            else:
                indicator = "="
        elif key == 'std_dev_speedup':
            # Lower std dev is better (more consistent), so reverse the logic
            if val2 < val1:
                indicator = "↑"
            elif val2 > val1:
                indicator = "↓"
            else:
                indicator = "="
        elif key == 'failed_task_count':
            if val2 < val1:
                indicator = "↑"
            elif val2 > val1:
                indicator = "↓"
            else:
                indicator = "="
        else:
            indicator = ""
        
        lines.append(f"{label:<40} {fmt1:<15} {fmt2:<15} {diff_str:<20} {indicator}")
    
    lines.append("")
    return lines


def compare_task_types(run1_data: Dict[str, Any], run2_data: Dict[str, Any]) -> list:
    """
    Compare task type breakdowns between two runs.
    
    Returns:
        List of formatted comparison lines
    """
    types1 = run1_data.get('task_types', {})
    types2 = run2_data.get('task_types', {})
    formal = (
        run1_data.get("overall", {}).get("formal_campaign") is True
        and run2_data.get("overall", {}).get("formal_campaign") is True
    )
    speedup_population = "Canonical-success-only " if formal else ""
    
    # Get all unique task types from both runs
    all_types = set(types1.keys()) | set(types2.keys())
    
    if not all_types:
        return ["No task type data available for comparison."]
    
    lines = [
        "=" * 80,
        "TASK TYPE BREAKDOWN COMPARISON",
        "=" * 80,
        "",
    ]
    
    for task_type in sorted(all_types):
        stats1 = types1.get(task_type, {})
        stats2 = types2.get(task_type, {})
        
        count1 = stats1.get('count', 0)
        count2 = stats2.get('count', 0)
        
        lines.append(f"{task_type.upper()} ({count1} tasks → {count2} tasks):")
        lines.append("-" * 80)
        
        if count1 == 0 and count2 == 0:
            lines.append("  No tasks in either run")
            lines.append("")
            continue
        
        # Compare key metrics
        metrics = [
            (f'{speedup_population}Average Speedup', 'average_speedup', False),
            (f'{speedup_population}Median Speedup', 'median_speedup', False),
            (f'{speedup_population}Std Dev Speedup', 'std_dev_speedup', False),
            (f'{speedup_population}P25 Speedup', 'p25_speedup', False),
            (f'{speedup_population}P75 Speedup', 'p75_speedup', False),
            (f'{speedup_population}P90 Speedup', 'p90_speedup', False),
            ('Compilation Pass Rate', 'compilation_pass_rate', True),
            ('Correctness Pass Rate', 'correctness_pass_rate', True),
            ('Speedup > 1.0 Rate', 'speedup_gt_1_rate', True),
            ('Average Score', 'average_score', False),
        ]
        
        for label, key, is_percentage in metrics:
            val1 = stats1.get(key, 0.0)
            val2 = stats2.get(key, 0.0)

            # Format both values first, then override with N/A as needed
            if is_percentage:
                fmt1 = f"{val1:.1f}%"
                fmt2 = f"{val2:.1f}%"
            elif key == 'average_score':
                fmt1 = f"{val1:.2f}"
                fmt2 = f"{val2:.2f}"
            else:
                fmt1 = f"{val1:.3f}"
                fmt2 = f"{val2:.3f}"

            if count1 == 0:
                fmt1 = "N/A"
                diff_str = "N/A (new)"
            elif count2 == 0:
                fmt2 = "N/A"
                diff_str = "N/A (removed)"
            else:
                diff_str = format_difference(val1, val2, is_percentage)
            
            if count1 > 0 and count2 > 0:
                # For std_dev_speedup, lower is better (more consistent)
                if key == 'std_dev_speedup':
                    if val2 < val1:
                        indicator = "↑ (improved)"
                    elif val2 > val1:
                        indicator = "↓ (regressed)"
                    else:
                        indicator = "= (same)"
                # For percentiles and other speedup metrics, higher is better
                elif key in ['p25_speedup', 'p75_speedup', 'p90_speedup', 'average_speedup', 'median_speedup']:
                    if val2 > val1:
                        indicator = "↑ (improved)"
                    elif val2 < val1:
                        indicator = "↓ (regressed)"
                    else:
                        indicator = "= (same)"
                else:
                    if val2 > val1:
                        indicator = "↑ (improved)"
                    elif val2 < val1:
                        indicator = "↓ (regressed)"
                    else:
                        indicator = "= (same)"
            else:
                indicator = ""
            
            if count1 == 0:
                lines.append(f"  {label:<35} {'N/A':<15} {fmt2:<15} {diff_str:<20} {indicator}")
            elif count2 == 0:
                lines.append(f"  {label:<35} {fmt1:<15} {'N/A':<15} {diff_str:<20} {indicator}")
            else:
                lines.append(f"  {label:<35} {fmt1:<15} {fmt2:<15} {diff_str:<20} {indicator}")
        
        lines.append("")
    
    return lines


def generate_comparison_report(run1_path: Path, run2_path: Path, output_path: Optional[Path] = None) -> str:
    """
    Generate a comparison report between two runs.
    
    Args:
        run1_path: Path to first run directory
        run2_path: Path to second run directory
        output_path: Optional path to save report (if None, auto-generates in comparisons/ directory)
    
    Returns:
        Comparison report as string
    """
    run1_data = load_run_data(run1_path)
    run2_data = load_run_data(run2_path)
    _validate_comparable_formal_runs(run1_data, run2_data)
    
    # Generate comparison report
    lines = [
        "=" * 80,
        "AgentKernelArena Run Comparison Report",
        "=" * 80,
        "",
    ]
    
    lines.extend(compare_overall(run1_data, run2_data))
    lines.extend(compare_task_types(run1_data, run2_data))
    
    lines.extend([
        "=" * 80,
        "Legend:",
        "  ↑ = Improvement (higher is better)",
        "  ↓ = Regression (lower is worse)",
        "  = = No change",
        "  pp = percentage points",
        "=" * 80,
    ])
    
    report = "\n".join(lines)
    
    # Determine output path
    if output_path is None:
        # Auto-generate path in comparisons/ directory at project root
        # Extract run directory names (e.g., "run_20260714_120000_baseline" from full path)
        run1_name = run1_path.name
        run2_name = run2_path.name
        
        # Keep generated comparisons at the project root even though this CLI
        # lives under src/tools/.
        project_root = Path(__file__).resolve().parents[2]
        
        comparisons_dir = project_root / "comparisons"
        comparisons_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename: comparison_report_{run1}_{run2}.txt
        filename = f"comparison_report_{run1_name}_{run2_name}.txt"
        output_path = comparisons_dir / filename
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)
    print(f"Comparison report written to: {output_path}")
    
    return report


def main():
    """Main entry point for comparison script."""
    parser = argparse.ArgumentParser(
        description="Compare two AgentKernelArena runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two runs
  python3 src/tools/compare_runs.py workspace_MI300_cursor/run_20260714_120000_baseline workspace_MI300_cursor/run_20260714_140000_treatment
  
  # Compare and save to file
  python3 src/tools/compare_runs.py run1 run2 --output comparison_report.txt
        """
    )
    
    parser.add_argument(
        'run1',
        type=str,
        help='Path to baseline/first run directory (e.g., workspace_MI300_cursor/run_20260714_120000_baseline)'
    )
    
    parser.add_argument(
        'run2',
        type=str,
        help='Path to treatment/second run directory (e.g., workspace_MI300_cursor/run_20260714_140000_treatment)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Optional output file path for comparison report (if not specified, auto-generates in comparisons/ directory)'
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    run1_path = Path(args.run1).resolve()
    run2_path = Path(args.run2).resolve()
    
    # Validate paths exist
    if not run1_path.exists():
        print(f"Error: Run 1 directory does not exist: {run1_path}", file=sys.stderr)
        sys.exit(1)
    
    if not run2_path.exists():
        print(f"Error: Run 2 directory does not exist: {run2_path}", file=sys.stderr)
        sys.exit(1)
    
    # Generate and print comparison report
    output_path = Path(args.output).resolve() if args.output else None
    try:
        report = generate_comparison_report(run1_path, run2_path, output_path)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    
    # Print to stdout
    print(report)


if __name__ == "__main__":
    main()
