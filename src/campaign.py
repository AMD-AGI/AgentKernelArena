# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Matched-campaign orchestration and provenance for Apex-versus-Codex runs."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import shutil
import sqlite3
import stat
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from src.agent_turn_budget import (
    FORMAL_MATCHED_MAX_TURNS,
    TURN_POLICY,
    budget_stop_reason_matches,
    context_packet_objective_matches,
    render_apex_run_control,
)
from src.campaign_isolation import CampaignIsolationError, runtime_isolation_receipt
from src.gpu_device_boundary import GpuBoundaryError, load_plan
from src.gpu_exclusivity import (
    GpuExclusivityError,
    load_receipt as load_gpu_lease_receipt,
)
from src.preprocessing import get_task_workspace_path


_CAMPAIGN_SCHEMA = "aka.matched-campaign/v1"
_TASK_SCHEMA = "aka.matched-task-attempts/v1"
_SELECTION_POLICY = "correctness_then_measured_rate_v1"
_MEASUREMENT_CONTRACT = "aka_native_100_repetition_external_score"
_OBJECTIVE_POLICY_ID = "aka.task-package-objective-and-protected-harness/v1"
_PROMPT_POLICY_ID = "aka.shared-objective-backend-native-context-receipted/v1"
_CODEX_RECEIPT_SCHEMA = "agentkernelarena.codex-attempt-receipt/v1"
_APEX_RECEIPT_SCHEMA_V1 = "agentkernelarena.apex-attempt-receipt/v1"
_APEX_RECEIPT_SCHEMA_V2 = "agentkernelarena.apex-attempt-receipt/v2"
_APEX_RECEIPT_SCHEMAS = {_APEX_RECEIPT_SCHEMA_V1, _APEX_RECEIPT_SCHEMA_V2}
_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class CampaignError(RuntimeError):
    """Raised when a matched campaign cannot preserve its fairness contract."""


@dataclass(frozen=True)
class CampaignPolicy:
    comparison: str
    attempts: int
    attempt_timeout_seconds: int
    apex_internal_allowance_seconds: int
    task_timeout_seconds: int
    evaluator_allowance_seconds: int
    selection_policy: str
    workspace_policy: str
    gpu_policy: str
    require_clean_checkouts: bool


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = yaml.safe_dump(payload, default_flow_style=False, sort_keys=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def _seal_evidence_file(path: Path, label: str) -> None:
    """Make a newly-published evidence file immutable to later pipeline stages."""
    try:
        metadata = path.lstat()
    except OSError as error:
        raise CampaignError(f"cannot inspect {label}: {path}: {error}") from error
    if not path.is_file() or path.is_symlink() or metadata.st_nlink != 1:
        raise CampaignError(f"cannot seal unsafe {label}: {path}")
    path.chmod(0o444)


def parse_campaign_policy(eval_config: dict[str, Any]) -> CampaignPolicy | None:
    raw = eval_config.get("campaign")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise CampaignError("campaign must be a mapping")
    policy = CampaignPolicy(
        comparison=str(raw.get("comparison") or ""),
        attempts=int(raw.get("attempts", 0)),
        attempt_timeout_seconds=int(raw.get("attempt_timeout_seconds", 0)),
        apex_internal_allowance_seconds=int(
            raw.get("apex_internal_allowance_seconds", 0)
        ),
        task_timeout_seconds=int(raw.get("task_timeout_seconds", 0)),
        evaluator_allowance_seconds=int(raw.get("evaluator_allowance_seconds", 0)),
        selection_policy=str(raw.get("selection_policy") or ""),
        workspace_policy=str(raw.get("workspace_policy") or ""),
        gpu_policy=str(raw.get("gpu_policy") or ""),
        require_clean_checkouts=raw.get("require_clean_checkouts") is True,
    )
    _validate_policy(policy)
    return policy


def _validate_policy(policy: CampaignPolicy) -> None:
    if policy.comparison != "apex_vs_codex":
        raise CampaignError("matched campaign comparison must be apex_vs_codex")
    if policy.attempts != 3:
        raise CampaignError("apex_vs_codex requires exactly three independent attempts")
    if policy.attempt_timeout_seconds != 3600:
        raise CampaignError("each apex_vs_codex attempt must have a 3600-second agent budget")
    if policy.apex_internal_allowance_seconds != 3600:
        raise CampaignError(
            "apex_vs_codex requires a separate 3600-second Apex internal-evaluator allowance"
        )
    minimum = policy.attempts * (
        policy.attempt_timeout_seconds + policy.apex_internal_allowance_seconds
    )
    minimum += policy.evaluator_allowance_seconds
    if policy.task_timeout_seconds < minimum:
        raise CampaignError(
            "task_timeout_seconds must cover all agent attempts plus evaluator allowance"
        )
    if policy.selection_policy != _SELECTION_POLICY:
        raise CampaignError(f"selection_policy must be {_SELECTION_POLICY}")
    if policy.workspace_policy != "fresh_per_attempt":
        raise CampaignError("workspace_policy must be fresh_per_attempt")
    if policy.gpu_policy != "deterministic_task_gpu_v1":
        raise CampaignError("gpu_policy must be deterministic_task_gpu_v1")
    if not policy.require_clean_checkouts:
        raise CampaignError("formal matched campaigns require clean checkouts")


def ordered_gpu_pool() -> list[str]:
    raw = os.environ.get("AGENT_KERNEL_ARENA_GPU_POOL", "")
    values = raw.split(",") if raw else []
    if not values or any(not value.isdigit() for value in values):
        raise CampaignError(
            "deterministic campaign requires an ordered numeric GPU pool from parallel-run"
        )
    if len(values) != len(set(values)):
        raise CampaignError("deterministic campaign GPU pool contains duplicates")
    return values


def deterministic_task_gpu_mapping(task_names: list[str]) -> list[dict[str, Any]]:
    pool = ordered_gpu_pool()
    return [
        {
            "task_index": index,
            "task_name": task_name,
            "assigned_host_gpu_id": pool[(index - 1) % len(pool)],
        }
        for index, task_name in enumerate(task_names, 1)
    ]


def _load_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise CampaignError(f"cannot read {label}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise CampaignError(f"{label} must contain a YAML mapping: {path}")
    return value


def _run_text(argv: list[str], *, cwd: Path | None = None) -> str:
    try:
        completed = subprocess.run(
            argv,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CampaignError(f"provenance command failed: {argv[0]}: {error}") from error
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()[-1000:]
        raise CampaignError(f"provenance command failed: {argv}: {detail}")
    return completed.stdout.rstrip("\r\n")


def _git_state(root: Path) -> dict[str, Any]:
    commit = _run_text(["git", "rev-parse", "HEAD"], cwd=root)
    if not _SHA1.fullmatch(commit):
        raise CampaignError(f"invalid Git commit for {root}: {commit!r}")
    status = _run_text(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=normal",
            "--",
            ".",
            ":(exclude).eval-tool-artifacts",
            ":(exclude)experiments",
        ],
        cwd=root,
    )
    return {
        "commit": commit,
        "dirty": bool(status),
        "status_sha256": _sha256_bytes(status.encode("utf-8")),
    }


def _apex_state_from_environment() -> dict[str, Any]:
    commit = os.environ.get("AGENT_KERNEL_ARENA_APEX_COMMIT", "")
    dirty = os.environ.get("AGENT_KERNEL_ARENA_APEX_DIRTY", "")
    status_digest = os.environ.get("AGENT_KERNEL_ARENA_APEX_STATUS_SHA256", "")
    if not _SHA1.fullmatch(commit):
        raise CampaignError("runner did not provide a valid Apex commit")
    if dirty not in {"true", "false"}:
        raise CampaignError("runner did not provide Apex dirty=true|false")
    if not _SHA256.fullmatch(status_digest):
        raise CampaignError("runner did not provide a valid Apex status digest")
    state = {
        "commit": commit,
        "dirty": dirty == "true",
        "status_sha256": status_digest,
    }
    if state["dirty"]:
        raise CampaignError("formal campaign requires a clean Apex checkout")
    return state


def _gpu_inventory(
    eval_config: dict[str, Any], task_names: list[str]
) -> dict[str, Any]:
    pool = ordered_gpu_pool()
    expected_arch = os.environ.get("AGENT_KERNEL_ARENA_GPU_ARCH", "")
    if not expected_arch:
        raise CampaignError("runner did not provide the selected GPU architecture")
    plan_path = os.environ.get("AGENT_KERNEL_ARENA_GPU_BOUNDARY_PLAN", "")
    expected_digest = os.environ.get(
        "AGENT_KERNEL_ARENA_GPU_BOUNDARY_PLAN_SHA256", ""
    )
    if not plan_path or not Path(plan_path).is_absolute():
        raise CampaignError("runner did not provide an absolute GPU boundary plan")
    if not _SHA256.fullmatch(expected_digest):
        raise CampaignError("runner did not provide a valid GPU boundary plan digest")
    try:
        plan = load_plan(Path(plan_path), expected_gpu_ids=pool)
    except GpuBoundaryError as error:
        raise CampaignError(f"cannot validate GPU boundary plan: {error}") from error
    if plan["sha256"] != expected_digest:
        raise CampaignError("GPU boundary plan digest differs from the runner receipt")
    exclusivity_path_value = os.environ.get(
        "AGENT_KERNEL_ARENA_GPU_EXCLUSIVITY_RECEIPT", ""
    )
    exclusivity_digest = os.environ.get(
        "AGENT_KERNEL_ARENA_GPU_EXCLUSIVITY_RECEIPT_SHA256", ""
    )
    exclusivity_path = Path(exclusivity_path_value)
    if not exclusivity_path.is_absolute() or not _SHA256.fullmatch(exclusivity_digest):
        raise CampaignError("runner did not provide immutable GPU exclusivity evidence")
    try:
        exclusivity = load_gpu_lease_receipt(
            exclusivity_path, expected_plan_sha256=plan["sha256"]
        )
    except GpuExclusivityError as error:
        raise CampaignError(f"cannot validate GPU exclusivity receipt: {error}") from error
    if exclusivity.get("sha256") != exclusivity_digest:
        raise CampaignError("GPU exclusivity receipt digest differs from runner evidence")

    devices: list[dict[str, Any]] = []
    for physical in plan["devices"]:
        if physical["gfx_version"] != expected_arch:
            raise CampaignError(
                f"card{physical['host_gpu_id']} architecture {physical['gfx_version']!r} "
                f"does not match selected {expected_arch!r}"
            )
        devices.append(
            {
                "host_device_id": physical["host_gpu_id"],
                "unique_id": physical["unique_id"],
                "serial_number": physical["serial_number"],
                "card_series": physical["card_series"],
                "observed_gfx_version": physical["gfx_version"],
                "render_nodes": [
                    render["path"] for render in physical["render_nodes"]
                ],
            }
        )
    return {
        "policy": "deterministic_task_gpu_v1",
        "ordered_host_gpu_ids": pool,
        "target_gpu_model": str(eval_config.get("target_gpu_model") or ""),
        "gpu_arch": expected_arch,
        "gpu_boundary_plan_sha256": plan["sha256"],
        "kfd_device": plan["kfd_device"],
        "exclusivity": exclusivity,
        "devices": devices,
        "task_mapping": deterministic_task_gpu_mapping(task_names),
    }


def _agent_manifest(repo_root: Path, agent_name: str, policy: CampaignPolicy) -> dict[str, Any]:
    config_path = repo_root / "agents" / agent_name / "agent_config.yaml"
    config = _load_mapping(config_path, "agent config")
    if config.get("model") != "gpt-5.5" or config.get("effort") != "xhigh":
        raise CampaignError("matched campaign requires Codex gpt-5.5 with xhigh effort")
    if config.get("permission_mode") != "workspace_write_isolated":
        raise CampaignError("matched campaign requires symmetric workspace-write isolation")
    if config.get("backend", "codex") != "codex":
        raise CampaignError("apex_vs_codex requires Codex as the backend on both paths")
    if int(config.get("campaign_max_iterations", 0)) != 1:
        raise CampaignError("matched campaign requires one inner iteration per fresh session")
    if int(config.get("timeout_seconds", 0)) != policy.attempt_timeout_seconds:
        raise CampaignError("agent timeout must match campaign attempt_timeout_seconds")
    if int(config.get("max_turns", 0)) != FORMAL_MATCHED_MAX_TURNS:
        raise CampaignError(
            f"matched campaign requires max_turns={FORMAL_MATCHED_MAX_TURNS}"
        )
    if int(config.get("structured_stream_output_limit_bytes", 0)) != 16 * 1024 * 1024:
        raise CampaignError("matched campaign requires a 16 MiB inner Codex stream bound")
    codex = shutil.which("codex")
    if not codex:
        raise CampaignError("codex CLI is unavailable for campaign provenance")
    binary_path = Path(codex).resolve()
    if not binary_path.is_file():
        raise CampaignError("resolved codex binary is not a regular file")
    return {
        "template": agent_name,
        "session_receipt_schema": (
            _APEX_RECEIPT_SCHEMA_V2
            if agent_name == "apex"
            else _CODEX_RECEIPT_SCHEMA
        ),
        "backend": config.get("backend", "codex"),
        "model": config["model"],
        "effort": config["effort"],
        "permission_mode": config["permission_mode"],
        "inner_max_iterations": config["campaign_max_iterations"],
        "attempt_timeout_seconds": config["timeout_seconds"],
        "max_turns": config["max_turns"],
        "turn_policy": TURN_POLICY,
        "max_process_output_bytes": config["max_process_output_bytes"],
        "structured_stream_output_limit_bytes": config[
            "structured_stream_output_limit_bytes"
        ],
        "structured_stream_overflow_policy": config.get(
            "structured_stream_overflow_policy"
        ),
        "codex_binary": str(binary_path),
        "codex_binary_sha256": _sha256_file(binary_path),
        "codex_version": _run_text([codex, "--version"]),
        "isolation": {
            "approval": "never_via_strict_config",
            "execpolicy_rules": "ignored",
            "project_instructions": "backend_default_may_load",
            "sandbox": "workspace-write",
            "session": "ephemeral",
            "user_config": "ignored",
            "mount_scope": "attempt_only_bubblewrap",
        },
        "agent_config_sha256": _sha256_file(config_path),
    }


def _regular_tree_manifest(root: Path) -> dict[str, str]:
    if not root.is_dir() or root.is_symlink():
        raise CampaignError(f"manifest root must be a regular directory: {root}")
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        if path.is_symlink():
            raise CampaignError(f"manifest tree contains symlink: {root}/{relative}")
        if path.is_dir():
            continue
        if not path.is_file() or metadata.st_nlink != 1:
            raise CampaignError(f"manifest tree contains unsafe file: {root}/{relative}")
        files[relative] = _sha256_file(path)
    if not files:
        raise CampaignError(f"manifest tree contains no files: {root}")
    return files


def _evaluator_manifest(repo_root: Path) -> dict[str, str]:
    relative_paths = (
        "main.py",
        "src/campaign.py",
        "src/campaign_isolation.py",
        "src/evaluator.py",
        "src/evaluator_utils.py",
        "src/harness_guard.py",
        "src/performance.py",
        "src/perf_helper_materialization.py",
        "src/score.py",
        "src/testcases.py",
    )
    manifest: dict[str, str] = {}
    for relative in relative_paths:
        path = repo_root / relative
        metadata = path.lstat()
        if not path.is_file() or path.is_symlink() or metadata.st_nlink != 1:
            raise CampaignError(f"unsafe evaluator source file: {path}")
        manifest[relative] = _sha256_file(path)
    return manifest


def _task_manifests(task_config_paths: dict[str, str]) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for index, (task_name, config_path_value) in enumerate(task_config_paths.items(), 1):
        config_path = Path(config_path_value).resolve()
        files = _regular_tree_manifest(config_path.parent)
        manifests.append(
            {
                "task_index": index,
                "task_name": task_name,
                "config_path": str(config_path),
                "config_sha256": _sha256_file(config_path),
                "package_files_sha256": files,
                "package_manifest_sha256": _sha256_bytes(
                    json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
                ),
            }
        )
    return manifests


def _load_verified_campaign_manifest(run_directory: Path) -> dict[str, Any]:
    """Load the sealed manifest and verify its self-contained comparison contract."""
    manifest_path = run_directory / "campaign_manifest.yaml"
    if not _safe_read_only_file(manifest_path):
        raise CampaignError("formal execution requires an immutable campaign manifest")
    manifest = _load_mapping(manifest_path, "campaign manifest")
    comparison = manifest.get("comparison_contract")
    comparison_digest = manifest.get("comparison_contract_sha256")
    if (
        manifest.get("schema") != _CAMPAIGN_SCHEMA
        or not isinstance(comparison, dict)
        or comparison.get("schema")
        != "aka.apex-vs-codex-comparison-contract/v1"
        or comparison.get("objective_policy_id") != _OBJECTIVE_POLICY_ID
        or comparison.get("prompt_policy_id") != _PROMPT_POLICY_ID
        or not isinstance(comparison_digest, str)
        or not _SHA256.fullmatch(comparison_digest)
        or comparison_digest
        != _sha256_bytes(
            json.dumps(comparison, sort_keys=True, separators=(",", ":")).encode()
        )
    ):
        raise CampaignError("campaign manifest comparison contract is invalid")
    return manifest


def validate_formal_task_binding(
    *,
    run_directory: Path,
    task_name: str,
    task_index: int,
    total_tasks: int,
    task_config_path: str,
    assigned_host_gpu_id: str,
) -> dict[str, Any]:
    """Bind one formal task invocation to the manifest and current package bytes."""
    manifest = _load_verified_campaign_manifest(run_directory)
    configuration = manifest.get("configuration")
    tasks = configuration.get("tasks") if isinstance(configuration, dict) else None
    runtime = manifest.get("runtime")
    gpu = runtime.get("gpu") if isinstance(runtime, dict) else None
    task_mapping = gpu.get("task_mapping") if isinstance(gpu, dict) else None
    comparison = manifest["comparison_contract"]
    comparison_runtime = comparison.get("runtime")
    comparison_gpu = (
        comparison_runtime.get("gpu")
        if isinstance(comparison_runtime, dict)
        else None
    )
    if (
        not isinstance(tasks, list)
        or not tasks
        or not isinstance(task_mapping, list)
        or len(task_mapping) != len(tasks)
        or comparison.get("tasks") != tasks
        or not isinstance(comparison_gpu, dict)
        or comparison_gpu.get("task_mapping") != task_mapping
    ):
        raise CampaignError("campaign task or GPU mapping is malformed")
    if total_tasks != len(tasks) or task_index < 1 or task_index > len(tasks):
        raise CampaignError("descriptor task count or index differs from campaign manifest")

    names: set[str] = set()
    indices: set[int] = set()
    for expected_index, (task, mapping) in enumerate(
        zip(tasks, task_mapping), 1
    ):
        if not isinstance(task, dict) or not isinstance(mapping, dict):
            raise CampaignError("campaign task mapping contains a non-mapping entry")
        expected_name = task.get("task_name")
        if (
            task.get("task_index") != expected_index
            or mapping.get("task_index") != expected_index
            or not isinstance(expected_name, str)
            or not expected_name
            or mapping.get("task_name") != expected_name
            or expected_name in names
            or expected_index in indices
            or not isinstance(mapping.get("assigned_host_gpu_id"), str)
            or not mapping["assigned_host_gpu_id"].isdigit()
        ):
            raise CampaignError("campaign task ordering or GPU mapping is inconsistent")
        names.add(expected_name)
        indices.add(expected_index)

    task = tasks[task_index - 1]
    mapping = task_mapping[task_index - 1]
    if task.get("task_name") != task_name or mapping.get("task_name") != task_name:
        raise CampaignError("descriptor task name/index differs from campaign manifest")
    if mapping.get("assigned_host_gpu_id") != assigned_host_gpu_id:
        raise CampaignError("descriptor GPU differs from campaign task mapping")

    expected_config = task.get("config_path")
    try:
        observed_config = Path(task_config_path).resolve(strict=True)
        metadata = observed_config.lstat()
    except OSError as error:
        raise CampaignError(f"cannot inspect formal task config: {error}") from error
    if (
        not isinstance(expected_config, str)
        or str(observed_config) != expected_config
        or not observed_config.is_file()
        or observed_config.is_symlink()
        or metadata.st_nlink != 1
        or _sha256_file(observed_config) != task.get("config_sha256")
    ):
        raise CampaignError("formal task config path or bytes differ from campaign manifest")

    files = _regular_tree_manifest(observed_config.parent)
    files_digest = _sha256_bytes(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    )
    if (
        files != task.get("package_files_sha256")
        or files_digest != task.get("package_manifest_sha256")
    ):
        raise CampaignError("formal task package bytes differ from campaign manifest")
    return {
        "task_index": task_index,
        "total_tasks": total_tasks,
        "task_name": task_name,
        "config_path": expected_config,
        "config_sha256": task["config_sha256"],
        "package_manifest_sha256": files_digest,
        "assigned_host_gpu_id": assigned_host_gpu_id,
    }


def _image_manifest() -> dict[str, Any]:
    image_ref = os.environ.get("AGENT_KERNEL_ARENA_DOCKER_IMAGE", "")
    image_id = os.environ.get("AGENT_KERNEL_ARENA_DOCKER_IMAGE_ID", "")
    raw_digests = os.environ.get("AGENT_KERNEL_ARENA_DOCKER_REPO_DIGESTS", "")
    if not image_ref:
        raise CampaignError("runner did not provide the inspected Docker image reference")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise CampaignError("runner did not provide a valid inspected Docker image ID")
    try:
        repo_digests = json.loads(raw_digests)
    except json.JSONDecodeError as error:
        raise CampaignError("runner provided invalid Docker RepoDigests JSON") from error
    digest_pattern = re.compile(r"[^@\s]+@sha256:[0-9a-f]{64}")
    if (
        not isinstance(repo_digests, list)
        or not repo_digests
        or any(not isinstance(value, str) or not digest_pattern.fullmatch(value) for value in repo_digests)
    ):
        raise CampaignError("formal campaign requires at least one inspected Docker repo digest")
    return {
        "reference": image_ref,
        "image_id": image_id,
        "repo_digests": sorted(set(repo_digests)),
    }


def _comparison_contract(
    *,
    policy: CampaignPolicy,
    measurement: dict[str, Any],
    repositories: dict[str, Any],
    agent: dict[str, Any],
    runtime: dict[str, Any],
    evaluator: dict[str, str],
    tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    effective_codex = {
        key: agent[key]
        for key in (
            "backend",
            "model",
            "effort",
            "permission_mode",
            "inner_max_iterations",
            "attempt_timeout_seconds",
            "max_turns",
            "turn_policy",
            "structured_stream_output_limit_bytes",
            "codex_version",
            "codex_binary_sha256",
            "isolation",
        )
    }
    comparison_runtime = dict(runtime)
    runtime_gpu = runtime.get("gpu")
    if isinstance(runtime_gpu, dict):
        comparison_gpu = dict(runtime_gpu)
        exclusivity = comparison_gpu.pop("exclusivity", None)
        if isinstance(exclusivity, dict):
            leases = exclusivity.get("leases")
            comparison_gpu["exclusivity_contract"] = {
                "policy": exclusivity.get("policy"),
                "gpu_boundary_plan_sha256": exclusivity.get(
                    "gpu_boundary_plan_sha256"
                ),
                "protected_device_paths": exclusivity.get("protected_device_paths"),
                "leased_unique_ids": sorted(
                    lease.get("unique_id")
                    for lease in leases
                    if isinstance(lease, dict)
                    and isinstance(lease.get("unique_id"), str)
                )
                if isinstance(leases, list)
                else None,
            }
        comparison_runtime["gpu"] = comparison_gpu
    return {
        "schema": "aka.apex-vs-codex-comparison-contract/v1",
        "objective_policy_id": _OBJECTIVE_POLICY_ID,
        "prompt_policy_id": _PROMPT_POLICY_ID,
        "policy": asdict(policy),
        "measurement": measurement,
        "repositories": repositories,
        "codex": effective_codex,
        "runtime": comparison_runtime,
        "evaluator_files_sha256": evaluator,
        "tasks": tasks,
    }


def build_campaign_manifest(
    *,
    eval_config: dict[str, Any],
    run_config_path: Path,
    task_config_paths: dict[str, str],
    agent_name: str,
) -> dict[str, Any] | None:
    policy = parse_campaign_policy(eval_config)
    if policy is None:
        return None
    repo_root = Path(__file__).resolve().parents[1]
    aka_state = _git_state(repo_root)
    if aka_state["dirty"]:
        raise CampaignError("formal campaign requires a clean AgentKernelArena checkout")
    repositories = {
        "agent_kernel_arena": aka_state,
        "apex": _apex_state_from_environment(),
    }
    measurement = {
        "contract": _MEASUREMENT_CONTRACT,
        "owner": "AgentKernelArena centralized evaluator",
        "configured_repetitions_per_test_case": 100,
        "is_apex_kernel_measurement_v1": False,
        "is_apex_canonical_300_sample_grade": False,
    }
    agent = _agent_manifest(repo_root, agent_name, policy)
    task_manifests = _task_manifests(task_config_paths)
    try:
        runtime_isolation = runtime_isolation_receipt()
    except CampaignIsolationError as error:
        raise CampaignError(f"formal runtime isolation is not proven: {error}") from error
    runtime = {
        "docker": _image_manifest(),
        "gpu": _gpu_inventory(eval_config, list(task_config_paths)),
        "isolation": runtime_isolation,
    }
    evaluator = _evaluator_manifest(repo_root)
    comparison = _comparison_contract(
        policy=policy,
        measurement=measurement,
        repositories=repositories,
        agent=agent,
        runtime=runtime,
        evaluator=evaluator,
        tasks=task_manifests,
    )
    return {
        "schema": _CAMPAIGN_SCHEMA,
        "policy": asdict(policy),
        "comparison_contract_sha256": _sha256_bytes(
            json.dumps(comparison, sort_keys=True, separators=(",", ":")).encode()
        ),
        "comparison_contract": comparison,
        "measurement": measurement,
        "repositories": repositories,
        "agent": agent,
        "runtime": runtime,
        "evaluator_files_sha256": evaluator,
        "configuration": {
            "run_config_path": str(run_config_path.resolve()),
            "run_config_sha256": _sha256_file(run_config_path),
            "tasks": task_manifests,
        },
    }


def ensure_campaign_manifest(
    *,
    run_directory: Path,
    eval_config: dict[str, Any],
    run_config_path: Path,
    task_config_paths: dict[str, str],
    agent_name: str,
) -> Path | None:
    manifest = build_campaign_manifest(
        eval_config=eval_config,
        run_config_path=run_config_path,
        task_config_paths=task_config_paths,
        agent_name=agent_name,
    )
    if manifest is None:
        return None
    path = run_directory / "campaign_manifest.yaml"
    if path.exists():
        if not _safe_read_only_file(path):
            raise CampaignError("existing campaign manifest is unsafe or mutable")
        existing = _load_mapping(path, "campaign manifest")
        if existing != manifest:
            raise CampaignError("campaign provenance changed after run initialization")
        return path
    _atomic_yaml(path, manifest)
    path.chmod(0o444)
    return path


def _attempt_record(
    *,
    attempt: int,
    workspace: Path | None,
    run_directory: Path,
    success: bool,
    receipt_path: Path,
    require_session_receipt: bool,
    expected_task_name: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "attempt": attempt,
        "session": f"fresh-{attempt:02d}",
        "attempt_completed": success,
        "workspace": (
            str(workspace.relative_to(run_directory)) if workspace is not None else None
        ),
        "central_evaluator_report": None,
        "selection_eligible": False,
        "measured_rate_per_ms": 0.0,
        "eligibility_errors": [],
    }
    receipt: dict[str, Any] | None = None
    if receipt_path.exists():
        receipt, receipt_errors = _validate_session_receipt(
            receipt_path=receipt_path,
            workspace=workspace,
            run_directory=run_directory,
            expected_task_name=expected_task_name,
        )
        record["session_receipt"] = str(receipt_path.relative_to(run_directory))
        if receipt is not None:
            record["session_receipt_sha256"] = _sha256_file(receipt_path)
            record["session_succeeded"] = receipt.get("session_succeeded") is True
            binding = {
                "schema": receipt.get("schema"),
                "comparison_contract_sha256": receipt.get(
                    "comparison_contract_sha256"
                ),
                "terminal_status": receipt.get("terminal_status"),
                "codex": receipt.get("codex"),
                "invocation_sha256": (
                    _canonical_json_digest(receipt["invocation"])
                    if isinstance(receipt.get("invocation"), dict)
                    else None
                ),
                "process_group_cleanup": receipt.get("process_group_cleanup"),
                "budgets": receipt.get("budgets"),
                "turn_budget": receipt.get("turn_budget"),
                "workspace_integrity": receipt.get("workspace_integrity"),
                "gpu": receipt.get("gpu"),
                "lineage": receipt.get("lineage"),
            }
            if receipt.get("schema") in _APEX_RECEIPT_SCHEMAS:
                lineage = receipt.get("lineage")
                prompt_event = (
                    lineage.get("prompt_event")
                    if isinstance(lineage, dict)
                    else None
                )
                lineage_errors = set(receipt_errors) - {
                    "apex_session_not_successful"
                }
                binding["lineage_verified"] = (
                    isinstance(lineage, dict) and not lineage_errors
                )
                binding["event_bound_prompt"] = (
                    {
                        "binding": prompt_event.get("binding"),
                        "event_id": prompt_event.get("event_id"),
                        "sha256": prompt_event.get("sha256"),
                        "size_bytes": prompt_event.get("size_bytes"),
                        "stdin_transport_attested": prompt_event.get(
                            "stdin_transport_attested"
                        ),
                    }
                    if isinstance(prompt_event, dict) and not lineage_errors
                    else None
                )
            record["session_receipt_binding"] = binding
            record["session_receipt_binding_sha256"] = _canonical_json_digest(binding)
        record["eligibility_errors"].extend(receipt_errors)
    elif require_session_receipt:
        record["eligibility_errors"].append("missing_agent_session_receipt")
    if require_session_receipt and record["eligibility_errors"]:
        record["attempt_completed"] = False
    if workspace is None:
        return record
    report_path = workspace / "task_result.yaml"
    if not report_path.is_file():
        return record
    report = _load_mapping(report_path, "attempt task result")
    errors = _evaluation_eligibility_errors(workspace, report)
    evaluation_mode = report.get("evaluation_mode")
    agent_session_score_eligible = report.get("agent_session_score_eligible")
    agent_session_terminal_status = report.get("agent_session_terminal_status")
    # The controller began writing these fields after diagnostic replays were
    # made explicit. Keep older sealed campaigns readable, but when either
    # field is present require the complete candidate-scoring contract.
    if evaluation_mode is not None or agent_session_score_eligible is not None:
        if evaluation_mode != "candidate_scoring_v1":
            errors.append("diagnostic_evaluation_not_scoreable")
        if agent_session_score_eligible is not True:
            errors.append("agent_session_not_score_eligible")
    if receipt is not None and receipt.get("schema") in _APEX_RECEIPT_SCHEMAS:
        receipt_terminal_status = receipt.get("terminal_status")
        if receipt_terminal_status != "candidate_ready":
            errors.append("apex_terminal_status_not_candidate_ready")
        if (
            agent_session_terminal_status is not None
            and agent_session_terminal_status != receipt_terminal_status
        ):
            errors.append("apex_report_terminal_status_mismatch")
    try:
        workspace_manifest = _regular_tree_manifest(workspace)
    except CampaignError:
        workspace_manifest = None
        errors.append("unsafe_attempt_workspace_tree")
    if workspace_manifest is not None:
        record["workspace_manifest_sha256"] = _sha256_bytes(
            json.dumps(
                workspace_manifest, sort_keys=True, separators=(",", ":")
            ).encode()
        )
    if not success:
        errors.append("agent_session_or_attempt_failed")
    if require_session_receipt and (receipt is None or receipt.get("session_succeeded") is not True):
        errors.append("agent_session_receipt_not_successful")
        record["attempt_completed"] = False
    errors.extend(record["eligibility_errors"])
    errors = sorted(set(errors))
    optimized_ms = _finite_positive(report.get("best_optimized_execution_time")) or 0.0
    correctness = report.get("pass_correctness") is True
    compilation = report.get("pass_compilation") is True
    consistent = report.get("benchmark_method_consistent") is True
    eligible = not errors
    record.update(
        {
            "central_evaluator_report": str(report_path.relative_to(run_directory)),
            "central_evaluator_report_sha256": _sha256_file(report_path),
            "pass_compilation": compilation,
            "pass_correctness": correctness,
            "optimized_execution_time_ms": optimized_ms,
            "speedup_ratio": float(report.get("speedup_ratio") or 0.0),
            "benchmark_method_consistent": consistent,
            "evaluation_mode": evaluation_mode,
            "agent_session_score_eligible": agent_session_score_eligible,
            "agent_session_terminal_status": agent_session_terminal_status,
            "selection_eligible": eligible,
            "measured_rate_per_ms": 1.0 / optimized_ms if eligible else 0.0,
            "eligibility_errors": errors,
        }
    )
    return record


def _safe_read_only_file(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return (
        path.is_file()
        and not path.is_symlink()
        and metadata.st_nlink == 1
        and not metadata.st_mode & 0o222
    )


def _expected_codex_contract(run_directory: Path) -> dict[str, Any] | None:
    manifest_path = run_directory / "campaign_manifest.yaml"
    if not _safe_read_only_file(manifest_path):
        return None
    try:
        manifest = _load_mapping(manifest_path, "campaign manifest")
    except (CampaignError, OSError, yaml.YAMLError):
        return None
    comparison = manifest.get("comparison_contract")
    if not isinstance(comparison, dict):
        return None
    codex = comparison.get("codex")
    return codex if isinstance(codex, dict) else None


def _expected_comparison_contract_sha256(run_directory: Path) -> str | None:
    manifest_path = run_directory / "campaign_manifest.yaml"
    if not _safe_read_only_file(manifest_path):
        return None
    try:
        manifest = _load_mapping(manifest_path, "campaign manifest")
    except (CampaignError, OSError, yaml.YAMLError):
        return None
    digest = manifest.get("comparison_contract_sha256")
    comparison = manifest.get("comparison_contract")
    if (
        not isinstance(digest, str)
        or not _SHA256.fullmatch(digest)
        or not isinstance(comparison, dict)
    ):
        return None
    observed = _sha256_bytes(
        json.dumps(comparison, sort_keys=True, separators=(",", ":")).encode()
    )
    return digest if digest == observed else None


def _expected_session_receipt_schema(run_directory: Path) -> str | None:
    """Resolve the only receipt schema allowed by the sealed agent manifest."""

    manifest_path = run_directory / "campaign_manifest.yaml"
    if not _safe_read_only_file(manifest_path):
        return None
    try:
        manifest = _load_mapping(manifest_path, "campaign manifest")
    except (CampaignError, OSError, yaml.YAMLError):
        return None
    agent = manifest.get("agent")
    if not isinstance(agent, dict):
        return None
    template = agent.get("template")
    expected = agent.get("session_receipt_schema")
    if template == "apex":
        # Sealed Apex manifests created before receipt v2 had no marker.
        if expected is None:
            return _APEX_RECEIPT_SCHEMA_V1
        return expected if expected in _APEX_RECEIPT_SCHEMAS else None
    if template == "codex":
        # Direct Codex has had one receipt generation; accept marker-less history.
        if expected is None:
            return _CODEX_RECEIPT_SCHEMA
        return expected if expected == _CODEX_RECEIPT_SCHEMA else None
    return None


def _comparison_contract_receipt_errors(
    receipt: dict[str, Any], run_directory: Path, *, prefix: str
) -> list[str]:
    expected = _expected_comparison_contract_sha256(run_directory)
    if expected is None:
        return ["missing_immutable_campaign_comparison_contract"]
    if receipt.get("comparison_contract_sha256") != expected:
        return [f"{prefix}_comparison_contract_digest_mismatch"]
    return []


def _expected_gpu_contract(run_directory: Path) -> dict[str, Any] | None:
    manifest_path = run_directory / "campaign_manifest.yaml"
    if not _safe_read_only_file(manifest_path):
        return None
    try:
        manifest = _load_mapping(manifest_path, "campaign manifest")
    except (CampaignError, OSError, yaml.YAMLError):
        return None
    runtime = manifest.get("runtime")
    gpu = runtime.get("gpu") if isinstance(runtime, dict) else None
    return gpu if isinstance(gpu, dict) else None


def _gpu_receipt_errors(
    receipt: dict[str, Any],
    run_directory: Path,
    *,
    expected_task_name: str | None = None,
) -> list[str]:
    observed = receipt.get("gpu")
    expected = _expected_gpu_contract(run_directory)
    if not isinstance(expected, dict):
        return ["missing_immutable_campaign_gpu_contract"]
    if not isinstance(observed, dict):
        return ["missing_attempt_gpu_boundary_receipt"]
    task_mapping = expected.get("task_mapping")
    expected_task_gpu: str | None = None
    if expected_task_name is not None:
        matching_mappings = [
            mapping
            for mapping in task_mapping
            if isinstance(mapping, dict)
            and mapping.get("task_name") == expected_task_name
        ] if isinstance(task_mapping, list) else []
        if len(matching_mappings) != 1:
            return ["campaign_task_gpu_mapping_missing_or_ambiguous"]
        mapped_gpu = matching_mappings[0].get("assigned_host_gpu_id")
        if not isinstance(mapped_gpu, str):
            return ["campaign_task_gpu_mapping_missing_or_ambiguous"]
        expected_task_gpu = mapped_gpu
    expected_exclusivity = expected.get("exclusivity")
    selected = [
        device
        for device in expected.get("devices", [])
        if isinstance(device, dict)
        and device.get("host_device_id") == observed.get("host_gpu_id")
    ]
    expected_render = selected[0].get("render_nodes") if len(selected) == 1 else None
    runtime_identity = observed.get("runtime_identity")
    runtime_rocm = (
        runtime_identity.get("rocm_smi_identity")
        if isinstance(runtime_identity, dict)
        else None
    )
    runtime_torch = (
        runtime_identity.get("torch") if isinstance(runtime_identity, dict) else None
    )
    if (
        observed.get("policy")
        != "physical_device_boundary_with_host_exclusivity_v1"
        or observed.get("plan_sha256") != expected.get("gpu_boundary_plan_sha256")
        or not isinstance(observed.get("boundary_receipt_sha256"), str)
        or not _SHA256.fullmatch(observed["boundary_receipt_sha256"])
        or not isinstance(expected_exclusivity, dict)
        or observed.get("exclusivity_receipt_sha256")
        != expected_exclusivity.get("sha256")
        or observed.get("exclusivity_verified") is not True
        or (
            expected_task_gpu is not None
            and observed.get("host_gpu_id") != expected_task_gpu
        )
        or len(selected) != 1
        or observed.get("unique_id") != selected[0].get("unique_id")
        or observed.get("allowed_render_nodes") != expected_render
        or not isinstance(runtime_identity, dict)
        or runtime_identity.get("visible_physical_gpu_count") != 1
        or not isinstance(runtime_rocm, dict)
        or runtime_rocm.get("unique_id") != selected[0].get("unique_id")
        or not isinstance(runtime_torch, dict)
        or runtime_torch.get("device_count") != 1
    ):
        return ["attempt_gpu_boundary_or_exclusivity_mismatch"]
    return []


def _validate_session_receipt(
    *,
    receipt_path: Path,
    workspace: Path | None,
    run_directory: Path,
    expected_task_name: str | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Recompute the manifest-selected receipt type before it affects selection."""
    errors: list[str] = []
    if not _safe_read_only_file(receipt_path):
        return None, ["unsafe_or_mutable_direct_codex_session_receipt"]
    try:
        receipt = _load_mapping(receipt_path, "session receipt")
    except (CampaignError, OSError, yaml.YAMLError):
        return None, ["unreadable_direct_codex_session_receipt"]

    expected_schema = _expected_session_receipt_schema(run_directory)
    if expected_schema is None:
        return receipt, ["session_receipt_manifest_schema_contract_invalid"]
    if receipt.get("schema") != expected_schema:
        mismatch = (
            "apex_receipt_schema_generation_mismatch"
            if expected_schema in _APEX_RECEIPT_SCHEMAS
            else "direct_codex_receipt_schema_generation_mismatch"
        )
        return receipt, [mismatch]

    if expected_schema in _APEX_RECEIPT_SCHEMAS:
        return _validate_apex_session_receipt(
            receipt=receipt,
            receipt_path=receipt_path,
            workspace=workspace,
            run_directory=run_directory,
            expected_task_name=expected_task_name,
            expected_receipt_schema=expected_schema,
        )

    errors.extend(
        _comparison_contract_receipt_errors(
            receipt, run_directory, prefix="direct_codex"
        )
    )
    errors.extend(
        _gpu_receipt_errors(
            receipt,
            run_directory,
            expected_task_name=expected_task_name,
        )
    )
    if not isinstance(receipt.get("session_succeeded"), bool):
        errors.append("direct_codex_receipt_invalid_session_status")
    if receipt.get("session_succeeded") is True:
        if receipt.get("timed_out") is not False or receipt.get("exit_code") != 0:
            errors.append("direct_codex_receipt_success_status_inconsistent")
        thread_id = receipt.get("thread_id")
        if not isinstance(thread_id, str) or not thread_id.strip():
            errors.append("direct_codex_receipt_missing_thread_id")

    cleanup = receipt.get("process_group_cleanup")
    if (
        not isinstance(cleanup, dict)
        or cleanup.get("verification_performed") is not True
        or cleanup.get("verified_absent") is not True
    ):
        errors.append("direct_codex_process_group_not_verified_absent")
    capture = receipt.get("capture")
    if (
        not isinstance(capture, dict)
        or capture.get("readers_completed") is not True
        or capture.get("errors") != []
    ):
        errors.append("direct_codex_capture_incomplete")
    else:
        for stream_name in ("stdout", "stderr"):
            stream = capture.get(stream_name)
            if (
                not isinstance(stream, dict)
                or stream.get("limit_bytes") != 16 * 1024 * 1024
                or not isinstance(stream.get("retained_bytes"), int)
                or not 0 <= stream["retained_bytes"] <= stream["limit_bytes"]
                or stream.get("discarded_bytes") != 0
                or stream.get("truncated") is not False
            ):
                errors.append(f"direct_codex_{stream_name}_capture_bound_invalid")

    turn_budget = receipt.get("turn_budget")
    if (
        not isinstance(turn_budget, dict)
        or turn_budget.get("policy") != TURN_POLICY
        or turn_budget.get("max_turns") != FORMAL_MATCHED_MAX_TURNS
        or not isinstance(turn_budget.get("observed_turns"), int)
        or not 1 <= turn_budget["observed_turns"] <= FORMAL_MATCHED_MAX_TURNS
        or turn_budget.get("budget_exceeded") is not False
        or turn_budget.get("enforcement_failed") is not False
        or turn_budget.get("stop_reason") is not None
    ):
        errors.append("direct_codex_turn_budget_invalid")

    workspace_integrity = receipt.get("workspace_integrity")
    if (
        not isinstance(workspace_integrity, dict)
        or workspace_integrity.get("policy") != "declared_source_only_sanitized_v1"
        or workspace_integrity.get("passed") is not True
        or workspace_integrity.get("errors") != []
    ):
        errors.append("direct_codex_workspace_integrity_invalid")

    try:
        effective_timeout = float(receipt.get("effective_timeout_seconds"))
    except (TypeError, ValueError):
        effective_timeout = 0.0
    if not math.isfinite(effective_timeout) or not 0 < effective_timeout <= 3600:
        errors.append("direct_codex_effective_timeout_invalid")

    usage = receipt.get("aggregated_usage")
    if not isinstance(usage, dict) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in usage.values()
    ):
        errors.append("direct_codex_usage_invalid")

    expected_codex = _expected_codex_contract(run_directory)
    observed_codex = receipt.get("codex")
    if expected_codex is None:
        errors.append("missing_immutable_campaign_codex_contract")
    elif not isinstance(observed_codex, dict):
        errors.append("direct_codex_identity_missing")
    else:
        comparisons = {
            "binary_sha256": "codex_binary_sha256",
            "version": "codex_version",
            "model": "model",
            "effort": "effort",
        }
        if any(
            observed_codex.get(receipt_key) != expected_codex.get(contract_key)
            for receipt_key, contract_key in comparisons.items()
        ):
            errors.append("direct_codex_identity_contract_mismatch")
        if (
            expected_codex.get("max_turns") != FORMAL_MATCHED_MAX_TURNS
            or expected_codex.get("turn_policy") != TURN_POLICY
            or expected_codex.get("structured_stream_output_limit_bytes")
            != 16 * 1024 * 1024
        ):
            errors.append("direct_codex_immutable_budget_contract_mismatch")

    invocation = receipt.get("invocation")
    if not isinstance(invocation, dict):
        errors.append("direct_codex_invocation_missing")
    else:
        expected_isolation = (
            expected_codex.get("isolation") if isinstance(expected_codex, dict) else None
        )
        if invocation.get("isolation") != expected_isolation:
            errors.append("direct_codex_isolation_contract_mismatch")
        argv = invocation.get("argv_without_prompt")
        required_flags = {
            "--strict-config",
            "--ignore-user-config",
            "--ignore-rules",
            "--ephemeral",
        }
        if not isinstance(argv, list) or not required_flags.issubset(set(argv)):
            errors.append("direct_codex_invocation_flags_missing")
        if not isinstance(invocation.get("prompt_sha256"), str) or not _SHA256.fullmatch(
            invocation["prompt_sha256"]
        ):
            errors.append("direct_codex_prompt_digest_invalid")
        if (
            invocation.get("max_turns") != FORMAL_MATCHED_MAX_TURNS
            or invocation.get("turn_policy") != TURN_POLICY
            or invocation.get("structured_stream_output_limit_bytes")
            != 16 * 1024 * 1024
        ):
            errors.append("direct_codex_budget_invocation_mismatch")
        editable = invocation.get("editable_files")
        if (
            not isinstance(editable, list)
            or not editable
            or any(not isinstance(path, str) or not path for path in editable)
            or not isinstance(workspace_integrity, dict)
            or workspace_integrity.get("editable_files") != editable
        ):
            errors.append("direct_codex_editable_scope_mismatch")

    artifact_dir = receipt_path.parent / f".{receipt_path.stem}.artifacts"
    try:
        artifact_metadata = artifact_dir.lstat()
    except OSError:
        artifact_metadata = None
    if (
        artifact_metadata is None
        or not artifact_dir.is_dir()
        or artifact_dir.is_symlink()
        or artifact_metadata.st_mode & 0o222
    ):
        errors.append("direct_codex_artifact_directory_unsafe")

    artifacts = receipt.get("artifacts")
    expected_artifacts = {
        "rendered_prompt": artifact_dir / "rendered_prompt.txt",
        "raw_stdout": artifact_dir / "raw_stdout.jsonl",
        "raw_stderr": artifact_dir / "raw_stderr.txt",
        "formatted_transcript": artifact_dir / "formatted_transcript.txt",
        "workspace_before_manifest": artifact_dir / "workspace_before_manifest.json",
        "workspace_after_manifest": artifact_dir / "workspace_after_manifest.json",
    }
    if not isinstance(artifacts, dict) or set(artifacts) != set(expected_artifacts):
        errors.append("direct_codex_artifact_set_mismatch")
    else:
        workspace_root = workspace.resolve() if workspace is not None else None
        for name, expected_path in expected_artifacts.items():
            evidence = artifacts.get(name)
            if not isinstance(evidence, dict):
                errors.append(f"direct_codex_{name}_metadata_invalid")
                continue
            raw_path = evidence.get("path")
            if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
                errors.append(f"direct_codex_{name}_path_invalid")
                continue
            artifact_path = Path(raw_path)
            try:
                resolved_path = artifact_path.resolve(strict=True)
                resolved_expected = expected_path.resolve(strict=True)
            except OSError:
                errors.append(f"direct_codex_{name}_missing")
                continue
            if resolved_path != resolved_expected:
                errors.append(f"direct_codex_{name}_path_mismatch")
                continue
            if workspace_root is not None:
                try:
                    resolved_path.relative_to(workspace_root)
                except ValueError:
                    pass
                else:
                    errors.append(f"direct_codex_{name}_inside_scored_workspace")
            if not _safe_read_only_file(resolved_path):
                errors.append(f"direct_codex_{name}_unsafe_or_mutable")
                continue
            observed_hash = _sha256_file(resolved_path)
            if evidence.get("sha256") != observed_hash:
                errors.append(f"direct_codex_{name}_hash_mismatch")
            if evidence.get("size_bytes") != resolved_path.stat().st_size:
                errors.append(f"direct_codex_{name}_size_mismatch")

        rendered_prompt = expected_artifacts["rendered_prompt"]
        if _safe_read_only_file(rendered_prompt):
            prompt_sha256 = _sha256_file(rendered_prompt)
            if (
                not isinstance(invocation, dict)
                or invocation.get("prompt_sha256") != prompt_sha256
            ):
                errors.append("direct_codex_prompt_digest_mismatch")

        try:
            before_manifest = _load_mapping(
                expected_artifacts["workspace_before_manifest"],
                "direct Codex before-workspace manifest",
            )
            after_manifest = _load_mapping(
                expected_artifacts["workspace_after_manifest"],
                "direct Codex after-workspace manifest",
            )
        except (CampaignError, OSError, yaml.YAMLError):
            errors.append("direct_codex_workspace_manifests_unreadable")
        else:
            before_digest = _canonical_json_digest(before_manifest)
            after_digest = _canonical_json_digest(after_manifest)
            final_changes = (
                workspace_integrity.get("final_changes")
                if isinstance(workspace_integrity, dict)
                else None
            )
            sanitization = (
                workspace_integrity.get("sanitization")
                if isinstance(workspace_integrity, dict)
                else None
            )
            editable = set(
                workspace_integrity.get("editable_files", [])
                if isinstance(workspace_integrity, dict)
                else []
            )
            raw_changes = (
                workspace_integrity.get("raw_changes")
                if isinstance(workspace_integrity, dict)
                else None
            )
            if (
                not isinstance(raw_changes, dict)
                or workspace_integrity.get("raw_manifest_error") is not None
                or not isinstance(workspace_integrity.get("raw_after_manifest_sha256"), str)
                or not _SHA256.fullmatch(
                    workspace_integrity["raw_after_manifest_sha256"]
                )
                or raw_changes.get("after_manifest_sha256")
                != workspace_integrity.get("raw_after_manifest_sha256")
            ):
                errors.append("direct_codex_raw_workspace_diff_invalid")
            if (
                not isinstance(final_changes, dict)
                or final_changes.get("before_manifest_sha256") != before_digest
                or final_changes.get("after_manifest_sha256") != after_digest
                or final_changes.get("created_files") != []
                or final_changes.get("deleted_files") != []
                or final_changes.get("unauthorized_changed_files") != []
                or final_changes.get("editable_mode_changes") != []
                or not set(final_changes.get("changed_files", [])).issubset(editable)
            ):
                errors.append("direct_codex_sanitized_manifest_contract_mismatch")
            if (
                not isinstance(sanitization, dict)
                or sanitization.get("performed") is not True
                or sanitization.get("candidate_retained") is not True
                or sanitization.get("baseline_restored") is not True
            ):
                errors.append("direct_codex_workspace_sanitization_unverified")

    return receipt, sorted(set(errors))


def _canonical_json_digest(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _validate_receipt_artifacts(
    *,
    receipt: dict[str, Any],
    receipt_path: Path,
    workspace: Path | None,
    expected_names: dict[str, str],
    prefix: str,
) -> tuple[dict[str, Path], list[str]]:
    errors: list[str] = []
    artifact_dir = receipt_path.parent / f".{receipt_path.stem}.artifacts"
    try:
        metadata = artifact_dir.lstat()
    except OSError:
        metadata = None
    if (
        metadata is None
        or not artifact_dir.is_dir()
        or artifact_dir.is_symlink()
        or metadata.st_mode & 0o222
    ):
        return {}, [f"{prefix}_artifact_directory_unsafe"]
    raw_artifacts = receipt.get("artifacts")
    if not isinstance(raw_artifacts, dict) or set(raw_artifacts) != set(expected_names):
        return {}, [f"{prefix}_artifact_set_mismatch"]
    verified: dict[str, Path] = {}
    workspace_root = workspace.resolve() if workspace is not None else None
    for name, filename in expected_names.items():
        evidence = raw_artifacts.get(name)
        expected = artifact_dir / filename
        if not isinstance(evidence, dict):
            errors.append(f"{prefix}_{name}_metadata_invalid")
            continue
        raw_path = evidence.get("path")
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
            errors.append(f"{prefix}_{name}_path_invalid")
            continue
        try:
            path = Path(raw_path).resolve(strict=True)
            if path != expected.resolve(strict=True):
                errors.append(f"{prefix}_{name}_path_mismatch")
                continue
        except OSError:
            errors.append(f"{prefix}_{name}_missing")
            continue
        if workspace_root is not None:
            try:
                path.relative_to(workspace_root)
            except ValueError:
                pass
            else:
                errors.append(f"{prefix}_{name}_inside_scored_workspace")
        if not _safe_read_only_file(path):
            errors.append(f"{prefix}_{name}_unsafe_or_mutable")
            continue
        if evidence.get("sha256") != _sha256_file(path):
            errors.append(f"{prefix}_{name}_hash_mismatch")
        if evidence.get("size_bytes") != path.stat().st_size:
            errors.append(f"{prefix}_{name}_size_mismatch")
        verified[name] = path
    return verified, errors


def _validate_apex_journal_snapshot(
    *,
    journal_path: Path,
    result: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    try:
        connection = sqlite3.connect(
            f"file:{journal_path.as_posix()}?mode=ro&immutable=1", uri=True
        )
        connection.row_factory = sqlite3.Row
        event_rows = connection.execute(
            "SELECT * FROM events WHERE run_id = ? ORDER BY sequence",
            (result.get("run_id"),),
        ).fetchall()
        transaction_rows = connection.execute(
            "SELECT * FROM transactions ORDER BY first_sequence"
        ).fetchall()
    except sqlite3.Error:
        return [], ["apex_event_journal_unreadable"]
    finally:
        try:
            connection.close()
        except UnboundLocalError:
            pass
    if not event_rows:
        return [], ["apex_event_journal_empty"]
    events: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    previous_id: str | None = None
    previous_sequence = 0
    for row in event_rows:
        try:
            payload = json.loads(str(row["payload_json"]))
            event = {
                "sequence": int(row["sequence"]),
                "event_id": str(row["event_id"]),
                "run_id": str(row["run_id"]),
                "event_type": str(row["event_type"]),
                "payload": payload,
                "parent_event_id": row["parent_event_id"],
                "idempotency_key": str(row["idempotency_key"]),
                "transaction_id": str(row["transaction_id"]),
                "created_at_ns": int(row["created_at_ns"]),
                "checksum": str(row["checksum"]),
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            errors.append("apex_event_journal_malformed")
            continue
        material = {key: value for key, value in event.items() if key != "checksum"}
        if event["checksum"] != _canonical_json_digest(material):
            errors.append("apex_event_checksum_mismatch")
        if event["sequence"] <= previous_sequence or event["parent_event_id"] != previous_id:
            errors.append("apex_event_parent_or_sequence_mismatch")
        previous_id = event["event_id"]
        previous_sequence = event["sequence"]
        events.append(event)
        grouped.setdefault(event["transaction_id"], []).append(event)
    transactions = {str(row["transaction_id"]): row for row in transaction_rows}
    if set(transactions) != set(grouped):
        errors.append("apex_transaction_set_mismatch")
    for transaction_id, tx_events in grouped.items():
        row = transactions.get(transaction_id)
        if row is None:
            continue
        expected = _canonical_json_digest(
            {
                "transaction_id": transaction_id,
                "event_checksums": [event["checksum"] for event in tx_events],
            }
        )
        if (
            int(row["first_sequence"]) != tx_events[0]["sequence"]
            or int(row["last_sequence"]) != tx_events[-1]["sequence"]
            or int(row["event_count"]) != len(tx_events)
            or str(row["checksum"]) != expected
        ):
            errors.append("apex_transaction_receipt_mismatch")
    head_ref = result.get("event_journal_ref")
    if not isinstance(head_ref, dict) or not events or (
        head_ref.get("head_event_id") != events[-1]["event_id"]
        or head_ref.get("head_checksum") != events[-1]["checksum"]
    ):
        errors.append("apex_result_journal_head_mismatch")
    return events, errors


def _apex_instruction_adaptation_errors(
    *,
    receipt: dict[str, Any],
    task_spec: dict[str, Any],
    original_prompt_path: Path,
) -> list[str]:
    """Recompute the caller-owned original/adapted prompt binding."""
    errors: list[str] = []
    adaptation = receipt.get("instruction_adaptation")
    if (
        not isinstance(adaptation, dict)
        or adaptation.get("schema") != "aka.apex-instruction-adaptation/v1"
        or task_spec.get("instruction_adaptation") != adaptation
    ):
        return ["apex_instruction_adaptation_contract_mismatch"]
    try:
        original_bytes = original_prompt_path.read_bytes()
        original_text = original_bytes.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return ["apex_original_prompt_unreadable"]
    original_summary = {
        "characters": len(original_text),
        "bytes": len(original_bytes),
        "sha256": _sha256_bytes(original_bytes),
    }
    if adaptation.get("original") != original_summary:
        errors.append("apex_original_prompt_digest_mismatch")
    instructions = task_spec.get("instructions")
    if not isinstance(instructions, str):
        errors.append("apex_adapted_prompt_digest_mismatch")
    else:
        adapted_bytes = instructions.encode("utf-8")
        adapted_summary = {
            "characters": len(instructions),
            "bytes": len(adapted_bytes),
            "sha256": _sha256_bytes(adapted_bytes),
        }
        if adaptation.get("adapted") != adapted_summary:
            errors.append("apex_adapted_prompt_digest_mismatch")
    return errors


def _apex_run_control_errors(task_spec: dict[str, Any]) -> list[str]:
    """Validate the caller-owned formal budget and verifier command contract."""

    control = task_spec.get("caller_run_control")
    commands = task_spec.get("commands")
    if not isinstance(control, dict) or not isinstance(commands, dict):
        return ["apex_caller_run_control_invalid"]
    turn_budget = control.get("structured_turn_budget")
    interpreter = control.get("python_interpreter")
    verifier_argv = control.get("verifier_argv")
    if (
        control.get("schema") != "aka.apex-caller-run-control/v1"
        or control.get("deliverable_versions") != 1
        or control.get("candidate_persistence")
        != "leave_best_source_before_budget_boundary"
        or turn_budget
        != {
            "policy": TURN_POLICY,
            "max_turns": FORMAL_MATCHED_MAX_TURNS,
            "counting": "assistant_message_and_tool_call_start_each_count_once",
        }
        or not isinstance(interpreter, dict)
        or interpreter.get("environment_variable")
        != "AGENT_KERNEL_ARENA_PYTHON"
        or not isinstance(interpreter.get("path"), str)
        or not Path(interpreter["path"]).is_absolute()
        or not isinstance(interpreter.get("resolved_path"), str)
        or not Path(interpreter["resolved_path"]).is_absolute()
        or not isinstance(interpreter.get("sha256"), str)
        or not _SHA256.fullmatch(interpreter["sha256"])
        or not isinstance(verifier_argv, dict)
    ):
        return ["apex_caller_run_control_invalid"]
    for phase in ("compile", "correctness", "performance"):
        command = commands.get(phase)
        argv = command.get("argv") if isinstance(command, dict) else None
        if (
            not isinstance(argv, list)
            or not argv
            or verifier_argv.get(phase) != argv
            or argv[0] != interpreter["path"]
        ):
            return ["apex_caller_run_control_invalid"]
    if set(verifier_argv) != {"compile", "correctness", "performance"}:
        return ["apex_caller_run_control_invalid"]
    instructions = task_spec.get("instructions")
    expected_suffix = f"\n\n{render_apex_run_control(control)}"
    if not isinstance(instructions, str) or not instructions.endswith(expected_suffix):
        return ["apex_caller_run_control_invalid"]
    return []


def _apex_turn_evidence_errors(
    *,
    transcript: dict[str, Any],
    payload: dict[str, Any],
    task_spec: dict[str, Any],
    budget_exceeded: bool,
) -> list[str]:
    semantic_events = transcript.get("semantic_events")
    budget = transcript.get("budget")
    if not isinstance(semantic_events, list) or not isinstance(budget, dict):
        return ["apex_agent_turn_evidence_invalid"]
    message_count = sum(
        1
        for event in semantic_events
        if isinstance(event, dict) and event.get("kind") == "agent_message"
    )
    tool_call_count = sum(
        1
        for event in semantic_events
        if isinstance(event, dict) and event.get("kind") == "tool_called"
    )
    observed_turns = message_count + tool_call_count
    max_turns = task_spec.get("budget", {}).get("max_turns")
    if (
        payload.get("observed_turns") != observed_turns
        or payload.get("message_event_count") != message_count
        or payload.get("tool_call_event_count") != tool_call_count
        or payload.get("semantic_event_count") != len(semantic_events)
        or budget.get("turn_policy") != TURN_POLICY
        or budget.get("max_turns") != max_turns
        or budget.get("observed_turns") != observed_turns
        or budget.get("exceeded") is not budget_exceeded
        or budget.get("enforcement_failed") is not False
        or budget.get("reason") != payload.get("budget_reason")
        or (
            not budget_exceeded
            and (
                type(max_turns) is not int
                or not 1 <= observed_turns <= max_turns
            )
        )
    ):
        return ["apex_agent_turn_evidence_invalid"]
    return []


def _apex_prompt_event_errors(
    *,
    events: list[dict[str, Any]],
    agent_event: dict[str, Any],
    lineage: dict[str, Any] | None,
    invocation: dict[str, Any] | None,
    agent_prompt_path: Path,
    expected_objective: object,
) -> list[str]:
    prompt_events = [event for event in events if event["event_type"] == "prompt_sent"]
    if len(prompt_events) != 1:
        return ["apex_prompt_event_binding_mismatch"]
    prompt_event = prompt_events[0]
    bindings = prompt_event["payload"].get("artifacts")
    prompt_bindings = [
        binding
        for binding in bindings
        if isinstance(binding, dict) and binding.get("role") == "prompt"
    ] if isinstance(bindings, list) else []
    prompt_receipt = (
        prompt_bindings[0].get("receipt") if len(prompt_bindings) == 1 else None
    )
    summary = lineage.get("prompt_event") if isinstance(lineage, dict) else None
    errors: list[str] = []
    if (
        not isinstance(prompt_receipt, dict)
        or not isinstance(summary, dict)
        or summary.get("binding") != "apex.prompt_sent_event_cas/v1"
        or summary.get("event_id") != prompt_event["event_id"]
        or summary.get("sha256") != prompt_receipt.get("digest")
        or summary.get("size_bytes") != prompt_receipt.get("size")
        or summary.get("sha256") != _sha256_file(agent_prompt_path)
        or summary.get("size_bytes") != agent_prompt_path.stat().st_size
        or not isinstance(summary.get("artifact_path"), str)
        or not Path(summary["artifact_path"]).is_absolute()
        or summary.get("stdin_transport_attested") is not False
        or not isinstance(summary.get("sha256"), str)
        or not _SHA256.fullmatch(summary["sha256"])
        or isinstance(summary.get("size_bytes"), bool)
        or not isinstance(summary.get("size_bytes"), int)
        or summary["size_bytes"] < 0
        or prompt_event["sequence"] >= agent_event["sequence"]
        or agent_event["parent_event_id"] != prompt_event["event_id"]
        or prompt_event["payload"].get("attempt_id")
        != agent_event["payload"].get("attempt_id")
        or not isinstance(invocation, dict)
        or invocation.get("prompt_transport") != "stdin"
    ):
        errors.append("apex_prompt_event_binding_mismatch")
    try:
        prompt_bytes = agent_prompt_path.read_bytes()
    except OSError:
        prompt_bytes = b""
    if not context_packet_objective_matches(prompt_bytes, expected_objective):
        errors.append("apex_prompt_objective_binding_mismatch")
    return errors


def _validate_apex_session_receipt(
    *,
    receipt: dict[str, Any],
    receipt_path: Path,
    workspace: Path | None,
    run_directory: Path,
    expected_receipt_schema: str,
    expected_task_name: str | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    errors.extend(
        _comparison_contract_receipt_errors(receipt, run_directory, prefix="apex")
    )
    errors.extend(
        _gpu_receipt_errors(
            receipt,
            run_directory,
            expected_task_name=expected_task_name,
        )
    )
    receipt_schema = receipt.get("schema")
    if receipt_schema not in _APEX_RECEIPT_SCHEMAS:
        errors.append("apex_receipt_schema_mismatch")
    if receipt_schema != expected_receipt_schema:
        errors.append("apex_receipt_schema_generation_mismatch")
    terminal_status = receipt.get("terminal_status")
    session_succeeded = receipt.get("session_succeeded")
    budget_exhausted = (
        terminal_status == "budget_exhausted" and session_succeeded is False
    )
    successful_terminal = (
        terminal_status in {"candidate_ready", "no_gain"}
        and session_succeeded is True
    )
    if session_succeeded is not True:
        errors.append("apex_session_not_successful")
    exit_code = receipt.get("exit_code")
    status_consistent = (
        successful_terminal
        and receipt.get("timed_out") is False
        and exit_code == 0
    ) or (
        budget_exhausted
        and receipt.get("timed_out") is False
        and isinstance(exit_code, int)
        and not isinstance(exit_code, bool)
        and exit_code != 0
    )
    if not status_consistent:
        errors.append("apex_receipt_success_status_inconsistent")
    cleanup = receipt.get("process_group_cleanup")
    if (
        not isinstance(cleanup, dict)
        or cleanup.get("verification_performed") is not True
        or cleanup.get("verified_absent") is not True
    ):
        errors.append("apex_process_group_not_verified_absent")
    capture = receipt.get("capture")
    if (
        not isinstance(capture, dict)
        or capture.get("readers_completed") is not True
        or capture.get("errors") != []
    ):
        errors.append("apex_capture_incomplete")
    budgets = receipt.get("budgets")
    expected_budgets = {
        "inner_agent_timeout_seconds": 3600,
        "apex_internal_allowance_seconds": 3600,
        "outer_timeout_seconds": 7200.0,
    }
    if not isinstance(budgets, dict) or any(
        budgets.get(key) != value for key, value in expected_budgets.items()
    ):
        errors.append("apex_independent_budget_contract_mismatch")
    try:
        effective_outer = float((budgets or {}).get("effective_outer_timeout_seconds"))
    except (TypeError, ValueError):
        effective_outer = 0.0
    if not math.isfinite(effective_outer) or not 0 < effective_outer <= 7200:
        errors.append("apex_effective_outer_timeout_invalid")

    expected_codex = _expected_codex_contract(run_directory)
    observed_codex = receipt.get("codex")
    if not isinstance(expected_codex, dict):
        errors.append("missing_immutable_campaign_codex_contract")
    elif not isinstance(observed_codex, dict) or any(
        observed_codex.get(receipt_key) != expected_codex.get(contract_key)
        for receipt_key, contract_key in {
            "binary_sha256": "codex_binary_sha256",
            "version": "codex_version",
            "model": "model",
            "effort": "effort",
        }.items()
    ):
        errors.append("apex_codex_identity_contract_mismatch")
    if isinstance(expected_codex, dict) and (
        expected_codex.get("max_turns") != FORMAL_MATCHED_MAX_TURNS
        or expected_codex.get("turn_policy") != TURN_POLICY
    ):
        errors.append("apex_immutable_budget_contract_mismatch")
    if (
        isinstance(expected_codex, dict)
        and receipt.get("outer_isolation") != expected_codex.get("isolation")
    ):
        errors.append("apex_outer_isolation_contract_mismatch")
    workspace_integrity = receipt.get("workspace_integrity")
    if (
        not isinstance(workspace_integrity, dict)
        or workspace_integrity.get("policy")
        != "read_only_until_adapter_bundle_apply_v1"
        or workspace_integrity.get("pre_apply_unchanged") is not True
        or not isinstance(workspace_integrity.get("baseline_manifest_sha256"), str)
        or not _SHA256.fullmatch(workspace_integrity["baseline_manifest_sha256"])
        or workspace_integrity.get("pre_apply_manifest_sha256")
        != workspace_integrity.get("baseline_manifest_sha256")
    ):
        errors.append("apex_workspace_pre_apply_integrity_invalid")

    task_spec_contract = receipt.get("task_spec_contract")
    contract_path: Path | None = None
    if (
        not isinstance(task_spec_contract, dict)
        or task_spec_contract.get("policy")
        != "prelaunch_read_only_sibling_bind_v1"
        or not isinstance(task_spec_contract.get("path"), str)
        or not isinstance(task_spec_contract.get("sha256"), str)
        or not _SHA256.fullmatch(task_spec_contract["sha256"])
        or type(task_spec_contract.get("size_bytes")) is not int
        or task_spec_contract["size_bytes"] <= 0
        or task_spec_contract.get("file_mode") != "0444"
        or task_spec_contract.get("directory_mode") != "0555"
        or task_spec_contract.get("read_only_bind") is not True
        or task_spec_contract.get("postlaunch_unchanged") is not True
    ):
        errors.append("apex_task_spec_prelaunch_contract_invalid")
    else:
        contract_path = Path(task_spec_contract["path"])
        try:
            contract_resolved = contract_path.resolve(strict=True)
            contract_resolved.relative_to(receipt_path.parent.resolve(strict=True))
            parent_metadata = contract_resolved.parent.lstat()
            if (
                contract_resolved.name != "task_spec.json"
                or not contract_resolved.parent.name.endswith(".contract")
                or not _safe_read_only_file(contract_resolved)
                or contract_resolved.parent.is_symlink()
                or not stat.S_ISDIR(parent_metadata.st_mode)
                or stat.S_IMODE(parent_metadata.st_mode) != 0o555
                or _sha256_file(contract_resolved)
                != task_spec_contract["sha256"]
                or contract_resolved.stat().st_size
                != task_spec_contract["size_bytes"]
            ):
                errors.append("apex_task_spec_prelaunch_contract_invalid")
        except (OSError, ValueError):
            errors.append("apex_task_spec_prelaunch_contract_invalid")

    new_prompt_receipt = expected_receipt_schema == _APEX_RECEIPT_SCHEMA_V2
    expected_artifacts = {
        "task_spec": "task_spec.json",
        "apex_stdout": "apex_stdout.txt",
        "apex_stderr": "apex_stderr.txt",
        "apex_result": "apex_result.json",
        "event_journal": "event_journal.sqlite",
        "agent_transcript": "agent_transcript.json",
    }
    if new_prompt_receipt:
        expected_artifacts["original_arena_prompt"] = "original_arena_prompt.txt"
        expected_artifacts["agent_prompt"] = "agent_prompt.txt"
    artifacts, artifact_errors = _validate_receipt_artifacts(
        receipt=receipt,
        receipt_path=receipt_path,
        workspace=workspace,
        expected_names=expected_artifacts,
        prefix="apex",
    )
    errors.extend(artifact_errors)
    if set(artifacts) != set(expected_artifacts):
        return receipt, sorted(set(errors))
    try:
        task_spec = _load_mapping(artifacts["task_spec"], "Apex TaskSpec receipt")
        result = _load_mapping(artifacts["apex_result"], "Apex result receipt")
        transcript = _load_mapping(
            artifacts["agent_transcript"], "Apex agent transcript receipt"
        )
    except CampaignError:
        return receipt, sorted(set(errors + ["apex_lineage_json_unreadable"]))
    if new_prompt_receipt:
        errors.extend(
            _apex_instruction_adaptation_errors(
                receipt=receipt,
                task_spec=task_spec,
                original_prompt_path=artifacts["original_arena_prompt"],
            )
        )
        errors.extend(_apex_run_control_errors(task_spec))
    if receipt.get("task_spec_sha256") != _sha256_file(artifacts["task_spec"]):
        errors.append("apex_task_spec_digest_mismatch")
    if isinstance(task_spec_contract, dict) and (
        task_spec_contract.get("sha256") != _sha256_file(artifacts["task_spec"])
        or task_spec_contract.get("size_bytes")
        != artifacts["task_spec"].stat().st_size
    ):
        errors.append("apex_task_spec_prelaunch_artifact_mismatch")
    task_budget = task_spec.get("budget")
    if (
        not isinstance(task_budget, dict)
        or task_budget.get("max_iterations") != 1
        or task_budget.get("max_turns") != FORMAL_MATCHED_MAX_TURNS
        or task_budget.get("timeout_seconds") != 3600
    ):
        errors.append("apex_task_spec_budget_contract_mismatch")
    lineage = receipt.get("lineage")
    if not isinstance(lineage, dict) or lineage.get("result_sha256") != _sha256_file(
        artifacts["apex_result"]
    ):
        errors.append("apex_result_digest_mismatch")
    status = result.get("status")
    verdict = {
        "candidate_ready": "keep",
        "no_gain": "revert",
        "budget_exhausted": "reject",
    }.get(status)
    result_contract_invalid = (
        verdict is None
        or receipt.get("terminal_status") != status
        or result.get("internal_verdict") != verdict
        or result.get("task_id") != task_spec.get("task_id")
        or result.get("baseline_lock", {}).get("file_hashes")
        != task_spec.get("baseline", {}).get("file_hashes")
    )
    failed_terminal = status == "budget_exhausted"
    if failed_terminal:
        result_error = result.get("error")
        result_contract_invalid = result_contract_invalid or (
            result.get("reason_code") != "agent_turn_budget_exceeded"
            or not isinstance(result_error, dict)
            or result_error.get("reason_code") != "agent_turn_budget_exceeded"
        )
    else:
        result_contract_invalid = result_contract_invalid or result.get("error") is not None
    if result_contract_invalid:
        errors.append("apex_terminal_result_contract_mismatch")
    no_candidate_payload = (
        result.get("bundle_path") is not None
        or result.get("bundle_digest") is not None
        or result.get("changed_files") != []
    )
    if status == "no_gain" and no_candidate_payload:
        errors.append("apex_no_gain_receipt_invalid")
    elif status == "budget_exhausted" and no_candidate_payload:
        errors.append("apex_budget_exhausted_receipt_invalid")

    events, journal_errors = _validate_apex_journal_snapshot(
        journal_path=artifacts["event_journal"], result=result
    )
    errors.extend(journal_errors)
    completed = [event for event in events if event["event_type"] == "agent_completed"]
    failed = [event for event in events if event["event_type"] == "agent_failed"]
    expected_agent_events = failed if failed_terminal else completed
    unexpected_agent_events = completed if failed_terminal else failed
    agent_event = expected_agent_events[0] if len(expected_agent_events) == 1 else None
    if agent_event is None or unexpected_agent_events:
        errors.append("apex_agent_completion_event_invalid")
    else:
        payload = agent_event["payload"]
        invocation = payload.get("invocation")
        outcome_invalid = (
            payload.get("backend") != "codex"
            or payload.get("model") != task_spec.get("agent_options", {}).get("model")
            or payload.get("effort") != task_spec.get("agent_options", {}).get("effort")
            or payload.get("timed_out") is not False
            or payload.get("budget_enforcement_failed") is not False
            or invocation != receipt.get("invocation")
        )
        if failed_terminal:
            observed_turns = payload.get("observed_turns")
            outcome_invalid = outcome_invalid or (
                type(payload.get("exit_code")) is not int
                or payload.get("budget_exceeded") is not True
                or not budget_stop_reason_matches(
                    reason=payload.get("budget_reason"),
                    observed_turns=observed_turns,
                    max_turns=FORMAL_MATCHED_MAX_TURNS,
                )
            )
        else:
            outcome_invalid = outcome_invalid or (
                payload.get("exit_code") != 0
                or payload.get("budget_exceeded") is not False
                or (new_prompt_receipt and payload.get("budget_reason") is not None)
            )
        if outcome_invalid:
            errors.append("apex_agent_completion_receipt_mismatch")
        if (
            transcript.get("schema") != "apex.agent-transcript/v1"
            or transcript.get("backend") != "codex"
            or transcript.get("model") != payload.get("model")
            or transcript.get("effort") != payload.get("effort")
            or transcript.get("invocation") != invocation
            or not isinstance(lineage, dict)
            or lineage.get("transcript_sha256")
            != _sha256_file(artifacts["agent_transcript"])
        ):
            errors.append("apex_agent_transcript_receipt_mismatch")
        expected_isolation = (
            expected_codex.get("isolation")
            if isinstance(expected_codex, dict)
            else {}
        )
        inner_isolation = invocation.get("isolation") if isinstance(invocation, dict) else None
        common_isolation = {
            key: value
            for key, value in (inner_isolation or {}).items()
            if key in expected_isolation and key != "mount_scope"
        }
        expected_common = {
            key: value for key, value in expected_isolation.items() if key != "mount_scope"
        }
        if common_isolation != expected_common:
            errors.append("apex_inner_codex_isolation_contract_mismatch")
        argv = invocation.get("argv") if isinstance(invocation, dict) else None
        if (
            not isinstance(invocation, dict)
            or invocation.get("schema") != "apex.agent-invocation/v1"
            or invocation.get("cli_name") != "codex"
            or invocation.get("cli_version")
            != (expected_codex or {}).get("codex_version")
            or invocation.get("entrypoint_sha256")
            != (expected_codex or {}).get("codex_binary_sha256")
            or invocation.get("max_turns") != FORMAL_MATCHED_MAX_TURNS
            or invocation.get("turn_policy") != TURN_POLICY
            or not isinstance(argv, list)
            or not {
                "--strict-config",
                "--ignore-user-config",
                "--ignore-rules",
                "--ephemeral",
            }.issubset(set(argv or []))
        ):
            errors.append("apex_inner_codex_invocation_contract_mismatch")
        if new_prompt_receipt:
            agent_bindings = payload.get("artifacts")
            transcript_bindings = [
                binding
                for binding in agent_bindings
                if isinstance(binding, dict)
                and binding.get("role") == "agent_transcript"
            ] if isinstance(agent_bindings, list) else []
            transcript_event_receipt = (
                transcript_bindings[0].get("receipt")
                if len(transcript_bindings) == 1
                else None
            )
            if (
                not isinstance(transcript_event_receipt, dict)
                or transcript_event_receipt.get("digest")
                != _sha256_file(artifacts["agent_transcript"])
                or transcript_event_receipt.get("size")
                != artifacts["agent_transcript"].stat().st_size
            ):
                errors.append("apex_agent_transcript_event_binding_mismatch")
            errors.extend(
                _apex_turn_evidence_errors(
                    transcript=transcript,
                    payload=payload,
                    task_spec=task_spec,
                    budget_exceeded=failed_terminal,
                )
            )
            errors.extend(
                _apex_prompt_event_errors(
                    events=events,
                    agent_event=agent_event,
                    lineage=lineage if isinstance(lineage, dict) else None,
                    invocation=invocation if isinstance(invocation, dict) else None,
                    agent_prompt_path=artifacts["agent_prompt"],
                    expected_objective=task_spec.get("instructions"),
                )
            )
    verdict_ref = result.get("internal_verdict_ref")
    matching_verdicts = [event for event in events if event["event_id"] == verdict_ref]
    if len(matching_verdicts) != 1 or matching_verdicts[0]["event_type"] not in {
        "decision",
        "action.aborted",
        "action.failed",
    }:
        errors.append("apex_internal_verdict_ref_invalid")
    elif failed_terminal and (
        matching_verdicts[0]["event_type"] != "decision"
        or matching_verdicts[0]["payload"].get("verdict") != "reject"
        or matching_verdicts[0]["payload"].get("reason")
        != "agent_turn_budget_exceeded"
    ):
        errors.append("apex_internal_verdict_ref_invalid")
    elif status == "candidate_ready" and (
        matching_verdicts[0]["event_type"] != "decision"
        or matching_verdicts[0]["payload"].get("verdict") != "keep"
    ):
        errors.append("apex_internal_verdict_ref_invalid")
    if new_prompt_receipt and events:
        expected_head_type = "run.failed" if failed_terminal else "run.succeeded"
        expected_head_reason = (
            "agent_turn_budget_exceeded"
            if failed_terminal
            else "candidate_ready" if status == "candidate_ready" else result.get("reason_code")
        )
        if (
            events[-1]["event_type"] != expected_head_type
            or events[-1]["payload"].get("reason") != expected_head_reason
        ):
            errors.append("apex_terminal_run_event_mismatch")
    if isinstance(lineage, dict) and events and (
        lineage.get("journal_head_event_id") != events[-1]["event_id"]
        or lineage.get("journal_head_checksum") != events[-1]["checksum"]
        or lineage.get("event_count") != len(events)
        or lineage.get("run_id") != result.get("run_id")
    ):
        errors.append("apex_lineage_summary_mismatch")
    event_artifact_digests: set[str] = set()
    for event in events:
        bindings = event["payload"].get("artifacts", [])
        if not isinstance(bindings, list):
            errors.append("apex_event_artifact_bindings_malformed")
            continue
        for binding in bindings:
            receipt_value = binding.get("receipt") if isinstance(binding, dict) else None
            digest = receipt_value.get("digest") if isinstance(receipt_value, dict) else None
            if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
                errors.append("apex_event_artifact_receipt_malformed")
            else:
                event_artifact_digests.add(digest)
    if not isinstance(lineage, dict) or lineage.get("event_artifact_digests") != sorted(
        event_artifact_digests
    ):
        errors.append("apex_event_artifact_summary_mismatch")
    if new_prompt_receipt:
        store_ref = result.get("artifact_store_ref")
        declared_digests = (
            store_ref.get("receipt_digests") if isinstance(store_ref, dict) else None
        )
        if (
            not isinstance(declared_digests, list)
            or any(
                not isinstance(digest, str) or not _SHA256.fullmatch(digest)
                for digest in declared_digests
            )
            or len(declared_digests) != len(set(declared_digests))
            or not set(declared_digests).issubset(event_artifact_digests)
        ):
            errors.append("apex_artifact_store_receipt_set_mismatch")
    return receipt, sorted(set(errors))


def _finite_positive(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def _load_performance_cases(path: Path) -> list[dict[str, Any]]:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise CampaignError(f"cannot inspect performance evidence: {path}") from error
    if not path.is_file() or path.is_symlink() or metadata.st_nlink != 1:
        raise CampaignError(f"unsafe performance evidence file: {path}")
    payload = _load_mapping(path, "performance evidence")
    cases = payload.get("test_cases")
    if not isinstance(cases, list) or not cases or any(not isinstance(case, dict) for case in cases):
        raise CampaignError(f"performance evidence has no valid test_cases list: {path}")
    return cases


def _case_identity(case: dict[str, Any]) -> str:
    material = {
        "test_case_id": case.get("test_case_id"),
        "shape": case.get("shape"),
        "params": case.get("params"),
    }
    return json.dumps(material, sort_keys=True, separators=(",", ":"))


def _evaluation_eligibility_errors(
    workspace: Path, report: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    if report.get("pass_compilation") is not True:
        errors.append("central_compilation_failed")
    if report.get("pass_correctness") is not True:
        errors.append("central_correctness_failed")
    try:
        baseline = _load_performance_cases(workspace / "baseline_perf.yaml")
        optimized = _load_performance_cases(workspace / "optimized_perf.yaml")
    except (CampaignError, FileNotFoundError):
        return ["missing_or_unsafe_performance_case_evidence"]
    baseline_ids = [_case_identity(case) for case in baseline]
    optimized_ids = [_case_identity(case) for case in optimized]
    if len(set(baseline_ids)) != len(baseline_ids) or len(set(optimized_ids)) != len(optimized_ids):
        errors.append("duplicate_testcase_identity")
    if baseline_ids != optimized_ids:
        errors.append("baseline_optimized_testcase_set_or_order_mismatch")
    if report.get("valid_baseline_cases") != len(baseline):
        errors.append("baseline_case_count_mismatch")
    if report.get("valid_optimized_cases") != len(optimized):
        errors.append("optimized_case_count_mismatch")
    speedup_error = report.get("speedup_calculation_error_message")
    if "speedup_calculation_error_message" not in report:
        errors.append("missing_speedup_calculation_status")
    elif speedup_error is not None and speedup_error != "":
        errors.append("speedup_calculation_error")
    if report.get("benchmark_method_consistent") is not True:
        errors.append("benchmark_method_inconsistent")
    for label, cases in (("baseline", baseline), ("optimized", optimized)):
        for case in cases:
            if _finite_positive(case.get("execution_time_ms")) is None:
                errors.append(f"{label}_nonfinite_or_nonpositive_timing")
            if case.get("benchmark_samples") != 100:
                errors.append(f"{label}_benchmark_samples_not_100")
    for base_case, opt_case in zip(baseline, optimized):
        base_method = base_case.get("benchmark_method")
        opt_method = opt_case.get("benchmark_method")
        if not base_method or base_method != opt_method:
            errors.append("per_testcase_benchmark_method_mismatch")
    expected_baseline_methods = sorted(
        {
            str(case.get("benchmark_method"))
            for case in baseline
            if case.get("benchmark_method")
        }
    )
    expected_optimized_methods = sorted(
        {
            str(case.get("benchmark_method"))
            for case in optimized
            if case.get("benchmark_method")
        }
    )
    if report.get("baseline_benchmark_methods") != expected_baseline_methods:
        errors.append("baseline_benchmark_method_summary_mismatch")
    if report.get("optimized_benchmark_methods") != expected_optimized_methods:
        errors.append("optimized_benchmark_method_summary_mismatch")
    aggregate_fields = (
        "base_execution_time",
        "best_optimized_execution_time",
        "speedup_ratio",
    )
    if any(_finite_positive(report.get(field)) is None for field in aggregate_fields):
        errors.append("nonfinite_or_nonpositive_aggregate")
    observed_baseline = _finite_positive(report.get("base_execution_time"))
    observed_optimized = _finite_positive(report.get("best_optimized_execution_time"))
    baseline_times = [_finite_positive(case.get("execution_time_ms")) for case in baseline]
    optimized_times = [_finite_positive(case.get("execution_time_ms")) for case in optimized]
    if all(value is not None for value in baseline_times + optimized_times):
        exact_baseline_times = [float(value) for value in baseline_times]
        exact_optimized_times = [float(value) for value in optimized_times]
        expected_baseline = sum(exact_baseline_times) / len(exact_baseline_times)
        expected_optimized = sum(exact_optimized_times) / len(exact_optimized_times)
        if observed_baseline is not None and not math.isclose(
            observed_baseline, expected_baseline, rel_tol=1e-12, abs_tol=1e-12
        ):
            errors.append("baseline_aggregate_mismatch")
        if observed_optimized is not None and not math.isclose(
            observed_optimized, expected_optimized, rel_tol=1e-12, abs_tol=1e-12
        ):
            errors.append("optimized_aggregate_mismatch")
        if len(exact_baseline_times) == len(exact_optimized_times):
            expected_speedup = sum(
                baseline_time / optimized_time
                for baseline_time, optimized_time in zip(
                    exact_baseline_times, exact_optimized_times
                )
            ) / len(exact_baseline_times)
            observed_speedup = _finite_positive(report.get("speedup_ratio"))
            if observed_speedup is not None and not math.isclose(
                observed_speedup, expected_speedup, rel_tol=1e-12, abs_tol=1e-12
            ):
                errors.append("speedup_aggregate_mismatch")
    return sorted(set(errors))


def _select_attempt(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    evaluated = [record for record in records if record["central_evaluator_report"]]
    if not evaluated:
        return None
    # Stable attempt number is the final tie-break, so identical measurements
    # always select the earlier independent session.
    return max(
        evaluated,
        key=lambda record: (
            bool(record["selection_eligible"]),
            float(record["measured_rate_per_ms"]),
            -int(record["attempt"]),
        ),
    )


def _campaign_failure_reasons(
    task_evidence: dict[str, Any],
) -> list[str]:
    """Return stable reason codes explaining why a formal task is not canonical."""
    reasons: list[str] = []
    if task_evidence.get("campaign_manifest_unchanged") is not True:
        reasons.append("campaign_manifest_changed")
    if task_evidence.get("all_attempts_centrally_evaluated") is not True:
        reasons.append("central_evaluation_incomplete")
    if task_evidence.get("all_agent_sessions_succeeded") is not True:
        reasons.append("agent_session_failed")
    if task_evidence.get("within_evaluator_allowance") is not True:
        reasons.append("evaluator_allowance_exceeded")
    if task_evidence.get("within_task_timeout") is not True:
        reasons.append("task_timeout_exceeded")

    attempts = task_evidence.get("attempts")
    if isinstance(attempts, list):
        for record in attempts:
            if not isinstance(record, dict):
                continue
            attempt = record.get("attempt")
            prefix = f"attempt_{attempt}" if isinstance(attempt, int) else "attempt_unknown"
            if record.get("attempt_completed") is not True:
                reasons.append(f"{prefix}:session_incomplete")
            if record.get("central_evaluator_report") is None:
                reasons.append(f"{prefix}:central_evaluator_report_missing")
            errors = record.get("eligibility_errors")
            if isinstance(errors, list):
                reasons.extend(
                    f"{prefix}:{error}"
                    for error in errors
                    if isinstance(error, str) and error
                )

    selected_attempt = task_evidence.get("selected_attempt")
    if selected_attempt is None:
        reasons.append("no_centrally_evaluated_attempt")
    elif isinstance(attempts, list) and not any(
        isinstance(record, dict)
        and record.get("attempt") == selected_attempt
        and record.get("selection_eligible") is True
        for record in attempts
    ):
        reasons.append("selected_attempt_ineligible")
    return sorted(set(reasons))


def _safe_copy_workspace(source: Path, destination: Path) -> dict[str, str]:
    if destination.exists():
        raise CampaignError(f"workspace projection already exists: {destination}")
    source_manifest = _regular_tree_manifest(source)
    if len(source_manifest) > 20_000:
        raise CampaignError("selected workspace exceeds projection file-count limit")
    total_size = sum((source / relative).stat().st_size for relative in source_manifest)
    if total_size > 2 * 1024 * 1024 * 1024:
        raise CampaignError("selected workspace exceeds projection byte limit")
    destination.mkdir(parents=True)
    for relative in source_manifest:
        source_path = source / relative
        destination_path = destination / relative
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path, follow_symlinks=False)
    projected_manifest = _regular_tree_manifest(destination)
    if projected_manifest != source_manifest:
        raise CampaignError("canonical workspace projection changed selected bytes")
    return source_manifest


def run_matched_task_campaign(
    *,
    eval_config: dict[str, Any],
    agent: Any,
    agent_launcher: Any,
    task_name: str,
    task_config_dir: str,
    run_directory: Path,
    timestamp: str,
    logger: logging.Logger,
    task_index: int,
    total_tasks: int,
    single_attempt: Callable[..., tuple[bool, Path | None]],
    clock: Callable[[], float] = time.monotonic,
) -> tuple[bool, Path | None]:
    policy = parse_campaign_policy(eval_config)
    if policy is None:
        raise CampaignError("run_matched_task_campaign requires campaign policy")
    assigned_gpu = str(eval_config.get("assigned_host_gpu_id") or "")
    worker_gpu = os.environ.get("AGENT_KERNEL_ARENA_HOST_GPU_ID", "")
    if not assigned_gpu or worker_gpu != assigned_gpu:
        raise CampaignError(
            f"task GPU affinity mismatch: assigned={assigned_gpu!r} worker={worker_gpu!r}"
        )
    task_binding = validate_formal_task_binding(
        run_directory=run_directory,
        task_name=task_name,
        task_index=task_index,
        total_tasks=total_tasks,
        task_config_path=task_config_dir,
        assigned_host_gpu_id=assigned_gpu,
    )
    gpu_contract = _expected_gpu_contract(run_directory)
    gpu_exclusivity_verified = (
        isinstance(gpu_contract, dict)
        and isinstance(gpu_contract.get("exclusivity"), dict)
        and gpu_contract["exclusivity"].get("exclusivity_verified") is True
    )
    if not gpu_exclusivity_verified:
        raise CampaignError("matched task lacks verified host GPU exclusivity")
    campaign_manifest_path = run_directory / "campaign_manifest.yaml"
    if not _safe_read_only_file(campaign_manifest_path):
        raise CampaignError("matched task requires an immutable campaign manifest")
    campaign_manifest_sha256 = _sha256_file(campaign_manifest_path)
    campaign_manifest = _load_mapping(campaign_manifest_path, "campaign manifest")
    comparison_contract_sha256 = campaign_manifest.get("comparison_contract_sha256")
    if not isinstance(comparison_contract_sha256, str) or not _SHA256.fullmatch(
        comparison_contract_sha256
    ):
        raise CampaignError("campaign manifest lacks a valid comparison contract digest")
    comparison_contract = campaign_manifest.get("comparison_contract")
    if not isinstance(comparison_contract, dict) or comparison_contract_sha256 != _sha256_bytes(
        json.dumps(
            comparison_contract, sort_keys=True, separators=(",", ":")
        ).encode()
    ):
        raise CampaignError("campaign comparison contract digest does not match its contents")
    attempt_root = run_directory / ".campaign_attempts" / task_name.replace("/", "_")
    if attempt_root.exists():
        raise CampaignError(
            f"attempt root already exists; use a new run for fresh sessions: {attempt_root}"
        )
    attempt_root.mkdir(parents=True)
    started = clock()
    deadline = started + policy.task_timeout_seconds
    records: list[dict[str, Any]] = []
    evaluator_elapsed = 0.0
    require_session_receipt = getattr(agent, "value", str(agent)) in {"apex", "codex"}

    for attempt in range(1, policy.attempts + 1):
        # Re-hash the source package for every independent attempt. A descriptor
        # or task tree changed after queue initialization must never be executed.
        task_binding = validate_formal_task_binding(
            run_directory=run_directory,
            task_name=task_name,
            task_index=task_index,
            total_tasks=total_tasks,
            task_config_path=task_config_dir,
            assigned_host_gpu_id=assigned_gpu,
        )
        remaining_attempts = policy.attempts - attempt + 1
        reserved_per_attempt = (
            policy.attempt_timeout_seconds + policy.apex_internal_allowance_seconds
        )
        if deadline - clock() < remaining_attempts * reserved_per_attempt:
            records.append(
                {
                    "attempt": attempt,
                    "session": f"fresh-{attempt:02d}",
                    "attempt_completed": False,
                    "central_evaluator_report": None,
                    "selection_eligible": False,
                    "measured_rate_per_ms": 0.0,
                    "eligibility_errors": ["outer_task_deadline_cannot_cover_remaining_sessions"],
                }
            )
            break
        attempt_run = attempt_root / f"attempt_{attempt:02d}"
        attempt_run.mkdir()
        receipt_path = attempt_run / "session_receipt.json"
        attempt_config = dict(eval_config)
        attempt_config["campaign_attempt"] = {
            "index": attempt,
            "count": policy.attempts,
            "fresh_session": True,
            "timeout_seconds": policy.attempt_timeout_seconds,
            "apex_internal_allowance_seconds": policy.apex_internal_allowance_seconds,
            "task_deadline_monotonic": deadline,
            "receipt_path": str(receipt_path),
            "comparison_contract_sha256": comparison_contract_sha256,
            "task_package_manifest_sha256": task_binding[
                "package_manifest_sha256"
            ],
        }
        success, workspace = single_attempt(
            eval_config=attempt_config,
            agent=agent,
            agent_launcher=agent_launcher,
            task_name=task_name,
            task_config_dir=task_config_dir,
            run_directory=attempt_run,
            timestamp=timestamp,
            logger=logger,
            task_index=task_index,
            total_tasks=total_tasks,
        )
        validate_formal_task_binding(
            run_directory=run_directory,
            task_name=task_name,
            task_index=task_index,
            total_tasks=total_tasks,
            task_config_path=task_config_dir,
            assigned_host_gpu_id=assigned_gpu,
        )
        records.append(
            _attempt_record(
                attempt=attempt,
                workspace=workspace,
                run_directory=run_directory,
                success=success,
                receipt_path=receipt_path,
                require_session_receipt=require_session_receipt,
                expected_task_name=task_name,
            )
        )
        evaluator_elapsed += float(
            attempt_config["campaign_attempt"].get("evaluation_elapsed_seconds", 0.0)
        )
        if evaluator_elapsed > policy.evaluator_allowance_seconds:
            records[-1]["eligibility_errors"] = sorted(
                set(records[-1]["eligibility_errors"] + ["evaluator_allowance_exceeded"])
            )
            records[-1]["selection_eligible"] = False

    # Mount isolation prevents later agent sessions from seeing earlier attempts.
    # Re-read every evaluator report, receipt, artifact, and workspace manifest
    # anyway, so host-side mutation can never leave stale evidence eligible.
    for index, original in enumerate(records):
        relative_workspace = original.get("workspace")
        if not isinstance(relative_workspace, str):
            continue
        attempt_number = int(original["attempt"])
        refreshed = _attempt_record(
            attempt=attempt_number,
            workspace=run_directory / relative_workspace,
            run_directory=run_directory,
            success=original.get("attempt_completed") is True,
            receipt_path=(
                attempt_root
                / f"attempt_{attempt_number:02d}"
                / "session_receipt.json"
            ),
            require_session_receipt=require_session_receipt,
            expected_task_name=task_name,
        )
        if (
            original.get("workspace_manifest_sha256")
            != refreshed.get("workspace_manifest_sha256")
        ):
            refreshed["eligibility_errors"] = sorted(
                set(
                    refreshed["eligibility_errors"]
                    + ["attempt_workspace_changed_after_central_evaluation"]
                )
            )
            refreshed["selection_eligible"] = False
            refreshed["attempt_completed"] = False
        if "evaluator_allowance_exceeded" in original.get("eligibility_errors", []):
            refreshed["eligibility_errors"] = sorted(
                set(refreshed["eligibility_errors"] + ["evaluator_allowance_exceeded"])
            )
            refreshed["selection_eligible"] = False
        records[index] = refreshed

    try:
        manifest_unchanged = (
            _safe_read_only_file(campaign_manifest_path)
            and _sha256_file(campaign_manifest_path) == campaign_manifest_sha256
        )
    except OSError:
        manifest_unchanged = False
    if not manifest_unchanged:
        for record in records:
            record["eligibility_errors"] = sorted(
                set(record.get("eligibility_errors", []) + ["campaign_manifest_changed"])
            )
            record["selection_eligible"] = False
            record["attempt_completed"] = False

    selected = _select_attempt(records)
    task_evidence = {
        "schema": _TASK_SCHEMA,
        "task_name": task_name,
        "assigned_host_gpu_id": assigned_gpu,
        "task_index": task_index,
        "total_tasks": total_tasks,
        "task_config_path": task_binding["config_path"],
        "task_config_sha256": task_binding["config_sha256"],
        "task_package_manifest_sha256": task_binding[
            "package_manifest_sha256"
        ],
        "gpu_exclusivity_verified": gpu_exclusivity_verified,
        "campaign_manifest_sha256": campaign_manifest_sha256,
        "comparison_contract_sha256": comparison_contract_sha256,
        "campaign_manifest_unchanged": manifest_unchanged,
        "policy": asdict(policy),
        "measurement_contract": _MEASUREMENT_CONTRACT,
        "is_apex_canonical_300_sample_grade": False,
        "attempts": records,
        "selected_attempt": selected["attempt"] if selected else None,
        "all_attempts_centrally_evaluated": len(records) == policy.attempts and all(
            record["central_evaluator_report"] is not None for record in records
        ),
        "all_agent_sessions_succeeded": len(records) == policy.attempts and all(
            record.get("attempt_completed") is True for record in records
        ),
        "evaluator_elapsed_seconds": evaluator_elapsed,
        "within_evaluator_allowance": evaluator_elapsed <= policy.evaluator_allowance_seconds,
        "elapsed_seconds": clock() - started,
        "within_task_timeout": clock() <= deadline,
    }
    task_evidence["failure_reasons"] = _campaign_failure_reasons(task_evidence)
    task_campaign_path = attempt_root / "task_campaign.yaml"
    _atomic_yaml(task_campaign_path, task_evidence)
    _seal_evidence_file(task_campaign_path, "matched task campaign evidence")
    completed = task_evidence["all_attempts_centrally_evaluated"]
    completed = completed and task_evidence["all_agent_sessions_succeeded"]
    completed = completed and task_evidence["within_evaluator_allowance"]
    completed = completed and task_evidence["within_task_timeout"]
    completed = completed and bool(selected and selected["selection_eligible"])
    if not completed or selected is None:
        return False, None

    selected_workspace = run_directory / str(selected["workspace"])
    canonical = get_task_workspace_path(run_directory, task_name, timestamp)
    if canonical.exists():
        raise CampaignError(f"canonical task projection already exists: {canonical}")
    selected_manifest = _safe_copy_workspace(selected_workspace, canonical)
    result_path = canonical / "task_result.yaml"
    result = _load_mapping(result_path, "selected task result")
    result["campaign_evidence"] = {
        "schema": _TASK_SCHEMA,
        "campaign_manifest_sha256": campaign_manifest_sha256,
        "comparison_contract_sha256": comparison_contract_sha256,
        "task_campaign_sha256": _sha256_file(task_campaign_path),
        "attempt_count": policy.attempts,
        "selected_attempt": selected["attempt"],
        "selection_policy": policy.selection_policy,
        "selected_measured_rate_per_ms": selected["measured_rate_per_ms"],
        "attempt_manifest": str(
            (attempt_root / "task_campaign.yaml").relative_to(run_directory)
        ),
        "measurement_contract": _MEASUREMENT_CONTRACT,
        "is_apex_canonical_300_sample_grade": False,
        "selected_central_evaluator_report_sha256": selected[
            "central_evaluator_report_sha256"
        ],
        "selected_performance_evidence_sha256": {
            name: selected_manifest[name]
            for name in ("baseline_perf.yaml", "optimized_perf.yaml")
        },
        "selected_workspace_manifest_sha256": _sha256_bytes(
            json.dumps(selected_manifest, sort_keys=True, separators=(",", ":")).encode()
        ),
    }
    _atomic_yaml(result_path, result)
    for evidence_name in (
        "baseline_perf.yaml",
        "optimized_perf.yaml",
        "task_result.yaml",
    ):
        evidence_path = canonical / evidence_name
        if not evidence_path.exists():
            raise CampaignError(
                f"canonical matched task lacks final evidence: {evidence_path}"
            )
        _seal_evidence_file(
            evidence_path, f"canonical matched task {evidence_name}"
        )
    return True, canonical


__all__ = [
    "CampaignError",
    "CampaignPolicy",
    "build_campaign_manifest",
    "deterministic_task_gpu_mapping",
    "ensure_campaign_manifest",
    "ordered_gpu_pool",
    "parse_campaign_policy",
    "run_matched_task_campaign",
    "validate_formal_task_binding",
]
