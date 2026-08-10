# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Matched-campaign orchestration and provenance for Apex-versus-Codex runs."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import os
import re
import shutil
import signal
import sqlite3
import stat
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable

import yaml

from src.aka_runtime import (
    AkaRuntimeError,
    BACKEND_CLOSURE_SCHEMA,
    EXECUTION_MANIFEST_SCHEMA,
    ENGINE_EVIDENCE_SCHEMA,
    ENGINE_SERVICE_SCHEMA,
    HOST_ACCESS_POLICY_SCHEMA,
    IMMUTABLE_MOUNT_RECEIPT_SCHEMA,
    capture_backend_closure,
    capture_execution_manifest,
    load_runtime_service_evidence,
    validate_immutable_mount_receipt,
    verify_backend_closure,
    verify_materialized_snapshot,
)
from src.apex_runtime import (
    ApexRuntimeError,
    RUNTIME_BOOTSTRAP_NAME,
    RUNTIME_BOOTSTRAP_POLICY_ID,
    RUNTIME_BOOTSTRAP_SHA256,
    RUNTIME_IMAGE_INPUT_SCHEMA,
    RUNTIME_IMMUTABLE_MOUNT_POLICY_ID,
    RUNTIME_IMMUTABLE_MOUNT_SCHEMA,
    RUNTIME_WRAPPER_ALIASES,
    RUNTIME_WRAPPER_NAME,
    RUNTIME_WRAPPER_POLICY_ID,
    RUNTIME_WRAPPER_SHA256,
    runtime_environment,
    validate_immutable_mount_receipt as validate_apex_immutable_mount_receipt,
    verify_runtime_snapshot,
)
from src.agent_turn_budget import (
    AGENT_PROCESS_CONTAINMENT_POLICY,
    BOUNDARY_QUIESCENCE_POLICY,
    CANDIDATE_PERSISTENCE_POLICY,
    FORMAL_MATCHED_MAX_TURNS,
    TURN_POLICY,
    context_packet_objective_matches,
    render_apex_run_control,
)
from src.campaign_isolation import (
    APEX_RUNTIME_MOUNT_POLICY,
    APEX_RUNTIME_MOUNT_SCHEMA,
    ATTEMPT_MOUNT_RECEIPT_SCHEMA,
    ATTEMPT_CONTAINMENT_POLICY,
    CODEX_CLOUD_CONFIG_BOOTSTRAP_POLICY,
    CODEX_CLOUD_CONFIG_BOOTSTRAP_SCHEMA,
    CampaignIsolationError,
    attempt_cleanup_verified,
    codex_cloud_config_contract,
    runtime_isolation_receipt,
)
from src.gpu_device_boundary import GpuBoundaryError, load_plan
from src.gpu_exclusivity import (
    GpuExclusivityError,
    load_receipt as load_gpu_lease_receipt,
)
from src.evaluator_utils import (
    FORMAL_SOURCE_ANTI_TAMPER_POLICY,
    FORMAL_SOURCE_ANTI_TAMPER_RULES_SHA256,
    FORMAL_SOURCE_ANTI_TAMPER_SCHEMA,
    canonical_json_sha256,
    inspect_formal_source_anti_tamper,
)
from src.preprocessing import get_task_workspace_path


_CAMPAIGN_SCHEMA = "aka.matched-campaign/v1"
_TASK_SCHEMA = "aka.matched-task-attempts/v1"
_CAMPAIGN_BINDING_SCHEMA = "aka.attempt-campaign-binding/v1"
_RUN_CONFIG_CONTRACT_SCHEMA = "aka.formal-run-config/v1"
_DURABLE_RUN_CONFIG_DIRECTORY = ".formal-run-config"
_DURABLE_RUN_CONFIG_NAME = "run_config.yaml"
_MAX_RUN_CONFIG_BYTES = 1024 * 1024
_CAMPAIGN_BINDING_KEYS = frozenset(
    {
        "schema",
        "formal_execution_sha256",
        "campaign_manifest_path",
        "campaign_manifest_sha256",
        "comparison_contract_sha256",
        "backend_runtime_closure_sha256",
        "task_package_manifest_sha256",
        "task_config_sha256",
        "task_name",
        "task_index",
        "total_tasks",
        "attempt_index",
        "attempt_count",
        "assigned_host_gpu_id",
    }
)
_SELECTION_POLICY = "correctness_then_measured_rate_v1"
_MEASUREMENT_CONTRACT = "aka_native_100_repetition_external_score"
_OBJECTIVE_POLICY_ID = "aka.task-package-objective-and-protected-harness/v1"
_PROMPT_POLICY_ID = "aka.shared-objective-backend-native-context-receipted/v1"
_COMPARISON_CONTRACT_SCHEMA = "aka.apex-vs-codex-comparison-contract/v7"
_CODEX_RECEIPT_SCHEMA = "agentkernelarena.codex-attempt-receipt/v6"
_APEX_RECEIPT_SCHEMA = "agentkernelarena.apex-attempt-receipt/v7"
_SESSION_RECEIPT_SCHEMA_BY_AGENT = {
    "apex": _APEX_RECEIPT_SCHEMA,
    "codex": _CODEX_RECEIPT_SCHEMA,
}
_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FORMAL_LIVE_COMMITMENT = {
    "mode": "live_formal_scoring",
    "comparison_generation": 7,
    "historical_compatibility": False,
    "policy_id": "aka.live-formal-v7-only/v1",
}
_FORMAL_LIVE_COMMITMENT_SHA256 = hashlib.sha256(
    json.dumps(
        _FORMAL_LIVE_COMMITMENT, sort_keys=True, separators=(",", ":")
    ).encode()
).hexdigest()

# Public, immutable generation marker for queue/worker entrypoints that must
# reject historical campaign artifacts before they perform any evaluation.
FORMAL_LIVE_EXECUTION_SHA256 = _FORMAL_LIVE_COMMITMENT_SHA256
FORMAL_AGENT_TRANSPORT_TREATMENTS = {
    "apex": {
        "max_process_output_bytes": 4 * 1024 * 1024,
        "structured_stream_output_limit_bytes": 16 * 1024 * 1024,
        "overflow_policy": "apex_inner_supervisor_bounded_truncation",
    },
    "codex": {
        "max_process_output_bytes": 16 * 1024 * 1024,
        "structured_stream_output_limit_bytes": 16 * 1024 * 1024,
        "overflow_policy": "fail_closed_after_bounded_drain",
    },
}


class CampaignError(RuntimeError):
    """Raised when a matched campaign cannot preserve its fairness contract."""


def resolve_session_receipt_schema(
    agent_name: str, declared_schema: object | None
) -> str | None:
    """Resolve the one live receipt schema; all superseded artifacts fail closed."""

    supported = _SESSION_RECEIPT_SCHEMA_BY_AGENT.get(agent_name)
    return supported if declared_schema == supported else None


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


def campaign_task_path_component(task_name: str) -> str:
    """Map an untrusted task name to one collision-resistant path component."""

    if not isinstance(task_name, str) or not task_name or "\x00" in task_name:
        raise CampaignError("campaign task name must be non-empty UTF-8 text")
    try:
        encoded = task_name.encode("utf-8")
    except UnicodeEncodeError as error:
        raise CampaignError("campaign task name is not valid UTF-8 text") from error
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", task_name).strip("._-")
    slug = (slug[:80] or "task").rstrip("._-") or "task"
    digest = _sha256_bytes(encoded)
    component = f"{slug}--{digest}"
    if component in {".", ".."} or "/" in component or "\\" in component:
        raise CampaignError("campaign task path component is unsafe")
    return component


def _campaign_task_paths_valid(manifest: dict[str, Any]) -> bool:
    configuration = manifest.get("configuration")
    tasks = configuration.get("tasks") if isinstance(configuration, dict) else None
    if not isinstance(tasks, list) or not tasks:
        return False
    try:
        names = [task.get("task_name") for task in tasks if isinstance(task, dict)]
        components = [campaign_task_path_component(name) for name in names]
    except (CampaignError, TypeError):
        return False
    return bool(
        len(names) == len(tasks)
        and len(names) == len(set(names))
        and len(components) == len(set(components))
    )


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
    """Prove exact committed bytes without trusting Git's mutable index."""

    try:
        manifest = capture_execution_manifest(root)
    except AkaRuntimeError as error:
        raise CampaignError(f"cannot prove exact AgentKernelArena checkout: {error}") from error
    source = manifest["source"]
    return {
        "commit": source["commit"],
        "tree": source["tree"],
        "dirty": False,
        "status_sha256": _sha256_bytes(b""),
        "execution_manifest_schema": EXECUTION_MANIFEST_SCHEMA,
        "execution_manifest_sha256": manifest["manifest_sha256"],
        "git_evidence_policy_id": manifest["policy_id"],
    }


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot read {label}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise CampaignError(f"{label} must contain a JSON object")
    return value


def _runtime_evidence_path(environment_key: str, label: str) -> Path:
    raw = os.environ.get(environment_key, "")
    if not raw:
        raise CampaignError(f"runner did not provide {label}")
    path = Path(raw)
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise CampaignError(f"cannot inspect runner-provided {label}: {path}") from error
    if (
        not path.is_absolute()
        or resolved != path
        or path.is_symlink()
        or not path.is_file()
        or metadata.st_nlink != 1
    ):
        raise CampaignError(f"runner-provided {label} is unsafe")
    return path


def _aka_state_from_environment(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the mounted AKA image and bind its host attestation."""

    runtime_root_raw = os.environ.get("AGENT_KERNEL_ARENA_AKA_RUNTIME_ROOT", "")
    try:
        runtime_lexical = Path(runtime_root_raw)
        runtime_metadata = runtime_lexical.lstat()
        runtime_root = runtime_lexical.resolve(strict=True)
        resolved_repo = repo_root.resolve(strict=True)
    except OSError as error:
        raise CampaignError("runner did not provide an available AKA runtime root") from error
    if (
        not runtime_root_raw
        or not runtime_lexical.is_absolute()
        or runtime_lexical != Path(os.path.abspath(runtime_lexical))
        or runtime_lexical != runtime_root
        or runtime_lexical.is_symlink()
        or not stat.S_ISDIR(runtime_metadata.st_mode)
        or runtime_root != resolved_repo
    ):
        raise CampaignError("AKA code is not executing from the attested runtime root")
    manifest_path = _runtime_evidence_path(
        "AGENT_KERNEL_ARENA_AKA_RUNTIME_MANIFEST", "AKA runtime manifest"
    )
    manifest_digest = os.environ.get(
        "AGENT_KERNEL_ARENA_AKA_RUNTIME_MANIFEST_SHA256", ""
    )
    manifest_file_digest = os.environ.get(
        "AGENT_KERNEL_ARENA_AKA_RUNTIME_MANIFEST_FILE_SHA256", ""
    )
    receipt_path = _runtime_evidence_path(
        "AGENT_KERNEL_ARENA_AKA_RUNTIME_MOUNT_RECEIPT",
        "AKA runtime mount receipt",
    )
    receipt_digest = os.environ.get(
        "AGENT_KERNEL_ARENA_AKA_RUNTIME_MOUNT_RECEIPT_SHA256", ""
    )
    receipt_file_digest = os.environ.get(
        "AGENT_KERNEL_ARENA_AKA_RUNTIME_MOUNT_RECEIPT_FILE_SHA256", ""
    )
    service_path = _runtime_evidence_path(
        "AGENT_KERNEL_ARENA_AKA_RUNTIME_SERVICE_EVIDENCE",
        "AKA runtime service evidence",
    )
    service_file_digest = os.environ.get(
        "AGENT_KERNEL_ARENA_AKA_RUNTIME_SERVICE_EVIDENCE_FILE_SHA256", ""
    )
    service_content_digest = os.environ.get(
        "AGENT_KERNEL_ARENA_AKA_RUNTIME_SERVICE_EVIDENCE_CONTENT_SHA256", ""
    )
    if any(
        not _SHA256.fullmatch(value)
        for value in (
            manifest_digest,
            manifest_file_digest,
            receipt_digest,
            receipt_file_digest,
            service_file_digest,
            service_content_digest,
        )
    ):
        raise CampaignError("runner provided an invalid AKA runtime evidence digest")
    if _sha256_file(manifest_path) != manifest_file_digest:
        raise CampaignError("AKA runtime manifest file digest differs from runner evidence")
    if _sha256_file(receipt_path) != receipt_file_digest:
        raise CampaignError("AKA runtime mount receipt digest differs from runner evidence")
    source_manifest = _load_json_object(manifest_path, "AKA runtime manifest")
    mount_receipt = _load_json_object(receipt_path, "AKA runtime mount receipt")
    try:
        service_evidence = load_runtime_service_evidence(
            service_path,
            file_sha256=service_file_digest,
            content_sha256=service_content_digest,
            manifest_sha256=manifest_digest,
            image_sha256=str(mount_receipt.get("image_sha256") or ""),
        )
        verify_materialized_snapshot(runtime_root, source_manifest, manifest_digest)
        validate_immutable_mount_receipt(
            mount_receipt, manifest_digest, expected_root=runtime_root
        )
    except AkaRuntimeError as error:
        raise CampaignError(f"AKA runtime attestation is invalid: {error}") from error
    source = source_manifest.get("source")
    if not isinstance(source, dict):
        raise CampaignError("AKA runtime manifest lacks source Git evidence")
    if mount_receipt.get("sha256") != receipt_digest:
        raise CampaignError("AKA mount receipt content digest differs from runner evidence")
    if (
        mount_receipt.get("runtime_service_evidence_sha256")
        != service_content_digest
        or mount_receipt.get("runtime_engine_evidence_sha256")
        != service_evidence["engine_evidence"]["sha256"]
        or mount_receipt.get("host_access_policy")
        != service_evidence["mount_receipt"]["host_access_policy"]
    ):
        raise CampaignError("AKA mount receipt is not bound to its host engine evidence")
    state = {
        "commit": source.get("commit"),
        "tree": source.get("tree"),
        "dirty": False,
        "status_sha256": _sha256_bytes(b""),
        "execution_manifest_schema": source_manifest.get("schema"),
        "execution_manifest_sha256": manifest_digest,
        "git_evidence_policy_id": source_manifest.get("policy_id"),
    }
    if not _SHA1.fullmatch(str(state["commit"])) or not _SHA1.fullmatch(
        str(state["tree"])
    ):
        raise CampaignError("AKA runtime source Git identity is invalid")
    runtime = {
        "schema": "aka.execution-snapshot-runtime/v2",
        "root": str(runtime_root),
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": manifest_file_digest,
        "manifest_sha256": manifest_digest,
        "mount_receipt_path": str(receipt_path),
        "mount_receipt_file_sha256": receipt_file_digest,
        "mount_receipt_sha256": receipt_digest,
        "mount_receipt_schema": IMMUTABLE_MOUNT_RECEIPT_SCHEMA,
        "mount_receipt": mount_receipt,
        "runtime_service_evidence_path": str(service_path),
        "runtime_service_evidence_file_sha256": service_file_digest,
        "runtime_service_evidence_content_sha256": service_content_digest,
        "runtime_engine_evidence_sha256": service_evidence["engine_evidence"][
            "sha256"
        ],
        "runtime_service_evidence": service_evidence,
    }
    return state, runtime


def _revalidate_aka_runtime(manifest: dict[str, Any]) -> bool:
    runtime = manifest.get("runtime")
    evidence = runtime.get("aka_execution_snapshot") if isinstance(runtime, dict) else None
    if not isinstance(evidence, dict) or set(evidence) != {
        "schema",
        "root",
        "manifest_path",
        "manifest_file_sha256",
        "manifest_sha256",
        "mount_receipt_path",
        "mount_receipt_file_sha256",
        "mount_receipt_sha256",
        "mount_receipt_schema",
        "mount_receipt",
        "runtime_service_evidence_path",
        "runtime_service_evidence_file_sha256",
        "runtime_service_evidence_content_sha256",
        "runtime_engine_evidence_sha256",
        "runtime_service_evidence",
    }:
        return False
    try:
        root_lexical = Path(evidence["root"])
        source_lexical = Path(evidence["manifest_path"])
        receipt_lexical = Path(evidence["mount_receipt_path"])
        service_lexical = Path(evidence["runtime_service_evidence_path"])
        root_metadata = root_lexical.lstat()
        source_metadata = source_lexical.lstat()
        receipt_metadata = receipt_lexical.lstat()
        service_metadata = service_lexical.lstat()
        root = root_lexical.resolve(strict=True)
        source_path = source_lexical.resolve(strict=True)
        receipt_path = receipt_lexical.resolve(strict=True)
        service_path = service_lexical.resolve(strict=True)
        if (
            evidence.get("schema") != "aka.execution-snapshot-runtime/v2"
            or evidence.get("mount_receipt_schema")
            != IMMUTABLE_MOUNT_RECEIPT_SCHEMA
            or root != root_lexical
            or source_path != source_lexical
            or receipt_path != receipt_lexical
            or service_path != service_lexical
            or root_lexical.is_symlink()
            or source_lexical.is_symlink()
            or receipt_lexical.is_symlink()
            or service_lexical.is_symlink()
            or not stat.S_ISDIR(root_metadata.st_mode)
            or not stat.S_ISREG(source_metadata.st_mode)
            or not stat.S_ISREG(receipt_metadata.st_mode)
            or not stat.S_ISREG(service_metadata.st_mode)
            or source_metadata.st_nlink != 1
            or receipt_metadata.st_nlink != 1
            or service_metadata.st_nlink != 1
            or not _SHA256.fullmatch(str(evidence.get("manifest_sha256") or ""))
            or not _SHA256.fullmatch(
                str(evidence.get("mount_receipt_sha256") or "")
            )
            or _sha256_file(source_path) != evidence["manifest_file_sha256"]
            or _sha256_file(receipt_path) != evidence["mount_receipt_file_sha256"]
            or _sha256_file(service_path)
            != evidence["runtime_service_evidence_file_sha256"]
        ):
            return False
        source = _load_json_object(source_path, "AKA runtime manifest")
        receipt = _load_json_object(receipt_path, "AKA runtime mount receipt")
        service = load_runtime_service_evidence(
            service_path,
            file_sha256=evidence["runtime_service_evidence_file_sha256"],
            content_sha256=evidence[
                "runtime_service_evidence_content_sha256"
            ],
            manifest_sha256=evidence["manifest_sha256"],
            image_sha256=str(receipt.get("image_sha256") or ""),
        )
        verify_materialized_snapshot(root, source, evidence["manifest_sha256"])
        validate_immutable_mount_receipt(receipt, evidence["manifest_sha256"], root)
    except (AkaRuntimeError, CampaignError, KeyError, OSError, TypeError, ValueError):
        return False
    repositories = manifest.get("repositories")
    aka = repositories.get("agent_kernel_arena") if isinstance(repositories, dict) else None
    return bool(
        isinstance(aka, dict)
        and aka.get("execution_manifest_sha256") == evidence["manifest_sha256"]
        and receipt.get("sha256") == evidence["mount_receipt_sha256"]
        and receipt == evidence["mount_receipt"]
        and service == evidence["runtime_service_evidence"]
        and service.get("sha256")
        == evidence["runtime_service_evidence_content_sha256"]
        and service.get("engine_evidence", {}).get("sha256")
        == evidence["runtime_engine_evidence_sha256"]
        and receipt.get("runtime_service_evidence_sha256")
        == service.get("sha256")
        and receipt.get("runtime_engine_evidence_sha256")
        == service.get("engine_evidence", {}).get("sha256")
    )


def _apex_state_from_environment() -> dict[str, Any]:
    commit = os.environ.get("AGENT_KERNEL_ARENA_APEX_COMMIT", "")
    dirty = os.environ.get("AGENT_KERNEL_ARENA_APEX_DIRTY", "")
    status_digest = os.environ.get("AGENT_KERNEL_ARENA_APEX_STATUS_SHA256", "")
    runtime_manifest_digest = os.environ.get(
        "AGENT_KERNEL_ARENA_APEX_RUNTIME_MANIFEST_SHA256", ""
    )
    if not _SHA1.fullmatch(commit):
        raise CampaignError("runner did not provide a valid Apex commit")
    if dirty not in {"true", "false"}:
        raise CampaignError("runner did not provide Apex dirty=true|false")
    if not _SHA256.fullmatch(status_digest):
        raise CampaignError("runner did not provide a valid Apex status digest")
    if not _SHA256.fullmatch(runtime_manifest_digest):
        raise CampaignError("runner did not provide a valid Apex runtime manifest digest")
    state = {
        "commit": commit,
        "dirty": dirty == "true",
        "status_sha256": status_digest,
        "runtime_manifest_sha256": runtime_manifest_digest,
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


def _agent_manifest(
    repo_root: Path,
    agent_name: str,
    policy: CampaignPolicy,
    *,
    apex_runtime_manifest_sha256: str,
) -> dict[str, Any]:
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
    transport = FORMAL_AGENT_TRANSPORT_TREATMENTS.get(agent_name)
    if transport is None or (
        int(config.get("max_process_output_bytes", 0))
        != transport["max_process_output_bytes"]
        or int(config.get("structured_stream_output_limit_bytes", 0))
        != transport["structured_stream_output_limit_bytes"]
        or config.get("structured_stream_overflow_policy")
        != transport["overflow_policy"]
    ):
        raise CampaignError(
            f"matched campaign {agent_name} transport policy violates its pin"
        )
    codex = shutil.which("codex")
    if not codex:
        raise CampaignError("codex CLI is unavailable for campaign provenance")
    binary_path = Path(codex).resolve()
    if not binary_path.is_file():
        raise CampaignError("resolved codex binary is not a regular file")
    try:
        backend_closure = capture_backend_closure("codex", codex)
    except AkaRuntimeError as error:
        raise CampaignError(f"cannot capture complete Codex runtime: {error}") from error
    try:
        cloud_config_contract = codex_cloud_config_contract()
    except CampaignIsolationError as error:
        raise CampaignError(
            f"cannot bind formal Codex cloud-config bootstrap: {error}"
        ) from error
    manifest = {
        "template": agent_name,
        "session_receipt_schema": (
            _APEX_RECEIPT_SCHEMA
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
        "agent_process_containment_policy_id": (
            AGENT_PROCESS_CONTAINMENT_POLICY
        ),
        "attempt_containment_policy_id": ATTEMPT_CONTAINMENT_POLICY,
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
        "backend_runtime_closure_schema": BACKEND_CLOSURE_SCHEMA,
        "backend_runtime_closure_sha256": backend_closure["closure_sha256"],
        "backend_runtime_closure": backend_closure,
        "isolation": {
            "approval": "never_via_strict_config",
            "execpolicy_rules": "ignored",
            "project_instructions": "backend_default_may_load",
            "sandbox": "workspace-write",
            "session": "ephemeral",
            "user_config": "ignored",
            "mount_scope": "attempt_only_bubblewrap",
            "attempt_containment_policy_id": ATTEMPT_CONTAINMENT_POLICY,
        },
        "agent_config_sha256": _sha256_file(config_path),
    }
    manifest.update(cloud_config_contract)
    if agent_name == "apex":
        if not _SHA256.fullmatch(apex_runtime_manifest_sha256):
            raise CampaignError("Apex runtime manifest digest is unavailable")
        manifest.update(
            {
                "apex_runtime_mount_policy_id": APEX_RUNTIME_MOUNT_POLICY,
                "attempt_mount_receipt_schema": ATTEMPT_MOUNT_RECEIPT_SCHEMA,
                "apex_runtime_mount_schema": APEX_RUNTIME_MOUNT_SCHEMA,
                "runtime_manifest_sha256": apex_runtime_manifest_sha256,
            }
        )
    return manifest


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


def _evaluator_manifest(aka_state: dict[str, Any]) -> dict[str, Any]:
    """Bind evaluation to the complete AKA execution snapshot, never a file subset."""

    digest = aka_state.get("execution_manifest_sha256")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise CampaignError("complete AKA execution manifest is unavailable")
    return {
        "schema": "aka.evaluator-source-binding/v2",
        "coverage": "all_committed_files",
        "execution_manifest_schema": aka_state.get("execution_manifest_schema"),
        "execution_manifest_sha256": digest,
        "commit": aka_state.get("commit"),
        "tree": aka_state.get("tree"),
    }


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


def _run_config_contract(
    run_config_path: Path, *, agent_name: str
) -> dict[str, Any]:
    """Bind every formal run option except the intentional agent treatment."""

    try:
        supplied = run_config_path.absolute()
        metadata = supplied.lstat()
        path = supplied.resolve(strict=True)
    except OSError as error:
        raise CampaignError("formal run config is unavailable") from error
    if not path.is_file() or supplied.is_symlink() or metadata.st_nlink != 1:
        raise CampaignError("formal run config is not a safe regular file")
    document = _load_mapping(path, "formal run config")
    if document.get("agent") != {"template": agent_name}:
        raise CampaignError(
            "formal run config agent must contain exactly the selected template"
        )
    projection = {key: value for key, value in document.items() if key != "agent"}
    if not projection:
        raise CampaignError("formal run config projection is empty")
    return {
        "schema": _RUN_CONFIG_CONTRACT_SCHEMA,
        "effective_config": projection,
        "effective_config_sha256": _canonical_json_digest(projection),
    }


def _read_run_config_bytes(path: Path) -> tuple[bytes, Path]:
    """Read one bounded regular config without following or racing a symlink."""

    supplied = path.absolute()
    descriptor = -1
    try:
        lexical = supplied.lstat()
        resolved = supplied.resolve(strict=True)
        descriptor = os.open(
            supplied,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        if (
            resolved != supplied
            or stat.S_ISLNK(lexical.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (lexical.st_dev, lexical.st_ino)
            or opened.st_size <= 0
            or opened.st_size > _MAX_RUN_CONFIG_BYTES
        ):
            raise CampaignError("formal run config is not a safe bounded regular file")
        payload = bytearray()
        while len(payload) <= opened.st_size:
            chunk = os.read(descriptor, opened.st_size + 1 - len(payload))
            if not chunk:
                break
            payload.extend(chunk)
        rechecked = os.fstat(descriptor)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_uid",
            "st_gid",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if len(payload) != opened.st_size or any(
            getattr(rechecked, field) != getattr(opened, field)
            for field in stable_fields
        ):
            raise CampaignError("formal run config changed while it was read")
        return bytes(payload), resolved
    except CampaignError:
        raise
    except OSError as error:
        raise CampaignError("formal run config is unavailable") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _publish_durable_run_config(path: Path, payload: bytes) -> None:
    """Publish exact config bytes once without replacing an existing artifact."""

    temporary_fd = -1
    temporary_path: Path | None = None
    try:
        temporary_fd, raw_temporary = tempfile.mkstemp(
            prefix=f".{_DURABLE_RUN_CONFIG_NAME}.", dir=path.parent
        )
        temporary_path = Path(raw_temporary)
        offset = 0
        while offset < len(payload):
            offset += os.write(temporary_fd, payload[offset:])
        os.fchmod(temporary_fd, 0o444)
        os.fsync(temporary_fd)
        os.close(temporary_fd)
        temporary_fd = -1
        try:
            os.link(temporary_path, path, follow_symlinks=False)
        except FileExistsError:
            pass
        else:
            temporary_path.unlink()
            temporary_path = None
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except OSError as error:
        raise CampaignError("cannot publish durable formal run config") from error
    finally:
        if temporary_fd >= 0:
            os.close(temporary_fd)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _materialize_durable_run_config(
    run_directory: Path, source: Path
) -> Path:
    """Copy the ephemeral sealed-runtime config into durable run evidence."""

    payload, _source = _read_run_config_bytes(source)
    try:
        run_metadata = run_directory.lstat()
        run = run_directory.resolve(strict=True)
    except OSError as error:
        raise CampaignError("formal run directory is unavailable") from error
    if (
        run != run_directory.absolute()
        or stat.S_ISLNK(run_metadata.st_mode)
        or not stat.S_ISDIR(run_metadata.st_mode)
    ):
        raise CampaignError("formal run directory is unsafe")
    evidence = run / _DURABLE_RUN_CONFIG_DIRECTORY
    try:
        evidence.mkdir(mode=0o700)
    except FileExistsError:
        pass
    try:
        metadata = evidence.lstat()
        resolved = evidence.resolve(strict=True)
    except OSError as error:
        raise CampaignError("durable run-config directory is unavailable") from error
    if (
        resolved != evidence
        or resolved.parent != run
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) not in {0o700, 0o555}
    ):
        raise CampaignError("durable run-config directory is unsafe")
    destination = evidence / _DURABLE_RUN_CONFIG_NAME
    if not destination.exists():
        if stat.S_IMODE(metadata.st_mode) != 0o700:
            raise CampaignError("sealed run-config directory lacks its config")
        _publish_durable_run_config(destination, payload)
    observed, resolved_destination = _read_run_config_bytes(destination)
    destination_metadata = destination.lstat()
    if (
        resolved_destination != destination
        or observed != payload
        or destination_metadata.st_nlink != 1
        or destination_metadata.st_mode & 0o222
    ):
        raise CampaignError("durable formal run config differs from its source")
    evidence.chmod(0o555)
    return destination


def _v7_apex_treatment_contract(repositories: Any) -> dict[str, Any] | None:
    if not isinstance(repositories, dict):
        return None
    apex = repositories.get("apex")
    if not isinstance(apex, dict):
        return None
    runtime_digest = apex.get("runtime_manifest_sha256")
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


def _v7_top_level_agent_valid(
    agent: Any, apex_treatment: dict[str, Any]
) -> bool:
    if not isinstance(agent, dict):
        return False
    template = agent.get("template")
    closure = agent.get("backend_runtime_closure")
    try:
        closure_valid = bool(
            isinstance(closure, dict)
            and agent.get("backend_runtime_closure_schema")
            == BACKEND_CLOSURE_SCHEMA
            and agent.get("backend_runtime_closure_sha256")
            == closure.get("closure_sha256")
            and verify_backend_closure(
                closure, agent.get("backend_runtime_closure_sha256")
            )
        )
    except (AkaRuntimeError, OSError, TypeError, ValueError):
        closure_valid = False
    cloud_config_valid = bool(
        agent.get("cloud_config_bootstrap_schema")
        == CODEX_CLOUD_CONFIG_BOOTSTRAP_SCHEMA
        and agent.get("cloud_config_bootstrap_policy")
        == CODEX_CLOUD_CONFIG_BOOTSTRAP_POLICY
        and isinstance(agent.get("cloud_config_bundle_sha256"), str)
        and _SHA256.fullmatch(agent["cloud_config_bundle_sha256"])
        and isinstance(
            agent.get("cloud_config_host_runtime_closure_sha256"), str
        )
        and _SHA256.fullmatch(
            agent["cloud_config_host_runtime_closure_sha256"]
        )
        and isinstance(
            agent.get("cloud_config_initial_refresh_receipt_sha256"), str
        )
        and _SHA256.fullmatch(
            agent["cloud_config_initial_refresh_receipt_sha256"]
        )
    )
    if template == "apex":
        return cloud_config_valid and closure_valid and all(
            agent.get(key) == value for key, value in apex_treatment.items()
        )
    forbidden = {
        "apex_runtime_mount_policy_id",
        "attempt_mount_receipt_schema",
        "apex_runtime_mount_schema",
        "runtime_manifest_sha256",
    }
    return bool(
        template == "codex"
        and agent.get("session_receipt_schema") == _CODEX_RECEIPT_SCHEMA
        and cloud_config_valid
        and closure_valid
        and not forbidden.intersection(agent)
    )


def _v7_manifest_contract_valid(
    manifest: dict[str, Any], comparison: dict[str, Any]
) -> bool:
    repositories = manifest.get("repositories")
    comparison_repositories = comparison.get("repositories")
    apex_treatment = _v7_apex_treatment_contract(repositories)
    apex = repositories.get("apex") if isinstance(repositories, dict) else None
    aka = (
        repositories.get("agent_kernel_arena")
        if isinstance(repositories, dict)
        else None
    )
    runtime = manifest.get("runtime")
    comparison_runtime = comparison.get("runtime")
    projected_runtime = comparison_runtime_projection(runtime)
    agent = manifest.get("agent")
    comparison_codex = comparison.get("codex")
    transport_treatments = comparison.get("agent_transport_treatments")
    evaluator = manifest.get("evaluator_files_sha256")
    agent_transport = (
        transport_treatments.get(agent.get("template"))
        if isinstance(agent, dict) and isinstance(transport_treatments, dict)
        else None
    )
    configuration = manifest.get("configuration")
    run_config = comparison.get("run_config")
    try:
        run_config_path = Path(str(configuration.get("run_config_path") or ""))
        expected_run_config = (
            _run_config_contract(
                run_config_path,
                agent_name=str(agent.get("template") or ""),
            )
            if isinstance(configuration, dict) and isinstance(agent, dict)
            else None
        )
        run_config_metadata = run_config_path.lstat()
        declared_run_config_size = configuration.get("run_config_size_bytes")
        durable_run_config_valid = bool(
            type(declared_run_config_size) is int
            and declared_run_config_size == run_config_metadata.st_size
            and _safe_read_only_file(run_config_path)
        )
        run_config_file_valid = bool(
            expected_run_config is not None
            and durable_run_config_valid
            and configuration.get("run_config_sha256") == _sha256_file(run_config_path)
        )
    except (CampaignError, OSError, TypeError, ValueError):
        expected_run_config = None
        run_config_file_valid = False
    return bool(
        isinstance(repositories, dict)
        and comparison_repositories == repositories
        and apex_treatment is not None
        and comparison.get("apex_treatment") == apex_treatment
        and _v7_top_level_agent_valid(
            agent, apex_treatment
        )
        and isinstance(agent, dict)
        and isinstance(comparison_codex, dict)
        and comparison_codex.get("backend_runtime_closure_sha256")
        == agent.get("backend_runtime_closure_sha256")
        and comparison_codex.get("backend_runtime_closure")
        == agent.get("backend_runtime_closure")
        and comparison_codex.get("cloud_config_bootstrap_schema")
        == agent.get("cloud_config_bootstrap_schema")
        and comparison_codex.get("cloud_config_bootstrap_policy")
        == agent.get("cloud_config_bootstrap_policy")
        and comparison_codex.get("cloud_config_bundle_sha256")
        == agent.get("cloud_config_bundle_sha256")
        and comparison_codex.get("cloud_config_host_runtime_closure_sha256")
        == agent.get("cloud_config_host_runtime_closure_sha256")
        and transport_treatments == FORMAL_AGENT_TRANSPORT_TREATMENTS
        and isinstance(agent_transport, dict)
        and agent.get("max_process_output_bytes")
        == agent_transport.get("max_process_output_bytes")
        and agent.get("structured_stream_output_limit_bytes")
        == agent_transport.get("structured_stream_output_limit_bytes")
        and agent.get("structured_stream_overflow_policy")
        == agent_transport.get("overflow_policy")
        and projected_runtime is not None
        and comparison_runtime == projected_runtime
        and comparison.get("evaluator_files_sha256") == evaluator
        and isinstance(apex, dict)
        and _SHA1.fullmatch(str(apex.get("commit") or ""))
        and apex.get("dirty") is False
        and _SHA256.fullmatch(str(apex.get("status_sha256") or ""))
        and _SHA256.fullmatch(
            str(apex.get("runtime_manifest_sha256") or "")
        )
        and isinstance(aka, dict)
        and _SHA1.fullmatch(str(aka.get("commit") or ""))
        and _SHA1.fullmatch(str(aka.get("tree") or ""))
        and aka.get("dirty") is False
        and aka.get("execution_manifest_schema") == EXECUTION_MANIFEST_SCHEMA
        and _SHA256.fullmatch(str(aka.get("execution_manifest_sha256") or ""))
        and manifest.get("formal_execution") == _FORMAL_LIVE_COMMITMENT
        and manifest.get("formal_execution_sha256")
        == _FORMAL_LIVE_COMMITMENT_SHA256
        and comparison.get("formal_execution") == _FORMAL_LIVE_COMMITMENT
        and comparison.get("formal_execution_sha256")
        == _FORMAL_LIVE_COMMITMENT_SHA256
        and run_config_file_valid
        and configuration.get("run_config_contract") == expected_run_config
        and run_config == expected_run_config
        and _campaign_task_paths_valid(manifest)
        and _revalidate_aka_runtime(manifest)
    )


def _load_verified_campaign_manifest(run_directory: Path) -> dict[str, Any]:
    """Load a live, scoreable v7 manifest; older generations always fail."""
    manifest_path = run_directory / "campaign_manifest.yaml"
    if not _safe_read_only_file(manifest_path):
        raise CampaignError("formal execution requires an immutable campaign manifest")
    manifest = _load_mapping(manifest_path, "campaign manifest")
    comparison = manifest.get("comparison_contract")
    comparison_digest = manifest.get("comparison_contract_sha256")
    comparison_schema = comparison.get("schema") if isinstance(comparison, dict) else None
    live_contract_valid = (
        comparison_schema == _COMPARISON_CONTRACT_SCHEMA
        and comparison.get("candidate_persistence_policy_id")
        == CANDIDATE_PERSISTENCE_POLICY
        and comparison.get("boundary_quiescence_policy_id")
        == BOUNDARY_QUIESCENCE_POLICY
        and comparison.get("agent_process_containment_policy_id")
        == AGENT_PROCESS_CONTAINMENT_POLICY
        and comparison.get("attempt_containment_policy_id")
        == ATTEMPT_CONTAINMENT_POLICY
        and _v7_manifest_contract_valid(manifest, comparison)
    )
    if (
        manifest.get("schema") != _CAMPAIGN_SCHEMA
        or not isinstance(comparison, dict)
        or not live_contract_valid
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
        "formal_execution_sha256": _FORMAL_LIVE_COMMITMENT_SHA256,
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


def _comparison_aka_runtime_snapshot(snapshot: Any) -> dict[str, Any] | None:
    """Project one per-run mount receipt into stable A/B contract evidence."""

    if not isinstance(snapshot, dict):
        return None
    receipt = snapshot.get("mount_receipt")
    mount = receipt.get("mount") if isinstance(receipt, dict) else None
    service = snapshot.get("runtime_service_evidence")
    engine = service.get("engine_evidence") if isinstance(service, dict) else None
    host_policy = receipt.get("host_access_policy") if isinstance(receipt, dict) else None
    ancestor = (
        host_policy.get("private_ancestor")
        if isinstance(host_policy, dict)
        else None
    )
    if not all(
        isinstance(value, dict)
        for value in (receipt, mount, service, engine, host_policy, ancestor)
    ):
        return None
    return {
        "schema": "aka.execution-snapshot-comparison/v2",
        "runtime_schema": snapshot.get("schema"),
        "manifest_sha256": snapshot.get("manifest_sha256"),
        "manifest_file_sha256": snapshot.get("manifest_file_sha256"),
        "mount_receipt_schema": snapshot.get("mount_receipt_schema"),
        "mount_contract": {
            "schema": receipt.get("schema"),
            "policy_id": receipt.get("policy_id"),
            "manifest_sha256": receipt.get("manifest_sha256"),
            "image_sha256": receipt.get("image_sha256"),
            "memfd_seals": receipt.get("memfd_seals"),
            "requested_mount_options": receipt.get(
                "requested_mount_options"
            ),
            "mount": {
                "filesystem_type": mount.get("filesystem_type"),
                "read_only": mount.get("read_only"),
                "root": mount.get("root"),
                "nested_mounts": mount.get("nested_mounts"),
            },
        },
        "host_access_contract": {
            "schema": host_policy.get("schema"),
            "policy_id": host_policy.get("policy_id"),
            "requested_mount_options": host_policy.get(
                "requested_mount_options"
            ),
            "private_ancestor": {
                key: ancestor.get(key) for key in ("uid", "gid", "mode")
            },
            "fuse_config": host_policy.get("fuse_config"),
            "mount_owner": host_policy.get("mount_owner"),
            "worker": host_policy.get("worker"),
            "docker_daemon": host_policy.get("docker_daemon"),
        },
        "engine_contract": {
            "service_schema": service.get("schema"),
            "service_policy_id": service.get("policy_id"),
            "engine_schema": engine.get("schema"),
            "engine_policy_id": engine.get("policy_id"),
            "requested_mount_options": engine.get(
                "requested_mount_options"
            ),
            "tools": engine.get("tools"),
        },
    }


def comparison_runtime_projection(runtime: Any) -> dict[str, Any] | None:
    """Return stable runtime evidence shared by matched formal campaign arms."""

    if not isinstance(runtime, dict):
        return None
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
    comparison_runtime["aka_execution_snapshot"] = (
        _comparison_aka_runtime_snapshot(runtime.get("aka_execution_snapshot"))
    )
    return comparison_runtime


def _comparison_contract(
    *,
    policy: CampaignPolicy,
    measurement: dict[str, Any],
    repositories: dict[str, Any],
    agent: dict[str, Any],
    runtime: dict[str, Any],
    evaluator: dict[str, Any],
    tasks: list[dict[str, Any]],
    run_config: dict[str, Any],
) -> dict[str, Any]:
    apex_treatment = _v7_apex_treatment_contract(repositories)
    if (
        apex_treatment is None
        or (
            agent.get("template") == "apex"
            and any(
                agent.get(key) != value
                for key, value in apex_treatment.items()
            )
        )
    ):
        raise CampaignError("comparison lacks a bound Apex runtime manifest")
    effective_codex = {
        key: (
            agent.get(key, AGENT_PROCESS_CONTAINMENT_POLICY)
            if key == "agent_process_containment_policy_id"
            else agent.get(key, ATTEMPT_CONTAINMENT_POLICY)
            if key == "attempt_containment_policy_id"
            else agent[key]
        )
        for key in (
            "backend",
            "model",
            "effort",
            "permission_mode",
            "inner_max_iterations",
            "attempt_timeout_seconds",
            "max_turns",
            "turn_policy",
            "agent_process_containment_policy_id",
            "attempt_containment_policy_id",
            "structured_stream_output_limit_bytes",
            "codex_version",
            "codex_binary_sha256",
            "backend_runtime_closure_schema",
            "backend_runtime_closure_sha256",
            "backend_runtime_closure",
            "cloud_config_bootstrap_schema",
            "cloud_config_bootstrap_policy",
            "cloud_config_bundle_sha256",
            "cloud_config_host_runtime_closure_sha256",
            "isolation",
        )
    }
    comparison_runtime = comparison_runtime_projection(runtime)
    if comparison_runtime is None:
        raise CampaignError("comparison runtime evidence is malformed")
    return {
        "schema": _COMPARISON_CONTRACT_SCHEMA,
        "formal_execution": dict(_FORMAL_LIVE_COMMITMENT),
        "formal_execution_sha256": _FORMAL_LIVE_COMMITMENT_SHA256,
        "objective_policy_id": _OBJECTIVE_POLICY_ID,
        "prompt_policy_id": _PROMPT_POLICY_ID,
        "candidate_persistence_policy_id": CANDIDATE_PERSISTENCE_POLICY,
        "boundary_quiescence_policy_id": BOUNDARY_QUIESCENCE_POLICY,
        "agent_process_containment_policy_id": AGENT_PROCESS_CONTAINMENT_POLICY,
        "attempt_containment_policy_id": ATTEMPT_CONTAINMENT_POLICY,
        "policy": asdict(policy),
        "measurement": measurement,
        "repositories": repositories,
        "apex_treatment": apex_treatment,
        "agent_transport_treatments": {
            key: dict(value)
            for key, value in FORMAL_AGENT_TRANSPORT_TREATMENTS.items()
        },
        "codex": effective_codex,
        "runtime": comparison_runtime,
        "evaluator_files_sha256": evaluator,
        "tasks": tasks,
        "run_config": run_config,
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
    aka_state, aka_runtime = _aka_state_from_environment(repo_root)
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
    apex_repository = repositories["apex"]
    agent = _agent_manifest(
        repo_root,
        agent_name,
        policy,
        apex_runtime_manifest_sha256=apex_repository["runtime_manifest_sha256"],
    )
    task_manifests = _task_manifests(task_config_paths)
    run_config = _run_config_contract(run_config_path, agent_name=agent_name)
    run_config_payload, canonical_run_config_path = _read_run_config_bytes(
        run_config_path
    )
    try:
        runtime_isolation = runtime_isolation_receipt()
    except CampaignIsolationError as error:
        raise CampaignError(f"formal runtime isolation is not proven: {error}") from error
    runtime = {
        "docker": _image_manifest(),
        "gpu": _gpu_inventory(eval_config, list(task_config_paths)),
        "isolation": runtime_isolation,
        "aka_execution_snapshot": aka_runtime,
    }
    evaluator = _evaluator_manifest(aka_state)
    comparison = _comparison_contract(
        policy=policy,
        measurement=measurement,
        repositories=repositories,
        agent=agent,
        runtime=runtime,
        evaluator=evaluator,
        tasks=task_manifests,
        run_config=run_config,
    )
    return {
        "schema": _CAMPAIGN_SCHEMA,
        "formal_execution": dict(_FORMAL_LIVE_COMMITMENT),
        "formal_execution_sha256": _FORMAL_LIVE_COMMITMENT_SHA256,
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
            "run_config_path": str(canonical_run_config_path),
            "run_config_sha256": _sha256_bytes(run_config_payload),
            "run_config_size_bytes": len(run_config_payload),
            "run_config_contract": run_config,
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
    if parse_campaign_policy(eval_config) is None:
        return None
    durable_run_config = _materialize_durable_run_config(
        run_directory, run_config_path
    )
    manifest = build_campaign_manifest(
        eval_config=eval_config,
        run_config_path=durable_run_config,
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
        "source_anti_tamper_sha256": None,
        "source_anti_tamper_source_manifest_sha256": None,
        "source_anti_tamper_rules_sha256": None,
        "selection_eligible": False,
        "measured_rate_per_ms": 0.0,
        "eligibility_errors": [],
    }
    receipt: dict[str, Any] | None = None
    source_delta_files: list[str] | None = None
    if receipt_path.exists():
        receipt, receipt_errors = _validate_session_receipt(
            receipt_path=receipt_path,
            workspace=workspace,
            run_directory=run_directory,
            expected_task_name=expected_task_name,
        )
        record["session_receipt"] = str(receipt_path.relative_to(run_directory))
        if receipt is not None:
            receipt_schema = receipt.get("schema")
            if receipt_schema == _CODEX_RECEIPT_SCHEMA:
                integrity = receipt.get("workspace_integrity")
                final_changes = (
                    integrity.get("final_changes")
                    if isinstance(integrity, dict)
                    else None
                )
                changed_files = (
                    final_changes.get("changed_files")
                    if isinstance(final_changes, dict)
                    else None
                )
                if isinstance(changed_files, list) and all(
                    isinstance(path, str) and path for path in changed_files
                ):
                    source_delta_files = changed_files
            record["session_receipt_sha256"] = _sha256_file(receipt_path)
            record["session_succeeded"] = receipt.get("session_succeeded") is True
            binding = {
                "schema": receipt.get("schema"),
                "campaign_binding": receipt.get("campaign_binding"),
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
                "attempt_process_cleanup": receipt.get(
                    "attempt_process_cleanup"
                ),
                "budgets": receipt.get("budgets"),
                "turn_budget": receipt.get("turn_budget"),
                "workspace_integrity": receipt.get("workspace_integrity"),
                "gpu": receipt.get("gpu"),
                "lineage": receipt.get("lineage"),
                "source_delta_files": source_delta_files,
            }
            if receipt.get("schema") == _APEX_RECEIPT_SCHEMA:
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
    source_anti_tamper = report.get("source_anti_tamper")
    if isinstance(source_anti_tamper, dict):
        record["source_anti_tamper_sha256"] = canonical_json_sha256(
            source_anti_tamper
        )
        record["source_anti_tamper_source_manifest_sha256"] = (
            source_anti_tamper.get("source_manifest_sha256")
        )
        record["source_anti_tamper_rules_sha256"] = source_anti_tamper.get(
            "rules_sha256"
        )
    evaluation_mode = report.get("evaluation_mode")
    agent_session_score_eligible = report.get("agent_session_score_eligible")
    agent_session_succeeded = report.get("agent_session_succeeded")
    agent_session_terminal_status = report.get("agent_session_terminal_status")
    apex_receipt = (
        receipt is not None and receipt.get("schema") == _APEX_RECEIPT_SCHEMA
    )
    no_candidate_attempt = bool(
        receipt is not None
        and not record["eligibility_errors"]
        and (
            (apex_receipt and receipt.get("terminal_status") == "no_gain")
            or (
                receipt.get("schema") == _CODEX_RECEIPT_SCHEMA
                and source_delta_files == []
            )
        )
    )
    if no_candidate_attempt:
        if evaluation_mode != "no_candidate_baseline_replay_v1":
            errors.append("no_candidate_evaluation_mode_mismatch")
        if agent_session_score_eligible is not False:
            errors.append("no_candidate_score_eligibility_mismatch")
        if agent_session_succeeded is not True:
            errors.append("no_candidate_session_success_mismatch")
        if agent_session_terminal_status != "no_gain":
            errors.append("no_candidate_terminal_status_mismatch")
    else:
        if evaluation_mode != "candidate_scoring_v1":
            errors.append("diagnostic_evaluation_not_scoreable")
        if agent_session_score_eligible is not True:
            errors.append("agent_session_not_score_eligible")
    if apex_receipt:
        receipt_terminal_status = receipt.get("terminal_status")
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
    eligible = not errors and not no_candidate_attempt
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
    try:
        manifest = _load_verified_campaign_manifest(run_directory)
    except (CampaignError, OSError, yaml.YAMLError):
        return None
    comparison = manifest.get("comparison_contract")
    if not isinstance(comparison, dict):
        return None
    codex = comparison.get("codex")
    return codex if isinstance(codex, dict) else None


def _expected_comparison_contract_sha256(run_directory: Path) -> str | None:
    try:
        manifest = _load_verified_campaign_manifest(run_directory)
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
    if digest != observed:
        return None
    return digest


def _expected_session_receipt_schema(run_directory: Path) -> str | None:
    """Resolve the only receipt schema allowed by the sealed agent manifest."""

    try:
        manifest = _load_verified_campaign_manifest(run_directory)
    except (CampaignError, OSError, yaml.YAMLError):
        return None
    agent = manifest.get("agent")
    if not isinstance(agent, dict):
        return None
    template = agent.get("template")
    expected = agent.get("session_receipt_schema")
    return resolve_session_receipt_schema(template, expected)


def _expected_apex_runtime_mount(
    run_directory: Path,
) -> dict[str, Any] | None:
    """Return the sealed Apex mount contract for the sole live generation."""

    try:
        manifest = _load_verified_campaign_manifest(run_directory)
    except (CampaignError, OSError, yaml.YAMLError):
        return {"invalid": True}
    comparison = manifest.get("comparison_contract")
    if (
        not isinstance(comparison, dict)
        or comparison.get("schema") != _COMPARISON_CONTRACT_SCHEMA
    ):
        return {"invalid": True}
    comparison = manifest["comparison_contract"]
    agent = manifest.get("agent")
    treatment = comparison.get("apex_treatment")
    repositories = comparison.get("repositories")
    apex = repositories.get("apex") if isinstance(repositories, dict) else None
    if not isinstance(agent, dict) or agent.get("template") != "apex":
        return None
    if not isinstance(treatment, dict):
        return {"invalid": True}
    return {
        "policy_id": treatment.get("apex_runtime_mount_policy_id"),
        "mount_receipt_schema": treatment.get("attempt_mount_receipt_schema"),
        "runtime_mount_schema": treatment.get("apex_runtime_mount_schema"),
        "runtime_manifest_sha256": treatment.get("runtime_manifest_sha256"),
        "repository": apex,
    }


def _receipt_digest_matches(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    material = dict(value)
    observed = material.pop("sha256", None)
    try:
        return (
            isinstance(observed, str)
            and _SHA256.fullmatch(observed) is not None
            and observed == _canonical_json_digest(material)
        )
    except (TypeError, ValueError):
        return False


def _absolute_receipt_path(value: Any, *, specific: bool = False) -> Path | None:
    if not isinstance(value, str):
        return None
    path = Path(value)
    if (
        not path.is_absolute()
        or path != Path(os.path.abspath(path))
        or ".." in path.parts
        or (
            specific
            and (
                len(path.parts) < 3
                or path in {Path("/tmp"), Path("/var/tmp"), Path("/dev/shm")}
            )
        )
    ):
        return None
    return path


def _canonical_receipt_directory(
    value: Any, *, specific: bool = False
) -> Path | None:
    path = _absolute_receipt_path(value, specific=specific)
    if path is None:
        return None
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError:
        return None
    if (
        resolved != path
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        return None
    return path


def _mount_identity_matches(identity: Any, expected: Path) -> bool:
    if not isinstance(identity, dict) or set(identity) != {
        "path",
        "device",
        "inode",
        "mode",
        "mount",
        "nested_mounts",
        "source",
    }:
        return False
    observed_path = _canonical_receipt_directory(identity.get("path"))
    mount = identity.get("mount")
    if (
        observed_path != expected
        or not isinstance(mount, dict)
        or set(mount)
        != {"mount_id", "parent_id", "major_minor", "root", "mount_point"}
        or type(mount.get("mount_id")) is not int
        or mount["mount_id"] <= 0
        or type(mount.get("parent_id")) is not int
        or mount["parent_id"] <= 0
        or not isinstance(mount.get("major_minor"), str)
        or re.fullmatch(r"[0-9]+:[0-9]+", mount["major_minor"]) is None
        or _absolute_receipt_path(mount.get("root")) is None
        or _absolute_receipt_path(mount.get("mount_point")) is None
        or not expected.is_relative_to(Path(mount["mount_point"]))
        or identity.get("nested_mounts") != []
        or identity.get("source") != "o_path_nofollow_bind_fd"
    ):
        return False
    try:
        metadata = expected.lstat()
    except OSError:
        return False
    return bool(
        type(identity.get("device")) is int
        and identity["device"] == metadata.st_dev
        and type(identity.get("inode")) is int
        and identity["inode"] == metadata.st_ino
        and type(identity.get("mode")) is int
        and identity["mode"] == stat.S_IMODE(metadata.st_mode)
    )


def _source_mount_identity_matches(identity: Any, expected: Path) -> bool:
    if not isinstance(identity, dict) or set(identity) != {
        "path",
        "device",
        "inode",
        "mode",
        "mount",
        "nested_mounts",
        "source",
    }:
        return False
    observed_path = _canonical_receipt_directory(identity.get("path"))
    mount = identity.get("mount")
    if (
        observed_path != expected
        or not _source_mount_record_valid(mount)
        or not expected.is_relative_to(Path(mount["mount_point"]))
        or identity.get("nested_mounts") != []
        or identity.get("source") != "o_path_nofollow_bind_fd"
    ):
        return False
    try:
        metadata = expected.lstat()
    except OSError:
        return False
    return bool(
        type(identity.get("device")) is int
        and identity["device"] == metadata.st_dev
        and type(identity.get("inode")) is int
        and identity["inode"] == metadata.st_ino
        and type(identity.get("mode")) is int
        and identity["mode"] == stat.S_IMODE(metadata.st_mode)
    )


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first.is_relative_to(second) or second.is_relative_to(first)


def _mount_record_valid(value: Any, *, expected_path: Path | None = None) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "mount_id",
        "parent_id",
        "major_minor",
        "root",
        "mount_point",
    }:
        return False
    return bool(
        type(value.get("mount_id")) is int
        and value["mount_id"] > 0
        and type(value.get("parent_id")) is int
        and value["parent_id"] > 0
        and isinstance(value.get("major_minor"), str)
        and re.fullmatch(r"[0-9]+:[0-9]+", value["major_minor"])
        and _absolute_receipt_path(value.get("root")) is not None
        and _absolute_receipt_path(value.get("mount_point")) is not None
        and (
            expected_path is None
            or Path(value["mount_point"]) == expected_path
        )
    )


def _source_mount_record_valid(value: Any) -> bool:
    base_fields = {
        "mount_id",
        "parent_id",
        "major_minor",
        "root",
        "mount_point",
    }
    if not isinstance(value, dict) or set(value) != base_fields | {
        "access",
        "filesystem_type",
        "source",
        "super_options",
    }:
        return False
    base = {field: value[field] for field in base_fields}
    return bool(
        _mount_record_valid(base)
        and value.get("access") in {"read_only", "read_write"}
        and isinstance(value.get("filesystem_type"), str)
        and bool(value["filesystem_type"])
        and isinstance(value.get("source"), str)
        and bool(value["source"])
        and isinstance(value.get("super_options"), list)
        and all(
            isinstance(option, str) and bool(option)
            for option in value["super_options"]
        )
    )


def _outer_bubblewrap_valid(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "policy",
        "canonical_path",
        "source",
        "sealed_exec",
    }:
        return False
    source = value.get("source")
    sealed = value.get("sealed_exec")
    canonical = Path("/usr/bin/bwrap")
    if (
        value.get("policy") != "canonical_source_to_sealed_memfd_exec_v1"
        or value.get("canonical_path") != str(canonical)
        or not isinstance(source, dict)
        or set(source)
        != {
            "device",
            "inode",
            "mode",
            "uid",
            "gid",
            "nlink",
            "size_bytes",
            "sha256",
        }
        or not isinstance(sealed, dict)
        or set(sealed) != {"transport", "size_bytes", "sha256", "seals"}
    ):
        return False
    try:
        metadata = canonical.lstat()
        digest = _sha256_file(canonical)
    except OSError:
        return False
    return bool(
        canonical.is_file()
        and not canonical.is_symlink()
        and source
        == {
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "mode": stat.S_IMODE(metadata.st_mode),
            "uid": metadata.st_uid,
            "gid": metadata.st_gid,
            "nlink": metadata.st_nlink,
            "size_bytes": metadata.st_size,
            "sha256": digest,
        }
        and sealed
        == {
            "transport": "sealed_memfd_proc_self_fd",
            "size_bytes": metadata.st_size,
            "sha256": digest,
            "seals": [
                "F_SEAL_WRITE",
                "F_SEAL_SHRINK",
                "F_SEAL_GROW",
                "F_SEAL_SEAL",
            ],
        }
    )


def _namespace_private_mount_valid(value: Any, expected: Path) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "path",
        "device",
        "inode",
        "access",
        "filesystem_type",
        "mount",
        "mount_options",
        "covered_mount_ids",
    }:
        return False
    options = value.get("mount_options")
    covered = value.get("covered_mount_ids")
    visible_mount = value.get("mount")
    return bool(
        value.get("path") == str(expected)
        and type(value.get("device")) is int
        and type(value.get("inode")) is int
        and value.get("access") == "read_write"
        and value.get("filesystem_type") == "tmpfs"
        and _mount_record_valid(value.get("mount"), expected_path=expected)
        and isinstance(options, list)
        and "rw" in options
        and "ro" not in options
        and isinstance(covered, list)
        and all(type(item) is int and item > 0 for item in covered)
        and covered == sorted(covered)
        and len(covered) == len(set(covered))
        and isinstance(visible_mount, dict)
        and visible_mount.get("mount_id") not in covered
    )


def _namespace_role_mount_valid(
    value: Any,
    *,
    role: str,
    expected: Path,
    source_identity: dict[str, Any],
    expected_access: str,
) -> bool:
    if not isinstance(value, dict) or set(value) != {"source", "target"}:
        return False
    source = value.get("source")
    target = value.get("target")
    if (
        not isinstance(source, dict)
        or set(source) != {"path", "device", "inode", "mount"}
        or not isinstance(target, dict)
        or set(target)
        != {
            "path",
            "device",
            "inode",
            "access",
            "mount",
            "mount_options",
            "covered_mount_ids",
        }
        or source.get("path") != str(expected)
        or source.get("device") != source_identity.get("device")
        or source.get("inode") != source_identity.get("inode")
        or not _source_mount_record_valid(source.get("mount"))
        or target.get("path") != str(expected)
        or target.get("device") != source.get("device")
        or target.get("inode") != source.get("inode")
        or target.get("access") != expected_access
        or not _mount_record_valid(target.get("mount"), expected_path=expected)
        or source_identity.get("mount") != source.get("mount")
    ):
        return False
    options = target.get("mount_options")
    covered = target.get("covered_mount_ids")
    if (
        not isinstance(options, list)
        or not isinstance(covered, list)
        or any(type(item) is not int or item <= 0 for item in covered)
        or covered != sorted(covered)
        or len(covered) != len(set(covered))
        or target["mount"]["mount_id"] in covered
    ):
        return False
    expected_option = "ro" if expected_access == "read_only" else "rw"
    forbidden_option = "rw" if expected_option == "ro" else "ro"
    source_mount = source["mount"]
    target_mount = target["mount"]
    try:
        relative = expected.relative_to(Path(source_mount["mount_point"]))
    except ValueError:
        return False
    inherited_exact_mount = Path(source_mount["mount_point"]) == expected
    expected_covered_count = 1 if role == "apex_runtime" and inherited_exact_mount else 0
    return bool(
        expected_option in options
        and forbidden_option not in options
        and target_mount["major_minor"] == source_mount["major_minor"]
        and Path(target_mount["root"]) == Path(source_mount["root"]) / relative
        and len(covered) == expected_covered_count
        and (
            role != "apex_runtime"
            or not inherited_exact_mount
            or source_mount["access"] == "read_only"
        )
    )


def _namespace_mounts_valid(
    value: Any,
    *,
    data_root: Path,
    concrete: dict[str, Path],
    identities: dict[str, Any],
) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "policy",
        "visible_mount_resolution_policy",
        "namespace_init_pid",
        "mount_namespace_id",
        "root",
        "campaign_data_root",
        "private_tmpfs",
        "roles",
        "declared_mount_points",
        "observed_mount_points_below_campaign_data",
        "closed_set",
        "aliases_absent",
    }:
        return False
    root = value.get("root")
    private = value.get("private_tmpfs")
    roles = value.get("roles")
    declared = sorted([str(data_root), *(str(path) for path in concrete.values())])
    if (
        value.get("policy") != "blocked_namespace_mount_attestation_v2"
        or value.get("visible_mount_resolution_policy")
        != "proc_root_o_path_fdinfo_mnt_id_v1"
        or type(value.get("namespace_init_pid")) is not int
        or value["namespace_init_pid"] <= 0
        or type(value.get("mount_namespace_id")) is not int
        or value["mount_namespace_id"] <= 0
        or not isinstance(root, dict)
        or set(root)
        != {
            "path",
            "device",
            "inode",
            "access",
            "mount",
            "mount_options",
            "covered_mount_ids",
        }
        or root.get("path") != "/"
        or root.get("access") != "read_only"
        or not _mount_record_valid(root.get("mount"), expected_path=Path("/"))
        or not isinstance(root.get("mount_options"), list)
        or "ro" not in root["mount_options"]
        or "rw" in root["mount_options"]
        or root.get("covered_mount_ids") != []
        or not _namespace_private_mount_valid(value.get("campaign_data_root"), data_root)
        or not isinstance(private, dict)
        or set(private) != {"tmp", "dev_shm"}
        or not _namespace_private_mount_valid(private.get("tmp"), Path("/tmp"))
        or not _namespace_private_mount_valid(private.get("dev_shm"), Path("/dev/shm"))
        or not isinstance(roles, dict)
        or set(roles) != {"persistent_writable", "read_only"}
        or value.get("declared_mount_points") != declared
        or value.get("observed_mount_points_below_campaign_data") != declared
        or value.get("closed_set") is not True
        or value.get("aliases_absent") is not True
    ):
        return False
    expected_groups = {
        "persistent_writable": {"apex_artifacts", "backend_home"},
        "read_only": {"scored_workspace", "sealed_task_contract", "apex_runtime"},
    }
    if any(
        not isinstance(roles.get(group), dict)
        or set(roles[group]) != names
        for group, names in expected_groups.items()
    ):
        return False
    for group, names in expected_groups.items():
        access = "read_write" if group == "persistent_writable" else "read_only"
        if any(
            not _namespace_role_mount_valid(
                roles[group][name],
                role=name,
                expected=concrete[name],
                source_identity=identities[name],
                expected_access=access,
            )
            for name in names
        ):
            return False
    targets = [
        roles[group][name]["target"]
        for group, names in expected_groups.items()
        for name in names
    ]
    observations = [
        root,
        value["campaign_data_root"],
        private["tmp"],
        private["dev_shm"],
        *targets,
    ]
    pairs = [(target["device"], target["inode"]) for target in targets]
    mount_ids = [observation["mount"]["mount_id"] for observation in observations]
    covered_ids = [
        mount_id
        for observation in observations
        for mount_id in observation["covered_mount_ids"]
    ]
    return bool(
        len(pairs) == len(set(pairs))
        and len(mount_ids) == len(set(mount_ids))
        and len(covered_ids) == len(set(covered_ids))
        and set(covered_ids).isdisjoint(mount_ids)
    )


def _apex_attempt_mount_role_errors(
    *,
    receipt: dict[str, Any],
    receipt_path: Path,
    workspace: Path | None,
    task_spec: dict[str, Any],
    contract_path: Path | None,
    runtime_root: Path,
) -> list[str]:
    """Validate the closed set of Apex outer-mount roles."""

    mounts = receipt.get("attempt_mounts")
    if (
        not isinstance(mounts, dict)
        or set(mounts)
        != {
            "schema",
            "campaign_data_root",
            "campaign_data_root_hidden",
            "campaign_data_identity",
            "outer_bubblewrap",
            "namespace_mounts",
            "roles",
            "sha256",
        }
        or mounts.get("schema") != ATTEMPT_MOUNT_RECEIPT_SCHEMA
        or mounts.get("campaign_data_root_hidden") is not True
        or not _outer_bubblewrap_valid(mounts.get("outer_bubblewrap"))
        or not _receipt_digest_matches(mounts)
    ):
        return ["apex_attempt_mount_role_contract_mismatch"]
    data_root = _canonical_receipt_directory(
        mounts.get("campaign_data_root"), specific=True
    )
    roles = mounts.get("roles")
    if (
        data_root is None
        or not _mount_identity_matches(
            mounts.get("campaign_data_identity"), data_root
        )
        or not isinstance(roles, dict)
        or set(roles) != {"persistent_writable", "read_only", "private_tmpfs"}
    ):
        return ["apex_attempt_mount_role_contract_mismatch"]

    configured_data_root = os.environ.get(
        "AGENT_KERNEL_ARENA_CAMPAIGN_DATA_ROOT"
    )
    if configured_data_root:
        configured = _canonical_receipt_directory(configured_data_root, specific=True)
        if configured != data_root:
            return ["apex_attempt_mount_role_contract_mismatch"]

    writable = roles.get("persistent_writable")
    read_only = roles.get("read_only")
    private_tmpfs = roles.get("private_tmpfs")
    if (
        not isinstance(writable, dict)
        or set(writable) != {"apex_artifacts", "backend_home"}
        or not isinstance(read_only, dict)
        or set(read_only)
        != {"scored_workspace", "sealed_task_contract", "apex_runtime"}
        or private_tmpfs
        != {
            "tmp": {"path": "/tmp", "persistence": "private"},
            "dev_shm": {"path": "/dev/shm", "persistence": "private"},
        }
    ):
        return ["apex_attempt_mount_role_contract_mismatch"]

    task_workspace = _canonical_receipt_directory(task_spec.get("workspace"))
    artifact_root = _canonical_receipt_directory(task_spec.get("results_dir"))
    try:
        expected_workspace = (
            _canonical_receipt_directory(str(workspace.resolve(strict=True)))
            if workspace is not None
            else None
        )
        receipt_below_data = receipt_path.resolve(strict=True).is_relative_to(
            data_root
        )
    except OSError:
        expected_workspace = None
        receipt_below_data = False
    task_contract = (
        _canonical_receipt_directory(str(contract_path.parent))
        if contract_path is not None
        else None
    )
    backend_home = _canonical_receipt_directory(
        str(receipt_path.parent / ".agent-home")
    )
    expected_runtime = _canonical_receipt_directory(str(runtime_root), specific=True)
    expected = {
        "apex_artifacts": artifact_root,
        "backend_home": backend_home,
        "scored_workspace": expected_workspace,
        "sealed_task_contract": task_contract,
        "apex_runtime": expected_runtime,
    }
    if (
        any(path is None for path in expected.values())
        or task_workspace != expected_workspace
        or any(
            not path.is_relative_to(data_root)
            for name, path in expected.items()
            if path is not None and name != "apex_runtime"
        )
        or expected_runtime is None
        or expected_runtime.is_relative_to(data_root)
        or not receipt_below_data
    ):
        return ["apex_attempt_mount_role_contract_mismatch"]

    identities = {
        "apex_artifacts": writable["apex_artifacts"],
        "backend_home": writable["backend_home"],
        "scored_workspace": read_only["scored_workspace"],
        "sealed_task_contract": read_only["sealed_task_contract"],
        "apex_runtime": read_only["apex_runtime"],
    }
    concrete = {name: path for name, path in expected.items() if path is not None}
    if any(
        not _source_mount_identity_matches(identities[name], path)
        for name, path in concrete.items()
    ):
        return ["apex_attempt_mount_role_contract_mismatch"]
    roots = list(concrete.values())
    if any(
        _paths_overlap(root, other)
        for index, root in enumerate(roots)
        for other in roots[index + 1 :]
    ):
        return ["apex_attempt_mount_role_contract_mismatch"]
    identity_values = list(identities.values())
    for index, identity in enumerate(identity_values):
        for other in identity_values[index + 1 :]:
            if (identity["device"], identity["inode"]) == (
                other["device"],
                other["inode"],
            ):
                return ["apex_attempt_mount_role_contract_mismatch"]
            mount = identity["mount"]
            other_mount = other["mount"]
            mount_root = Path(mount["root"])
            other_root = Path(other_mount["root"])
            if (
                mount["mount_id"] != other_mount["mount_id"]
                and mount["major_minor"] == other_mount["major_minor"]
                and _paths_overlap(mount_root, other_root)
            ):
                return ["apex_attempt_mount_role_contract_mismatch"]
    if not _namespace_mounts_valid(
        mounts.get("namespace_mounts"),
        data_root=data_root,
        concrete=concrete,
        identities=identities,
    ):
        return ["apex_attempt_mount_role_contract_mismatch"]
    return []


def _canonical_receipt_file(value: Any) -> Path | None:
    path = _absolute_receipt_path(value)
    if path is None:
        return None
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError:
        return None
    if (
        resolved != path
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
    ):
        return None
    return path


def _verified_runtime_manifest(root: Path, expected: Any) -> dict[str, Any] | None:
    if not isinstance(expected, str) or not _SHA256.fullmatch(expected):
        return None
    try:
        return verify_runtime_snapshot(root, expected)
    except (ApexRuntimeError, OSError, TypeError, ValueError):
        return None


def _apex_runtime_mount_errors(
    receipt: dict[str, Any],
    run_directory: Path,
    *,
    receipt_path: Path | None = None,
    workspace: Path | None = None,
    task_spec: dict[str, Any] | None = None,
    contract_path: Path | None = None,
) -> list[str]:
    expected = _expected_apex_runtime_mount(run_directory)
    if expected is None:
        return ["apex_runtime_mount_contract_missing"]
    mounts = receipt.get("attempt_mounts")
    runtime = receipt.get("apex_runtime_mount")
    apex = receipt.get("apex")
    if (
        receipt.get("schema") != _APEX_RECEIPT_SCHEMA
        or expected.get("mount_receipt_schema") != ATTEMPT_MOUNT_RECEIPT_SCHEMA
        or expected.get("runtime_mount_schema") != APEX_RUNTIME_MOUNT_SCHEMA
        or expected.get("policy_id") != APEX_RUNTIME_MOUNT_POLICY
        or not isinstance(mounts, dict)
        or mounts.get("schema") != ATTEMPT_MOUNT_RECEIPT_SCHEMA
        or mounts.get("campaign_data_root_hidden") is not True
        or not _receipt_digest_matches(mounts)
        or not isinstance(runtime, dict)
        or runtime.get("schema") != APEX_RUNTIME_MOUNT_SCHEMA
        or runtime.get("policy_id") != APEX_RUNTIME_MOUNT_POLICY
        or runtime.get("mode") != "read_only"
        or runtime.get("repository") != expected.get("repository")
        or runtime.get("runtime_manifest_sha256")
        != expected.get("runtime_manifest_sha256")
        or runtime.get("attempt_mounts_sha256") != mounts.get("sha256")
        or not _receipt_digest_matches(runtime)
        or not isinstance(apex, dict)
        or receipt_path is None
        or not isinstance(task_spec, dict)
    ):
        return ["apex_runtime_mount_contract_mismatch"]

    if set(runtime) != {
        "schema",
        "policy_id",
        "mode",
        "source_root",
        "root",
        "repository",
        "runtime_manifest_sha256",
        "runtime_manifest_path",
        "runtime_manifest_relative_path",
        "entrypoint",
        "python",
        "immutability",
        "attempt_mounts_sha256",
        "sha256",
    }:
        return ["apex_runtime_mount_contract_mismatch"]
    root = _canonical_receipt_directory(runtime.get("root"), specific=True)
    source_root = _absolute_receipt_path(
        runtime.get("source_root"), specific=True
    )
    configured_source_raw = os.environ.get(
        "AGENT_KERNEL_ARENA_APEX_SOURCE_ROOT"
    )
    configured_source = _absolute_receipt_path(
        configured_source_raw, specific=True
    )
    configured_execution_raw = os.environ.get("APEX_ROOT")
    configured_execution = _canonical_receipt_directory(
        configured_execution_raw, specific=True
    )
    configured_python_raw = os.environ.get("APEX_PYTHON")
    configured_python = _canonical_receipt_file(configured_python_raw)
    manifest_path = _canonical_receipt_file(runtime.get("runtime_manifest_path"))
    expected_manifest_path = root / "runtime_manifest.json" if root else None
    verified_manifest = (
        _verified_runtime_manifest(root, expected.get("runtime_manifest_sha256"))
        if root is not None
        else None
    )
    if (
        root is None
        or source_root is None
        or runtime.get("root") != str(root)
        or runtime.get("source_root") != str(source_root)
        or configured_source is None
        or configured_source_raw != str(configured_source)
        or source_root != configured_source
        or configured_execution != root / "repo"
        or configured_execution_raw != str(root / "repo")
        or configured_python != root / RUNTIME_WRAPPER_NAME
        or configured_python_raw != str(root / RUNTIME_WRAPPER_NAME)
        or _paths_overlap(root, source_root)
        or root.name != expected["runtime_manifest_sha256"]
        or runtime.get("runtime_manifest_relative_path") != "runtime_manifest.json"
        or manifest_path != expected_manifest_path
        or runtime.get("runtime_manifest_path") != str(expected_manifest_path)
        or verified_manifest is None
    ):
        return ["apex_runtime_mount_contract_mismatch"]

    manifest_git = verified_manifest.get("git")
    manifest_launcher = verified_manifest.get("launcher")
    manifest_system_python = (
        manifest_launcher.get("system_python")
        if isinstance(manifest_launcher, dict)
        else None
    )
    manifest_roots = verified_manifest.get("roots")
    execution = verified_manifest.get("execution")
    manifest_immutability = verified_manifest.get("immutability")
    apex_roots = (
        [
            item
            for item in manifest_roots
            if isinstance(item, dict) and item.get("role") == "apex"
        ]
        if isinstance(manifest_roots, list)
        else []
    )
    apex_source = (
        apex_roots[0].get("source") if len(apex_roots) == 1 else None
    )
    apex_files = apex_roots[0].get("files") if len(apex_roots) == 1 else None
    main_entries = (
        [
            item
            for item in apex_files
            if isinstance(item, dict) and item.get("path") == "main.py"
        ]
        if isinstance(apex_files, list)
        else []
    )
    main_entry = main_entries[0] if len(main_entries) == 1 else None
    expected_repository = expected.get("repository")
    if (
        not isinstance(manifest_git, dict)
        or not isinstance(expected_repository, dict)
        or any(
            manifest_git.get(key) != expected_repository.get(key)
            for key in ("commit", "dirty", "status_sha256")
        )
        or not isinstance(apex_source, dict)
        or apex_source.get("path") != str(source_root)
        or apex_roots[0].get("destination") != "repo"
        or not isinstance(main_entry, dict)
        or main_entry.get("type") != "file"
        or not isinstance(main_entry.get("sha256"), str)
        or _SHA256.fullmatch(main_entry["sha256"]) is None
        or not isinstance(manifest_system_python, dict)
        or set(manifest_system_python)
        != {"path", "binding", "size", "sha256", "mode", "device", "inode"}
        or manifest_system_python.get("binding")
        != "formal_docker_image_plus_attempt_receipt_v1"
        or not isinstance(manifest_system_python.get("sha256"), str)
        or _SHA256.fullmatch(manifest_system_python["sha256"]) is None
        or type(manifest_system_python.get("size")) is not int
        or manifest_system_python["size"] <= 0
        or not isinstance(execution, dict)
        or set(execution)
        != {
            "interpreter",
            "underlying_interpreter",
            "flags",
            "bootstrap",
            "bootstrap_policy_id",
            "bootstrap_sha256",
            "wrapper_policy_id",
            "wrapper_sha256",
            "wrapper_aliases",
            "entrypoint",
            "pythonpath",
            "site_hook_policy",
            "no_live_interpreter_fallback",
        }
        or execution.get("interpreter") != RUNTIME_WRAPPER_NAME
        or execution.get("underlying_interpreter") != "venv/bin/python"
        or execution.get("flags") != ["-I", "-S", "-u"]
        or execution.get("bootstrap") != RUNTIME_BOOTSTRAP_NAME
        or execution.get("bootstrap_policy_id")
        != RUNTIME_BOOTSTRAP_POLICY_ID
        or execution.get("bootstrap_sha256") != RUNTIME_BOOTSTRAP_SHA256
        or execution.get("wrapper_policy_id") != RUNTIME_WRAPPER_POLICY_ID
        or execution.get("wrapper_sha256") != RUNTIME_WRAPPER_SHA256
        or execution.get("wrapper_aliases")
        != [f"sealed-bin/{alias}" for alias in RUNTIME_WRAPPER_ALIASES]
        or execution.get("entrypoint") != "repo/main.py"
        or execution.get("site_hook_policy")
        != {
            "primary_invocation": "forced_isolated_no_site",
            "python_alias_children": "forced_isolated_no_site",
            "sys_executable_rebound_to_wrapper": True,
            "pth_execution_via_contract": False,
            "sitecustomize_execution_via_contract": False,
            "raw_interpreter_is_not_an_execution_contract": True,
        }
        or execution.get("no_live_interpreter_fallback") is not True
        or manifest_immutability
        != {
            "required_for_execution": True,
            "receipt_schema": RUNTIME_IMMUTABLE_MOUNT_SCHEMA,
            "receipt_policy_id": RUNTIME_IMMUTABLE_MOUNT_POLICY_ID,
            "image_input_schema": RUNTIME_IMAGE_INPUT_SCHEMA,
            "host_mode_bits_are_evidence_only": True,
            "runtime_service_evidence_schema": ENGINE_SERVICE_SCHEMA,
            "runtime_engine_evidence_schema": ENGINE_EVIDENCE_SCHEMA,
            "host_access_policy_schema": HOST_ACCESS_POLICY_SCHEMA,
            "requested_mount_options": [
                "ro", "nodev", "nosuid", "default_permissions",
                "allow_other", "subtype=squashfuse",
            ],
        }
    ):
        return ["apex_runtime_mount_contract_mismatch"]

    bootstrap_path = _canonical_receipt_file(str(root / RUNTIME_BOOTSTRAP_NAME))
    if (
        bootstrap_path != root / RUNTIME_BOOTSTRAP_NAME
        or _sha256_file(bootstrap_path) != RUNTIME_BOOTSTRAP_SHA256
    ):
        return ["apex_runtime_mount_contract_mismatch"]

    entrypoint = runtime.get("entrypoint")
    python = runtime.get("python")
    immutability = runtime.get("immutability")
    if (
        not isinstance(entrypoint, dict)
        or not isinstance(python, dict)
        or not isinstance(immutability, dict)
    ):
        return ["apex_runtime_mount_contract_mismatch"]
    entrypoint_relative = execution.get("entrypoint")
    interpreter_relative = execution.get("interpreter")
    underlying_relative = execution.get("underlying_interpreter")
    manifest_pythonpath = execution.get("pythonpath")
    if not all(
        isinstance(value, str)
        for value in (
            entrypoint_relative,
            interpreter_relative,
            underlying_relative,
        )
    ) or not isinstance(manifest_pythonpath, list):
        return ["apex_runtime_mount_contract_mismatch"]
    expected_launcher = root / str(interpreter_relative)
    launcher = _canonical_receipt_file(python.get("launcher_path"))
    expected_underlying = root / str(underlying_relative)
    underlying = _absolute_receipt_path(python.get("underlying_path"))
    expected_entrypoint = root / str(entrypoint_relative)
    observed_entrypoint = _canonical_receipt_file(entrypoint.get("path"))
    pythonpath = python.get("pythonpath")
    try:
        environment = runtime_environment(root, verified_manifest)
        expected_pythonpath = environment["PYTHONPATH"].split(os.pathsep)
        underlying_metadata = underlying.lstat() if underlying else None
        resolved_underlying = underlying.resolve(strict=True) if underlying else None
        pythonpath_roots = (
            [Path(value).resolve(strict=True) for value in pythonpath]
            if isinstance(pythonpath, list)
            else []
        )
    except (ApexRuntimeError, KeyError, OSError, TypeError, ValueError):
        return ["apex_runtime_mount_contract_mismatch"]
    expected_environment = {
        key: environment[key]
        for key in (
            "PATH",
            "APEX_RUNTIME_PYTHON",
            "PYTHONNOUSERSITE",
            "PYTHONSAFEPATH",
            "PYTHONDONTWRITEBYTECODE",
        )
    }
    if (
        set(entrypoint) != {"path", "relative_path", "sha256"}
        or set(python)
        != {
            "source_launcher_relative_path",
            "launcher_path",
            "launcher_sha256",
            "underlying_path",
            "underlying_sha256",
            "flags",
            "pythonpath",
            "environment",
        }
        or set(immutability)
        != {
            "schema",
            "policy_id",
            "receipt_sha256",
            "runtime_image_input_sha256",
            "image_sha256",
            "backing",
            "requested_mount_options",
            "runtime_service_evidence_sha256",
            "runtime_engine_evidence_sha256",
            "host_access_policy",
            "mount",
        }
        or observed_entrypoint != expected_entrypoint
        or entrypoint.get("path") != str(expected_entrypoint)
        or entrypoint.get("relative_path") != entrypoint_relative
        or not isinstance(entrypoint.get("sha256"), str)
        or not _SHA256.fullmatch(entrypoint["sha256"])
        or _sha256_file(expected_entrypoint) != entrypoint["sha256"]
        or main_entry.get("sha256") != entrypoint["sha256"]
        or apex.get("entrypoint") != str(expected_entrypoint)
        or apex.get("entrypoint_sha256") != entrypoint.get("sha256")
        or python.get("source_launcher_relative_path") != underlying_relative
        or launcher != expected_launcher
        or python.get("launcher_path") != str(expected_launcher)
        or not os.access(expected_launcher, os.X_OK)
        or python.get("launcher_sha256") != RUNTIME_WRAPPER_SHA256
        or _sha256_file(expected_launcher) != python.get("launcher_sha256")
        or underlying != expected_underlying
        or python.get("underlying_path") != str(expected_underlying)
        or underlying_metadata is None
        or not (
            stat.S_ISREG(underlying_metadata.st_mode)
            or stat.S_ISLNK(underlying_metadata.st_mode)
        )
        or resolved_underlying is None
        or not resolved_underlying.is_file()
        or not resolved_underlying.is_relative_to(root)
        or not os.access(expected_underlying, os.X_OK)
        or not isinstance(python.get("underlying_sha256"), str)
        or _SHA256.fullmatch(python["underlying_sha256"]) is None
        or _sha256_file(expected_underlying) != python["underlying_sha256"]
        or manifest_system_python.get("sha256") != python["underlying_sha256"]
        or python.get("flags") != ["-I", "-S", "-u"]
        or python.get("flags") != execution.get("flags")
        or python.get("environment") != expected_environment
        or not pythonpath_roots
        or pythonpath != expected_pythonpath
        or len(pythonpath_roots) != len(expected_pythonpath)
        or any(
            not path.is_dir() or not path.is_relative_to(root)
            for path in pythonpath_roots
        )
        or len(set(pythonpath_roots)) != len(pythonpath_roots)
        or set(apex)
        != {"entrypoint", "entrypoint_sha256", "python", "python_sha256"}
        or apex.get("python") != str(expected_launcher)
        or apex.get("python_sha256") != python.get("launcher_sha256")
    ):
        return ["apex_runtime_mount_contract_mismatch"]

    immutable_mount_receipt = {
        "schema": immutability.get("schema"),
        "policy_id": immutability.get("policy_id"),
        "root": str(root),
        "runtime_manifest_sha256": runtime.get("runtime_manifest_sha256"),
        "runtime_image_input_sha256": immutability.get(
            "runtime_image_input_sha256"
        ),
        "image_sha256": immutability.get("image_sha256"),
        "backing": immutability.get("backing"),
        "requested_mount_options": immutability.get(
            "requested_mount_options"
        ),
        "runtime_service_evidence_sha256": immutability.get(
            "runtime_service_evidence_sha256"
        ),
        "runtime_engine_evidence_sha256": immutability.get(
            "runtime_engine_evidence_sha256"
        ),
        "host_access_policy": immutability.get("host_access_policy"),
        "mount": immutability.get("mount"),
        "sha256": immutability.get("receipt_sha256"),
    }
    try:
        validate_apex_immutable_mount_receipt(
            root, verified_manifest, immutable_mount_receipt
        )
    except (ApexRuntimeError, OSError, TypeError, ValueError):
        return ["apex_runtime_mount_contract_mismatch"]
    return _apex_attempt_mount_role_errors(
        receipt=receipt,
        receipt_path=receipt_path,
        workspace=workspace,
        task_spec=task_spec,
        contract_path=contract_path,
        runtime_root=root,
    )


def _comparison_contract_receipt_errors(
    receipt: dict[str, Any], run_directory: Path, *, prefix: str
) -> list[str]:
    expected = _expected_comparison_contract_sha256(run_directory)
    if expected is None:
        return ["missing_immutable_campaign_comparison_contract"]
    if receipt.get("comparison_contract_sha256") != expected:
        return [f"{prefix}_comparison_contract_digest_mismatch"]
    return []


def _campaign_binding_receipt_errors(
    receipt: dict[str, Any],
    *,
    receipt_path: Path,
    run_directory: Path,
    expected_task_name: str | None,
) -> list[str]:
    """Rebuild the exact attempt-to-campaign binding from immutable evidence."""

    binding = receipt.get("campaign_binding")
    if not isinstance(binding, dict):
        return ["missing_attempt_campaign_binding"]
    if set(binding) != _CAMPAIGN_BINDING_KEYS:
        return ["attempt_campaign_binding_key_set_mismatch"]
    try:
        manifest = _load_verified_campaign_manifest(run_directory)
        manifest_path = (run_directory / "campaign_manifest.yaml").resolve(strict=True)
        if not _safe_read_only_file(manifest_path):
            raise CampaignError("campaign manifest is not immutable")
        configuration = manifest.get("configuration")
        tasks = configuration.get("tasks") if isinstance(configuration, dict) else None
        runtime = manifest.get("runtime")
        gpu = runtime.get("gpu") if isinstance(runtime, dict) else None
        mappings = gpu.get("task_mapping") if isinstance(gpu, dict) else None
        policy = manifest.get("policy")
        comparison = manifest.get("comparison_contract")
        comparison_policy = (
            comparison.get("policy") if isinstance(comparison, dict) else None
        )
        agent = manifest.get("agent")
        if (
            not isinstance(tasks, list)
            or not tasks
            or not isinstance(mappings, list)
            or len(mappings) != len(tasks)
            or not isinstance(policy, dict)
            or comparison_policy != policy
            or not isinstance(agent, dict)
        ):
            raise CampaignError("campaign binding source is malformed")
        task_index = binding.get("task_index")
        attempt_index = binding.get("attempt_index")
        attempt_count = binding.get("attempt_count")
        if (
            type(task_index) is not int
            or not 1 <= task_index <= len(tasks)
            or type(attempt_index) is not int
            or type(attempt_count) is not int
            or attempt_count != policy.get("attempts")
            or not 1 <= attempt_index <= attempt_count
        ):
            raise CampaignError("campaign binding indices are invalid")
        task = tasks[task_index - 1]
        mapping = mappings[task_index - 1]
        if not isinstance(task, dict) or not isinstance(mapping, dict):
            raise CampaignError("campaign binding task entry is malformed")
        task_name = task.get("task_name")
        if (
            not isinstance(task_name, str)
            or not task_name
            or task.get("task_index") != task_index
            or mapping.get("task_index") != task_index
            or mapping.get("task_name") != task_name
            or (
                expected_task_name is not None
                and task_name != expected_task_name
            )
        ):
            raise CampaignError("campaign binding task identity is invalid")
        task_config_path = Path(str(task.get("config_path") or "")).resolve(
            strict=True
        )
        task_config_metadata = task_config_path.lstat()
        if (
            not task_config_path.is_file()
            or task_config_path.is_symlink()
            or task_config_metadata.st_nlink != 1
            or _sha256_file(task_config_path) != task.get("config_sha256")
        ):
            raise CampaignError("campaign binding task config changed")
        package_files = _regular_tree_manifest(task_config_path.parent)
        package_digest = _sha256_bytes(
            json.dumps(
                package_files, sort_keys=True, separators=(",", ":")
            ).encode()
        )
        if (
            package_files != task.get("package_files_sha256")
            or package_digest != task.get("package_manifest_sha256")
        ):
            raise CampaignError("campaign binding task package changed")
        if expected_task_name is not None:
            expected_receipt_path = (
                run_directory
                / ".campaign_attempts"
                / campaign_task_path_component(task_name)
                / f"attempt_{attempt_index:02d}"
                / "session_receipt.json"
            ).resolve(strict=True)
            if receipt_path.resolve(strict=True) != expected_receipt_path:
                raise CampaignError("campaign binding receipt path is noncanonical")
        expected = {
            "schema": _CAMPAIGN_BINDING_SCHEMA,
            "formal_execution_sha256": manifest.get(
                "formal_execution_sha256"
            ),
            "campaign_manifest_path": str(manifest_path),
            "campaign_manifest_sha256": _sha256_file(manifest_path),
            "comparison_contract_sha256": manifest.get(
                "comparison_contract_sha256"
            ),
            "backend_runtime_closure_sha256": agent.get(
                "backend_runtime_closure_sha256"
            ),
            "task_package_manifest_sha256": package_digest,
            "task_config_sha256": task.get("config_sha256"),
            "task_name": task_name,
            "task_index": task_index,
            "total_tasks": len(tasks),
            "attempt_index": attempt_index,
            "attempt_count": attempt_count,
            "assigned_host_gpu_id": mapping.get("assigned_host_gpu_id"),
        }
    except (CampaignError, OSError, TypeError, ValueError):
        return ["attempt_campaign_binding_source_invalid"]
    return [] if binding == expected else ["attempt_campaign_binding_mismatch"]


def _expected_gpu_contract(run_directory: Path) -> dict[str, Any] | None:
    try:
        manifest = _load_verified_campaign_manifest(run_directory)
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


def _codex_cloud_config_bootstrap_valid(
    observed_codex: dict[str, Any],
    expected_codex: dict[str, Any],
) -> bool:
    bootstrap = observed_codex.get("cloud_config_bootstrap")
    expected_keys = {
        "schema",
        "policy",
        "relative_path",
        "present",
        "sha256",
        "size_bytes",
        "bundle_sha256",
        "signed_envelope_shape_validated",
        "payload_recorded",
    }
    return bool(
        isinstance(bootstrap, dict)
        and set(bootstrap) == expected_keys
        and bootstrap.get("schema") == CODEX_CLOUD_CONFIG_BOOTSTRAP_SCHEMA
        and bootstrap.get("schema")
        == expected_codex.get("cloud_config_bootstrap_schema")
        and bootstrap.get("policy") == CODEX_CLOUD_CONFIG_BOOTSTRAP_POLICY
        and bootstrap.get("policy")
        == expected_codex.get("cloud_config_bootstrap_policy")
        and bootstrap.get("relative_path")
        == ".codex/cloud-config-bundle-cache.json"
        and bootstrap.get("present") is True
        and isinstance(bootstrap.get("sha256"), str)
        and _SHA256.fullmatch(bootstrap["sha256"])
        and type(bootstrap.get("size_bytes")) is int
        and 0 < bootstrap["size_bytes"] <= 1024 * 1024
        and isinstance(bootstrap.get("bundle_sha256"), str)
        and _SHA256.fullmatch(bootstrap["bundle_sha256"])
        and bootstrap.get("bundle_sha256")
        == expected_codex.get("cloud_config_bundle_sha256")
        and bootstrap.get("signed_envelope_shape_validated") is True
        and bootstrap.get("payload_recorded") is False
    )


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
            if expected_schema == _APEX_RECEIPT_SCHEMA
            else "direct_codex_receipt_schema_generation_mismatch"
        )
        return receipt, [mismatch]

    errors.extend(
        _campaign_binding_receipt_errors(
            receipt,
            receipt_path=receipt_path,
            run_directory=run_directory,
            expected_task_name=expected_task_name,
        )
    )

    if expected_schema == _APEX_RECEIPT_SCHEMA:
        apex_receipt, apex_errors = _validate_apex_session_receipt(
            receipt=receipt,
            receipt_path=receipt_path,
            workspace=workspace,
            run_directory=run_directory,
            expected_task_name=expected_task_name,
            expected_receipt_schema=expected_schema,
        )
        return apex_receipt, sorted(set(errors + apex_errors))

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
    candidate_persistence = receipt.get("candidate_persistence")
    persistence = (
        candidate_persistence if isinstance(candidate_persistence, dict) else {}
    )
    checkpoint_termination = (
        persistence.get("termination") == "exact_turn_boundary"
    )
    cleanup = receipt.get("attempt_process_cleanup")
    if not isinstance(receipt.get("session_succeeded"), bool):
        errors.append("direct_codex_receipt_invalid_session_status")
    if receipt.get("session_succeeded") is True:
        allowed_exit = receipt.get("exit_code") == 0 or (
            checkpoint_termination
            and receipt.get("exit_code") == 128 + int(signal.SIGKILL)
            and isinstance(cleanup, dict)
            and cleanup.get("reason") == "exact_turn_boundary"
            and cleanup.get("teardown_mode") == "pidfd_sigkill"
            and cleanup.get("sigkill_sent") is True
        )
        if receipt.get("timed_out") is not False or not allowed_exit:
            errors.append("direct_codex_receipt_success_status_inconsistent")
        thread_id = receipt.get("thread_id")
        if not isinstance(thread_id, str) or not thread_id.strip():
            errors.append("direct_codex_receipt_missing_thread_id")

    cleanup_valid = _pid_namespace_cleanup_valid(
        cleanup,
        exit_code=receipt.get("exit_code"),
        allowed_reasons={"normal_exit", "exact_turn_boundary"},
    )
    if not cleanup_valid:
        errors.append("direct_codex_attempt_namespace_not_verified_absent")
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
    expected_turn_policy = TURN_POLICY
    normal_turn_budget = (
        isinstance(turn_budget, dict)
        and turn_budget.get("policy") == expected_turn_policy
        and turn_budget.get("max_turns") == FORMAL_MATCHED_MAX_TURNS
        and isinstance(turn_budget.get("observed_turns"), int)
        and 1 <= turn_budget["observed_turns"] <= FORMAL_MATCHED_MAX_TURNS
        and turn_budget.get("budget_exceeded") is False
        and turn_budget.get("enforcement_failed") is False
        and turn_budget.get("stop_reason") is None
        and turn_budget.get("exact_boundary_reached") is False
        and turn_budget.get("post_boundary_turns") == 0
    )
    checkpoint_turn_budget = (
        isinstance(turn_budget, dict)
        and turn_budget.get("policy") == TURN_POLICY
        and turn_budget.get("max_turns") == FORMAL_MATCHED_MAX_TURNS
        and turn_budget.get("observed_turns") == FORMAL_MATCHED_MAX_TURNS
        and turn_budget.get("exact_boundary_reached") is True
        and turn_budget.get("post_boundary_turns") == 0
        and turn_budget.get("budget_exceeded") is False
        and turn_budget.get("enforcement_failed") is False
        and turn_budget.get("stop_reason") == "exact_turn_boundary"
        and checkpoint_termination
    )
    if not (normal_turn_budget or checkpoint_turn_budget):
        errors.append("direct_codex_turn_budget_invalid")

    workspace_integrity = receipt.get("workspace_integrity")
    if (
        not isinstance(workspace_integrity, dict)
        or workspace_integrity.get("policy") != "declared_source_only_sanitized_v1"
        or workspace_integrity.get("passed") is not True
        or workspace_integrity.get("errors") != []
    ):
        errors.append("direct_codex_workspace_integrity_invalid")
    checkpoint = persistence.get("checkpoint")
    expected_termination = (
        "exact_turn_boundary" if checkpoint_turn_budget else "completed"
    )
    if (
        not isinstance(candidate_persistence, dict)
        or persistence.get("schema") != "aka.candidate-persistence-receipt/v4"
        or persistence.get("policy_id") != CANDIDATE_PERSISTENCE_POLICY
        or persistence.get("attempt_contained") is not True
        or persistence.get("termination") != expected_termination
        or (checkpoint_turn_budget and not isinstance(checkpoint, dict))
        or (not checkpoint_turn_budget and checkpoint is not None)
    ):
        errors.append("direct_codex_candidate_persistence_invalid")
    if (
        persistence.get("agent_process_containment_policy_id")
        != AGENT_PROCESS_CONTAINMENT_POLICY
        or persistence.get("attempt_containment_policy_id")
        != ATTEMPT_CONTAINMENT_POLICY
    ):
        errors.append("direct_codex_attempt_containment_policy_mismatch")
    if checkpoint_turn_budget:
        boundary_snapshot = persistence.get("boundary_snapshot")
        output_tail = persistence.get("output_tail")
        persisted_cleanup = persistence.get("attempt_cleanup")
        boundary_resolution = persistence.get("boundary_resolution")
        namespace_route = (
            boundary_resolution == "pid_namespace_destroyed_before_snapshot"
            and persisted_cleanup == cleanup
            and _pid_namespace_cleanup_valid(
                cleanup,
                exit_code=receipt.get("exit_code"),
                allowed_reasons={"exact_turn_boundary"},
            )
            and isinstance(cleanup, dict)
            and cleanup.get("boundary_signal")
            == {
                "attempted": True,
                "stdout_character_offset": (
                    output_tail.get("stdout_character_offset")
                    if isinstance(output_tail, dict)
                    else None
                ),
            }
        )
        if not namespace_route:
            errors.append("direct_codex_checkpoint_suspension_invalid")
        if (
            not isinstance(boundary_snapshot, dict)
            or boundary_snapshot.get("policy_id") != ATTEMPT_CONTAINMENT_POLICY
            or boundary_snapshot.get("complete") is not True
            or boundary_snapshot.get("errors") != []
            or not isinstance(boundary_snapshot.get("manifest_sha256"), str)
            or not _SHA256.fullmatch(boundary_snapshot["manifest_sha256"])
            or not isinstance(boundary_snapshot.get("files"), list)
            or not boundary_snapshot["files"]
            or boundary_snapshot.get("capture_mode")
            != "post_namespace_teardown_checkpoint"
        ):
            errors.append("direct_codex_boundary_snapshot_invalid")
        if (
            not isinstance(output_tail, dict)
            or output_tail.get("policy") != "retained_and_digested_v1"
            or not isinstance(output_tail.get("stdout_size_bytes"), int)
            or output_tail["stdout_size_bytes"] < 0
            or not isinstance(output_tail.get("stdout_sha256"), str)
            or not _SHA256.fullmatch(output_tail["stdout_sha256"])
            or output_tail.get("post_boundary_turns") != 0
            or output_tail.get("capture_truncated") is not False
            or output_tail.get("readers_completed") is not True
        ):
            errors.append("direct_codex_boundary_output_tail_invalid")
        if isinstance(checkpoint, dict) and all(
            isinstance(value, dict)
            for value in (boundary_snapshot, output_tail, cleanup)
        ):
            digest_fields = {
                "boundary_snapshot_sha256": boundary_snapshot,
                "output_tail_sha256": output_tail,
                "attempt_cleanup_sha256": cleanup,
            }
            if any(
                checkpoint.get(field) != _canonical_json_digest(value)
                for field, value in digest_fields.items()
            ):
                errors.append("direct_codex_checkpoint_evidence_digest_mismatch")

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
            "runtime_closure_sha256": "backend_runtime_closure_sha256",
            "version": "codex_version",
            "model": "model",
            "effort": "effort",
        }
        if any(
            observed_codex.get(receipt_key) != expected_codex.get(contract_key)
            for receipt_key, contract_key in comparisons.items()
        ):
            errors.append("direct_codex_identity_contract_mismatch")
        if not _codex_cloud_config_bootstrap_valid(
            observed_codex, expected_codex
        ):
            errors.append("direct_codex_cloud_config_bootstrap_invalid")
        if (
            expected_codex.get("max_turns") != FORMAL_MATCHED_MAX_TURNS
            or expected_codex.get("turn_policy") != expected_turn_policy
            or expected_codex.get("agent_process_containment_policy_id")
            != AGENT_PROCESS_CONTAINMENT_POLICY
            or expected_codex.get("attempt_containment_policy_id")
            != ATTEMPT_CONTAINMENT_POLICY
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
            or invocation.get("turn_policy") != expected_turn_policy
            or invocation.get("structured_stream_output_limit_bytes")
            != 16 * 1024 * 1024
            or invocation.get("candidate_persistence_policy_id")
            != CANDIDATE_PERSISTENCE_POLICY
            or (
                invocation.get("agent_process_containment_policy_id")
                != AGENT_PROCESS_CONTAINMENT_POLICY
                or invocation.get("attempt_containment_policy_id")
                != ATTEMPT_CONTAINMENT_POLICY
                or not isinstance(invocation.get("attempt_process_boundary"), dict)
                or not isinstance(cleanup, dict)
                or invocation.get("attempt_process_boundary")
                != cleanup.get("boundary")
            )
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

        if checkpoint_turn_budget and isinstance(candidate_persistence, dict):
            output_tail = persistence.get("output_tail")
            try:
                raw_stdout_text = expected_artifacts["raw_stdout"].read_text(
                    encoding="utf-8"
                )
            except (OSError, UnicodeDecodeError):
                errors.append("direct_codex_boundary_output_tail_unreadable")
            else:
                offset = (
                    output_tail.get("stdout_character_offset")
                    if isinstance(output_tail, dict)
                    else None
                )
                if not isinstance(offset, int) or not 0 <= offset <= len(
                    raw_stdout_text
                ):
                    errors.append("direct_codex_boundary_output_tail_invalid")
                else:
                    tail_bytes = raw_stdout_text[offset:].encode("utf-8")
                    if (
                        output_tail.get("stdout_size_bytes") != len(tail_bytes)
                        or output_tail.get("stdout_sha256")
                        != _sha256_bytes(tail_bytes)
                    ):
                        errors.append(
                            "direct_codex_boundary_output_tail_digest_mismatch"
                        )

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
            before_paths = set(before_manifest)
            after_paths = set(after_manifest)
            recomputed_created = sorted(after_paths - before_paths)
            recomputed_deleted = sorted(before_paths - after_paths)
            recomputed_changed = sorted(
                path
                for path in before_paths & after_paths
                if before_manifest[path] != after_manifest[path]
            )
            recomputed_unauthorized = sorted(
                path for path in recomputed_changed if path not in editable
            )
            recomputed_mode_changes = sorted(
                path
                for path in recomputed_changed
                if path in editable
                and isinstance(before_manifest.get(path), dict)
                and isinstance(after_manifest.get(path), dict)
                and before_manifest[path].get("mode")
                != after_manifest[path].get("mode")
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
                or final_changes.get("created_files") != recomputed_created
                or final_changes.get("deleted_files") != recomputed_deleted
                or final_changes.get("changed_files") != recomputed_changed
                or final_changes.get("unauthorized_changed_files")
                != recomputed_unauthorized
                or final_changes.get("editable_mode_changes")
                != recomputed_mode_changes
                or bool(recomputed_created)
                or bool(recomputed_deleted)
                or bool(recomputed_unauthorized)
                or bool(recomputed_mode_changes)
                or not set(recomputed_changed).issubset(editable)
            ):
                errors.append("direct_codex_sanitized_manifest_contract_mismatch")
            if (
                not isinstance(sanitization, dict)
                or sanitization.get("performed") is not True
                or sanitization.get("candidate_retained") is not True
                or sanitization.get("baseline_restored") is not True
            ):
                errors.append("direct_codex_workspace_sanitization_unverified")
            if checkpoint_turn_budget:
                checkpoint = (
                    persistence.get("checkpoint")
                )
                expected_checkpoint = {
                    "before_manifest_sha256": before_digest,
                    "after_manifest_sha256": after_digest,
                    "changed_files": final_changes.get("changed_files", [])
                    if isinstance(final_changes, dict)
                    else None,
                    "editable_files": workspace_integrity.get("editable_files"),
                }
                expected_checkpoint.update(
                    {
                        "boundary_snapshot_sha256": _canonical_json_digest(
                            persistence.get("boundary_snapshot")
                        ),
                        "output_tail_sha256": _canonical_json_digest(
                            persistence.get("output_tail")
                        ),
                        "attempt_cleanup_sha256": _canonical_json_digest(
                            persistence.get("attempt_cleanup")
                        ),
                    }
                )
                if checkpoint != expected_checkpoint:
                    errors.append("direct_codex_checkpoint_manifest_mismatch")

    return receipt, sorted(set(errors))


def _pid_namespace_cleanup_valid(
    cleanup: Any,
    *,
    exit_code: Any,
    allowed_reasons: set[str],
    required_procfs: str = "private_attempt_procfs",
) -> bool:
    """Validate the common direct/Apex per-attempt containment proof."""
    return attempt_cleanup_verified(
        cleanup,
        exit_code=exit_code,
        allowed_reasons=allowed_reasons,
        required_procfs=required_procfs,
    )


def _apex_agent_containment_valid(
    receipt: Any, *, forced_stop: bool
) -> bool:
    """Recompute Apex's inner backend PID-namespace proof from typed fields."""
    if not isinstance(receipt, dict):
        return False
    positive_fields = (
        "namespace_init_host_pid",
        "namespace_init_starttime",
        "namespace_init_inner_pid",
        "pid_namespace_inode",
        "mount_namespace_inode",
        "ipc_namespace_inode",
        "user_namespace_inode",
    )
    launcher = receipt.get("launcher_path")
    try:
        launcher_path = Path(launcher) if isinstance(launcher, str) else None
        launcher_valid = (
            launcher_path is not None
            and launcher_path.is_absolute()
            and launcher_path.is_file()
            and not launcher_path.is_symlink()
            and _sha256_file(launcher_path) == receipt.get("launcher_sha256")
        )
    except OSError:
        launcher_valid = False
    terminal_verified = receipt.get("terminal_status_verified") is True
    terminal_absent = (
        receipt.get("terminal_status_absent_after_sigkill") is True
    )
    terminal_route = terminal_verified != terminal_absent
    terminal_fields_typed = (
        type(receipt.get("terminal_status_verified")) is bool
        and type(receipt.get("terminal_status_absent_after_sigkill")) is bool
    )
    common = (
        receipt.get("schema") == "apex.agent-process-containment/v1"
        and receipt.get("policy_id") == AGENT_PROCESS_CONTAINMENT_POLICY
        and launcher_valid
        and all(
            type(receipt.get(field)) is int and receipt[field] > 0
            for field in positive_fields
        )
        and receipt.get("namespace_init_inner_pid") == 1
        and receipt.get("private_procfs_verified") is True
        and receipt.get("pidfd_opened") is True
        and receipt.get("namespace_init_exit_verified") is True
        and receipt.get("wrapper_exit_verified") is True
        and receipt.get("wrapper_force_killed") is False
        and terminal_fields_typed
        and terminal_route
        and receipt.get("status_eof_verified") is True
        and receipt.get("namespace_membership_scan_complete") is True
        and receipt.get("live_namespace_members_after") == []
        and receipt.get("namespace_empty_verified") is True
    )
    if not common:
        return False
    if forced_stop:
        return (
            receipt.get("termination_reason") == "stdout_budget_boundary"
            and receipt.get("teardown_mode") == "pidfd_sigkill"
            and receipt.get("pidfd_sigkill_sent") is True
            and terminal_route
        )
    return (
        receipt.get("termination_reason") == "natural_exit"
        and receipt.get("teardown_mode") == "natural_exit"
        and receipt.get("pidfd_sigkill_sent") is False
        and terminal_verified
        and not terminal_absent
    )


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


def _validate_bundle_snapshot(
    *, path: Path, result: dict[str, Any], task_spec: dict[str, Any]
) -> list[str]:
    """Recompute an immutable Apex bundle snapshot without its live CAS tree."""
    try:
        snapshot = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return ["apex_checkpoint_bundle_snapshot_unreadable"]
    manifest = snapshot.get("manifest") if isinstance(snapshot, dict) else None
    patches = snapshot.get("patches") if isinstance(snapshot, dict) else None
    if (
        not isinstance(snapshot, dict)
        or snapshot.get("schema") != "aka.apex-source-bundle-snapshot/v1"
        or not isinstance(manifest, dict)
        or not isinstance(patches, list)
        or not patches
    ):
        return ["apex_checkpoint_bundle_snapshot_invalid"]
    patch_bytes: list[bytes] = []
    observed_paths: list[str] = []
    for item in patches:
        if not isinstance(item, dict):
            return ["apex_checkpoint_bundle_snapshot_invalid"]
        relative = item.get("path")
        if not isinstance(relative, str):
            return ["apex_checkpoint_bundle_snapshot_invalid"]
        parsed = PurePosixPath(relative)
        if (
            parsed.is_absolute()
            or parsed.as_posix() != relative
            or any(part in {"", ".", ".."} for part in parsed.parts)
        ):
            return ["apex_checkpoint_bundle_snapshot_invalid"]
        try:
            content = base64.b64decode(item.get("content_base64", ""), validate=True)
        except (ValueError, TypeError):
            return ["apex_checkpoint_bundle_snapshot_invalid"]
        if (
            item.get("size_bytes") != len(content)
            or item.get("sha256") != _sha256_bytes(content)
        ):
            return ["apex_checkpoint_bundle_snapshot_hash_mismatch"]
        observed_paths.append(relative)
        patch_bytes.append(content)
    declared_patches = manifest.get("patches")
    if (
        len(observed_paths) != len(set(observed_paths))
        or not isinstance(declared_patches, list)
        or [
            (item.get("path"), item.get("sha256"))
            for item in declared_patches
            if isinstance(item, dict)
        ]
        != [
            (item["path"], item["sha256"])
            for item in patches
        ]
    ):
        return ["apex_checkpoint_bundle_snapshot_manifest_mismatch"]
    digest = hashlib.sha256(
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for content in patch_bytes:
        digest.update(content)
    bundle_digest = digest.hexdigest()
    if (
        snapshot.get("bundle_digest") != bundle_digest
        or result.get("bundle_digest") != bundle_digest
        or manifest.get("task_id") != task_spec.get("task_id")
        or manifest.get("baseline", {}).get("file_hashes")
        != task_spec.get("baseline", {}).get("file_hashes")
        or manifest.get("changed_files") != result.get("changed_files")
        or manifest.get("delivery") != {"mode": "bundle", "applied": False}
    ):
        return ["apex_checkpoint_bundle_digest_or_contract_mismatch"]
    return []


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


def _apex_checkpoint_gate_chain_errors(
    *,
    events: list[dict[str, Any]],
    result: dict[str, Any],
    persistence: dict[str, Any],
    lineage: dict[str, Any],
    bundle_path: Path,
    task_spec: dict[str, Any],
) -> list[str]:
    """Recompute the post-boundary freeze, trusted gates, and bundle binding."""
    errors = _validate_bundle_snapshot(
        path=bundle_path, result=result, task_spec=task_spec
    )
    checkpoint = persistence.get("checkpoint")
    if not isinstance(checkpoint, dict):
        return sorted(set(errors + ["apex_checkpoint_lineage_missing"]))
    agent_events = [
        event
        for event in events
        if event["event_type"] == "agent_completed"
        and event["event_id"] == checkpoint.get("agent_event_id")
    ]
    if len(agent_events) != 1:
        errors.append("apex_checkpoint_agent_event_mismatch")
        return sorted(set(errors))
    agent = agent_events[0]
    attempt_id = agent["payload"].get("attempt_id")
    required_types = (
        "candidate_frozen",
        "action.artifacts_ready",
        "compile_result",
        "correctness_result",
        "safety_result",
        "performance_command_result",
        "measurement_result",
        "reward_committed",
        "action.verified",
        "decision",
        "run.succeeded",
    )
    selected: list[dict[str, Any]] = []
    declared_chain = checkpoint.get("gate_chain")
    if not isinstance(declared_chain, dict) or set(declared_chain) != set(required_types):
        errors.append("apex_checkpoint_gate_chain_manifest_invalid")
        return sorted(set(errors))
    for event_type in required_types:
        matches = [
            event
            for event in events
            if event["event_type"] == event_type
            and event["event_id"] == declared_chain.get(event_type)
            and (
                event_type == "run.succeeded"
                or event["payload"].get(
                    "attempt_id", event["payload"].get("action_id")
                )
                == attempt_id
            )
        ]
        if len(matches) != 1:
            errors.append("apex_checkpoint_gate_chain_event_mismatch")
            return sorted(set(errors))
        selected.append(matches[0])
    if not (
        agent["sequence"] < selected[0]["sequence"]
        and [event["sequence"] for event in selected]
        == sorted(event["sequence"] for event in selected)
    ):
        errors.append("apex_checkpoint_gate_chain_order_invalid")
    candidate, artifacts_ready, compiled, correct, safety, performance, measured, reward, verified, decision, finished = selected
    candidate_bindings = candidate["payload"].get("artifacts")
    if (
        candidate["payload"].get("changed_files") != result.get("changed_files")
        or not isinstance(candidate_bindings, list)
        or not candidate_bindings
        or any(
            not isinstance(binding, dict) or binding.get("role") != "candidate"
            for binding in candidate_bindings
        )
        or not artifacts_ready["payload"].get("artifact_refs")
    ):
        errors.append("apex_checkpoint_candidate_freeze_invalid")
    if compiled["payload"].get("passed") is not True:
        errors.append("apex_checkpoint_compile_gate_failed")
    if correct["payload"].get("passed") is not True:
        errors.append("apex_checkpoint_correctness_gate_failed")
    if (
        safety["payload"].get("allowed_to_measure") is not True
        or safety["payload"].get("promotion_eligible") is not True
    ):
        errors.append("apex_checkpoint_safety_gate_failed")
    if (
        performance["payload"].get("passed") is not True
        or performance["payload"].get("runtime") != "normal_uninstrumented"
        or performance["payload"].get("status")
        != "command_completed_without_robust_timing_grade"
    ):
        errors.append("apex_checkpoint_normal_performance_gate_failed")
    if (
        measured["payload"].get("measurement_status") != "valid"
        or measured["payload"].get("evidence_class") != "measured"
        or reward["payload"].get("evidence_class") != "measured"
        or isinstance(reward["payload"].get("scalar_reward"), bool)
        or not isinstance(reward["payload"].get("scalar_reward"), (int, float))
    ):
        errors.append("apex_checkpoint_measurement_reward_invalid")
    if (
        not isinstance(verified["payload"].get("verification_id"), str)
        or not _SHA256.fullmatch(verified["payload"]["verification_id"])
        or decision["event_id"] != result.get("internal_verdict_ref")
        or decision["payload"].get("verdict") != "keep"
        or decision["payload"].get("bundle_digest") != result.get("bundle_digest")
        or finished["payload"].get("reason") != "candidate_ready"
    ):
        errors.append("apex_checkpoint_decision_bundle_invalid")
    artifact_sha = _sha256_file(bundle_path)
    artifact_size = bundle_path.stat().st_size
    expected_bundle = {
        "bundle_digest": result.get("bundle_digest"),
        "snapshot_sha256": artifact_sha,
        "snapshot_size_bytes": artifact_size,
    }
    if (
        any(checkpoint.get(key) != value for key, value in expected_bundle.items())
        or lineage.get("bundle") != expected_bundle
    ):
        errors.append("apex_checkpoint_bundle_lineage_mismatch")
    return sorted(set(errors))


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


def _apex_run_control_errors(
    task_spec: dict[str, Any], *, expected_turn_policy: str
) -> list[str]:
    """Validate the caller-owned formal budget and verifier command contract."""

    control = task_spec.get("caller_run_control")
    commands = task_spec.get("commands")
    if not isinstance(control, dict) or not isinstance(commands, dict):
        return ["apex_caller_run_control_invalid"]
    turn_budget = control.get("structured_turn_budget")
    interpreter = control.get("python_interpreter")
    verifier_argv = control.get("verifier_argv")
    persistence_valid = (
        control.get("candidate_persistence_policy_id")
        == CANDIDATE_PERSISTENCE_POLICY
        and "candidate_persistence" not in control
    )
    if (
        control.get("schema") != "aka.apex-caller-run-control/v1"
        or control.get("deliverable_versions") != 1
        or not persistence_valid
        or control.get("process_containment_policy_id")
        != AGENT_PROCESS_CONTAINMENT_POLICY
        or turn_budget
        != {
            "policy": expected_turn_policy,
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


def _apex_checkpoint_evidence_errors(
    *,
    transcript: dict[str, Any],
    payload: dict[str, Any],
    persistence: Any,
    attempt_cleanup: Any,
    status: str,
) -> list[str]:
    """Validate typed inner containment before source may become persistent."""
    termination = transcript.get("termination")
    semantic_events = transcript.get("semantic_events")
    if not isinstance(termination, dict) or not isinstance(semantic_events, list):
        return ["apex_checkpoint_termination_evidence_invalid"]
    observed = sum(
        1
        for event in semantic_events
        if isinstance(event, dict)
        and event.get("kind") in {"agent_message", "tool_called"}
    )
    kind = payload.get("termination_kind")
    exact = kind == "exact_turn_boundary"
    budget_overrun = status == "budget_exhausted" and kind == "turn_overrun"
    expected_fields = {
        "kind": "termination_kind",
        "reason": "termination_reason",
        "capture_status": "capture_status",
        "candidate_capture_allowed": "candidate_capture_allowed",
        "observer_stop_sent": "observer_stop_sent",
        "observed_turns": "observed_turns",
    }
    process_containment = payload.get("process_containment")
    expected_discarded_tail = {
        "lines": payload.get("discarded_stdout_lines"),
        "bytes": payload.get("discarded_stdout_bytes"),
        "sha256": payload.get("discarded_stdout_sha256"),
    }
    tail_lines = payload.get("discarded_stdout_lines")
    tail_bytes = payload.get("discarded_stdout_bytes")
    tail_sha256 = payload.get("discarded_stdout_sha256")
    tail_present = (
        isinstance(tail_lines, int)
        and not isinstance(tail_lines, bool)
        and tail_lines > 0
    ) or (
        isinstance(tail_bytes, int)
        and not isinstance(tail_bytes, bool)
        and tail_bytes > 0
    )
    tail_invalid = (
        isinstance(tail_lines, bool)
        or not isinstance(tail_lines, int)
        or tail_lines < 0
        or isinstance(tail_bytes, bool)
        or not isinstance(tail_bytes, int)
        or tail_bytes < 0
        or (
            tail_present
            and (
                tail_lines == 0
                or tail_bytes == 0
                or not isinstance(tail_sha256, str)
                or not _SHA256.fullmatch(tail_sha256)
            )
        )
        or (not tail_present and tail_sha256 is not None)
    )
    invalid = (
        transcript.get("schema") != "apex.agent-transcript/v3"
        or any(
            termination.get(transcript_key) != payload.get(payload_key)
            for transcript_key, payload_key in expected_fields.items()
        )
        or termination.get("max_turns") != FORMAL_MATCHED_MAX_TURNS
        or termination.get("turn_policy") != TURN_POLICY
        or termination.get("process_containment") != process_containment
        or termination.get("discarded_stdout_tail")
        != expected_discarded_tail
        or tail_invalid
        or payload.get("observed_turns") != observed
        or payload.get("capture_status") != "complete"
        or payload.get("timed_out") is not False
        or not isinstance(payload.get("observer_stop_sent"), bool)
        or payload.get("process_containment_policy_id")
        != AGENT_PROCESS_CONTAINMENT_POLICY
        or not _apex_agent_containment_valid(
            process_containment, forced_stop=exact or budget_overrun
        )
        or (
            status != "budget_exhausted"
            and (
                kind not in {"completed", "exact_turn_boundary"}
                or payload.get("candidate_capture_allowed") is not True
            )
        )
        or (
            status == "budget_exhausted"
            and (
                not budget_overrun
                or payload.get("candidate_capture_allowed") is not False
                or payload.get("termination_reason") != "max_turns_overrun"
                or observed <= FORMAL_MATCHED_MAX_TURNS
                or payload.get("observer_stop_sent") is not True
                or payload.get("exit_code") != 128 + int(signal.SIGKILL)
            )
        )
        or (
            status != "budget_exhausted"
            and exact
            and (
                observed != FORMAL_MATCHED_MAX_TURNS
                or payload.get("termination_reason")
                != "max_turns_exact_boundary"
                or payload.get("observer_stop_sent") is not True
                or payload.get("exit_code") != 128 + int(signal.SIGKILL)
            )
        )
        or (
            status != "budget_exhausted"
            and not exact
            and (
                not 1 <= observed <= FORMAL_MATCHED_MAX_TURNS
                or payload.get("termination_reason") is not None
                or payload.get("exit_code") != 0
                or payload.get("observer_stop_sent") is not False
            )
        )
    )
    if invalid:
        return ["apex_checkpoint_termination_evidence_invalid"]
    expected_persistence = {
        "schema": "aka.candidate-persistence-receipt/v4",
        "policy_id": CANDIDATE_PERSISTENCE_POLICY,
        "agent_process_containment_policy_id": (
            AGENT_PROCESS_CONTAINMENT_POLICY
        ),
        "agent_process_containment_sha256": _canonical_json_digest(
            process_containment
        ),
        "attempt_containment_policy_id": ATTEMPT_CONTAINMENT_POLICY,
        "attempt_process_cleanup_sha256": _canonical_json_digest(
            attempt_cleanup
        ),
        "termination_kind": kind,
        "termination_reason": payload.get("termination_reason"),
        "capture_status": "complete",
        "candidate_capture_allowed": payload.get("candidate_capture_allowed"),
        "observer_stop_sent": payload.get("observer_stop_sent"),
        "discarded_stdout_tail": expected_discarded_tail,
        "observed_turns": observed,
    }
    if not isinstance(persistence, dict) or any(
        persistence.get(key) != value for key, value in expected_persistence.items()
    ):
        return ["apex_candidate_persistence_receipt_invalid"]
    checkpoint = persistence.get("checkpoint")
    if exact and status == "candidate_ready":
        if not isinstance(checkpoint, dict):
            return ["apex_checkpoint_lineage_missing"]
    elif checkpoint is not None:
        return ["apex_unexpected_checkpoint_lineage"]
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
    budget_failure_reason = "agent_turn_budget_overrun"
    if receipt_schema != _APEX_RECEIPT_SCHEMA:
        errors.append("apex_receipt_schema_mismatch")
    if receipt_schema != expected_receipt_schema:
        errors.append("apex_receipt_schema_generation_mismatch")
    candidate_persistence = receipt.get("candidate_persistence")
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
    cleanup = receipt.get("attempt_process_cleanup")
    if (
        receipt.get("agent_process_containment_policy_id")
        != AGENT_PROCESS_CONTAINMENT_POLICY
        or receipt.get("attempt_containment_policy_id")
        != ATTEMPT_CONTAINMENT_POLICY
    ):
        errors.append("apex_attempt_containment_policy_mismatch")
    cleanup_valid = _pid_namespace_cleanup_valid(
        cleanup,
        exit_code=receipt.get("exit_code"),
        allowed_reasons={"normal_exit"},
        required_procfs="trusted_orchestrator_inherited_procfs",
    )
    if not cleanup_valid:
        errors.append("apex_outer_attempt_namespace_not_verified_absent")
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
    apex_expected_turn_policy = (
        expected_codex.get("turn_policy")
        if isinstance(expected_codex, dict)
        else None
    )
    observed_codex = receipt.get("codex")
    if not isinstance(expected_codex, dict):
        errors.append("missing_immutable_campaign_codex_contract")
    elif not isinstance(observed_codex, dict):
        errors.append("apex_codex_identity_contract_mismatch")
    else:
        if any(
            observed_codex.get(receipt_key) != expected_codex.get(contract_key)
            for receipt_key, contract_key in {
                "binary_sha256": "codex_binary_sha256",
                "runtime_closure_sha256": "backend_runtime_closure_sha256",
                "version": "codex_version",
                "model": "model",
                "effort": "effort",
            }.items()
        ):
            errors.append("apex_codex_identity_contract_mismatch")
        if not _codex_cloud_config_bootstrap_valid(
            observed_codex, expected_codex
        ):
            errors.append("apex_codex_cloud_config_bootstrap_invalid")
    if isinstance(expected_codex, dict) and (
        expected_codex.get("max_turns") != FORMAL_MATCHED_MAX_TURNS
        or expected_codex.get("turn_policy") != TURN_POLICY
        or expected_codex.get("agent_process_containment_policy_id")
        != AGENT_PROCESS_CONTAINMENT_POLICY
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

    expected_artifacts = {
        "task_spec": "task_spec.json",
        "apex_stdout": "apex_stdout.txt",
        "apex_stderr": "apex_stderr.txt",
        "apex_result": "apex_result.json",
        "event_journal": "event_journal.sqlite",
        "agent_transcript": "agent_transcript.json",
        "original_arena_prompt": "original_arena_prompt.txt",
        "agent_prompt": "agent_prompt.txt",
    }
    if terminal_status == "candidate_ready":
        expected_artifacts["source_bundle"] = "source_bundle_snapshot.json"
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
    campaign_binding = receipt.get("campaign_binding")
    if task_spec.get("campaign_binding") != campaign_binding:
        errors.append("apex_task_spec_campaign_binding_mismatch")
    errors.extend(
        _apex_runtime_mount_errors(
            receipt,
            run_directory,
            receipt_path=receipt_path,
            workspace=workspace,
            task_spec=task_spec,
            contract_path=contract_path,
        )
    )
    errors.extend(
        _apex_instruction_adaptation_errors(
            receipt=receipt,
            task_spec=task_spec,
            original_prompt_path=artifacts["original_arena_prompt"],
        )
    )
    errors.extend(
        _apex_run_control_errors(
            task_spec,
            expected_turn_policy=str(apex_expected_turn_policy),
        )
    )
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
    if (
        not isinstance(lineage, dict)
        or lineage.get("campaign_binding_sha256")
        != _canonical_json_digest(receipt.get("campaign_binding"))
    ):
        errors.append("apex_lineage_campaign_binding_mismatch")
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
            result.get("reason_code") != budget_failure_reason
            or not isinstance(result_error, dict)
            or result_error.get("reason_code") != budget_failure_reason
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
            or invocation != receipt.get("invocation")
        )
        errors.extend(
            _apex_checkpoint_evidence_errors(
                transcript=transcript,
                payload=payload,
                persistence=candidate_persistence,
                attempt_cleanup=cleanup,
                status=status,
            )
        )
        if outcome_invalid:
            errors.append("apex_agent_completion_receipt_mismatch")
        if (
            transcript.get("schema") != "apex.agent-transcript/v3"
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
            or invocation.get("schema") != "apex.agent-invocation/v3"
            or invocation.get("cli_name") != "codex"
            or invocation.get("cli_version")
            != (expected_codex or {}).get("codex_version")
            or invocation.get("entrypoint_sha256")
            != (expected_codex or {}).get("codex_binary_sha256")
            or invocation.get("runtime_closure_sha256")
            != (expected_codex or {}).get("backend_runtime_closure_sha256")
            or invocation.get("max_turns") != FORMAL_MATCHED_MAX_TURNS
            or invocation.get("turn_policy") != apex_expected_turn_policy
            or invocation.get("process_containment_policy_id")
            != AGENT_PROCESS_CONTAINMENT_POLICY
            or not isinstance(argv, list)
            or not {
                "--strict-config",
                "--ignore-user-config",
                "--ignore-rules",
                "--ephemeral",
            }.issubset(set(argv or []))
        ):
            errors.append("apex_inner_codex_invocation_contract_mismatch")
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
        != budget_failure_reason
    ):
        errors.append("apex_internal_verdict_ref_invalid")
    elif status == "candidate_ready" and (
        matching_verdicts[0]["event_type"] != "decision"
        or matching_verdicts[0]["payload"].get("verdict") != "keep"
    ):
        errors.append("apex_internal_verdict_ref_invalid")
    if events:
        expected_head_type = "run.failed" if failed_terminal else "run.succeeded"
        expected_head_reason = (
            budget_failure_reason
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
    if status == "candidate_ready":
        bundle_path = artifacts.get("source_bundle")
        if not isinstance(bundle_path, Path):
            errors.append("apex_checkpoint_bundle_snapshot_missing")
        elif (
            isinstance(candidate_persistence, dict)
            and candidate_persistence.get("termination_kind")
            == "exact_turn_boundary"
            and isinstance(lineage, dict)
        ):
            errors.extend(
                _apex_checkpoint_gate_chain_errors(
                    events=events,
                    result=result,
                    persistence=candidate_persistence,
                    lineage=lineage,
                    bundle_path=bundle_path,
                    task_spec=task_spec,
                )
            )
        else:
            errors.extend(
                _validate_bundle_snapshot(
                    path=bundle_path,
                    result=result,
                    task_spec=task_spec,
                )
            )
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


def _formal_source_anti_tamper_errors(
    workspace: Path,
    report: dict[str, Any],
) -> list[str]:
    evidence = report.get("source_anti_tamper")
    if not isinstance(evidence, dict):
        return ["missing_formal_source_anti_tamper_report"]
    errors: list[str] = []
    if evidence.get("schema") != FORMAL_SOURCE_ANTI_TAMPER_SCHEMA:
        errors.append("formal_source_anti_tamper_schema_mismatch")
    if evidence.get("policy") != FORMAL_SOURCE_ANTI_TAMPER_POLICY:
        errors.append("formal_source_anti_tamper_policy_mismatch")
    if evidence.get("rules_sha256") != FORMAL_SOURCE_ANTI_TAMPER_RULES_SHA256:
        errors.append("formal_source_anti_tamper_rules_mismatch")
    if evidence.get("verdict") != "PASS":
        errors.append("formal_source_anti_tamper_not_passed")
    source_manifest_sha256 = evidence.get("source_manifest_sha256")
    if (
        not isinstance(source_manifest_sha256, str)
        or len(source_manifest_sha256) != 64
        or evidence.get("expected_source_manifest_sha256")
        != source_manifest_sha256
    ):
        errors.append("formal_source_manifest_anchor_mismatch")
    try:
        task_config = _load_mapping(workspace / "config.yaml", "attempt task config")
        recomputed = inspect_formal_source_anti_tamper(
            workspace,
            task_config,
            expected_source_manifest_sha256=source_manifest_sha256,
        )
    except (CampaignError, OSError, ValueError, TypeError):
        errors.append("formal_source_anti_tamper_recomputation_failed")
    else:
        if recomputed != evidence:
            errors.append("formal_source_anti_tamper_report_mismatch")
    return sorted(set(errors))


def _evaluation_eligibility_errors(
    workspace: Path, report: dict[str, Any]
) -> list[str]:
    errors: list[str] = _formal_source_anti_tamper_errors(workspace, report)
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
    codex_contract = comparison_contract.get("codex")
    backend_runtime_closure_sha256 = (
        codex_contract.get("backend_runtime_closure_sha256")
        if isinstance(codex_contract, dict)
        else None
    )
    if not isinstance(backend_runtime_closure_sha256, str) or not _SHA256.fullmatch(
        backend_runtime_closure_sha256
    ):
        raise CampaignError("campaign lacks a bound backend runtime closure")
    attempt_root = (
        run_directory
        / ".campaign_attempts"
        / campaign_task_path_component(task_name)
    )
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
            "formal_execution_sha256": _FORMAL_LIVE_COMMITMENT_SHA256,
            "index": attempt,
            "count": policy.attempts,
            "fresh_session": True,
            "timeout_seconds": policy.attempt_timeout_seconds,
            "apex_internal_allowance_seconds": policy.apex_internal_allowance_seconds,
            "task_deadline_monotonic": deadline,
            "receipt_path": str(receipt_path),
            "comparison_contract_sha256": comparison_contract_sha256,
            "backend_runtime_closure_sha256": backend_runtime_closure_sha256,
            "task_package_manifest_sha256": task_binding[
                "package_manifest_sha256"
            ],
            "task_config_sha256": task_binding["config_sha256"],
            "task_name": task_name,
            "task_index": task_index,
            "total_tasks": total_tasks,
            "assigned_host_gpu_id": assigned_gpu,
            "campaign_manifest_path": str(campaign_manifest_path.resolve()),
            "campaign_manifest_sha256": campaign_manifest_sha256,
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
        "formal_execution_sha256": _FORMAL_LIVE_COMMITMENT_SHA256,
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
    # Formal postprocessing requires an entirely clean three-attempt campaign.
    # Fail here too, rather than publishing a canonical projection that the
    # strict validator is guaranteed to reject later.
    completed = completed and task_evidence["failure_reasons"] == []
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
        "selected_source_anti_tamper_sha256": selected[
            "source_anti_tamper_sha256"
        ],
        "selected_source_manifest_sha256": selected[
            "source_anti_tamper_source_manifest_sha256"
        ],
        "source_anti_tamper_rules_sha256": selected[
            "source_anti_tamper_rules_sha256"
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
    "FORMAL_AGENT_TRANSPORT_TREATMENTS",
    "FORMAL_LIVE_EXECUTION_SHA256",
    "build_campaign_manifest",
    "campaign_task_path_component",
    "comparison_runtime_projection",
    "deterministic_task_gpu_mapping",
    "ensure_campaign_manifest",
    "ordered_gpu_pool",
    "parse_campaign_policy",
    "resolve_session_receipt_schema",
    "run_matched_task_campaign",
    "validate_formal_task_binding",
]
