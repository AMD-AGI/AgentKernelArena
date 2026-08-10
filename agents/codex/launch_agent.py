# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import hashlib
import json
import logging
import math
import os
import shlex
import shutil
import signal
import stat
import subprocess
import threading
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from agents import register_agent
from src.campaign import CampaignError, campaign_task_path_component
from src.module_registration import AgentType, load_prompt_builder
from src.runtime_env import build_subprocess_env
from src.agent_turn_budget import (
    AGENT_PROCESS_CONTAINMENT_POLICY,
    BOUNDARY_QUIESCENCE_POLICY,
    CANDIDATE_PERSISTENCE_POLICY,
    FORMAL_MATCHED_MAX_TURNS,
    AgentTurnBudget,
    TURN_POLICY,
)
from src.campaign_isolation import (
    ATTEMPT_CONTAINMENT_POLICY,
    attempt_cleanup_verified,
    attempt_command_pass_fds,
    codex_cloud_config_bootstrap_receipt,
    establish_attempt_boundary,
    finalize_attempt_boundary,
    formal_gpu_evidence,
    is_formal_campaign,
    isolated_environment,
    prepare_attempt_home,
    release_attempt_command_fds,
    wrap_attempt_command,
)


_RECEIPT_SCHEMA = "agentkernelarena.codex-attempt-receipt/v6"
_TERM_GRACE_SECONDS = 10.0
_KILL_GRACE_SECONDS = 5.0
_SUSPEND_GRACE_SECONDS = 2.0
_SUSPEND_STABLE_POLLS = 2
_DEFAULT_OUTPUT_LIMIT = 16 * 1024 * 1024
_MAX_EVENT_LINE_CHARS = 1024 * 1024
_MAX_WORKSPACE_FILES = 20_000
_MAX_WORKSPACE_BYTES = 2 * 1024 * 1024 * 1024
_LOWER_HEX = frozenset("0123456789abcdef")
_CAMPAIGN_BINDING_SCHEMA = "aka.attempt-campaign-binding/v1"
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


class CodexSessionError(RuntimeError):
    """Raised when a direct Codex session is not trustworthy or successful."""


class CodexSessionTimeout(CodexSessionError):
    """Raised after a timed-out Codex attempt tree has been cleaned up."""


def integrate_agent_config(
    prompt: str,
    agent_config: dict[str, Any],
    python_path: str | None,
) -> str:
    """Append agent-specific guidance to the prompt."""
    max_iters = agent_config.get("max_iterations")
    if max_iters is not None:
        prompt = prompt.rstrip() + f"\n\nFor this optimization, you must iterate up to {max_iters} versions."
    if python_path:
        prompt = prompt.rstrip() + (
            f"\n\nUse this Python interpreter: `{python_path}`. "
            f"When running pytest, use `{python_path} -m pytest` instead of bare `pytest`."
        )
    return prompt


def _prompt_agent_config(
    agent_config: dict[str, Any], eval_config: dict[str, Any]
) -> dict[str, Any]:
    """Return campaign-specific prompt settings without changing global defaults."""
    effective = dict(agent_config)
    if (eval_config.get("campaign") or {}).get("comparison") == "apex_vs_codex":
        effective["max_iterations"] = int(agent_config.get("campaign_max_iterations", 0))
    return effective


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _comparison_contract_sha256(
    eval_config: dict[str, Any], *, formal_campaign: bool
) -> str | None:
    if not formal_campaign:
        return None
    attempt = eval_config.get("campaign_attempt")
    digest = (
        attempt.get("comparison_contract_sha256")
        if isinstance(attempt, dict)
        else None
    )
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in _LOWER_HEX for character in digest)
    ):
        raise CodexSessionError(
            "formal direct Codex attempt lacks a valid comparison contract digest"
        )
    return digest


def _runtime_closure_sha256(
    eval_config: dict[str, Any], *, formal_campaign: bool
) -> str | None:
    if not formal_campaign:
        return None
    attempt = eval_config.get("campaign_attempt")
    digest = (
        attempt.get("backend_runtime_closure_sha256")
        if isinstance(attempt, dict)
        else None
    )
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in _LOWER_HEX for character in digest)
    ):
        raise CodexSessionError(
            "formal direct Codex attempt lacks a backend runtime closure digest"
        )
    return digest


def _valid_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in _LOWER_HEX for character in value)
    )


def _read_immutable_campaign_manifest(path: Path) -> tuple[dict[str, Any], str]:
    """Read canonical manifest bytes without following the final path component."""

    if not path.is_absolute():
        raise CodexSessionError("campaign manifest path must be absolute")
    try:
        canonical = path.resolve(strict=True)
        if canonical != path:
            raise CodexSessionError("campaign manifest path must be canonical")
        descriptor = os.open(
            canonical,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_mode & 0o222
            ):
                raise CodexSessionError(
                    "campaign manifest must be a single-link read-only regular file"
                )
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                payload = stream.read()
        finally:
            os.close(descriptor)
        manifest = yaml.safe_load(payload.decode("utf-8")) or {}
    except CodexSessionError:
        raise
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise CodexSessionError(
            f"cannot read immutable campaign manifest: {error}"
        ) from error
    if not isinstance(manifest, dict):
        raise CodexSessionError("campaign manifest must contain a mapping")
    return manifest, _sha256_bytes(payload)


def _task_package_manifest(root: Path) -> tuple[dict[str, str], str]:
    if not root.is_dir() or root.is_symlink():
        raise CodexSessionError("campaign task package root is unsafe")
    files: dict[str, str] = {}
    try:
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root).as_posix()
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise CodexSessionError(
                    f"campaign task package contains a symlink: {relative}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                continue
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise CodexSessionError(
                    f"campaign task package contains an unsafe file: {relative}"
                )
            files[relative] = _sha256_file(path)
    except OSError as error:
        raise CodexSessionError(
            f"cannot hash campaign task package: {error}"
        ) from error
    if not files:
        raise CodexSessionError("campaign task package contains no files")
    digest = _sha256_bytes(_canonical_json_bytes(files))
    return files, digest


def _campaign_binding(
    eval_config: dict[str, Any],
    task_config_path: Path,
    *,
    formal_campaign: bool,
) -> dict[str, Any] | None:
    """Validate and normalize the runner-owned attempt binding before spawn."""

    if not formal_campaign:
        return None
    attempt = eval_config.get("campaign_attempt")
    if not isinstance(attempt, dict):
        raise CodexSessionError("formal direct Codex attempt binding is missing")
    raw_manifest_path = attempt.get("campaign_manifest_path")
    if not isinstance(raw_manifest_path, str):
        raise CodexSessionError("formal direct Codex campaign manifest path is missing")
    manifest_path = Path(raw_manifest_path)
    manifest, manifest_digest = _read_immutable_campaign_manifest(manifest_path)
    comparison = manifest.get("comparison_contract")
    comparison_digest = (
        _sha256_bytes(_canonical_json_bytes(comparison))
        if isinstance(comparison, dict)
        else None
    )
    configuration = manifest.get("configuration")
    tasks = configuration.get("tasks") if isinstance(configuration, dict) else None
    runtime = manifest.get("runtime")
    gpu = runtime.get("gpu") if isinstance(runtime, dict) else None
    mappings = gpu.get("task_mapping") if isinstance(gpu, dict) else None
    policy = manifest.get("policy")
    agent = manifest.get("agent")
    comparison_codex = (
        comparison.get("codex") if isinstance(comparison, dict) else None
    )
    if (
        manifest.get("schema") != "aka.matched-campaign/v1"
        or not isinstance(tasks, list)
        or not tasks
        or not isinstance(mappings, list)
        or len(mappings) != len(tasks)
        or not isinstance(policy, dict)
        or not isinstance(agent, dict)
        or not isinstance(comparison_codex, dict)
        or comparison.get("policy") != policy
        or manifest.get("comparison_contract_sha256") != comparison_digest
        or agent.get("backend_runtime_closure_sha256")
        != comparison_codex.get("backend_runtime_closure_sha256")
    ):
        raise CodexSessionError("formal direct Codex campaign manifest is malformed")
    try:
        task_names: list[str] = []
        task_components: list[str] = []
        for expected_index, (manifest_task, manifest_mapping) in enumerate(
            zip(tasks, mappings), 1
        ):
            if not isinstance(manifest_task, dict) or not isinstance(
                manifest_mapping, dict
            ):
                raise CodexSessionError(
                    "formal direct Codex campaign task mapping is malformed"
                )
            manifest_name = manifest_task.get("task_name")
            if (
                not isinstance(manifest_name, str)
                or not manifest_name
                or manifest_task.get("task_index") != expected_index
                or manifest_mapping.get("task_index") != expected_index
                or manifest_mapping.get("task_name") != manifest_name
                or not isinstance(
                    manifest_mapping.get("assigned_host_gpu_id"), str
                )
            ):
                raise CodexSessionError(
                    "formal direct Codex campaign task mapping is malformed"
                )
            task_names.append(manifest_name)
            task_components.append(campaign_task_path_component(manifest_name))
    except CampaignError as error:
        raise CodexSessionError(
            f"formal direct Codex campaign task path is unsafe: {error}"
        ) from error
    if (
        len(task_names) != len(set(task_names))
        or len(task_components) != len(set(task_components))
    ):
        raise CodexSessionError(
            "formal direct Codex campaign task paths are not injective"
        )
    task_index = attempt.get("task_index")
    total_tasks = attempt.get("total_tasks")
    attempt_index = attempt.get("index")
    attempt_count = attempt.get("count")
    if (
        type(task_index) is not int
        or not 1 <= task_index <= len(tasks)
        or total_tasks != len(tasks)
        or type(attempt_index) is not int
        or type(attempt_count) is not int
        or attempt_count != policy.get("attempts")
        or not 1 <= attempt_index <= attempt_count
    ):
        raise CodexSessionError("formal direct Codex campaign indices are invalid")
    task = tasks[task_index - 1]
    mapping = mappings[task_index - 1]
    if not isinstance(task, dict) or not isinstance(mapping, dict):
        raise CodexSessionError("formal direct Codex campaign task is malformed")
    task_name = attempt.get("task_name")
    assigned_gpu = attempt.get("assigned_host_gpu_id")
    if (
        task.get("task_index") != task_index
        or task.get("task_name") != task_name
        or mapping.get("task_index") != task_index
        or mapping.get("task_name") != task_name
        or mapping.get("assigned_host_gpu_id") != assigned_gpu
        or eval_config.get("assigned_host_gpu_id") != assigned_gpu
        or os.environ.get("AGENT_KERNEL_ARENA_HOST_GPU_ID") != assigned_gpu
    ):
        raise CodexSessionError(
            "formal direct Codex task or GPU mapping differs from campaign"
        )
    raw_receipt_path = attempt.get("receipt_path")
    if not isinstance(raw_receipt_path, str) or not Path(
        raw_receipt_path
    ).is_absolute():
        raise CodexSessionError("formal direct Codex receipt path is missing")
    receipt_path = Path(raw_receipt_path).resolve(strict=False)
    expected_receipt_path = (
        manifest_path.parent
        / ".campaign_attempts"
        / campaign_task_path_component(str(task_name))
        / f"attempt_{attempt_index:02d}"
        / "session_receipt.json"
    )
    if receipt_path != expected_receipt_path:
        raise CodexSessionError(
            "formal direct Codex receipt is outside its bound campaign attempt"
        )
    try:
        canonical_config = task_config_path.resolve(strict=True)
        config_metadata = canonical_config.lstat()
    except OSError as error:
        raise CodexSessionError(
            f"cannot inspect formal direct Codex task config: {error}"
        ) from error
    if (
        str(canonical_config) != task.get("config_path")
        or not canonical_config.is_file()
        or canonical_config.is_symlink()
        or config_metadata.st_nlink != 1
    ):
        raise CodexSessionError(
            "formal direct Codex task config path is not campaign-bound"
        )
    config_digest = _sha256_file(canonical_config)
    package_files, package_digest = _task_package_manifest(canonical_config.parent)
    if (
        config_digest != task.get("config_sha256")
        or package_files != task.get("package_files_sha256")
        or package_digest != task.get("package_manifest_sha256")
    ):
        raise CodexSessionError(
            "formal direct Codex task package differs from campaign"
        )
    binding = {
        "schema": _CAMPAIGN_BINDING_SCHEMA,
        "formal_execution_sha256": attempt.get("formal_execution_sha256"),
        "campaign_manifest_path": str(manifest_path),
        "campaign_manifest_sha256": attempt.get("campaign_manifest_sha256"),
        "comparison_contract_sha256": attempt.get(
            "comparison_contract_sha256"
        ),
        "backend_runtime_closure_sha256": attempt.get(
            "backend_runtime_closure_sha256"
        ),
        "task_package_manifest_sha256": attempt.get(
            "task_package_manifest_sha256"
        ),
        "task_config_sha256": attempt.get("task_config_sha256"),
        "task_name": task_name,
        "task_index": task_index,
        "total_tasks": total_tasks,
        "attempt_index": attempt_index,
        "attempt_count": attempt_count,
        "assigned_host_gpu_id": assigned_gpu,
    }
    expected = {
        **binding,
        "formal_execution_sha256": manifest.get("formal_execution_sha256"),
        "campaign_manifest_sha256": manifest_digest,
        "comparison_contract_sha256": comparison_digest,
        "backend_runtime_closure_sha256": agent.get(
            "backend_runtime_closure_sha256"
        ),
        "task_package_manifest_sha256": package_digest,
        "task_config_sha256": config_digest,
    }
    if set(binding) != _CAMPAIGN_BINDING_KEYS or binding != expected or any(
        not _valid_sha256(binding[key])
        for key in (
            "formal_execution_sha256",
            "campaign_manifest_sha256",
            "comparison_contract_sha256",
            "backend_runtime_closure_sha256",
            "task_package_manifest_sha256",
            "task_config_sha256",
        )
    ):
        raise CodexSessionError(
            "formal direct Codex attempt binding differs from immutable campaign"
        )
    return binding


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _editable_files(task_config_path: Path, workspace: Path) -> tuple[str, ...]:
    try:
        config = yaml.safe_load(task_config_path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise CodexSessionError(f"cannot read task config for editable scope: {error}") from error
    raw = config.get("source_file_path") if isinstance(config, dict) else None
    values = [raw] if isinstance(raw, str) else raw
    if not isinstance(values, list) or not values:
        raise CodexSessionError("formal direct Codex task requires source_file_path")
    editable: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
            raise CodexSessionError("source_file_path contains an unsafe path")
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or path.as_posix() != value
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise CodexSessionError("source_file_path must be normalized and workspace-relative")
        candidate = workspace.joinpath(*path.parts)
        try:
            metadata = candidate.lstat()
        except OSError as error:
            raise CodexSessionError(f"declared source is missing: {value}") from error
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            raise CodexSessionError(f"declared source is unsafe: {value}")
        if value in editable:
            raise CodexSessionError(f"duplicate declared source: {value}")
        editable.append(value)
    return tuple(editable)


def _workspace_manifest(workspace: Path) -> dict[str, dict[str, Any]]:
    manifest: dict[str, dict[str, Any]] = {}
    total_size = 0
    for path in sorted(workspace.rglob("*")):
        relative = path.relative_to(workspace).as_posix()
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise CodexSessionError(f"workspace contains a symlink: {relative}")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise CodexSessionError(f"workspace contains an unsafe file: {relative}")
        total_size += metadata.st_size
        if len(manifest) >= _MAX_WORKSPACE_FILES or total_size > _MAX_WORKSPACE_BYTES:
            raise CodexSessionError("workspace manifest exceeds formal campaign limits")
        manifest[relative] = {
            "sha256": _sha256_file(path),
            "size_bytes": metadata.st_size,
            "mode": format(stat.S_IMODE(metadata.st_mode), "04o"),
        }
    if not manifest:
        raise CodexSessionError("formal direct Codex workspace is empty")
    return manifest


def _workspace_integrity(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
    editable_files: tuple[str, ...],
) -> dict[str, Any]:
    """Describe a workspace diff without deciding whether it was temporary."""
    before_paths = set(before)
    after_paths = set(after)
    created = sorted(after_paths - before_paths)
    deleted = sorted(before_paths - after_paths)
    changed = sorted(
        path for path in before_paths & after_paths if before[path] != after[path]
    )
    editable = set(editable_files)
    unauthorized = sorted(path for path in changed if path not in editable)
    mode_changes = sorted(
        path
        for path in changed
        if path in editable and before[path]["mode"] != after[path]["mode"]
    )
    return {
        "before_manifest_sha256": _sha256_bytes(_canonical_json_bytes(before)),
        "after_manifest_sha256": _sha256_bytes(_canonical_json_bytes(after)),
        "created_files": created,
        "deleted_files": deleted,
        "changed_files": changed,
        "unauthorized_changed_files": unauthorized,
        "editable_mode_changes": mode_changes,
    }


def _copy_workspace_snapshot(workspace: Path, destination: Path) -> None:
    """Copy a previously validated workspace outside the agent's mount view."""
    shutil.copytree(workspace, destination, copy_function=shutil.copy2, symlinks=False)


def _remove_tree_entry(path: Path) -> None:
    metadata = path.lstat()
    if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
        for child in os.scandir(path):
            _remove_tree_entry(Path(child.path))
        path.rmdir()
    else:
        path.unlink()


def _clear_workspace(workspace: Path) -> None:
    """Remove workspace entries without following an agent-created link."""
    for entry in os.scandir(workspace):
        _remove_tree_entry(Path(entry.path))


def _capture_editable_candidates(
    workspace: Path,
    editable_files: tuple[str, ...],
    destination: Path,
) -> list[str]:
    """Capture only safe declared source bytes before evaluator-owned cleanup."""
    errors: list[str] = []
    total_size = 0
    destination.mkdir(mode=0o700)
    for relative in editable_files:
        relative_path = PurePosixPath(relative)
        source = workspace
        try:
            for part in relative_path.parts[:-1]:
                source /= part
                parent_metadata = source.lstat()
                if not stat.S_ISDIR(parent_metadata.st_mode) or stat.S_ISLNK(
                    parent_metadata.st_mode
                ):
                    raise OSError("editable parent is not a real directory")
            source /= relative_path.parts[-1]
            metadata = source.lstat()
        except OSError:
            errors.append(f"editable_candidate_missing:{relative}")
            continue
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            errors.append(f"editable_candidate_unsafe:{relative}")
            continue
        total_size += metadata.st_size
        if total_size > _MAX_WORKSPACE_BYTES:
            errors.append("editable_candidates_exceed_formal_campaign_limit")
            continue
        target = destination.joinpath(*relative_path.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target, follow_symlinks=False)
    return errors


def _restore_workspace(
    *,
    workspace: Path,
    baseline_snapshot: Path,
    candidate_snapshot: Path,
    editable_files: tuple[str, ...],
    retain_candidates: bool,
) -> list[str]:
    """Restore baseline exactly, then optionally reapply declared source bytes."""
    errors: list[str] = []
    try:
        _clear_workspace(workspace)
        directory_modes: list[tuple[Path, int]] = []
        for source in sorted(baseline_snapshot.rglob("*")):
            relative = source.relative_to(baseline_snapshot)
            target = workspace / relative
            metadata = source.lstat()
            if stat.S_ISDIR(metadata.st_mode):
                target.mkdir(parents=True, exist_ok=True, mode=0o700)
                directory_modes.append((target, stat.S_IMODE(metadata.st_mode)))
            elif stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target, follow_symlinks=False)
                target.chmod(stat.S_IMODE(metadata.st_mode))
            else:
                raise CodexSessionError(f"unsafe baseline snapshot entry: {relative}")
        if retain_candidates:
            for relative in editable_files:
                candidate = candidate_snapshot.joinpath(*PurePosixPath(relative).parts)
                if not candidate.is_file() or candidate.is_symlink():
                    continue
                target = workspace.joinpath(*PurePosixPath(relative).parts)
                shutil.copyfile(candidate, target, follow_symlinks=False)
                baseline_mode = stat.S_IMODE((baseline_snapshot / relative).stat().st_mode)
                target.chmod(baseline_mode)
        for directory, mode in reversed(directory_modes):
            directory.chmod(mode)
        workspace.chmod(stat.S_IMODE(baseline_snapshot.stat().st_mode))
    except (OSError, CodexSessionError) as error:
        errors.append(f"workspace_sanitization_failed:{type(error).__name__}:{error}")
    return errors


def _sanitize_formal_workspace(
    *,
    workspace: Path,
    baseline_snapshot: Path,
    before: dict[str, dict[str, Any]],
    editable_files: tuple[str, ...],
    retain_candidates: bool,
    artifact_dir: Path,
    captured_candidate_snapshot: Path | None = None,
    captured_raw_after: dict[str, dict[str, Any]] | None = None,
    captured_errors: tuple[str, ...] = (),
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Record raw changes and reduce the scored workspace to source-only output."""
    errors: list[str] = list(captured_errors)
    raw_after: dict[str, dict[str, Any]] = captured_raw_after or {}
    raw_scan_error: str | None = None
    if captured_raw_after is None:
        try:
            raw_after = _workspace_manifest(workspace)
        except CodexSessionError as error:
            raw_scan_error = str(error)
            errors.append("raw_workspace_manifest_unavailable")
    raw_diff = _workspace_integrity(before, raw_after, editable_files)

    candidate_snapshot = (
        captured_candidate_snapshot
        if captured_candidate_snapshot is not None
        else artifact_dir / ".editable-candidates"
    )
    candidate_errors: list[str] = []
    if captured_candidate_snapshot is None:
        candidate_errors = _capture_editable_candidates(
            workspace, editable_files, candidate_snapshot
        )
        errors.extend(candidate_errors)
    restore_errors = _restore_workspace(
        workspace=workspace,
        baseline_snapshot=baseline_snapshot,
        candidate_snapshot=candidate_snapshot,
        editable_files=editable_files,
        retain_candidates=retain_candidates and not candidate_errors,
    )
    errors.extend(restore_errors)
    try:
        after = _workspace_manifest(workspace)
    except CodexSessionError as error:
        after = {}
        errors.append(f"sanitized_workspace_manifest_unavailable:{error}")
    final_diff = _workspace_integrity(before, after, editable_files)
    final_errors: list[str] = []
    if final_diff["created_files"]:
        final_errors.append("workspace_files_created_after_sanitization")
    if final_diff["deleted_files"]:
        final_errors.append("workspace_files_deleted_after_sanitization")
    if final_diff["unauthorized_changed_files"]:
        final_errors.append("noneditable_files_changed_after_sanitization")
    if final_diff["editable_mode_changes"]:
        final_errors.append("editable_file_modes_changed_after_sanitization")
    errors.extend(final_errors)
    integrity = {
        "policy": "declared_source_only_sanitized_v1",
        "editable_files": list(editable_files),
        "raw_after_manifest_sha256": (
            _sha256_bytes(_canonical_json_bytes(raw_after)) if raw_after else None
        ),
        "raw_manifest_error": raw_scan_error,
        "raw_changes": raw_diff,
        "sanitization": {
            "performed": True,
            "candidate_retained": retain_candidates and not candidate_errors,
            "baseline_restored": not restore_errors,
        },
        "final_changes": final_diff,
        "errors": errors,
        "passed": not errors,
    }
    shutil.rmtree(candidate_snapshot, ignore_errors=True)
    return after, integrity


def _capture_workspace_checkpoint(
    *,
    workspace: Path,
    editable_files: tuple[str, ...],
    destination: Path,
    capture_mode: str = "verified_process_tree_suspension",
) -> tuple[dict[str, dict[str, Any]], tuple[str, ...], dict[str, Any]]:
    """Capture source after verified tree suspension or complete natural exit."""

    errors: list[str] = []
    try:
        manifest = _workspace_manifest(workspace)
    except CodexSessionError as error:
        manifest = {}
        errors.append(f"boundary_workspace_manifest_unavailable:{error}")
    errors.extend(_capture_editable_candidates(workspace, editable_files, destination))

    files: list[dict[str, Any]] = []
    for relative in editable_files:
        candidate = destination.joinpath(*PurePosixPath(relative).parts)
        expected = manifest.get(relative)
        if not candidate.is_file() or candidate.is_symlink() or not isinstance(
            expected, dict
        ):
            errors.append(f"boundary_candidate_not_bound_to_manifest:{relative}")
            continue
        observed_hash = _sha256_file(candidate)
        observed_size = candidate.stat().st_size
        if (
            expected.get("sha256") != observed_hash
            or expected.get("size_bytes") != observed_size
        ):
            errors.append(f"boundary_candidate_manifest_mismatch:{relative}")
            continue
        files.append(
            {
                "path": relative,
                "sha256": observed_hash,
                "size_bytes": observed_size,
            }
        )

    receipt = {
        "policy_id": (
            ATTEMPT_CONTAINMENT_POLICY
            if capture_mode == "post_namespace_teardown_checkpoint"
            else BOUNDARY_QUIESCENCE_POLICY
        ),
        "capture_mode": capture_mode,
        "manifest_sha256": (
            _sha256_bytes(_canonical_json_bytes(manifest)) if manifest else None
        ),
        "files": files,
        "errors": errors,
        "complete": bool(manifest) and not errors,
    }
    return manifest, tuple(errors), receipt


def _candidate_persistence_receipt(
    *,
    turn_budget: AgentTurnBudget,
    workspace_integrity: dict[str, Any] | None,
    evidence_complete: bool,
    attempt_contained: bool,
    boundary_snapshot: dict[str, Any] | None,
    output_tail: dict[str, Any] | None,
    boundary_resolution: str | None,
    attempt_cleanup: dict[str, Any],
) -> dict[str, Any]:
    """Bind retained source bytes to the shared matched-campaign policy."""
    exact_boundary = turn_budget.exact_boundary_reached
    integrity = workspace_integrity if isinstance(workspace_integrity, dict) else {}
    final_changes = integrity.get("final_changes")
    checkpoint = None
    if (
        exact_boundary
        and evidence_complete
        and attempt_contained
        and integrity.get("passed") is True
    ):
        checkpoint = {
            "before_manifest_sha256": (final_changes or {}).get(
                "before_manifest_sha256"
            ),
            "after_manifest_sha256": (final_changes or {}).get(
                "after_manifest_sha256"
            ),
            "changed_files": (final_changes or {}).get("changed_files"),
            "editable_files": integrity.get("editable_files"),
            "boundary_snapshot_sha256": _sha256_bytes(
                _canonical_json_bytes(boundary_snapshot)
            ),
            "output_tail_sha256": _sha256_bytes(
                _canonical_json_bytes(output_tail)
            ),
            "attempt_cleanup_sha256": _sha256_bytes(
                _canonical_json_bytes(attempt_cleanup)
            ),
        }
    return {
        "schema": "aka.candidate-persistence-receipt/v4",
        "policy_id": CANDIDATE_PERSISTENCE_POLICY,
        "attempt_contained": attempt_contained,
        "agent_process_containment_policy_id": AGENT_PROCESS_CONTAINMENT_POLICY,
        "attempt_containment_policy_id": ATTEMPT_CONTAINMENT_POLICY,
        "termination": (
            "exact_turn_boundary"
            if exact_boundary and attempt_contained
            else "rejected"
            if exact_boundary
            else "completed"
            if evidence_complete
            else "rejected"
        ),
        "checkpoint": checkpoint,
        "boundary_snapshot": (
            boundary_snapshot if exact_boundary and attempt_contained else None
        ),
        "output_tail": output_tail if exact_boundary and attempt_contained else None,
        "boundary_resolution": (
            boundary_resolution if exact_boundary and attempt_contained else None
        ),
        "attempt_cleanup": (
            attempt_cleanup if exact_boundary and attempt_contained else None
        ),
    }


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _attempt_receipt_path(
    eval_config: dict[str, Any], workspace: Path
) -> tuple[Path, Path]:
    """Reserve an evidence directory outside the scored workspace.

    A campaign supplies an absolute receipt path in its attempt directory. For
    ordinary direct-agent runs, use a hidden sibling of the scored workspace.
    In both cases artifacts live in a fresh directory and existing evidence is
    never overwritten.
    """
    attempt = eval_config.get("campaign_attempt") or {}
    if not isinstance(attempt, dict):
        raise CodexSessionError("campaign_attempt must be a mapping")

    workspace = workspace.resolve(strict=True)
    raw_receipt = attempt.get("receipt_path")
    if raw_receipt is not None:
        if not isinstance(raw_receipt, (str, os.PathLike)):
            raise CodexSessionError("campaign_attempt.receipt_path must be a path")
        receipt_path = Path(raw_receipt)
        if not receipt_path.is_absolute():
            raise CodexSessionError("campaign_attempt.receipt_path must be absolute")
        resolved_parent = receipt_path.parent.resolve(strict=False)
        resolved_receipt = resolved_parent / receipt_path.name
        artifact_dir = resolved_parent / f".{receipt_path.stem}.artifacts"
    else:
        resolved_parent = workspace.parent.resolve(strict=True)
        artifact_dir = resolved_parent / f".{workspace.name}.codex-attempt"
        resolved_receipt = artifact_dir / "attempt_receipt.json"

    if _is_within(resolved_receipt, workspace) or _is_within(artifact_dir, workspace):
        raise CodexSessionError("Codex evidence must be outside the scored workspace")
    resolved_parent.mkdir(parents=True, exist_ok=True)
    if resolved_receipt.exists():
        raise CodexSessionError(f"Codex attempt receipt already exists: {resolved_receipt}")
    try:
        artifact_dir.mkdir(mode=0o700)
    except FileExistsError as error:
        raise CodexSessionError(
            f"Codex attempt artifact directory already exists: {artifact_dir}"
        ) from error
    return resolved_receipt, artifact_dir


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_read_only_atomic(path: Path, payload: bytes) -> dict[str, Any]:
    """Publish one immutable artifact atomically without overwriting a target."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        # link(2) provides no-replace publication, unlike os.replace(). The
        # temporary file is in the same directory, so the operation is atomic.
        os.link(temporary, path, follow_symlinks=False)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    metadata = path.stat()
    return {
        "path": str(path),
        "sha256": _sha256_bytes(payload),
        "size_bytes": len(payload),
        "mode": format(metadata.st_mode & 0o777, "04o"),
    }


def _stream_metadata(raw_stdout: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract session identifiers and additive usage from Codex JSONL."""
    thread_ids: list[str] = []
    session_ids: list[str] = []
    usage: dict[str, int] = {}
    usage_events = 0
    json_events = 0
    malformed_lines = 0

    def _append_identifier(collection: list[str], value: Any) -> None:
        if isinstance(value, str) and value.strip() and value not in collection:
            collection.append(value)

    for line in raw_stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            malformed_lines += 1
            continue
        if not isinstance(event, dict):
            continue
        json_events += 1
        event_type = event.get("type")
        if event_type == "thread.started":
            _append_identifier(thread_ids, event.get("thread_id"))
            thread = event.get("thread")
            if isinstance(thread, dict):
                _append_identifier(thread_ids, thread.get("id"))
        if event_type in {"session.started", "session_created"}:
            _append_identifier(session_ids, event.get("session_id"))
            session = event.get("session")
            if isinstance(session, dict):
                _append_identifier(session_ids, session.get("id"))

        event_usage = event.get("usage")
        if not isinstance(event_usage, dict):
            continue
        usage_events += 1
        for key, value in event_usage.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                continue
            usage[key] = usage.get(key, 0) + value

    for field in ("input_tokens", "cached_input_tokens", "output_tokens"):
        usage.setdefault(field, 0)
    session = {
        "thread_id": thread_ids[0] if thread_ids else None,
        "session_id": session_ids[0] if session_ids else None,
        "thread_ids": thread_ids,
        "session_ids": session_ids,
        "json_events": json_events,
        "malformed_lines": malformed_lines,
    }
    aggregated_usage = {"events": usage_events, **usage}
    return session, aggregated_usage


def _effective_timeout_seconds(
    agent_config: dict[str, Any], eval_config: dict[str, Any]
) -> float:
    attempt = eval_config.get("campaign_attempt") or {}
    if not isinstance(attempt, dict):
        raise CodexSessionError("campaign_attempt must be a mapping")
    try:
        configured = float(agent_config.get("timeout_seconds", 600))
        raw_deadline = attempt.get("task_deadline_monotonic")
        deadline = float(raw_deadline) if raw_deadline is not None else None
    except (TypeError, ValueError) as error:
        raise CodexSessionError("Codex timeout and campaign deadline must be numeric") from error
    if not math.isfinite(configured) or (
        deadline is not None and not math.isfinite(deadline)
    ):
        raise CodexSessionError("Codex timeout and campaign deadline must be finite")
    effective = configured
    if deadline is not None:
        effective = min(configured, deadline - time.monotonic())
    if effective <= 0:
        raise CodexSessionError("no positive Codex session budget remains")
    return effective


def _process_group_exists(pgid: int) -> bool:
    states = _linux_process_group_states(pgid)
    if states is not None:
        return any(state != "Z" for state in states.values())
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _linux_process_group_states(pgid: int) -> dict[int, str] | None:
    """Return Linux process states for a process group, or ``None`` off Linux.

    A killed orphan can briefly remain as a zombie when PID 1 is slow to reap
    it. Such a process cannot execute or retain file descriptors and therefore
    does not make cleanup unsafe. ``killpg(pgid, 0)`` alone cannot distinguish
    that state.
    """
    proc = Path("/proc")
    if not proc.is_dir():
        return None
    try:
        entries = list(proc.iterdir())
    except OSError:
        return None
    states: dict[int, str] = {}
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            stat_line = (entry / "stat").read_text(encoding="utf-8")
            fields = stat_line[stat_line.rfind(")") + 2 :].split()
            state = fields[0]
            process_group = int(fields[2])
        except (OSError, ValueError, IndexError):
            continue
        if process_group == pgid:
            states[int(entry.name)] = state
    return states


def _linux_process_table() -> dict[int, dict[str, int | str]] | None:
    """Return the process identities needed for descendant-lineage tracking."""

    proc = Path("/proc")
    if not proc.is_dir():
        return None
    try:
        entries = list(proc.iterdir())
    except OSError:
        return None
    table: dict[int, dict[str, int | str]] = {}
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            stat_line = (entry / "stat").read_text(encoding="utf-8")
            fields = stat_line[stat_line.rfind(")") + 2 :].split()
            table[int(entry.name)] = {
                "state": fields[0],
                "ppid": int(fields[1]),
                "pgrp": int(fields[2]),
                "session": int(fields[3]),
                "starttime": int(fields[19]),
            }
        except (OSError, ValueError, IndexError):
            continue
    return table


class _AttemptProcessTracker:
    """Track an attempt's descendants even when they leave the root PGID."""

    def __init__(self, root_pid: int, token: str) -> None:
        self.root_pid = root_pid
        self._token = token.encode("utf-8")
        self._known: dict[int, int] = {}
        self._errors: list[str] = []
        self._lock = threading.Lock()
        self.scan()

    def _has_token(self, pid: int) -> bool:
        try:
            environ = Path(f"/proc/{pid}/environ").read_bytes()
        except OSError:
            return False
        return any(value == self._token for value in environ.split(b"\0"))

    def scan(self) -> list[dict[str, int | str]]:
        table = _linux_process_table()
        if table is None:
            with self._lock:
                if "linux_proc_unavailable" not in self._errors:
                    self._errors.append("linux_proc_unavailable")
            return []
        with self._lock:
            identities = {
                pid
                for pid, record in table.items()
                if self._known.get(pid) == record["starttime"]
            }
            root = table.get(self.root_pid)
            if root is not None:
                identities.add(self.root_pid)
            identities.update(
                pid for pid in table if self._has_token(pid)
            )
            changed = True
            while changed:
                before = len(identities)
                identities.update(
                    pid
                    for pid, record in table.items()
                    if record["ppid"] in identities
                )
                changed = len(identities) != before
            for pid in identities:
                self._known[pid] = int(table[pid]["starttime"])
            return [
                {"pid": pid, **table[pid]}
                for pid in sorted(identities)
                if table[pid]["state"] != "Z"
            ]

    def errors(self) -> list[str]:
        with self._lock:
            return list(self._errors)


def _track_attempt_processes(
    tracker: _AttemptProcessTracker, stop: threading.Event
) -> None:
    while not stop.wait(0.01):
        tracker.scan()


def _suspend_process_group(
    process: subprocess.Popen[str],
    logger: logging.Logger,
    tracker: _AttemptProcessTracker,
) -> dict[str, Any]:
    """Stop and synchronously verify every live member before source capture.

    Sending ``SIGSTOP`` is not itself a barrier: the signal can be pending while
    a process is still able to mutate the workspace.  The verifier therefore
    waits until two consecutive ``/proc`` scans see the same live membership and
    every member is in a stopped state.  Formal checkpoint persistence fails
    closed on platforms where that state cannot be proven.
    """

    pgid = process.pid
    evidence: dict[str, Any] = {
        "policy_id": BOUNDARY_QUIESCENCE_POLICY,
        "method": "sigstop_process_tree",
        "scope": "proc_descendant_lineage_and_inherited_attempt_token_v1",
        "pgid": pgid,
        "sent": False,
        "root_group_signal_sent": False,
        "individually_signaled_pids": [],
        "verification_performed": False,
        "verified": False,
        "verification_polls": 0,
        "stable_polls": 0,
        "members": [],
        "members_sha256": None,
        "error": None,
    }
    try:
        os.killpg(pgid, signal.SIGSTOP)
        evidence["sent"] = True
        evidence["root_group_signal_sent"] = True
    except ProcessLookupError:
        pass
    except OSError as error:
        evidence["error"] = f"{type(error).__name__}: {error}"
        return evidence

    deadline = time.monotonic() + _SUSPEND_GRACE_SECONDS
    previous_live_identities: tuple[tuple[int, int], ...] | None = None
    stable_polls = 0
    while time.monotonic() < deadline:
        members = tracker.scan()
        evidence["verification_performed"] = True
        evidence["verification_polls"] += 1
        if tracker.errors():
            evidence["error"] = "process_tree_state_verification_unavailable"
            break
        signaled = set(evidence["individually_signaled_pids"])
        for member in members:
            pid = int(member["pid"])
            if member["state"] not in {"T", "t"}:
                try:
                    os.kill(pid, signal.SIGSTOP)
                    evidence["sent"] = True
                    signaled.add(pid)
                except ProcessLookupError:
                    pass
        evidence["individually_signaled_pids"] = sorted(signaled)
        members = tracker.scan()
        live_identities = tuple(
            sorted((int(item["pid"]), int(item["starttime"])) for item in members)
        )
        all_stopped = bool(members) and all(
            item["state"] in {"T", "t"} for item in members
        )
        if all_stopped and live_identities == previous_live_identities:
            stable_polls += 1
        elif all_stopped:
            stable_polls = 1
        else:
            stable_polls = 0
        previous_live_identities = live_identities
        if stable_polls >= _SUSPEND_STABLE_POLLS:
            evidence["stable_polls"] = stable_polls
            evidence["members"] = members
            evidence["members_sha256"] = _sha256_bytes(
                _canonical_json_bytes(members)
            )
            evidence["verified"] = True
            return evidence
        time.sleep(0.01)

    evidence["stable_polls"] = stable_polls
    if evidence["error"] is None:
        evidence["error"] = (
            "process_tree_empty_before_suspension"
            if not tracker.scan()
            else "process_tree_suspension_not_verified"
        )
    logger.error("Codex process-tree suspension failed: %s", evidence["error"])
    return evidence


def _wait_for_process_group_exit(
    process: subprocess.Popen[str],
    pgid: int,
    timeout: float,
    tracker: _AttemptProcessTracker | None = None,
) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        process.poll()
        tree_members = tracker.scan() if tracker is not None else []
        if (
            (tracker is None or not tracker.errors())
            and not _process_group_exists(pgid)
            and not tree_members
        ):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def _terminate_process_group(
    process: subprocess.Popen[str],
    logger: logging.Logger,
    *,
    reason: str,
    resume_stopped_group: bool = False,
    tracker: _AttemptProcessTracker | None = None,
) -> dict[str, Any]:
    """TERM, then KILL, the root group and tracked descendants; prove absence."""
    pgid = process.pid
    cleanup = {
        "required": True,
        "reason": reason,
        "pgid": pgid,
        "sigterm_sent": False,
        "sigcont_sent": False,
        "sigkill_sent": False,
        "verified_absent": False,
        "verification_performed": True,
        "scope": "process_tree" if tracker is not None else "process_group",
        "tracked_members_before_cleanup": (
            tracker.scan() if tracker is not None else []
        ),
    }
    try:
        os.killpg(pgid, signal.SIGTERM)
        cleanup["sigterm_sent"] = True
    except ProcessLookupError:
        pass

    tracked = tracker.scan() if tracker is not None else []
    for member in tracked:
        try:
            os.kill(int(member["pid"]), signal.SIGTERM)
            cleanup["sigterm_sent"] = True
        except ProcessLookupError:
            pass

    if resume_stopped_group:
        try:
            os.killpg(pgid, signal.SIGCONT)
            cleanup["sigcont_sent"] = True
        except ProcessLookupError:
            pass
        for member in tracked:
            try:
                os.kill(int(member["pid"]), signal.SIGCONT)
                cleanup["sigcont_sent"] = True
            except ProcessLookupError:
                pass

    if _wait_for_process_group_exit(
        process, pgid, _TERM_GRACE_SECONDS, tracker
    ):
        cleanup["verified_absent"] = True
        cleanup["tracked_members_after_cleanup"] = []
        return cleanup

    logger.warning("Codex attempt process tree survived SIGTERM; sending SIGKILL")
    try:
        os.killpg(pgid, signal.SIGKILL)
        cleanup["sigkill_sent"] = True
    except ProcessLookupError:
        pass
    for member in tracker.scan() if tracker is not None else []:
        try:
            os.kill(int(member["pid"]), signal.SIGKILL)
            cleanup["sigkill_sent"] = True
        except ProcessLookupError:
            pass
    cleanup["verified_absent"] = _wait_for_process_group_exit(
        process, pgid, _KILL_GRACE_SECONDS, tracker
    )
    cleanup["tracked_members_after_cleanup"] = (
        tracker.scan() if tracker is not None else []
    )
    return cleanup


def _formatted_transcript(stdout_lines: list[str], stderr_lines: list[str]) -> str:
    output = "\n".join(stdout_lines)
    if stderr_lines:
        separator = "\n" if output else ""
        output += separator + "=== STDERR ===\n" + "\n".join(stderr_lines)
    return output


def _write_attempt_receipt(
    *,
    receipt_path: Path,
    artifact_dir: Path,
    raw_stdout: str,
    raw_stderr: str,
    transcript: str,
    rendered_prompt: bytes,
    receipt: dict[str, Any],
    workspace_before: dict[str, dict[str, Any]] | None = None,
    workspace_after: dict[str, dict[str, Any]] | None = None,
) -> None:
    artifacts = {
        "rendered_prompt": _write_read_only_atomic(
            artifact_dir / "rendered_prompt.txt", rendered_prompt
        ),
        "raw_stdout": _write_read_only_atomic(
            artifact_dir / "raw_stdout.jsonl", raw_stdout.encode("utf-8")
        ),
        "raw_stderr": _write_read_only_atomic(
            artifact_dir / "raw_stderr.txt", raw_stderr.encode("utf-8")
        ),
        "formatted_transcript": _write_read_only_atomic(
            artifact_dir / "formatted_transcript.txt", transcript.encode("utf-8")
        ),
    }
    if workspace_before is not None and workspace_after is not None:
        artifacts["workspace_before_manifest"] = _write_read_only_atomic(
            artifact_dir / "workspace_before_manifest.json",
            _canonical_json_bytes(workspace_before) + b"\n",
        )
        artifacts["workspace_after_manifest"] = _write_read_only_atomic(
            artifact_dir / "workspace_after_manifest.json",
            _canonical_json_bytes(workspace_after) + b"\n",
        )
    receipt["artifacts"] = artifacts
    payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _write_read_only_atomic(receipt_path, payload)
    artifact_dir.chmod(0o555)
    _fsync_directory(artifact_dir.parent)


def _format_codex_event(raw_line: str) -> str:
    """Convert Codex JSONL events into readable log lines.

    Modern `codex exec --json` (>=0.x item-based stream) emits a thread/turn/item
    envelope, e.g.::

        {"type":"item.completed","item":{"type":"agent_message","text":"..."}}
        {"type":"item.started","item":{"type":"command_execution","command":"...","status":"in_progress"}}
        {"type":"turn.completed","usage":{...}}

    The assistant's final answer lives at ``item.text`` of an ``item.completed``
    event whose ``item.type == "agent_message"``. Older binaries used flat
    ``assistant_message``/``assistant`` events or a nested ``msg`` envelope; those
    are kept as fallbacks so logs stay readable across Codex versions.
    """
    try:
        data = json.loads(raw_line)
    except json.JSONDecodeError:
        return raw_line

    if not isinstance(data, dict):
        return raw_line

    ev_type = data.get("type", "")

    # --- Current item-based envelope -------------------------------------
    if ev_type in {"item.started", "item.completed", "item.updated"}:
        item = data.get("item") or {}
        if isinstance(item, dict):
            item_type = item.get("type", "")
            if item_type == "agent_message":
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    return f"assistant: {text.strip()}"
            elif item_type == "reasoning":
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    return f"reasoning: {text.strip()}"
            elif item_type == "command_execution":
                command = item.get("command", "")
                status = item.get("status", "")
                exit_code = item.get("exit_code")
                tail = f" exit={exit_code}" if exit_code is not None else ""
                return f"command[{status}] {command}{tail}".strip()
            elif item_type == "mcp_tool_call":
                server = item.get("server", "")
                tool = item.get("tool", "")
                status = item.get("status", "")
                return f"mcp_tool[{status}] {server}.{tool}".strip()
            elif item_type == "file_change":
                return f"file_change[{item.get('status', '')}]".strip()
            elif item_type == "error":
                return f"error: {item.get('message', raw_line)}"
        return raw_line

    if ev_type == "turn.completed":
        usage = data.get("usage")
        if isinstance(usage, dict):
            return (
                "turn.completed usage "
                f"in={usage.get('input_tokens')} out={usage.get('output_tokens')}"
            )
        return raw_line

    if ev_type in {"turn.failed", "error"}:
        err = data.get("error") or data.get("message")
        if isinstance(err, dict):
            err = err.get("message", err)
        return f"{ev_type}: {err}" if err else raw_line

    if ev_type in {"thread.started", "turn.started"}:
        return raw_line

    # --- Legacy fallbacks (older Codex binaries) -------------------------
    # Nested `msg` envelope: {"msg":{"type":"agent_message","message":"..."}}
    msg = data.get("msg")
    if isinstance(msg, dict) and msg.get("type") in {"agent_message", "assistant_message"}:
        text = msg.get("message") or msg.get("text")
        if isinstance(text, str) and text.strip():
            return f"assistant: {text.strip()}"

    # Oldest flat events.
    if ev_type in {"assistant_message", "assistant"}:
        message = data.get("message", {})
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return f"assistant: {content.strip()}"
        text = data.get("text")
        if isinstance(text, str) and text.strip():
            return f"assistant: {text.strip()}"

    text = data.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    return raw_line


def _get_codex_version(agent_cmd: str, env: dict[str, str] | None = None) -> str:
    """Best-effort Codex CLI version lookup for logging."""
    try:
        result = subprocess.run(
            [agent_cmd, "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
            env=env,
        )
    except Exception:
        return "unknown"

    text = (result.stdout or result.stderr or "").strip()
    return text or "unknown"


@register_agent("codex")
def launch_agent(eval_config: dict[str, Any], task_config_dir: str, workspace: str) -> str:
    """
    Launch Codex CLI in non-interactive mode with streaming output capture.

    Args:
        eval_config: Evaluator settings passed from main
        task_config_dir: Path to the task configuration used to build the prompt
        workspace: Workspace directory where the agent runs

    Returns:
        Combined stdout/stderr output captured from Codex CLI.
    """
    AGENT = "codex"
    codex_bin = shutil.which(AGENT)
    if not codex_bin:
        raise RuntimeError(
            f"Command '{AGENT}' not found. Please ensure Codex CLI is installed and in your PATH."
        )
    resolved_codex_bin = Path(codex_bin).resolve(strict=True)
    if not resolved_codex_bin.is_file():
        raise CodexSessionError(f"Codex CLI is not a regular file: {resolved_codex_bin}")
    codex_binary_sha256 = _sha256_file(resolved_codex_bin)

    config_path = Path(__file__).with_name("agent_config.yaml")
    with config_path.open("r") as f:
        agent_config = yaml.safe_load(f) or {}

    logger = logging.getLogger(__name__)
    workspace_path = Path(workspace).resolve(strict=True)
    if not workspace_path.is_dir() or workspace_path.is_symlink():
        raise CodexSessionError(f"unsafe Codex workspace: {workspace_path}")
    formal_campaign = is_formal_campaign(eval_config)
    task_config_path = Path(task_config_dir).resolve(strict=formal_campaign)
    campaign_binding = _campaign_binding(
        eval_config,
        task_config_path,
        formal_campaign=formal_campaign,
    )
    process_env = build_subprocess_env(agent_config.get("python_path"))
    prompt_builder = load_prompt_builder(AgentType.CODEX, logger)
    prompt = prompt_builder(task_config_dir, workspace, eval_config, logger)
    prompt_agent_config = _prompt_agent_config(agent_config, eval_config)
    prompt = integrate_agent_config(
        prompt,
        prompt_agent_config,
        process_env.get("AGENT_KERNEL_ARENA_PYTHON"),
    )
    rendered_prompt = prompt.encode("utf-8")
    comparison_contract_sha256 = _comparison_contract_sha256(
        eval_config, formal_campaign=formal_campaign
    )
    runtime_closure_sha256 = _runtime_closure_sha256(
        eval_config, formal_campaign=formal_campaign
    )
    configured_model = agent_config.get("model")
    configured_effort = agent_config.get("effort")
    try:
        max_turns = int(agent_config.get("max_turns", 0))
        output_limit = int(
            agent_config.get(
                "structured_stream_output_limit_bytes",
                agent_config.get("max_process_output_bytes", _DEFAULT_OUTPUT_LIMIT),
            )
        )
    except (TypeError, ValueError) as error:
        raise CodexSessionError("Codex turn and output limits must be integers") from error
    if max_turns <= 0 or output_limit <= 0:
        raise CodexSessionError("Codex turn and output limits must be positive")
    if formal_campaign and max_turns != FORMAL_MATCHED_MAX_TURNS:
        raise CodexSessionError(
            f"formal direct Codex requires max_turns={FORMAL_MATCHED_MAX_TURNS}"
        )
    turn_budget = AgentTurnBudget(max_turns)

    editable_files: tuple[str, ...] = ()
    workspace_before: dict[str, dict[str, Any]] | None = None
    gpu_evidence: dict[str, Any] | None = None
    if formal_campaign:
        editable_files = _editable_files(task_config_path, workspace_path)
        workspace_before = _workspace_manifest(workspace_path)
        gpu_evidence = formal_gpu_evidence(eval_config)
    effective_timeout = _effective_timeout_seconds(agent_config, eval_config)
    receipt_path, artifact_dir = _attempt_receipt_path(
        eval_config, workspace_path
    )
    baseline_snapshot: Path | None = None
    if formal_campaign:
        baseline_snapshot = artifact_dir / ".workspace-baseline"
        _copy_workspace_snapshot(workspace_path, baseline_snapshot)
    attempt_home = prepare_attempt_home(eval_config, backend="codex")
    if formal_campaign and attempt_home is None:
        raise CodexSessionError("formal direct Codex requires attempt containment")
    cloud_config_bootstrap = (
        codex_cloud_config_bootstrap_receipt(attempt_home)
        if attempt_home is not None
        else None
    )
    process_env = isolated_environment(process_env, attempt_home)
    attempt_process_token = f"AKA_ATTEMPT_PROCESS_TOKEN={uuid.uuid4().hex}"
    process_env["AKA_ATTEMPT_PROCESS_TOKEN"] = attempt_process_token.split("=", 1)[1]
    executed_codex_bin = str(resolved_codex_bin)
    codex_version = _get_codex_version(executed_codex_bin, process_env)

    cmd = [
        executed_codex_bin,
        "exec",
        "--json",
        "--color",
        "never",
        "--sandbox",
        "workspace-write",
        "--config",
        'approval_policy="never"',
        "--strict-config",
        "--ignore-user-config",
        "--ignore-rules",
        "--skip-git-repo-check",
        "--ephemeral",
        # Explicitly disable cross-session "Memories" so headless runs never read
        # or write persistent learned memory (off by default, pinned for safety).
        "-c",
        "features.memories=false",
        "--cd",
        str(workspace_path),
    ]
    if configured_model:
        cmd.extend(["--model", str(configured_model)])
    if configured_effort:
        # Codex has no --effort flag; reasoning effort is a config key.
        cmd.extend(["-c", f'model_reasoning_effort="{configured_effort}"'])
    cmd.append(prompt)

    logger.info("Codex Preflight")
    logger.info(f"  codex_binary: {resolved_codex_bin}")
    logger.info(f"  codex_binary_sha256: {codex_binary_sha256}")
    logger.info(f"  codex_version: {codex_version}")
    logger.info(f"  workspace: {workspace_path}")
    logger.info(f"  attempt_receipt: {receipt_path}")
    logger.info(f"  effective_timeout_seconds: {effective_timeout:.3f}")
    logger.info(f"  python_path: {process_env.get('AGENT_KERNEL_ARENA_PYTHON', '<unset>')}")
    if configured_model:
        logger.info(f"  model: {configured_model} (explicit via agents/codex/agent_config.yaml)")
    else:
        logger.info("  model: <codex CLI default/config> (not explicitly set)")
    logger.info(f"  effort: {configured_effort if configured_effort else '<codex config default>'} (model_reasoning_effort)")
    logger.info(f"Running command: {' '.join(shlex.quote(p) for p in cmd[:12])} ...")
    logger.info("=" * 80)
    logger.info("Agent Output (streaming):")
    logger.info("=" * 80)

    isolated_cmd = wrap_attempt_command(
        cmd,
        eval_config=eval_config,
        writable_roots=(
            workspace_path,
            *((attempt_home,) if attempt_home is not None else ()),
        ),
    )
    pass_fds = attempt_command_pass_fds(isolated_cmd)
    try:
        process = subprocess.Popen(
            isolated_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=workspace_path,
            bufsize=1,
            env=process_env,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
            pass_fds=pass_fds,
        )
    finally:
        release_attempt_command_fds(isolated_cmd)
    attempt_boundary = establish_attempt_boundary(isolated_cmd, process)
    attempt_boundary_receipt = (
        attempt_boundary.receipt() if attempt_boundary is not None else None
    )
    if process.stdin:
        process.stdin.close()
    process_tracker = (
        None
        if attempt_boundary is not None
        else _AttemptProcessTracker(process.pid, attempt_process_token)
    )
    process_tracker_stop = threading.Event() if process_tracker is not None else None
    process_tracker_thread = None
    if process_tracker is not None and process_tracker_stop is not None:
        process_tracker_thread = threading.Thread(
            target=_track_attempt_processes,
            args=(process_tracker, process_tracker_stop),
            daemon=True,
        )
        process_tracker_thread.start()

    raw_stdout_chunks: list[str] = []
    raw_stderr_chunks: list[str] = []
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    reader_errors: list[str] = []
    turn_stop_event = threading.Event()
    suspension_complete_event = threading.Event()
    boundary_signal: dict[str, Any] = {
        "attempted": False,
        "stdout_character_offset": None,
    }
    process_group_suspension: dict[str, Any] | None = None
    capture_state: dict[str, dict[str, Any]] = {
        "stdout": {
            "limit_bytes": output_limit,
            "retained_bytes": 0,
            "discarded_bytes": 0,
            "truncated": False,
        },
        "stderr": {
            "limit_bytes": output_limit,
            "retained_bytes": 0,
            "discarded_bytes": 0,
            "truncated": False,
        },
    }

    def retain_bounded(raw_chunks, value: str, state: dict[str, Any]) -> str:
        encoded = value.encode("utf-8", errors="replace")
        remaining = max(0, state["limit_bytes"] - state["retained_bytes"])
        retained = encoded[:remaining]
        discarded = len(encoded) - len(retained)
        if retained:
            state["retained_bytes"] += len(retained)
            decoded = retained.decode("utf-8", errors="replace")
            raw_chunks.append(decoded)
        else:
            decoded = ""
        if discarded:
            state["discarded_bytes"] += discarded
            state["truncated"] = True
        return decoded

    def read_stream(
        stream,
        raw_chunks,
        output_list,
        prefix,
        log_func,
        stream_name: str,
        observe_turns: bool,
    ):
        discarding_oversized = False
        try:
            while True:
                line = stream.readline(_MAX_EVENT_LINE_CHARS + 1)
                if not line:
                    break
                complete_line = line.endswith("\n")
                oversized_chunk = not complete_line and len(line) > _MAX_EVENT_LINE_CHARS
                oversized_line = oversized_chunk or discarding_oversized
                if observe_turns and oversized_line:
                    turn_budget.stop_for_observer_error("oversized_structured_event")
                    turn_stop_event.set()
                    discarding_oversized = not complete_line
                captured = retain_bounded(raw_chunks, line, capture_state[stream_name])
                if (
                    not captured
                    or oversized_line
                    or capture_state[stream_name]["truncated"]
                ):
                    continue
                raw_line = captured.rstrip("\r\n")
                if not raw_line.strip():
                    continue
                if observe_turns and turn_budget.observe(raw_line):
                    if (
                        turn_budget.exact_boundary_reached
                        and not boundary_signal["attempted"]
                    ):
                        boundary_signal["attempted"] = True
                        boundary_signal["stdout_character_offset"] = sum(
                            len(chunk) for chunk in raw_stdout_chunks
                        )
                        turn_stop_event.set()
                        if attempt_boundary is not None:
                            try:
                                attempt_boundary.terminate_now()
                                boundary_signal["namespace_sigkill_sent"] = (
                                    attempt_boundary.sigkill_sent
                                )
                            except OSError as error:
                                turn_budget.stop_for_observer_error(
                                    "pid_namespace_termination_failed"
                                )
                                reader_errors.append(
                                    f"boundary: {type(error).__name__}: {error}"
                                )
                        elif process_tracker is not None:
                            suspension = _suspend_process_group(
                                process, logger, process_tracker
                            )
                            boundary_signal["suspension"] = suspension
                        suspension_complete_event.set()
                    else:
                        turn_stop_event.set()
                formatted = _format_codex_event(raw_line)
                output_list.append(formatted)
                log_func(f"{prefix} {formatted[:240]}")
        except Exception as error:  # capture failures invalidate the receipt
            reader_errors.append(f"{prefix}: {type(error).__name__}: {error}")
            if observe_turns:
                turn_budget.stop_for_observer_error("turn_observer_failed")
                turn_stop_event.set()
        finally:
            stream.close()

    stdout_thread = threading.Thread(
        target=read_stream,
        args=(
            process.stdout,
            raw_stdout_chunks,
            stdout_lines,
            "[AGENT]",
            logger.info,
            "stdout",
            True,
        ),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=read_stream,
        args=(
            process.stderr,
            raw_stderr_chunks,
            stderr_lines,
            "[AGENT STDERR]",
            logger.warning,
            "stderr",
            False,
        ),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    timed_out = False
    cleanup: dict[str, Any] = {
        "required": False,
        "reason": "normal_exit",
        "pgid": process.pid,
        "sigterm_sent": False,
        "sigcont_sent": False,
        "sigkill_sent": False,
        "verified_absent": False,
        "verification_performed": False,
    }
    boundary_end_reason = "normal_exit"
    boundary_raw_after: dict[str, dict[str, Any]] | None = None
    boundary_capture_errors: tuple[str, ...] = ()
    boundary_snapshot_receipt: dict[str, Any] | None = None
    boundary_candidate_snapshot: Path | None = None
    deadline = time.monotonic() + effective_timeout
    while process.poll() is None:
        if turn_stop_event.is_set():
            logger.warning("Codex structured turn budget requires process termination")
            if turn_budget.exact_boundary_reached:
                suspension_complete_event.wait(_SUSPEND_GRACE_SECONDS + 0.5)
                candidate = boundary_signal.get("suspension")
                process_group_suspension = (
                    candidate if isinstance(candidate, dict) else None
                )
            boundary_end_reason = "exact_turn_boundary"
            if attempt_boundary is None:
                cleanup = _terminate_process_group(
                    process,
                    logger,
                    reason=boundary_end_reason,
                    resume_stopped_group=(
                        isinstance(process_group_suspension, dict)
                        and process_group_suspension.get("sent") is True
                    ),
                    tracker=process_tracker,
                )
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            logger.warning(
                "Codex agent timed out after %.3fs; terminating attempt process tree",
                effective_timeout,
            )
            boundary_end_reason = "timeout"
            if attempt_boundary is None:
                cleanup = _terminate_process_group(
                    process, logger, reason="timeout", tracker=process_tracker
                )
            break
        try:
            process.wait(timeout=min(0.1, remaining))
        except subprocess.TimeoutExpired:
            continue
    if attempt_boundary is not None:
        if boundary_signal["attempted"] and turn_budget.exact_boundary_reached:
            boundary_end_reason = "exact_turn_boundary"
        cleanup = finalize_attempt_boundary(
            process,
            attempt_boundary,
            reason=boundary_end_reason,
            terminate=boundary_end_reason != "normal_exit",
        )
        if boundary_signal["attempted"]:
            cleanup["boundary_signal"] = {
                "attempted": True,
                "stdout_character_offset": boundary_signal[
                    "stdout_character_offset"
                ],
            }
    else:
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            cleanup["verified_absent"] = False

    if attempt_boundary is None and cleanup["reason"] == "normal_exit":
        cleanup["verification_performed"] = True
        cleanup["scope"] = "process_tree"
        assert process_tracker is not None
        cleanup["tracked_members_before_cleanup"] = process_tracker.scan()
        cleanup["verified_absent"] = (
            not process_tracker.errors()
            and not _process_group_exists(process.pid)
            and not cleanup["tracked_members_before_cleanup"]
        )
        if not cleanup["verified_absent"]:
            logger.warning(
                "Codex leader exited but its attempt process tree is still live; cleaning it up"
            )
            cleanup = _terminate_process_group(
                process,
                logger,
                reason="post_exit_lingering_process_tree",
                tracker=process_tracker,
            )
    if (
        boundary_signal["attempted"]
        and cleanup["verified_absent"]
        and isinstance(process_group_suspension, dict)
        and process_group_suspension.get("verified") is True
    ):
        cleanup["required"] = True
        cleanup["reason"] = "exact_turn_boundary"
        cleanup["boundary_signal"] = {
            "attempted": True,
            "stdout_character_offset": boundary_signal[
                "stdout_character_offset"
            ],
        }

    if process_tracker is not None and process_tracker_stop is not None:
        process_tracker_stop.set()
        assert process_tracker_thread is not None
        process_tracker_thread.join(timeout=1)
        cleanup["process_tracker_errors"] = process_tracker.errors()
        cleanup["tracked_members_after_cleanup"] = process_tracker.scan()
        if cleanup["process_tracker_errors"] or cleanup["tracked_members_after_cleanup"]:
            cleanup["verified_absent"] = False

    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)
    readers_completed = not stdout_thread.is_alive() and not stderr_thread.is_alive()
    if not readers_completed:
        reader_errors.append("stream readers did not reach EOF")
    if (
        formal_campaign
        and turn_budget.exact_boundary_reached
        and cleanup.get("scope") == "private_pid_namespace"
        and cleanup.get("verified_absent") is True
        and readers_completed
    ):
        boundary_candidate_snapshot = artifact_dir / ".boundary-candidates"
        (
            boundary_raw_after,
            boundary_capture_errors,
            boundary_snapshot_receipt,
        ) = _capture_workspace_checkpoint(
            workspace=workspace_path,
            editable_files=editable_files,
            destination=boundary_candidate_snapshot,
            capture_mode="post_namespace_teardown_checkpoint",
        )

    if stderr_lines:
        logger.warning("=" * 80)
        logger.warning(f"Agent STDERR captured {len(stderr_lines)} lines")
        logger.warning("=" * 80)

    logger.info("=" * 80)
    logger.info(f"Agent completed with exit code: {process.returncode}")
    logger.info("=" * 80)

    raw_stdout = "".join(raw_stdout_chunks)
    raw_stderr = "".join(raw_stderr_chunks)
    boundary_output_tail: dict[str, Any] | None = None
    if turn_budget.exact_boundary_reached:
        offset = boundary_signal.get("stdout_character_offset")
        tail_text = (
            raw_stdout[offset:]
            if isinstance(offset, int) and 0 <= offset <= len(raw_stdout)
            else raw_stdout
        )
        tail_bytes = tail_text.encode("utf-8")
        boundary_output_tail = {
            "policy": "retained_and_digested_v1",
            "stdout_character_offset": offset,
            "stdout_size_bytes": len(tail_bytes),
            "stdout_sha256": _sha256_bytes(tail_bytes),
            "post_boundary_turns": turn_budget.post_boundary_turns,
            "capture_truncated": any(
                state["truncated"] for state in capture_state.values()
            ),
            "readers_completed": readers_completed,
        }
    output = _formatted_transcript(stdout_lines, stderr_lines)
    session, aggregated_usage = _stream_metadata(raw_stdout)
    exit_code = process.returncode
    capture_truncated = any(
        state["truncated"] for state in capture_state.values()
    )
    turn_budget.finalize(
        process_succeeded=(
            not timed_out
            and exit_code == 0
            and cleanup["verified_absent"]
            and readers_completed
            and not reader_errors
            and not capture_truncated
        ),
        observer_stopped=bool(reader_errors),
    )

    observed_suspension = boundary_signal.get("suspension")
    if process_group_suspension is None and isinstance(observed_suspension, dict):
        process_group_suspension = observed_suspension
    natural_exact_exit = (
        turn_budget.exact_boundary_reached
        and exit_code == 0
        and not timed_out
        and cleanup.get("reason") in {"normal_exit", "exact_turn_boundary"}
        and cleanup.get("sigterm_sent") is False
        and cleanup.get("sigcont_sent") is False
        and cleanup.get("sigkill_sent") is False
        and cleanup.get("tracked_members_before_cleanup") == []
        and cleanup.get("verified_absent") is True
        and cleanup.get("process_tracker_errors") == []
        and cleanup.get("tracked_members_after_cleanup") == []
        and readers_completed
        and not reader_errors
        and not capture_truncated
    )
    if natural_exact_exit and not (
        isinstance(process_group_suspension, dict)
        and process_group_suspension.get("verified") is True
    ):
        cleanup["reason"] = "natural_exact_boundary_exit"
        cleanup["boundary_signal"] = {
            "attempted": True,
            "stdout_character_offset": boundary_signal[
                "stdout_character_offset"
            ],
        }
        if formal_campaign and boundary_candidate_snapshot is None:
            boundary_candidate_snapshot = artifact_dir / ".boundary-candidates"
            (
                boundary_raw_after,
                boundary_capture_errors,
                boundary_snapshot_receipt,
            ) = _capture_workspace_checkpoint(
                workspace=workspace_path,
                editable_files=editable_files,
                destination=boundary_candidate_snapshot,
                capture_mode="complete_natural_exit_process_tree_absent",
            )

    workspace_after: dict[str, dict[str, Any]] | None = None
    workspace_integrity: dict[str, Any] | None = None
    formal_cleanup_verified = (
        attempt_cleanup_verified(
            cleanup,
            exit_code=exit_code,
            allowed_reasons={"normal_exit", "exact_turn_boundary"},
            required_procfs="private_attempt_procfs",
        )
        if formal_campaign
        else True
    )
    complete_capture_and_cleanup = (
        not timed_out
        and cleanup["verified_absent"]
        and formal_cleanup_verified
        and readers_completed
        and not reader_errors
        and not capture_truncated
        and not turn_budget.enforcement_failed
    )
    suspended_boundary_evidence = (
        isinstance(process_group_suspension, dict)
        and process_group_suspension.get("policy_id")
        == BOUNDARY_QUIESCENCE_POLICY
        and process_group_suspension.get("method") == "sigstop_process_tree"
        and process_group_suspension.get("sent") is True
        and process_group_suspension.get("verified") is True
        and cleanup.get("reason") == "exact_turn_boundary"
        and cleanup.get("sigcont_sent") is True
        and isinstance(boundary_output_tail, dict)
        and boundary_output_tail.get("capture_truncated") is False
        and boundary_output_tail.get("readers_completed") is True
        and (
            not formal_campaign
            or isinstance(boundary_snapshot_receipt, dict)
            and boundary_snapshot_receipt.get("complete") is True
            and not boundary_capture_errors
        )
    )
    natural_boundary_evidence = (
        natural_exact_exit
        and cleanup.get("reason") == "natural_exact_boundary_exit"
        and (
            not formal_campaign
            or isinstance(boundary_snapshot_receipt, dict)
            and boundary_snapshot_receipt.get("capture_mode")
            == "complete_natural_exit_process_tree_absent"
            and boundary_snapshot_receipt.get("complete") is True
            and not boundary_capture_errors
        )
    )
    namespace_boundary_evidence = (
        formal_campaign
        and formal_cleanup_verified
        and cleanup.get("reason") == "exact_turn_boundary"
        and isinstance(boundary_output_tail, dict)
        and boundary_output_tail.get("capture_truncated") is False
        and boundary_output_tail.get("readers_completed") is True
        and isinstance(boundary_snapshot_receipt, dict)
        and boundary_snapshot_receipt.get("capture_mode")
        == "post_namespace_teardown_checkpoint"
        and boundary_snapshot_receipt.get("complete") is True
        and not boundary_capture_errors
    )
    exact_boundary_evidence = (
        namespace_boundary_evidence
        or suspended_boundary_evidence
        or natural_boundary_evidence
    )
    boundary_resolution = (
        "pid_namespace_destroyed_before_snapshot"
        if namespace_boundary_evidence
        else "verified_process_tree_suspension"
        if suspended_boundary_evidence
        else "complete_natural_exit_process_tree_absent"
        if natural_boundary_evidence
        else None
    )
    execution_completed = complete_capture_and_cleanup and (
        (
            not turn_budget.exact_boundary_reached
            and exit_code == 0
        )
        or (
            turn_budget.exact_boundary_reached
            and exact_boundary_evidence
            and isinstance(exit_code, int)
            and not isinstance(exit_code, bool)
            and exit_code
            in (
                {0, 128 + int(signal.SIGKILL)}
                if formal_campaign
                else {0, -int(signal.SIGTERM), -int(signal.SIGKILL)}
            )
        )
    )
    if formal_campaign:
        assert workspace_before is not None
        assert baseline_snapshot is not None
        retain_candidates = (
            execution_completed
            and not turn_budget.budget_exceeded
        )
        workspace_after, workspace_integrity = _sanitize_formal_workspace(
            workspace=workspace_path,
            baseline_snapshot=baseline_snapshot,
            before=workspace_before,
            editable_files=editable_files,
            retain_candidates=retain_candidates,
            artifact_dir=artifact_dir,
            captured_candidate_snapshot=(
                boundary_candidate_snapshot
                if turn_budget.exact_boundary_reached
                else None
            ),
            captured_raw_after=(
                boundary_raw_after if turn_budget.exact_boundary_reached else None
            ),
            captured_errors=(
                boundary_capture_errors if turn_budget.exact_boundary_reached else ()
            ),
        )
        shutil.rmtree(baseline_snapshot, ignore_errors=True)

    persistence_evidence_complete = (
        execution_completed
        and not turn_budget.budget_exceeded
        and (
            not formal_campaign
            or (
                isinstance(workspace_integrity, dict)
                and workspace_integrity.get("passed") is True
            )
        )
    )
    candidate_persistence = _candidate_persistence_receipt(
        turn_budget=turn_budget,
        workspace_integrity=workspace_integrity,
        evidence_complete=persistence_evidence_complete,
        attempt_contained=formal_campaign,
        boundary_snapshot=boundary_snapshot_receipt,
        output_tail=boundary_output_tail,
        boundary_resolution=boundary_resolution,
        attempt_cleanup=cleanup,
    )
    session_succeeded = persistence_evidence_complete
    receipt = {
        "schema": _RECEIPT_SCHEMA,
        "campaign_binding": campaign_binding,
        "comparison_contract_sha256": comparison_contract_sha256,
        "session_succeeded": session_succeeded,
        "thread_id": session["thread_id"],
        "session_id": session["session_id"],
        "session": session,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "effective_timeout_seconds": effective_timeout,
        "capture": {
            "readers_completed": readers_completed,
            "errors": reader_errors,
            "stdout": capture_state["stdout"],
            "stderr": capture_state["stderr"],
        },
        "turn_budget": turn_budget.receipt(),
        "candidate_persistence": candidate_persistence,
        "workspace_integrity": workspace_integrity,
        "gpu": gpu_evidence,
        "codex": {
            "discovered_path": codex_bin,
            "executed_path": executed_codex_bin,
            "binary_sha256": codex_binary_sha256,
            "runtime_closure_sha256": runtime_closure_sha256,
            "version": codex_version,
            "model": configured_model,
            "effort": configured_effort,
            "cloud_config_bootstrap": cloud_config_bootstrap,
        },
        "invocation": {
            "argv_without_prompt": cmd[:-1],
            "prompt_sha256": _sha256_bytes(rendered_prompt),
            "workspace": str(workspace_path),
            "editable_files": list(editable_files),
            "max_turns": max_turns,
            "turn_policy": TURN_POLICY,
            "candidate_persistence_policy_id": CANDIDATE_PERSISTENCE_POLICY,
            "agent_process_containment_policy_id": (
                AGENT_PROCESS_CONTAINMENT_POLICY
            ),
            "attempt_containment_policy_id": ATTEMPT_CONTAINMENT_POLICY,
            "attempt_process_boundary": attempt_boundary_receipt,
            "structured_stream_output_limit_bytes": output_limit,
            "isolation": {
                "approval": "never_via_strict_config",
                "execpolicy_rules": "ignored",
                "project_instructions": "backend_default_may_load",
                "sandbox": "workspace-write",
                "session": "ephemeral",
                "user_config": "ignored",
                "mount_scope": (
                    "attempt_only_bubblewrap"
                    if attempt_home is not None
                    else "ordinary_run"
                ),
                **(
                    {"attempt_containment_policy_id": ATTEMPT_CONTAINMENT_POLICY}
                    if attempt_home is not None
                    else {}
                ),
            },
        },
        "aggregated_usage": aggregated_usage,
    }
    if formal_campaign:
        receipt["attempt_process_cleanup"] = cleanup
    else:
        receipt["process_group_cleanup"] = cleanup
        receipt["process_group_suspension"] = process_group_suspension
    _write_attempt_receipt(
        receipt_path=receipt_path,
        artifact_dir=artifact_dir,
        raw_stdout=raw_stdout,
        raw_stderr=raw_stderr,
        transcript=output,
        rendered_prompt=rendered_prompt,
        receipt=receipt,
        workspace_before=workspace_before,
        workspace_after=workspace_after,
    )

    if timed_out:
        verification = cleanup["verified_absent"]
        raise CodexSessionTimeout(
            "Codex session timed out; attempt cleanup "
            f"verified_absent={verification}; receipt={receipt_path}"
        )
    if capture_truncated:
        raise CodexSessionError(
            f"Codex output exceeded the bounded capture limit; receipt={receipt_path}"
        )
    if turn_budget.budget_exceeded or turn_budget.enforcement_failed:
        raise CodexSessionError(
            "Codex structured turn budget was exceeded or could not be enforced; "
            f"receipt={receipt_path}"
        )
    if exit_code != 0 and not turn_budget.exact_boundary_reached:
        raise CodexSessionError(
            f"Codex session exited with status {exit_code}; receipt={receipt_path}"
        )
    if formal_campaign and not (
        isinstance(workspace_integrity, dict)
        and workspace_integrity.get("passed") is True
    ):
        raise CodexSessionError(
            f"Codex workspace could not be reduced to declared sources; receipt={receipt_path}"
        )
    if not session_succeeded:
        raise CodexSessionError(
            f"Codex session evidence capture was incomplete; receipt={receipt_path}"
        )
    return output
