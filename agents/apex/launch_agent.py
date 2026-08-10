# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""AgentKernelArena adapter for Apex's caller-neutral kernel CLI.

The adapter translates an Arena task into Apex's versioned ``TaskSpec`` and
runs Apex out of process. Apex never edits the scored Arena workspace directly:
it returns a content-addressed source patch bundle, which this module validates
against the frozen source hashes before applying it. AgentKernelArena then runs
its existing harness guard and centralized compile/correctness/performance
pipeline. Apex's internal score or safety opinion is never consumed here.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import signal
import sqlite3
import stat
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import yaml

from agents import register_agent
from src.campaign import CampaignError, campaign_task_path_component
from src.agent_turn_budget import (
    AGENT_PROCESS_CONTAINMENT_POLICY,
    CANDIDATE_PERSISTENCE_POLICY,
    FORMAL_MATCHED_MAX_TURNS,
    TURN_POLICY,
    budget_stop_reason_matches,
    context_packet_objective_matches,
    render_apex_run_control,
)
from src.module_registration import AgentType, load_prompt_builder
from src.campaign_isolation import (
    APEX_RUNTIME_MOUNT_POLICY,
    APEX_RUNTIME_MOUNT_SCHEMA,
    ATTEMPT_MOUNT_RECEIPT_SCHEMA,
    ATTEMPT_CONTAINMENT_POLICY,
    attempt_cleanup_verified,
    attempt_command_pass_fds,
    attempt_mount_receipt,
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
from src.apex_runtime import (
    ApexRuntimeError,
    RUNTIME_IMMUTABLE_MOUNT_POLICY_ID,
    RUNTIME_IMMUTABLE_MOUNT_SCHEMA,
    RuntimePlan,
    runtime_command,
    runtime_environment,
    runtime_image_inputs,
    validate_immutable_mount_receipt,
    verify_runtime_snapshot,
)


_SCHEMA_VERSION = 1
_DEFAULT_RESULT_LIMIT = 8 * 1024 * 1024
_DEFAULT_BUNDLE_LIMIT = 64 * 1024 * 1024
_DEFAULT_OUTPUT_LIMIT = 4 * 1024 * 1024
_MAX_WORKSPACE_FILES = 20_000
_MAX_WORKSPACE_BYTES = 2 * 1024 * 1024 * 1024
_APEX_INSTRUCTION_LIMIT = 8_192
_APEX_GENERIC_CONTEXT_MARKER = (
    "\n# AMD MI355X (CDNA 4) Kernel Optimization Context & Directives"
)
_NORMAL_NO_PATCH_STATUSES = {"no_gain"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_APEX_RECEIPT_SCHEMA = "agentkernelarena.apex-attempt-receipt/v7"
_NAMESPACE_MOUNT_POLICY = "blocked_namespace_mount_attestation_v2"
_VISIBLE_MOUNT_RESOLUTION_POLICY = "proc_root_o_path_fdinfo_mnt_id_v1"
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
_CALLER_RUN_CONTROL_SCHEMA = "aka.apex-caller-run-control/v1"
_INSTRUCTION_ADAPTATION_SCHEMA = "aka.apex-instruction-adaptation/v1"
_ARENA_PYTHON_ENV = "AGENT_KERNEL_ARENA_PYTHON"
_TERM_GRACE_SECONDS = 10.0
_KILL_GRACE_SECONDS = 5.0
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHELL_OPERATOR_TOKENS = {
    "|",
    "||",
    "&",
    "&&",
    ";",
    "<",
    ">",
    ">>",
    "2>",
    "2>>",
}


class ApexAdapterError(RuntimeError):
    """Raised when the Apex handoff or returned bundle violates the contract."""


@dataclass(frozen=True)
class ApexProcessOutcome:
    exit_code: int | None
    stdout: bytes
    stderr: bytes
    timed_out: bool
    cleanup: dict[str, Any]
    readers_completed: bool
    capture_errors: tuple[str, ...]

    @property
    def output(self) -> str:
        return "\n".join(
            part
            for part in (
                self.stdout.decode("utf-8", "replace"),
                self.stderr.decode("utf-8", "replace"),
            )
            if part
        )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ApexAdapterError(f"YAML document must be a mapping: {path}")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _inner_agent_containment_valid(
    receipt: Any, *, forced_stop: bool
) -> bool:
    """Recompute Apex's backend PID-namespace proof before accepting bytes."""
    if not isinstance(receipt, dict):
        return False
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
    positive = (
        "namespace_init_host_pid",
        "namespace_init_starttime",
        "namespace_init_inner_pid",
        "pid_namespace_inode",
        "mount_namespace_inode",
        "ipc_namespace_inode",
        "user_namespace_inode",
    )
    terminal_verified = receipt.get("terminal_status_verified") is True
    terminal_absent = (
        receipt.get("terminal_status_absent_after_sigkill") is True
    )
    terminal = terminal_verified != terminal_absent
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
            for field in positive
        )
        and receipt.get("namespace_init_inner_pid") == 1
        and receipt.get("private_procfs_verified") is True
        and receipt.get("pidfd_opened") is True
        and receipt.get("namespace_init_exit_verified") is True
        and receipt.get("wrapper_exit_verified") is True
        and receipt.get("wrapper_force_killed") is False
        and terminal_fields_typed
        and terminal
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
            and terminal
        )
    return (
        receipt.get("termination_reason") == "natural_exit"
        and receipt.get("teardown_mode") == "natural_exit"
        and receipt.get("pidfd_sigkill_sent") is False
        and terminal_verified
        and not terminal_absent
    )


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
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise ApexAdapterError(
            "formal Apex attempt lacks a valid comparison contract digest"
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
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise ApexAdapterError(
            "formal Apex attempt lacks a backend runtime closure digest"
        )
    return digest


def _read_immutable_campaign_manifest(path: Path) -> tuple[dict[str, Any], str]:
    """Read canonical manifest bytes without following the final path component."""

    if not path.is_absolute():
        raise ApexAdapterError("campaign manifest path must be absolute")
    try:
        canonical = path.resolve(strict=True)
        if canonical != path:
            raise ApexAdapterError("campaign manifest path must be canonical")
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
                raise ApexAdapterError(
                    "campaign manifest must be a single-link read-only regular file"
                )
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                payload = stream.read()
        finally:
            os.close(descriptor)
        manifest = yaml.safe_load(payload.decode("utf-8")) or {}
    except ApexAdapterError:
        raise
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ApexAdapterError(
            f"cannot read immutable campaign manifest: {error}"
        ) from error
    if not isinstance(manifest, dict):
        raise ApexAdapterError("campaign manifest must contain a mapping")
    return manifest, _sha256_bytes(payload)


def _task_package_manifest(root: Path) -> tuple[dict[str, str], str]:
    if not root.is_dir() or root.is_symlink():
        raise ApexAdapterError("campaign task package root is unsafe")
    files: dict[str, str] = {}
    try:
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root).as_posix()
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ApexAdapterError(
                    f"campaign task package contains a symlink: {relative}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                continue
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ApexAdapterError(
                    f"campaign task package contains an unsafe file: {relative}"
                )
            files[relative] = _sha256_file(path)
    except OSError as error:
        raise ApexAdapterError(
            f"cannot hash campaign task package: {error}"
        ) from error
    if not files:
        raise ApexAdapterError("campaign task package contains no files")
    return files, _canonical_digest(files)


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
        raise ApexAdapterError("formal Apex attempt binding is missing")
    raw_manifest_path = attempt.get("campaign_manifest_path")
    if not isinstance(raw_manifest_path, str):
        raise ApexAdapterError("formal Apex campaign manifest path is missing")
    manifest_path = Path(raw_manifest_path)
    manifest, manifest_digest = _read_immutable_campaign_manifest(manifest_path)
    comparison = manifest.get("comparison_contract")
    comparison_digest = (
        _canonical_digest(comparison) if isinstance(comparison, dict) else None
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
        raise ApexAdapterError("formal Apex campaign manifest is malformed")
    try:
        task_names: list[str] = []
        task_components: list[str] = []
        for expected_index, (manifest_task, manifest_mapping) in enumerate(
            zip(tasks, mappings), 1
        ):
            if not isinstance(manifest_task, dict) or not isinstance(
                manifest_mapping, dict
            ):
                raise ApexAdapterError("formal Apex campaign task mapping is malformed")
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
                raise ApexAdapterError("formal Apex campaign task mapping is malformed")
            task_names.append(manifest_name)
            task_components.append(campaign_task_path_component(manifest_name))
    except CampaignError as error:
        raise ApexAdapterError(
            f"formal Apex campaign task path is unsafe: {error}"
        ) from error
    if (
        len(task_names) != len(set(task_names))
        or len(task_components) != len(set(task_components))
    ):
        raise ApexAdapterError("formal Apex campaign task paths are not injective")
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
        raise ApexAdapterError("formal Apex campaign indices are invalid")
    task = tasks[task_index - 1]
    mapping = mappings[task_index - 1]
    if not isinstance(task, dict) or not isinstance(mapping, dict):
        raise ApexAdapterError("formal Apex campaign task is malformed")
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
        raise ApexAdapterError("formal Apex task or GPU mapping differs from campaign")
    raw_receipt_path = attempt.get("receipt_path")
    if not isinstance(raw_receipt_path, str) or not Path(
        raw_receipt_path
    ).is_absolute():
        raise ApexAdapterError("formal Apex receipt path is missing")
    receipt_path = Path(raw_receipt_path).resolve(strict=False)
    expected_receipt_path = (
        manifest_path.parent
        / ".campaign_attempts"
        / campaign_task_path_component(str(task_name))
        / f"attempt_{attempt_index:02d}"
        / "session_receipt.json"
    )
    if receipt_path != expected_receipt_path:
        raise ApexAdapterError(
            "formal Apex receipt is outside its bound campaign attempt"
        )
    try:
        canonical_config = task_config_path.resolve(strict=True)
        config_metadata = canonical_config.lstat()
    except OSError as error:
        raise ApexAdapterError(
            f"cannot inspect formal Apex task config: {error}"
        ) from error
    if (
        str(canonical_config) != task.get("config_path")
        or not canonical_config.is_file()
        or canonical_config.is_symlink()
        or config_metadata.st_nlink != 1
    ):
        raise ApexAdapterError("formal Apex task config path is not campaign-bound")
    config_digest = _sha256_file(canonical_config)
    package_files, package_digest = _task_package_manifest(canonical_config.parent)
    if (
        config_digest != task.get("config_sha256")
        or package_files != task.get("package_files_sha256")
        or package_digest != task.get("package_manifest_sha256")
    ):
        raise ApexAdapterError("formal Apex task package differs from campaign")
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
    if (
        set(binding) != _CAMPAIGN_BINDING_KEYS
        or binding != expected
        or any(
            not isinstance(binding[key], str)
            or not _SHA256.fullmatch(binding[key])
            for key in (
                "formal_execution_sha256",
                "campaign_manifest_sha256",
                "comparison_contract_sha256",
                "backend_runtime_closure_sha256",
                "task_package_manifest_sha256",
                "task_config_sha256",
            )
        )
    ):
        raise ApexAdapterError(
            "formal Apex attempt binding differs from immutable campaign"
        )
    return binding


def _apex_task_instructions(
    prompt: str,
    *,
    workspace: Path,
    sources: Iterable[str],
    symbols: Iterable[str],
    caller_run_control: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Preserve the Arena task contract within Apex's bounded objective field.

    Arena's shared prompt builder appends a large generic MI355X/Triton
    knowledge section after the task-specific objective, source, harness, and
    completion contract. Apex supplies that generic knowledge through its own
    bounded knowledge layer. Unknown oversized layouts fail closed instead of
    being silently or heuristically truncated.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ApexAdapterError("Apex task prompt must be non-empty text")

    original_bytes = prompt.encode("utf-8")
    strategy = "verbatim"
    marker = None
    instructions = prompt
    if len(prompt) > _APEX_INSTRUCTION_LIMIT:
        marker_count = prompt.count(_APEX_GENERIC_CONTEXT_MARKER)
        if marker_count != 1:
            raise ApexAdapterError(
                "oversized Apex prompt does not contain exactly one known generic "
                f"context boundary; count={marker_count}"
            )
        boundary = prompt.index(_APEX_GENERIC_CONTEXT_MARKER)
        task_contract = prompt[:boundary].rstrip()
        if not task_contract:
            raise ApexAdapterError("Apex prompt task contract is empty before context boundary")
        source_lines = "\n".join(f"- `{value}`" for value in sources)
        symbol_lines = "\n".join(f"- `{value}`" for value in symbols)
        structural_handoff = (
            "### Structured Apex handoff\n\n"
            "AgentKernelArena's generic MI355X and Triton cheatsheets are "
            "intentionally not duplicated here. Apex supplies scoped "
            "architecture and optimization knowledge through its own "
            "provenance-aware ContextPacket.\n\n"
            f"Scored workspace: `{workspace}`\n\n"
            f"Editable source files:\n{source_lines}\n\n"
            f"Target kernel functions:\n{symbol_lines}\n\n"
            "The TaskSpec separately binds evaluator commands, source hashes, "
            "GPU architecture, and bundle delivery."
        )
        instructions = f"{task_contract}\n\n{structural_handoff}"
        strategy = "omit_known_generic_mi355x_triton_context_v1"
        marker = _APEX_GENERIC_CONTEXT_MARKER.lstrip("\n")

    if caller_run_control is not None:
        instructions = (
            f"{instructions.rstrip()}\n\n{render_apex_run_control(caller_run_control)}"
        )
        strategy = (
            "append_formal_run_control_v1"
            if strategy == "verbatim"
            else f"{strategy}_and_append_formal_run_control_v1"
        )

    if len(instructions) > _APEX_INSTRUCTION_LIMIT:
        raise ApexAdapterError(
            "Apex task instructions exceed the ContextPacket text limit after adaptation: "
            f"{len(instructions)} > {_APEX_INSTRUCTION_LIMIT}"
        )

    adapted_bytes = instructions.encode("utf-8")
    provenance = {
        "schema": _INSTRUCTION_ADAPTATION_SCHEMA,
        "strategy": strategy,
        "limit_characters": _APEX_INSTRUCTION_LIMIT,
        "boundary_marker": marker,
        "original": {
            "characters": len(prompt),
            "bytes": len(original_bytes),
            "sha256": _sha256_bytes(original_bytes),
        },
        "adapted": {
            "characters": len(instructions),
            "bytes": len(adapted_bytes),
            "sha256": _sha256_bytes(adapted_bytes),
        },
    }
    return instructions, provenance


def _formal_python_interpreter(*, required: bool) -> dict[str, Any] | None:
    """Resolve the caller-selected interpreter without silently choosing a fallback."""

    raw = os.environ.get(_ARENA_PYTHON_ENV)
    if not raw:
        if required:
            raise ApexAdapterError(
                f"formal Apex requires {_ARENA_PYTHON_ENV} to be set"
            )
        return None
    path = Path(raw)
    if not path.is_absolute():
        raise ApexAdapterError(f"{_ARENA_PYTHON_ENV} must be an absolute path")
    try:
        resolved = path.resolve(strict=True)
        metadata = resolved.stat()
    except OSError as error:
        raise ApexAdapterError(
            f"{_ARENA_PYTHON_ENV} does not resolve to a usable interpreter"
        ) from error
    if not stat.S_ISREG(metadata.st_mode) or not os.access(resolved, os.X_OK):
        raise ApexAdapterError(
            f"{_ARENA_PYTHON_ENV} must resolve to an executable regular file"
        )
    return {
        "environment_variable": _ARENA_PYTHON_ENV,
        "path": raw,
        "resolved_path": str(resolved),
        "sha256": _sha256_file(resolved),
    }


def _bind_formal_python(
    command: dict[str, Any], interpreter: dict[str, Any]
) -> dict[str, Any]:
    """Make Python verifier argv use the exact interpreter selected by the runner."""

    argv = list(command["argv"])
    launcher = Path(argv[0]).name if argv else ""
    if launcher in {"python", "python3"}:
        argv[0] = interpreter["path"]
    elif launcher in {"pytest", "py.test"}:
        argv = [interpreter["path"], "-m", "pytest", *argv[1:]]
    return {"argv": argv, "timeout_seconds": command["timeout_seconds"]}


def _caller_run_control(
    *,
    formal_campaign: bool,
    commands: dict[str, dict[str, Any]],
    max_turns: int,
    max_iterations: int,
) -> dict[str, Any] | None:
    if not formal_campaign:
        return None
    interpreter = _formal_python_interpreter(required=True)
    assert interpreter is not None
    bound_commands = {
        phase: _bind_formal_python(commands[phase], interpreter)
        for phase in ("compile", "correctness", "performance")
    }
    return {
        "schema": _CALLER_RUN_CONTROL_SCHEMA,
        "deliverable_versions": max_iterations,
        "structured_turn_budget": {
            "policy": TURN_POLICY,
            "max_turns": max_turns,
            "counting": "assistant_message_and_tool_call_start_each_count_once",
        },
        "candidate_persistence_policy_id": CANDIDATE_PERSISTENCE_POLICY,
        "process_containment_policy_id": AGENT_PROCESS_CONTAINMENT_POLICY,
        "python_interpreter": interpreter,
        "verifier_argv": {
            phase: list(bound_commands[phase]["argv"])
            for phase in ("compile", "correctness", "performance")
        },
    }


def _workspace_manifest(workspace: Path) -> dict[str, dict[str, Any]]:
    """Freeze the complete scored tree before an untrusted Apex subprocess."""
    manifest: dict[str, dict[str, Any]] = {}
    total_size = 0
    try:
        paths = sorted(workspace.rglob("*"))
        for path in paths:
            relative = path.relative_to(workspace).as_posix()
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ApexAdapterError(f"workspace contains a symlink: {relative}")
            if stat.S_ISDIR(metadata.st_mode):
                continue
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ApexAdapterError(f"workspace contains an unsafe file: {relative}")
            total_size += metadata.st_size
            if len(manifest) >= _MAX_WORKSPACE_FILES or total_size > _MAX_WORKSPACE_BYTES:
                raise ApexAdapterError("workspace manifest exceeds formal campaign limits")
            manifest[relative] = {
                "sha256": _sha256_file(path),
                "size_bytes": metadata.st_size,
                "mode": format(stat.S_IMODE(metadata.st_mode), "04o"),
            }
    except OSError as error:
        raise ApexAdapterError(f"cannot freeze scored workspace: {error}") from error
    if not manifest:
        raise ApexAdapterError("formal Apex workspace is empty")
    return manifest


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _normalize_relative_path(raw: Any, *, field: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ApexAdapterError(f"{field} entries must be non-empty strings")
    value = raw.strip()
    if "\x00" in value or "\\" in value:
        raise ApexAdapterError(f"{field} contains an unsafe path: {raw!r}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ApexAdapterError(f"{field} must be a normalized workspace-relative path: {raw!r}")
    return path.as_posix()


def _ensure_regular_workspace_file(workspace: Path, relative: str) -> Path:
    current = workspace
    for part in PurePosixPath(relative).parts:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as error:
            raise ApexAdapterError(
                f"declared source file does not exist in workspace: {relative}"
            ) from error
        if stat.S_ISLNK(mode):
            raise ApexAdapterError(f"declared source path traverses a symlink: {relative}")
    metadata = current.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ApexAdapterError(f"declared source is not a regular file: {relative}")
    if metadata.st_nlink != 1:
        raise ApexAdapterError(f"declared source must not be hard-linked: {relative}")
    return current


def _string_list(raw: Any, *, field: str, required: bool = True) -> list[str]:
    values = [raw] if isinstance(raw, str) else raw
    if values is None:
        values = []
    if not isinstance(values, list):
        raise ApexAdapterError(f"{field} must be a string or list of strings")
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ApexAdapterError(f"{field} entries must be non-empty strings")
        normalized = value.strip()
        if normalized not in result:
            result.append(normalized)
    if required and not result:
        raise ApexAdapterError(f"{field} is required")
    return result


def _declared_sources(task_config: dict[str, Any], workspace: Path) -> list[str]:
    sources = [
        _normalize_relative_path(value, field="source_file_path")
        for value in _string_list(task_config.get("source_file_path"), field="source_file_path")
    ]
    for source in sources:
        _ensure_regular_workspace_file(workspace, source)
    return sources


def _command_specs(
    task_config: dict[str, Any],
    phase: str,
    default_timeout: int,
) -> dict[str, Any]:
    commands = _string_list(
        task_config.get(f"{phase}_command"),
        field=f"{phase}_command",
    )
    if len(commands) != 1:
        raise ApexAdapterError(
            f"Apex V1 requires exactly one {phase}_command; got {len(commands)}"
        )
    timeout = int(task_config.get(f"{phase}_timeout", default_timeout))
    if timeout <= 0:
        raise ApexAdapterError(f"{phase}_timeout must be positive")
    command = commands[0]
    try:
        argv = shlex.split(command, posix=True)
    except ValueError as error:
        raise ApexAdapterError(f"invalid {phase}_command: {error}") from error
    if not argv or any(token in _SHELL_OPERATOR_TOKENS for token in argv):
        raise ApexAdapterError(
            f"{phase}_command must be representable as argv without shell operators: {command!r}"
        )
    return {"argv": argv, "timeout_seconds": timeout}


def _task_id(task_config_path: Path) -> str:
    parts = task_config_path.resolve().parts
    try:
        index = len(parts) - 1 - tuple(reversed(parts)).index("tasks")
    except ValueError:
        relative = (task_config_path.parent.name,)
    else:
        relative = parts[index + 1 : -1]
    logical = "/".join(relative) or task_config_path.parent.name
    candidate = ".".join(relative) or task_config_path.parent.name
    if _IDENTIFIER.fullmatch(candidate):
        return candidate
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", candidate).strip(".-_") or "task"
    suffix = hashlib.sha256(logical.encode("utf-8")).hexdigest()[:16]
    return f"{slug[:110]}-{suffix}"


def _framework(task_config_path: Path) -> str | None:
    known = {"aiter", "pytorch", "rocmbench", "sglang", "vllm"}
    for part in reversed(task_config_path.resolve().parts[:-1]):
        if part.lower() in known:
            return part.lower()
    return None


def _gpu_arch(eval_config: dict[str, Any]) -> str:
    observed = os.environ.get("PYTORCH_ROCM_ARCH", "").split(":", 1)[0].strip()
    if observed.startswith("gfx"):
        return observed
    model = str(eval_config.get("target_gpu_model") or "").strip().upper()
    mapping = {
        "MI250": "gfx90a",
        "MI250X": "gfx90a",
        "MI300": "gfx942",
        "MI300A": "gfx940",
        "MI300X": "gfx942",
        "MI355X": "gfx950",
    }
    try:
        return mapping[model]
    except KeyError as error:
        raise ApexAdapterError(
            f"cannot resolve GPU architecture from target_gpu_model={model!r}"
        ) from error


def _build_task_spec(
    *,
    task_config_path: Path,
    task_config: dict[str, Any],
    eval_config: dict[str, Any],
    agent_config: dict[str, Any],
    workspace: Path,
    artifact_root: Path,
    prompt: str,
    formal_campaign: bool | None = None,
    campaign_binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    formal_campaign = (
        is_formal_campaign(eval_config)
        if formal_campaign is None
        else formal_campaign
    )
    if formal_campaign and (
        not isinstance(campaign_binding, dict)
        or set(campaign_binding) != _CAMPAIGN_BINDING_KEYS
        or campaign_binding.get("schema") != _CAMPAIGN_BINDING_SCHEMA
    ):
        raise ApexAdapterError("formal Apex TaskSpec requires an exact campaign binding")
    if not formal_campaign and campaign_binding is not None:
        raise ApexAdapterError("ordinary Apex TaskSpec cannot claim a campaign binding")
    task_type = str(task_config.get("task_type") or "").strip()
    supported = set(_string_list(
        agent_config.get("supported_task_types", []),
        field="supported_task_types",
        required=False,
    ))
    if task_type not in supported:
        raise ApexAdapterError(
            f"Apex adapter does not support task_type={task_type!r}; supported={sorted(supported)}"
        )
    if task_type != "triton2triton":
        raise ApexAdapterError(
            f"task_type={task_type!r} lacks a trusted V1 recipe; only triton2triton is enabled"
        )

    sources = _declared_sources(task_config, workspace)
    symbols = _string_list(
        task_config.get("target_kernel_functions"),
        field="target_kernel_functions",
    )
    compile_commands = _command_specs(
        task_config,
        "compile",
        int(agent_config.get("compile_timeout_seconds", 3600)),
    )
    correctness_commands = _command_specs(
        task_config,
        "correctness",
        int(agent_config.get("correctness_timeout_seconds", 3600)),
    )
    performance_commands = _command_specs(
        task_config,
        "performance",
        int(agent_config.get("performance_timeout_seconds", 3600)),
    )

    file_hashes = {
        relative: _sha256_file(_ensure_regular_workspace_file(workspace, relative))
        for relative in sources
    }
    commands: dict[str, dict[str, Any]] = {
        "compile": compile_commands,
        "correctness": correctness_commands,
        "performance": performance_commands,
    }
    backend = str(agent_config.get("backend") or "codex").strip().lower()
    if backend not in {"codex", "claude", "cursor"}:
        raise ApexAdapterError(
            f"Apex backend must be codex, claude, or cursor; got {backend!r}"
        )
    task_id = _task_id(task_config_path)
    campaign = eval_config.get("campaign") or {}
    iteration_budget = int(agent_config.get("max_iterations", 1))
    max_turns = int(agent_config.get("max_turns", 25))
    if campaign.get("comparison") == "apex_vs_codex":
        iteration_budget = int(agent_config.get("campaign_max_iterations", 0))
        if max_turns != FORMAL_MATCHED_MAX_TURNS:
            raise ApexAdapterError(
                f"formal Apex requires max_turns={FORMAL_MATCHED_MAX_TURNS}"
            )
    if formal_campaign and iteration_budget != 1:
        raise ApexAdapterError("formal Apex requires exactly one deliverable version")
    caller_run_control = _caller_run_control(
        formal_campaign=formal_campaign,
        commands=commands,
        max_turns=max_turns,
        max_iterations=iteration_budget,
    )
    if caller_run_control is not None:
        interpreter = caller_run_control["python_interpreter"]
        commands = {
            phase: _bind_formal_python(commands[phase], interpreter)
            for phase in ("compile", "correctness", "performance")
        }
    recipe_material = {
        "task_config": _sha256_file(task_config_path),
        "commands": commands,
        "source_files": sources,
    }
    instructions, instruction_adaptation = _apex_task_instructions(
        prompt,
        workspace=workspace,
        sources=sources,
        symbols=symbols,
        caller_run_control=caller_run_control,
    )
    return {
        "schema_version": _SCHEMA_VERSION,
        "task_id": task_id,
        "workspace": str(workspace),
        "results_dir": str(artifact_root),
        "instructions": instructions,
        # Apex ignores unknown TaskSpec fields. This adapter-owned receipt makes
        # the transform immutable without expanding Apex's caller-neutral V1 API.
        "instruction_adaptation": instruction_adaptation,
        "caller_run_control": caller_run_control,
        "campaign_binding": campaign_binding,
        "language": "triton",
        "editable_files": sources,
        "target_functions": symbols,
        "commands": commands,
        "gpu_arch": _gpu_arch(eval_config),
        "mode": "optimize_existing",
        "agent_backend": backend,
        "agent_options": {
            "model": agent_config.get("model"),
            "effort": agent_config.get("effort"),
            "runtime_closure_sha256": _runtime_closure_sha256(
                eval_config, formal_campaign=formal_campaign
            ),
        },
        "budget": {
            "max_iterations": iteration_budget,
            "max_turns": max_turns,
            "timeout_seconds": int(agent_config.get("timeout_seconds", 3600)),
        },
        "recipe": {
            "kind": "python_triton",
            "recipe_id": "external-central-evaluator-v1",
            "sha256": _canonical_digest(recipe_material),
            "provenance": "external_evaluator",
        },
        "delivery": {"mode": "bundle"},
        # Caller-owned evidence is deliberately redundant with Apex's resolver.
        # It lets this adapter bind the returned patch to the exact source bytes
        # that existed before the untrusted optimizer subprocess was launched.
        "baseline": {
            "repository": None,
            "commit": None,
            "tree_hash": _canonical_digest(file_hashes),
            "dirty_policy": "reject",
            "file_hashes": file_hashes,
        },
        "framework": _framework(task_config_path),
    }


def _process_group_exists(process_group_id: int) -> bool:
    live_members = _linux_live_process_group_members(process_group_id)
    if live_members is not None:
        return live_members
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _linux_live_process_group_members(process_group_id: int) -> bool | None:
    """Distinguish executable processes from harmless unreaped zombies on Linux."""
    proc = Path("/proc")
    if not proc.is_dir():
        return None
    try:
        entries = list(proc.iterdir())
    except OSError:
        return None
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            stat_line = (entry / "stat").read_text(encoding="utf-8")
            fields = stat_line[stat_line.rfind(")") + 2 :].split()
            state = fields[0]
            observed_group = int(fields[2])
        except (OSError, ValueError, IndexError):
            continue
        if observed_group == process_group_id and state != "Z":
            return True
    return False


def _wait_for_process_group_exit(
    process: subprocess.Popen[bytes], process_group_id: int, timeout: float
) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        process.poll()
        if not _process_group_exists(process_group_id):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def _terminate_process_group(
    process: subprocess.Popen[bytes],
    logger: logging.Logger,
    *,
    reason: str,
) -> dict[str, Any]:
    process_group_id = process.pid
    cleanup: dict[str, Any] = {
        "required": True,
        "reason": reason,
        "pgid": process_group_id,
        "sigterm_sent": False,
        "sigkill_sent": False,
        "verification_performed": True,
        "verified_absent": False,
    }
    if not _process_group_exists(process_group_id):
        process.poll()
        cleanup["verified_absent"] = True
        return cleanup
    try:
        os.killpg(process_group_id, signal.SIGTERM)
        cleanup["sigterm_sent"] = True
    except ProcessLookupError:
        cleanup["verified_absent"] = not _process_group_exists(process_group_id)
        return cleanup
    if _wait_for_process_group_exit(process, process_group_id, _TERM_GRACE_SECONDS):
        cleanup["verified_absent"] = True
        return cleanup
    logger.warning("Force killing Apex process group")
    try:
        os.killpg(process_group_id, signal.SIGKILL)
        cleanup["sigkill_sent"] = True
    except ProcessLookupError:
        cleanup["verified_absent"] = not _process_group_exists(process_group_id)
        return cleanup
    cleanup["verified_absent"] = _wait_for_process_group_exit(
        process, process_group_id, _KILL_GRACE_SECONDS
    )
    return cleanup


def _capture_stream(
    stream,
    *,
    label: str,
    limit: int,
    chunks: list[bytes],
    errors: list[str],
    logger,
) -> None:
    captured = 0
    warned = False
    try:
        while True:
            chunk = stream.read(4096)
            if not chunk:
                break
            remaining = limit - captured
            if remaining > 0:
                retained = chunk[:remaining]
                chunks.append(retained)
                captured += len(retained)
                compact = " ".join(retained.decode("utf-8", "replace").split())
                if compact:
                    logger(f"{label} {compact[:500]}")
            if len(chunk) > max(remaining, 0) and not warned:
                warned = True
                logger(f"{label} output truncated at {limit} bytes")
    except Exception as error:
        errors.append(f"{label}: {type(error).__name__}: {error}")
    finally:
        try:
            stream.close()
        except Exception as error:
            errors.append(f"{label} close: {type(error).__name__}: {error}")


def _subprocess_environment(backend: str) -> dict[str, str]:
    environment = os.environ.copy()
    if backend != "codex":
        environment.pop("OPENAI_API_KEY", None)
        environment.pop("CODEX_HOME", None)
    if backend != "claude":
        for name in list(environment):
            if name.startswith("ANTHROPIC_") or name.startswith("CLAUDE_CODE_"):
                environment.pop(name, None)
    if backend != "cursor":
        for name in list(environment):
            if name.startswith("CURSOR_"):
                environment.pop(name, None)
    return environment


def _run_apex(
    command: list[str],
    *,
    cwd: Path,
    backend: str,
    timeout_seconds: float,
    output_limit: int,
    logger: logging.Logger,
    environment: dict[str, str] | None = None,
) -> ApexProcessOutcome:
    pass_fds = attempt_command_pass_fds(command)
    try:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment or _subprocess_environment(backend),
            start_new_session=True,
            pass_fds=pass_fds,
        )
    finally:
        release_attempt_command_fds(command)
    attempt_boundary = establish_attempt_boundary(command, process)
    assert process.stdout is not None
    assert process.stderr is not None
    stdout: list[bytes] = []
    stderr: list[bytes] = []
    capture_errors: list[str] = []
    stdout_thread = threading.Thread(
        target=_capture_stream,
        kwargs={
            "stream": process.stdout,
            "label": "[APEX]",
            "limit": output_limit,
            "chunks": stdout,
            "errors": capture_errors,
            "logger": logger.info,
        },
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_capture_stream,
        kwargs={
            "stream": process.stderr,
            "label": "[APEX STDERR]",
            "limit": output_limit,
            "chunks": stderr,
            "errors": capture_errors,
            "logger": logger.warning,
        },
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
        "sigkill_sent": False,
        "verification_performed": False,
        "verified_absent": False,
    }
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        if attempt_boundary is not None:
            cleanup = finalize_attempt_boundary(
                process,
                attempt_boundary,
                reason="timeout",
                terminate=True,
            )
        else:
            cleanup = _terminate_process_group(process, logger, reason="timeout")
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                cleanup["verified_absent"] = False
    else:
        if attempt_boundary is not None:
            cleanup = finalize_attempt_boundary(
                process,
                attempt_boundary,
                reason="normal_exit",
                terminate=False,
            )
        else:
            cleanup["verification_performed"] = True
            cleanup["verified_absent"] = not _process_group_exists(process.pid)
            if not cleanup["verified_absent"]:
                cleanup = _terminate_process_group(
                    process, logger, reason="post_exit_lingering_group"
                )
    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)
    readers_completed = not stdout_thread.is_alive() and not stderr_thread.is_alive()
    if not readers_completed:
        capture_errors.append("stream readers did not reach EOF")
    return ApexProcessOutcome(
        exit_code=process.returncode,
        stdout=b"".join(stdout),
        stderr=b"".join(stderr),
        timed_out=timed_out,
        cleanup=cleanup,
        readers_completed=readers_completed,
        capture_errors=tuple(capture_errors),
    )


def _normalize_process_outcome(value: Any) -> ApexProcessOutcome:
    """Keep ordinary adapter unit fixtures terse; production always returns evidence."""
    if isinstance(value, ApexProcessOutcome):
        return value
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], int)
        and isinstance(value[1], str)
    ):
        return ApexProcessOutcome(
            exit_code=value[0],
            stdout=value[1].encode("utf-8"),
            stderr=b"",
            timed_out=False,
            cleanup={
                "required": False,
                "reason": "test_fixture",
                "pgid": None,
                "sigterm_sent": False,
                "sigkill_sent": False,
                "verification_performed": True,
                "verified_absent": True,
            },
            readers_completed=True,
            capture_errors=(),
        )
    raise ApexAdapterError("Apex process supervisor returned an invalid outcome")


def _read_regular_json(path: Path, *, size_limit: int, label: str) -> dict[str, Any]:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise ApexAdapterError(f"Apex did not write {label}: {path}") from error
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise ApexAdapterError(f"{label} must be a regular non-symlink file: {path}")
    if metadata.st_size > size_limit:
        raise ApexAdapterError(f"{label} exceeds {size_limit} bytes: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ApexAdapterError(f"{label} is not valid UTF-8 JSON: {path}") from error
    if not isinstance(value, dict):
        raise ApexAdapterError(f"{label} must contain a JSON object: {path}")
    return value


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


def _validate_instruction_adaptation(
    task_spec: dict[str, Any], original_prompt_bytes: bytes
) -> dict[str, Any]:
    """Recompute both sides of the caller-owned prompt adaptation receipt."""

    adaptation = task_spec.get("instruction_adaptation")
    if (
        not isinstance(adaptation, dict)
        or adaptation.get("schema") != _INSTRUCTION_ADAPTATION_SCHEMA
    ):
        raise ApexAdapterError("TaskSpec instruction_adaptation is missing or invalid")
    try:
        original_text = original_prompt_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ApexAdapterError("original Arena prompt is not valid UTF-8") from error
    instructions = task_spec.get("instructions")
    if not isinstance(instructions, str):
        raise ApexAdapterError("TaskSpec instructions must be text")
    adapted_bytes = instructions.encode("utf-8")
    expected_original = {
        "characters": len(original_text),
        "bytes": len(original_prompt_bytes),
        "sha256": _sha256_bytes(original_prompt_bytes),
    }
    expected_adapted = {
        "characters": len(instructions),
        "bytes": len(adapted_bytes),
        "sha256": _sha256_bytes(adapted_bytes),
    }
    if adaptation.get("original") != expected_original:
        raise ApexAdapterError(
            "instruction_adaptation original digest/size does not match Arena prompt"
        )
    if adaptation.get("adapted") != expected_adapted:
        raise ApexAdapterError(
            "instruction_adaptation adapted digest/size does not match TaskSpec"
        )
    return adaptation


def _regular_path_below(root: Path, raw: Any, *, label: str) -> Path:
    if not isinstance(raw, str) or not Path(raw).is_absolute():
        raise ApexAdapterError(f"{label} must be an absolute path")
    root = root.resolve(strict=True)
    try:
        path = Path(raw).resolve(strict=True)
        path.relative_to(root)
        metadata = path.lstat()
    except (OSError, ValueError) as error:
        raise ApexAdapterError(f"{label} is outside the Apex artifact root") from error
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise ApexAdapterError(f"{label} must be a regular non-symlink file")
    if metadata.st_nlink != 1:
        raise ApexAdapterError(f"{label} must not be hard-linked")
    return path


def _verify_artifact_receipt(
    *, artifact_store: Path, receipt: Any, label: str
) -> tuple[Path, bytes]:
    if not isinstance(receipt, dict):
        raise ApexAdapterError(f"{label} artifact receipt is missing")
    digest = receipt.get("digest")
    size = receipt.get("size")
    relative = receipt.get("relative_path")
    if (
        not isinstance(digest, str)
        or not _SHA256.fullmatch(digest)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or relative != f"sha256/{digest[:2]}/{digest}"
    ):
        raise ApexAdapterError(f"{label} artifact receipt is malformed")
    path = _regular_path_below(
        artifact_store,
        str(artifact_store / relative),
        label=f"{label} artifact",
    )
    content = path.read_bytes()
    if len(content) != size or _sha256_bytes(content) != digest:
        raise ApexAdapterError(f"{label} artifact receipt does not match stored bytes")
    return path, content


def _validate_apex_lineage(
    *, result: dict[str, Any], task_spec: dict[str, Any], artifact_root: Path
) -> dict[str, Any]:
    """Validate Apex's terminal result through its journal and transcript CAS."""
    campaign_binding = task_spec.get("campaign_binding")
    if (
        not isinstance(campaign_binding, dict)
        or set(campaign_binding) != _CAMPAIGN_BINDING_KEYS
        or campaign_binding.get("schema") != _CAMPAIGN_BINDING_SCHEMA
    ):
        raise ApexAdapterError("formal Apex lineage lacks the sealed campaign binding")
    status = result.get("status")
    terminal_contracts = {
        "candidate_ready": {
            "verdict": "keep",
            "agent_event": "agent_completed",
            "reason": None,
            "run_reason": "candidate_ready",
        },
        "no_gain": {
            "verdict": "revert",
            "agent_event": "agent_completed",
            "reason": None,
            "run_reason": None,
        },
        "budget_exhausted": {
            "verdict": "reject",
            "agent_event": "agent_failed",
            "reason": "agent_turn_budget_overrun",
            "run_reason": "agent_turn_budget_overrun",
        },
    }
    contract = terminal_contracts.get(status)
    if contract is None:
        raise ApexAdapterError(
            "formal Apex lineage supports candidate_ready, no_gain, or budget_exhausted"
        )
    expected_verdict = contract["verdict"]
    failure_reason = contract["reason"]
    if failure_reason is None:
        if result.get("error") is not None:
            raise ApexAdapterError("successful Apex terminal result must have error=null")
    else:
        error = result.get("error")
        if (
            result.get("reason_code") != failure_reason
            or not isinstance(error, dict)
            or error.get("reason_code") != failure_reason
            or result.get("bundle_path") is not None
            or result.get("bundle_digest") is not None
            or result.get("changed_files") != []
        ):
            raise ApexAdapterError(
                "budget_exhausted result does not carry the exact failure contract"
            )
    baseline = result.get("baseline_lock")
    if not isinstance(baseline, dict) or baseline.get("file_hashes") != task_spec["baseline"]["file_hashes"]:
        raise ApexAdapterError("Apex result baseline_lock does not match TaskSpec")
    if result.get("internal_verdict") != expected_verdict:
        raise ApexAdapterError(
            f"Apex {status} requires internal_verdict={expected_verdict}"
        )

    journal_ref = result.get("event_journal_ref")
    store_ref = result.get("artifact_store_ref")
    if not isinstance(journal_ref, dict) or not isinstance(store_ref, dict):
        raise ApexAdapterError("Apex result is missing journal/artifact lineage")
    journal = _regular_path_below(
        artifact_root, journal_ref.get("path"), label="event journal"
    )
    store_raw = store_ref.get("path")
    if not isinstance(store_raw, str) or not Path(store_raw).is_absolute():
        raise ApexAdapterError("artifact store reference must be absolute")
    store = Path(store_raw).resolve(strict=True)
    try:
        store.relative_to(artifact_root.resolve(strict=True))
        store_metadata = store.lstat()
    except (OSError, ValueError) as error:
        raise ApexAdapterError("artifact store is outside the Apex artifact root") from error
    if not stat.S_ISDIR(store_metadata.st_mode) or stat.S_ISLNK(store_metadata.st_mode):
        raise ApexAdapterError("artifact store must be a regular directory")

    uri = f"file:{journal.as_posix()}?mode=ro&immutable=1"
    try:
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        event_rows = connection.execute(
            "SELECT * FROM events WHERE run_id = ? ORDER BY sequence",
            (result.get("run_id"),),
        ).fetchall()
        transaction_rows = connection.execute(
            "SELECT * FROM transactions ORDER BY first_sequence"
        ).fetchall()
    except sqlite3.Error as error:
        raise ApexAdapterError(f"cannot read Apex event journal: {error}") from error
    finally:
        try:
            connection.close()
        except UnboundLocalError:
            pass
    if not event_rows:
        raise ApexAdapterError("Apex event journal has no events for result run_id")

    events: list[dict[str, Any]] = []
    transaction_events: dict[str, list[dict[str, Any]]] = {}
    previous_id: str | None = None
    previous_sequence = 0
    for row in event_rows:
        try:
            payload = json.loads(str(row["payload_json"]))
        except (json.JSONDecodeError, TypeError) as error:
            raise ApexAdapterError("Apex journal contains malformed payload JSON") from error
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
        checksum_material = {key: value for key, value in event.items() if key != "checksum"}
        if event["checksum"] != _canonical_json_digest(checksum_material):
            raise ApexAdapterError("Apex journal event checksum mismatch")
        if event["sequence"] <= previous_sequence or event["parent_event_id"] != previous_id:
            raise ApexAdapterError("Apex journal sequence/parent chain is invalid")
        previous_sequence = event["sequence"]
        previous_id = event["event_id"]
        events.append(event)
        transaction_events.setdefault(event["transaction_id"], []).append(event)

    transactions = {str(row["transaction_id"]): row for row in transaction_rows}
    if set(transactions) != set(transaction_events):
        raise ApexAdapterError("Apex journal transaction set is inconsistent")
    for transaction_id, tx_events in transaction_events.items():
        row = transactions[transaction_id]
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
            raise ApexAdapterError("Apex journal transaction checksum/bounds mismatch")

    head = events[-1]
    if (
        journal_ref.get("head_event_id") != head["event_id"]
        or journal_ref.get("head_checksum") != head["checksum"]
    ):
        raise ApexAdapterError("Apex result journal head does not match verified journal")
    expected_run_event = "run.failed" if failure_reason else "run.succeeded"
    expected_run_reason = contract["run_reason"] or result.get("reason_code")
    if (
        head["event_type"] != expected_run_event
        or head["payload"].get("reason") != expected_run_reason
    ):
        raise ApexAdapterError("Apex terminal run event disagrees with result status")
    verdict_ref = result.get("internal_verdict_ref")
    verdict_events = [event for event in events if event["event_id"] == verdict_ref]
    if len(verdict_events) != 1 or verdict_events[0]["event_type"] not in {
        "decision",
        "action.aborted",
        "action.failed",
    }:
        raise ApexAdapterError("Apex internal_verdict_ref is not a terminal decision event")
    verdict_payload = verdict_events[0]["payload"]
    if verdict_payload.get("verdict", expected_verdict) != expected_verdict:
        raise ApexAdapterError("Apex terminal decision disagrees with result verdict")
    if failure_reason and (
        verdict_events[0]["event_type"] != "decision"
        or verdict_payload.get("reason") != failure_reason
    ):
        raise ApexAdapterError("Apex failed decision does not bind the failure reason")

    completed = [event for event in events if event["event_type"] == "agent_completed"]
    failed = [event for event in events if event["event_type"] == "agent_failed"]
    expected_events = completed if contract["agent_event"] == "agent_completed" else failed
    unexpected_events = failed if contract["agent_event"] == "agent_completed" else completed
    if len(expected_events) != 1 or unexpected_events:
        raise ApexAdapterError(
            f"formal Apex {status} requires exactly one {contract['agent_event']} event"
        )
    agent_event = expected_events[0]
    payload = agent_event["payload"]
    expected_options = task_spec["agent_options"]
    common_agent_mismatch = (
        payload.get("backend") != task_spec["agent_backend"]
        or payload.get("model") != expected_options.get("model")
        or payload.get("effort") != expected_options.get("effort")
        or payload.get("timed_out") is not False
    )
    typed_termination = payload.get("termination_kind")
    if typed_termination is None:
        raise ApexAdapterError("formal Apex agent event lacks typed termination")
    exact_checkpoint = typed_termination == "exact_turn_boundary"
    budget_overrun = (
        failure_reason is not None and typed_termination == "turn_overrun"
    )
    process_containment = payload.get("process_containment")
    discarded_lines = payload.get("discarded_stdout_lines")
    discarded_bytes = payload.get("discarded_stdout_bytes")
    discarded_sha256 = payload.get("discarded_stdout_sha256")
    has_discarded_tail = (
        isinstance(discarded_lines, int)
        and not isinstance(discarded_lines, bool)
        and discarded_lines > 0
    ) or (
        isinstance(discarded_bytes, int)
        and not isinstance(discarded_bytes, bool)
        and discarded_bytes > 0
    )
    discarded_tail_invalid = (
        isinstance(discarded_lines, bool)
        or not isinstance(discarded_lines, int)
        or discarded_lines < 0
        or isinstance(discarded_bytes, bool)
        or not isinstance(discarded_bytes, int)
        or discarded_bytes < 0
        or (
            has_discarded_tail
            and (
                discarded_lines == 0
                or discarded_bytes == 0
                or not isinstance(discarded_sha256, str)
                or not _SHA256.fullmatch(discarded_sha256)
            )
        )
        or (not has_discarded_tail and discarded_sha256 is not None)
    )
    outcome_mismatch = (
        payload.get("capture_status") != "complete"
        or payload.get("timed_out") is not False
        or not isinstance(payload.get("observer_stop_sent"), bool)
        or payload.get("process_containment_policy_id")
        != AGENT_PROCESS_CONTAINMENT_POLICY
        or not _inner_agent_containment_valid(
            process_containment,
            forced_stop=exact_checkpoint or budget_overrun,
        )
        or discarded_tail_invalid
        or (
            failure_reason is None
            and (
                typed_termination not in {"completed", "exact_turn_boundary"}
                or payload.get("candidate_capture_allowed") is not True
            )
        )
        or (
            failure_reason is not None
            and (
                not budget_overrun
                or payload.get("candidate_capture_allowed") is not False
                or payload.get("termination_reason") != "max_turns_overrun"
                or type(payload.get("observed_turns")) is not int
                or payload["observed_turns"]
                <= task_spec["budget"]["max_turns"]
                or payload.get("observer_stop_sent") is not True
                or type(payload.get("exit_code")) is not int
                or payload["exit_code"] != 128 + signal.SIGKILL
            )
        )
        or (
            failure_reason is None
            and exact_checkpoint
            and (
                payload.get("termination_reason")
                != "max_turns_exact_boundary"
                or payload.get("observed_turns")
                != task_spec["budget"]["max_turns"]
                or payload.get("observer_stop_sent") is not True
                or payload.get("exit_code") != 128 + signal.SIGKILL
            )
        )
        or (
            failure_reason is None
            and not exact_checkpoint
            and (
                payload.get("termination_reason") is not None
                or payload.get("exit_code") != 0
            )
        )
    )
    if common_agent_mismatch or outcome_mismatch:
        raise ApexAdapterError(
            f"Apex {contract['agent_event']} outcome/identity is inconsistent"
        )
    invocation = payload.get("invocation")
    expected_invocation_schema = "apex.agent-invocation/v3"
    if (
        not isinstance(invocation, dict)
        or invocation.get("schema") != expected_invocation_schema
    ):
        raise ApexAdapterError("Apex agent invocation receipt is missing")
    expected_isolation = {
        "approval": "never_via_strict_config",
        "execpolicy_rules": "ignored",
        "project_instructions": "backend_default_may_load",
        "response_token_limit": "not_supported_context_advisory_only",
        "sandbox": "workspace-write",
        "session": "ephemeral",
        "user_config": "ignored",
    }
    argv = invocation.get("argv")
    if (
        invocation.get("cli_name") != "codex"
        or not isinstance(invocation.get("cli_version"), str)
        or not invocation["cli_version"].strip()
        or not isinstance(invocation.get("entrypoint_sha256"), str)
        or not _SHA256.fullmatch(invocation["entrypoint_sha256"])
        or invocation.get("runtime_closure_sha256")
        != expected_options.get("runtime_closure_sha256")
        or invocation.get("max_turns") != task_spec["budget"]["max_turns"]
        or invocation.get("turn_policy") != TURN_POLICY
        or invocation.get("process_containment_policy_id")
        != AGENT_PROCESS_CONTAINMENT_POLICY
        or invocation.get("isolation") != expected_isolation
        or not isinstance(argv, list)
        or "--strict-config" not in argv
        or "--ignore-user-config" not in argv
        or "--ignore-rules" not in argv
        or "--ephemeral" not in argv
    ):
        raise ApexAdapterError("Apex Codex invocation does not meet the campaign contract")
    executed = Path(str(invocation.get("resolved_executable_path", "")))
    if not executed.is_absolute() or not executed.is_file() or _sha256_file(executed) != invocation["entrypoint_sha256"]:
        raise ApexAdapterError("Apex Codex entrypoint identity is not reproducible")

    prompt_events = [event for event in events if event["event_type"] == "prompt_sent"]
    if len(prompt_events) != 1:
        raise ApexAdapterError("formal Apex attempt requires exactly one prompt_sent event")
    prompt_event = prompt_events[0]
    prompt_bindings = [
        item
        for item in prompt_event["payload"].get("artifacts", [])
        if isinstance(item, dict) and item.get("role") == "prompt"
    ]
    if (
        len(prompt_bindings) != 1
        or prompt_event["sequence"] >= agent_event["sequence"]
        or agent_event["parent_event_id"] != prompt_event["event_id"]
        or prompt_event["payload"].get("attempt_id")
        != agent_event["payload"].get("attempt_id")
    ):
        raise ApexAdapterError("Apex prompt event is not uniquely bound to agent invocation")
    prompt_path, prompt_bytes = _verify_artifact_receipt(
        artifact_store=store,
        receipt=prompt_bindings[0].get("receipt"),
        label="agent prompt",
    )
    if not context_packet_objective_matches(
        prompt_bytes, task_spec.get("instructions")
    ):
        raise ApexAdapterError(
            "Apex event-bound prompt does not bind TaskSpec instructions as "
            "ContextPacket role.objective"
        )

    bindings = payload.get("artifacts")
    if not isinstance(bindings, list):
        raise ApexAdapterError("Apex agent_completed artifacts are missing")
    transcript_bindings = [
        item
        for item in bindings
        if isinstance(item, dict) and item.get("role") == "agent_transcript"
    ]
    if len(transcript_bindings) != 1:
        raise ApexAdapterError("Apex agent transcript binding is not unique")
    transcript_path, transcript_bytes = _verify_artifact_receipt(
        artifact_store=store,
        receipt=transcript_bindings[0].get("receipt"),
        label="agent transcript",
    )
    try:
        transcript = json.loads(transcript_bytes)
    except json.JSONDecodeError as error:
        raise ApexAdapterError("Apex agent transcript is not valid JSON") from error
    if (
        not isinstance(transcript, dict)
        or transcript.get("schema") != "apex.agent-transcript/v3"
        or transcript.get("backend") != task_spec["agent_backend"]
        or transcript.get("model") != expected_options.get("model")
        or transcript.get("effort") != expected_options.get("effort")
        or transcript.get("invocation") != invocation
    ):
        raise ApexAdapterError("Apex transcript does not bind the verified invocation")
    semantic_events = transcript.get("semantic_events")
    budget = transcript.get("termination")
    if not isinstance(semantic_events, list) or not isinstance(budget, dict):
        raise ApexAdapterError("Apex transcript lacks semantic turn evidence")
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
    max_turns = task_spec["budget"]["max_turns"]
    expected_budget_exceeded = failure_reason is not None
    common_turn_mismatch = (
        payload.get("observed_turns") != observed_turns
        or budget.get("turn_policy") != TURN_POLICY
        or budget.get("max_turns") != max_turns
        or budget.get("observed_turns") != observed_turns
    )
    if transcript.get("schema") == "apex.agent-transcript/v3":
        turn_mismatch = common_turn_mismatch or any(
            budget.get(transcript_key) != payload.get(payload_key)
            for transcript_key, payload_key in {
                "kind": "termination_kind",
                "reason": "termination_reason",
                "capture_status": "capture_status",
                "candidate_capture_allowed": "candidate_capture_allowed",
                "observer_stop_sent": "observer_stop_sent",
            }.items()
        )
        discarded_tail = budget.get("discarded_stdout_tail")
        turn_mismatch = turn_mismatch or (
            budget.get("process_containment")
            != payload.get("process_containment")
            or not isinstance(discarded_tail, dict)
            or discarded_tail
            != {
                "lines": payload.get("discarded_stdout_lines"),
                "bytes": payload.get("discarded_stdout_bytes"),
                "sha256": payload.get("discarded_stdout_sha256"),
            }
        )
        turn_mismatch = (
            turn_mismatch
            or (exact_checkpoint and observed_turns != max_turns)
            or (budget_overrun and observed_turns <= max_turns)
            or (
                not exact_checkpoint
                and not budget_overrun
                and not 1 <= observed_turns <= max_turns
            )
        )
    else:
        turn_mismatch = (
            common_turn_mismatch
            or payload.get("message_event_count") != message_count
            or payload.get("tool_call_event_count") != tool_call_count
            or payload.get("semantic_event_count") != len(semantic_events)
            or budget.get("exceeded") is not expected_budget_exceeded
            or budget.get("enforcement_failed") is not False
            or budget.get("reason") != payload.get("budget_reason")
            or (
                not expected_budget_exceeded
                and (
                    type(max_turns) is not int
                    or not 1 <= observed_turns <= max_turns
                )
            )
        )
    if turn_mismatch:
        raise ApexAdapterError("Apex transcript turn evidence is inconsistent")

    declared_receipts = store_ref.get("receipt_digests")
    if not isinstance(declared_receipts, list) or any(
        not isinstance(value, str) or not _SHA256.fullmatch(value)
        for value in declared_receipts
    ):
        raise ApexAdapterError("Apex result artifact receipt set is malformed")
    for digest in declared_receipts:
        artifact = store / "sha256" / digest[:2] / digest
        _regular_path_below(store, str(artifact), label="declared result artifact")
        if _sha256_file(artifact) != digest:
            raise ApexAdapterError("Apex result artifact digest mismatch")

    event_artifact_digests: set[str] = set()
    for event in events:
        raw_bindings = event["payload"].get("artifacts", [])
        if not isinstance(raw_bindings, list):
            raise ApexAdapterError("Apex event artifact bindings are malformed")
        for index, binding in enumerate(raw_bindings):
            if not isinstance(binding, dict):
                raise ApexAdapterError("Apex event artifact binding is malformed")
            _, _ = _verify_artifact_receipt(
                artifact_store=store,
                receipt=binding.get("receipt"),
                label=f"event {event['event_id']} artifact {index}",
            )
            event_artifact_digests.add(binding["receipt"]["digest"])
    if len(declared_receipts) != len(set(declared_receipts)) or not set(
        declared_receipts
    ).issubset(event_artifact_digests):
        raise ApexAdapterError(
            "Apex result artifact receipt set is not event-bound"
        )

    checkpoint_gate_chain = None
    if exact_checkpoint and status == "candidate_ready":
        checkpoint_gate_chain = _validate_checkpoint_gate_chain(
            events=events,
            agent_event=agent_event,
            result=result,
        )

    return {
        "journal_path": journal,
        "journal_head_event_id": head["event_id"],
        "journal_head_checksum": head["checksum"],
        "event_count": len(events),
        "transcript_path": transcript_path,
        "transcript_bytes": transcript_bytes,
        "transcript_digest": _sha256_bytes(transcript_bytes),
        "event_artifact_digests": sorted(event_artifact_digests),
        "invocation": invocation,
        "termination_kind": typed_termination,
        "termination_reason": payload.get("termination_reason"),
        "capture_status": payload.get("capture_status"),
        "candidate_capture_allowed": payload.get("candidate_capture_allowed"),
        "observed_turns": observed_turns,
        "observer_stop_sent": payload.get("observer_stop_sent"),
        "process_containment": payload.get("process_containment"),
        "discarded_stdout_tail": {
            "lines": payload.get("discarded_stdout_lines"),
            "bytes": payload.get("discarded_stdout_bytes"),
            "sha256": payload.get("discarded_stdout_sha256"),
        },
        "agent_event_id": agent_event["event_id"],
        "checkpoint_gate_chain": checkpoint_gate_chain,
        "prompt_bytes": prompt_bytes,
        "prompt_event": {
            "binding": "apex.prompt_sent_event_cas/v1",
            "event_id": prompt_event["event_id"],
            "sha256": _sha256_bytes(prompt_bytes),
            "size_bytes": len(prompt_bytes),
            "artifact_path": str(prompt_path),
            "stdin_transport_attested": False,
        },
        "codex": {
            "binary_sha256": invocation["entrypoint_sha256"],
            "runtime_closure_sha256": invocation["runtime_closure_sha256"],
            "version": invocation["cli_version"],
            "model": transcript["model"],
            "effort": transcript["effort"],
        },
    }


def _validate_checkpoint_gate_chain(
    *,
    events: list[dict[str, Any]],
    agent_event: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, str]:
    """Require the evaluator-owned freeze/gate chain after an exact boundary."""
    attempt_id = agent_event["payload"].get("attempt_id")
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
    for event_type in required_types:
        matches = [
            event
            for event in events
            if event["event_type"] == event_type
            and (
                event_type == "run.succeeded"
                or event["payload"].get("attempt_id", event["payload"].get("action_id"))
                == attempt_id
            )
        ]
        if len(matches) != 1:
            raise ApexAdapterError(
                f"exact-boundary checkpoint requires one {event_type} event"
            )
        selected.append(matches[0])
    if not (
        agent_event["sequence"] < selected[0]["sequence"]
        and [event["sequence"] for event in selected]
        == sorted(event["sequence"] for event in selected)
    ):
        raise ApexAdapterError("exact-boundary trusted gate events are out of order")

    candidate, artifacts_ready, compiled, correct, safety, performance, measured, reward, verified, decision, finished = selected
    candidate_bindings = candidate["payload"].get("artifacts")
    changed_files = result.get("changed_files")
    if (
        not isinstance(changed_files, list)
        or not changed_files
        or candidate["payload"].get("changed_files") != changed_files
        or not isinstance(candidate_bindings, list)
        or not candidate_bindings
        or any(
            not isinstance(binding, dict) or binding.get("role") != "candidate"
            for binding in candidate_bindings
        )
        or not isinstance(artifacts_ready["payload"].get("artifact_refs"), list)
        or not artifacts_ready["payload"]["artifact_refs"]
    ):
        raise ApexAdapterError("exact-boundary frozen candidate lineage is invalid")
    if compiled["payload"].get("passed") is not True:
        raise ApexAdapterError("exact-boundary compile gate did not pass")
    if correct["payload"].get("passed") is not True:
        raise ApexAdapterError("exact-boundary correctness gate did not pass")
    if (
        safety["payload"].get("allowed_to_measure") is not True
        or safety["payload"].get("promotion_eligible") is not True
    ):
        raise ApexAdapterError("exact-boundary safety gate did not permit promotion")
    if (
        performance["payload"].get("passed") is not True
        or performance["payload"].get("runtime") != "normal_uninstrumented"
        or performance["payload"].get("status")
        != "command_completed_without_robust_timing_grade"
    ):
        raise ApexAdapterError("exact-boundary normal-performance gate did not pass")
    if (
        measured["payload"].get("measurement_status") != "valid"
        or measured["payload"].get("evidence_class") != "measured"
        or reward["payload"].get("evidence_class") != "measured"
        or not isinstance(reward["payload"].get("scalar_reward"), (int, float))
    ):
        raise ApexAdapterError("exact-boundary measurement/reward chain is invalid")
    if (
        not isinstance(verified["payload"].get("verification_id"), str)
        or not _SHA256.fullmatch(verified["payload"]["verification_id"])
        or decision["event_id"] != result.get("internal_verdict_ref")
        or decision["payload"].get("verdict") != "keep"
        or decision["payload"].get("bundle_digest") != result.get("bundle_digest")
        or finished["payload"].get("reason") != "candidate_ready"
    ):
        raise ApexAdapterError("exact-boundary decision/bundle lineage is invalid")
    return {event["event_type"]: event["event_id"] for event in selected}


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_read_only_atomic(path: Path, payload: bytes) -> dict[str, Any]:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        os.link(temporary, path, follow_symlinks=False)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_bytes(payload),
        "size_bytes": len(payload),
        "mode": "0444",
    }


def _write_apex_attempt_receipt(
    *,
    receipt_path: Path,
    task_spec_bytes: bytes,
    original_prompt_bytes: bytes,
    result_path: Path,
    outcome: ApexProcessOutcome,
    receipt: dict[str, Any],
    lineage: dict[str, Any] | None,
) -> None:
    try:
        task_spec = json.loads(
            task_spec_bytes,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ApexAdapterError("sealed TaskSpec bytes are not valid JSON") from error
    if not isinstance(task_spec, dict):
        raise ApexAdapterError("sealed TaskSpec must contain an object")
    adaptation = _validate_instruction_adaptation(task_spec, original_prompt_bytes)
    if receipt_path.exists():
        raise ApexAdapterError(f"Apex attempt receipt already exists: {receipt_path}")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir = receipt_path.parent / f".{receipt_path.stem}.artifacts"
    try:
        artifact_dir.mkdir(mode=0o700)
    except FileExistsError as error:
        raise ApexAdapterError(f"Apex receipt artifact directory exists: {artifact_dir}") from error
    artifacts = {
        "task_spec": _write_read_only_atomic(
            artifact_dir / "task_spec.json", task_spec_bytes
        ),
        "original_arena_prompt": _write_read_only_atomic(
            artifact_dir / "original_arena_prompt.txt", original_prompt_bytes
        ),
        "apex_stdout": _write_read_only_atomic(
            artifact_dir / "apex_stdout.txt", outcome.stdout
        ),
        "apex_stderr": _write_read_only_atomic(
            artifact_dir / "apex_stderr.txt", outcome.stderr
        ),
    }
    if result_path.is_file():
        artifacts["apex_result"] = _write_read_only_atomic(
            artifact_dir / "apex_result.json", result_path.read_bytes()
        )
    if lineage is not None:
        artifacts["event_journal"] = _write_read_only_atomic(
            artifact_dir / "event_journal.sqlite",
            Path(lineage["journal_path"]).read_bytes(),
        )
        artifacts["agent_transcript"] = _write_read_only_atomic(
            artifact_dir / "agent_transcript.json", lineage["transcript_bytes"]
        )
        artifacts["agent_prompt"] = _write_read_only_atomic(
            artifact_dir / "agent_prompt.txt", lineage["prompt_bytes"]
        )
        if isinstance(lineage.get("bundle_snapshot_bytes"), bytes):
            artifacts["source_bundle"] = _write_read_only_atomic(
                artifact_dir / "source_bundle_snapshot.json",
                lineage["bundle_snapshot_bytes"],
            )
    original_artifact = artifacts["original_arena_prompt"]
    if (
        original_artifact["sha256"] != adaptation["original"]["sha256"]
        or original_artifact["size_bytes"] != adaptation["original"]["bytes"]
    ):
        raise ApexAdapterError(
            "sealed original Arena prompt artifact disagrees with adaptation receipt"
        )
    if lineage is not None:
        prompt_artifact = artifacts["agent_prompt"]
        prompt_event = lineage["prompt_event"]
        if (
            prompt_artifact["sha256"] != prompt_event["sha256"]
            or prompt_artifact["size_bytes"] != prompt_event["size_bytes"]
        ):
            raise ApexAdapterError(
                "sealed agent prompt artifact disagrees with prompt event receipt"
            )
        bundle_summary = lineage.get("bundle")
        if isinstance(bundle_summary, dict):
            bundle_artifact = artifacts.get("source_bundle")
            if (
                not isinstance(bundle_artifact, dict)
                or bundle_artifact["sha256"]
                != bundle_summary.get("snapshot_sha256")
                or bundle_artifact["size_bytes"]
                != bundle_summary.get("snapshot_size_bytes")
            ):
                raise ApexAdapterError(
                    "sealed source bundle disagrees with checkpoint lineage"
                )
    receipt["artifacts"] = artifacts
    payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _write_read_only_atomic(receipt_path, payload)
    artifact_dir.chmod(0o555)
    _fsync_directory(artifact_dir.parent)


def _sealed_task_spec_matches(path: Path, expected: bytes) -> bool:
    """Recheck the exact formal request after the untrusted subprocess exits."""
    try:
        metadata = path.lstat()
        parent_metadata = path.parent.lstat()
        return (
            stat.S_ISREG(metadata.st_mode)
            and not path.is_symlink()
            and metadata.st_nlink == 1
            and stat.S_IMODE(metadata.st_mode) == 0o444
            and stat.S_ISDIR(parent_metadata.st_mode)
            and not path.parent.is_symlink()
            and stat.S_IMODE(parent_metadata.st_mode) == 0o555
            and path.read_bytes() == expected
        )
    except OSError:
        return False


def _bundle_files(root: Path, *, size_limit: int) -> set[str]:
    total_size = 0
    files: set[str] = set()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ApexAdapterError(f"bundle contains a symlink: {relative}")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ApexAdapterError(f"bundle contains a non-regular or hard-linked file: {relative}")
        total_size += metadata.st_size
        if total_size > size_limit:
            raise ApexAdapterError(f"bundle exceeds {size_limit} bytes")
        files.add(relative)
    return files


def _bundle_digest(manifest: dict[str, Any], patch_paths: Iterable[Path]) -> str:
    """Hash canonical ``bundle.json`` data followed by patch bytes in manifest order."""
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    )
    for patch_path in patch_paths:
        with patch_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _bundle_snapshot_bytes(
    *, result: dict[str, Any], artifact_root: Path, size_limit: int
) -> bytes:
    """Serialize validated bundle bytes for immutable outer-receipt verification."""
    bundle_path = _resolve_below(
        artifact_root, result.get("bundle_path"), field="bundle_path"
    )
    files = _bundle_files(bundle_path, size_limit=size_limit)
    manifest = _read_regular_json(
        bundle_path / "bundle.json",
        size_limit=_DEFAULT_RESULT_LIMIT,
        label="bundle manifest",
    )
    raw_patches = manifest.get("patches")
    if not isinstance(raw_patches, list) or not raw_patches:
        raise ApexAdapterError("bundle snapshot requires declared patches")
    patches: list[dict[str, Any]] = []
    patch_paths: list[Path] = []
    for index, entry in enumerate(raw_patches):
        if not isinstance(entry, dict):
            raise ApexAdapterError(f"bundle.patches[{index}] must be an object")
        relative = _normalize_relative_path(
            entry.get("path"), field=f"bundle.patches[{index}].path"
        )
        path = _resolve_below(bundle_path, relative, field="patch path")
        content = path.read_bytes()
        patch_paths.append(path)
        patches.append(
            {
                "path": relative,
                "sha256": _sha256_bytes(content),
                "size_bytes": len(content),
                "content_base64": base64.b64encode(content).decode("ascii"),
            }
        )
    if files != {"bundle.json", *(item["path"] for item in patches)}:
        raise ApexAdapterError("bundle snapshot file set disagrees with manifest")
    bundle_digest = _bundle_digest(manifest, patch_paths)
    if bundle_digest != result.get("bundle_digest"):
        raise ApexAdapterError("bundle snapshot digest disagrees with Apex result")
    snapshot = {
        "schema": "aka.apex-source-bundle-snapshot/v1",
        "bundle_digest": bundle_digest,
        "manifest": manifest,
        "patches": patches,
    }
    return json.dumps(
        snapshot,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _resolve_below(root: Path, raw: Any, *, field: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ApexAdapterError(f"{field} must be a non-empty path string")
    resolved_root = root.resolve()
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = resolved_root / candidate
    candidate = Path(os.path.abspath(candidate))
    if not candidate.is_relative_to(resolved_root):
        raise ApexAdapterError(f"{field} escapes the Apex artifact root: {raw!r}")
    relative = candidate.relative_to(resolved_root)
    current = resolved_root
    for part in relative.parts:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ApexAdapterError(f"{field} traverses a symlink: {raw!r}")
    return candidate


def _validate_patch_text(patch_path: Path) -> None:
    forbidden_prefixes = (
        "new file mode ",
        "deleted file mode ",
        "old mode ",
        "new mode ",
        "rename from ",
        "rename to ",
        "copy from ",
        "copy to ",
    )
    try:
        lines = patch_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ApexAdapterError(f"patch must be UTF-8 text: {patch_path}") from error
    for line in lines:
        if line.startswith(forbidden_prefixes):
            raise ApexAdapterError(f"patch contains unsupported file operation: {line}")
        if line in {"--- /dev/null", "+++ /dev/null"}:
            raise ApexAdapterError("patch may not create or delete files")


def _git_patch_changed_files(workspace: Path, patch_paths: list[Path]) -> set[str]:
    """Ask Git's patch parser which files would change, using NUL-safe output."""
    parsed = subprocess.run(
        ["git", "-C", str(workspace), "apply", "--numstat", "-z", *map(str, patch_paths)],
        capture_output=True,
        check=False,
    )
    if parsed.returncode != 0:
        detail = parsed.stderr.decode("utf-8", "replace")[-4000:]
        raise ApexAdapterError(f"git apply --numstat failed: {detail}")
    changed: set[str] = set()
    for record in parsed.stdout.split(b"\0"):
        if not record:
            continue
        fields = record.split(b"\t", 2)
        if len(fields) != 3:
            raise ApexAdapterError("git apply --numstat returned malformed output")
        try:
            raw_path = fields[2].decode("utf-8")
        except UnicodeDecodeError as error:
            raise ApexAdapterError("patch target path must be UTF-8") from error
        changed.add(_normalize_relative_path(raw_path, field="patch target"))
    if not changed:
        raise ApexAdapterError("bundle patches contain no source changes")
    return changed


def _contract_path_list(raw: Any, *, field: str, required: bool = True) -> list[str]:
    if not isinstance(raw, list) or any(not isinstance(value, str) for value in raw):
        raise ApexAdapterError(f"{field} must be a list of path strings")
    normalized = [_normalize_relative_path(value, field=field) for value in raw]
    if len(normalized) != len(set(normalized)):
        raise ApexAdapterError(f"{field} contains duplicate paths")
    if required and not normalized:
        raise ApexAdapterError(f"{field} must not be empty")
    return normalized


def _validate_and_apply_bundle(
    *,
    result: dict[str, Any],
    task_spec: dict[str, Any],
    workspace: Path,
    artifact_root: Path,
    max_result_bytes: int,
    max_bundle_bytes: int,
) -> list[str]:
    raw_bundle_path = result.get("bundle_path")
    raw_bundle_digest = result.get("bundle_digest")
    bundle_path = _resolve_below(artifact_root, raw_bundle_path, field="bundle_path")
    if not isinstance(raw_bundle_digest, str) or not _SHA256.fullmatch(raw_bundle_digest):
        raise ApexAdapterError("bundle_digest must be a lowercase 64-hex SHA-256 digest")
    if not bundle_path.is_dir() or bundle_path.is_symlink():
        raise ApexAdapterError("bundle_path must name a regular directory")
    bundle_files = _bundle_files(bundle_path, size_limit=max_bundle_bytes)

    manifest = _read_regular_json(
        bundle_path / "bundle.json",
        size_limit=max_result_bytes,
        label="bundle manifest",
    )
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != _SCHEMA_VERSION:
        raise ApexAdapterError("unsupported bundle schema_version")
    if manifest.get("task_id") != task_spec["task_id"]:
        raise ApexAdapterError("bundle task_id does not match request")

    expected_baseline = task_spec["baseline"]["file_hashes"]
    manifest_baseline = manifest.get("baseline")
    if not isinstance(manifest_baseline, dict) or manifest_baseline.get("file_hashes") != expected_baseline:
        raise ApexAdapterError("bundle baseline file hashes do not match the frozen request")
    for relative, expected_digest in expected_baseline.items():
        current_digest = _sha256_file(_ensure_regular_workspace_file(workspace, relative))
        if current_digest != expected_digest:
            raise ApexAdapterError(
                f"workspace baseline changed before bundle apply: {relative}"
            )

    editable = set(task_spec["editable_files"])
    result_changed = set(
        _contract_path_list(result.get("changed_files"), field="result.changed_files")
    )
    manifest_changed = set(
        _contract_path_list(manifest.get("changed_files"), field="bundle.changed_files")
    )
    if result_changed != manifest_changed:
        raise ApexAdapterError("result and bundle changed_files disagree")
    if not manifest_changed.issubset(editable):
        raise ApexAdapterError(
            f"bundle changes undeclared files: {sorted(manifest_changed - editable)}"
        )

    raw_patches = manifest.get("patches")
    if not isinstance(raw_patches, list) or not raw_patches:
        raise ApexAdapterError("bundle.patches must be a non-empty list")
    patch_paths: list[Path] = []
    patch_relatives: set[str] = set()
    for index, entry in enumerate(raw_patches):
        if not isinstance(entry, dict):
            raise ApexAdapterError(f"bundle.patches[{index}] must be an object")
        patch_relative = _normalize_relative_path(
            entry.get("path"), field=f"bundle.patches[{index}].path"
        )
        if patch_relative in patch_relatives:
            raise ApexAdapterError(f"bundle contains duplicate patch path: {patch_relative}")
        patch_path = _resolve_below(bundle_path, patch_relative, field="patch path")
        if not patch_path.is_file() or patch_path.is_symlink():
            raise ApexAdapterError(f"patch is not a regular file: {patch_path}")
        expected_digest = entry.get("sha256")
        if not isinstance(expected_digest, str) or not _SHA256.fullmatch(expected_digest):
            raise ApexAdapterError(f"patch digest is not lowercase 64-hex: {patch_path.name}")
        if expected_digest != _sha256_file(patch_path):
            raise ApexAdapterError(f"patch digest mismatch: {patch_path.name}")
        _validate_patch_text(patch_path)
        patch_paths.append(patch_path)
        patch_relatives.add(patch_relative)
    expected_bundle_files = {"bundle.json", *patch_relatives}
    if bundle_files != expected_bundle_files:
        raise ApexAdapterError(
            "bundle contains undeclared files: "
            f"extra={sorted(bundle_files - expected_bundle_files)} "
            f"missing={sorted(expected_bundle_files - bundle_files)}"
        )
    actual_bundle_digest = _bundle_digest(manifest, patch_paths)
    if actual_bundle_digest != raw_bundle_digest:
        raise ApexAdapterError(
            f"bundle digest mismatch: expected={raw_bundle_digest} actual={actual_bundle_digest}"
        )
    patch_changed = _git_patch_changed_files(workspace, patch_paths)
    if patch_changed != manifest_changed:
        raise ApexAdapterError(
            "Git-parsed patch targets do not match declared changed_files: "
            f"patch={sorted(patch_changed)} declared={sorted(manifest_changed)}"
        )
    candidate_hashes = manifest.get("candidate_file_hashes")
    if not isinstance(candidate_hashes, dict) or set(candidate_hashes) != manifest_changed:
        raise ApexAdapterError("bundle candidate_file_hashes do not match changed_files")
    if any(
        not isinstance(value, str) or not _SHA256.fullmatch(value)
        for value in candidate_hashes.values()
    ):
        raise ApexAdapterError("bundle candidate_file_hashes must be lowercase 64-hex digests")
    if manifest.get("delivery") != {"mode": "bundle", "applied": False}:
        raise ApexAdapterError("bundle delivery contract must be mode=bundle, applied=false")

    command = ["git", "-C", str(workspace), "apply", "--check", *map(str, patch_paths)]
    checked = subprocess.run(command, capture_output=True, text=True, check=False)
    if checked.returncode != 0:
        raise ApexAdapterError(f"git apply --check failed: {checked.stderr[-4000:]}")
    backups = {
        relative: (workspace / relative).read_bytes() for relative in manifest_changed
    }
    applied = subprocess.run(
        ["git", "-C", str(workspace), "apply", *map(str, patch_paths)],
        capture_output=True,
        text=True,
        check=False,
    )
    if applied.returncode != 0:
        for relative, content in backups.items():
            (workspace / relative).write_bytes(content)
        raise ApexAdapterError(f"git apply failed: {applied.stderr[-4000:]}")

    for relative in manifest_changed:
        path = _ensure_regular_workspace_file(workspace, relative)
        applied_digest = _sha256_file(path)
        if applied_digest != candidate_hashes[relative]:
            for restore_path, content in backups.items():
                (workspace / restore_path).write_bytes(content)
            raise ApexAdapterError(f"applied source hash does not match bundle: {relative}")
        if applied_digest == expected_baseline[relative]:
            for restore_path, content in backups.items():
                (workspace / restore_path).write_bytes(content)
            raise ApexAdapterError(f"bundle did not change declared file: {relative}")
    return sorted(manifest_changed)


def _resolve_apex_command(agent_config: dict[str, Any]) -> tuple[Path, str]:
    root_value = os.environ.get("APEX_ROOT") or agent_config.get("apex_root")
    if not isinstance(root_value, str) or not root_value.strip():
        raise ApexAdapterError(
            "Apex checkout is not configured; set APEX_ROOT or agents/apex/agent_config.yaml apex_root"
        )
    apex_root = Path(root_value).expanduser().resolve()
    entrypoint = apex_root / "main.py"
    if not entrypoint.is_file():
        raise ApexAdapterError(f"Apex entrypoint not found: {entrypoint}")
    bootstrapped_python = apex_root / ".venv" / "bin" / "python"
    python_value = (
        os.environ.get("APEX_PYTHON")
        or agent_config.get("python_path")
        or (str(bootstrapped_python) if bootstrapped_python.is_file() else None)
        or sys.executable
    )
    python_path = str(Path(str(python_value)).expanduser())
    return entrypoint, python_path


def _formal_runtime_plan(entrypoint: Path, python_path: str) -> RuntimePlan:
    raw_root = os.environ.get("APEX_ROOT")
    source_root = os.environ.get("AGENT_KERNEL_ARENA_APEX_SOURCE_ROOT")
    configured_python = os.environ.get("APEX_PYTHON")
    raw_snapshot = os.environ.get(
        "AGENT_KERNEL_ARENA_APEX_RUNTIME_SNAPSHOT_ROOT"
    )
    expected_digest = os.environ.get(
        "AGENT_KERNEL_ARENA_APEX_RUNTIME_MANIFEST_SHA256"
    )
    if (
        not isinstance(raw_root, str)
        or not isinstance(source_root, str)
        or not Path(source_root).is_absolute()
        or not isinstance(configured_python, str)
        or configured_python != python_path
        or not isinstance(raw_snapshot, str)
        or not isinstance(expected_digest, str)
        or _SHA256.fullmatch(expected_digest) is None
    ):
        raise ApexAdapterError("formal Apex runtime contract is incomplete")
    execution_root = Path(os.path.abspath(Path(raw_root).expanduser()))
    snapshot = Path(os.path.abspath(Path(raw_snapshot).expanduser()))
    if (
        entrypoint != execution_root / "main.py"
        or execution_root != snapshot / "repo"
        or Path(configured_python) != snapshot / "sealed-bin" / "python"
    ):
        raise ApexAdapterError("formal Apex entrypoint differs from APEX_ROOT")
    try:
        manifest = verify_runtime_snapshot(snapshot, expected_digest)
    except ApexRuntimeError as error:
        raise ApexAdapterError(f"formal Apex runtime is invalid: {error}") from error
    roots = manifest.get("roots")
    apex_source = (
        roots[0].get("source")
        if isinstance(roots, list) and roots and isinstance(roots[0], dict)
        else None
    )
    if (
        not isinstance(apex_source, dict)
        or apex_source.get("path") != source_root
    ):
        raise ApexAdapterError("formal Apex snapshot source differs from APEX_ROOT")
    repository = manifest.get("git")
    expected_dirty = os.environ.get("AGENT_KERNEL_ARENA_APEX_DIRTY")
    expected_repository = {
        "commit": os.environ.get("AGENT_KERNEL_ARENA_APEX_COMMIT"),
        "dirty": expected_dirty == "true" if expected_dirty in {"true", "false"} else None,
        "status_sha256": os.environ.get("AGENT_KERNEL_ARENA_APEX_STATUS_SHA256"),
    }
    actual_repository = (
        {
            "commit": repository.get("commit"),
            "dirty": repository.get("dirty"),
            "status_sha256": repository.get("status_sha256"),
        }
        if isinstance(repository, dict)
        else None
    )
    if manifest.get("sha256") != expected_digest or actual_repository != expected_repository:
        raise ApexAdapterError(
            "formal Apex runtime differs from the comparison contract"
        )
    launcher = manifest.get("launcher")
    system_python = (
        launcher.get("system_python", {}).get("path")
        if isinstance(launcher, dict)
        else None
    )
    if not isinstance(system_python, str) or not Path(system_python).is_absolute():
        raise ApexAdapterError("formal Apex system interpreter contract is invalid")
    return RuntimePlan(
        manifest=manifest,
        roots=(),
        system_python=Path(system_python),
    )


def _runtime_snapshot_receipt(
    *,
    plan: RuntimePlan,
    snapshot: Path,
    immutable_mount: dict[str, Any],
    attempt_mounts_sha256: str | None = None,
) -> dict[str, Any]:
    try:
        manifest = verify_runtime_snapshot(snapshot, plan.sha256)
        environment = runtime_environment(snapshot, manifest)
        execution = manifest["execution"]
        entrypoint = snapshot / execution["entrypoint"]
        launcher = snapshot / execution["interpreter"]
        underlying = snapshot / execution["underlying_interpreter"]
        image_inputs = runtime_image_inputs(snapshot, manifest)
        validate_immutable_mount_receipt(snapshot, manifest, immutable_mount)
        repository = manifest["git"]
        material: dict[str, Any] = {
            "schema": APEX_RUNTIME_MOUNT_SCHEMA,
            "policy_id": APEX_RUNTIME_MOUNT_POLICY,
            "mode": "read_only",
            "source_root": os.environ["AGENT_KERNEL_ARENA_APEX_SOURCE_ROOT"],
            "root": str(snapshot),
            "repository": {
                "commit": repository["commit"],
                "dirty": repository["dirty"],
                "status_sha256": repository["status_sha256"],
                "runtime_manifest_sha256": plan.sha256,
            },
            "runtime_manifest_sha256": plan.sha256,
            "runtime_manifest_path": str(snapshot / "runtime_manifest.json"),
            "runtime_manifest_relative_path": "runtime_manifest.json",
            "entrypoint": {
                "path": str(entrypoint),
                "relative_path": execution["entrypoint"],
                "sha256": _sha256_file(entrypoint),
            },
            "python": {
                "source_launcher_relative_path": execution["underlying_interpreter"],
                "launcher_path": str(launcher),
                "launcher_sha256": _sha256_file(launcher),
                "underlying_path": str(underlying),
                "underlying_sha256": _sha256_file(underlying),
                "flags": execution["flags"],
                "pythonpath": environment["PYTHONPATH"].split(os.pathsep),
                "environment": {
                    "PATH": environment["PATH"],
                    "APEX_RUNTIME_PYTHON": environment["APEX_RUNTIME_PYTHON"],
                    "PYTHONNOUSERSITE": environment["PYTHONNOUSERSITE"],
                    "PYTHONSAFEPATH": environment["PYTHONSAFEPATH"],
                    "PYTHONDONTWRITEBYTECODE": environment[
                        "PYTHONDONTWRITEBYTECODE"
                    ],
                },
            },
            "immutability": {
                "schema": immutable_mount["schema"],
                "policy_id": immutable_mount["policy_id"],
                "receipt_sha256": immutable_mount["sha256"],
                "runtime_image_input_sha256": image_inputs["sha256"],
                "image_sha256": immutable_mount["image_sha256"],
                "backing": immutable_mount["backing"],
                "requested_mount_options": immutable_mount[
                    "requested_mount_options"
                ],
                "runtime_service_evidence_sha256": immutable_mount[
                    "runtime_service_evidence_sha256"
                ],
                "runtime_engine_evidence_sha256": immutable_mount[
                    "runtime_engine_evidence_sha256"
                ],
                "host_access_policy": immutable_mount[
                    "host_access_policy"
                ],
                "mount": immutable_mount["mount"],
            },
            "attempt_mounts_sha256": attempt_mounts_sha256,
        }
    except (ApexRuntimeError, KeyError, OSError) as error:
        raise ApexAdapterError(f"formal Apex runtime snapshot is invalid: {error}") from error
    return {**material, "sha256": _canonical_digest(material)}


def _attempt_mount_receipt_valid(receipt: Any, *, runtime_root: Path) -> bool:
    if not isinstance(receipt, dict):
        return False
    material = dict(receipt)
    digest = material.pop("sha256", None)
    roles = receipt.get("roles")
    read_only = roles.get("read_only") if isinstance(roles, dict) else None
    namespace = receipt.get("namespace_mounts")
    return (
        receipt.get("schema") == ATTEMPT_MOUNT_RECEIPT_SCHEMA
        and receipt.get("campaign_data_root_hidden") is True
        and isinstance(read_only, dict)
        and isinstance(read_only.get("apex_runtime"), dict)
        and read_only["apex_runtime"].get("path") == str(runtime_root)
        and isinstance(namespace, dict)
        and _namespace_mount_v3_valid(
            namespace,
            runtime_root=runtime_root,
            source_roles=roles,
        )
        and isinstance(digest, str)
        and _SHA256.fullmatch(digest) is not None
        and digest == _canonical_digest(material)
    )


def _covered_mount_ids_valid(value: Any, *, expected_count: int | None) -> bool:
    if not isinstance(value, dict):
        return False
    covered = value.get("covered_mount_ids")
    mount = value.get("mount")
    if (
        not isinstance(covered, list)
        or any(type(item) is not int or item <= 0 for item in covered)
        or covered != sorted(covered)
        or len(covered) != len(set(covered))
        or not isinstance(mount, dict)
        or type(mount.get("mount_id")) is not int
        or mount["mount_id"] <= 0
        or mount["mount_id"] in covered
    ):
        return False
    return expected_count is None or len(covered) == expected_count


def _namespace_mount_v3_valid(
    namespace: Any,
    *,
    runtime_root: Path,
    source_roles: Any,
) -> bool:
    if (
        not isinstance(namespace, dict)
        or namespace.get("policy") != _NAMESPACE_MOUNT_POLICY
        or namespace.get("visible_mount_resolution_policy")
        != _VISIBLE_MOUNT_RESOLUTION_POLICY
        or namespace.get("closed_set") is not True
        or not isinstance(namespace.get("root"), dict)
        or namespace["root"].get("covered_mount_ids") != []
    ):
        return False
    roles = namespace.get("roles")
    if not isinstance(roles, dict) or set(roles) != {
        "persistent_writable",
        "read_only",
    }:
        return False
    expected = {
        "persistent_writable": {"apex_artifacts", "backend_home"},
        "read_only": {"scored_workspace", "sealed_task_contract", "apex_runtime"},
    }
    role_targets: list[dict[str, Any]] = []
    for group, names in expected.items():
        entries = roles.get(group)
        source_entries = (
            source_roles.get(group) if isinstance(source_roles, dict) else None
        )
        if (
            not isinstance(entries, dict)
            or set(entries) != names
            or not isinstance(source_entries, dict)
            or set(source_entries) != names
        ):
            return False
        for name, observation in entries.items():
            source_identity = source_entries[name]
            source = (
                observation.get("source") if isinstance(observation, dict) else None
            )
            target = observation.get("target") if isinstance(observation, dict) else None
            if (
                not isinstance(source_identity, dict)
                or not isinstance(source, dict)
                or source
                != {
                    "path": source_identity.get("path"),
                    "device": source_identity.get("device"),
                    "inode": source_identity.get("inode"),
                    "mount": source_identity.get("mount"),
                }
                or not isinstance(target, dict)
                or target.get("device") != source.get("device")
                or target.get("inode") != source.get("inode")
            ):
                return False
            role_targets.append(target)
            if name != "apex_runtime" and not _covered_mount_ids_valid(
                target, expected_count=0
            ):
                return False
    runtime = roles["read_only"]["apex_runtime"]
    source = runtime.get("source") if isinstance(runtime, dict) else None
    target = runtime.get("target") if isinstance(runtime, dict) else None
    source_mount = source.get("mount") if isinstance(source, dict) else None
    target_options = target.get("mount_options") if isinstance(target, dict) else None
    expected_covered = (
        1
        if isinstance(source_mount, dict)
        and source_mount.get("mount_point") == str(runtime_root)
        else 0
    )
    if (
        not isinstance(target, dict)
        or target.get("path") != str(runtime_root)
        or target.get("access") != "read_only"
        or not isinstance(target_options, list)
        or "ro" not in target_options
        or "rw" in target_options
        or (
            expected_covered == 1
            and source_mount.get("access") != "read_only"
        )
        or not _covered_mount_ids_valid(target, expected_count=expected_covered)
    ):
        return False
    private = namespace.get("private_tmpfs")
    data = namespace.get("campaign_data_root")
    if not isinstance(private, dict) or set(private) != {"tmp", "dev_shm"}:
        return False
    observations = [
        namespace["root"],
        data,
        private["tmp"],
        private["dev_shm"],
        *role_targets,
    ]
    if any(
        not _covered_mount_ids_valid(observation, expected_count=None)
        for observation in observations
    ):
        return False
    visible_ids = [observation["mount"]["mount_id"] for observation in observations]
    covered_ids = [
        mount_id
        for observation in observations
        for mount_id in observation["covered_mount_ids"]
    ]
    return bool(
        len(visible_ids) == len(set(visible_ids))
        and len(covered_ids) == len(set(covered_ids))
        and set(visible_ids).isdisjoint(covered_ids)
    )


def _immutable_runtime_receipt(
    *, snapshot: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    raw_path = os.environ.get("AGENT_KERNEL_ARENA_APEX_RUNTIME_MOUNT_RECEIPT")
    expected_digest = os.environ.get(
        "AGENT_KERNEL_ARENA_APEX_RUNTIME_MOUNT_RECEIPT_SHA256"
    )
    expected_file_digest = os.environ.get(
        "AGENT_KERNEL_ARENA_APEX_RUNTIME_MOUNT_RECEIPT_FILE_SHA256"
    )
    if (
        not isinstance(raw_path, str)
        or not Path(raw_path).is_absolute()
        or not isinstance(expected_digest, str)
        or not _SHA256.fullmatch(expected_digest)
        or not isinstance(expected_file_digest, str)
        or not _SHA256.fullmatch(expected_file_digest)
    ):
        raise ApexAdapterError("formal Apex immutable mount evidence is incomplete")
    path = Path(raw_path)
    receipt = _read_regular_json(
        path, size_limit=1024 * 1024, label="Apex immutable runtime mount receipt"
    )
    if (
        _sha256_file(path) != expected_file_digest
        or receipt.get("sha256") != expected_digest
        or receipt.get("schema") != RUNTIME_IMMUTABLE_MOUNT_SCHEMA
        or receipt.get("policy_id") != RUNTIME_IMMUTABLE_MOUNT_POLICY_ID
    ):
        raise ApexAdapterError("formal Apex immutable mount evidence changed")
    try:
        return validate_immutable_mount_receipt(snapshot, manifest, receipt)
    except ApexRuntimeError as error:
        raise ApexAdapterError(f"formal Apex immutable mount is invalid: {error}") from error


@register_agent("apex")
def launch_agent(
    eval_config: dict[str, Any],
    task_config_dir: str,
    workspace: str,
) -> str:
    """Run Apex and import only its independently validated source patch bundle."""
    logger = logging.getLogger(__name__)
    agent_config = _load_yaml(Path(__file__).with_name("agent_config.yaml"))
    task_config_path = Path(task_config_dir).resolve()
    task_config = _load_yaml(task_config_path)
    workspace_path = Path(workspace).resolve()
    if not workspace_path.is_dir():
        raise ApexAdapterError(f"Arena workspace does not exist: {workspace_path}")
    formal_campaign = is_formal_campaign(eval_config)
    campaign_binding = _campaign_binding(
        eval_config,
        task_config_path,
        formal_campaign=formal_campaign,
    )
    comparison_contract_sha256 = _comparison_contract_sha256(
        eval_config, formal_campaign=formal_campaign
    )
    workspace_before = _workspace_manifest(workspace_path) if formal_campaign else None

    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        + f"_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    )
    artifact_root = workspace_path.parent / f".{workspace_path.name}_apex" / run_id
    artifact_root.mkdir(parents=True, exist_ok=False)
    contract_root = (
        artifact_root.with_name(f"{artifact_root.name}.contract")
        if formal_campaign
        else artifact_root
    )
    if formal_campaign:
        contract_root.mkdir(mode=0o700, exist_ok=False)
    task_spec_path = contract_root / "task_spec.json"
    result_path = artifact_root / "result.json"

    prompt_builder = load_prompt_builder(AgentType.APEX, logger)
    prompt = prompt_builder(task_config_dir, str(workspace_path), eval_config, logger)
    task_spec = _build_task_spec(
        task_config_path=task_config_path,
        task_config=task_config,
        eval_config=eval_config,
        agent_config=agent_config,
        workspace=workspace_path,
        artifact_root=artifact_root,
        prompt=prompt,
        formal_campaign=formal_campaign,
        campaign_binding=campaign_binding,
    )
    original_prompt_bytes = prompt.encode("utf-8")
    instruction_adaptation = _validate_instruction_adaptation(
        task_spec, original_prompt_bytes
    )
    task_spec_bytes = (
        json.dumps(task_spec, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if formal_campaign:
        _write_read_only_atomic(task_spec_path, task_spec_bytes)
        contract_root.chmod(0o555)
        _fsync_directory(contract_root.parent)
        if not _sealed_task_spec_matches(task_spec_path, task_spec_bytes):
            raise ApexAdapterError("formal Apex TaskSpec contract could not be sealed")
    else:
        _atomic_write_json(task_spec_path, task_spec)
        task_spec_bytes = task_spec_path.read_bytes()
    task_spec_sha256 = _sha256_bytes(task_spec_bytes)

    entrypoint, python_path = _resolve_apex_command(agent_config)
    runtime_plan: RuntimePlan | None = None
    runtime_root: Path | None = None
    immutable_runtime_mount: dict[str, Any] | None = None
    apex_runtime_mount: dict[str, Any] | None = None
    apex_arguments = [
        "optimize",
        "kernel",
        "--task-spec",
        str(task_spec_path),
        "--result-json",
        str(result_path),
        "--non-interactive",
    ]
    if formal_campaign:
        runtime_plan = _formal_runtime_plan(entrypoint, python_path)
        runtime_root = Path(
            os.environ["AGENT_KERNEL_ARENA_APEX_RUNTIME_SNAPSHOT_ROOT"]
        )
        try:
            immutable_runtime_mount = _immutable_runtime_receipt(
                snapshot=runtime_root, manifest=runtime_plan.manifest
            )
            command = runtime_command(
                runtime_root,
                runtime_plan.manifest,
                apex_arguments,
                immutable_mount_receipt=immutable_runtime_mount,
            )
        except ApexRuntimeError as error:
            raise ApexAdapterError(
                f"formal Apex runtime snapshot failed: {error}"
            ) from error
    else:
        command = [python_path, str(entrypoint), *apex_arguments]
    gpu_evidence = formal_gpu_evidence(eval_config) if formal_campaign else None
    inner_agent_timeout = int(agent_config.get("timeout_seconds", 3600))
    internal_allowance = 0
    if formal_campaign:
        campaign = eval_config.get("campaign") or {}
        internal_allowance = int(campaign.get("apex_internal_allowance_seconds", 0))
        if inner_agent_timeout != 3600 or internal_allowance != 3600:
            raise ApexAdapterError(
                "formal campaign requires independent 3600-second agent and Apex evaluator budgets"
            )
    outer_timeout_contract = float(inner_agent_timeout + internal_allowance)
    timeout_seconds = outer_timeout_contract
    campaign_attempt = eval_config.get("campaign_attempt") or {}
    deadline = campaign_attempt.get("task_deadline_monotonic")
    if deadline is not None:
        remaining = float(deadline) - time.monotonic()
        if remaining <= 0:
            raise ApexAdapterError("hard task deadline expired before Apex launch")
        timeout_seconds = min(timeout_seconds, remaining)
    receipt_path: Path | None = None
    attempt_home: Path | None = None
    cloud_config_bootstrap: dict[str, Any] | None = None
    process_environment = _subprocess_environment(task_spec["agent_backend"])
    isolated_command = command
    attempt_mounts: dict[str, Any] | None = None
    if formal_campaign:
        assert runtime_plan is not None and runtime_root is not None
        raw_receipt = campaign_attempt.get("receipt_path")
        if not isinstance(raw_receipt, str) or not Path(raw_receipt).is_absolute():
            raise ApexAdapterError("formal campaign requires an absolute attempt receipt path")
        receipt_path = Path(raw_receipt)
        attempt_home = prepare_attempt_home(eval_config, backend=task_spec["agent_backend"])
        if attempt_home is None:
            raise ApexAdapterError("formal campaign did not create an isolated agent home")
        cloud_config_bootstrap = codex_cloud_config_bootstrap_receipt(attempt_home)
        process_environment = isolated_environment(process_environment, attempt_home)
        process_environment.update(
            runtime_environment(runtime_root, runtime_plan.manifest)
        )
    output_limit = int(agent_config.get("max_process_output_bytes", _DEFAULT_OUTPUT_LIMIT))
    backend = task_spec["agent_backend"]
    logger.info("Apex preflight")
    logger.info("  entrypoint: %s", entrypoint)
    logger.info("  workspace: %s", workspace_path)
    logger.info("  artifact root: %s", artifact_root)
    logger.info("  backend: %s", backend)
    logger.info("  task: %s", task_spec["task_id"])
    logger.info("  editable files: %s", task_spec["editable_files"])
    logger.info("  inner agent budget: %s", inner_agent_timeout)
    logger.info("  Apex internal allowance: %s", internal_allowance)
    logger.info("  outer process budget: %.3f", timeout_seconds)

    run_kwargs: dict[str, Any] = {
        "cwd": artifact_root,
        "backend": backend,
        "timeout_seconds": timeout_seconds,
        "output_limit": output_limit,
        "logger": logger,
    }
    if formal_campaign:
        assert runtime_plan is not None and runtime_root is not None
        run_kwargs["environment"] = process_environment
        isolated_command = wrap_attempt_command(
            command,
            eval_config=eval_config,
            writable_roots=(artifact_root, attempt_home),
            read_only_roots=(workspace_path, contract_root),
            trusted_read_only_roots=(runtime_root,),
            mount_roles={
                "apex_artifacts": artifact_root,
                "backend_home": attempt_home,
                "scored_workspace": workspace_path,
                "sealed_task_contract": contract_root,
                "apex_runtime": runtime_root,
            },
            # Apex is the trusted inner containment owner.  Its supervisor
            # creates the agent-visible private procfs; mounting another procfs
            # here breaks the required Apex -> Codex nested user namespace.
            private_proc=False,
        )
        attempt_mounts = attempt_mount_receipt(isolated_command)
        if not isinstance(attempt_mounts, dict):
            release_attempt_command_fds(isolated_command)
            raise ApexAdapterError(
                "formal Apex isolation did not create a mount receipt"
            )
    outcome = _normalize_process_outcome(_run_apex(isolated_command, **run_kwargs))
    if formal_campaign:
        assert (
            runtime_plan is not None
            and runtime_root is not None
            and immutable_runtime_mount is not None
        )
        attempt_mounts = attempt_mount_receipt(isolated_command)
        if not _attempt_mount_receipt_valid(
            attempt_mounts, runtime_root=runtime_root
        ):
            raise ApexAdapterError(
                "formal Apex isolation did not attest the exact runtime mount"
            )
        try:
            verify_runtime_snapshot(runtime_root, runtime_plan.sha256)
            immutable_runtime_mount = _immutable_runtime_receipt(
                snapshot=runtime_root, manifest=runtime_plan.manifest
            )
        except ApexRuntimeError as error:
            raise ApexAdapterError(
                f"formal Apex runtime changed during execution: {error}"
            ) from error
        assert attempt_mounts is not None
        apex_runtime_mount = _runtime_snapshot_receipt(
            plan=runtime_plan,
            snapshot=runtime_root,
            immutable_mount=immutable_runtime_mount,
            attempt_mounts_sha256=attempt_mounts["sha256"],
        )
    task_spec_postlaunch_unchanged = (
        _sealed_task_spec_matches(task_spec_path, task_spec_bytes)
        if formal_campaign
        else task_spec_path.read_bytes() == task_spec_bytes
    )
    return_code = outcome.exit_code
    process_output = outcome.output
    max_result_bytes = int(agent_config.get("max_result_bytes", _DEFAULT_RESULT_LIMIT))
    result: dict[str, Any] | None = None
    lineage: dict[str, Any] | None = None
    status = ""
    receipt: dict[str, Any] = {
        "schema": _APEX_RECEIPT_SCHEMA,
        "campaign_binding": campaign_binding,
        "comparison_contract_sha256": comparison_contract_sha256,
        "session_succeeded": False,
        "terminal_status": None,
        "exit_code": return_code,
        "timed_out": outcome.timed_out,
        "budgets": {
            "inner_agent_timeout_seconds": inner_agent_timeout,
            "apex_internal_allowance_seconds": internal_allowance,
            "outer_timeout_seconds": outer_timeout_contract,
            "effective_outer_timeout_seconds": timeout_seconds,
        },
        "attempt_containment_policy_id": ATTEMPT_CONTAINMENT_POLICY,
        "agent_process_containment_policy_id": (
            AGENT_PROCESS_CONTAINMENT_POLICY
        ),
        "attempt_process_cleanup": outcome.cleanup,
        "capture": {
            "readers_completed": outcome.readers_completed,
            "errors": list(outcome.capture_errors),
        },
        "gpu": gpu_evidence,
        "attempt_mounts": attempt_mounts,
        "apex_runtime_mount": apex_runtime_mount,
        "apex": {
            "entrypoint": (
                apex_runtime_mount["entrypoint"]["path"]
                if apex_runtime_mount is not None
                else str(entrypoint.resolve(strict=True))
            ),
            "entrypoint_sha256": (
                apex_runtime_mount["entrypoint"]["sha256"]
                if apex_runtime_mount is not None
                else _sha256_file(entrypoint.resolve(strict=True))
            ),
            "python": (
                apex_runtime_mount["python"]["launcher_path"]
                if apex_runtime_mount is not None
                else str(Path(python_path).resolve(strict=True))
            ),
            "python_sha256": (
                apex_runtime_mount["python"]["launcher_sha256"]
                if apex_runtime_mount is not None
                else _sha256_file(Path(python_path).resolve(strict=True))
            ),
        },
        "task_spec_sha256": task_spec_sha256,
        "instruction_adaptation": instruction_adaptation,
        "task_spec_contract": (
            {
                "policy": "prelaunch_read_only_sibling_bind_v1",
                "path": str(task_spec_path.resolve()),
                "sha256": task_spec_sha256,
                "size_bytes": len(task_spec_bytes),
                "file_mode": "0444",
                "directory_mode": "0555",
                "read_only_bind": True,
                "postlaunch_unchanged": task_spec_postlaunch_unchanged,
            }
            if formal_campaign
            else None
        ),
        "outer_isolation": (
            {
                "approval": "never_via_strict_config",
                "execpolicy_rules": "ignored",
                "project_instructions": "backend_default_may_load",
                "sandbox": "workspace-write",
                "session": "ephemeral",
                "user_config": "ignored",
                "mount_scope": "attempt_only_bubblewrap",
                "attempt_containment_policy_id": ATTEMPT_CONTAINMENT_POLICY,
            }
            if formal_campaign
            else None
        ),
        "workspace_integrity": (
            {
                "policy": "read_only_until_adapter_bundle_apply_v1",
                "baseline_manifest_sha256": _canonical_digest(workspace_before),
                "pre_apply_manifest_sha256": None,
                "pre_apply_unchanged": False,
            }
            if workspace_before is not None
            else None
        ),
        "codex": None,
        "invocation": None,
        "lineage": None,
        "candidate_persistence": None,
    }
    try:
        if formal_campaign and not task_spec_postlaunch_unchanged:
            raise ApexAdapterError(
                "formal Apex TaskSpec contract changed during subprocess execution"
            )
        if outcome.timed_out:
            raise ApexAdapterError(f"Apex timed out after {timeout_seconds:.3f} seconds")
        cleanup_verified = (
            attempt_cleanup_verified(
                outcome.cleanup,
                exit_code=return_code,
                allowed_reasons={"normal_exit"},
                required_procfs="trusted_orchestrator_inherited_procfs",
            )
            if formal_campaign
            else outcome.cleanup.get("verification_performed") is True
            and outcome.cleanup.get("verified_absent") is True
        )
        if not cleanup_verified:
            raise ApexAdapterError("Apex attempt cleanup was not verified")
        if not outcome.readers_completed or outcome.capture_errors:
            raise ApexAdapterError("Apex process output capture was incomplete")
        if workspace_before is not None:
            pre_apply_manifest = _workspace_manifest(workspace_path)
            pre_apply_digest = _canonical_digest(pre_apply_manifest)
            workspace_integrity = receipt["workspace_integrity"]
            assert isinstance(workspace_integrity, dict)
            workspace_integrity["pre_apply_manifest_sha256"] = pre_apply_digest
            workspace_integrity["pre_apply_unchanged"] = (
                pre_apply_manifest == workspace_before
            )
            if pre_apply_manifest != workspace_before:
                raise ApexAdapterError(
                    "scored workspace changed before adapter-owned bundle apply"
                )

        result = _read_regular_json(
            result_path,
            size_limit=max_result_bytes,
            label="Apex result",
        )
        if type(result.get("schema_version")) is not int or result["schema_version"] != _SCHEMA_VERSION:
            raise ApexAdapterError("unsupported Apex result schema_version")
        if result.get("task_id") != task_spec["task_id"]:
            raise ApexAdapterError("Apex result task_id does not match request")
        if result.get("applied") is not False:
            raise ApexAdapterError("Apex external-evaluator result must have applied=false")
        if result.get("external_verification_required") is not True:
            raise ApexAdapterError(
                "Apex result must acknowledge external_verification_required=true"
            )

        status_value = result.get("status")
        status = status_value if isinstance(status_value, str) else ""
        receipt["terminal_status"] = status
        if formal_campaign:
            lineage = _validate_apex_lineage(
                result=result, task_spec=task_spec, artifact_root=artifact_root
            )
            receipt["codex"] = {
                **lineage["codex"],
                "cloud_config_bootstrap": cloud_config_bootstrap,
            }
            receipt["invocation"] = lineage["invocation"]
            receipt["lineage"] = {
                "run_id": result.get("run_id"),
                "result_sha256": _sha256_file(result_path),
                "campaign_binding_sha256": _canonical_digest(
                    campaign_binding
                ),
                "journal_head_event_id": lineage["journal_head_event_id"],
                "journal_head_checksum": lineage["journal_head_checksum"],
                "event_count": lineage["event_count"],
                "transcript_sha256": lineage["transcript_digest"],
                "event_artifact_digests": lineage["event_artifact_digests"],
                "prompt_event": lineage["prompt_event"],
                "internal_verdict": result.get("internal_verdict"),
                "internal_verdict_ref": result.get("internal_verdict_ref"),
            }
            receipt["candidate_persistence"] = {
                "schema": "aka.candidate-persistence-receipt/v4",
                "policy_id": CANDIDATE_PERSISTENCE_POLICY,
                "agent_process_containment_policy_id": (
                    AGENT_PROCESS_CONTAINMENT_POLICY
                ),
                "agent_process_containment_sha256": _canonical_digest(
                    lineage["process_containment"]
                ),
                "attempt_containment_policy_id": ATTEMPT_CONTAINMENT_POLICY,
                "attempt_process_cleanup_sha256": _canonical_digest(
                    outcome.cleanup
                ),
                "termination_kind": lineage["termination_kind"],
                "termination_reason": lineage["termination_reason"],
                "capture_status": lineage["capture_status"],
                "candidate_capture_allowed": lineage[
                    "candidate_capture_allowed"
                ],
                "observer_stop_sent": lineage["observer_stop_sent"],
                "discarded_stdout_tail": lineage["discarded_stdout_tail"],
                "observed_turns": lineage["observed_turns"],
                "checkpoint": None,
            }
        if return_code != 0:
            raise ApexAdapterError(
                f"Apex returned {status or 'an invalid result'} with process exit "
                f"code {return_code}"
            )

        if status == "candidate_ready":
            changed = _validate_and_apply_bundle(
                result=result,
                task_spec=task_spec,
                workspace=workspace_path,
                artifact_root=artifact_root,
                max_result_bytes=max_result_bytes,
                max_bundle_bytes=int(agent_config.get("max_bundle_bytes", _DEFAULT_BUNDLE_LIMIT)),
            )
            if formal_campaign:
                assert lineage is not None
                snapshot = _bundle_snapshot_bytes(
                    result=result,
                    artifact_root=artifact_root,
                    size_limit=int(
                        agent_config.get(
                            "max_bundle_bytes", _DEFAULT_BUNDLE_LIMIT
                        )
                    ),
                )
                bundle_summary = {
                    "bundle_digest": result["bundle_digest"],
                    "snapshot_sha256": _sha256_bytes(snapshot),
                    "snapshot_size_bytes": len(snapshot),
                }
                lineage["bundle_snapshot_bytes"] = snapshot
                lineage["bundle"] = bundle_summary
                receipt_lineage = receipt["lineage"]
                assert isinstance(receipt_lineage, dict)
                receipt_lineage["bundle"] = bundle_summary
                persistence = receipt["candidate_persistence"]
                assert isinstance(persistence, dict)
                if lineage["termination_kind"] == "exact_turn_boundary":
                    persistence["checkpoint"] = {
                        "agent_event_id": lineage["agent_event_id"],
                        "gate_chain": lineage["checkpoint_gate_chain"],
                        **bundle_summary,
                    }
            logger.info("Applied validated Apex source bundle: %s", changed)
        elif status in _NORMAL_NO_PATCH_STATUSES:
            if result.get("bundle_path") is not None or result.get("bundle_digest") is not None:
                raise ApexAdapterError("no_gain result must not declare a bundle")
            if result.get("changed_files") != []:
                raise ApexAdapterError("no_gain result must have changed_files=[]")
            logger.info("Apex produced no accepted gain; Arena workspace remains at baseline")
        else:
            reason = result.get("reason_code") or result.get("error") or "unknown"
            raise ApexAdapterError(
                f"Apex did not produce a scoreable candidate: status={status!r} reason={reason} "
                f"exit_code={return_code}"
            )
        receipt["session_succeeded"] = True
    except Exception:
        if receipt_path is not None:
            _write_apex_attempt_receipt(
                receipt_path=receipt_path,
                task_spec_bytes=task_spec_bytes,
                original_prompt_bytes=original_prompt_bytes,
                result_path=result_path,
                outcome=outcome,
                receipt=receipt,
                lineage=lineage,
            )
        raise
    else:
        if receipt_path is not None:
            _write_apex_attempt_receipt(
                receipt_path=receipt_path,
                task_spec_bytes=task_spec_bytes,
                original_prompt_bytes=original_prompt_bytes,
                result_path=result_path,
                outcome=outcome,
                receipt=receipt,
                lineage=lineage,
            )

    public_result = {
        "schema_version": result.get("schema_version"),
        "task_id": result.get("task_id"),
        "status": status,
        "reason_code": result.get("reason_code"),
        "bundle_digest": result.get("bundle_digest"),
    }
    return process_output + "\n" + json.dumps(public_result, sort_keys=True)


__all__ = [
    "ApexAdapterError",
    "_apex_task_instructions",
    "_bundle_digest",
    "_build_task_spec",
    "_validate_and_apply_bundle",
    "launch_agent",
]
