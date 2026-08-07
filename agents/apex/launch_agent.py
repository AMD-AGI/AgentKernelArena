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

import hashlib
import json
import logging
import os
import re
import shlex
import signal
import stat
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import yaml

from agents import register_agent
from src.module_registration import AgentType, load_prompt_builder


_SCHEMA_VERSION = 1
_DEFAULT_RESULT_LIMIT = 8 * 1024 * 1024
_DEFAULT_BUNDLE_LIMIT = 64 * 1024 * 1024
_DEFAULT_OUTPUT_LIMIT = 4 * 1024 * 1024
_NORMAL_NO_PATCH_STATUSES = {"no_gain"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
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
) -> dict[str, Any]:
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
    commands = {
        "compile": compile_commands,
        "correctness": correctness_commands,
        "performance": performance_commands,
    }
    recipe_material = {
        "task_config": _sha256_file(task_config_path),
        "commands": commands,
        "source_files": sources,
    }
    backend = str(agent_config.get("backend") or "codex").strip().lower()
    if backend not in {"codex", "claude", "cursor"}:
        raise ApexAdapterError(
            f"Apex backend must be codex, claude, or cursor; got {backend!r}"
        )
    task_id = _task_id(task_config_path)
    return {
        "schema_version": _SCHEMA_VERSION,
        "task_id": task_id,
        "workspace": str(workspace),
        "results_dir": str(artifact_root),
        "instructions": prompt,
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
        },
        "budget": {
            "max_iterations": int(agent_config.get("max_iterations", 1)),
            "max_turns": int(agent_config.get("max_turns", 25)),
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
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_process_group(process: subprocess.Popen[bytes], logger: logging.Logger) -> None:
    process_group_id = process.pid
    if not _process_group_exists(process_group_id):
        process.poll()
        return
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and _process_group_exists(process_group_id):
        process.poll()
        time.sleep(0.1)
    if not _process_group_exists(process_group_id):
        return
    logger.warning("Force killing Apex process group")
    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        return


def _capture_stream(
    stream,
    *,
    label: str,
    limit: int,
    chunks: list[bytes],
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
    finally:
        stream.close()


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
    timeout_seconds: int,
    output_limit: int,
    logger: logging.Logger,
) -> tuple[int, str]:
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_subprocess_environment(backend),
        start_new_session=True,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    stdout: list[bytes] = []
    stderr: list[bytes] = []
    stdout_thread = threading.Thread(
        target=_capture_stream,
        kwargs={
            "stream": process.stdout,
            "label": "[APEX]",
            "limit": output_limit,
            "chunks": stdout,
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
            "logger": logger.warning,
        },
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    timed_out = False
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
    finally:
        _terminate_process_group(process, logger)
        stdout_thread.join(timeout=2)
        stderr_thread.join(timeout=2)
    if timed_out:
        raise TimeoutError(f"Apex timed out after {timeout_seconds} seconds")
    output = b"".join(stdout).decode("utf-8", "replace")
    error_output = b"".join(stderr).decode("utf-8", "replace")
    return int(process.returncode or 0), "\n".join(
        part for part in (output, error_output) if part
    )


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

    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        + f"_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    )
    artifact_root = workspace_path.parent / f".{workspace_path.name}_apex" / run_id
    artifact_root.mkdir(parents=True, exist_ok=False)
    task_spec_path = artifact_root / "task_spec.json"
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
    )
    _atomic_write_json(task_spec_path, task_spec)

    entrypoint, python_path = _resolve_apex_command(agent_config)
    command = [
        python_path,
        str(entrypoint),
        "optimize",
        "kernel",
        "--task-spec",
        str(task_spec_path),
        "--result-json",
        str(result_path),
        "--non-interactive",
    ]
    timeout_seconds = int(agent_config.get("timeout_seconds", 14400))
    output_limit = int(agent_config.get("max_process_output_bytes", _DEFAULT_OUTPUT_LIMIT))
    backend = task_spec["agent_backend"]
    logger.info("Apex preflight")
    logger.info("  entrypoint: %s", entrypoint)
    logger.info("  workspace: %s", workspace_path)
    logger.info("  artifact root: %s", artifact_root)
    logger.info("  backend: %s", backend)
    logger.info("  task: %s", task_spec["task_id"])
    logger.info("  editable files: %s", task_spec["editable_files"])

    return_code, process_output = _run_apex(
        command,
        cwd=artifact_root,
        backend=backend,
        timeout_seconds=timeout_seconds,
        output_limit=output_limit,
        logger=logger,
    )
    max_result_bytes = int(agent_config.get("max_result_bytes", _DEFAULT_RESULT_LIMIT))
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
    if status == "candidate_ready":
        if return_code != 0:
            raise ApexAdapterError(
                f"Apex returned candidate_ready with process exit code {return_code}"
            )
        changed = _validate_and_apply_bundle(
            result=result,
            task_spec=task_spec,
            workspace=workspace_path,
            artifact_root=artifact_root,
            max_result_bytes=max_result_bytes,
            max_bundle_bytes=int(agent_config.get("max_bundle_bytes", _DEFAULT_BUNDLE_LIMIT)),
        )
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
    "_bundle_digest",
    "_build_task_spec",
    "_validate_and_apply_bundle",
    "launch_agent",
]
