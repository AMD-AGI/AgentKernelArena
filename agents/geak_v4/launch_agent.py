# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""AgentKernelArena adapter for GEAK v4's deterministic kernel workflow."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from agents import register_agent
from src.harness_guard import (
    is_protected_workspace_path,
    snapshot_workspace_harness,
    verify_workspace_harness,
)
from src.module_registration import AgentType, load_prompt_builder


_COPY_IGNORED_DIRS = {
    ".git",
    ".rocprofv3",
    ".torch_ext",
    "__pycache__",
    "build",
}
_COPY_IGNORED_SUFFIXES = {".o", ".pyc", ".so"}
_PATCH_SIZE_LIMIT = 16 * 1024 * 1024
_JSON_SIZE_LIMIT = 8 * 1024 * 1024
_PROCESS_OUTPUT_LIMIT = 4 * 1024 * 1024
_UNSAFE_SOURCE_NAME = re.compile(
    r"(?:test_.*|.*_(?:test|harness))\.(?:py|c|cc|cpp|cxx|cu|hip)",
    re.IGNORECASE,
)


def _load_agent_config() -> dict[str, Any]:
    path = Path(__file__).with_name("agent_config.yaml")
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _load_task_config(task_config_dir: str) -> dict[str, Any]:
    with Path(task_config_dir).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _single_declared_source(
    task_config: dict[str, Any],
    workspace: Path,
) -> PurePosixPath:
    raw = task_config.get("source_file_path")
    values = [raw] if isinstance(raw, str) else raw
    if not isinstance(values, list) or len(values) != 1:
        raise ValueError(
            "GEAK v4 V1 requires exactly one source_file_path; "
            f"got {raw!r}"
        )
    value = values[0]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("GEAK v4 source_file_path must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"source_file_path must not contain outer whitespace: {value!r}")
    if "\\" in value or any(ord(char) < 32 for char in value):
        raise ValueError(f"unsafe source_file_path: {value!r}")

    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or relative.as_posix() != value
        or any(part in ("", ".", "..") for part in relative.parts)
    ):
        raise ValueError(f"source_file_path must be a safe relative path: {value!r}")
    if is_protected_workspace_path(Path(*relative.parts)):
        raise ValueError(f"declared source is protected by the Arena harness guard: {value}")
    if _UNSAFE_SOURCE_NAME.fullmatch(relative.name):
        raise ValueError(
            "GEAK v4 V1 does not support a source file that also looks like a "
            f"co-located test/harness: {value}"
        )

    source = workspace.joinpath(*relative.parts)
    try:
        metadata = source.lstat()
    except OSError as exc:
        raise ValueError(f"declared source is not readable: {source}") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise ValueError(f"declared source must be an existing regular file: {source}")
    try:
        source.resolve().relative_to(workspace.resolve())
    except ValueError as exc:
        raise ValueError(f"declared source escapes the workspace: {source}") from exc
    return relative


def _snapshot_ignore(directory: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        path = Path(directory) / name
        if name in _COPY_IGNORED_DIRS:
            ignored.add(name)
        elif path.is_file() and path.suffix in _COPY_IGNORED_SUFFIXES:
            ignored.add(name)
        elif name in {"task_result.yaml", "validation_report.yaml"}:
            ignored.add(name)
    return ignored


def _materialize_disposable_input(workspace: Path, destination: Path) -> None:
    """Copy the task so Workflow tools never receive the scoring workspace path."""
    if destination.exists():
        raise FileExistsError(f"disposable GEAK input already exists: {destination}")
    symlinks = sorted(
        str(path.relative_to(workspace))
        for path in workspace.rglob("*")
        if path.is_symlink()
    )
    if symlinks:
        raise ValueError(
            "GEAK v4 does not accept symlinks in a task workspace because a "
            f"disposable copy could retain references to protected data: {symlinks[:10]}"
        )
    shutil.copytree(
        workspace,
        destination,
        symlinks=False,
        ignore=_snapshot_ignore,
    )


def _logical_gpu_ids(eval_config: dict[str, Any]) -> str:
    """Return GPU IDs in the process-visible namespace.

    Arena parallel workers mask one physical GPU with ROCR_VISIBLE_DEVICES and
    expose it as logical HIP/CUDA device 0. GEAK's gpu_lock wrapper rewrites
    HIP_VISIBLE_DEVICES again, so forwarding the host ID would hide the GPU.
    """
    if os.environ.get("AGENT_KERNEL_ARENA_HOST_GPU_ID") is not None:
        return "0"

    visible = (
        os.environ.get("HIP_VISIBLE_DEVICES")
        or os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    if visible:
        count = len([part for part in visible.split(",") if part.strip()])
        if count:
            return ",".join(str(index) for index in range(count))

    override = os.environ.get("GEAK_V4_GPU_IDS")
    configured = override if override is not None else eval_config.get("gpu_ids", "0")
    if isinstance(configured, (list, tuple)):
        configured = ",".join(str(item) for item in configured)
    return str(configured)


def _directory_identity(path: Path) -> tuple[int, int]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RuntimeError(f"artifact directory is not readable: {path}") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(
            f"artifact directory must be a real directory, not a symlink: {path}"
        )
    return metadata.st_dev, metadata.st_ino


def _open_directory_fd(
    path: Path,
    expected_identity: tuple[int, int] | None = None,
) -> int:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    metadata = os.fstat(descriptor)
    identity = (metadata.st_dev, metadata.st_ino)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or (
            expected_identity is not None
            and identity != expected_identity
        )
    ):
        os.close(descriptor)
        raise RuntimeError(f"artifact directory identity changed: {path}")
    return descriptor


def _verify_directory_identity(
    path: Path,
    expected_identity: tuple[int, int],
) -> None:
    if _directory_identity(path) != expected_identity:
        raise RuntimeError(f"artifact directory identity changed: {path}")


def _verify_artifact_directories(
    artifact_root: Path,
    artifact_root_identity: tuple[int, int],
    run_dir: Path,
    run_dir_identity: tuple[int, int],
) -> None:
    _verify_directory_identity(artifact_root, artifact_root_identity)
    _verify_directory_identity(run_dir, run_dir_identity)


def _new_run_paths(workspace: Path) -> dict[str, Path]:
    root = workspace.parent / f".{workspace.name}_geak_v4"
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        + f"_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    )
    run_dir = root / run_id
    root.mkdir(mode=0o700, exist_ok=True)
    root_identity = _directory_identity(root)
    root_fd = _open_directory_fd(root, root_identity)
    try:
        os.mkdir(run_id, mode=0o700, dir_fd=root_fd)
    finally:
        os.close(root_fd)
    _verify_directory_identity(root, root_identity)
    _directory_identity(run_dir)
    return {
        "artifact_root": root,
        "run_dir": run_dir,
        "input": run_dir / "input",
        "eval": run_dir / "eval",
        "exp_root": run_dir / "runs",
        "handoff": run_dir / "handoff.json",
        "result": run_dir / "result.json",
    }


def _atomic_write_json(
    path: Path,
    value: dict[str, Any],
    *,
    expected_parent_identity: tuple[int, int] | None = None,
) -> None:
    encoded = (
        json.dumps(value, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    directory_fd = _open_directory_fd(path.parent, expected_parent_identity)
    temporary_fd = -1
    temporary_name = ""
    try:
        proc_directory = Path(f"/proc/self/fd/{directory_fd}")
        temporary_fd, temporary_path = tempfile.mkstemp(
            prefix=f".{path.name}.tmp.",
            dir=proc_directory,
        )
        temporary_name = Path(temporary_path).name
        metadata = os.fstat(temporary_fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise OSError("atomic JSON temporary is not a private regular file")
        with os.fdopen(temporary_fd, "wb", closefd=True) as stream:
            temporary_fd = -1
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(
            temporary_name,
            path.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        temporary_name = ""
        os.fsync(directory_fd)
    finally:
        if temporary_fd >= 0:
            os.close(temporary_fd)
        if temporary_name:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)


def _read_bounded_text(path: Path, size_limit: int) -> str | None:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return None
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size > size_limit
        ):
            return None
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            content = stream.read(size_limit + 1)
        if len(content) > size_limit:
            return None
        return content.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    finally:
        os.close(descriptor)


def _read_json(path: Path) -> dict[str, Any] | None:
    content = _read_bounded_text(path, _JSON_SIZE_LIMIT)
    if content is None:
        return None
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_group_exit(process: subprocess.Popen[str], pgid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        process.poll()
        if not _process_group_exists(pgid):
            return True
        time.sleep(0.1)
    return not _process_group_exists(pgid)


def _terminate_process_group(
    process: subprocess.Popen[str],
    logger: logging.Logger,
) -> None:
    pgid = process.pid
    if not _process_group_exists(pgid):
        process.poll()
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    if _wait_group_exit(process, pgid, 10):
        return
    logger.warning("Force killing GEAK runner process group")
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return
    _wait_group_exit(process, pgid, 5)


def _stream_pipe(
    stream,
    prefix: str,
    output: list[str],
    log,
) -> None:
    captured = 0
    truncated = False
    try:
        while True:
            chunk = stream.read(4096)
            if not chunk:
                break
            remaining = _PROCESS_OUTPUT_LIMIT - captured
            if remaining > 0:
                retained = chunk[:remaining]
                output.append(retained)
                captured += len(retained)
                compact = " ".join(retained[:2000].split())
                if compact:
                    log(f"{prefix} {compact[:500]}")
            if len(chunk) > remaining and not truncated:
                truncated = True
                log(f"{prefix} output truncated at {_PROCESS_OUTPUT_LIMIT} characters")
    finally:
        stream.close()


def _run_workflow_runner(
    handoff_path: Path,
    result_path: Path,
    *,
    timeout_seconds: int,
    logger: logging.Logger,
) -> str:
    runner = Path(__file__).with_name("workflow_runner.py")
    command = [sys.executable, str(runner), str(handoff_path), str(result_path)]
    process = subprocess.Popen(
        command,
        cwd=str(handoff_path.parent),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        bufsize=1,
        start_new_session=True,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    stdout: list[str] = []
    stderr: list[str] = []
    stdout_thread = threading.Thread(
        target=_stream_pipe,
        args=(process.stdout, "[GEAK]", stdout, logger.info),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_stream_pipe,
        args=(process.stderr, "[GEAK STDERR]", stderr, logger.warning),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    timed_out = False
    try:
        process.wait(timeout=timeout_seconds + 30)
    except subprocess.TimeoutExpired:
        timed_out = True
        logger.error("GEAK v4 runner exceeded its hard timeout")
    finally:
        # Also clean descendants if the runner leader exited while a background
        # Workflow/Claude process remained in the session.
        _terminate_process_group(process, logger)
        stdout_thread.join(timeout=2)
        stderr_thread.join(timeout=2)

    if timed_out:
        raise TimeoutError(f"GEAK v4 timed out after {timeout_seconds} seconds")

    result = _read_json(result_path)
    stderr_text = "".join(stderr)
    if process.returncode != 0:
        detail = result.get("error") if result else stderr_text[-4000:]
        raise RuntimeError(
            f"GEAK v4 runner failed with exit {process.returncode}: {detail}"
        )
    if result is None:
        raise RuntimeError(f"GEAK v4 runner did not write a valid result: {result_path}")
    return "\n".join(
        part for part in ("".join(stdout), stderr_text) if part
    )


def _run_git_apply_inspection(
    patch: Path,
    *options: str,
    cwd: Path | None = None,
    binary: bool = False,
    isolate_from_parent_repo: bool = False,
) -> subprocess.CompletedProcess:
    environment = None
    if isolate_from_parent_repo:
        if cwd is None:
            raise ValueError("isolated git apply inspection requires cwd")
        environment = dict(os.environ)
        environment.pop("GIT_DIR", None)
        environment.pop("GIT_WORK_TREE", None)
        environment["GIT_CEILING_DIRECTORIES"] = str(cwd.parent.resolve())
    return subprocess.run(
        ["git", "apply", *options, str(patch)],
        cwd=str(cwd) if cwd else None,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=not binary,
        check=False,
    )


def _patch_paths(patch: Path) -> list[PurePosixPath]:
    result = _run_git_apply_inspection(patch, "--numstat", "-z", binary=True)
    if result.returncode != 0:
        stderr = os.fsdecode(result.stderr)
        raise RuntimeError(f"cannot parse GEAK patch: {stderr.strip()}")

    paths: list[PurePosixPath] = []
    for record in result.stdout.split(b"\0"):
        if not record:
            continue
        fields = record.split(b"\t", 2)
        if len(fields) != 3:
            raise RuntimeError("GEAK patch has malformed numstat output")
        added, deleted, raw_path = fields
        if not added.isdigit() or not deleted.isdigit():
            raise RuntimeError("binary GEAK patches are not accepted")
        value = os.fsdecode(raw_path)
        if (
            not value
            or "\\" in value
            or any(ord(char) < 32 for char in value)
        ):
            raise RuntimeError(f"GEAK patch contains an unsafe path: {value!r}")
        path = PurePosixPath(value)
        if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
            raise RuntimeError(f"GEAK patch path escapes the workspace: {value!r}")
        paths.append(path)
    return paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _workspace_manifest(workspace: Path) -> dict[str, tuple[Any, ...]]:
    """Capture all workspace entries so Workflow cannot bypass patch import."""
    manifest: dict[str, tuple[Any, ...]] = {}
    for path in sorted(workspace.rglob("*")):
        relative = str(path.relative_to(workspace))
        metadata = path.lstat()
        mode = stat.S_IMODE(metadata.st_mode)
        if stat.S_ISLNK(metadata.st_mode):
            manifest[relative] = ("symlink", mode, os.readlink(path))
        elif stat.S_ISREG(metadata.st_mode):
            manifest[relative] = ("file", mode, metadata.st_size, _sha256(path))
        elif stat.S_ISDIR(metadata.st_mode):
            manifest[relative] = ("directory", mode)
        else:
            manifest[relative] = ("special", stat.S_IFMT(metadata.st_mode), mode)
    return manifest


def _verify_workspace_manifest(
    expected: dict[str, tuple[Any, ...]],
    workspace: Path,
) -> None:
    """Reject any direct mutation of the Arena scoring workspace."""
    current = _workspace_manifest(workspace)
    if current == expected:
        return

    expected_paths = set(expected)
    current_paths = set(current)
    added = sorted(current_paths - expected_paths)
    deleted = sorted(expected_paths - current_paths)
    changed = sorted(
        path
        for path in expected_paths & current_paths
        if expected[path] != current[path]
    )
    raise RuntimeError(
        "GEAK v4 detected a direct mutation of the Arena scoring workspace; "
        "only the validated single-file patch import is allowed "
        f"(added={added[:10]}, deleted={deleted[:10]}, changed={changed[:10]})"
    )


def _apply_validated_patch(
    *,
    result: dict[str, Any],
    expected_eval_dir: Path,
    workspace: Path,
    source_path: PurePosixPath,
    min_improve: float,
    run_dir: Path,
    expected_workspace_manifest: dict[str, tuple[Any, ...]] | None = None,
    artifact_root: Path | None = None,
    artifact_root_identity: tuple[int, int] | None = None,
    run_dir_identity: tuple[int, int] | None = None,
) -> bool:
    if result.get("schema_version") != 1:
        raise RuntimeError(
            "GEAK result schema is not supported: "
            f"{result.get('schema_version')!r}"
        )
    status = str(result.get("status") or "")
    if status in {"no_gain", "rejected"}:
        return False
    if status != "ok":
        raise RuntimeError(f"GEAK result is not importable: status={status!r}")
    if result.get("validation_status") != "accepted":
        raise RuntimeError("GEAK result was not accepted by its Director")
    if result.get("correctness") != "pass":
        raise RuntimeError("GEAK Director did not report correctness=pass")
    if str(result.get("applied_to_original", "unknown")).lower() != "false":
        raise RuntimeError("GEAK unexpectedly wrote directly to its input snapshot")
    if not math.isfinite(min_improve) or min_improve < 0:
        raise RuntimeError("GEAK min_improve policy must be finite and non-negative")

    try:
        speedup = float(result["final_speedup"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("GEAK result has no numeric final_speedup") from exc
    if not math.isfinite(speedup):
        raise RuntimeError("GEAK final_speedup must be finite")
    if speedup < 1.0 + min_improve:
        return False

    raw_eval_dir = result.get("eval_dir")
    if not isinstance(raw_eval_dir, str):
        raise RuntimeError("GEAK result is missing eval_dir")
    if raw_eval_dir != str(expected_eval_dir):
        raise RuntimeError("GEAK result eval_dir does not match the pinned directory")
    if expected_eval_dir.is_symlink() or not expected_eval_dir.is_dir():
        raise RuntimeError(
            f"GEAK eval_dir is missing, not a directory, or a symlink: {expected_eval_dir}"
        )

    # The patch location is fixed by the workflow contract. Never trust an
    # arbitrary absolute path returned by an agent.
    patch = expected_eval_dir / "final_patch.diff"
    for field in ("final_patch", "director_final_patch"):
        reported = result.get(field)
        if not isinstance(reported, str) or reported != str(patch):
            raise RuntimeError(
                f"GEAK {field} does not match the Director-validated pinned patch"
            )
    workflow_patch = result.get("workflow_final_patch")
    if workflow_patch is not None and workflow_patch != str(patch):
        raise RuntimeError(
            "GEAK workflow_final_patch does not match the pinned patch"
        )
    if patch.is_symlink() or not patch.is_file():
        raise RuntimeError(f"GEAK final patch is missing or not a regular file: {patch}")
    size = patch.stat().st_size
    if size <= 0 or size > _PATCH_SIZE_LIMIT:
        raise RuntimeError(f"GEAK final patch has an invalid size: {size} bytes")

    touched = _patch_paths(patch)
    if touched != [source_path]:
        raise RuntimeError(
            "GEAK patch must modify exactly the declared source file; "
            f"declared={source_path}, touched={touched}"
        )
    if is_protected_workspace_path(Path(*source_path.parts)):
        raise RuntimeError(f"GEAK patch targets a protected harness path: {source_path}")

    if (
        artifact_root is not None
        and artifact_root_identity is not None
        and run_dir_identity is not None
    ):
        _verify_artifact_directories(
            artifact_root,
            artifact_root_identity,
            run_dir,
            run_dir_identity,
        )

    baseline_manifest = (
        dict(expected_workspace_manifest)
        if expected_workspace_manifest is not None
        else _workspace_manifest(workspace)
    )
    _verify_workspace_manifest(baseline_manifest, workspace)

    summary = _run_git_apply_inspection(patch, "--summary")
    if summary.returncode != 0:
        raise RuntimeError(f"cannot summarize GEAK patch: {summary.stderr.strip()}")
    if summary.stdout.strip():
        raise RuntimeError(
            "GEAK patch contains a create/delete/rename/mode operation: "
            + summary.stdout.strip()
        )

    destination = workspace.joinpath(*source_path.parts)
    destination_metadata = destination.lstat()
    if (
        not stat.S_ISREG(destination_metadata.st_mode)
        or destination_metadata.st_nlink != 1
    ):
        raise RuntimeError(f"Arena source changed type before patch import: {destination}")
    original_mode = destination_metadata.st_mode
    original_digest = _sha256(destination)
    harness_snapshot = snapshot_workspace_harness(workspace)

    with tempfile.TemporaryDirectory(prefix="patch_stage_", dir=run_dir) as temporary:
        staging = Path(temporary)
        staged_source = staging.joinpath(*source_path.parts)
        staged_source.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(destination, staged_source)

        check = _run_git_apply_inspection(
            patch,
            "--check",
            cwd=staging,
            isolate_from_parent_repo=True,
        )
        if check.returncode != 0:
            raise RuntimeError(f"GEAK patch does not apply cleanly: {check.stderr.strip()}")
        apply_result = _run_git_apply_inspection(
            patch,
            cwd=staging,
            isolate_from_parent_repo=True,
        )
        if apply_result.returncode != 0:
            raise RuntimeError(f"failed to apply GEAK patch: {apply_result.stderr.strip()}")
        staged_metadata = staged_source.lstat()
        if (
            not stat.S_ISREG(staged_metadata.st_mode)
            or staged_metadata.st_nlink != 1
        ):
            raise RuntimeError("GEAK patch changed the source file type")
        if staged_metadata.st_mode != original_mode:
            raise RuntimeError("GEAK patch changed the source file mode")
        if _sha256(staged_source) == original_digest:
            raise RuntimeError("GEAK patch reported success but did not change the source")

        current_metadata = destination.lstat()
        if (
            not stat.S_ISREG(current_metadata.st_mode)
            or current_metadata.st_nlink != 1
            or current_metadata.st_mode != original_mode
            or _sha256(destination) != original_digest
        ):
            raise RuntimeError("Arena source changed while GEAK patch was staged")
        # TemporaryDirectory lives beside the workspace, so os.replace is an
        # atomic same-filesystem mutation of the one approved source file.
        os.replace(staged_source, destination)

    verify_workspace_harness(harness_snapshot)
    updated_metadata = destination.lstat()
    expected_after = dict(baseline_manifest)
    expected_after[str(source_path)] = (
        "file",
        stat.S_IMODE(updated_metadata.st_mode),
        updated_metadata.st_size,
        _sha256(destination),
    )
    _verify_workspace_manifest(expected_after, workspace)

    if (
        artifact_root is not None
        and artifact_root_identity is not None
        and run_dir_identity is not None
    ):
        _verify_artifact_directories(
            artifact_root,
            artifact_root_identity,
            run_dir,
            run_dir_identity,
        )
    audit_parent_identity = (
        run_dir_identity
        if run_dir_identity is not None
        else _directory_identity(run_dir)
    )
    _atomic_write_json(
        run_dir / "applied_patch.json",
        {
            "source_file": str(source_path),
            "patch": str(patch),
            "patch_sha256": _sha256(patch),
            "source_sha256": _sha256(destination),
            "director_speedup": speedup,
        },
        expected_parent_identity=audit_parent_identity,
    )
    _verify_workspace_manifest(expected_after, workspace)
    return True


@register_agent("geak_v4")
def launch_agent(
    eval_config: dict[str, Any],
    task_config_dir: str,
    workspace: str,
) -> str:
    """Run GEAK v4 against a disposable copy, then import one validated patch."""
    logger = logging.getLogger(__name__)
    agent_config = _load_agent_config()
    task_config = _load_task_config(task_config_dir)
    workspace_path = Path(workspace).resolve()
    if not workspace_path.is_dir():
        raise FileNotFoundError(f"Arena workspace does not exist: {workspace_path}")

    task_type = str(task_config.get("task_type") or "")
    supported = {str(value) for value in agent_config.get("supported_task_types", [])}
    if task_type not in supported:
        raise ValueError(
            f"GEAK v4 V1 does not support task_type={task_type!r}; "
            f"supported task types: {sorted(supported)}"
        )
    source_path = _single_declared_source(task_config, workspace_path)
    workspace_manifest = _workspace_manifest(workspace_path)

    workflow_dir = Path(
        os.environ.get("GEAK_V4_WORKFLOW_DIR")
        or agent_config.get("workflow_dir")
        or "/opt/geak/kernel_workflow"
    ).resolve()
    workflow_script = workflow_dir / "kernel_workflow.js"
    if not workflow_script.is_file():
        raise FileNotFoundError(
            f"GEAK v4 workflow not found: {workflow_script}. "
            "Set AKA_GEAK_ROOT on the host and run make docker-setup-geak."
        )
    claude_binary = shutil.which("claude")
    if not claude_binary:
        raise RuntimeError("Claude Code CLI not found on PATH")

    paths = _new_run_paths(workspace_path)
    artifact_root_identity = _directory_identity(paths["artifact_root"])
    run_dir_identity = _directory_identity(paths["run_dir"])
    _materialize_disposable_input(workspace_path, paths["input"])

    prompt_builder = load_prompt_builder(AgentType.GEAK_V4, logger)
    task_prompt = prompt_builder(
        task_config_dir,
        str(paths["input"]),
        eval_config,
        logger,
    )
    task_prompt += (
        "\n\n### GEAK/Arena Integration Contract\n"
        "The config.yaml compile, correctness, and performance commands are the "
        "measurement source of truth. Do not create, modify, or replace any test, "
        "harness, config, eval_tools, reference, or timing file. Optimize only "
        f"`{source_path}`. The caller will accept a patch only when it modifies "
        "that one existing file."
    )

    timeout_seconds = int(agent_config.get("timeout_seconds", 43200))
    handoff = {
        "schema_version": 1,
        "kernel_path": str(paths["input"]),
        "workflow_dir": str(workflow_dir),
        "eval_dir": str(paths["eval"]),
        "exp_root": str(paths["exp_root"]),
        "gpu_ids": _logical_gpu_ids(eval_config),
        "budget": int(agent_config.get("budget", 6)),
        "min_improve": float(agent_config.get("min_improve", 0.02)),
        "deep_cost": int(agent_config.get("deep_cost", 2)),
        "use_expert_skills": bool(agent_config.get("use_expert_skills", False)),
        "task": task_prompt,
        "model": str(agent_config.get("model", "claude-opus-4-8")),
        "effort": str(agent_config.get("effort", "ultracode")),
        "claude_cli_path": claude_binary,
        "timeout_seconds": timeout_seconds,
        "done_grace_seconds": float(agent_config.get("done_grace_seconds", 1800)),
        "done_poll_seconds": float(agent_config.get("done_poll_seconds", 5)),
    }
    _atomic_write_json(
        paths["handoff"],
        handoff,
        expected_parent_identity=run_dir_identity,
    )

    logger.info("GEAK v4 preflight")
    logger.info("  workflow: %s", workflow_script)
    logger.info("  disposable input: %s", paths["input"])
    logger.info("  eval dir: %s", paths["eval"])
    logger.info("  source allowlist: %s", source_path)
    logger.info("  logical GPU IDs: %s", handoff["gpu_ids"])
    logger.info("  budget: %s", handoff["budget"])
    logger.info("  timeout: %ss", timeout_seconds)

    try:
        output = _run_workflow_runner(
            paths["handoff"],
            paths["result"],
            timeout_seconds=timeout_seconds,
            logger=logger,
        )
    finally:
        try:
            _verify_artifact_directories(
                paths["artifact_root"],
                artifact_root_identity,
                paths["run_dir"],
                run_dir_identity,
            )
        finally:
            _verify_workspace_manifest(workspace_manifest, workspace_path)
    result = _read_json(paths["result"])
    if result is None:
        raise RuntimeError("GEAK v4 result disappeared after runner completion")

    applied = _apply_validated_patch(
        result=result,
        expected_eval_dir=paths["eval"],
        workspace=workspace_path,
        source_path=source_path,
        min_improve=float(agent_config.get("min_improve", 0.02)),
        run_dir=paths["run_dir"],
        expected_workspace_manifest=workspace_manifest,
        artifact_root=paths["artifact_root"],
        artifact_root_identity=artifact_root_identity,
        run_dir_identity=run_dir_identity,
    )
    _verify_artifact_directories(
        paths["artifact_root"],
        artifact_root_identity,
        paths["run_dir"],
        run_dir_identity,
    )
    if applied:
        logger.info("Imported GEAK v4 Director-validated patch into Arena workspace")
    else:
        logger.info("GEAK v4 produced no accepted gain; Arena workspace is unchanged")
    return output + "\n" + json.dumps(result, sort_keys=True)
