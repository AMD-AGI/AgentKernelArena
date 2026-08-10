#!/usr/bin/env python3
"""Lifecycle and CLI for formal Codex cloud-config refreshes."""

from __future__ import annotations

import argparse
import base64
import binascii
import datetime as dt
import hashlib
import os
import resource
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

try:
    from .codex_cloud_config_evidence import (
        ALLOWED_ENVIRONMENT,
        HEALTH_SCHEMA,
        MAX_JSON_BYTES,
        RECEIPT_POLICY,
        RECEIPT_SCHEMA,
        CommandResult,
        Policy,
        RefreshError,
        RefreshOutcome,
        RefreshState,
        SupervisorContext,
        SupervisorStop,
        _atomic_write,
        _canonical_bytes,
        _load_state,
        _parse_timestamp,
        _process_identity_matches,
        _process_starttime,
        _read_account_id,
        _stable_read,
        _strict_json,
        _timestamp,
        _validate_cli_identity,
        _validate_private_root,
        _write_state,
        prepare_state,
    )
except ImportError:  # Script execution places src/ directly on sys.path.
    from codex_cloud_config_evidence import (
        ALLOWED_ENVIRONMENT,
        HEALTH_SCHEMA,
        MAX_JSON_BYTES,
        RECEIPT_POLICY,
        RECEIPT_SCHEMA,
        CommandResult,
        Policy,
        RefreshError,
        RefreshOutcome,
        RefreshState,
        SupervisorContext,
        SupervisorStop,
        _atomic_write,
        _canonical_bytes,
        _load_state,
        _parse_timestamp,
        _process_identity_matches,
        _process_starttime,
        _read_account_id,
        _stable_read,
        _strict_json,
        _timestamp,
        _validate_cli_identity,
        _validate_private_root,
        _write_state,
        prepare_state,
    )


def _clean_environment(state: RefreshState) -> dict[str, str]:
    environment = {
        "HOME": state.work_home,
        "CODEX_HOME": str(Path(state.work_home) / ".codex"),
        "TMPDIR": str(Path(state.root) / "tmp"),
        "XDG_CACHE_HOME": str(Path(state.root) / "cache"),
        "PATH": f"{state.cli.node_prefix}/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    for name in ALLOWED_ENVIRONMENT:
        value = os.environ.get(name)
        if value:
            environment[name] = value
    return environment


def _limit_output(size: int) -> None:
    resource.setrlimit(resource.RLIMIT_FSIZE, (size, size))


def _terminate_process(process: subprocess.Popen[bytes], grace_seconds: int) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass


def _execute_command(
    state: RefreshState,
    stdout_path: Path,
    stderr_path: Path,
    context: SupervisorContext | None,
) -> CommandResult:
    command = [state.cli.launcher_path, "app-server", "--listen", "stdio://"]
    try:
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            try:
                process = subprocess.Popen(
                    command,
                    cwd=Path(state.root) / "empty-cwd",
                    env=_clean_environment(state),
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                    preexec_fn=lambda: _limit_output(
                        state.policy.output_limit_bytes
                    ),
                )
            except OSError:
                return CommandResult(125, False, "command_spawn_failed")
            if context is not None:
                context.process = process
                if context.stopped.is_set():
                    _terminate_process(process, state.policy.term_grace_seconds)
            try:
                process.wait(timeout=state.policy.timeout_seconds)
            except subprocess.TimeoutExpired:
                _terminate_process(process, state.policy.term_grace_seconds)
                return CommandResult(124, True, "command_timeout")
            finally:
                if context is not None:
                    context.process = None
            if context is not None and context.stopped.is_set():
                raise SupervisorStop()
            return CommandResult(process.returncode, False, None)
    except FileExistsError as error:
        raise RefreshError("unsafe_output_capture") from error


def _file_evidence(path: Path, limit: int) -> dict[str, Any]:
    raw = _stable_read(path, limit, allow_empty=True)
    return {"sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


def _ensure_capture_exists(path: Path) -> None:
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_CLOEXEC
            | os.O_NOFOLLOW,
            0o600,
        )
    except FileExistsError:
        return
    os.close(descriptor)


def _atomic_publish(published: Path, name: str, raw: bytes) -> None:
    _atomic_write(published / name, raw, 0o600)


def _validate_cache_and_publish(
    state: RefreshState,
    *,
    cancelled: Callable[[], bool] | None = None,
) -> tuple[dict[str, Any], str, int, bool]:
    auth_raw, account_id = _read_account_id(
        Path(state.work_home) / ".codex" / "auth.json"
    )
    cache_path = Path(state.work_home) / ".codex" / "cloud-config-bundle-cache.json"
    cache_raw = _stable_read(cache_path, MAX_JSON_BYTES)
    cache = _strict_json(cache_raw, "invalid_cache_json")
    if not isinstance(cache, dict) or set(cache) != {"signature", "signed_payload"}:
        raise RefreshError("invalid_envelope_shape")
    try:
        signature = base64.b64decode(cache["signature"], validate=True)
    except (TypeError, ValueError, binascii.Error) as error:
        raise RefreshError("invalid_signature_shape") from error
    payload = cache.get("signed_payload")
    if (
        len(signature) != 32
        or not isinstance(payload, dict)
        or payload.get("version") != 1
        or payload.get("account_id") != account_id
        or not isinstance(payload.get("chatgpt_user_id"), str)
        or not payload["chatgpt_user_id"]
    ):
        raise RefreshError("invalid_envelope_identity")
    bundle = payload.get("bundle")
    if (
        not isinstance(bundle, dict)
        or set(bundle) != {"config_toml", "requirements_toml"}
        or not all(isinstance(value, dict) for value in bundle.values())
    ):
        raise RefreshError("invalid_bundle_shape")
    cached_at = _parse_timestamp(payload.get("cached_at"))
    expires = _parse_timestamp(payload.get("expires_at"))
    now = dt.datetime.now(dt.timezone.utc)
    lifetime_delta = expires - cached_at
    ttl_delta = expires - now
    lifetime = int(lifetime_delta.total_seconds())
    ttl = int(ttl_delta.total_seconds())
    if cached_at > now + dt.timedelta(seconds=state.policy.clock_skew_seconds):
        raise RefreshError("cached_at_in_future")
    if (
        expires <= cached_at
        or lifetime_delta
        > dt.timedelta(seconds=state.policy.maximum_envelope_lifetime_seconds)
    ):
        raise RefreshError("invalid_envelope_lifetime")
    if expires <= now + dt.timedelta(seconds=state.policy.minimum_ttl_seconds):
        raise RefreshError("insufficient_envelope_ttl")
    if expires <= now + dt.timedelta(seconds=state.policy.refresh_early_seconds):
        raise RefreshError("insufficient_envelope_refresh_window")
    bundle_sha256 = hashlib.sha256(_canonical_bytes(bundle)).hexdigest()
    bundle_matches = (
        not state.anchor_bundle_sha256
        or bundle_sha256 == state.anchor_bundle_sha256
    )
    published = Path(state.published_directory)
    unexpected = {entry.name for entry in published.iterdir()} - {
        "auth.json",
        "cloud-config-bundle-cache.json",
    }
    if unexpected:
        raise RefreshError("published_state_not_minimal")
    evidence = {
        "sha256": hashlib.sha256(cache_raw).hexdigest(),
        "size_bytes": len(cache_raw),
        "bundle_sha256": bundle_sha256,
        "envelope_lifetime_seconds": lifetime,
        "remaining_ttl_seconds": ttl,
        "signed_envelope_shape_validated": True,
        "signature_verified_by_runner": False,
    }
    next_refresh = int(expires.timestamp()) - state.policy.refresh_early_seconds
    if not bundle_matches:
        return evidence, bundle_sha256, next_refresh, False
    if cancelled is not None and cancelled():
        raise SupervisorStop()
    _atomic_publish(published, "auth.json", auth_raw)
    _atomic_publish(published, "cloud-config-bundle-cache.json", cache_raw)
    return evidence, bundle_sha256, next_refresh, True


def _safe_failure(error: BaseException) -> str:
    if isinstance(error, RefreshError):
        return error.code
    return f"internal_{type(error).__name__.lower()}"


def _persist_receipt(
    state: RefreshState, sequence: int, material: dict[str, Any]
) -> tuple[str, str]:
    canonical = _canonical_bytes(material)
    digest = hashlib.sha256(canonical).hexdigest()
    material = dict(material)
    material["sha256"] = digest
    raw = _canonical_bytes(material) + b"\n"
    destination = Path(state.campaign_data_root) / (
        f"codex-cloud-config-refresh-{sequence:06d}-{digest}.json"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".codex-cloud-config-refresh.", suffix=".json", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o444)
        remaining = memoryview(raw)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise RefreshError("short_receipt_write")
            remaining = remaining[written:]
        os.fsync(descriptor)
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError:
            observed: bytes | None = None
            for _ in range(50):
                try:
                    observed = _stable_read(destination, MAX_JSON_BYTES)
                    break
                except RefreshError as error:
                    if error.code not in {
                        "unsafe_regular_file",
                        "unsafe_or_missing_file",
                    }:
                        raise
                    time.sleep(0.01)
            if observed != raw:
                raise RefreshError("receipt_digest_collision")
        temporary.unlink()
        directory = os.open(
            destination.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    metadata = destination.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o444
        or metadata.st_nlink != 1
        or metadata.st_uid != os.getuid()
        or metadata.st_gid != os.getgid()
        or metadata.st_size != len(raw)
    ):
        raise RefreshError("unsafe_published_receipt")
    return digest, str(destination)


def refresh_once(
    state: RefreshState,
    reason: str,
    *,
    context: SupervisorContext | None = None,
) -> RefreshOutcome:
    if reason not in {"prelaunch", "scheduled", "diagnostic"}:
        raise RefreshError("invalid_refresh_reason")
    _validate_private_root(Path(state.root), state.root_device, state.root_inode)
    sequence = state.sequence + 1
    io_root = Path(tempfile.mkdtemp(prefix="io.", dir=state.root))
    os.chmod(io_root, 0o700)
    stdout_path = io_root / "stdout"
    stderr_path = io_root / "stderr"
    started_at = _timestamp()
    command = CommandResult(125, False, "refresh_not_started")
    cache_evidence: dict[str, Any] | None = None
    status = "fatal"
    failure: str | None = None
    bundle_sha256 = ""
    next_refresh_epoch = 0
    promoted = False
    try:
        _validate_cli_identity(state.cli)
        cache_path = Path(state.work_home) / ".codex" / "cloud-config-bundle-cache.json"
        try:
            cache_path.unlink()
        except FileNotFoundError:
            pass
        command = _execute_command(state, stdout_path, stderr_path, context)
        if command.failure is not None or command.exit_code != 0:
            raise RefreshError(command.failure or "command_failed")
        _validate_cli_identity(state.cli)
        cache_evidence, bundle_sha256, next_refresh_epoch, promoted = (
            _validate_cache_and_publish(
                state,
                cancelled=(context.stopped.is_set if context is not None else None),
            )
        )
        if promoted:
            status = "success"
        else:
            failure = "bundle_changed"
    except SupervisorStop:
        raise
    except BaseException as error:
        failure = _safe_failure(error)
    finally:
        finished_at = _timestamp()
    _ensure_capture_exists(stdout_path)
    _ensure_capture_exists(stderr_path)
    try:
        stdout_evidence = _file_evidence(
            stdout_path, state.policy.output_limit_bytes
        )
        stderr_evidence = _file_evidence(
            stderr_path, state.policy.output_limit_bytes
        )
    except RefreshError as error:
        status = "fatal"
        promoted = False
        failure = error.code
        stdout_evidence = None
        stderr_evidence = None
    material = {
        "schema": RECEIPT_SCHEMA,
        "policy_id": RECEIPT_POLICY,
        "sequence": sequence,
        "reason": reason,
        "status": status,
        "failure": failure,
        "started_at": started_at,
        "finished_at": finished_at,
        "command": {
            "argv": ["app-server", "--listen", "stdio://"],
            "model_invocation": False,
            "timeout_seconds": state.policy.timeout_seconds,
            "output_limit_bytes_per_stream": state.policy.output_limit_bytes,
            "exit_code": command.exit_code,
            "timed_out": command.timed_out,
            "stdout": stdout_evidence,
            "stderr": stderr_evidence,
        },
        "cli": {
            "launcher_path": state.cli.launcher_path,
            "resolved_path": state.cli.launcher_resolved_path,
            "sha256": state.cli.launcher_sha256,
            "node_resolved_path": state.cli.node_resolved_path,
            "node_sha256": state.cli.node_sha256,
            "backend_runtime_closure_sha256": (
                state.cli.backend_runtime_closure_sha256
            ),
        },
        "cache": cache_evidence,
        "bundle_matches_initial": (
            None
            if not bundle_sha256
            else not state.anchor_bundle_sha256
            or bundle_sha256 == state.anchor_bundle_sha256
        ),
        "promoted": promoted,
        "payload_recorded": False,
    }
    try:
        receipt_sha256, receipt_path = _persist_receipt(state, sequence, material)
    finally:
        shutil.rmtree(io_root, ignore_errors=True)
    return RefreshOutcome(
        status=status,
        failure=failure,
        sequence=sequence,
        bundle_sha256=bundle_sha256,
        next_refresh_epoch=next_refresh_epoch,
        receipt_sha256=receipt_sha256,
        receipt_path=receipt_path,
        promoted=promoted,
    )


def _write_fatal(state: RefreshState, receipt_sha256: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", receipt_sha256):
        raise RefreshError("invalid_fatal_receipt_digest")
    _atomic_write(Path(state.root) / "fatal", (receipt_sha256 + "\n").encode())


def _write_health(
    state: RefreshState, status_value: str, receipt_sha256: str
) -> None:
    material = {
        "schema": HEALTH_SCHEMA,
        "status": status_value,
        "pid": os.getpid(),
        "starttime": _process_starttime(os.getpid()),
        "sequence": state.sequence,
        "anchor_bundle_sha256": state.anchor_bundle_sha256,
        "last_receipt_sha256": receipt_sha256,
        "updated_at": _timestamp(),
    }
    material["sha256"] = hashlib.sha256(_canonical_bytes(material)).hexdigest()
    _atomic_write(
        Path(state.root) / "health.json", _canonical_bytes(material) + b"\n"
    )


def _persist_supervisor_failure(
    state: RefreshState, failure: str
) -> tuple[str, str]:
    sequence = state.sequence + 1
    now = _timestamp()
    material = {
        "schema": RECEIPT_SCHEMA,
        "policy_id": RECEIPT_POLICY,
        "sequence": sequence,
        "reason": "supervisor",
        "status": "fatal",
        "failure": failure,
        "started_at": now,
        "finished_at": now,
        "command": None,
        "cli": {
            "launcher_path": state.cli.launcher_path,
            "resolved_path": state.cli.launcher_resolved_path,
            "sha256": state.cli.launcher_sha256,
            "node_resolved_path": state.cli.node_resolved_path,
            "node_sha256": state.cli.node_sha256,
            "backend_runtime_closure_sha256": (
                state.cli.backend_runtime_closure_sha256
            ),
        },
        "cache": None,
        "bundle_matches_initial": None,
        "promoted": False,
        "payload_recorded": False,
    }
    return _persist_receipt(state, sequence, material)


def _signal_owner(pid: int, starttime: int) -> None:
    if _process_identity_matches(pid, starttime):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def bootstrap(
    auth_source: Path,
    node_prefix: Path,
    campaign_data_root: Path,
    owner_pid: int,
    owner_starttime: int,
    *,
    policy: Policy | None = None,
    reason: str = "prelaunch",
) -> tuple[RefreshState, RefreshOutcome]:
    state = prepare_state(
        auth_source,
        node_prefix,
        campaign_data_root,
        owner_pid,
        owner_starttime,
        policy=policy,
    )
    try:
        outcome = refresh_once(state, reason)
        if outcome.status != "success" or not outcome.bundle_sha256:
            _write_fatal(state, outcome.receipt_sha256)
            raise RefreshError(outcome.failure or "prelaunch_refresh_failed")
        state.sequence = outcome.sequence
        state.anchor_bundle_sha256 = outcome.bundle_sha256
        state.next_refresh_epoch = outcome.next_refresh_epoch
        _write_state(state)
        return state, outcome
    except BaseException:
        cleanup_private_root(
            Path(state.root), state.root_device, state.root_inode
        )
        raise


def supervise(root: Path, device: int, inode: int) -> int:
    state = _load_state(root, device, inode)
    context = SupervisorContext()

    def request_stop(_signum: int, _frame: Any) -> None:
        context.stopped.set()
        if context.process is not None:
            _terminate_process(context.process, state.policy.term_grace_seconds)

    for selected_signal in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        signal.signal(selected_signal, request_stop)
    _write_health(state, "running", "")
    while not context.stopped.is_set():
        delay = max(1, state.next_refresh_epoch - int(time.time()))
        if context.stopped.wait(delay):
            break
        outcome = refresh_once(state, "scheduled", context=context)
        if context.stopped.is_set():
            break
        if outcome.status != "success":
            _write_fatal(state, outcome.receipt_sha256)
            _write_health(state, "fatal", outcome.receipt_sha256)
            _signal_owner(state.owner_pid, state.owner_starttime)
            return 1
        state.sequence = outcome.sequence
        state.next_refresh_epoch = outcome.next_refresh_epoch
        _write_state(state)
        _write_health(state, "running", outcome.receipt_sha256)
    return 0


def supervisor_main(root: Path, device: int, inode: int) -> int:
    state: RefreshState | None = None
    try:
        state = _load_state(root, device, inode)
        return supervise(root, device, inode)
    except SupervisorStop:
        return 0
    except BaseException as error:
        if state is not None:
            try:
                digest, _ = _persist_supervisor_failure(
                    state, _safe_failure(error)
                )
                _write_fatal(state, digest)
            except BaseException:
                pass
            _signal_owner(state.owner_pid, state.owner_starttime)
        return 1


def health_check(
    root: Path, device: int, inode: int, pid: int, starttime: int
) -> bool:
    _validate_private_root(root, device, inode)
    if not _process_identity_matches(pid, starttime):
        return False
    material = _strict_json(
        _stable_read(root / "health.json", MAX_JSON_BYTES), "invalid_health_json"
    )
    if not isinstance(material, dict):
        return False
    digest = material.pop("sha256", None)
    return bool(
        material.get("schema") == HEALTH_SCHEMA
        and material.get("status") == "running"
        and material.get("pid") == pid
        and material.get("starttime") == starttime
        and isinstance(digest, str)
        and digest == hashlib.sha256(_canonical_bytes(material)).hexdigest()
    )


def mark_unexpected_exit(root: Path, device: int, inode: int) -> str:
    state = _load_state(root, device, inode)
    if (root / "fatal").is_file():
        return _stable_read(root / "fatal", 128).decode("ascii").strip()
    digest, _ = _persist_supervisor_failure(state, "supervisor_unexpected_exit")
    _write_fatal(state, digest)
    return digest


def cleanup_private_root(root: Path, device: int, inode: int) -> None:
    _validate_private_root(root, device, inode)
    shutil.rmtree(root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    bootstrap_parser = commands.add_parser("bootstrap")
    bootstrap_parser.add_argument("--auth-source", required=True, type=Path)
    bootstrap_parser.add_argument("--node-prefix", required=True, type=Path)
    bootstrap_parser.add_argument("--campaign-data-root", required=True, type=Path)
    bootstrap_parser.add_argument("--owner-pid", required=True, type=int)
    bootstrap_parser.add_argument("--owner-starttime", required=True, type=int)
    bootstrap_parser.add_argument(
        "--reason", choices=("prelaunch", "diagnostic"), default="prelaunch"
    )
    for name in ("supervise", "mark-unexpected-exit", "cleanup"):
        selected = commands.add_parser(name)
        selected.add_argument("--root", required=True, type=Path)
        selected.add_argument("--device", required=True, type=int)
        selected.add_argument("--inode", required=True, type=int)
    health_parser = commands.add_parser("health")
    health_parser.add_argument("--root", required=True, type=Path)
    health_parser.add_argument("--device", required=True, type=int)
    health_parser.add_argument("--inode", required=True, type=int)
    health_parser.add_argument("--pid", required=True, type=int)
    health_parser.add_argument("--starttime", required=True, type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "bootstrap":
            state, outcome = bootstrap(
                arguments.auth_source,
                arguments.node_prefix,
                arguments.campaign_data_root,
                arguments.owner_pid,
                arguments.owner_starttime,
                reason=arguments.reason,
            )
            print(
                "\t".join(
                    (
                        state.root,
                        str(state.root_device),
                        str(state.root_inode),
                        str(Path(state.root) / "published"),
                        state.anchor_bundle_sha256,
                        state.cli.backend_runtime_closure_sha256,
                        str(state.next_refresh_epoch),
                        outcome.receipt_sha256,
                    )
                )
            )
            return 0
        if arguments.command == "supervise":
            return supervisor_main(arguments.root, arguments.device, arguments.inode)
        if arguments.command == "health":
            return 0 if health_check(
                arguments.root,
                arguments.device,
                arguments.inode,
                arguments.pid,
                arguments.starttime,
            ) else 1
        if arguments.command == "mark-unexpected-exit":
            mark_unexpected_exit(arguments.root, arguments.device, arguments.inode)
            return 0
        if arguments.command == "cleanup":
            cleanup_private_root(arguments.root, arguments.device, arguments.inode)
            return 0
    except RefreshError as error:
        print(f"Codex cloud-config refresh failed: {error.code}", file=sys.stderr)
        return 1
    except OSError:
        print("Codex cloud-config refresh failed: operating_system_error", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
