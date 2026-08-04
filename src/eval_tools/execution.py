"""Bounded subprocess execution for isolated evaluation-tool workers.

The normal Arena evaluator historically uses :func:`subprocess.run`.  Evaluation
tools need a stricter contract: their output can be very large, and a timed-out
GPU launcher frequently has descendants that outlive its shell.  This module
therefore starts every command in a new process group, drains stdout/stderr to
bounded files, and terminates the complete group on exit or timeout.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Mapping, Sequence


DEFAULT_LOG_LIMIT_BYTES = 64 * 1024 * 1024
DEFAULT_TERM_GRACE_SECONDS = 10.0
DEFAULT_KILL_GRACE_SECONDS = 5.0
_COPY_CHUNK_BYTES = 64 * 1024


@dataclass(frozen=True)
class LogCapture:
    """Metadata for one drained subprocess stream."""

    path: str
    bytes_seen: int
    bytes_written: int
    truncated: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ExecutionResult:
    """Transport-neutral result returned by :func:`execute_command`."""

    exit_code: int | None
    signal: int | None
    timed_out: bool
    termination: str
    duration_ms: int
    stdout: LogCapture
    stderr: LogCapture

    @property
    def succeeded(self) -> bool:
        return not self.timed_out and self.exit_code == 0

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["succeeded"] = self.succeeded
        return value


class _BoundedDrain:
    """Drain one pipe without allowing its artifact to grow without bound."""

    def __init__(self, source: BinaryIO, destination: Path, limit_bytes: int):
        self._source = source
        self._destination = destination
        self._limit_bytes = limit_bytes
        self.bytes_seen = 0
        self.bytes_written = 0
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float | None = None) -> bool:
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _run(self) -> None:
        try:
            self._destination.parent.mkdir(parents=True, exist_ok=True)
            with self._destination.open("wb") as output:
                while True:
                    chunk = self._source.read(_COPY_CHUNK_BYTES)
                    if not chunk:
                        break
                    self.bytes_seen += len(chunk)
                    remaining = self._limit_bytes - self.bytes_written
                    if remaining > 0:
                        kept = chunk[:remaining]
                        output.write(kept)
                        self.bytes_written += len(kept)
                output.flush()
        except BaseException as error:  # surfaced in the caller after cleanup
            self.error = error
        finally:
            try:
                self._source.close()
            except OSError:
                pass

    def metadata(self) -> LogCapture:
        return LogCapture(
            path=str(self._destination),
            bytes_seen=self.bytes_seen,
            bytes_written=self.bytes_written,
            truncated=self.bytes_seen > self.bytes_written,
        )


def _validate_argv(argv: Sequence[str]) -> list[str]:
    if isinstance(argv, (str, bytes)) or not isinstance(argv, Sequence) or not argv:
        raise ValueError("argv must be a non-empty sequence of strings")
    normalized: list[str] = []
    for argument in argv:
        if not isinstance(argument, str) or "\x00" in argument:
            raise ValueError("argv entries must be NUL-free strings")
        normalized.append(argument)
    return normalized


def _validate_environment(environment: Mapping[str, str] | None) -> dict[str, str]:
    result = os.environ.copy()
    if environment is None:
        return result
    if not isinstance(environment, Mapping):
        raise ValueError("env must be a string mapping")
    for key, value in environment.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("env keys and values must be strings")
        if not key or "=" in key or "\x00" in key or "\x00" in value:
            raise ValueError("env contains an invalid key or NUL byte")
        result[key] = value
    return result


def _group_alive(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Treat a group we cannot signal as alive; the caller will surface the
        # failed cleanup rather than silently claiming success.
        return True
    return True


def _wait_group_gone(process_group: int, timeout: float) -> bool:
    deadline = time.monotonic() + max(0.0, timeout)
    while _group_alive(process_group):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)
    return True


def _signal_group(process_group: int, sig: signal.Signals) -> bool:
    try:
        os.killpg(process_group, sig)
        return True
    except ProcessLookupError:
        return False


def _terminate_group(
    process: subprocess.Popen[bytes],
    process_group: int,
    *,
    term_grace_seconds: float,
    kill_grace_seconds: float,
) -> str:
    """Terminate all members of ``process_group`` and reap the leader."""

    if not _group_alive(process_group):
        process.poll()
        return "none"

    _signal_group(process_group, signal.SIGTERM)
    try:
        process.wait(timeout=max(0.0, term_grace_seconds))
    except subprocess.TimeoutExpired:
        pass
    if _wait_group_gone(process_group, term_grace_seconds):
        return "sigterm"

    _signal_group(process_group, signal.SIGKILL)
    try:
        process.wait(timeout=max(0.0, kill_grace_seconds))
    except subprocess.TimeoutExpired:
        pass
    _wait_group_gone(process_group, kill_grace_seconds)
    return "sigkill"


def execute_command(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None,
    timeout_seconds: float,
    stdout_path: Path,
    stderr_path: Path,
    stdout_limit_bytes: int = DEFAULT_LOG_LIMIT_BYTES,
    stderr_limit_bytes: int = DEFAULT_LOG_LIMIT_BYTES,
    term_grace_seconds: float = DEFAULT_TERM_GRACE_SECONDS,
    kill_grace_seconds: float = DEFAULT_KILL_GRACE_SECONDS,
) -> ExecutionResult:
    """Execute ``argv`` and capture bounded logs.

    ``argv`` is deliberately a vector, not a shell string.  Shell behavior must
    be explicit (for example ``["bash", "-lc", command]``) in a trusted plugin.
    """

    normalized_argv = _validate_argv(argv)
    normalized_env = _validate_environment(env)
    cwd = Path(cwd)
    stdout_path = Path(stdout_path)
    stderr_path = Path(stderr_path)

    if not cwd.is_dir():
        raise ValueError(f"cwd is not a directory: {cwd}")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if stdout_limit_bytes < 0 or stderr_limit_bytes < 0:
        raise ValueError("log limits must be non-negative")
    if term_grace_seconds < 0 or kill_grace_seconds < 0:
        raise ValueError("termination grace periods must be non-negative")
    if stdout_path == stderr_path:
        raise ValueError("stdout_path and stderr_path must be different")

    started = time.monotonic()
    process = subprocess.Popen(
        normalized_argv,
        cwd=str(cwd),
        env=normalized_env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        close_fds=True,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    process_group = process.pid
    stdout_drain = _BoundedDrain(process.stdout, stdout_path, stdout_limit_bytes)
    stderr_drain = _BoundedDrain(process.stderr, stderr_path, stderr_limit_bytes)
    stdout_drain.start()
    stderr_drain.start()

    timed_out = False
    termination = "none"
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        termination = _terminate_group(
            process,
            process_group,
            term_grace_seconds=term_grace_seconds,
            kill_grace_seconds=kill_grace_seconds,
        )
    else:
        # A launcher may exit after spawning a daemon that inherited our pipes.
        # Evaluation commands are not allowed to leave background descendants.
        if _group_alive(process_group):
            termination = _terminate_group(
                process,
                process_group,
                term_grace_seconds=term_grace_seconds,
                kill_grace_seconds=kill_grace_seconds,
            )
    finally:
        if process.poll() is None:
            termination = _terminate_group(
                process,
                process_group,
                term_grace_seconds=term_grace_seconds,
                kill_grace_seconds=kill_grace_seconds,
            )
        # Once the complete group is gone both pipe writers are closed.  Keep the
        # wait bounded so an OS/filesystem failure cannot wedge the worker.
        drain_wait = term_grace_seconds + kill_grace_seconds + 1.0
        stdout_done = stdout_drain.join(drain_wait)
        stderr_done = stderr_drain.join(drain_wait)
        if not stdout_done or not stderr_done:
            try:
                process.stdout.close()
                process.stderr.close()
            except OSError:
                pass
            stdout_drain.join(1.0)
            stderr_drain.join(1.0)

    if stdout_drain.error is not None:
        raise RuntimeError(f"failed to capture stdout: {stdout_drain.error}")
    if stderr_drain.error is not None:
        raise RuntimeError(f"failed to capture stderr: {stderr_drain.error}")

    return_code = process.returncode
    exit_code = return_code if return_code is not None and return_code >= 0 else None
    signal_number = -return_code if return_code is not None and return_code < 0 else None
    duration_ms = max(0, int((time.monotonic() - started) * 1000))
    return ExecutionResult(
        exit_code=exit_code,
        signal=signal_number,
        timed_out=timed_out,
        termination=termination,
        duration_ms=duration_ms,
        stdout=stdout_drain.metadata(),
        stderr=stderr_drain.metadata(),
    )
