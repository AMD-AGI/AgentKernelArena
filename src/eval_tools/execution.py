"""Bounded subprocess execution for isolated evaluation-tool workers.

The normal Arena evaluator historically uses :func:`subprocess.run`.  Evaluation
tools need a stricter contract: their output can be very large, and a timed-out
GPU launcher frequently has descendants that outlive its shell.  This module
therefore starts every command in a new process group, drains stdout/stderr to
bounded files, and terminates the complete group on exit or timeout.
"""

from __future__ import annotations

import ctypes
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
    cleanup_required: bool
    duration_ms: int
    stdout: LogCapture
    stderr: LogCapture

    @property
    def succeeded(self) -> bool:
        return (
            not self.timed_out
            and not self.cleanup_required
            and self.exit_code == 0
        )

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


@dataclass(frozen=True)
class _ProcessInfo:
    pid: int
    ppid: int
    state: str
    start_time: int

    @property
    def identity(self) -> tuple[int, int]:
        return self.pid, self.start_time


def enable_child_subreaper() -> None:
    """Reparent double-forked invocation descendants to the sidecar worker."""

    if os.name != "posix" or not Path("/proc/self/stat").is_file():
        raise RuntimeError("process containment requires Linux /proc")
    libc = ctypes.CDLL(None, use_errno=True)
    # Linux prctl(PR_SET_CHILD_SUBREAPER, 1). This is unprivileged and ensures
    # descendants cannot escape ancestry tracking merely by double-forking.
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))


def _process_table() -> dict[int, _ProcessInfo]:
    result: dict[int, _ProcessInfo] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text(encoding="utf-8")
            fields = raw[raw.rfind(")") + 2 :].split()
            info = _ProcessInfo(
                pid=int(entry.name),
                state=fields[0],
                ppid=int(fields[1]),
                start_time=int(fields[19]),
            )
        except (FileNotFoundError, IndexError, OSError, ValueError):
            continue
        result[info.pid] = info
    return result


def _descendants(owner_pid: int) -> list[_ProcessInfo]:
    table = _process_table()
    ancestor_pids = {owner_pid}
    result: dict[int, _ProcessInfo] = {}
    while True:
        added = {
            pid: info
            for pid, info in table.items()
            if pid not in ancestor_pids and info.ppid in ancestor_pids
        }
        if not added:
            break
        result.update(added)
        ancestor_pids.update(added)
    return list(result.values())


class LinuxProcessContainment:
    """Track and clean every process created inside one sequential invocation."""

    def __init__(
        self,
        owner_pid: int,
        baseline: set[tuple[int, int]],
        *,
        fatal_on_failure: bool = False,
    ):
        self.owner_pid = owner_pid
        self.baseline = baseline
        self.fatal_on_failure = fatal_on_failure

    @classmethod
    def capture(
        cls, *, fatal_on_failure: bool = False
    ) -> "LinuxProcessContainment":
        owner_pid = os.getpid()
        return cls(
            owner_pid,
            {info.identity for info in _descendants(owner_pid)},
            fatal_on_failure=fatal_on_failure,
        )

    def _targets(self) -> list[_ProcessInfo]:
        return [
            info
            for info in _descendants(self.owner_pid)
            if info.identity not in self.baseline
        ]

    def _reap_direct_children(self, targets: Sequence[_ProcessInfo]) -> None:
        for info in targets:
            if info.ppid != self.owner_pid:
                continue
            try:
                os.waitpid(info.pid, os.WNOHANG)
            except (ChildProcessError, ProcessLookupError):
                pass

    def _signal_until_gone(self, sig: signal.Signals, timeout: float) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            targets = self._targets()
            if not targets:
                return True
            for info in targets:
                if info.state == "Z":
                    continue
                try:
                    os.kill(info.pid, sig)
                except ProcessLookupError:
                    pass
                except PermissionError as error:
                    raise RuntimeError(
                        f"cannot signal escaped invocation process {info.pid}"
                    ) from error
            self._reap_direct_children(targets)
            if not self._targets():
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.02)

    def cleanup(
        self, *, term_grace_seconds: float, kill_grace_seconds: float
    ) -> tuple[str, bool]:
        if not self._targets():
            return "none", False
        if self._signal_until_gone(signal.SIGTERM, term_grace_seconds):
            return "sigterm", True
        if self._signal_until_gone(signal.SIGKILL, kill_grace_seconds):
            return "sigkill", True
        survivors = sorted(info.pid for info in self._targets())
        if self.fatal_on_failure:
            # Exiting PID 1 tears down the complete Docker PID namespace. Never
            # leave a poisoned persistent worker alive for another request.
            os._exit(70)
        raise RuntimeError(
            f"failed to clean invocation PID containment; survivors={survivors}"
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
    containment: LinuxProcessContainment | None = None,
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
    cleanup_required = False
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
            cleanup_required = True
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
        if containment is not None:
            containment_termination, containment_required = containment.cleanup(
                term_grace_seconds=term_grace_seconds,
                kill_grace_seconds=kill_grace_seconds,
            )
            cleanup_required = cleanup_required or containment_required
            if containment_termination != "none":
                termination = containment_termination
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
        cleanup_required=cleanup_required,
        duration_ms=duration_ms,
        stdout=stdout_drain.metadata(),
        stderr=stderr_drain.metadata(),
    )
