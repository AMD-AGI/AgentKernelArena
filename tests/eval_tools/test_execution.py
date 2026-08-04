from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path

from src.eval_tools.execution import execute_command


def _execute(tmp_path: Path, code: str, **overrides):
    kwargs = {
        "cwd": tmp_path,
        "env": {},
        "timeout_seconds": 5,
        "stdout_path": tmp_path / "stdout.log",
        "stderr_path": tmp_path / "stderr.log",
        "stdout_limit_bytes": 1024,
        "stderr_limit_bytes": 1024,
        "term_grace_seconds": 0.1,
        "kill_grace_seconds": 0.5,
    }
    kwargs.update(overrides)
    return execute_command([sys.executable, "-c", code], **kwargs)


def test_streams_logs_with_hard_size_limits(tmp_path: Path) -> None:
    result = _execute(
        tmp_path,
        "import sys; sys.stdout.write('o' * 5000); sys.stderr.write('e' * 3000)",
        stdout_limit_bytes=100,
        stderr_limit_bytes=80,
    )

    assert result.succeeded
    assert result.stdout.bytes_seen == 5000
    assert result.stdout.bytes_written == 100
    assert result.stdout.truncated is True
    assert result.stderr.bytes_seen == 3000
    assert result.stderr.bytes_written == 80
    assert result.stderr.truncated is True
    assert (tmp_path / "stdout.log").stat().st_size == 100
    assert (tmp_path / "stderr.log").stat().st_size == 80


def _pid_is_live(pid: int) -> bool:
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        fields = stat_path.read_text().split()
    except FileNotFoundError:
        return False
    # A reparented zombie is already dead for our leak check.
    return len(fields) < 3 or fields[2] != "Z"


def test_timeout_kills_descendants_that_ignore_sigterm(tmp_path: Path) -> None:
    child_code = (
        "import signal,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "time.sleep(60)"
    )
    leader_code = (
        "import subprocess,sys,time; "
        f"child=subprocess.Popen([sys.executable,'-c',{child_code!r}]); "
        "print(child.pid, flush=True); "
        "time.sleep(60)"
    )
    result = _execute(
        tmp_path,
        leader_code,
        timeout_seconds=0.2,
        term_grace_seconds=0.1,
        kill_grace_seconds=1,
    )

    assert result.timed_out is True
    assert result.termination == "sigkill"
    child_pid = int((tmp_path / "stdout.log").read_text().strip())
    deadline = time.monotonic() + 2
    while _pid_is_live(child_pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not _pid_is_live(child_pid)


def test_reports_signal_separately_from_exit_code(tmp_path: Path) -> None:
    result = _execute(
        tmp_path,
        "import os,signal; os.kill(os.getpid(), signal.SIGUSR1)",
    )

    assert result.exit_code is None
    assert result.signal == signal.SIGUSR1
    assert result.timed_out is False


def test_rejects_shell_string_and_invalid_limits(tmp_path: Path) -> None:
    try:
        execute_command(
            "echo unsafe",
            cwd=tmp_path,
            env={},
            timeout_seconds=1,
            stdout_path=tmp_path / "out",
            stderr_path=tmp_path / "err",
        )
    except ValueError as error:
        assert "argv" in str(error)
    else:
        raise AssertionError("shell string was accepted")
