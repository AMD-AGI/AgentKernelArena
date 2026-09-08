"""Bounded subprocesses with file logs and whole-process-group cleanup."""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path


def run_process(argv: list[str], cwd: Path, log: Path, timeout: float, env=None) -> dict:
    log.parent.mkdir(parents=True, exist_ok=True)
    timed_out = False
    owns_group = os.environ.get("SIKL_TASK_BUILDER_CHILD") != "1"
    child_env = dict(os.environ if env is None else env)
    child_env["SIKL_TASK_BUILDER_CHILD"] = "1"
    with log.open("w") as output:
        process = subprocess.Popen(argv, cwd=cwd, env=child_env, stdin=subprocess.DEVNULL,
                                   stdout=output, stderr=subprocess.STDOUT, start_new_session=owns_group)
        try:
            process.wait(timeout=max(0.01, timeout))
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # Descendants may survive their leader. Never leave GPU/agent work
            # running while the controller accepts files or starts another task.
            try:
                if owns_group:
                    os.killpg(process.pid, signal.SIGKILL)
                elif timed_out:
                    process.kill()
            except ProcessLookupError:
                pass
            process.wait()
    with log.open("rb") as output:
        output.seek(max(0, log.stat().st_size - 6000))
        tail = output.read().decode(errors="replace")
    return {"ok": process.returncode == 0 and not timed_out, "exit_code": process.returncode,
            "timed_out": timed_out, "log": str(log), "tail": tail}
