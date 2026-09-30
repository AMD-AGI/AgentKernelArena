"""Shared process-group cleanup and absolute deadlines; no agent policy."""
from __future__ import annotations

import os
import signal
import subprocess
import time


def stop_process_group(process: subprocess.Popen, grace_s: float = 5) -> None:
    """Also kill surviving children after their group leader exits."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        if sig == signal.SIGTERM:
            try:
                process.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                pass
    process.wait()


def bounded_timeout(local_limit: float, deadline_epoch: float | None = None) -> float:
    remaining = local_limit if deadline_epoch is None else min(local_limit, deadline_epoch - time.time())
    if remaining <= 0:
        raise TimeoutError("Task wall-clock budget exhausted")
    return remaining
