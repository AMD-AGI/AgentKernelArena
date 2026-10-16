"""Shared process-group cleanup, absolute deadlines and pre-measurement sweeps.

No agent policy lives here. Launchers decide when to stop their own process;
this module only knows how to stop a group, bound a timeout by the task
deadline, and clear a dedicated worker container before final evaluation.
"""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import time

CONTAINER_MARKER = "AGENT_KERNEL_ARENA_DEDICATED_CONTAINER"


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


def dedicated_container() -> bool:
    """True only inside a worker container the Docker runner started for one run."""
    return os.environ.get(CONTAINER_MARKER) == "1"


def _read(path: Path) -> str | None:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return None


def _stat_fields(text: str) -> tuple[str, str, int] | None:
    """Return (comm, state, ppid) from a /proc/<pid>/stat line."""
    # The command name is parenthesized and may itself contain spaces or ')'.
    close = text.rfind(")")
    if close < 0:
        return None
    comm = text[text.find("(") + 1:close]
    rest = text[close + 1:].split()
    if len(rest) < 2 or not rest[1].isdigit():
        return None
    return comm, rest[0], int(rest[1])


def list_foreign_processes(proc: Path = Path("/proc"), *, self_pid: int | None = None) -> list[dict]:
    """Processes in this PID namespace that are neither the caller nor its ancestors.

    Descendants count as foreign: after the agent phase nothing the framework
    started should still be running, and orphans re-parented to the namespace
    init are exactly the leftovers this sweep exists to find.
    """
    self_pid = os.getpid() if self_pid is None else self_pid
    table: dict[int, dict] = {}
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        text = _read(entry / "stat")
        fields = _stat_fields(text) if text else None
        if fields is None:
            continue
        comm, state, ppid = fields
        cmdline = (_read(entry / "cmdline") or "").replace("\0", " ").strip()
        table[int(entry.name)] = {"pid": int(entry.name), "ppid": ppid, "state": state,
                                  "comm": comm, "cmdline": cmdline}
    protected = {1}
    pid = self_pid
    while pid in table and pid not in protected:
        protected.add(pid)
        pid = table[pid]["ppid"]
    protected.add(self_pid)
    return [row for pid, row in sorted(table.items()) if pid not in protected and row["state"] != "Z"]


def _alive(pid: int, proc: Path) -> bool:
    try:
        reaped, _ = os.waitpid(pid, os.WNOHANG)
        if reaped == pid:
            return False
    except ChildProcessError:
        pass
    text = _read(proc / str(pid) / "stat")
    if text is not None:
        fields = _stat_fields(text)
        return fields is not None and fields[1] not in ("Z", "X")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def terminate_processes(pids, *, proc: Path = Path("/proc"), grace_s: float = 5.0) -> list[int]:
    """SIGTERM, wait for the grace period, SIGKILL; return the PIDs still alive."""
    live: set[int] = set()
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            live.add(pid)
        except ProcessLookupError:
            continue
        except PermissionError:
            live.add(pid)
    deadline = time.monotonic() + grace_s
    while live and time.monotonic() < deadline:
        live = {pid for pid in live if _alive(pid, proc)}
        if live:
            time.sleep(0.1)
    for pid in live:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    if live:
        time.sleep(0.2)
    return sorted(pid for pid in live if _alive(pid, proc))


def sweep_foreign_processes(*, proc: Path = Path("/proc"), grace_s: float = 5.0) -> dict:
    """Stop everything outside the framework's own ancestry before final evaluation.

    Refuses to run outside a dedicated container: on a shared host the same
    sweep would kill unrelated user processes.
    """
    if not dedicated_container():
        raise RuntimeError(f"Process sweep requires {CONTAINER_MARKER}=1 in a dedicated worker container")
    found = list_foreign_processes(proc)
    survivors = terminate_processes([row["pid"] for row in found], proc=proc, grace_s=grace_s)
    return {"status": "leftovers" if survivors else "clean", "terminated": found, "survivors": survivors}
