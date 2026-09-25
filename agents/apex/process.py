"""Bound the optimizer process, output and descendant lifetime."""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import threading
import time


def _identity(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def _descendants(pid: int) -> dict[int, str]:
    children: dict[int, set[int]] = {}
    identities = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            children.setdefault(int(fields[1]), set()).add(int(path.parent.name))
            identities[int(path.parent.name)] = fields[19]
        except (OSError, ValueError, IndexError):
            continue
    found, pending = set(), [pid]
    while pending:
        for child in children.get(pending.pop(), set()) - found:
            found.add(child)
            pending.append(child)
    return {child: identities[child] for child in found}


def run_worker(command: list[str], *, cwd: Path, env: dict, deadline: float, log: Path) -> int:
    limit = 4 * 1024 * 1024
    chunks: list[bytes] = []
    overflow = threading.Event()

    def drain(stream) -> None:
        total = 0
        try:
            while data := stream.read(8192):
                remaining = max(0, limit - total)
                if remaining:
                    chunks.append(data[:remaining])
                total += len(data)
                if total > limit:
                    overflow.set()
        finally:
            stream.close()

    descendants: dict[int, str] = {}
    with subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          start_new_session=True) as process:
        reader = threading.Thread(target=drain, args=(process.stdout,), daemon=True)
        reader.start()
        try:
            while process.poll() is None:
                descendants.update(_descendants(process.pid))
                if time.monotonic() >= deadline:
                    raise TimeoutError("Apex exceeded the invocation timeout")
                if overflow.is_set():
                    raise RuntimeError("Apex exceeded the process output limit")
                time.sleep(0.05)
            returncode = process.returncode
        finally:
            for pid, identity in (descendants | _descendants(process.pid)).items():
                if _identity(pid) != identity:
                    continue
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            reader.join(timeout=5)
            log.write_bytes(b"".join(chunks))
        if reader.is_alive() or overflow.is_set():
            raise RuntimeError("Apex process output did not close within its limits")
        return returncode
