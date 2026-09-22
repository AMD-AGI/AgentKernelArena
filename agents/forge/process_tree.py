"""Reap provider descendants, including subprocesses that create new sessions."""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def _descendants(parent: int) -> set[int]:
    children = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            # comm may contain spaces and parentheses; the suffix begins at state.
            fields = path.read_text().rsplit(")", 1)[1].split()
            children.setdefault(int(fields[1]), []).append(int(path.parent.name))
        except (OSError, ValueError, IndexError):
            continue
    result, pending = set(), [parent]
    while pending:
        for child in children.get(pending.pop(), ()):
            if child not in result:
                result.add(child)
                pending.append(child)
    return result


def _stop_children():
    # Kill every captured descendant, not merely our process group: nested
    # forge-loop, task runners and SDKs deliberately use start_new_session.
    deadline = time.monotonic() + 3
    while True:
        children = _descendants(os.getpid())
        for pid in children:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        while True:
            try:
                if os.waitpid(-1, os.WNOHANG)[0] == 0:
                    break
            except ChildProcessError:
                break
        remaining = _descendants(os.getpid())
        if not remaining:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Forge child processes did not exit: {sorted(remaining)}")
        time.sleep(0.05)


@contextmanager
def managed_children():
    """Linux worker scope; adopted orphans stay attributable to this invocation."""
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "Cannot install Forge child subreaper")
    previous = {}
    def terminate(sig, frame):
        raise SystemExit(128 + sig)
    for sig in (signal.SIGTERM, signal.SIGINT):
        previous[sig] = signal.signal(sig, terminate)
    try:
        yield
    finally:
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        try:
            _stop_children()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def main(argv=None):
    """Supervise a command without importing or modifying the engine.

    Keep this process alive around the CLI: exec would discard the signal
    handler, and a plain process-group kill misses children that call setsid().
    The subreaper also adopts children left behind when the CLI exits normally.
    """
    command = list(sys.argv[1:] if argv is None else argv)
    if not command:
        raise SystemExit("Usage: process_tree.py COMMAND [ARG ...]")
    process = None
    try:
        with managed_children():
            # Inherit cwd, environment and streams; preserve the native argv and
            # interpreter. No shell, extra mounts or provider state is involved.
            process = subprocess.Popen(command, shell=False)
            returncode = process.wait()
    finally:
        if process is not None:
            process.poll()  # managed_children may already have reaped the CLI.
    if returncode < 0:
        # Preserve signal termination after cleaning the descendants, rather than
        # turning a native -SIGTERM into Python's modulo-256 exit status.
        sig = -returncode
        if sig != signal.SIGKILL:
            signal.signal(sig, signal.SIG_DFL)
        os.kill(os.getpid(), sig)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
