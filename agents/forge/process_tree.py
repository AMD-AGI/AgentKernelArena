"""Reap provider descendants, including subprocesses that create new sessions."""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import os
from pathlib import Path
import signal
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
def managed_children(*, after_stop=None):
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
            if after_stop is not None:
                after_stop()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
