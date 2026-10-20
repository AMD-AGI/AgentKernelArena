"""CPU checks for the pre-measurement process sweep; no GPU or container is used."""
import subprocess
import sys
from types import SimpleNamespace

import pytest

from src import process_control as pc


def test_disappeared_darwin_group_is_checked_before_ignoring_eperm(monkeypatch):
    monkeypatch.setattr(pc.sys, "platform", "darwin")
    waited = []
    process = SimpleNamespace(pid=4242, wait=lambda **kwargs: waited.append(kwargs))

    def denied(*args):
        raise PermissionError("group disappeared")

    def groups(command, **kwargs):
        assert command == ["/bin/ps", "-axo", "pgid="]
        assert kwargs["timeout"] == 1
        return SimpleNamespace(returncode=0, stdout="1\n 14242\n")

    monkeypatch.setattr(pc.os, "killpg", denied)
    monkeypatch.setattr(pc.subprocess, "run", groups)
    pc.stop_process_group(process)
    assert waited == [{}]


@pytest.mark.parametrize("platform,returncode,stdout", [
    ("darwin", 0, "1\n4242\n"), ("darwin", 1, ""), ("darwin", 0, ""),
    ("darwin", 0, "invalid"), ("linux", 0, ""),
])
def test_group_permission_errors_are_not_hidden_without_proven_absence(
        monkeypatch, platform, returncode, stdout):
    monkeypatch.setattr(pc.sys, "platform", platform)
    process = SimpleNamespace(pid=4242, wait=lambda **kwargs: None)

    def denied(*args):
        raise PermissionError("cannot signal group")

    monkeypatch.setattr(pc.os, "killpg", denied)
    monkeypatch.setattr(pc.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=returncode, stdout=stdout))
    with pytest.raises(PermissionError):
        pc.stop_process_group(process)


def fake_proc(root, rows):
    for pid, ppid, comm, state, cmdline in rows:
        entry = root / str(pid)
        entry.mkdir()
        (entry / "stat").write_text(f"{pid} ({comm}) {state} {ppid} {pid} {pid} 0 -1 4194560 0\n")
        (entry / "cmdline").write_text(cmdline.replace(" ", "\0") + ("\0" if cmdline else ""))
    (root / "self").mkdir()
    (root / "cpuinfo").write_text("")
    return root


def test_ancestry_is_protected_and_orphans_are_foreign(tmp_path):
    proc = fake_proc(tmp_path, [
        (1, 0, "python", "S", "python main.py"),
        (40, 1, "claude", "S", "claude --print"),
        (41, 40, "python", "S", "python -m sglang.launch_server"),
        (77, 1, "sleep", "S", "sleep 100"),
        (78, 1, "python", "Z", ""),
        (90, 1, "bash (login)", "S", "bash"),
    ])
    rows = pc.list_foreign_processes(proc, self_pid=40)
    assert [row["pid"] for row in rows] == [41, 77, 90]
    assert rows[0]["cmdline"] == "python -m sglang.launch_server"
    assert rows[2]["comm"] == "bash (login)"
    # The namespace init is protected even when the caller is not its descendant.
    assert 1 not in {row["pid"] for row in pc.list_foreign_processes(proc, self_pid=77)}


def test_sweep_refuses_outside_dedicated_container(monkeypatch, tmp_path):
    monkeypatch.delenv(pc.CONTAINER_MARKER, raising=False)
    with pytest.raises(RuntimeError, match="dedicated worker container"):
        pc.sweep_foreign_processes(proc=fake_proc(tmp_path, [(1, 0, "python", "S", "python main.py")]))


def test_sweep_records_terminated_processes(monkeypatch, tmp_path):
    monkeypatch.setenv(pc.CONTAINER_MARKER, "1")
    proc = fake_proc(tmp_path, [(1, 0, "python", "S", "python main.py"), (4242, 1, "sleep", "S", "sleep 9")])
    signalled = []

    def fake_kill(pid, sig):
        signalled.append((pid, sig))
        raise ProcessLookupError

    monkeypatch.setattr(pc.os, "kill", fake_kill)
    record = pc.sweep_foreign_processes(proc=proc, grace_s=0.1)
    assert record["status"] == "clean"
    assert [row["pid"] for row in record["terminated"]] == [4242]
    assert record["survivors"] == []
    assert signalled == [(4242, pc.signal.SIGTERM)]


def test_terminate_processes_stops_a_detached_child():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    try:
        assert pc.terminate_processes([child.pid], grace_s=5) == []
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
    assert pc.terminate_processes([child.pid], grace_s=0.1) == []
