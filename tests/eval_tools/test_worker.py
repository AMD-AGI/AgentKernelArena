from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from src.eval_tools.runtime_client import RuntimeRPCError, UnixSocketRuntimeClient
from src.eval_tools import worker


def _worker(tmp_path: Path):
    roots = {name: tmp_path / name for name in ("input", "scratch", "artifacts")}
    for root in roots.values():
        root.mkdir(parents=True)
    (roots["input"] / "workspace").mkdir()
    socket_path = tmp_path / "sockets" / "gpu_asan.sock"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "src.eval_tools.worker",
            "--tool",
            "gpu_asan",
            "--socket",
            str(socket_path),
            "--input-root",
            str(roots["input"]),
            "--scratch-root",
            str(roots["scratch"]),
            "--artifact-root",
            str(roots["artifacts"]),
            "--max-timeout-s",
            "5",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "AKA_EVAL_TOOL_SKIP_POSITIVE_CONTROL": "1"},
    )
    deadline = time.monotonic() + 5
    while not socket_path.exists() and process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    if not socket_path.exists():
        stdout, stderr = process.communicate(timeout=2)
        raise AssertionError(f"worker did not start: stdout={stdout!r} stderr={stderr!r}")
    return process, roots, UnixSocketRuntimeClient(socket_path, timeout_seconds=10)


def _cleanup(process: subprocess.Popen[str], client: UnixSocketRuntimeClient) -> None:
    if process.poll() is not None:
        return
    try:
        client.shutdown()
        process.wait(timeout=3)
    except Exception:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)


def test_execute_writes_only_relative_artifact_paths(tmp_path: Path) -> None:
    process, roots, client = _worker(tmp_path)
    try:
        response = client.execute(
            {
                "argv": [
                    sys.executable,
                    "-c",
                    "import os,sys; print(os.getcwd()); sys.stderr.write('diagnostic')",
                ],
                "cwd": {"root": "input", "path": "workspace"},
                "artifact_dir": "run/task/gpu_asan",
                "timeout_s": 2,
            }
        )
        execution = response["execution"]
        assert execution["exit_code"] == 0
        assert execution["timed_out"] is False
        assert execution["stdout"]["path"] == "run/task/gpu_asan/stdout.log"
        assert execution["stderr"]["path"] == "run/task/gpu_asan/stderr.log"
        assert str(roots["input"] / "workspace") in (
            roots["artifacts"] / execution["stdout"]["path"]
        ).read_text()
        assert (
            roots["artifacts"] / execution["stderr"]["path"]
        ).read_text() == "diagnostic"
    finally:
        _cleanup(process, client)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("artifact_dir", "../escape"),
        ("artifact_dir", "/absolute"),
        ("cwd", {"root": "input", "path": "../../escape"}),
    ],
)
def test_rejects_path_escape(tmp_path: Path, field: str, value) -> None:
    process, _roots, client = _worker(tmp_path)
    request = {
        "argv": [sys.executable, "-c", "print('must not run')"],
        "cwd": {"root": "input", "path": "workspace"},
        "artifact_dir": "safe",
        "timeout_s": 2,
    }
    request[field] = value
    try:
        with pytest.raises(RuntimeRPCError) as raised:
            client.execute(request)
        assert raised.value.code == "INVALID_PATH"
    finally:
        _cleanup(process, client)


def test_symlink_escape_is_rejected(tmp_path: Path) -> None:
    process, roots, client = _worker(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (roots["artifacts"] / "link").symlink_to(outside, target_is_directory=True)
    try:
        with pytest.raises(RuntimeRPCError) as raised:
            client.execute(
                {
                    "argv": [sys.executable, "-c", "print('must not run')"],
                    "cwd": {"root": "input", "path": "workspace"},
                    "artifact_dir": "link/escape",
                    "timeout_s": 2,
                }
            )
        assert raised.value.code == "INVALID_PATH"
        assert not (outside / "escape").exists()
    finally:
        _cleanup(process, client)


def test_timeout_above_worker_cap_is_rejected(tmp_path: Path) -> None:
    process, _roots, client = _worker(tmp_path)
    try:
        with pytest.raises(RuntimeRPCError) as raised:
            client.execute(
                {
                    "argv": [sys.executable, "-c", "print('must not run')"],
                    "cwd": {"root": "input", "path": "workspace"},
                    "artifact_dir": "safe",
                    "timeout_s": 6,
                }
            )
        assert raised.value.code == "INVALID_REQUEST"
    finally:
        _cleanup(process, client)


@pytest.mark.parametrize("field", ["timeout_s", "term_grace_s", "kill_grace_s"])
def test_non_finite_timing_values_are_rejected(
    tmp_path: Path, field: str
) -> None:
    process, _roots, client = _worker(tmp_path)
    request = {
        "argv": [sys.executable, "-c", "print('must not run')"],
        "cwd": {"root": "input", "path": "workspace"},
        "artifact_dir": "safe",
        "timeout_s": 2,
    }
    request[field] = float("nan")
    try:
        with pytest.raises(RuntimeRPCError) as raised:
            client.execute(request)
        assert raised.value.code == "INVALID_REQUEST"
    finally:
        _cleanup(process, client)


def test_detached_descendant_is_killed_by_worker_containment(tmp_path: Path) -> None:
    process, roots, client = _worker(tmp_path)
    child_code = "import time; time.sleep(60)"
    leader_code = (
        "import subprocess,sys; "
        f"child=subprocess.Popen([sys.executable,'-c',{child_code!r}],"
        "start_new_session=True); "
        "print(child.pid, flush=True)"
    )
    try:
        response = client.execute(
            {
                "argv": [sys.executable, "-c", leader_code],
                "cwd": {"root": "input", "path": "workspace"},
                "artifact_dir": "detached",
                "timeout_s": 2,
                "term_grace_s": 0.2,
                "kill_grace_s": 1,
            }
        )
        execution = response["execution"]
        child_pid = int(
            (roots["artifacts"] / execution["stdout"]["path"])
            .read_text(encoding="utf-8")
            .strip()
        )
        deadline = time.monotonic() + 2
        while Path(f"/proc/{child_pid}").exists() and time.monotonic() < deadline:
            time.sleep(0.02)

        assert execution["cleanup_required"] is True
        assert execution["succeeded"] is False
        assert execution["termination"] in {"sigterm", "sigkill"}
        assert not Path(f"/proc/{child_pid}").exists()
    finally:
        _cleanup(process, client)


def test_hip_fpsan_positive_control_requires_both_processes_to_exit_zero(
    tmp_path: Path, monkeypatch
) -> None:
    calls = iter(
        [
            {"returncode": 0, "_stdout": ""},
            {
                "returncode": 0,
                "_stdout": (
                    'AKA_FPSAN_RESULT {"instrumented": true, '
                    '"reference_digest": "same", "candidate_digest": "same"}'
                ),
            },
            {
                "returncode": 139,
                "_stdout": (
                    'AKA_FPSAN_RESULT {"instrumented": true, '
                    '"reference_digest": "good", "candidate_digest": "bad"}'
                ),
            },
        ]
    )
    monkeypatch.setattr(worker, "_run_probe_step", lambda *args, **kwargs: next(calls))

    result = worker._hip_fpsan_positive(
        {"include_dir": "/opt/hip-fpsan/include"},
        tmp_path / "probes",
        tmp_path / "work",
        tmp_path / "artifacts",
    )

    assert result["passed"] is False
    assert result["steps"]["mismatch"]["returncode"] == 139
