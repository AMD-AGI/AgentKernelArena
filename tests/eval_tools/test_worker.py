from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.eval_tools import worker
from src.eval_tools.plugins.base import FINDING, PASS
from src.eval_tools.runtime_client import RuntimeRPCError, UnixSocketRuntimeClient


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


def _waitcheck_result_line(
    sha256: str, *, passed: bool, diagnostic: bool
) -> str:
    diagnostics = []
    if diagnostic:
        diagnostics.append(
            {
                "code": "wait-counter",
                "kernel_name": "waitcheck_probe_kernel",
                "kernel_entry": 0,
                "section_name": ".text",
                "section_offset": 8,
                "message": "missing s_waitcnt lgkmcnt(0)",
            }
        )
    payload = {
        "schema_version": 1,
        "code_object_sha256": sha256,
        "target": "gfx950",
        "expected_kernel": "waitcheck_probe_kernel",
        "kernel_entry": 0,
        "inventory_attested": True,
        "api_status": 0,
        "analysis_complete": True,
        "instructions_analyzed": 4,
        "memory_events_tracked": 1,
        "kernels_discovered": 1,
        "kernels_analyzed": 1,
        "diagnostics_observed": len(diagnostics),
        "diagnostics_reported": len(diagnostics),
        "diagnostics_truncated": False,
        "stopped_early": False,
        "passed": passed,
        "diagnostics": diagnostics,
    }
    return "AKA_WAITCHECK_RESULT " + worker.json.dumps(payload)


def test_waitcheck_positive_control_runs_production_capi_and_parser(
    tmp_path: Path, monkeypatch
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    safe = work / "waitcheck-safe.hsaco"
    hazard = work / "waitcheck-hazard.hsaco"
    safe.write_bytes(b"safe")
    hazard.write_bytes(b"hazard")
    calls: dict[str, tuple[list[str], dict[str, str]]] = {}

    def run_step(name, argv, *, environment, **_kwargs):
        calls[name] = (argv, dict(environment))
        output = ""
        returncode = 0
        if name == "inventory-safe":
            output = worker.json.dumps(
                {
                    "kind": "kernel",
                    "kernel_name": "waitcheck_probe_kernel",
                    "kernel_entry": 0,
                }
            )
        elif name == "safe-production":
            output = _waitcheck_result_line(
                worker._sha256_file(safe), passed=True, diagnostic=False
            )
        elif name == "hazard-production":
            output = _waitcheck_result_line(
                worker._sha256_file(hazard), passed=False, diagnostic=True
            )
        elif name == "hazard-cli":
            output = "missing s_waitcnt lgkmcnt(0)"
            returncode = 4
        return {
            "command": argv,
            "returncode": returncode,
            "_stdout": output,
            "_stderr": "",
        }

    monkeypatch.setattr(worker, "_run_probe_step", run_step)
    result = worker._waitcheck_positive(
        {
            "waitcheck_binary": "/opt/rocjitsu/bin/rj_waitcheck",
            "waitcheck_capi_wrapper": "/opt/rocjitsu/bin/aka-waitcheck-capi",
        },
        tmp_path / "framework" / "probes",
        work,
        tmp_path / "artifacts",
    )

    assert result["passed"] is True
    safe_command = calls["safe-production"][0]
    assert any(value.endswith("waitcheck_entrypoint.py") for value in safe_command)
    assert "/opt/rocjitsu/bin/aka-waitcheck-capi" in safe_command
    assert "hazard-production" in calls
    assert calls["hazard-cli"][0][0] == "/opt/rocjitsu/bin/rj_waitcheck"


def test_waitcheck_positive_control_rejects_broken_production_entrypoint(
    tmp_path: Path, monkeypatch
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    (work / "waitcheck-safe.hsaco").write_bytes(b"safe")
    (work / "waitcheck-hazard.hsaco").write_bytes(b"hazard")

    def run_step(name, argv, **_kwargs):
        output = ""
        returncode = 0
        if name == "inventory-safe":
            output = worker.json.dumps(
                {
                    "kernel_name": "waitcheck_probe_kernel",
                    "kernel_entry": 0,
                }
            )
        elif name == "safe-production":
            returncode = 2
        elif name == "hazard-cli":
            output = "missing wait"
            returncode = 4
        return {
            "command": argv,
            "returncode": returncode,
            "_stdout": output,
            "_stderr": "",
        }

    monkeypatch.setattr(worker, "_run_probe_step", run_step)
    result = worker._waitcheck_positive(
        {
            "waitcheck_binary": "/opt/rocjitsu/bin/rj_waitcheck",
            "waitcheck_capi_wrapper": "/opt/rocjitsu/bin/aka-waitcheck-capi",
        },
        tmp_path / "framework" / "probes",
        work,
        tmp_path / "artifacts",
    )

    assert result["passed"] is False


def test_consan_positive_control_runs_production_entrypoint_and_oracle_split(
    tmp_path: Path, monkeypatch
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    safe = work / "consan-safe.hsaco"
    racy = work / "consan-racy.hsaco"
    safe.write_bytes(b"safe")
    racy.write_bytes(b"racy")
    calls: dict[str, tuple[list[str], dict[str, str]]] = {}

    def run_step(name, argv, *, environment, **_kwargs):
        calls[name] = (argv, dict(environment))
        output = (
            "CONSAN_ORACLE_ENV_CLEAN\nAKA_CONSAN_RUN {}"
            if name.endswith("-production")
            else ""
        )
        return {
            "command": argv,
            "returncode": 0,
            "_stdout": output,
            "_stderr": "",
        }

    parsed_statuses = iter((PASS, FINDING))
    monkeypatch.setattr(worker, "_run_probe_step", run_step)
    monkeypatch.setattr(
        worker,
        "parse_consan",
        lambda *_args, **_kwargs: SimpleNamespace(status=next(parsed_statuses)),
    )
    result = worker._consan_positive(
        {"consan_hook": "/opt/rocjitsu/lib/librocjitsu_dbi_hooks.so"},
        tmp_path / "framework" / "probes",
        work,
        tmp_path / "artifacts",
    )

    assert result["passed"] is True
    for name in ("safe-production", "racy-production"):
        command, environment = calls[name]
        assert any(value.endswith("consan_entrypoint.py") for value in command)
        assert sum(value.startswith("--command-arg=") for value in command) == 4
        assert sum(value.startswith("--oracle-arg=") for value in command) == 4
        assert "HSA_TOOLS_LIB" not in environment
        assert not any(key.startswith("RJ_CONSAN_") for key in environment)
