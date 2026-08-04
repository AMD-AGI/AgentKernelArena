from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from src.eval_tools.runtime_client import (
    RuntimeRPCError,
    SidecarRuntimeClient,
    UnixSocketRuntimeClient,
    socket_path_for_tool,
    validate_relative_rpc_path,
)
from src.eval_tools.contracts import (
    ArtifactKind,
    CapabilityState,
    InstrumentationControl,
    KernelLanguage,
    TaskProfile,
    ToolContext,
    ToolInvocation,
)


def _start_worker(
    tmp_path: Path, tool: str = "test-tool", extra_env: dict[str, str] | None = None
):
    input_root = tmp_path / "input"
    scratch_root = tmp_path / "scratch"
    artifact_root = tmp_path / "artifacts"
    socket_path = tmp_path / "sockets" / f"{tool}.sock"
    for path in (input_root, scratch_root, artifact_root):
        path.mkdir(parents=True)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "src.eval_tools.worker",
            "--tool",
            tool,
            "--socket",
            str(socket_path),
            "--input-root",
            str(input_root),
            "--scratch-root",
            str(scratch_root),
            "--artifact-root",
            str(artifact_root),
            "--max-timeout-s",
            "10",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={
            **os.environ,
            "AKA_EVAL_TOOL_SKIP_POSITIVE_CONTROL": "1",
            **(extra_env or {}),
        },
    )
    deadline = time.monotonic() + 5
    while not socket_path.exists() and process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    if not socket_path.exists():
        stdout, stderr = process.communicate(timeout=2)
        raise AssertionError(f"worker did not start: stdout={stdout!r} stderr={stderr!r}")
    return process, socket_path


def _stop_worker(process: subprocess.Popen[str], client: UnixSocketRuntimeClient) -> None:
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


def test_health_round_trip(tmp_path: Path) -> None:
    process, socket_path = _start_worker(tmp_path)
    client = UnixSocketRuntimeClient(socket_path)
    try:
        health = client.health()
        assert health["status"] == "ready"
        assert health["tool"] == "test-tool"
        assert health["protocol_version"] == 1
    finally:
        _stop_worker(process, client)


@pytest.mark.parametrize(
    ("tool", "evidence_keys"),
    [
        ("triton_fpsan", {"triton_version", "triton_fpsan", "triton_asan"}),
        (
            "gpu_asan",
            {
                "asan_runtime_dir",
                "hip_asan_runtime",
                "host_asan_preload",
                "host_asan_lib_dir",
                "normal_rocm_lib_dir",
                "xnack_supported",
                "triton_asan",
            },
        ),
        ("rocjitsu", {"rocjitsu_binary", "config_path", "target_arch"}),
        ("hip_fpsan", {"include_dir", "public_header", "hip_fpsan_headers"}),
    ],
)
def test_health_returns_tool_specific_runtime_evidence(
    tmp_path: Path, tool: str, evidence_keys: set[str]
) -> None:
    process, socket_path = _start_worker(tmp_path, tool=tool)
    client = UnixSocketRuntimeClient(socket_path)
    try:
        health = client.health()
        assert health["status"] == "degraded"
        assert evidence_keys <= set(health["evidence"])
        assert health["evidence"]["positive_control"]["passed"] is False
        assert health["evidence"]["positive_control"]["kind"] == "skipped"
    finally:
        _stop_worker(process, client)


def test_unavailable_socket_has_stable_error_code(tmp_path: Path) -> None:
    client = UnixSocketRuntimeClient(tmp_path / "missing.sock", timeout_seconds=0.1)
    with pytest.raises(RuntimeRPCError) as raised:
        client.health()
    assert raised.value.code == "UNAVAILABLE_RUNTIME"


def test_typed_probe_fails_closed_when_tool_assets_are_missing(tmp_path: Path) -> None:
    process, socket_path = _start_worker(tmp_path, tool="hip_fpsan")
    low_level = UnixSocketRuntimeClient(socket_path)
    typed = SidecarRuntimeClient(
        socket_dir=socket_path.parent,
        scoring_root=tmp_path / "input",
        artifact_scoring_root=tmp_path / "artifacts",
    )
    context = _context(tmp_path / "input", tmp_path / "artifacts" / "tool")
    try:
        capability = typed.probe("hip_fpsan", context)
        assert capability.state == CapabilityState.UNAVAILABLE_RUNTIME
        assert capability.reason_code == "RUNTIME_ASSET_MISSING"
        assert capability.evidence["hip_fpsan_headers"] is False
    finally:
        _stop_worker(process, low_level)


def test_tool_socket_names_and_relative_paths_are_validated(tmp_path: Path) -> None:
    assert socket_path_for_tool("gpu_asan", tmp_path) == tmp_path / "gpu_asan.sock"
    with pytest.raises(ValueError):
        socket_path_for_tool("../../escape", tmp_path)
    assert validate_relative_rpc_path("task/tool", field="artifact_dir") == "task/tool"
    with pytest.raises(ValueError):
        validate_relative_rpc_path("../escape", field="artifact_dir")
    with pytest.raises(ValueError):
        validate_relative_rpc_path("/host/path", field="artifact_dir")


def _context(
    workspace: Path,
    artifact_dir: Path,
    *,
    options: dict | None = None,
    runtime_ref: str | None = None,
) -> ToolContext:
    return ToolContext(
        workspace=str(workspace),
        task_config={},
        profile=TaskProfile(
            task_type="hip",
            language=KernelLanguage.HIP,
            artifact_kind=ArtifactKind.SOURCE_AOT,
            framework="hip",
            instrumentation_control=InstrumentationControl.RECOMPILE,
            adapter=None,
            source_available=True,
        ),
        artifact_dir=str(artifact_dir),
        env={"CONTEXT_VALUE": "context"},
        options=options or {},
        runtime_ref=runtime_ref,
    )


def test_typed_probe_enforces_runtime_ref_and_required_positive_control(
    tmp_path: Path,
) -> None:
    process, socket_path = _start_worker(
        tmp_path,
        tool="test_tool",
        extra_env={"AKA_EVAL_TOOL_RUNTIME_REF": "sha256:actual"},
    )
    low_level = UnixSocketRuntimeClient(socket_path)
    typed = SidecarRuntimeClient(
        socket_dir=socket_path.parent,
        scoring_root=tmp_path / "input",
        artifact_scoring_root=tmp_path / "artifacts",
    )
    try:
        mismatch = typed.probe(
            "test_tool",
            _context(
                tmp_path / "input",
                tmp_path / "artifacts" / "tool",
                runtime_ref="sha256:expected",
            ),
        )
        assert mismatch.reason_code == "RUNTIME_REF_MISMATCH"

        failed_control = typed.probe(
            "test_tool",
            _context(
                tmp_path / "input",
                tmp_path / "artifacts" / "tool",
                runtime_ref="sha256:actual",
                options={"positive_control_required": True},
            ),
        )
        assert failed_control.reason_code == "POSITIVE_CONTROL_FAILED"
        assert failed_control.evidence["positive_control"]["passed"] is False
    finally:
        _stop_worker(process, low_level)


def test_typed_sidecar_client_maps_paths_and_returns_logs(tmp_path: Path) -> None:
    process, socket_path = _start_worker(tmp_path, tool="test_tool")
    input_root = tmp_path / "input"
    workspace = input_root / "task"
    workspace.mkdir()
    artifact_root = tmp_path / "artifacts"
    artifact_dir = artifact_root / "run" / "test_tool"
    client = UnixSocketRuntimeClient(socket_path)
    typed = SidecarRuntimeClient(
        socket_dir=socket_path.parent,
        scoring_root=input_root,
        artifact_scoring_root=artifact_root,
    )
    context = _context(workspace, artifact_dir)
    try:
        capability = typed.probe("test_tool", context)
        assert capability.state == CapabilityState.READY

        record = typed.execute(
            ToolInvocation(
                tool="test_tool",
                command=(
                    sys.executable,
                    "-c",
                    "import os,sys; print(os.environ['CONTEXT_VALUE']); "
                    "sys.stderr.write(os.getcwd())",
                ),
                cwd=str(workspace),
                env={},
                timeout_s=2,
            ),
            context,
        )
        assert record.returncode == 0
        assert record.stdout == "context\n"
        assert record.stderr == str(workspace)
        assert record.metadata["runtime"] == "unix_socket_sidecar"
    finally:
        _stop_worker(process, client)


def test_typed_sidecar_client_rejects_workspace_and_artifact_escape(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    workspace = input_root / "task"
    workspace.mkdir(parents=True)
    artifact_root = tmp_path / "artifacts"
    artifact_dir = artifact_root / "run" / "gpu_asan"
    typed = SidecarRuntimeClient(
        socket_dir=tmp_path / "sockets",
        scoring_root=input_root,
        artifact_scoring_root=artifact_root,
    )
    context = _context(workspace, artifact_dir)

    with pytest.raises(ValueError, match="outside its allowed root"):
        typed.execute(
            ToolInvocation(
                tool="gpu_asan",
                command=(sys.executable, "-c", "pass"),
                cwd=str(input_root),
                timeout_s=1,
            ),
            context,
        )
    with pytest.raises(ValueError, match="outside its allowed root"):
        typed.execute(
            ToolInvocation(
                tool="gpu_asan",
                command=(sys.executable, "-c", "pass"),
                cwd=str(workspace),
                artifact_dir=str(artifact_root / "other-task"),
                timeout_s=1,
            ),
            context,
        )
