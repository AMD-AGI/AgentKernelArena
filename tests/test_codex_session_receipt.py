# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only contract tests for direct Codex process and attempt evidence."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import signal
import stat
import tempfile
import time
from pathlib import Path

import pytest


launcher = importlib.import_module("agents.codex.launch_agent")


_FAKE_CODEX = r'''#!/usr/bin/env python3
import json
import os
import signal
import subprocess
import sys
import time

if "--version" in sys.argv:
    print("codex-cli 9.9.9-test")
    raise SystemExit(0)

mode = os.environ.get("FAKE_CODEX_MODE", "success")
print(json.dumps({"type": "thread.started", "thread_id": "thread-test-123"}), flush=True)
print(json.dumps({"type": "session.started", "session_id": "session-test-456"}), flush=True)
print(json.dumps({"type": "probe", "pid": os.getpid(), "pgid": os.getpgrp()}), flush=True)
print(json.dumps({
    "type": "isolation.probe",
    "home": os.environ.get("HOME"),
    "codex_home": os.environ.get("CODEX_HOME"),
    "dont_write_bytecode": os.environ.get("PYTHONDONTWRITEBYTECODE") == "1",
    "forbidden_exists": os.path.exists(os.environ.get("FAKE_FORBIDDEN_PATH", "/missing")),
}), flush=True)

if mode == "temporary_workspace_changes":
    with open("kernel.py", "w", encoding="utf-8") as stream:
        stream.write("optimized = True\n")
    os.makedirs("build", exist_ok=True)
    with open("build/compile_report.json", "w", encoding="utf-8") as stream:
        stream.write('{"temporary":true}\n')
    os.makedirs("__pycache__", exist_ok=True)
    with open("__pycache__/kernel.pyc", "wb") as stream:
        stream.write(b"temporary")

if mode == "turn_limit":
    for index in range(51):
        print(json.dumps({
            "type": "item.completed",
            "item": {"type": "agent_message", "text": f"turn-{index}"},
        }), flush=True)
    time.sleep(30)

if mode == "exact_boundary":
    with open("kernel.py", "w", encoding="utf-8") as stream:
        stream.write("optimized_at_boundary = True\n")
    for index in range(50):
        print(json.dumps({
            "type": "item.completed",
            "item": {"type": "agent_message", "text": f"turn-{index}"},
        }), flush=True)
    time.sleep(30)

if mode == "exact_boundary_late_write":
    with open("kernel.py", "w", encoding="utf-8") as stream:
        stream.write("optimized_at_boundary = True\n")

    def late_write(_signum, _frame):
        with open("kernel.py", "w", encoding="utf-8") as stream:
            stream.write("poisoned_after_boundary = True\n")
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, late_write)
    for index in range(50):
        print(json.dumps({
            "type": "item.completed",
            "item": {"type": "agent_message", "text": f"turn-{index}"},
        }), flush=True)
    time.sleep(30)

if mode == "exact_boundary_natural_exit":
    with open("kernel.py", "w", encoding="utf-8") as stream:
        stream.write("optimized_natural_exit = True\n")
    for index in range(50):
        print(json.dumps({
            "type": "item.completed",
            "item": {"type": "agent_message", "text": f"turn-{index}"},
        }), flush=True)
    raise SystemExit(0)

if mode == "exact_boundary_setsid_double_fork":
    child_code = r"""
import os, sys
os.setsid()
child = os.fork()
if child:
    os._exit(0)
late_code = (
    "import os, signal, sys, time\n"
    "for descriptor in (0, 1, 2):\n"
    "    try:\n"
    "        os.close(descriptor)\n"
    "    except OSError:\n"
    "        pass\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "time.sleep(0.5)\n"
    "with open(sys.argv[1], 'w', encoding='utf-8') as stream:\n"
    "    stream.write('poisoned_by_env_i_double_fork = True\\\\n')\n"
)
os.execve(sys.executable, [sys.executable, "-c", late_code, os.path.abspath("kernel.py")], {})
"""
    subprocess.Popen([sys.executable, "-c", child_code])
    time.sleep(0.1)
    with open("kernel.py", "w", encoding="utf-8") as stream:
        stream.write("optimized_with_descendant = True\n")
    for index in range(50):
        print(json.dumps({
            "type": "item.completed",
            "item": {"type": "agent_message", "text": f"turn-{index}"},
        }), flush=True)
    time.sleep(30)

if mode == "output_flood":
    payload = "x" * 1024
    for index in range(17000):
        print(json.dumps({"type": "probe.output", "index": index, "payload": payload}))

if mode == "timeout":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    child_code = (
        "import signal,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "time.sleep(30)"
    )
    child = subprocess.Popen([sys.executable, "-c", child_code])
    print(json.dumps({"type": "child", "pid": child.pid}), flush=True)
    time.sleep(30)

print(json.dumps({
    "type": "item.completed",
    "item": {"type": "agent_message", "text": "optimized"},
}), flush=True)
print(json.dumps({
    "type": "turn.completed",
    "usage": {"input_tokens": 11, "cached_input_tokens": 2, "output_tokens": 3},
}), flush=True)
print(json.dumps({
    "type": "turn.completed",
    "usage": {"input_tokens": 7, "cached_input_tokens": 1, "output_tokens": 5},
}), flush=True)
sys.stderr.write("raw warning\n")
sys.stderr.flush()
raise SystemExit(7 if mode == "nonzero" else 0)
'''


def _install_fake_codex(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    binary = binary_dir / "codex"
    binary.write_text(_FAKE_CODEX, encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", f"{binary_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setattr(
        "src.campaign_isolation._codex_requirements_identity",
        lambda: (Path("/etc/codex/requirements.toml"), {"sha256": "f" * 64}),
    )
    monkeypatch.setattr(
        launcher,
        "load_prompt_builder",
        lambda _agent_type, _logger: (
            lambda _task_config, _workspace, _eval_config, _inner_logger: "test prompt"
        ),
    )
    monkeypatch.setattr(
        launcher,
        "formal_gpu_evidence",
        lambda _config: {
            "policy": "physical_device_boundary_with_host_exclusivity_v1",
            "plan_sha256": "a" * 64,
            "boundary_receipt_sha256": "b" * 64,
            "exclusivity_receipt_sha256": "c" * 64,
            "exclusivity_verified": True,
            "host_gpu_id": "0",
            "unique_id": "0x0000000000000001",
            "allowed_render_nodes": ["/dev/dri/renderD128"],
            "observed_devices": [],
        },
    )
    return binary


def _attempt_config(receipt: Path, timeout_seconds: float = 30.0) -> dict:
    return {
        "campaign_attempt": {
            "receipt_path": str(receipt),
            "task_deadline_monotonic": launcher.time.monotonic() + timeout_seconds,
        }
    }


def _load_receipt(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_bytes(receipt: dict, name: str) -> bytes:
    return Path(receipt["artifacts"][name]["path"]).read_bytes()


def _make_artifact_dir_removable(receipt: dict) -> None:
    artifact = Path(receipt["artifacts"]["raw_stdout"]["path"])
    artifact.parent.chmod(0o700)


def test_success_receipt_binds_new_session_cli_usage_and_external_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = _install_fake_codex(tmp_path, monkeypatch)
    workspace = tmp_path / "scored_workspace"
    workspace.mkdir()
    receipt_path = tmp_path / "attempt" / "session_receipt.json"

    output = launcher.launch_agent(
        _attempt_config(receipt_path), str(tmp_path / "task.yaml"), str(workspace)
    )

    receipt = _load_receipt(receipt_path)
    try:
        assert receipt["schema"] == "agentkernelarena.codex-attempt-receipt/v3"
        assert receipt["comparison_contract_sha256"] is None
        assert receipt["session_succeeded"] is True
        assert receipt["timed_out"] is False
        assert receipt["exit_code"] == 0
        assert receipt["process_group_cleanup"]["verification_performed"] is True
        assert receipt["process_group_cleanup"]["verified_absent"] is True
        assert receipt["thread_id"] == "thread-test-123"
        assert receipt["session_id"] == "session-test-456"
        assert receipt["codex"]["version"] == "codex-cli 9.9.9-test"
        assert receipt["codex"]["model"] == "gpt-5.5"
        assert receipt["codex"]["effort"] == "xhigh"
        assert receipt["codex"]["binary_sha256"] == hashlib.sha256(
            binary.read_bytes()
        ).hexdigest()
        assert receipt["invocation"]["isolation"] == {
            "approval": "never_via_strict_config",
            "execpolicy_rules": "ignored",
            "project_instructions": "backend_default_may_load",
            "sandbox": "workspace-write",
            "session": "ephemeral",
            "user_config": "ignored",
            "mount_scope": "ordinary_run",
        }
        invocation = receipt["invocation"]["argv_without_prompt"]
        assert "--strict-config" in invocation
        assert "--ignore-user-config" in invocation
        assert "--ignore-rules" in invocation
        assert "--ephemeral" in invocation
        assert "--sandbox" in invocation
        assert invocation[invocation.index("--sandbox") + 1] == "workspace-write"
        assert "--dangerously-bypass-approvals-and-sandbox" not in invocation
        assert receipt["aggregated_usage"] == {
            "events": 2,
            "input_tokens": 18,
            "cached_input_tokens": 3,
            "output_tokens": 8,
        }

        raw_stdout = _artifact_bytes(receipt, "raw_stdout")
        probe = next(
            json.loads(line)
            for line in raw_stdout.decode("utf-8").splitlines()
            if json.loads(line).get("type") == "probe"
        )
        assert probe["pid"] == probe["pgid"], "Codex must start a new process session"
        assert _artifact_bytes(receipt, "raw_stderr") == b"raw warning\n"
        assert _artifact_bytes(receipt, "formatted_transcript").decode() == output
        rendered_prompt = _artifact_bytes(receipt, "rendered_prompt")
        assert rendered_prompt.startswith(b"test prompt")
        assert receipt["invocation"]["prompt_sha256"] == hashlib.sha256(
            rendered_prompt
        ).hexdigest()
        assert "assistant: optimized" in output
        assert "=== STDERR ===\nraw warning" in output

        for evidence in receipt["artifacts"].values():
            artifact_path = Path(evidence["path"])
            assert workspace not in artifact_path.parents
            assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == evidence["sha256"]
            assert stat.S_IMODE(artifact_path.stat().st_mode) == 0o444
        assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o444
        assert stat.S_IMODE(
            Path(receipt["artifacts"]["raw_stdout"]["path"]).parent.stat().st_mode
        ) == 0o555
    finally:
        _make_artifact_dir_removable(receipt)


def test_nonzero_exit_writes_failure_receipt_then_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_codex(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_CODEX_MODE", "nonzero")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    receipt_path = tmp_path / "evidence" / "session_receipt.json"

    with pytest.raises(launcher.CodexSessionError, match="status 7"):
        launcher.launch_agent(
            _attempt_config(receipt_path), str(tmp_path / "task.yaml"), str(workspace)
        )

    receipt = _load_receipt(receipt_path)
    try:
        assert receipt["session_succeeded"] is False
        assert receipt["timed_out"] is False
        assert receipt["exit_code"] == 7
        assert receipt["capture"]["readers_completed"] is True
    finally:
        _make_artifact_dir_removable(receipt)


def test_formal_session_uses_auth_only_home_and_cannot_see_sibling_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary_parent = Path(tempfile.mkdtemp(prefix="aka-codex-test-", dir="/var/tmp"))
    _install_fake_codex(binary_parent, monkeypatch)
    data_root = tmp_path / "campaign"
    attempt = data_root / "run/task/attempt_01"
    workspace = attempt / "workspace"
    sibling = data_root / "run/task/attempt_02"
    workspace.mkdir(parents=True)
    (workspace / "kernel.py").write_text("baseline = True\n", encoding="utf-8")
    task_config = tmp_path / "task.yaml"
    task_config.write_text("source_file_path: kernel.py\n", encoding="utf-8")
    sibling.mkdir(parents=True)
    forbidden = sibling / "prior-result.json"
    forbidden.write_text("{}\n", encoding="utf-8")
    state_root = tmp_path / "agent-state/.codex"
    state_root.mkdir(parents=True)
    (state_root / "auth.json").write_text('{"token":"fixture"}\n', encoding="utf-8")
    (state_root / "history.jsonl").write_text("prior context\n", encoding="utf-8")
    monkeypatch.setenv("AGENT_STATE_MOUNT_ROOT", str(state_root.parent))
    monkeypatch.setenv("AGENT_KERNEL_ARENA_CAMPAIGN_DATA_ROOT", str(data_root))
    monkeypatch.setenv("FAKE_FORBIDDEN_PATH", str(forbidden))
    receipt_path = attempt / "session_receipt.json"
    eval_config = {
        "campaign": {"comparison": "apex_vs_codex"},
        "campaign_attempt": {
            "fresh_session": True,
            "receipt_path": str(receipt_path),
            "task_deadline_monotonic": launcher.time.monotonic() + 30,
            "comparison_contract_sha256": "d" * 64,
        },
    }

    receipt: dict = {}
    try:
        launcher.launch_agent(eval_config, str(task_config), str(workspace))
        receipt = _load_receipt(receipt_path)
        events = [
            json.loads(line)
            for line in _artifact_bytes(receipt, "raw_stdout").decode().splitlines()
        ]
        probe = next(event for event in events if event.get("type") == "isolation.probe")
        attempt_home = attempt / ".agent-home"
        assert probe == {
            "type": "isolation.probe",
            "home": str(attempt_home),
            "codex_home": str(attempt_home / ".codex"),
            "dont_write_bytecode": True,
            "forbidden_exists": False,
        }
        assert (attempt_home / ".codex/auth.json").is_file()
        assert not (attempt_home / ".codex/history.jsonl").exists()
        assert receipt["invocation"]["isolation"]["mount_scope"] == (
            "attempt_only_bubblewrap"
        )
        assert receipt["turn_budget"] == {
            "policy": "structured_agent_turn_checkpoint_v2",
            "max_turns": 50,
            "observed_turns": 1,
            "exact_boundary_reached": False,
            "post_boundary_turns": 0,
            "budget_exceeded": False,
            "enforcement_failed": False,
            "stop_reason": None,
        }
        assert receipt["workspace_integrity"]["passed"] is True
        assert receipt["comparison_contract_sha256"] == "d" * 64
        assert set(receipt["artifacts"]) == {
            "rendered_prompt",
            "raw_stdout",
            "raw_stderr",
            "formatted_transcript",
            "workspace_before_manifest",
            "workspace_after_manifest",
        }
    finally:
        if receipt:
            _make_artifact_dir_removable(receipt)
        shutil.rmtree(binary_parent)


def test_formal_session_rejects_missing_comparison_contract_digest() -> None:
    with pytest.raises(
        launcher.CodexSessionError,
        match="lacks a valid comparison contract digest",
    ):
        launcher._comparison_contract_sha256({}, formal_campaign=True)


def test_formal_workspace_is_sanitized_to_declared_source_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary_parent = Path(tempfile.mkdtemp(prefix="aka-codex-test-", dir="/var/tmp"))
    _install_fake_codex(binary_parent, monkeypatch)
    monkeypatch.setenv("FAKE_CODEX_MODE", "temporary_workspace_changes")
    data_root = tmp_path / "campaign"
    attempt = data_root / "run/task/attempt_01"
    workspace = attempt / "workspace"
    workspace.mkdir(parents=True)
    source = workspace / "kernel.py"
    source.write_text("baseline = True\n", encoding="utf-8")
    build = workspace / "build"
    build.mkdir()
    report = build / "compile_report.json"
    report.write_text('{"baseline":true}\n', encoding="utf-8")
    task_config = tmp_path / "task.yaml"
    task_config.write_text("source_file_path: kernel.py\n", encoding="utf-8")
    state_root = tmp_path / "agent-state/.codex"
    state_root.mkdir(parents=True)
    (state_root / "auth.json").write_text('{}\n', encoding="utf-8")
    monkeypatch.setenv("AGENT_STATE_MOUNT_ROOT", str(state_root.parent))
    monkeypatch.setenv("AGENT_KERNEL_ARENA_CAMPAIGN_DATA_ROOT", str(data_root))
    receipt_path = attempt / "session_receipt.json"
    eval_config = {
        "campaign": {"comparison": "apex_vs_codex"},
        "campaign_attempt": {
            "fresh_session": True,
            "receipt_path": str(receipt_path),
            "task_deadline_monotonic": time.monotonic() + 30,
            "comparison_contract_sha256": "d" * 64,
        },
    }

    receipt: dict = {}
    try:
        launcher.launch_agent(eval_config, str(task_config), str(workspace))
        receipt = _load_receipt(receipt_path)
        integrity = receipt["workspace_integrity"]
        assert integrity["passed"] is True
        assert integrity["raw_changes"]["created_files"] == ["__pycache__/kernel.pyc"]
        assert integrity["raw_changes"]["unauthorized_changed_files"] == [
            "build/compile_report.json"
        ]
        assert integrity["final_changes"]["changed_files"] == ["kernel.py"]
        assert source.read_text(encoding="utf-8") == "optimized = True\n"
        assert report.read_text(encoding="utf-8") == '{"baseline":true}\n'
        assert not (workspace / "__pycache__").exists()
    finally:
        if receipt:
            _make_artifact_dir_removable(receipt)
        shutil.rmtree(binary_parent)


def test_direct_codex_rejects_post_boundary_turn_without_reporting_fifty_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_codex(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_CODEX_MODE", "turn_limit")
    monkeypatch.setattr(launcher, "_TERM_GRACE_SECONDS", 0.1)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    receipt_path = tmp_path / "evidence" / "session_receipt.json"

    with pytest.raises(launcher.CodexSessionError, match="turn budget"):
        launcher.launch_agent(
            _attempt_config(receipt_path), str(tmp_path / "task.yaml"), str(workspace)
        )

    receipt = _load_receipt(receipt_path)
    try:
        assert receipt["turn_budget"]["max_turns"] == 50
        assert receipt["turn_budget"]["observed_turns"] == 50
        assert receipt["turn_budget"]["post_boundary_turns"] == 1
        assert receipt["turn_budget"]["budget_exceeded"] is True
        assert receipt["session_succeeded"] is False
    finally:
        _make_artifact_dir_removable(receipt)


def test_formal_direct_codex_persists_exact_boundary_source_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary_parent = Path(tempfile.mkdtemp(prefix="aka-codex-test-", dir="/var/tmp"))
    _install_fake_codex(binary_parent, monkeypatch)
    monkeypatch.setenv("FAKE_CODEX_MODE", "exact_boundary")
    monkeypatch.setattr(launcher, "_TERM_GRACE_SECONDS", 0.1)
    data_root = tmp_path / "campaign"
    attempt = data_root / "run/task/attempt_01"
    workspace = attempt / "workspace"
    workspace.mkdir(parents=True)
    source = workspace / "kernel.py"
    source.write_text("baseline = True\n", encoding="utf-8")
    task_config = tmp_path / "task.yaml"
    task_config.write_text("source_file_path: kernel.py\n", encoding="utf-8")
    state_root = tmp_path / "agent-state/.codex"
    state_root.mkdir(parents=True)
    (state_root / "auth.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("AGENT_STATE_MOUNT_ROOT", str(state_root.parent))
    monkeypatch.setenv("AGENT_KERNEL_ARENA_CAMPAIGN_DATA_ROOT", str(data_root))
    receipt_path = attempt / "session_receipt.json"
    eval_config = {
        "campaign": {"comparison": "apex_vs_codex"},
        "campaign_attempt": {
            "fresh_session": True,
            "receipt_path": str(receipt_path),
            "task_deadline_monotonic": time.monotonic() + 30,
            "comparison_contract_sha256": "d" * 64,
        },
    }

    receipt: dict = {}
    try:
        launcher.launch_agent(eval_config, str(task_config), str(workspace))
        receipt = _load_receipt(receipt_path)
        assert receipt["session_succeeded"] is True
        assert receipt["schema"] == "agentkernelarena.codex-attempt-receipt/v4"
        assert receipt["exit_code"] == 128 + int(signal.SIGKILL)
        assert receipt["turn_budget"] == {
            "policy": "structured_agent_turn_checkpoint_v2",
            "max_turns": 50,
            "observed_turns": 50,
            "exact_boundary_reached": True,
            "post_boundary_turns": 0,
            "budget_exceeded": False,
            "enforcement_failed": False,
            "stop_reason": "exact_turn_boundary",
        }
        persistence = receipt["candidate_persistence"]
        assert persistence["policy_id"] == "structured_agent_turn_checkpoint_v2"
        assert persistence["schema"] == "aka.candidate-persistence-receipt/v4"
        assert persistence["attempt_containment_policy_id"] == (
            "private_pid_namespace_init_pidfd_v1"
        )
        assert persistence["termination"] == "exact_turn_boundary"
        checkpoint = persistence["checkpoint"]
        assert checkpoint["before_manifest_sha256"] == receipt[
            "workspace_integrity"
        ]["final_changes"]["before_manifest_sha256"]
        assert checkpoint["after_manifest_sha256"] == receipt[
            "workspace_integrity"
        ]["final_changes"]["after_manifest_sha256"]
        assert checkpoint["changed_files"] == ["kernel.py"]
        assert checkpoint["editable_files"] == ["kernel.py"]
        assert len(checkpoint["attempt_cleanup_sha256"]) == 64
        assert len(checkpoint["boundary_snapshot_sha256"]) == 64
        assert len(checkpoint["output_tail_sha256"]) == 64
        cleanup = receipt["attempt_process_cleanup"]
        assert cleanup["teardown_mode"] == "pidfd_sigkill"
        assert cleanup["sigkill_sent"] is True
        assert cleanup["namespace_init_exit_verified"] is True
        assert cleanup["namespace_membership_enumeration_completed"] is True
        assert cleanup["namespace_membership_inaccessible_entries_count"] >= 0
        assert cleanup["namespace_membership_scan_complete"] is (
            cleanup["namespace_membership_inaccessible_entries_count"] == 0
        )
        assert cleanup["live_visible_namespace_members_after"] == []
        assert cleanup["verified_absent"] is True
        assert receipt["capture"]["readers_completed"] is True
        assert source.read_text(encoding="utf-8") == "optimized_at_boundary = True\n"
    finally:
        if receipt:
            _make_artifact_dir_removable(receipt)
        shutil.rmtree(binary_parent)


def test_exact_boundary_snapshot_excludes_deterministic_late_cleanup_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary_parent = Path(tempfile.mkdtemp(prefix="aka-codex-test-", dir="/var/tmp"))
    _install_fake_codex(binary_parent, monkeypatch)
    monkeypatch.setenv("FAKE_CODEX_MODE", "exact_boundary_late_write")
    monkeypatch.setattr(launcher, "_TERM_GRACE_SECONDS", 0.2)
    data_root = tmp_path / "campaign"
    attempt = data_root / "run/task/attempt_01"
    workspace = attempt / "workspace"
    workspace.mkdir(parents=True)
    source = workspace / "kernel.py"
    source.write_text("baseline = True\n", encoding="utf-8")
    task_config = tmp_path / "task.yaml"
    task_config.write_text("source_file_path: kernel.py\n", encoding="utf-8")
    state_root = tmp_path / "agent-state/.codex"
    state_root.mkdir(parents=True)
    (state_root / "auth.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("AGENT_STATE_MOUNT_ROOT", str(state_root.parent))
    monkeypatch.setenv("AGENT_KERNEL_ARENA_CAMPAIGN_DATA_ROOT", str(data_root))
    receipt_path = attempt / "session_receipt.json"
    eval_config = {
        "campaign": {"comparison": "apex_vs_codex"},
        "campaign_attempt": {
            "fresh_session": True,
            "receipt_path": str(receipt_path),
            "task_deadline_monotonic": time.monotonic() + 30,
            "comparison_contract_sha256": "d" * 64,
        },
    }

    receipt: dict = {}
    try:
        launcher.launch_agent(eval_config, str(task_config), str(workspace))
        receipt = _load_receipt(receipt_path)
        assert receipt["exit_code"] == 128 + int(signal.SIGKILL)
        assert receipt["attempt_process_cleanup"]["sigkill_sent"] is True
        assert receipt["attempt_process_cleanup"]["verified_absent"] is True
        assert receipt["candidate_persistence"]["boundary_snapshot"][
            "complete"
        ] is True
        assert source.read_text(encoding="utf-8") == (
            "optimized_at_boundary = True\n"
        )
        assert "poisoned_after_boundary" not in source.read_text(encoding="utf-8")
    finally:
        if receipt:
            _make_artifact_dir_removable(receipt)
        shutil.rmtree(binary_parent)


def _formal_boundary_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mode: str,
) -> tuple[dict, Path, Path]:
    binary_parent = Path(tempfile.mkdtemp(prefix="aka-codex-test-", dir="/var/tmp"))
    _install_fake_codex(binary_parent, monkeypatch)
    monkeypatch.setenv("FAKE_CODEX_MODE", mode)
    monkeypatch.setattr(launcher, "_TERM_GRACE_SECONDS", 0.2)
    monkeypatch.setattr(launcher, "_KILL_GRACE_SECONDS", 2.0)
    data_root = tmp_path / "campaign"
    attempt = data_root / "run/task/attempt_01"
    workspace = attempt / "workspace"
    workspace.mkdir(parents=True)
    source = workspace / "kernel.py"
    source.write_text("baseline = True\n", encoding="utf-8")
    task_config = tmp_path / "task.yaml"
    task_config.write_text("source_file_path: kernel.py\n", encoding="utf-8")
    state_root = tmp_path / "agent-state/.codex"
    state_root.mkdir(parents=True)
    (state_root / "auth.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("AGENT_STATE_MOUNT_ROOT", str(state_root.parent))
    monkeypatch.setenv("AGENT_KERNEL_ARENA_CAMPAIGN_DATA_ROOT", str(data_root))
    receipt_path = attempt / "session_receipt.json"
    launcher.launch_agent(
        {
            "campaign": {"comparison": "apex_vs_codex"},
            "campaign_attempt": {
                "fresh_session": True,
                "receipt_path": str(receipt_path),
                "task_deadline_monotonic": time.monotonic() + 30,
                "comparison_contract_sha256": "d" * 64,
            },
        },
        str(task_config),
        str(workspace),
    )
    return _load_receipt(receipt_path), source, binary_parent


def test_exact_boundary_accepts_only_fully_reaped_exit_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt: dict = {}
    binary_parent: Path | None = None
    try:
        receipt, source, binary_parent = _formal_boundary_fixture(
            tmp_path,
            monkeypatch,
            mode="exact_boundary_natural_exit",
        )
        cleanup = receipt["attempt_process_cleanup"]
        assert receipt["exit_code"] in {0, 128 + int(signal.SIGKILL)}
        assert cleanup["reason"] == "exact_turn_boundary"
        assert cleanup["teardown_mode"] in {"natural_exit", "pidfd_sigkill"}
        assert cleanup["verified_absent"] is True
        assert receipt["capture"]["readers_completed"] is True
        persistence = receipt["candidate_persistence"]
        assert persistence["boundary_resolution"] == (
            "pid_namespace_destroyed_before_snapshot"
        )
        assert persistence["boundary_snapshot"]["capture_mode"] == (
            "post_namespace_teardown_checkpoint"
        )
        assert source.read_text(encoding="utf-8") == (
            "optimized_natural_exit = True\n"
        )
    finally:
        if receipt:
            _make_artifact_dir_removable(receipt)
        if binary_parent is not None:
            shutil.rmtree(binary_parent)


def test_exact_boundary_tracks_and_reaps_setsid_double_fork_descendant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt: dict = {}
    binary_parent: Path | None = None
    try:
        receipt, source, binary_parent = _formal_boundary_fixture(
            tmp_path,
            monkeypatch,
            mode="exact_boundary_setsid_double_fork",
        )
        cleanup = receipt["attempt_process_cleanup"]
        assert cleanup["scope"] == "private_pid_namespace"
        assert cleanup["method"] == "namespace_init_pidfd_v1"
        assert cleanup["sigkill_sent"] is True
        assert cleanup["verified_absent"] is True
        assert cleanup["namespace_membership_enumeration_completed"] is True
        assert cleanup["namespace_membership_inaccessible_entries_count"] >= 0
        assert cleanup["namespace_membership_scan_complete"] is (
            cleanup["namespace_membership_inaccessible_entries_count"] == 0
        )
        assert cleanup["live_visible_namespace_members_after"] == []
        time.sleep(0.7)
        assert source.read_text(encoding="utf-8") == (
            "optimized_with_descendant = True\n"
        )
    finally:
        if receipt:
            _make_artifact_dir_removable(receipt)
        if binary_parent is not None:
            shutil.rmtree(binary_parent)


def test_direct_codex_output_is_bounded_and_truncation_invalidates_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_codex(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_CODEX_MODE", "output_flood")
    monkeypatch.setattr(launcher, "_TERM_GRACE_SECONDS", 0.1)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    receipt_path = tmp_path / "evidence" / "session_receipt.json"

    with pytest.raises(launcher.CodexSessionError, match="bounded capture"):
        launcher.launch_agent(
            _attempt_config(receipt_path), str(tmp_path / "task.yaml"), str(workspace)
        )

    receipt = _load_receipt(receipt_path)
    try:
        stdout = receipt["capture"]["stdout"]
        assert stdout["truncated"] is True
        assert stdout["retained_bytes"] == stdout["limit_bytes"] == 16 * 1024 * 1024
        assert stdout["discarded_bytes"] > 0
        assert len(_artifact_bytes(receipt, "raw_stdout")) <= 16 * 1024 * 1024
        assert receipt["session_succeeded"] is False
    finally:
        _make_artifact_dir_removable(receipt)


def test_timeout_terms_then_kills_entire_group_and_records_verified_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_codex(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_CODEX_MODE", "timeout")
    monkeypatch.setattr(launcher, "_TERM_GRACE_SECONDS", 0.1)
    monkeypatch.setattr(launcher, "_KILL_GRACE_SECONDS", 2.0)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    receipt_path = tmp_path / "evidence" / "session_receipt.json"

    with pytest.raises(launcher.CodexSessionTimeout, match="verified_absent=True"):
        launcher.launch_agent(
            _attempt_config(receipt_path, timeout_seconds=0.2),
            str(tmp_path / "task.yaml"),
            str(workspace),
        )

    receipt = _load_receipt(receipt_path)
    try:
        cleanup = receipt["process_group_cleanup"]
        assert receipt["session_succeeded"] is False
        assert receipt["timed_out"] is True
        assert receipt["exit_code"] == -int(signal.SIGKILL)
        assert cleanup["required"] is True
        assert cleanup["sigterm_sent"] is True
        assert cleanup["sigkill_sent"] is True
        assert cleanup["verified_absent"] is True
    finally:
        _make_artifact_dir_removable(receipt)


def test_default_receipt_is_reserved_in_hidden_workspace_sibling(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    receipt, artifacts = launcher._attempt_receipt_path({}, workspace)

    assert receipt == tmp_path / ".workspace.codex-attempt" / "attempt_receipt.json"
    assert artifacts == tmp_path / ".workspace.codex-attempt"
    assert workspace not in receipt.parents
    artifacts.rmdir()


def test_receipt_inside_scored_workspace_is_rejected_without_creating_it(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    unsafe = workspace / "evidence" / "receipt.json"

    with pytest.raises(launcher.CodexSessionError, match="outside the scored workspace"):
        launcher._attempt_receipt_path(
            {"campaign_attempt": {"receipt_path": str(unsafe)}}, workspace
        )

    assert not unsafe.parent.exists()


def test_effective_timeout_is_capped_by_campaign_deadline_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher.time, "monotonic", lambda: 100.0)
    assert launcher._effective_timeout_seconds(
        {"timeout_seconds": 60},
        {"campaign_attempt": {"task_deadline_monotonic": 112.5}},
    ) == 12.5
    with pytest.raises(launcher.CodexSessionError, match="no positive"):
        launcher._effective_timeout_seconds(
            {"timeout_seconds": 60},
            {"campaign_attempt": {"task_deadline_monotonic": 100.0}},
        )


def test_atomic_artifact_publication_never_overwrites_existing_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.txt"
    path.write_bytes(b"original")

    with pytest.raises(FileExistsError):
        launcher._write_read_only_atomic(path, b"replacement")

    assert path.read_bytes() == b"original"
