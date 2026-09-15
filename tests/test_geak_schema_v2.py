"""CPU/interface tests; reported timings are synthetic, never GPU validation."""
from __future__ import annotations

import fcntl
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest
import yaml

from agents.geak.bridge import (Bridge, TaskContext, candidate_files, copy_task,
                                copy_tree, digests, remaining_budget, write_json)
from agents.geak.compatibility import adapt_lane, prepare_engine, verify_upstream
from agents.geak.launch_agent import prepare_job
from src.task_session import TaskSession
from src.task_spec import TaskSpec


RUNNER = r'''
import ast, json, os, runpy, sys, time
from pathlib import Path
import yaml
config = yaml.safe_load(Path("config.yaml").read_text())
role, action = ("task", "validate-task") if len(sys.argv) == 2 else sys.argv[1:]
if Path("probe.json").exists():
    phase_probe = json.loads(Path("probe.json").read_text())
    if role == "baseline" and phase_probe.get("baseline_initial_phase_only"):
        if os.environ.get("ARENA_EVAL_PHASE") != "task_validation":
            raise RuntimeError("Baseline rejects the final candidate language phase")
state = config["candidate"]["initial_state"]
rows = [{"test_case_id": name, "shape": [n], "dtype": "int64", "params": {"seed": 7},
         "status": "PASS"} for name, n in (("small", 2), ("wide", 5))]
metadata = {}
status = "PASS"
if action == "validate-task":
    metadata["candidate_state"] = state
    for row in rows:
        row["checks"] = ["correctness", "performance"]
elif action == "compile":
    rows = []
    if role == "candidate":
        for name in ("source/nested/implementation.py", "source/helpers/math.py"):
            ast.parse(Path(name).read_text())
else:
    if role == "baseline" and config["baseline"]["kind"] == "provided":
        value = int(Path("reference/value.txt").read_text())
    else:
        value = runpy.run_path("source/nested/implementation.py")["value"]
        value += runpy.run_path("source/helpers/math.py")["offset"]
    for row in rows:
        if action == "correctness":
            row["metrics"] = {"absolute_error": abs(value - 7)}
            if value != 7:
                status = row["status"] = "FAIL"
        else:
            # Deliberately synthetic device timing: protocol test only.
            row.update(execution_time_ms=2.0 if role == "baseline" else 4.0,
                       benchmark_method="cuda_graph")
result = {"protocol": "arena-eval-v1", "role": role, "action": action,
          "status": status, "cases": rows, "metadata": metadata}
if status == "FAIL":
    result["reason"] = "wrong numerical value"
if Path("probe.json").exists():
    probe = json.loads(Path("probe.json").read_text())
    if role == "candidate" and action == probe.get("action"):
        time.sleep(probe.get("sleep", 0))
        if probe.get("drop_case") and rows:
            rows.pop()
        if probe.get("wrong_shape") and rows:
            rows[0]["shape"] = [999]
        if probe.get("wrong_method") and rows:
            rows[0]["benchmark_method"] = "cuda_event_fallback"
        if probe.get("mutate"):
            Path("source/helpers/math.py").write_text("offset = 100\n")
        if probe.get("exit_code"):
            print("ARENA_EVAL_RESULT=" + json.dumps(result))
            sys.exit(probe["exit_code"])
print("ARENA_EVAL_RESULT=" + json.dumps(result))
sys.exit(0 if status == "PASS" else 1)
'''


@pytest.fixture
def task_factory(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_KERNEL_ARENA_PYTHON", sys.executable)
    def create(language="hip", state="implemented", baseline="provided", probe=None, tree=False,
               scoped=False, initial_language=None):
        root = tmp_path / (language + "-" + state + "-" + baseline)
        root.mkdir()
        for directory in ("scripts", "source/nested", "source/helpers", "reference"):
            (root / directory).mkdir(parents=True)
        (root / "source/nested/implementation.py").write_text("value = 7\n" if state == "implemented" else "")
        if scoped:
            (root / "source/nested/implementation.py").write_text("def compute():\n    return 7\nvalue = compute()\n")
        (root / "source/helpers/math.py").write_text("offset = 0\n" if state == "implemented" else "")
        (root / "reference/value.txt").write_text("7\n")
        (root / "scripts/runner.py").write_text(RUNNER)
        if probe:
            write_json(root / "probe.json", probe)
        config = {"schema_version": 2,
                  "candidate": {"language": language, "initial_state": state,
                                "editable": ([{"path": "source", "scope": "tree"}] if tree else
                                             ["source/nested/implementation.py", "source/helpers/math.py"])},
                  "baseline": {"kind": baseline, "source_files": ["reference/value.txt"]},
                  "evaluation": {"runner": ["python3", "scripts/runner.py"], "timeout_s": 5}}
        if scoped:
            config["candidate"]["editable"][0] = {"path": "source/nested/implementation.py",
                                                    "scope": "symbols", "symbols": ["compute"],
                                                    "allow_new_helpers": True}
        if initial_language:
            config["candidate"]["initial_language"] = initial_language
        (root / "config.yaml").write_text(yaml.safe_dump(config))
        spec = TaskSpec.from_mapping(config, task_id="opaque/arbitrary-name")
        session = TaskSession.create(spec, root, tmp_path / (root.name + "-state"))
        if probe and probe.get("action") in {"correctness", "performance"}:
            # Initial fixture validation must complete before testing candidate faults.
            # TaskContext here is a captured public manifest, not runner auto-discovery.
            raw = {"version": 1, "task_id": spec.task_id, "task_config": spec.to_mapping(),
                   "workspace": str(root), "baseline_workspace": str(session.baseline_workspace),
                   "manifest": {"protocol": "arena-eval-v1", "role": "task", "action": "validate-task",
                                "status": "PASS", "metadata": {"candidate_state": state},
                                "cases": [{"test_case_id": name, "shape": [n], "dtype": "int64",
                                           "params": {"seed": 7}, "status": "PASS",
                                           "checks": ["correctness", "performance"]}
                                          for name, n in (("small", 2), ("wide", 5))]}}
            write_json(session.state_directory / "agent_context.json", raw)
        else:
            initial = session.validate_initial()
            assert initial.accepted, initial.errors
        context = TaskContext.load(session.state_directory / "agent_context.json")
        options = {"budget": 1, "deep_cost": 2, "min_improve": 0.02, "gpu_ids": "0",
                   "timeout_seconds": 30, "model": None, "effort": "high", "claude_cli_path": "claude"}
        bridge = prepare_job(context, tmp_path / (root.name + "-geak"), options,
                             deadline=time.monotonic() + 30, deadline_epoch=time.time() + 30)
        return bridge
    return create


@pytest.mark.parametrize("language", ["hip", "triton", "flydsl"])
@pytest.mark.parametrize("state,baseline", [("implemented", "initial_candidate"),
                                           ("implemented", "provided"), ("unimplemented", "provided")])
def test_languages_states_baselines_and_nested_delivery(task_factory, language, state, baseline):
    bridge = task_factory(language, state, baseline)
    root = bridge.eval_dir / "workspace"
    # A correct slower candidate must still be delivered, including an author seed.
    (root / "source/nested/implementation.py").write_text("value = 5\n")
    (root / "source/helpers/math.py").write_text("offset = 2\n")
    before = digests(candidate_files(bridge.spec, bridge.context.baseline))
    result = bridge.deliver(root)
    assert result["status"] == "DELIVERED"
    assert result["arena_acceptance"] == "PENDING"
    assert result["agent_measurement"]["speedup_geomean"] == 0.5
    assert result["files"] == ["source/helpers/math.py", "source/nested/implementation.py"]
    assert (bridge.context.workspace / "source/helpers/math.py").read_text() == "offset = 2\n"
    assert digests(candidate_files(bridge.spec, bridge.context.baseline)) == before
    assert not (bridge.context.workspace / "task_result.yaml").exists()
    assert not (bridge.context.workspace / "implementation.py").exists()


def test_wrong_candidate_never_replaced_by_baseline(task_factory):
    bridge = task_factory()
    (bridge.eval_dir / "workspace/source/nested/implementation.py").write_text("value = 99\n")
    with pytest.raises(Exception, match="candidate.correctness"):
        bridge.deliver(bridge.eval_dir / "workspace")
    assert (bridge.context.workspace / "source/nested/implementation.py").read_text() == "value = 7\n"


@pytest.mark.parametrize("probe", [{"action": "correctness", "drop_case": True},
                                   {"action": "performance", "wrong_shape": True},
                                   {"action": "performance", "wrong_method": True},
                                   {"action": "correctness", "exit_code": 4},
                                   {"action": "correctness", "mutate": True}])
def test_bad_public_evidence_is_rejected(task_factory, probe):
    bridge = task_factory(probe=probe)
    with pytest.raises(Exception):
        bridge.deliver(bridge.eval_dir / "workspace")
    assert digests(candidate_files(bridge.spec, bridge.context.workspace)) == bridge.job["original_sources"]


def test_one_deadline_bounds_public_commands(task_factory):
    bridge = task_factory(probe={"action": "correctness", "sleep": 3})
    bridge.deadline = time.monotonic() + 0.25
    started = time.monotonic()
    with pytest.raises(Exception, match="deadline"):
        bridge.check(bridge.eval_dir / "workspace", performance=True)
    assert time.monotonic() - started < 2


def test_expired_deadline_prevents_new_commands(task_factory, monkeypatch):
    bridge = task_factory()
    bridge.deadline = time.monotonic() - 1
    monkeypatch.setattr("agents.geak.bridge.run_action", lambda *a, **k: pytest.fail("action started"))
    with pytest.raises(TimeoutError):
        bridge.action("candidate", "compile", bridge.eval_dir / "workspace")


@pytest.mark.parametrize("target", ["scripts/runner.py", "reference/value.txt", "config.yaml"])
def test_harness_reference_and_config_are_protected(task_factory, target):
    bridge = task_factory()
    (bridge.eval_dir / "workspace" / target).write_text("tampered\n")
    with pytest.raises(RuntimeError, match="Protected"):
        bridge.deliver(bridge.eval_dir / "workspace")


def test_frozen_baseline_cannot_be_replaced(task_factory):
    bridge = task_factory(baseline="initial_candidate")
    (bridge.context.baseline / "source/nested/implementation.py").write_text("value = 5\n")
    with pytest.raises(ValueError, match="baseline sources changed"):
        bridge.action("baseline", "performance")


def test_baseline_keeps_initial_language_phase_and_candidate_uses_final_phase(task_factory):
    bridge = task_factory("flydsl", baseline="initial_candidate", initial_language="triton",
                          probe={"baseline_initial_phase_only": True})
    # The phase-sensitive fixture must also pass prepare_job's baseline actions.
    bridge.action("baseline", "compile")
    bridge.action("baseline", "performance")
    bridge.check(bridge.eval_dir / "workspace", performance=True)
    records = [json.loads(path.read_text()) for path in (bridge.root / "checks").glob("*.json")]
    assert {record["phase"] for record in records if record["role"] == "baseline"} == {"task_validation"}
    assert {record["phase"] for record in records if record["role"] == "candidate"} == {"candidate_evaluation"}


def test_copy_deadline_interrupts_one_large_file(tmp_path, monkeypatch):
    from agents.geak import bridge as module

    source = tmp_path / "source"
    source.mkdir()
    (source / "large.bin").write_bytes(b"x" * 100)
    monkeypatch.setattr(module, "_COPY_CHUNK_SIZE", 4)
    destination = tmp_path / "destination"
    ticks = 0
    def remaining():
        nonlocal ticks
        ticks += 1
        # Allow setup and a few chunk writes, then exhaust this one budget.
        if ticks >= 15:
            raise TimeoutError("copy deadline")
        return 1
    with pytest.raises(TimeoutError, match="copy deadline"):
        copy_task(source, destination, remaining=remaining)
    assert 0 < (destination / "large.bin").stat().st_size < 100
    assert (source / "large.bin").stat().st_size == 100


def test_copy_checks_deadline_after_last_metadata_operation(tmp_path, monkeypatch):
    from agents.geak import bridge as module

    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "destination"
    now = [10.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    original = module.shutil.copystat
    def expired_metadata(src, dst):
        original(src, dst)
        now[0] = 12.0
    monkeypatch.setattr(module.shutil, "copystat", expired_metadata)
    with pytest.raises(TimeoutError):
        copy_tree(source, destination, remaining=lambda: remaining_budget(11.0))


def test_expired_preparation_does_not_create_run_directory(task_factory, tmp_path):
    bridge = task_factory()
    destination = tmp_path / "never-created"
    with pytest.raises(TimeoutError):
        prepare_job(bridge.context, destination, bridge.job["options"],
                    deadline=time.monotonic() - 1, deadline_epoch=time.time() - 1)
    assert not destination.exists()


def test_prepare_job_copy_timeout_stops_before_git_and_actions(task_factory, monkeypatch, tmp_path):
    bridge = task_factory()
    launcher = importlib.import_module("agents.geak.launch_agent")
    original_copy = launcher.copy_task
    destinations = []
    def expire_after_copy(source, destination, *, remaining):
        original_copy(source, destination, remaining=remaining)
        destinations.append(destination)
        raise TimeoutError("copy exhausted preparation budget")
    monkeypatch.setattr(launcher, "copy_task", expire_after_copy)
    monkeypatch.setattr(launcher, "_run_process", lambda *a, **kw: pytest.fail("git launched after failed copy"))
    with pytest.raises(TimeoutError):
        prepare_job(bridge.context, tmp_path / "partial-copy", bridge.job["options"],
                    deadline=time.monotonic() + 10, deadline_epoch=time.time() + 10)
    assert len(destinations) == 1
    assert destinations[0].name == "original"


def test_tree_helpers_delivered_and_obsolete_editable_files_removed(task_factory):
    bridge = task_factory(tree=True)
    root = bridge.eval_dir / "workspace"
    (root / "source/helpers/extra.py").write_text("helper = 1\n")
    result = bridge.deliver(root)
    assert "source/helpers/extra.py" in result["files"]


def test_symbol_scopes_preserve_colocated_harness(task_factory):
    bridge = task_factory("triton", scoped=True)
    path = bridge.eval_dir / "workspace/source/nested/implementation.py"
    path.write_text("def compute():\n    return helper()\ndef helper():\n    return 7\nvalue = compute()\n")
    assert bridge.check(bridge.eval_dir / "workspace", performance=False)["correctness"] == "pass"
    path.write_text(path.read_text().replace("value = compute()", "value = 7"))
    with pytest.raises(RuntimeError, match="Protected"):
        bridge.deliver(bridge.eval_dir / "workspace")


def test_parallel_device_lock_consumes_the_same_deadline(task_factory):
    bridge = task_factory()
    with (bridge.root / "device.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        bridge.deadline = time.monotonic() + 0.1
        with pytest.raises(TimeoutError):
            bridge.action("baseline", "compile")


def test_materialization_preserves_nested_inputs_and_private_git(task_factory):
    bridge = task_factory(tree=True)
    source = bridge.eval_dir / "workspace"
    (source / "source/helpers/extra.py").write_text("helper = 1\n")
    destination = bridge.eval_dir / "round_1/engineer_0/workspace"
    bridge.materialize(source, destination)
    assert (destination / "scripts/runner.py").read_text() == RUNNER
    assert (destination / ".git/HEAD").is_file()
    assert (destination / "source/helpers/extra.py").is_file()
    with pytest.raises(ValueError, match="fresh"):
        bridge.materialize(source, destination)


@pytest.mark.parametrize("alias", ["geak_v3", "geak_v3_triton", "geak_v4"])
def test_old_aliases_delegate_v2_without_legacy_cli(task_factory, monkeypatch, alias):
    bridge = task_factory()
    generic = importlib.import_module("agents.geak.launch_agent")
    monkeypatch.setattr(generic, "launch_agent", lambda *args: "generic GEAK v2")
    legacy = importlib.import_module(f"agents.{alias}.launch_agent")
    assert legacy.launch_agent({}, str(bridge.context.workspace / "config.yaml"),
                               str(bridge.context.workspace)) == "generic GEAK v2"


def test_hard_deadline_kills_detached_runner_children(task_factory, tmp_path):
    bridge = task_factory()
    launcher = importlib.import_module("agents.geak.launch_agent")
    pid_path = tmp_path / "child.pid"
    interpreter = tmp_path / "fake-python"
    interpreter.write_text(f"#!{sys.executable}\n" +
                           "import subprocess, sys, time\nfrom pathlib import Path\n" +
                           "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)\n" +
                           f"Path({str(pid_path)!r}).write_text(str(child.pid))\n" +
                           "time.sleep(30)\n")
    interpreter.chmod(0o755)
    bridge.deadline = time.monotonic() + 0.3
    with pytest.raises(subprocess.TimeoutExpired):
        launcher._run_engine(bridge, str(interpreter))
    child_pid = int(pid_path.read_text())
    for _ in range(20):
        state = Path(f"/proc/{child_pid}/stat")
        try:
            exited = state.read_text().rsplit(")", 1)[1].split()[0] == "Z"
        except (FileNotFoundError, ProcessLookupError):
            exited = True
        if exited:
            break
        time.sleep(0.02)
    else:
        pytest.fail("detached public-runner child survived GEAK's deadline")


def test_outside_workspace_and_symlink_artifacts_rejected(task_factory, tmp_path):
    bridge = task_factory()
    with pytest.raises(ValueError, match="private"):
        bridge.check(bridge.context.workspace, performance=False)
    path = bridge.eval_dir / "workspace/source/nested/implementation.py"
    path.unlink()
    external = tmp_path / "external.py"
    external.write_text("value = 7\n")
    path.symlink_to(external)
    with pytest.raises(ValueError):
        bridge.deliver(bridge.eval_dir / "workspace")


def test_context_rejects_wrong_version_and_overlapping_baseline(task_factory):
    bridge = task_factory()
    path = bridge.root / "context.json"
    raw = json.loads(path.read_text())
    for changes in ({"version": True}, {"version": 2},
                    {"baseline_workspace": str(bridge.context.workspace)}):
        write_json(path, {**raw, **changes})
        with pytest.raises(ValueError):
            TaskContext.load(path)


def test_bridge_cli_failure_is_nonzero_and_does_not_print_secret(task_factory):
    bridge = task_factory()
    path = bridge.eval_dir / "workspace/source/nested/implementation.py"
    path.write_text('raise RuntimeError("FAKE_RUNTIME_SECRET_123")\n')
    command = [sys.executable, "agents/geak/bridge.py", "--job", str(bridge.job_path),
               "correctness", "--workspace", str(bridge.eval_dir / "workspace")]
    result = subprocess.run(command, text=True, capture_output=True, timeout=10)
    assert result.returncode == 1
    assert "FAKE_RUNTIME_SECRET_123" not in result.stdout + result.stderr
    assert '"status": "FAIL"' in result.stdout


def test_launcher_retains_delivery_but_reports_engine_failure(task_factory, monkeypatch):
    bridge = task_factory()
    launcher = importlib.import_module("agents.geak.launch_agent")
    monkeypatch.setenv("ARENA_TASK_CONTEXT", str(bridge.root / "context.json"))
    monkeypatch.setenv("GEAK_HOME", str(bridge.root))
    monkeypatch.setenv("GEAK_CLAUDE_BIN", "unused")
    monkeypatch.setattr("agents.geak.compatibility.verify_upstream", lambda *a, **kw: None)
    monkeypatch.setattr(launcher, "prepare_engine", lambda *a, **kw: {})
    def failed_engine(new_bridge, python):
        (new_bridge.eval_dir / "workspace/source/nested/implementation.py").write_text("value = 5\n")
        (new_bridge.eval_dir / "workspace/source/helpers/math.py").write_text("offset = 2\n")
        write_json(new_bridge.root / "engine_result.json", {"status": "FAILED", "workflow_completed": False})
        return 1
    monkeypatch.setattr(launcher, "_run_engine", failed_engine)
    with pytest.raises(RuntimeError, match="GEAK failed"):
        launcher.launch_agent({}, str(bridge.context.workspace / "config.yaml"), str(bridge.context.workspace))
    assert (bridge.context.workspace / "source/nested/implementation.py").read_text() == "value = 5\n"
    outputs = list(bridge.context.workspace.parent.glob(".*_geak/*/delivery.json"))
    assert len(outputs) == 1
    report = json.loads(outputs[0].read_text())
    assert report["status"] == "FAILED"
    assert report["delivery"] == "DELIVERED"
    assert report["retained_workspace_sources"] == report["candidate"]["sources"]


def test_launcher_retains_observed_identity_after_worker_timeout(task_factory, monkeypatch):
    bridge = task_factory()
    launcher = importlib.import_module("agents.geak.launch_agent")
    monkeypatch.setenv("ARENA_TASK_CONTEXT", str(bridge.root / "context.json"))
    monkeypatch.setenv("GEAK_HOME", str(bridge.root))
    monkeypatch.setenv("GEAK_CLAUDE_BIN", "unused")
    monkeypatch.setattr("agents.geak.compatibility.verify_upstream", lambda *a, **kw: None)
    monkeypatch.setattr(launcher, "prepare_engine", lambda *a, **kw: {})

    def timed_out(new_bridge, python):
        write_json(new_bridge.root / "runtime_identity.json", {
            "requested_model": "requested", "assistant_models": ["observed"],
            "untrusted_extra": "FAKE_SECRET_DO_NOT_LOG"})
        raise subprocess.TimeoutExpired("worker", 1)

    monkeypatch.setattr(launcher, "_run_engine", timed_out)
    with pytest.raises(RuntimeError, match="GEAK failed"):
        launcher.launch_agent({}, str(bridge.context.workspace / "config.yaml"), str(bridge.context.workspace))
    path = next(bridge.context.workspace.parent.glob(".*_geak/*/delivery.json"))
    report = json.loads(path.read_text())
    assert report["status"] == "FAILED"
    assert report["delivery"] == "NOT_ATTEMPTED"
    assert report["engine"] == {"status": "MISSING", "workflow_completed": False,
                                "runtime": {"requested_model": "requested", "assistant_models": ["observed"]}}
    assert "FAKE_SECRET_DO_NOT_LOG" not in path.read_text()


@pytest.mark.parametrize("terminal", ["valid", "missing", "wrong_path", "sdk_error", "oauth_expired"])
def test_engine_worker_requires_truthful_terminal_result(task_factory, monkeypatch, terminal):
    from agents.geak import engine_worker

    bridge = task_factory()
    bridge.job["engine"] = {"script_path": str(bridge.root / "engine/kernel_workflow.js"),
                            "args": {"eval_dir": str(bridge.eval_dir), "mode": "optimize",
                                     "target_language": "hip", "apply_to_original": "false"}}
    write_json(bridge.job_path, bridge.job)
    def sdk(prompt, **kwargs):
        assert kwargs["quiet"] is True
        assert kwargs["require_workflow_result"] is True
        assert 0 < kwargs["timeout_seconds"] <= bridge.remaining() + 0.1
        kwargs["runtime_metadata"].update(init_model="probe-model", assistant_models=["probe-model"])
        if terminal == "oauth_expired":
            kwargs["runtime_metadata"]["runtime_error_codes"] = ["authentication_failed", "oauth_session_expired"]
            return ""
        result = {"eval_dir": str(bridge.eval_dir), "validation_status": "accepted", "final_geomean": 0.5,
                  "final_patch": str(bridge.eval_dir / "final_patch.diff")}
        if terminal == "wrong_path":
            result["eval_dir"] = "unrelated"
        if terminal != "missing":
            write_json(bridge.eval_dir / "workflow_return.json", result)
        if terminal == "sdk_error":
            raise RuntimeError("FAKE_PROVIDER_SECRET_DO_NOT_LOG")
        return ""
    monkeypatch.setattr(engine_worker, "invoke_via_sdk", sdk)
    code = engine_worker.run(bridge.job_path)
    output = (bridge.root / "engine_result.json").read_text()
    assert code == (0 if terminal == "valid" else 1)
    assert "FAKE_PROVIDER_SECRET_DO_NOT_LOG" not in output
    assert json.loads(output)["workflow_completed"] is (terminal == "valid")
    assert json.loads(output)["runtime"]["assistant_models"] == ["probe-model"]
    if terminal == "oauth_expired":
        assert json.loads(output)["error_code"] == "oauth_session_expired"


def test_runtime_identity_records_observed_models_without_credentials(tmp_path):
    from types import SimpleNamespace
    from agents.geak_v4.workflow_runner import _record_runtime_identity

    identity = {}
    def message(name, **kwargs):
        return type(name, (SimpleNamespace,), {})(**kwargs)
    _record_runtime_identity(message("SystemMessage", subtype="init", data={
        "model": "configured-alias", "claude_code_version": "probe-cli",
        "authToken": "secret-never-recorded"}), identity)
    _record_runtime_identity(message("AssistantMessage", model="actual-model"), identity)
    _record_runtime_identity(message("AssistantMessage", model="actual-model"), identity)
    path = tmp_path / "workflow-output.json"
    write_json(path, {"workflowProgress": [{"model": "actual-child-model", "promptPreview": "private"}]})
    _record_runtime_identity(message("TaskNotificationMessage", status="completed", output_file=str(path)), identity)
    assert identity == {"init_model": "configured-alias", "cli_version": "probe-cli",
                        "assistant_models": ["actual-model"], "workflow_models": ["actual-child-model"]}


def test_runtime_identity_handles_synchronous_and_unknown_workflow_output(tmp_path):
    from types import SimpleNamespace
    from agents.geak_v4.workflow_runner import _record_runtime_identity

    def message(name, **kwargs):
        return type(name, (SimpleNamespace,), {})(**kwargs)
    identity = {}
    _record_runtime_identity(message("SystemMessage", subtype="init", data=None), identity)
    path = tmp_path / "output.json"
    write_json(path, {"workflowProgress": None})
    _record_runtime_identity(message("TaskNotificationMessage", status="completed", output_file=str(path)), identity)
    payload = json.dumps({"workflowProgress": [{"model": "observed-child"}, None, {"model": 42}]})
    _record_runtime_identity(message("AssistantMessage", content=[message("TextBlock", text=payload)]), identity)
    assert identity == {}
    _record_runtime_identity(message("UserMessage", content=[
        message("ToolResultBlock", content="unstructured"),
        message("ToolResultBlock", content=[{"type": "text", "text": payload}])]), identity)
    assert identity == {"workflow_models": ["observed-child"]}
    # Real CLI completion can arrive while its output JSON is still partial.
    path.write_text('{"workflowProgress": [')
    notification = message("TaskNotificationMessage", status="completed", output_file=str(path))
    _record_runtime_identity(notification, identity)
    assert identity == {"workflow_models": ["observed-child"]}
    write_json(path, {"workflowProgress": [{"model": "late-child"}]})
    _record_runtime_identity(notification, identity)
    assert identity == {"workflow_models": ["late-child", "observed-child"]}


def test_runtime_identity_retains_error_codes_without_error_text(tmp_path):
    from types import SimpleNamespace
    from agents.geak_v4.workflow_runner import _record_runtime_identity

    output = tmp_path / "output.json"
    write_json(output, {"workflowProgress": [{"label": "tech_lead:plan r1", "state": "error",
                                            "error": "[reasoning_extraction] FAKE_PROVIDER_SECRET"}]})
    message = type("TaskNotificationMessage", (SimpleNamespace,), {})(status="completed", output_file=str(output))
    identity = {}
    _record_runtime_identity(message, identity)
    assert identity == {"workflow_agent_errors": [{"label": "tech_lead:plan r1", "code": "reasoning_extraction"}]}


@pytest.mark.parametrize("model", ["<synthetic>", "real-model"])
def test_runtime_identity_classifies_only_synthetic_oauth_failures(model):
    from types import SimpleNamespace
    from agents.geak_v4.workflow_runner import _record_runtime_identity

    message = type("AssistantMessage", (SimpleNamespace,), {})(model=model, content=[
        SimpleNamespace(text="Failed to authenticate: OAuth session expired and could not be refreshed. "
                             "FAKE_PROVIDER_SECRET_DO_NOT_LOG")])
    identity = {}
    _record_runtime_identity(message, identity)
    if model == "<synthetic>":
        assert identity["runtime_error_codes"] == ["authentication_failed", "oauth_refresh_failed", "oauth_session_expired"]
    else:
        assert "runtime_error_codes" not in identity
    assert "FAKE_PROVIDER_SECRET_DO_NOT_LOG" not in json.dumps(identity)


def test_engine_rejects_child_errors_before_search(task_factory, monkeypatch):
    from agents.geak import engine_worker

    bridge = task_factory()
    bridge.job["engine"] = {"script_path": str(bridge.root / "engine.js"),
                            "args": {"eval_dir": str(bridge.eval_dir), "mode": "optimize", "target_language": "hip"}}
    write_json(bridge.job_path, bridge.job)
    def sdk(prompt, **kwargs):
        kwargs["runtime_metadata"]["workflow_agent_errors"] = [{"label": "tech_lead:plan r1", "code": "agent_error"}]
        return json.dumps({"eval_dir": str(bridge.eval_dir), "validation_status": "accepted", "final_geomean": 1,
                           "final_patch": str(bridge.eval_dir / "final_patch.diff"), "rounds": 0, "budget_used": 0})
    monkeypatch.setattr(engine_worker, "invoke_via_sdk", sdk)
    assert engine_worker.run(bridge.job_path) == 1
    result = json.loads((bridge.root / "engine_result.json").read_text())
    assert result["error_code"] == "workflow_agent_errors_before_search"
    assert (bridge.eval_dir / "workflow_return.json").is_file()


def test_compatibility_rejects_unknown_engine_source():
    with pytest.raises(ValueError, match="unknown GEAK lane"):
        adapt_lane("export const meta = {}; return {};")


@pytest.fixture
def upstream():
    value = os.environ.get("GEAK_TEST_CHECKOUT")
    if not value:
        pytest.skip("Set GEAK_TEST_CHECKOUT for read-only upstream engine interface probes")
    path = Path(value)
    verify_upstream(path)
    return path


@pytest.mark.parametrize("language", ["hip", "triton", "flydsl"])
@pytest.mark.parametrize("state", ["implemented", "unimplemented"])
def test_pinned_upstream_preparation(task_factory, upstream, language, state):
    bridge = task_factory(language, state)
    bridge.job["options"]["model"] = "explicit-model"
    engine = prepare_engine(upstream, bridge, python=sys.executable, options=bridge.job["options"])
    args = engine["args"]
    assert args["mode"] == ("author" if state == "unimplemented" else "optimize")
    assert args["target_language"] == language
    assert args["arena_model"] == "explicit-model"
    assert args["arena_setup"]["baseline_dir"] == str(bridge.context.baseline)
    assert args["apply_to_original"] == "false"
    assert args["warm_start"] == args["update_experience"] == "off"
    lane = Path(args["kernel_lane_script"]).read_text()
    assert "const setup = A.arena_setup;" in lane
    assert "const bench = A.arena_benchmark;" in lane
    assert "const results = await pipeline(" in lane
    assert "roleAgent('author_engineer'" in lane
    assert "roleAgent('director', 'validate'" in lane
    assert "A.arena_contract" in lane
    assert "plan.decision_summary" in lane
    assert "reasoning: { type:" not in lane
    assert '"reasoning":' not in (Path(engine["script_path"]).parent / "roles/tech_lead.md").read_text()
    assert "const KB_WRITE_OK = false;" in lane
    assert args["kb_remote"] == "off"
    # Verify preparation never mutates the engine being shared with other runs.
    verify_upstream(upstream)


@pytest.mark.parametrize("expire_at", ["kernel_workflow", "perf_knowledge"])
def test_engine_and_knowledge_copy_deadlines(task_factory, upstream, monkeypatch, expire_at):
    from agents.geak import compatibility

    bridge = task_factory()
    original_copy = compatibility.copy_tree
    completed = []
    def expiring_copy(source, destination, *, remaining):
        if source.name == expire_at:
            bridge.deadline = time.monotonic() - 1
        original_copy(source, destination, remaining=remaining)
        completed.append(source.name)
    monkeypatch.setattr(compatibility, "copy_tree", expiring_copy)
    with pytest.raises(TimeoutError):
        prepare_engine(upstream, bridge, python=sys.executable, options=bridge.job["options"])
    assert completed == ([] if expire_at == "kernel_workflow" else ["kernel_workflow"])
    assert not (bridge.root / "engine_identity.json").exists()


@pytest.mark.parametrize("language", ["hip", "triton", "flydsl"])
@pytest.mark.parametrize("state", ["implemented", "unimplemented"])
def test_actual_geak_dispatcher_and_lane_run_on_cpu(task_factory, upstream, tmp_path, language, state):
    node = os.environ.get("GEAK_TEST_NODE") or shutil.which("node")
    if not node:
        pytest.skip("Node required for the real GEAK JavaScript interface probe")
    bridge = task_factory(language, state)
    bridge.job["options"]["model"] = "explicit-model"
    engine = prepare_engine(upstream, bridge, python=sys.executable, options=bridge.job["options"])
    args = engine["args"]
    args.update(deadline_epoch=0, agent_timeout_ms=0)
    probe = tmp_path / "probe.js"
    probe.write_text(r'''
const fs = require('fs');
const input = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const calls = [];
const globals = {
  phase: () => {}, log: () => {},
  agent: async (prompt, options) => {
    const label = options.label;
    calls.push(label);
    if (!prompt.includes('Arena task contract')) throw Error('missing task contract');
    if (options.model !== 'explicit-model') throw Error('wrong child model');
    if (label.startsWith('author:')) return {authored: true, correctness: 'pass'};
    if (label === 'tech_lead:analyze') return {kernel_type:'arbitrary', roadmap_summary:'probe'};
    if (label.includes('profile_engineer')) return {bottleneck:'unknown',top_opportunities:[],device:'gfx950'};
    if (label.startsWith('tech_lead:plan')) return {stop:false,directions:[{id:'cpu',specialty:'compute'}]};
    if (label.startsWith('eng ')) return {status:'ok',speedup_geomean:2};
    if (label.startsWith('verify ')) return {status:'verified',correctness:'pass',verified_geomean:2};
    if (label.startsWith('commit ')) return {committed:true};
    if (label === 'tech_lead:report') return {final_patch:input.args.eval_dir+'/final_patch.diff',final_speedup_geomean:0.5};
    if (label === 'director:validate') return {validation_status:'accepted',correctness:'pass',director_verified_speedup_geomean:2};
    return {};
  },
  parallel: async (thunks) => Promise.all(thunks.map(t => t())),
  pipeline: async (items, ...stages) => Promise.all(items.map(async item => {
    let value = item; for (const stage of stages) value = await stage(value); return value;
  })),
  budget: {spent:()=>0,remaining:()=>Infinity}
};
async function run(script, args) {
  const body=fs.readFileSync(script,'utf8').replace(/^export const meta/m,'const meta');
  const context={...globals,args,workflow:async (ref,a)=>run(ref.scriptPath,a)};
  return new Function(...Object.keys(context), 'return (async()=>{'+body+'})();')(...Object.values(context));
}
run(input.script_path,input.args).then(result=>{
  console.log(JSON.stringify({result,calls}));
}).catch(error=>{console.error(error);process.exit(1);});
''')
    handoff = tmp_path / "engine.json"
    write_json(handoff, engine)
    completed = subprocess.run([node, str(probe), str(handoff)], text=True, capture_output=True, timeout=10)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["result"]["validation_status"] == "accepted"
    assert result["result"]["mode"] == ("author" if state == "unimplemented" else "optimize")
    if state == "unimplemented":
        assert result["result"]["target_language"] == language
        assert f"author:{language}" in result["calls"]
    else:
        assert not any(label.startswith("author:") for label in result["calls"])
    assert "eng cpu:compute" in result["calls"]
    assert "verify cpu" in result["calls"]
    assert "commit r1" in result["calls"]
    assert "director:validate" in result["calls"]
    assert "director:setup" not in result["calls"]
    assert "benchmark_engineer" not in result["calls"]
    assert not any(label.startswith("kb:") or label == "update_experience" for label in result["calls"])


def test_initial_language_translation_uses_author_mode(task_factory, upstream):
    bridge = task_factory("flydsl", initial_language="triton")
    engine = prepare_engine(upstream, bridge, python=sys.executable, options=bridge.job["options"])
    assert engine["args"]["mode"] == "author"
    assert engine["args"]["target_language"] == "flydsl"
