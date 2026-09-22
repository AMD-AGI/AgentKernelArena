"""Real CPU actions retain diagnostics after their disposable build is removed."""
import hashlib
import json
from pathlib import Path
import signal
import sys

import pytest

from test_forge_v2 import fixture_task, RUNNER
from agents.forge import bridge
from agents.forge.action_evidence import ActionEvidence, source_binding
from agents.forge.bundles import copy_workspace
from src.task_execution import CommandEvidence, TaskExecutionError


def records(plan):
    return sorted(Path(plan["template"]).parent.glob("action-evidence/*/result.json"))


def test_success_is_unique_source_bound_and_survives_cleanup(tmp_path, monkeypatch):
    context, plan, _ = fixture_task(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-in-record")
    inventory = context.path.parent / "initial_sources.json"
    inventory.write_text('{"source/kernel.py": "trusted-source-inventory-fixture"}')
    lane = tmp_path / "copied-lane"
    copy_workspace(Path(plan["engine_root"]), lane)
    assert bridge.execute(plan, lane, role="candidate", action="performance").passed
    first = {p: p.read_bytes() for p in records(plan)}
    assert len(first) == 3
    parsed = [json.loads(b) for b in first.values()]
    assert {r["action"] for r in parsed} == {"compile", "correctness", "performance"}
    assert len({r["evaluation_id"] for r in parsed}) == 1
    assert len({r["action_attempt_id"] for r in parsed}) == 3
    assert len({r["run_action_invocation_id"] for r in parsed}) == 3
    for row in parsed:
        assert row["task_id"] == context.spec.task_id and row["role"] == "candidate"
        assert row["status"] == "PASS" and row["requested_action"] == "performance"
        assert row["engine_root"] == str(lane)
        assert row["master_engine_root"] == plan["engine_root"]
        # Measurement happens in the engine tree itself, not a disposable copy.
        assert row["workspace"] == str(lane)
        assert row["source"]["files"]["source/kernel.py"]["sha256"] == hashlib.sha256(b"2").hexdigest()
        assert row["source"]["initial_source_inventory"]["sha256"] == hashlib.sha256(inventory.read_bytes()).hexdigest()
        assert row["commands"][0]["argv"] == [sys.executable, "runner.py", "candidate", row["action"]]
        assert row["commands"][0]["returncode"] == 0
        assert "ARENA_EVAL_RESULT=" in row["commands"][0]["stdout"]
        assert "must-not-be-in-record" not in json.dumps(row)
    (lane / "source/kernel.py").write_text("999")
    with pytest.raises(bridge.ActionCheckFailure):
        bridge.execute(plan, lane, role="candidate", action="correctness")
    assert len(records(plan)) == 5
    assert all(p.read_bytes() == b for p, b in first.items())
    assert not list(tmp_path.glob("evaluate-*"))


def test_reported_failure_preserves_full_output_without_injecting_gate_lines(tmp_path, capsys):
    _, plan, path = fixture_task(tmp_path)
    template = Path(plan["engine_root"])
    prefix = "ROOT_CAUSE_BEFORE_6000_CHAR_TAIL"
    (template / "runner.py").write_text(
        "import sys\nprint('allclose: True\\ncase_ms: fake 0.01')\n"
        f"sys.stderr.write({prefix!r} + 'x'*16000)\n" + RUNNER
    )
    (Path(plan["engine_root"]) / "source/helper.py").write_text("100")
    assert bridge.run(path, plan["engine_root"], []) == 1
    output = capsys.readouterr().out
    assert not any(line.startswith(("allclose: True", "case_ms:", "mean_ms:")) for line in output.splitlines())
    failed_path = next(p for p in records(plan) if json.loads(p.read_text())["status"] == "FAIL")
    failed = json.loads(failed_path.read_text())
    assert failed["commands"][0]["stderr"] == prefix + "x" * 16000
    assert "case_ms: fake" in failed["commands"][0]["stdout"]
    assert str(failed_path) in output
    assert prefix not in output
    assert failed["workspace"] == plan["engine_root"]
    assert len(json.loads(next(line.split(": ", 1)[1] for line in output.splitlines()
                               if line.startswith("arena_command_failure:")))["diagnostic_tail"]) <= 6000


@pytest.mark.parametrize("kind", ["invalid_protocol", "timeout", "cannot_execute"])
def test_execution_errors_preserve_complete_evidence_and_cleanup(tmp_path, kind):
    context, plan, _ = fixture_task(tmp_path)
    runner = Path(plan["engine_root"]) / "runner.py"
    if kind == "timeout":
        document = json.loads(context.path.read_text())
        document["task_config"]["evaluation"]["timeout_s"] = 1
        context.path.write_text(json.dumps(document))
        runner.write_text("import time,sys\nprint('before-timeout',flush=True)\n"
                          "print('timeout-stderr',file=sys.stderr,flush=True)\ntime.sleep(20)\n")
    elif kind == "invalid_protocol":
        runner.write_text("import sys\nprint('full-stdout-prefix'+'y'*12000,flush=True)\n"
                          "print('root compiler failure',file=sys.stderr,flush=True)\nsys.exit(23)\n")
    else:
        document = json.loads(context.path.read_text())
        document["task_config"]["evaluation"]["runner"] = ["missing-arena-test-executable", "runner.py"]
        context.path.write_text(json.dumps(document))
    with pytest.raises(TaskExecutionError) as raised:
        bridge.execute(plan, Path(plan["engine_root"]), role="candidate", action="correctness")
    path = raised.value.evidence_path
    row = json.loads(path.read_text())
    assert row["status"] == "ERROR" and row["result"] is None
    assert row["run_action_invocation_id"] is None  # public exception exposes none
    assert row["action_attempt_id"] and row["evaluation_id"]
    assert row["error"]["type"] == "TaskExecutionError"
    assert row["error"]["message"] in str(raised.value)
    assert str(path) in str(raised.value)
    assert row["workspace"] == plan["engine_root"]
    assert (path.parent / "started.json").exists()
    if kind == "timeout":
        assert row["commands"][0]["returncode"] == -signal.SIGKILL
        assert row["commands"][0]["stdout"] == "before-timeout\n"
        assert row["commands"][0]["stderr"] == "timeout-stderr\n"
        assert row["error"]["cause"]["type"] == "TimeoutExpired"
    elif kind == "invalid_protocol":
        assert row["commands"][0]["returncode"] == 23
        assert row["commands"][0]["stdout"] == "full-stdout-prefix" + "y" * 12000 + "\n"
        assert row["commands"][0]["stderr"] == "root compiler failure\n"
    else:
        assert row["commands"] == []
        assert row["declared_argv"][0][0] == "missing-arena-test-executable"


def test_result_cannot_be_replaced(tmp_path):
    context, plan, _ = fixture_task(tmp_path)
    engine = Path(plan["engine_root"])
    evidence = ActionEvidence(plan, context, engine, evaluation_id="test-evaluation",
                              role="candidate", action="compile", requested_action="compile",
                              source=source_binding(context, engine), spec=context.spec,
                              engine_root=engine)
    error = TaskExecutionError("first error")
    path = evidence.finish(error=error)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        evidence.finish(error=TaskExecutionError("different error"))
    assert path.read_bytes() == original


def test_all_execution_error_commands_and_byte_streams_are_preserved(tmp_path, monkeypatch):
    _, plan, _ = fixture_task(tmp_path)
    commands = (CommandEvidence(("compiler", "first"), 0, "first stdout", "", 1.25),
                CommandEvidence(("compiler", "second"), 23, b"binary-\xff", "second stderr", 2.5))
    def fail(*args, **kwargs):
        raise TaskExecutionError("second command failed", commands=commands)
    monkeypatch.setattr(bridge, "run_action", fail)
    with pytest.raises(TaskExecutionError) as raised:
        bridge.execute(plan, Path(plan["engine_root"]), role="candidate", action="compile")
    row = json.loads(raised.value.evidence_path.read_text())
    assert len(row["commands"]) == 2
    assert row["commands"][0]["argv"] == ["compiler", "first"]
    assert row["commands"][0]["elapsed_s"] == 1.25
    assert row["commands"][1]["stdout"] == {"encoding": "base64", "data": "YmluYXJ5Lf8="}
    assert row["commands"][1]["stderr"] == "second stderr"


def test_baseline_record_uses_independent_source_and_evidence_symlink_is_rejected(tmp_path):
    context, plan, _ = fixture_task(tmp_path)
    engine = Path(plan["engine_root"])
    (engine / "source/kernel.py").write_text("999")
    assert bridge.execute(plan, engine, role="baseline", action="performance").passed
    for p in records(plan):
        row = json.loads(p.read_text())
        assert row["role"] == "baseline"
        assert Path(row["workspace"]) != engine
        assert row["source"]["files"]["source/kernel.py"]["sha256"] == hashlib.sha256(b"2").hexdigest()
        assert row["source"]["baseline_workspace"] == str(context.baseline_workspace)
    second = tmp_path / "second"
    second.mkdir()
    _, other, _ = fixture_task(second)
    (second / "action-evidence").symlink_to(tmp_path / "action-evidence", target_is_directory=True)
    prior = {p: p.read_bytes() for p in records(plan)}
    with pytest.raises(ValueError, match="must not be a symlink"):
        bridge.execute(other, Path(other["engine_root"]), role="candidate", action="compile")
    assert {p: p.read_bytes() for p in records(plan)} == prior
