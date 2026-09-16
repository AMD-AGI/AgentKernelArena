"""Private diagnostic regressions using real SDK types, no provider or GPU."""
import json
import stat
import sys
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip("anyio")
sdk_types = pytest.importorskip("claude_agent_sdk").types
from claude_agent_sdk import ProcessError
from test_geak_schema_v2 import task_factory

try:
    from builtins import ExceptionGroup
except ImportError:
    from exceptiongroup import ExceptionGroup


SECRET = "FAKE_PROVIDER_SECRET_DO_NOT_RETAIN"


def _job(task_factory):
    from agents.geak.bridge import write_json

    bridge = task_factory()
    bridge.job["engine"] = {
        "script_path": str(bridge.root / "workflow.js"),
        "args": {"mode": "optimize", "target_language": "hip", "budget": 2,
                 "task": SECRET, "eval_dir": str(bridge.eval_dir)},
    }
    write_json(bridge.job_path, bridge.job)
    return bridge


def _read_private(bridge):
    path = bridge.root / "runtime_identity.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert SECRET not in path.read_text()
    report = (bridge.root / "engine_result.json").read_text()
    assert SECRET not in report
    return json.loads(path.read_text())["sdk_diagnostics"], json.loads(report)


def test_nested_sdk_failure_retains_leaf_reason_and_actual_exit_only(task_factory, monkeypatch):
    from agents.geak import engine_worker
    from agents.geak import workflow_runner as runner

    bridge = _job(task_factory)
    process_error = ProcessError(SECRET, exit_code=17, stderr=SECRET)
    leaf = RuntimeError("GEAK Workflow tool returned an error")
    leaf.__cause__ = ValueError(SECRET)
    group = ExceptionGroup(SECRET, [ExceptionGroup(SECRET, [process_error]), leaf])

    def fail(*args, **kwargs):
        raise group

    monkeypatch.setattr(engine_worker, "invoke_via_sdk", fail)
    assert engine_worker.run(bridge.job_path) == 1
    diagnostic, report = _read_private(bridge)
    assert report["status"] == "FAILED"
    assert report["error_type"] == "ExceptionGroup"
    assert report["error_code"] == "sdk_workflow_failed"
    assert report["workflow_completed"] is False
    assert diagnostic["failure"] == {"truncated": False, "leaves": [
        {"exception_class": "ProcessError", "reason": "cli_process_failed", "exit_code": 17},
        {"exception_class": "RuntimeError", "reason": "workflow_tool_error"},
        {"exception_class": "ValueError", "reason": "unclassified"},
    ]}
    # A self-referencing cause cannot hang diagnostics, and oversize groups are
    # explicitly truncated instead of silently presenting complete evidence.
    leaf.__cause__ = leaf
    bounded = {}
    runner._record_sdk_failure(bounded, ExceptionGroup(SECRET, [leaf] * 100))
    assert bounded["sdk_diagnostics"]["failure"]["truncated"] is True
    assert len(bounded["sdk_diagnostics"]["failure"]["leaves"]) == 1


@pytest.mark.parametrize("scenario,calls,matches,reason", [
    ("zero", 0, 0, "native_return_missing"),
    ("mismatch", 1, 0, "workflow_arguments_mismatch"),
    ("input_type", 1, 0, "workflow_arguments_mismatch"),
    ("duplicate", 2, 2, "workflow_count_invalid"),
    ("tool_error", 1, 1, "workflow_tool_error"),
    ("extra_only", 1, 0, "workflow_arguments_mismatch"),
])
def test_observed_calls_persist_before_strict_failure(
    task_factory, monkeypatch, scenario, calls, matches, reason,
):
    from agents.geak import engine_worker
    from agents.geak import workflow_runner as runner

    bridge = _job(task_factory)
    inputs = {"scriptPath": bridge.job["engine"]["script_path"],
              "args": dict(bridge.job["engine"]["args"])}
    if scenario == "mismatch":
        inputs["args"]["budget"] = SECRET
        inputs["args"][SECRET] = SECRET
        inputs[SECRET] = SECRET
    if scenario == "input_type":
        inputs = [SECRET]
    if scenario == "extra_only":
        inputs["run_in_background"] = False
    terminal = {"eval_dir": str(bridge.eval_dir), "validation_status": "accepted",
                "final_geomean": 1.0, "final_patch": str(bridge.eval_dir / "final_patch.diff"),
                "budget_used": 2}

    class Client:
        def __init__(self, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def query(self, prompt):
            pass
        async def receive_messages(self):
            if scenario != "zero":
                blocks = [sdk_types.ToolUseBlock(id="tool1", name="Workflow", input=inputs)]
                if scenario == "duplicate":
                    blocks.append(sdk_types.ToolUseBlock(id="tool2", name="Workflow", input=inputs))
                yield sdk_types.AssistantMessage(content=blocks, model="fixture-model")
                if scenario in {"tool_error", "extra_only"}:
                    yield sdk_types.UserMessage(content=[sdk_types.ToolResultBlock(tool_use_id="tool1",
                        content=SECRET if scenario == "tool_error" else json.dumps({"result": terminal}),
                        is_error=scenario == "tool_error")])
            yield type("ResultMessage", (SimpleNamespace,), {})()

    _install_client(monkeypatch, runner, Client)
    assert engine_worker.run(bridge.job_path) == (0 if reason is None else 1)
    diagnostic, report = _read_private(bridge)
    assert report["workflow_completed"] is (reason is None)
    assert diagnostic["observed_workflow_calls"] == calls
    assert diagnostic["matched_workflow_calls"] == matches
    assert diagnostic["match_checked"] is True
    if reason:
        assert reason in {row["reason"] for row in diagnostic["failure"]["leaves"]}
    else:
        assert "failure" not in diagnostic
    if scenario == "mismatch":
        row = diagnostic["calls"][0]
        assert row["extra_keys"] == row["args_extra_keys"] == 1
        assert {"field": ["args", "budget"], "expected_type": "integer", "actual_type": "string"} in row["differences"]
    if scenario == "input_type":
        assert diagnostic["calls"][0]["input_type"] == "array"
    if scenario == "extra_only":
        # Unexpected native tool keys fail the exact declared invocation contract.
        assert diagnostic["calls"][0]["extra_keys"] == 1
        assert diagnostic["calls"][0]["extra_key_types"] == {"run_in_background": "boolean"}


def _install_client(monkeypatch, runner, client):
    sdk = ModuleType("claude_agent_sdk")
    # Exercise the installed option contract, not a permissive **kwargs stub.
    from claude_agent_sdk import ClaudeAgentOptions
    sdk.ClaudeAgentOptions = ClaudeAgentOptions
    sdk.ClaudeSDKClient = client
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", sdk)
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda _: "offline-sdk")
    monkeypatch.setattr(runner.os, "geteuid", lambda: 1000)


def test_sdk_stderr_is_bounded_private_and_never_an_acceptance_override(task_factory, monkeypatch, capsys):
    from agents.geak import engine_worker
    from agents.geak import workflow_runner as runner

    bridge = _job(task_factory)

    class Client:
        def __init__(self, *, options):
            self.stderr = options.stderr
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def query(self, prompt):
            self.stderr("OAuth session expired; could not be refreshed: " + SECRET)
            saved = json.loads((bridge.root / "runtime_identity.json").read_text())
            assert saved["sdk_diagnostics"]["stderr_codes"] == ["oauth_refresh_failed", "oauth_session_expired"]
            assert "runtime_error_codes" not in saved
            for _ in range(1100):
                self.stderr(SECRET * 1000)
            raise ProcessError(SECRET, exit_code=23, stderr=SECRET)
        async def receive_messages(self):
            if False:
                yield None

    _install_client(monkeypatch, runner, Client)
    assert engine_worker.run(bridge.job_path) == 1
    diagnostic, report = _read_private(bridge)
    assert report["error_code"] == "sdk_workflow_failed"
    assert report["workflow_completed"] is False
    assert diagnostic["stderr_truncated"] is True
    assert diagnostic["stderr_callbacks"] == 1025
    assert diagnostic["failure"]["leaves"] == [
        {"exception_class": "ProcessError", "reason": "cli_process_failed", "exit_code": 23}]
    assert len(json.dumps(diagnostic)) < 2048
    assert SECRET not in capsys.readouterr().err
