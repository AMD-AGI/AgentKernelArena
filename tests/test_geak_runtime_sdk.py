"""SDK lifecycle fixtures; requires anyio, makes no model or GPU calls."""
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

anyio = pytest.importorskip("anyio")


def test_director_marker_does_not_preempt_runtime_return(tmp_path, monkeypatch):
    from agents.geak_v4 import workflow_runner as runner

    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    output = tmp_path / "workflow-output.json"
    terminal = {"eval_dir": str(eval_dir), "validation_status": "accepted", "final_geomean": 1.1,
                "final_patch": str(eval_dir / "final_patch.diff"), "budget_used": 1}
    notification = type("TaskNotificationMessage", (SimpleNamespace,), {})(
        task_id="workflow1", status="completed", output_file=str(output))
    def message(name, **kwargs):
        return type(name, (SimpleNamespace,), {})(**kwargs)

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
            yield message("TaskStartedMessage", task_id="workflow1")
            (eval_dir / "director_validation.json").write_text(json.dumps({
                "validation_status": "accepted", "correctness": "pass",
                "director_verified_speedup_geomean": 1.1, "applied_to_original": "false",
                "final_patch": str(eval_dir / "final_patch.diff")}))
            output.write_text('{"result":')
            yield notification
            await anyio.sleep(0.4)
            output.write_text(json.dumps({"result": terminal,
                                          "workflowProgress": [{"model": "actual-child"}]}))
            yield message("ResultMessage")

    sdk = ModuleType("claude_agent_sdk")
    sdk.ClaudeAgentOptions = lambda **kwargs: kwargs
    sdk.ClaudeSDKClient = Client
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", sdk)
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda _: "probe-sdk")
    monkeypatch.setattr(runner.os, "geteuid", lambda: 1000)
    # Force a full transcript before completion; the authoritative terminal
    # result must still survive truncation and be the last compact JSON line.
    original_text = runner._iter_message_text
    def text(message):
        if type(message).__name__ == "TaskStartedMessage":
            yield "x" * 4096
        yield from original_text(message)
    monkeypatch.setattr(runner, "_iter_message_text", text)
    monkeypatch.setattr(runner, "_TRANSCRIPT_SIZE_LIMIT", 2048)
    identity = {}
    transcript = runner.invoke_via_sdk("fixture", workflow_dir=tmp_path, eval_dir=eval_dir,
        model="explicit-model", effort="low", settings="{}", cli_path="unused", timeout_seconds=5,
        done_grace_seconds=1, done_poll_seconds=0.1, quiet=True, runtime_metadata=identity,
        require_workflow_result=True)
    assert runner._extract_workflow_return(transcript, eval_dir) == terminal
    assert len(transcript) <= 2048
    assert identity["workflow_models"] == ["actual-child"]
    assert not (eval_dir / "workflow_return.json").exists()
