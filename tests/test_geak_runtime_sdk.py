"""SDK lifecycle fixtures; requires anyio, makes no model or GPU calls."""
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

anyio = pytest.importorskip("anyio")


@pytest.mark.parametrize("status", ["allowed_warning", "rejected"])
def test_actual_sdk_rate_limit_event_retains_only_bounded_metadata(status):
    sdk = pytest.importorskip("claude_agent_sdk")
    from agents.geak_v4.workflow_runner import _record_runtime_identity

    info = sdk.types.RateLimitInfo(status=status, resets_at=1789462800,
        rate_limit_type="five_hour", overage_status="rejected",
        raw={"credential": "FAKE_PROVIDER_SECRET"}, overage_disabled_reason="FAKE_PROVIDER_SECRET")
    message = sdk.types.RateLimitEvent(info, uuid="private-id", session_id="private-session")
    identity = {}
    _record_runtime_identity(message, identity)
    assert identity["rate_limit"] == {"status": status, "resets_at": 1789462800,
                                      "rate_limit_type": "five_hour", "overage_status": "rejected"}
    assert identity.get("runtime_error_codes", []) == (["rate_limit"] if status == "rejected" else [])
    assert "FAKE_PROVIDER_SECRET" not in json.dumps(identity)
    assert "private-id" not in json.dumps(identity)


def test_actual_sdk_structured_assistant_error_does_not_retain_provider_text():
    sdk = pytest.importorskip("claude_agent_sdk")
    from agents.geak_v4.workflow_runner import _record_runtime_identity

    identity = {}
    _record_runtime_identity(sdk.AssistantMessage(
        content=[sdk.TextBlock(text="FAKE_PROVIDER_SECRET")], model="<synthetic>", error="rate_limit"), identity)
    assert identity == {"assistant_models": ["<synthetic>"], "runtime_error_codes": ["rate_limit"]}


def test_director_marker_does_not_preempt_runtime_return(tmp_path, monkeypatch):
    from agents.geak_v4 import workflow_runner as runner

    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    output = tmp_path / "workflow-output.json"
    terminal = {"eval_dir": str(eval_dir), "validation_status": "accepted", "final_geomean": 1.1,
                "final_patch": str(eval_dir / "final_patch.diff"), "budget_used": 1}
    notification = type("TaskNotificationMessage", (SimpleNamespace,), {})(
        task_id="workflow1", status="completed", output_file=str(output))
    def message(class_name, **kwargs):
        return type(class_name, (SimpleNamespace,), {})(**kwargs)

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
            yield message("AssistantMessage", content=[message("ToolUseBlock", id="tool1", name="Workflow", input={})])
            yield message("TaskStartedMessage", task_id="workflow1", tool_use_id="tool1")
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


def test_observed_identity_persists_before_sdk_timeout(tmp_path, monkeypatch):
    from agents.geak_v4 import workflow_runner as runner

    path = tmp_path / "runtime_identity.json"
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()

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
            yield message("SystemMessage", subtype="init", data={
                "model": "init-model", "claude_code_version": "probe-cli",
                "apiKey": "FAKE_SECRET_DO_NOT_LOG"})
            yield message("AssistantMessage", model="observed", content=[])
            # Evidence exists while the Workflow is still running, before any
            # terminal worker result or cleanup can be written.
            assert json.loads(path.read_text())["assistant_models"] == ["observed"]
            yield message("TaskStartedMessage", task_id="unfinished")
            await anyio.sleep_forever()

    sdk = ModuleType("claude_agent_sdk")
    sdk.ClaudeAgentOptions = lambda **kwargs: kwargs
    sdk.ClaudeSDKClient = Client
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", sdk)
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda _: "probe-sdk")
    monkeypatch.setattr(runner.os, "geteuid", lambda: 1000)
    with pytest.raises(TimeoutError):
        runner.invoke_via_sdk("fixture", workflow_dir=tmp_path, eval_dir=eval_dir,
            model="requested", effort="low", settings="{}", cli_path="unused", timeout_seconds=0.2,
            done_grace_seconds=1, done_poll_seconds=0.1, quiet=True,
            require_workflow_result=True, runtime_metadata_path=path)
    assert json.loads(path.read_text()) == {"requested_model": "requested", "sdk_version": "probe-sdk",
        "init_model": "init-model", "cli_version": "probe-cli", "assistant_models": ["observed"]}
    assert "FAKE_SECRET_DO_NOT_LOG" not in path.read_text()
