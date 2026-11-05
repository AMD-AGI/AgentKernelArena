import json
import logging
import sys

import pytest

from agents.stream_log import StreamLog


def test_event_log_flushes_redacts_and_preserves_previous_invocations(tmp_path):
    workspace = tmp_path / "task"
    workspace.mkdir()
    logger = logging.getLogger(__name__)
    env = {"BACKEND_API_KEY": "synthetic-env-secret"}
    log = StreamLog(workspace, "test", logger, env=env)
    event = {"message": "synthetic-env-secret Bearer synthetic-bearer-secret",
             "nested": {"refresh_token": "synthetic-refresh-secret"},
             "usage": {"input_tokens": 123}, "tool_output": "x" * 5000 + "TAIL"}
    with log.open("stdout") as record:
        record(json.dumps(event) + "\n")
        # Must be available before process completion or closing the stream.
        saved = json.loads((log.directory / "stdout.jsonl").read_text())
        assert saved["tool_output"].endswith("TAIL")
        assert saved["usage"] == {"input_tokens": 123}
        assert saved["nested"]["refresh_token"] == "[REDACTED]"
        assert "synthetic-" not in saved["message"]
    previous = (log.directory / "stdout.jsonl").read_bytes()
    other = StreamLog(workspace, "test", logger, env=env)
    assert other.directory != log.directory
    assert (log.directory / "stdout.jsonl").read_bytes() == previous
    with pytest.raises(FileExistsError):
        with log.open("stdout"):
            pass
    assert log.error == "Agent event recording failed: FileExistsError"
    assert (log.directory / "stdout.jsonl").read_bytes() == previous


def test_backend_retains_terminal_error_details_outside_the_workspace(tmp_path):
    from agents.backend_cli import run_backend
    workspace = tmp_path / "task"
    workspace.mkdir()
    event = {"type": "result", "is_error": True, "subtype": "error_during_execution",
             "errors": ["Session expired and could not be refreshed"],
             "authorization": "synthetic-auditor-secret"}
    result = run_backend(
        [sys.executable, "-c", f"print({json.dumps(event)!r}); raise SystemExit(7)"],
        backend="claude_code", workspace=str(workspace), timeout_seconds=5,
        logger=logging.getLogger(__name__),
    )
    assert result.returncode == 7 and not result.timed_out
    assert result.error == "Backend emitted a terminal failure event"
    directory, = tmp_path.glob(".task-backend-claude_code-stream-*")
    saved = (directory / "stdout.jsonl").read_text()
    assert "Session expired and could not be refreshed" in saved
    assert "synthetic-auditor-secret" not in saved + result.output
    assert not list(workspace.iterdir())


@pytest.mark.parametrize("events,error", [
    ([{"type": "turn.completed"}], None),
    ([{"type": "turn.failed"}], "Backend emitted a terminal failure event"),
    ([], "Backend did not emit a successful terminal event"),
])
def test_backend_success_requires_a_terminal_success_event(tmp_path, events, error):
    from agents.backend_cli import run_backend
    workspace = tmp_path / "task"
    workspace.mkdir()
    lines = "".join(f"print({json.dumps(json.dumps(event))});" for event in events)
    result = run_backend([sys.executable, "-c", lines or "pass"], backend="codex", workspace=str(workspace),
                         timeout_seconds=5, logger=logging.getLogger(__name__))
    assert result.returncode == 0
    assert result.error == error
