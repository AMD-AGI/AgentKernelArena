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


def test_validator_retains_terminal_error_details_outside_candidate(tmp_path):
    from agents.task_validator.launch_agent import _run_backend
    workspace = tmp_path / "task"
    workspace.mkdir()
    event = {"type": "result", "is_error": True, "subtype": "error_during_execution",
             "errors": ["Session expired and could not be refreshed"],
             "authorization": "synthetic-validator-secret"}
    result = _run_backend(
        [sys.executable, "-c", f"print({json.dumps(event)!r}); raise SystemExit(7)"],
        backend="claude_code", workspace=str(workspace), timeout_seconds=5,
        logger=logging.getLogger(__name__),
    )
    assert result.returncode == 7 and result.error
    directory, = tmp_path.glob(".task-validator-claude_code-stream-*")
    saved = (directory / "stdout.jsonl").read_text()
    assert "Session expired and could not be refreshed" in saved
    assert "synthetic-validator-secret" not in saved + result.output
    assert not list(workspace.iterdir())
