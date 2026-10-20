"""Offline adapter regression tests; the fake CLI never contacts a model or GPU."""

import importlib
import json
import logging
import os
from pathlib import Path
import sys
import time

import pytest
import yaml

from src.module_registration import (
    AgentType, load_agent_launcher, load_post_processing_handler, load_prompt_builder,
)
from src.postprocessing import general_post_processing
from src.prompt_builder import prompt_builder


launcher = importlib.import_module("agents.deepseek_harness.launch_agent")


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    config = launcher._load_config()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "private-test-key")
    monkeypatch.setenv("DSH_HOME", str(tmp_path / "host-history"))
    monkeypatch.setenv("DSH_TOOLS_MODE", "inherited-experimental-mode")
    monkeypatch.setenv("DSH_TELEMETRY_MODE", "FEEDBACK_ONLY")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])
    binary = bin_dir / "dsh"
    binary.write_text(f"#!{sys.executable}\n" + '''
import json, os, pathlib, sys, time
if sys.argv[1:] == ["--version"]:
    print(os.environ.get("FAKE_DSH_VERSION", "0.1.6-alpha.2"))
    sys.exit(0)
payload = sys.stdin.read()
home = pathlib.Path(os.environ["DSH_HOME"])
home.mkdir(parents=True, exist_ok=True)
history_path = home / "history.json"
resume_id = sys.argv[sys.argv.index("--session-id") + 1] if "--session-id" in sys.argv else None
if resume_id is None:
    history = {"sessionId": "fixture-session", "prompts": []}
elif not history_path.exists() or json.loads(history_path.read_text())["sessionId"] != resume_id:
    print(json.dumps({"type": "error", "message": "session does not exist"}), flush=True)
    sys.exit(23)
else:
    history = json.loads(history_path.read_text())
history["prompts"].append(payload)
history_path.write_text(json.dumps(history))
mode = os.environ.get("FAKE_DSH_MODE")
observed = {
    "argv": sys.argv[1:], "cwd": os.getcwd(), "prompt": payload,
    "dsh_home": os.environ["DSH_HOME"],
    "permission": os.environ["DSH_PERMISSION_MODE"],
    "telemetry": os.environ["DSH_TELEMETRY_MODE"],
    "telemetry_disabled": os.environ["DSH_TELEMETRY_DISABLED"],
    "tools_mode": os.environ.get("DSH_TOOLS_MODE"),
    "gpu": os.environ.get("HIP_VISIBLE_DEVICES"),
    "python": os.environ["AGENT_KERNEL_ARENA_PYTHON"],
    "arena_context": os.environ.get("ARENA_TASK_CONTEXT"),
    "validation_context": os.environ.get("ARENA_VALIDATION_CONTEXT"),
    "arena_phase": os.environ.get("ARENA_EVAL_PHASE"),
    "history_length": len(history["prompts"]),
}
pathlib.Path("observed.json").write_text(json.dumps(observed))
with pathlib.Path("observed.jsonl").open("a") as log:
    log.write(json.dumps(observed) + "\\n")
if mode != "missing_id":
    session_id = "wrong-session" if mode == "wrong_id" and resume_id else history["sessionId"]
    print(json.dumps({"type": "session", "sessionId": session_id}), flush=True)
if mode == "timeout_first" or (mode == "timeout_resume" and resume_id):
    time.sleep(30)
if mode == "lost_history":
    history_path.unlink()
final = {"type": "final", "text": "finished"}
if os.environ.get("FAKE_DSH_USAGE"):
    final["usage"] = {"input_tokens": 100, "output_tokens": 10}
print(json.dumps(final))
print("diagnostic " + os.environ["DEEPSEEK_API_KEY"], file=sys.stderr)
exit_code = os.environ.get("FAKE_DSH_RESUME_EXIT", "0") if resume_id else "0"
sys.exit(int(os.environ.get("FAKE_DSH_EXIT", exit_code)))
''')
    binary.chmod(0o700)
    workspace = tmp_path / "task with spaces"
    workspace.mkdir()
    return config, binary, workspace


def test_registration_and_shared_routes():
    agent = AgentType.from_string("deepseek_harness")
    assert AgentType.from_string("deepseek-harness") == agent
    logger = logging.getLogger(__name__)
    assert load_agent_launcher(agent, logger) is launcher.launch_agent
    assert load_prompt_builder(agent, logger) is prompt_builder
    assert load_post_processing_handler(agent, logger) is general_post_processing


def test_launch_passes_full_prompt_isolates_state_and_redacts_key(runtime, monkeypatch, caplog):
    config, _, workspace = runtime
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    # Exceeds Linux's single-argument limit, and contains shell metacharacters.
    original_prompt = "Optimize kernel $(touch forbidden); `literal` 中文\n" * 5000
    calls = []

    def build(task, root, evaluation, logger):
        calls.append((task, root, evaluation))
        return original_prompt

    monkeypatch.setattr(launcher, "load_prompt_builder", lambda *args: build)
    with caplog.at_level(logging.INFO):
        result = launcher.launch_agent({"target_gpu_model": "MI300"}, "task/config.yaml", str(workspace))
    observed = json.loads((workspace / "observed.json").read_text())
    assert calls == [("task/config.yaml", str(workspace), {"target_gpu_model": "MI300"})]
    assert observed["prompt"].startswith(original_prompt)
    assert "iterate up to 9 versions" in observed["prompt"]
    assert f"budget for this agent invocation is {config['timeout_seconds']} seconds" in observed["prompt"]
    assert observed["cwd"] == str(workspace)
    assert observed["argv"][:3] == ["--profile", "headless", "--patch"]
    assert observed["argv"][-1] == "--json"
    assert original_prompt not in observed["argv"]
    assert observed["permission"] == "danger-full-access"
    assert observed["telemetry"] == "DISABLED"
    assert observed["telemetry_disabled"] == "1"
    assert observed["tools_mode"] is None
    assert observed["gpu"] == "0"
    assert observed["python"] == sys.executable
    state = Path(observed["dsh_home"]).parent
    assert state.parent == workspace
    patch = yaml.safe_load((state / "cordis.patch.yml").read_text())
    rows = {row["id"]: row["config"] for row in patch}
    assert rows["agent-default-model"]["model"] == config["model"]
    assert rows["llm-deepseek"]["apiKeyEnv"] == "DEEPSEEK_API_KEY"
    assert rows["llm-deepseek"]["reasoningEffort"] == config["reasoning_effort"] == "max"
    assert rows["session-log-deepseek"] == {"enabled": False}
    assert not (workspace / "forbidden").exists()
    assert "private-test-key" not in result + caplog.text
    assert "[REDACTED]" in result
    for artifact in state.rglob("*"):
        if artifact.is_file():
            assert "private-test-key" not in artifact.read_text()

    launcher.launch_agent({}, "task/config.yaml", str(workspace))
    second = json.loads((workspace / "observed.json").read_text())
    assert second["dsh_home"] != observed["dsh_home"]
    assert (state / "stdout.log").exists()  # Retry preserves the first invocation.


def test_preflight_checks_version_and_key_without_model_call(runtime, monkeypatch):
    _, _, workspace = runtime
    assert "API access not checked" in launcher.check_installation()
    assert not (workspace / "observed.json").exists()
    monkeypatch.setenv("FAKE_DSH_VERSION", "0.1.5-rc.2")
    with pytest.raises(RuntimeError, match="CLI version does not match"):
        launcher.check_installation()
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    with pytest.raises(RuntimeError, match="requires DEEPSEEK_API_KEY"):
        launcher.check_installation()


def test_missing_cli(runtime, monkeypatch):
    monkeypatch.setattr(launcher.shutil, "which", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="dsh not found"):
        launcher.check_installation()


def test_cli_failure_is_not_treated_as_completed(runtime, monkeypatch):
    _, _, workspace = runtime
    monkeypatch.setenv("FAKE_DSH_EXIT", "17")
    monkeypatch.setattr(launcher, "load_prompt_builder", lambda *args: lambda *args: "task")
    with pytest.raises(RuntimeError, match="exited with code 17"):
        launcher.launch_agent({}, "task/config.yaml", str(workspace))
    assert list(workspace.glob(".deepseek_harness-*/stderr.log"))


@pytest.mark.parametrize("mode", ["timeout", "success_with_child"])
def test_tool_processes_are_stopped_before_return(tmp_path, mode):
    state = tmp_path / "state"
    state.mkdir()
    (state / "prompt.txt").write_text("task")
    child_code = "import time; time.sleep(2); open('escaped', 'w').write('bad'); time.sleep(30)"
    code = (
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        "print('started', flush=True); "
        + ("time.sleep(30)" if mode == "timeout" else "sys.exit(0)")
    )
    call = lambda: launcher._run(
        [sys.executable, "-c", code], tmp_path, state, dict(os.environ), 1,
        logging.getLogger(__name__),
    )
    started = time.monotonic()
    output, outcome = call()
    assert outcome.timed_out is (mode == "timeout")
    assert "started" in output
    if mode != "timeout":
        assert outcome.ok
    # Wait past the child's write deadline even if it was reaped immediately.
    time.sleep(max(0, 2.2 - (time.monotonic() - started)))
    assert not (tmp_path / "escaped").exists()


def test_endpoint_override_and_off_effort_do_not_embed_credentials(runtime):
    config, _, _ = runtime
    config.update(base_url="https://gateway.example/v1", reasoning_effort="off")
    provider = launcher._patch(config)[1]["config"]
    assert provider["baseURL"] == "https://gateway.example/v1"
    assert provider["reasoningEffort"] == "off"
    assert "apiKey" not in provider


def test_v2_prompt_context_and_run_settings_reach_cli(runtime, monkeypatch):
    defaults, _, workspace = runtime
    config_path = workspace / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "schema_version": 2,
        "candidate": {"language": "hip", "editable": ["kernel.hip"]},
        "evaluation": {"runner": ["python3", "evaluate.py"]},
    }))
    (workspace / "README.md").write_text("Preserve the fixture operator contract.")
    (workspace / "kernel.hip").write_text("// fixture candidate\n")
    (workspace / "evaluate.py").write_text("# Declared action path; this test only invokes the fake CLI.\n")
    task_context = str(workspace.parent / "agent_context.json")
    validation_context = str(workspace.parent / "validation_context.json")
    monkeypatch.setenv("ARENA_TASK_CONTEXT", task_context)
    monkeypatch.setenv("ARENA_VALIDATION_CONTEXT", validation_context)
    monkeypatch.setenv("ARENA_EVAL_PHASE", "candidate_evaluation")
    launcher.launch_agent({
        "target_gpu_model": "MI355X", "_task_id": "fixture/materialized-task",
        "agent": {"template": "deepseek_harness", "model": "fixture-model",
                  "reasoning_effort": "low", "timeout_seconds": 45, "max_iterations": 1,
                  "cli_version": "not-an-allowed-run-override"},
    }, str(config_path), str(workspace))
    observed = json.loads((workspace / "observed.json").read_text())
    assert observed["arena_context"] == task_context
    assert observed["validation_context"] == validation_context
    assert observed["arena_phase"] == "candidate_evaluation"
    assert "Task: fixture/materialized-task" in observed["prompt"]
    assert "Preserve the fixture operator contract." in observed["prompt"]
    assert "python3 evaluate.py candidate correctness" in observed["prompt"]
    assert "ARENA_EVAL_RESULT" in observed["prompt"]
    assert "iterate up to 1 versions" in observed["prompt"]
    assert "budget for this agent invocation is 45 seconds" in observed["prompt"]
    state = Path(observed["dsh_home"]).parent
    invocation = json.loads((state / "invocation.json").read_text())
    assert invocation["agent_config"]["timeout_seconds"] == 45
    assert invocation["cli_version"] == defaults["cli_version"]
    patch = yaml.safe_load((state / "cordis.patch.yml").read_text())
    assert patch[0]["config"]["model"] == "fixture-model"
    assert patch[1]["config"]["reasoningEffort"] == "low"
    assert launcher._load_config() == defaults


@pytest.mark.parametrize("overrides", [
    {"timeout_seconds": 0}, {"timeout_seconds": True}, {"max_iterations": -1},
    {"model": ""}, {"reasoning_effort": "invalid"}, {"protocol": "invalid"},
    {"budget_mode": "tokens"}, {"budget_mode": ""}, {"budget_mode": False},
    "not-a-mapping", [], False,
])
def test_invalid_run_settings_fail_before_cli(runtime, monkeypatch, overrides):
    _, _, workspace = runtime
    def unexpected_preflight(*args):
        pytest.fail("Invalid run settings reached the CLI")
    monkeypatch.setattr(launcher, "_preflight", unexpected_preflight)
    with pytest.raises(ValueError):
        launcher.launch_agent({"agent": overrides}, "unused", str(workspace))


def _observations(workspace):
    return [json.loads(line) for line in (workspace / "observed.jsonl").read_text().splitlines()]


def _scripted_sessions(monkeypatch, durations):
    """Use real fake-CLI processes but a deterministic clock for their budget."""
    from src.tools import agent_time_budget

    def run(invoke, **settings):
        elapsed = 0.0
        remaining_durations = iter(durations)

        def measured(*args):
            nonlocal elapsed
            outcome = invoke(*args)
            elapsed += next(remaining_durations)
            return outcome

        return agent_time_budget.run_sessions(measured, clock=lambda: elapsed, **settings)

    monkeypatch.setattr(launcher, "run_sessions", run)


def test_time_mode_preserves_history_and_uses_one_decreasing_deadline(runtime, monkeypatch):
    from src.tools.agent_cost import AgentCostCollector

    _, _, workspace = runtime
    monkeypatch.setattr(launcher, "load_prompt_builder", lambda *args: lambda *args: "original task")
    monkeypatch.setenv("FAKE_DSH_USAGE", "1")
    _scripted_sessions(monkeypatch, [1000, 1200, 850])
    config = {"agent": {"budget_mode": "time", "timeout_seconds": 3600, "max_iterations": 2}}
    collector = AgentCostCollector(agent="deepseek_harness", task_name="fixture/task")
    output = collector.wrap(launcher.launch_agent)(
        eval_config=config, task_config_dir="unused", workspace=str(workspace))
    observed = _observations(workspace)
    assert [row["history_length"] for row in observed] == [1, 2, 3]
    assert len({row["dsh_home"] for row in observed}) == 1
    assert all(row["cwd"] == str(workspace) for row in observed)
    assert "--session-id" not in observed[0]["argv"]
    for row in observed[1:]:
        assert row["argv"][-2:] == ["--session-id", "fixture-session"]
    assert "no fixed number of versions" in observed[0]["prompt"]
    assert "iterate up to" not in observed[0]["prompt"]
    assert "budget for this agent invocation is 3600 seconds" in observed[0]["prompt"]
    assert "43 minutes" in observed[1]["prompt"]
    assert "23 minutes" in observed[2]["prompt"]
    state = Path(observed[0]["dsh_home"]).parent
    sessions = sorted(state.glob("session-*"))
    metadata = [json.loads((path / "invocation.json").read_text()) for path in sessions]
    assert [row["timeout_seconds"] for row in metadata] == [3600, 2600, 1400]
    assert [row["resume_session_id"] for row in metadata] == [None, "fixture-session", "fixture-session"]
    for path, row in zip(sessions, observed):
        assert (path / "prompt.txt").read_text() == row["prompt"]
        assert '"type": "final"' in (path / "stdout.log").read_text()
        assert "[REDACTED]" in (path / "stderr.log").read_text()
    history = json.loads((Path(observed[0]["dsh_home"]) / "history.json").read_text())
    assert history["prompts"] == [row["prompt"] for row in observed]
    assert "private-test-key" not in output
    attempt = collector.attempt()
    assert [row["session_id"] for row in attempt["invocations"]] == ["fixture-session"] * 3
    assert attempt["usage"] == {"input_tokens": 300, "output_tokens": 30}
    assert attempt["usage_status"] == "complete"
    assert attempt["total_cost_usd"] is None and attempt["cost_status"] == "not_reported"
    # A new task launch still starts a separate home and never adopts this history.
    launcher.launch_agent({"agent": {"budget_mode": "time", "timeout_seconds": 60}},
                          "unused", str(workspace))
    retry = _observations(workspace)[-1]
    assert retry["dsh_home"] != observed[0]["dsh_home"]
    assert retry["history_length"] == 1 and "--session-id" not in retry["argv"]


@pytest.mark.parametrize("mode,reason,launches", [
    ("missing_id", None, 1),
    ("lost_history", "exited with code 23", 1),
    ("wrong_id", "resumed a different session", 2),
])
def test_time_mode_never_substitutes_a_new_session(runtime, monkeypatch, mode, reason, launches):
    _, _, workspace = runtime
    monkeypatch.setattr(launcher, "load_prompt_builder", lambda *args: lambda *args: "task")
    monkeypatch.setenv("FAKE_DSH_MODE", mode)
    _scripted_sessions(monkeypatch, [10, 10, 10])
    config = {"agent": {"budget_mode": "time", "timeout_seconds": 3600}}
    if reason is None:
        launcher.launch_agent(config, "unused", str(workspace))
    else:
        with pytest.raises(RuntimeError, match=reason):
            launcher.launch_agent(config, "unused", str(workspace))
    assert len(_observations(workspace)) == launches
    state = next(workspace.glob(".deepseek_harness-*"))
    assert len(list(state.glob("session-*"))) == (1 if mode == "missing_id" else 2)


def test_failed_continuation_stops_instead_of_restarting(runtime, monkeypatch):
    _, _, workspace = runtime
    monkeypatch.setattr(launcher, "load_prompt_builder", lambda *args: lambda *args: "task")
    monkeypatch.setenv("FAKE_DSH_RESUME_EXIT", "17")
    _scripted_sessions(monkeypatch, [10, 10])
    with pytest.raises(RuntimeError, match="exited with code 17"):
        launcher.launch_agent({"agent": {"budget_mode": "time", "timeout_seconds": 3600}},
                              "unused", str(workspace))
    assert len(_observations(workspace)) == 2


@pytest.mark.parametrize("mode", ["iterations", "time"])
def test_first_session_timeout_remains_a_failure(runtime, monkeypatch, mode):
    _, _, workspace = runtime
    monkeypatch.setattr(launcher, "load_prompt_builder", lambda *args: lambda *args: "task")
    monkeypatch.setenv("FAKE_DSH_MODE", "timeout_first")
    with pytest.raises(TimeoutError, match="timed out after 1s"):
        launcher.launch_agent({"agent": {"budget_mode": mode, "timeout_seconds": 1}},
                              "unused", str(workspace))
    assert len(_observations(workspace)) == 1


def test_continuation_timeout_exhausts_the_original_budget(runtime, monkeypatch):
    from src.tools import agent_time_budget
    from src.tools.agent_cost import AgentCostCollector

    _, _, workspace = runtime
    monkeypatch.setattr(launcher, "load_prompt_builder", lambda *args: lambda *args: "task")
    monkeypatch.setenv("FAKE_DSH_MODE", "timeout_resume")
    monkeypatch.setenv("FAKE_DSH_USAGE", "1")
    monkeypatch.setattr(agent_time_budget, "MIN_RESUME_SECONDS", 0)
    collector = AgentCostCollector(agent="deepseek_harness", task_name="fixture/task")
    start = time.monotonic()
    collector.wrap(launcher.launch_agent)(
        eval_config={"agent": {"budget_mode": "time", "timeout_seconds": 2}},
        task_config_dir="unused", workspace=str(workspace))
    assert time.monotonic() - start < 5
    assert len(_observations(workspace)) == 2
    attempt = collector.attempt()
    assert attempt["launcher_error"] is None
    assert attempt["invocations"][-1]["timed_out"] is True
    assert attempt["usage"] == {"input_tokens": 100, "output_tokens": 10}
    assert attempt["usage_status"] == "partial"
