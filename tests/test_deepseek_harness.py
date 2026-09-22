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
import json, os, pathlib, sys
if sys.argv[1:] == ["--version"]:
    print(os.environ.get("FAKE_DSH_VERSION", "0.1.6-alpha.2"))
    sys.exit(0)
payload = sys.stdin.read()
pathlib.Path("observed.json").write_text(json.dumps({
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
}))
print(json.dumps({"type": "final", "text": "finished", "turn_end": "completed"}))
print("diagnostic " + os.environ["DEEPSEEK_API_KEY"], file=sys.stderr)
sys.exit(int(os.environ.get("FAKE_DSH_EXIT", "0")))
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
    assert "iterate up to 3 versions" in observed["prompt"]
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
    assert rows["llm-deepseek"]["reasoningEffort"] == config["reasoning_effort"]
    assert rows["session-log-deepseek"] == {"enabled": False}
    assert not (workspace / "forbidden").exists()
    assert "private-test-key" not in result + caplog.text
    assert "[REDACTED]" in result
    for artifact in state.iterdir():
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
    if mode == "timeout":
        with pytest.raises(TimeoutError, match="exceeded 1s"):
            call()
    else:
        assert "started" in call()
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
    "not-a-mapping", [], False,
])
def test_invalid_run_settings_fail_before_cli(runtime, monkeypatch, overrides):
    _, _, workspace = runtime
    def unexpected_preflight(*args):
        pytest.fail("Invalid run settings reached the CLI")
    monkeypatch.setattr(launcher, "_preflight", unexpected_preflight)
    with pytest.raises(ValueError):
        launcher.launch_agent({"agent": overrides}, "unused", str(workspace))
