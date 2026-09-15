"""Exercise CLI argv, run overrides, failures, and cleanup without GPU or auth."""

import importlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import tomllib

import pytest


AGENTS = ("codex", "claude_code", "cursor")
DEFAULTS = {
    "codex": "gpt-6-astra",
    "claude_code": "claude-fable-5-1",
    "cursor": "composer-2.5",
}


@pytest.fixture(params=AGENTS)
def launcher(request):
    return importlib.import_module(f"agents.{request.param}.launch_agent")


def agent_name(module):
    return module.__name__.split(".")[1]


def test_defaults_and_run_model_override(launcher):
    defaults = launcher._load_agent_config({})
    assert defaults["model"] == DEFAULTS[agent_name(launcher)]
    overridden = launcher._load_agent_config({"agent": {
        "model": "explicit-provider-model", "timeout_seconds": 123,
        "max_iterations": 5, "python_path": "/chosen/python",
    }})
    assert overridden["model"] == "explicit-provider-model"
    assert overridden["timeout_seconds"] == 123
    assert overridden["max_iterations"] == 5
    assert overridden["python_path"] == "/chosen/python"
    assert launcher._load_agent_config({}) == defaults


def test_null_model_explicitly_uses_cli_default(launcher):
    config = launcher._load_agent_config({"agent": {"model": None}})
    if agent_name(launcher) == "claude_code":
        cmd = launcher._build_command("cli", "prompt", config)
    else:
        cmd = launcher._build_command("cli", "/workspace", "prompt", config)
    assert "--model" not in cmd


@pytest.mark.parametrize("timeout", [0, -1, True, "10", 1.5, None])
def test_invalid_timeout_rejected_before_launch(launcher, timeout):
    with pytest.raises(ValueError, match="timeout_seconds"):
        launcher._load_agent_config({"agent": {"timeout_seconds": timeout}})


@pytest.mark.parametrize("model", ["", "  ", ["model"], False])
def test_invalid_model_config_rejected(launcher, model):
    with pytest.raises(ValueError, match="model"):
        launcher._load_agent_config({"agent": {"model": model}})


@pytest.mark.parametrize("value", [["model"], [], False, ""])
def test_invalid_run_agent_mapping_rejected(launcher, value):
    with pytest.raises(ValueError, match="mapping"):
        launcher._load_agent_config({"agent": value})


def test_codex_effort_is_literal_toml_string():
    module = importlib.import_module("agents.codex.launch_agent")
    effort = 'xhigh"\nmodel="other-model'
    config = module._load_agent_config({"agent": {"effort": effort}})
    cmd = module._build_command("codex", "/workspace", "prompt", config)
    value = next(arg for arg in cmd if arg.startswith("model_reasoning_effort="))
    assert tomllib.loads(value) == {"model_reasoning_effort": effort}
    assert "--effort" not in cmd


@pytest.mark.parametrize("budget", [0, -1, True, "1", float("nan"), float("inf")])
def test_invalid_claude_budget_rejected(budget):
    module = importlib.import_module("agents.claude_code.launch_agent")
    with pytest.raises(ValueError, match="max_budget_usd"):
        module._load_agent_config({"agent": {"max_budget_usd": budget}})


def test_cursor_effort_must_be_in_model_selection():
    module = importlib.import_module("agents.cursor.launch_agent")
    with pytest.raises(ValueError, match="parameterized model"):
        module._load_agent_config({"agent": {"effort": "high"}})
    config = module._load_agent_config({"agent": {
        "model": "claude-opus-5[context=1m,effort=high,fast=false]",
    }})
    cmd = module._build_command("cursor-agent", "/workspace", "prompt", config)
    assert cmd[cmd.index("--model") + 1] == config["model"]
    assert "--effort" not in cmd


@pytest.fixture
def fake_cli(tmp_path, monkeypatch, launcher):
    """A real subprocess records argv/env and emits the provider's JSON envelope."""
    bin_dir = tmp_path / "bin with spaces"
    bin_dir.mkdir()
    binary = bin_dir / "fake-cli"
    binary.write_text(f"#!{sys.executable}\n" + '''
import json, os, subprocess, sys, time
from pathlib import Path
if '--version' in sys.argv:
    print('fake-cli 1.0')
    raise SystemExit(0)
Path('invocation.json').write_text(json.dumps({
    'argv': sys.argv[1:], 'cwd': os.getcwd(),
    'python': os.environ.get('AGENT_KERNEL_ARENA_PYTHON'),
    'memory': os.environ.get('CLAUDE_CODE_DISABLE_AUTO_MEMORY'),
    'sandbox': os.environ.get('IS_SANDBOX'),
    'stdin': sys.stdin.read(),
}))
mode = os.environ.get('ARENA_TEST_CLI_MODE')
if mode == 'timeout':
    child = subprocess.Popen([sys.executable, '-c',
        "import os,signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "Path('child.pid').write_text(str(os.getpid())); time.sleep(60)"])
    time.sleep(60)
if mode == 'nonzero':
    print('Unsupported model', file=sys.stderr)
    raise SystemExit(7)
codex = sys.argv[1] == 'exec'
if mode == 'failed_event':
    event = ({'type':'turn.failed','error':{'message':'request failed'}} if codex
             else {'type':'result','is_error':True,'subtype':'error_max_budget_usd'})
    print(json.dumps(event))
    raise SystemExit(0)
if codex:
    print(json.dumps({'type':'item.completed', 'item':{
        'type':'agent_message','text':'ARENA_FAKE_OK'}}))
    print(json.dumps({'type':'turn.completed','usage':{}}))
else:
    print(json.dumps({'type':'assistant','message':{
        'content':[{'type':'text','text':'ARENA_FAKE_OK'}]}}))
    print(json.dumps({'type':'result','subtype':'success','is_error':False}))
''')
    binary.chmod(0o755)
    monkeypatch.setattr(launcher.shutil, "which", lambda _: str(binary))
    prompt = '--prompt-as-data $(touch SHELL_RAN) `touch SHELL_RAN`\nsecond line'
    monkeypatch.setattr(launcher, "load_prompt_builder", lambda *args: lambda *args: prompt)
    workspace = tmp_path / "workspace with spaces"
    workspace.mkdir()
    return workspace, prompt


def test_launch_forwards_literal_argv_and_run_settings(launcher, fake_cli, caplog):
    workspace, prompt = fake_cli
    name = agent_name(launcher)
    model = "provider-model[effort=high] $(touch MODEL_SHELL_RAN)"
    config = {"agent": {
        "template": name, "model": model, "max_iterations": None,
        "timeout_seconds": 10, "python_path": sys.executable,
    }}
    if name != "cursor":
        config["agent"]["effort"] = "medium"
    if name == "claude_code":
        config["agent"]["max_budget_usd"] = 0.5
    with caplog.at_level("INFO"):
        output = launcher.launch_agent(config, "never-read-a-task-config", str(workspace))
    invocation = json.loads((workspace / "invocation.json").read_text())
    args = invocation["argv"]
    assert args[args.index("--model") + 1] == model
    if name == "cursor":
        assert args[-2] == "--"
        delivered_prompt = args[-1]
    else:
        delivered_prompt = invocation["stdin"]
        assert prompt not in args
    assert delivered_prompt.startswith(prompt)
    assert "up to" not in delivered_prompt
    assert invocation["cwd"] == str(workspace)
    assert invocation["python"] == sys.executable
    assert not (workspace / "SHELL_RAN").exists()
    assert not (workspace / "MODEL_SHELL_RAN").exists()
    assert "second line" not in caplog.text  # Do not print the entire input prompt.
    assert "ARENA_FAKE_OK" in output
    if name == "codex":
        assert args[-2:] == ["--", "-"]
        assert 'model_reasoning_effort="medium"' in args
        assert "--ephemeral" in args
    if name == "claude_code":
        assert args[args.index("--input-format") + 1] == "text"
        assert args[args.index("--effort") + 1] == "medium"
        assert args[args.index("--max-budget-usd") + 1] == "0.5"
        assert invocation["memory"] == "1"
        assert invocation["sandbox"] == "1"
        assert "--no-session-persistence" in args
    if name == "cursor":
        assert "--trust" in args
        assert args[args.index("--workspace") + 1] == str(workspace)


@pytest.mark.parametrize("launcher", ["codex", "claude_code"], indirect=True)
def test_large_prompts_use_stdin_without_exec_argument_limits(launcher, fake_cli, monkeypatch):
    workspace, _ = fake_cli
    prompt = "literal 汉字 `x` $(x)\n" * 20000
    monkeypatch.setattr(launcher, "load_prompt_builder", lambda *args: lambda *args: prompt)
    launcher.launch_agent({"agent": {"max_iterations": None, "timeout_seconds": 10}},
                          "unused", str(workspace))
    invocation = json.loads((workspace / "invocation.json").read_text())
    assert invocation["stdin"].startswith(prompt.rstrip())
    assert sum(len(arg.encode()) for arg in invocation["argv"]) < 4096


@pytest.mark.parametrize("mode", ["nonzero", "failed_event"])
def test_cli_failure_is_not_returned_as_success(launcher, fake_cli, monkeypatch, mode):
    workspace, _ = fake_cli
    monkeypatch.setenv("ARENA_TEST_CLI_MODE", mode)
    with pytest.raises(RuntimeError, match="exited with code 7|failed"):
        launcher.launch_agent({}, "unused", str(workspace))


def test_timeout_stops_tool_child_and_reports_failure(launcher, fake_cli, monkeypatch):
    workspace, _ = fake_cli
    monkeypatch.setenv("ARENA_TEST_CLI_MODE", "timeout")
    pid = None
    try:
        with pytest.raises(TimeoutError, match="timed out"):
            launcher.launch_agent({"agent": {"timeout_seconds": 1}}, "unused", str(workspace))
        pid = int((workspace / "child.pid").read_text())
        # An adopted child may briefly remain a zombie until init reaps it.
        for _ in range(50):
            stat = Path(f"/proc/{pid}/stat")
            if not stat.exists() or stat.read_text().split()[2] == "Z":
                break
            time.sleep(0.02)
        else:
            pytest.fail("agent's tool child survived the timeout")
    finally:
        if pid is None and (workspace / "child.pid").exists():
            pid = int((workspace / "child.pid").read_text())
        if pid is not None:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
