"""Exercise CLI argv, run overrides, failures, and cleanup without GPU or auth."""

import importlib
import json
import os
from pathlib import Path
import signal
import sys
import time
try:
    import tomllib
except ModuleNotFoundError:  # pytest depends on tomli for Python < 3.11.
    import tomli as tomllib

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
    print(os.environ.get('FAKE_CLI_VERSION', 'fake-cli 1.0'))
    raise SystemExit(0)
record = json.dumps({
    'argv': sys.argv[1:], 'cwd': os.getcwd(),
    'python': os.environ.get('AGENT_KERNEL_ARENA_PYTHON'),
    'memory': os.environ.get('CLAUDE_CODE_DISABLE_AUTO_MEMORY'),
    'sandbox': os.environ.get('IS_SANDBOX'),
    'stdin': sys.stdin.read(),
})
Path('invocation.json').write_text(record)
with open('invocations.jsonl', 'a') as log:
    log.write(record + '\\n')
n = len(Path('invocations.jsonl').read_text().splitlines())
mode = os.environ.get('ARENA_TEST_CLI_MODE')
if mode in ('trace_timeout', 'trace_nonzero'):
    print(json.dumps({'type':'item.completed','item':{
        'type':'command_execution','command':'test-kernel','status':'completed',
        'aggregated_output':'x' * 2000 + 'FULL_TOOL_OUTPUT_TAIL'}}), flush=True)
    print('FULL_ERROR_DETAIL ' + os.environ.get('ARENA_TEST_API_KEY', ''), file=sys.stderr, flush=True)
    if mode == 'trace_timeout':
        time.sleep(60)
    raise SystemExit(7)
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
    # Codex reports the thread's running token total on every turn.completed.
    print(json.dumps({'type':'thread.started','thread_id':'fake-thread'}))
    print(json.dumps({'type':'item.completed', 'item':{
        'type':'agent_message','text':'ARENA_FAKE_OK'}}))
    print(json.dumps({'type':'turn.completed','usage':{
        'input_tokens':100 * n,'cached_input_tokens':0,'output_tokens':10 * n}}))
elif '--trust' in sys.argv:
    # Cursor reports this turn's tokens in camelCase and no dollars.
    print(json.dumps({'type':'assistant','message':{
        'content':[{'type':'text','text':'ARENA_FAKE_OK'}]}}))
    print(json.dumps({'type':'result','subtype':'success','is_error':False,
                      'session_id':'fake-session',
                      'usage':{'inputTokens':100,'outputTokens':10,'cacheReadTokens':5}}))
else:
    # Claude Code: total_cost_usd is a running session total on newer CLIs.
    print(json.dumps({'type':'assistant','message':{
        'content':[{'type':'text','text':'ARENA_FAKE_OK'}]}}))
    print(json.dumps({'type':'result','subtype':'success','is_error':False,
                      'session_id':'fake-session','total_cost_usd':0.25 * n,
                      'usage':{'input_tokens':100,'output_tokens':10}}))
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
    # The actual subprocess must see the run override, not a task timeout or
    # the launcher's default. All three CLI transports share this guidance.
    assert "budget for this agent invocation is 10 seconds" in delivered_prompt
    assert "task's declared candidate paths" in delivered_prompt
    assert "3600 seconds" not in delivered_prompt
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


@pytest.mark.parametrize("launcher", ["codex"], indirect=True)
@pytest.mark.parametrize("mode", ["trace_timeout", "trace_nonzero"])
def test_codex_preserves_full_private_events_after_failure(launcher, fake_cli, monkeypatch, mode):
    workspace, prompt = fake_cli
    secret = "synthetic-private-credential"
    monkeypatch.setenv("ARENA_TEST_CLI_MODE", mode)
    monkeypatch.setenv("ARENA_TEST_API_KEY", secret)
    with pytest.raises((TimeoutError, RuntimeError)):
        launcher.launch_agent({"agent": {"timeout_seconds": 1}}, "unused", str(workspace))
    directory, = workspace.parent.glob(f".{workspace.name}-codex-stream-*")
    output = (directory / "stdout.jsonl").read_text()
    errors = (directory / "stderr.jsonl").read_text()
    assert "FULL_TOOL_OUTPUT_TAIL" in output
    assert "FULL_ERROR_DETAIL" in errors
    assert secret not in output + errors and "[REDACTED]" in errors
    assert prompt not in output + errors
    assert directory.stat().st_mode & 0o777 == 0o700
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in directory.iterdir())


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


def _invocations(workspace):
    return [json.loads(line) for line in (workspace / "invocations.jsonl").read_text().splitlines()]


def _delivered_prompt(name, invocation):
    return invocation["argv"][-1] if name == "cursor" else invocation["stdin"]


def _launch_collecting(launcher, config, workspace):
    from src.tools.agent_cost import AgentCostCollector

    collector = AgentCostCollector(agent=agent_name(launcher), task_name="suite/task",
                                   run_settings=config["agent"])
    collector.wrap(launcher.launch_agent)(eval_config=config, task_config_dir="unused",
                                          workspace=str(workspace))
    return collector.attempt()


def test_iterations_mode_is_unchanged(launcher, fake_cli):
    workspace, prompt = fake_cli
    name = agent_name(launcher)
    config = {"agent": {"max_iterations": 3, "timeout_seconds": 10, "python_path": sys.executable}}
    attempt = _launch_collecting(launcher, config, workspace)
    runs = _invocations(workspace)
    assert len(runs) == 1
    expected = prompt.rstrip() + "\n\nFor this optimization, you must iterate up to 3 versions."
    assert _delivered_prompt(name, runs[0]).startswith(expected + "\n\nThe total wall-clock budget")
    assert "no fixed number" not in _delivered_prompt(name, runs[0])
    args = runs[0]["argv"]
    assert "--resume" not in args and "resume" not in args
    if name == "claude_code":
        assert "--no-session-persistence" in args and "--session-id" not in args
    if name == "codex":
        assert "--ephemeral" in args
    # One invocation is still recorded through the cost tool.
    assert len(attempt["invocations"]) == 1
    assert attempt["usage"]["input_tokens"] == 100


def test_time_mode_resumes_the_session_and_counts_cost_once(launcher, fake_cli, monkeypatch):
    from src.tools import agent_time_budget

    workspace, _ = fake_cli
    name = agent_name(launcher)
    monkeypatch.setattr(agent_time_budget, "MIN_RESUME_SECONDS", 0)
    monkeypatch.setattr(agent_time_budget, "MAX_SESSIONS", 3)
    monkeypatch.setenv("FAKE_CLI_VERSION", "2.1.290 (Claude Code)")
    config = {"agent": {"budget_mode": "time", "max_iterations": 3, "timeout_seconds": 60}}
    attempt = _launch_collecting(launcher, config, workspace)
    runs = _invocations(workspace)
    assert len(runs) == 3
    first = runs[0]["argv"]
    first_prompt = _delivered_prompt(name, runs[0])
    assert "up to 3 versions" not in first_prompt
    assert "no fixed number of versions" in first_prompt
    for later in runs[1:]:
        args = later["argv"]
        assert "minutes of the wall-clock budget remain" in _delivered_prompt(name, later)
        if name == "claude_code":
            assert args[args.index("--resume") + 1] == first[first.index("--session-id") + 1]
            assert "--no-session-persistence" not in args
        elif name == "cursor":
            assert args[args.index("--resume") + 1] == "fake-session"
        else:
            assert args[args.index("resume") + 1] == "fake-thread"
            assert args[-2:] == ["--", "-"]
            assert "--ephemeral" not in args
    assert len(attempt["invocations"]) == 3
    if name == "claude_code":
        # Running totals 0.25, 0.50, 0.75 for one session: spend is 0.75, not 1.50.
        assert attempt["total_cost_usd"] == 0.75 and attempt["cost_status"] == "complete"
        assert attempt["usage"]["input_tokens"] == 300
    elif name == "codex":
        # Thread totals 100, 200, 300 input tokens: usage is 300, not 600.
        assert attempt["usage"] == {"input_tokens": 300, "cached_input_tokens": 0, "output_tokens": 30}
        assert attempt["total_cost_usd"] is None and attempt["cost_status"] == "not_reported"
    else:
        assert attempt["usage"] == {"input_tokens": 300, "output_tokens": 30,
                                    "cache_read_input_tokens": 15}
        assert attempt["total_cost_usd"] is None and attempt["cost_status"] == "not_reported"


@pytest.mark.parametrize("launcher", ["claude_code"], indirect=True)
def test_claude_cost_with_unknown_cli_version_is_not_guessed_after_resume(launcher, fake_cli, monkeypatch):
    from src.tools import agent_time_budget

    workspace, _ = fake_cli
    monkeypatch.setattr(agent_time_budget, "MIN_RESUME_SECONDS", 0)
    monkeypatch.setattr(agent_time_budget, "MAX_SESSIONS", 2)
    attempt = _launch_collecting(
        launcher, {"agent": {"budget_mode": "time", "timeout_seconds": 60}}, workspace)
    assert len(attempt["invocations"]) == 2
    assert attempt["total_cost_usd"] is None and attempt["cost_status"] == "ambiguous"


def test_time_mode_does_not_resume_without_budget_left(launcher, fake_cli):
    workspace, _ = fake_cli
    launcher.launch_agent({"agent": {"budget_mode": "time", "timeout_seconds": 60}},
                          "unused", str(workspace))
    assert len(_invocations(workspace)) == 1


@pytest.mark.parametrize("value", ["tokens", "", False])
def test_invalid_budget_mode_rejected(launcher, value):
    with pytest.raises(ValueError, match="budget_mode"):
        launcher._load_agent_config({"agent": {"budget_mode": value}})
