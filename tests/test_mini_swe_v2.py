"""Capability rejection and legacy process isolation; no model or GPU runtime."""
import importlib
import json
import logging
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import pytest
import yaml

from src.task_spec import TaskSpec


mini = importlib.import_module("agents.mini_swe_triton.launch_agent")


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    for name in ("ARENA_TASK_CONTEXT", "GEAK_SRC", "GEAK_GPU_IDS"):
        monkeypatch.delenv(name, raising=False)


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes()
            for p in root.rglob("*") if p.is_file()}


def write_task(tmp_path, config):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    path = workspace / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return workspace, path


def no_process(*args, **kwargs):
    pytest.fail("Unsupported task/runtime must not start any subprocess")


def test_registered_launcher_keeps_the_explicit_capability_gate():
    from src.module_registration import AgentType, load_agent_launcher
    selected = load_agent_launcher(AgentType.MINI_SWE_TRITON, logging.getLogger(__name__))
    assert selected is mini.launch_agent


@pytest.mark.parametrize("language", ["triton", "hip", "flydsl"])
@pytest.mark.parametrize("state", ["implemented", "unimplemented"])
@pytest.mark.parametrize("scope", ["file", "symbols", "tree"])
def test_v2_rejected_before_default_files_or_processes(tmp_path, monkeypatch, language, state, scope):
    edit = {"path": "source" if scope == "tree" else "source/nested/kernel.py", "scope": scope}
    if scope == "symbols":
        edit.update(symbols=["compute"], allow_new_helpers=True)
    candidate = {"language": language, "initial_state": state,
                 "editable": [edit],
                 "entrypoints": [{"file": "source/nested/kernel.py", "kind": "function",
                                  "symbol": "compute"}]}
    if scope != "tree":
        candidate["editable"].append("source/helpers/math.py")
    config = {"schema_version": 2, "candidate": candidate,
              "evaluation": {"runner": ["python3", "scripts/evaluate.py"]}}
    TaskSpec.from_mapping(config, task_id="arbitrary/task")  # Real public-schema fixture.
    workspace, path = write_task(tmp_path, config)
    # No candidate files exist: the capability error must precede file inference.
    before = snapshot(tmp_path)
    monkeypatch.setattr(mini.subprocess, "Popen", no_process)
    monkeypatch.setattr(mini.subprocess, "run", no_process)
    with pytest.raises(mini.MiniSweCapabilityError, match="MINI_SWE_V2_UNSUPPORTED"):
        mini.launch_agent({"_task_id": "arbitrary/task"}, str(path), str(workspace))
    assert snapshot(tmp_path) == before
    assert not (workspace / ".git").exists()
    assert list(tmp_path.iterdir()) == [workspace]


@pytest.mark.parametrize("version", [0, 1, 3, "2", True, None])
def test_versioned_config_cannot_fall_back_to_legacy(tmp_path, monkeypatch, version):
    workspace, path = write_task(tmp_path, {"schema_version": version})
    monkeypatch.setattr(mini.subprocess, "Popen", no_process)
    with pytest.raises(mini.MiniSweCapabilityError, match="MINI_SWE_V2_UNSUPPORTED"):
        mini.launch_agent({}, str(path), str(workspace))


@pytest.mark.parametrize("field", ["candidate", "evaluation", "context"])
def test_stripped_schema_or_framework_context_does_not_enable_legacy(tmp_path, monkeypatch, field):
    config = {field: {}} if field != "context" else {}
    if field == "context":
        monkeypatch.setenv("ARENA_TASK_CONTEXT", str(tmp_path / "external-context.json"))
    workspace, path = write_task(tmp_path, config)
    monkeypatch.setattr(mini.subprocess, "Popen", no_process)
    with pytest.raises(mini.MiniSweCapabilityError, match="MINI_SWE_V2_UNSUPPORTED"):
        mini.launch_agent({}, str(path), str(workspace))


@pytest.mark.parametrize("config", [None, [], "task"])
def test_nonmapping_config_rejected(tmp_path, config):
    workspace, path = write_task(tmp_path, config)
    with pytest.raises(ValueError, match="must be a mapping"):
        mini.launch_agent({}, str(path), str(workspace))


@pytest.mark.parametrize("runtime", ["unset", "empty", "workflow", "relative"])
def test_missing_legacy_module_fails_before_workspace_mutation(tmp_path, monkeypatch, runtime):
    workspace, path = write_task(tmp_path, {})
    source = tmp_path / "upstream"
    source.mkdir()
    if runtime == "workflow":
        (source / "geak").mkdir()
        (source / "geak/__init__.py").write_text("")
    if runtime != "unset":
        monkeypatch.setenv("GEAK_SRC", "upstream" if runtime == "relative" else str(source))
    monkeypatch.setattr(mini.subprocess, "Popen", no_process)
    monkeypatch.setattr(mini.subprocess, "run", no_process)
    before = snapshot(tmp_path)
    with pytest.raises(mini.MiniSweCapabilityError, match="MINI_SWE_RUNTIME_UNAVAILABLE"):
        mini.launch_agent({}, str(path), str(workspace))
    assert snapshot(tmp_path) == before
    assert not (workspace / ".git").exists()


# This only exercises OS process/argv behavior; it is not an upstream agent,
# a scripted model, a task runner, or an optimization qualification fixture.
LOCAL_CLI = '''import argparse, json, os, pathlib, sys
p = argparse.ArgumentParser()
for flag in ("task", "test-command", "repo", "num-parallel", "gpu-ids", "model", "cost-limit"):
    p.add_argument("--" + flag, required=True)
p.add_argument("--yolo", action="store_true")
p.add_argument("--exit-immediately", action="store_true")
p.add_argument("-o", required=True)
a = p.parse_args()
out = pathlib.Path(a.o)
(out / "argv.json").write_text(json.dumps(vars(a)))
print("local CLI stdout", flush=True)
print("local CLI stderr", file=sys.stderr, flush=True)
if os.environ.get("MINI_TEST_FAIL"):
    (out / "unselected.patch").write_text("unselected patch")
    raise SystemExit(7)
if os.environ.get("MINI_TEST_DELIVER"):
    (pathlib.Path(a.repo) / os.environ["MINI_TEST_TARGET"]).write_text("delivered = 2\\n")
'''


@pytest.fixture
def legacy(tmp_path, monkeypatch):
    source = tmp_path / "legacy source"
    package = source / "minisweagent/run"
    package.mkdir(parents=True)
    (source / "minisweagent/__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "mini.py").write_text(LOCAL_CLI)
    monkeypatch.setenv("GEAK_SRC", str(source))
    # Use the same Python interpreter, without installing any agent dependency.
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "python3").symlink_to(sys.executable)
    monkeypatch.setenv("PATH", str(bindir) + os.pathsep + os.environ["PATH"])
    configdir = tmp_path / "agent"
    configdir.mkdir()
    (configdir / "agent_config.yaml").write_text(yaml.safe_dump({
        "timeout_seconds": 10, "agent": {"num_parallel": 1,
        "model": "literal'; touch MODEL_INJECTION; #"}, "geak_env": {}}))
    monkeypatch.setattr(mini, "__file__", str(configdir / "launch_agent.py"))
    config = {"source_file_path": ["source with space/nested/actual.py"],
              "harness_path": "checks/eval's harness.py"}
    workspace, config_path = write_task(tmp_path, config)
    for name in (config["source_file_path"][0], config["harness_path"]):
        p = workspace / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("original = 1\n")
    monkeypatch.setenv("MINI_TEST_TARGET", config["source_file_path"][0])
    return workspace, config_path, config


def test_legacy_keeps_nested_paths_literal_argv_and_own_delivery(legacy, monkeypatch):
    workspace, config_path, config = legacy
    monkeypatch.setenv("MINI_TEST_DELIVER", "1")
    sibling = workspace.parent / "other-run"
    sibling.mkdir()
    (sibling / "kernel.py").write_text("unrelated = 99\n")
    old_logs = workspace.parent / "workspace_logs"
    old_logs.mkdir()
    (old_logs / "best.patch").write_text("old report must survive")
    result = mini.launch_agent({"gpu_ids": "0", "target_gpu_model": "MI355X"},
                               str(config_path), str(workspace))
    assert result == "local CLI stdout"
    assert (workspace / config["source_file_path"][0]).read_text() == "delivered = 2\n"
    assert (workspace / config["harness_path"]).read_text() == "original = 1\n"
    assert (sibling / "kernel.py").read_text() == "unrelated = 99\n"
    assert (old_logs / "best.patch").read_text() == "old report must survive"
    logs = next(workspace.parent.glob("workspace_mini_*"))
    argv = json.loads((logs / "argv.json").read_text())
    assert argv["repo"] == str(workspace)
    assert argv["model"] == "literal'; touch MODEL_INJECTION; #"
    assert not (workspace / "MODEL_INJECTION").exists()
    parts = shlex.split(argv["test_command"])
    assert parts == ["python3", config["harness_path"], "--correctness", "&&",
                     "python3", config["harness_path"], "--full-benchmark", "--iterations", "30"]
    prompt = Path(argv["task"]).read_text()
    assert config["source_file_path"][0] in prompt and config["harness_path"] in prompt
    assert "MI355X" in prompt and "304 CUs" not in prompt
    assert (logs / "stdout.log").read_text() == "local CLI stdout"


def test_zero_exit_does_not_collect_old_or_sibling_outputs_and_logs_are_fresh(legacy):
    workspace, config_path, config = legacy
    other = workspace.parent / "unrelated"
    other.mkdir()
    (other / "kernel.py").write_text("not our candidate")
    before = (workspace / config["source_file_path"][0]).read_bytes()
    mini.launch_agent({}, str(config_path), str(workspace))
    first = next(workspace.parent.glob("workspace_mini_*"))
    (first / "kernel.py").write_text("not selected")
    (first / "best.patch").write_text("not selected")
    saved = snapshot(first)
    mini.launch_agent({}, str(config_path), str(workspace))
    assert len(list(workspace.parent.glob("workspace_mini_*"))) == 2
    assert snapshot(first) == saved
    assert (workspace / config["source_file_path"][0]).read_bytes() == before


def test_subprocess_failure_raises_without_patch_or_sibling_delivery(legacy, monkeypatch):
    workspace, config_path, config = legacy
    monkeypatch.setenv("MINI_TEST_FAIL", "1")
    sibling = workspace.parent / "unrelated"
    sibling.mkdir()
    (sibling / "kernel.py").write_text("not our candidate")
    before = (workspace / config["source_file_path"][0]).read_bytes()
    with pytest.raises(RuntimeError, match="exited with code 7"):
        mini.launch_agent({}, str(config_path), str(workspace))
    assert (workspace / config["source_file_path"][0]).read_bytes() == before
    logs = next(workspace.parent.glob("workspace_mini_*"))
    assert (logs / "unselected.patch").exists()
    assert (logs / "stderr.log").read_text() == "local CLI stderr"


def test_git_setup_failure_prevents_agent_launch(legacy, monkeypatch):
    workspace, config_path, _ = legacy
    def fail_git(cmd, **kwargs):
        assert kwargs["check"] is True and kwargs["timeout"] == 60
        raise subprocess.CalledProcessError(2, cmd, stderr="git fixture failed")
    monkeypatch.setattr(mini.subprocess, "run", fail_git)
    monkeypatch.setattr(mini, "_run_step", no_process)
    with pytest.raises(subprocess.CalledProcessError):
        mini.launch_agent({}, str(config_path), str(workspace))


@pytest.mark.parametrize("sources", [[], ["a.py", "b.py"], "a.py"])
def test_legacy_multifile_or_ambiguous_source_is_rejected(tmp_path, sources):
    with pytest.raises(mini.MiniSweCapabilityError, match="exactly one"):
        mini._legacy_paths({"source_file_path": sources}, tmp_path)


@pytest.mark.parametrize("path", ["../outside.py", "/outside.py", "a/../../outside.py"])
def test_legacy_candidate_path_cannot_escape(tmp_path, path):
    with pytest.raises(ValueError, match="task-relative"):
        mini._legacy_paths({"source_file_path": [path]}, tmp_path)


def test_legacy_symlink_escape_is_rejected(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("outside")
    (workspace / "kernel.py").symlink_to(outside)
    with pytest.raises(ValueError, match="within workspace"):
        mini._legacy_paths({}, workspace)
    assert outside.read_text() == "outside"


@pytest.mark.parametrize("harness", ["../outside.py", "kernel.py"])
def test_legacy_harness_escape_or_colocation_is_rejected(tmp_path, harness):
    (tmp_path / "kernel.py").write_text("value = 1\n")
    with pytest.raises(ValueError):
        mini._legacy_paths({"harness_path": harness}, tmp_path)


def test_timeout_kills_and_reaps_ordinary_descendants(tmp_path):
    pidfile = tmp_path / "child.pid"
    script = ("import pathlib, subprocess, sys, time\n"
              "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
              f"pathlib.Path({str(pidfile)!r}).write_text(str(p.pid))\n"
              "print('before timeout', flush=True)\n"
              "time.sleep(30)\n")
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired) as failure:
        mini._run_step([sys.executable, "-c", script], env=dict(os.environ), cwd=str(tmp_path),
                       label="CPU timeout fixture", logger=logging.getLogger(__name__), timeout=0.5)
    assert time.monotonic() - started < 5
    assert "before timeout" in failure.value.stdout
    stat = Path("/proc") / pidfile.read_text() / "stat"
    if stat.exists():
        assert stat.read_text().rsplit(")", 1)[1].split()[0] == "Z"


def test_launcher_timeout_preserves_logs_and_raises(legacy, monkeypatch):
    workspace, config_path, _ = legacy
    def timeout(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 1, output="partial stdout", stderr="partial stderr")
    monkeypatch.setattr(mini, "_run_step", timeout)
    with pytest.raises(RuntimeError, match="timed out"):
        mini.launch_agent({}, str(config_path), str(workspace))
    logs = next(workspace.parent.glob("workspace_mini_*"))
    assert (logs / "stdout.log").read_text() == "partial stdout"
    assert (logs / "stderr.log").read_text() == "partial stderr"
