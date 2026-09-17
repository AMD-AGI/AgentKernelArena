"""Real Git and process cleanup regressions; CPU only, no model or GPU calls."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

from agents.forge import adapter, bridge, common
from agents.forge.bundles import copy_workspace
from agents.forge.upstream import protected_agent_paths
from test_forge_v2 import ROOT, fixture_task, mock_engine


@pytest.mark.parametrize("rewrite", [False, True])
def test_protected_inventory_uses_engine_or_lane_root(tmp_path, monkeypatch, rewrite):
    context, plan, _ = fixture_task(tmp_path)
    engine = Path(plan["engine_root"])
    (engine / "arena_forge_driver.py").write_text("# generated driver\n")
    adapter._initialize_git(engine)
    if rewrite:
        plan["workflow"] = "rewrite"
        attempt = engine / ".forge_rewrite/current"
        copy_workspace(Path(plan["template"]), attempt)
        monkeypatch.setenv("KERNELFORGE_REWRITE_CANDIDATE_KERNEL", str(attempt / plan["anchor"]))
        with pytest.raises(ValueError, match="Git root"):
            protected_agent_paths(plan, context.spec, attempt)
    files, protected = protected_agent_paths(plan, context.spec, engine)
    assert str(engine / "config.yaml") in protected
    assert str(engine / "runner.py") in protected
    assert str(engine / "arena_forge_driver.py") in protected
    assert not set(map(str, files.values())) & set(protected)
    lane = tmp_path / "lane"
    subprocess.run(["git", "clone", "--quiet", str(engine), str(lane)], check=True)
    if rewrite:
        copy_workspace(attempt, lane / ".forge_rewrite/current")
    _, protected = protected_agent_paths(plan, context.spec, lane)
    assert str(lane / "runner.py") in protected
    assert not any(str(engine) in name for name in protected)


@pytest.mark.parametrize("missing", ["config.yaml", "runner.py", "arena_forge_driver.py"])
def test_untracked_protected_file_rejected(tmp_path, missing):
    context, plan, _ = fixture_task(tmp_path)
    engine = Path(plan["engine_root"])
    (engine / "arena_forge_driver.py").write_text("# generated driver\n")
    adapter._initialize_git(engine)
    subprocess.run(["git", "rm", "--cached", missing], cwd=engine, check=True, capture_output=True)
    with pytest.raises(ValueError, match="protected files are not tracked"):
        protected_agent_paths(plan, context.spec, engine)


@pytest.mark.parametrize("exit_mode", ["normal", "error", "sigterm"])
def test_campaign_reaps_killed_driver_children_before_cleaning_copies(tmp_path, exit_mode):
    context, plan, path = fixture_task(tmp_path)
    pool = tmp_path / "evaluation-workspaces"
    pool.mkdir()
    plan["evaluation_root"] = str(pool)
    path.write_text(json.dumps(plan))
    old = tmp_path / "evaluate-old-experiment"
    old.mkdir()
    (old / "keep.txt").write_text("previous experiment")
    (tmp_path / "engine.log").write_text("retain diagnostics")
    marker = tmp_path / "worker.pid"
    driver = r'''
import subprocess, sys, time
from pathlib import Path
from agents.forge.bridge import evaluation_workspace, load_plan
from agents.forge.task_context import TaskContext
plan = load_plan(Path(sys.argv[1]))
with evaluation_workspace(TaskContext.load(plan["context"]), plan, Path(plan["engine_root"]), "candidate"):
    worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
    Path(sys.argv[2]).write_text(str(worker.pid))
    time.sleep(120)
'''
    supervisor = r'''
import os, signal, subprocess, sys, time
from pathlib import Path
from agents.forge.bridge import load_plan, cleanup_evaluation_workspaces
from agents.forge.process_tree import managed_children
plan = load_plan(Path(sys.argv[1]))
def clean():
    worker_pid = int(Path(sys.argv[2]).read_text())
    assert not Path(f"/proc/{worker_pid}").exists(), "cleanup raced a detached task worker"
    cleanup_evaluation_workspaces(plan)
with managed_children(after_stop=clean):
    driver = subprocess.Popen([sys.executable, "-c", sys.argv[4], sys.argv[1], sys.argv[2]], start_new_session=True)
    deadline = time.monotonic() + 10
    while not Path(sys.argv[2]).exists() and time.monotonic() < deadline:
        time.sleep(.02)
    assert Path(sys.argv[2]).exists()
    driver.kill()
    driver.wait(timeout=5)
    assert list(Path(plan["evaluation_root"]).glob("evaluate-*/task")), "SIGKILL must leave a real copy"
    if sys.argv[3] == "error":
        raise RuntimeError("campaign failure")
    if sys.argv[3] == "sigterm":
        os.kill(os.getpid(), signal.SIGTERM)
'''
    result = subprocess.run([sys.executable, "-c", supervisor, str(path), str(marker), exit_mode, driver],
                            cwd=ROOT, capture_output=True, text=True, timeout=20)
    assert (result.returncode == 0) == (exit_mode == "normal"), result.stderr
    assert not pool.exists(), result.stderr
    assert not Path(f"/proc/{int(marker.read_text())}").exists()
    assert (old / "keep.txt").read_text() == "previous experiment"
    assert (tmp_path / "engine.log").read_text() == "retain diagnostics"
    for root in (context.workspace, context.baseline_workspace, Path(plan["template"]), Path(plan["engine_root"])):
        assert (root / "source/kernel.py").read_text() == "2"


def test_cleanup_refuses_symlink_or_other_directory(tmp_path):
    _, plan, _ = fixture_task(tmp_path)
    valuable = tmp_path / "valuable"
    valuable.mkdir()
    (valuable / "keep").touch()
    plan["evaluation_root"] = str(valuable)
    with pytest.raises(ValueError, match="cleanup directory"):
        bridge.cleanup_evaluation_workspaces(plan)
    pool = tmp_path / "evaluation-workspaces"
    pool.symlink_to(valuable, target_is_directory=True)
    plan["evaluation_root"] = str(pool)
    with pytest.raises(ValueError, match="cleanup directory"):
        bridge.cleanup_evaluation_workspaces(plan)
    assert (valuable / "keep").exists()


def test_launcher_reclaims_copies_when_slow_cleanup_supervisor_is_killed(tmp_path, monkeypatch):
    context, _, _ = fixture_task(tmp_path, language="hip")
    monkeypatch.setenv("ARENA_TASK_CONTEXT", str(context.path))
    mock_engine(monkeypatch)
    old = tmp_path / "evaluate-old-experiment"
    old.mkdir()
    (old / "keep.txt").write_text("previous experiment")
    terminate = common._terminate_process_group
    monkeypatch.setattr(common, "_terminate_process_group", lambda process, logger: terminate(
        process, logger, term_timeout=.5, kill_timeout=3))
    supervisor = r'''
import os, subprocess, sys, time
from pathlib import Path
from agents.forge import bridge
from agents.forge.bundles import copy_workspace
from agents.forge.process_tree import managed_children
plan = bridge.load_plan(Path(os.environ["ARENA_FORGE_PLAN"]))
artifact = Path(plan["template"]).parent
copy_workspace(Path(plan["template"]), Path(plan["evaluation_root"]) / "evaluate-abandoned/task")
def slow_remove(root):
    assert not Path(f"/proc/{worker.pid}").exists(), "cleanup raced a detached worker"
    (Path(root) / "evaluate-abandoned/task/source/helper.py").unlink()
    (artifact / "cleanup-started").write_text("all workers reaped")
    time.sleep(60)
bridge.shutil.rmtree = slow_remove
with managed_children(after_stop=lambda: bridge.cleanup_evaluation_workspaces(plan)):
    worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    (artifact / "worker.pid").write_text(str(worker.pid))
    print("engine started", flush=True)
    time.sleep(60)
'''
    observed = []

    def run(command, *, workspace, env, **kwargs):
        result = common.run_forge_subprocess(
            [sys.executable, "-c", supervisor], workspace=workspace, env=env,
            timeout_seconds=2, logger=kwargs["logger"])
        process, _, _, timed_out = result
        plan = bridge.load_plan(Path(env["ARENA_FORGE_PLAN"]))
        artifact = Path(plan["template"]).parent
        assert timed_out and process.returncode == -signal.SIGKILL
        assert (artifact / "cleanup-started").exists()
        assert (Path(plan["evaluation_root"]) / "evaluate-abandoned/task").is_dir()
        observed.append(plan)
        return result

    monkeypatch.setattr(adapter, "run_forge_subprocess", run)
    with pytest.raises(adapter.ForgeRunError, match="exceeded its shared deadline"):
        adapter.launch({"agent": {"timeout_seconds": 30}}, "unused", str(context.workspace))
    plan, = observed
    artifact = Path(plan["template"]).parent
    assert not Path(plan["evaluation_root"]).exists()
    assert not Path(f"/proc/{int((artifact / 'worker.pid').read_text())}").exists()
    assert (old / "keep.txt").read_text() == "previous experiment"
    assert "engine started" in (artifact / "engine.log").read_text()
    status = json.loads((artifact / "arena_forge_status.json").read_text())
    assert status["status"] == "FAILED" and status["timed_out"]
    assert status["exit_code"] == -signal.SIGKILL
    for root in (context.workspace, context.baseline_workspace,
                 Path(plan["template"]), Path(plan["engine_root"])):
        assert (root / "source/kernel.py").read_text() == "2"


@pytest.mark.parametrize("problem", ["missing", "partial", "wrong_pid", "other_run", "live"])
def test_recovery_requires_matching_reaped_supervisor_receipt(tmp_path, problem):
    _, plan, _ = fixture_task(tmp_path)
    pool = tmp_path / "evaluation-workspaces"
    pool.mkdir()
    (pool / "keep.txt").write_text("unconfirmed disposable copy")
    plan["evaluation_root"] = str(pool)
    marker = tmp_path / "evaluation-workspaces-reaped.json"
    receipt = {"supervisor_pid": os.getpid(), "evaluation_root": str(pool)}
    if problem == "wrong_pid":
        receipt["supervisor_pid"] += 1
    if problem == "other_run":
        receipt["evaluation_root"] = str(tmp_path / "other-experiment/evaluation-workspaces")
    if problem != "missing":
        marker.write_text("{" if problem == "partial" else json.dumps(receipt))
    process = SimpleNamespace(pid=os.getpid(), poll=lambda: None if problem == "live" else -signal.SIGKILL)
    assert bridge.cleanup_evaluation_workspaces(plan, supervisor=process) is False
    assert (pool / "keep.txt").read_text() == "unconfirmed disposable copy"


@pytest.mark.parametrize("recover", [False, True])
def test_cleanup_refuses_symlink_receipt(tmp_path, recover):
    _, plan, _ = fixture_task(tmp_path)
    pool = tmp_path / "evaluation-workspaces"
    pool.mkdir()
    plan["evaluation_root"] = str(pool)
    valuable = tmp_path / "valuable.json"
    contents = json.dumps({"supervisor_pid": os.getpid(), "evaluation_root": str(pool)})
    valuable.write_text(contents)
    (tmp_path / "evaluation-workspaces-reaped.json").symlink_to(valuable)
    process = SimpleNamespace(pid=os.getpid(), poll=lambda: -signal.SIGKILL) if recover else None
    with pytest.raises(ValueError, match="cleanup receipt"):
        bridge.cleanup_evaluation_workspaces(plan, supervisor=process)
    assert pool.exists()
    assert valuable.read_text() == contents
