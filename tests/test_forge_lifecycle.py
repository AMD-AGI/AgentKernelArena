"""Real Git and process cleanup regressions; CPU only, no model or GPU calls."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from agents.forge import adapter, bridge
from agents.forge.bundles import copy_workspace
from agents.forge.upstream import protected_agent_paths
from test_forge_v2 import ROOT, fixture_task


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
