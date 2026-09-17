"""Real Git and process cleanup regressions; CPU only, no model or GPU calls."""
from pathlib import Path
import subprocess

import pytest

from agents.forge import adapter, bridge
from agents.forge.bundles import copy_workspace
from agents.forge.upstream import protected_agent_paths
from test_forge_v2 import fixture_task


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


@pytest.mark.parametrize("rewrite", [False, True])
def test_measurement_uses_the_engine_tree_and_frozen_baseline(tmp_path, monkeypatch, rewrite):
    """No private per-invocation copy; a rewrite candidate leaves no shadow."""
    context, plan, _ = fixture_task(tmp_path)
    engine = Path(plan["engine_root"])
    anchor = plan["anchor"]
    if rewrite:
        plan["workflow"] = "rewrite"
        attempt = engine / ".forge_rewrite/current"
        copy_workspace(Path(plan["template"]), attempt)
        (attempt / anchor).write_text("9")
        (engine / anchor).unlink()
        monkeypatch.setenv("KERNELFORGE_REWRITE_CANDIDATE_KERNEL", str(attempt / anchor))

    with bridge.evaluation_workspace(context, plan, engine, "candidate") as root:
        assert root == engine
        assert (root / anchor).read_text() == ("9" if rewrite else "2")
    assert (engine / anchor).exists() is not rewrite

    with bridge.evaluation_workspace(context, plan, engine, "baseline") as root:
        assert root == context.baseline_workspace
    assert not list(tmp_path.glob("**/evaluate-*"))
