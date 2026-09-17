"""Forge measurement placement; CPU only, no model or GPU calls."""
from pathlib import Path

import pytest

from agents.forge import bridge
from agents.forge.bundles import copy_workspace
from test_forge_v2 import fixture_task


@pytest.mark.parametrize("rewrite", [False, True])
def test_candidate_measures_in_place_and_baseline_stays_independent(tmp_path, monkeypatch, rewrite):
    """Candidate actions reuse the engine tree; a rewrite leaves no shadow."""
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

    # The baseline anchor is rebuilt from the frozen snapshot, once per campaign,
    # so a candidate cannot move what it is measured against.
    with bridge.evaluation_workspace(context, plan, engine, "baseline") as root:
        assert root != engine and (root / anchor).read_text() == "2"
        baseline_root = root
    assert not baseline_root.exists()
    assert (context.baseline_workspace / anchor).read_text() == "2"
    assert (context.workspace / anchor).read_text() == "2"
