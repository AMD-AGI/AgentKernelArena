"""Forge measurement placement and process cleanup; no model or GPU calls."""
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from agents.forge import adapter, bridge
from agents.forge.bundles import copy_workspace
from test_forge_v2 import ROOT, fixture_task


def test_engine_budget_reserves_the_startup_margin_above_the_engine_floor():
    """The engine's clock starts after ours, so it is handed less than we have."""
    budget = adapter.engine_budget_hours({"deadline_unix": time.time() + 4 * 3600}) * 3600
    assert 4 * 3600 - adapter.ENGINE_STARTUP_MARGIN_SEC - 60 < budget <= 4 * 3600 - adapter.ENGINE_STARTUP_MARGIN_SEC
    # A shorter campaign gains nothing from the margin: the engine floors it.
    assert adapter.engine_budget_hours(
        {"deadline_unix": time.time() + 3600}) * 3600 == adapter.ENGINE_BUDGET_FLOOR_SEC


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


@pytest.mark.parametrize("workflow", ["optimize", "rewrite"])
@pytest.mark.parametrize("outcome", ["success", "error", "signal", "killed", "timeout"])
def test_campaign_entry_reaps_detached_descendants(tmp_path, workflow, outcome):
    """Exercise the production entry/launcher, including an orphaned grandchild."""
    package = tmp_path / "kernelforge"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "cli.py").write_text(r'''
import os, signal, subprocess, sys, time
from pathlib import Path
outcome, marker = sys.argv[1:]
worker = r"""
import os, signal, subprocess, sys, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
grandchild = subprocess.Popen([sys.executable, '-c',
    'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'],
    start_new_session=True)
Path(sys.argv[1]).write_text(str(os.getpid()) + ' ' + str(grandchild.pid))
time.sleep(60)
"""
subprocess.Popen([sys.executable, '-c', worker, marker], start_new_session=True)
deadline = time.monotonic() + 5
while not Path(marker).exists() and time.monotonic() < deadline:
    time.sleep(.01)
assert Path(marker).exists()
print('native stdout', flush=True)
print('native stderr', file=sys.stderr, flush=True)
if outcome == 'timeout':
    time.sleep(60)
if outcome == 'signal':
    os.kill(os.getpid(), signal.SIGTERM)
if outcome == 'killed':
    os.kill(os.getpid(), signal.SIGKILL)
raise SystemExit(7 if outcome == 'error' else 0)
''')
    # This outer test-only subreaper prevents leaks when run against the broken
    # implementation. Assertions happen before its cleanup, so it cannot mask
    # missing production supervision.
    script = r'''
import logging, os, signal, subprocess, sys
from pathlib import Path
from agents.forge.adapter import engine_entry
from agents.forge.common import run_forge_subprocess
from agents.forge.process_tree import managed_children
root, workflow, outcome = sys.argv[1:]
marker = Path(root) / 'workers.pid'
with managed_children():
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                                 start_new_session=True)
    command = engine_entry({'python': sys.executable}, workflow) + [outcome, str(marker)]
    process, stdout, stderr, timed_out = run_forge_subprocess(
        command, workspace=root, env=os.environ.copy(), timeout_seconds=2,
        logger=logging.getLogger('campaign-lifecycle-test'))
    assert marker.exists(), (stdout, stderr)
    survivors = [pid for pid in marker.read_text().split() if Path('/proc', pid).exists()]
    assert not survivors, 'campaign returned with live descendants: ' + str(survivors)
    assert unrelated.poll() is None, 'campaign cleanup killed an unrelated process'
    assert timed_out == (outcome == 'timeout')
    if not timed_out:
        expected = {'success': 0, 'error': 7, 'signal': -signal.SIGTERM, 'killed': -signal.SIGKILL}
        assert process.returncode == expected[outcome]
    else:
        assert process.returncode != 0
    assert 'native stdout' in stdout
    assert 'native stderr' in stderr
'''
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), workflow, outcome],
        cwd=ROOT, env={**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), str(tmp_path)])},
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
