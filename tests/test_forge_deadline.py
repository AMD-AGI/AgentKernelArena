"""Shared-clock regressions; native compatibility tests use the pinned engine."""
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest

from agents.forge.deadline import bound_session


@dataclass
class Session:
    timeout_sec: int | None = 1200
    user_prompt: str = "## Session deadline\nYou have 20 minutes.\n## Prior experience\nkeep this"


def test_session_cannot_consume_checkpoint_reserve():
    original = Session()
    with patch("agents.forge.deadline.time.time", return_value=1000):
        bounded = bound_session(original, {"deadline_unix": 1300})
    assert bounded.timeout_sec == 180
    assert "180 seconds" in bounded.user_prompt
    assert "20 minutes" not in bounded.user_prompt
    assert "## Prior experience\nkeep this" in bounded.user_prompt
    assert original.timeout_sec == 1200


def test_shorter_provider_limit_and_initialization_phase_are_preserved():
    with patch("agents.forge.deadline.time.time", return_value=1000):
        original = Session(timeout_sec=30)
        assert bound_session(original, {"deadline_unix": 1300}) is original
        result = bound_session(Session(), {"deadline_unix": 5000, "phase_deadline_unix": 1040})
        assert result.timeout_sec == 40
        with pytest.raises(TimeoutError, match="finalization reserve"):
            bound_session(Session(), {"deadline_unix": 1119})
        with pytest.raises(TimeoutError):
            bound_session(Session(), {"deadline_unix": 900, "phase_deadline_unix": 1100})


def test_pinned_native_loop_admission_uses_absolute_clock(tmp_path):
    python = os.environ.get("AKA_FORGE_PROBE_PYTHON")
    if not python:
        pytest.skip("Set AKA_FORGE_PROBE_PYTHON to the pinned Hyperloom[forge] interpreter")
    root = Path(__file__).resolve().parents[1]
    script = r'''
from types import SimpleNamespace
from unittest.mock import patch
from agents.forge import upstream, deadline
from kernelforge.loop import runner
upstream.probe()
original = runner.IterationLoop
deadline.install()
bounded = runner.IterationLoop
deadline.install()
assert runner.IterationLoop is bounded
loop = object.__new__(bounded)
loop.start_time = 1000
loop.ic = SimpleNamespace(max_time_hours=1, deadline_unix=1300, budget_reserve_sec=60)
with patch('time.time', return_value=1250):
    # Original would claim 55+ minutes and admit another costly search.
    assert original._time_remaining(loop) == 3350
    assert loop._time_remaining() == 50
    assert loop._is_budget_exhausted()
with patch('time.time', return_value=1400):
    assert loop._time_remaining() == 0
    assert loop._is_budget_exhausted()
loop.ic.deadline_unix = 10000
with patch('time.time', return_value=4550):
    assert loop._time_remaining() == 50  # wall budget still wins if shorter
loop.ic.deadline_unix = None
with patch('time.time', return_value=1250):
    assert loop._time_remaining() == original._time_remaining(loop)

received = []
def capture(self, ic, tracker, config, resume):
    received.append(ic)
with patch.object(original, '__init__', capture), patch('time.time', return_value=1000):
    ic = runner.IterationConfig(kernel_file='kernel.py', driver_script='driver.py',
                                max_time_hours=1, deadline_unix=2200)
    bounded(ic, None)
    assert received[-1].budget_reserve_sec == 120
    assert ic.budget_reserve_sec == 1800
    assert received[-1].validate_stage_timeout_sec == ic.validate_stage_timeout_sec
    assert received[-1].bench_timeout_sec == ic.bench_timeout_sec
'''
    result = subprocess.run([python, "-c", script], cwd=root,
                            env=dict(os.environ, PYTHONPATH=str(root)),
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
