"""Time-budget session loop and budget-mode validation, with a fake clock."""

import logging

import pytest

from src.prompts.agent_budget import iteration_budget_guidance, time_budget_continuation
from src.tools import agent_time_budget as budget
from src.tools.agent_time_budget import (
    SessionOutcome, raise_for_sessions, resolve_budget_mode, run_sessions, validate_run_budget_mode,
)

LOGGER = logging.getLogger(__name__)


class FakeAgent:
    """Each call advances the clock by the scripted session length."""

    def __init__(self, durations, *, outcomes=None, session_id="session-1"):
        self.now = 1000.0
        self.durations = list(durations)
        self.outcomes = outcomes or {}
        self.session_id = session_id
        self.calls = []

    def clock(self):
        return self.now

    def __call__(self, prompt, resume_id, timeout):
        index = len(self.calls)
        self.calls.append((prompt, resume_id, timeout))
        duration = self.durations[index]
        timed_out = duration >= timeout
        self.now += min(duration, timeout)
        outcome = dict(timed_out=timed_out, returncode=-15 if timed_out else 0,
                       session_id=self.session_id, seconds=min(duration, timeout))
        outcome.update(self.outcomes.get(index, {}))
        return SessionOutcome(**outcome)


def test_iterations_mode_is_exactly_one_session_with_full_timeout():
    agent = FakeAgent([10])
    runs = run_sessions(agent, mode="iterations", prompt="P", timeout_seconds=3600,
                        logger=LOGGER, clock=agent.clock, min_resume_seconds=0)
    assert len(runs) == 1
    assert agent.calls == [("P", None, 3600)]


def test_time_mode_resumes_with_remaining_time_until_deadline():
    agent = FakeAgent([1000, 1200, 5000])
    runs = run_sessions(agent, mode="time", prompt="P", timeout_seconds=3600,
                        logger=LOGGER, clock=agent.clock, min_resume_seconds=600)
    assert [call[1] for call in agent.calls] == [None, "session-1", "session-1"]
    assert [call[2] for call in agent.calls] == [3600, 2600, 1400]
    assert agent.calls[1][0] == time_budget_continuation(2600)
    assert "43 minutes" in agent.calls[1][0]
    # The last session was stopped at the original deadline: no overrun.
    assert agent.now == 1000.0 + 3600
    assert runs[-1].timed_out
    raise_for_sessions(runs, mode="time", agent_label="Fake", timeout_seconds=3600)


def test_time_mode_does_not_start_a_session_below_the_floor():
    agent = FakeAgent([3100])
    runs = run_sessions(agent, mode="time", prompt="P", timeout_seconds=3600,
                        logger=LOGGER, clock=agent.clock, min_resume_seconds=600)
    assert len(runs) == 1


@pytest.mark.parametrize("outcome", [{"returncode": 3}, {"failed": True}, {"session_id": None}])
def test_time_mode_stops_on_failure_or_missing_session(outcome):
    agent = FakeAgent([10, 10], outcomes={0: outcome})
    runs = run_sessions(agent, mode="time", prompt="P", timeout_seconds=3600,
                        logger=LOGGER, clock=agent.clock, min_resume_seconds=0)
    assert len(runs) == 1


def test_time_mode_session_count_is_bounded(monkeypatch):
    monkeypatch.setattr(budget, "MAX_SESSIONS", 4)
    agent = FakeAgent([1] * 10)
    runs = run_sessions(agent, mode="time", prompt="P", timeout_seconds=3600,
                        logger=LOGGER, clock=agent.clock, min_resume_seconds=0)
    assert len(runs) == 4


def test_failure_semantics_match_a_single_invocation():
    timed_out = SessionOutcome(timed_out=True, returncode=-15)
    with pytest.raises(TimeoutError, match="timed out after 60s"):
        raise_for_sessions([timed_out], mode="iterations", agent_label="Fake", timeout_seconds=60)
    with pytest.raises(TimeoutError):
        raise_for_sessions([timed_out], mode="time", agent_label="Fake", timeout_seconds=60)
    with pytest.raises(RuntimeError, match="exited with code 7"):
        raise_for_sessions([SessionOutcome(returncode=7)], mode="time", agent_label="Fake", timeout_seconds=60)
    with pytest.raises(RuntimeError, match="reported a failed turn"):
        raise_for_sessions([SessionOutcome(returncode=0, failed=True)], mode="iterations",
                           agent_label="Fake", timeout_seconds=60, failed_event="turn")
    raise_for_sessions([SessionOutcome(returncode=0)], mode="iterations", agent_label="Fake", timeout_seconds=60)


@pytest.mark.parametrize("settings", [None, {}, {"budget_mode": None}])
def test_default_budget_mode_is_iterations(settings):
    assert resolve_budget_mode(settings) == "iterations"


@pytest.mark.parametrize("value", ["tokens", "", "TIME", False, 0, ["time"]])
def test_invalid_budget_mode_rejected(value):
    with pytest.raises(ValueError, match="budget_mode"):
        resolve_budget_mode({"budget_mode": value})


@pytest.mark.parametrize("agent", ["claude_code", "codex", "cursor", "deepseek_harness"])
def test_run_validation_accepts_time_for_resumable_cli_agents(agent):
    assert validate_run_budget_mode(agent, {"budget_mode": "time"}) == "time"


@pytest.mark.parametrize("agent", ["geak", "forge", "task_validator"])
def test_run_validation_rejects_time_for_other_agents(agent):
    assert validate_run_budget_mode(agent, {"template": agent}) == "iterations"
    with pytest.raises(ValueError, match="not supported"):
        validate_run_budget_mode(agent, {"budget_mode": "time"})


def test_iteration_guidance_text_is_unchanged():
    assert iteration_budget_guidance(3) == "\n\nFor this optimization, you must iterate up to 3 versions."
    assert iteration_budget_guidance(None) == ""
