# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Shared time-budget runtime for resumable CLI agents.

``agent.budget_mode`` selects how a CLI agent spends ``timeout_seconds``:

* ``iterations`` (default): one session asked for up to ``max_iterations``
  versions; ``timeout_seconds`` is only its hard cap. Unchanged behavior.
* ``time``: no iteration count. If the agent ends its session early, the
  launcher resumes the same provider session with the time remaining until the
  original deadline, while at least ``MIN_RESUME_SECONDS`` remain and at most
  ``MAX_SESSIONS`` sessions have run.

Provider-specific resume flags and output parsing stay in ``agents/<name>/``.
Prompt text lives in ``src/prompts/agent_budget.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any, Callable, Mapping

from src.prompts.agent_budget import time_budget_continuation

BUDGET_MODES = ("iterations", "time")
DEFAULT_BUDGET_MODE = "iterations"
# Agents whose launchers implement session resume. Other agents reject ``time``.
TIME_BUDGET_AGENTS = frozenset({"claude_code", "codex", "cursor", "deepseek_harness"})
# Do not start a continuation too short for a measure/verify cycle.
MIN_RESUME_SECONDS = 600
# Bound on sessions per task, so an agent that returns immediately cannot spin.
MAX_SESSIONS = 20


def resolve_budget_mode(agent_settings: Mapping[str, Any] | None) -> str:
    """Return the validated mode; only a missing or null value means the default."""
    if agent_settings is None:
        return DEFAULT_BUDGET_MODE
    if not isinstance(agent_settings, Mapping):
        raise ValueError("agent must be a mapping")
    mode = agent_settings.get("budget_mode")
    if mode is None:
        return DEFAULT_BUDGET_MODE
    if not isinstance(mode, str) or mode not in BUDGET_MODES:
        raise ValueError(f"agent.budget_mode must be one of {list(BUDGET_MODES)}, got {mode!r}")
    return mode


def validate_run_budget_mode(agent_template: str, agent_settings: Mapping[str, Any] | None) -> str:
    """Fail fast at run start instead of failing every task's agent launch."""
    mode = resolve_budget_mode(agent_settings)
    if mode == "time" and agent_template not in TIME_BUDGET_AGENTS:
        raise ValueError(
            f"agent.budget_mode 'time' is not supported by agent {agent_template!r}; "
            f"supported agents: {sorted(TIME_BUDGET_AGENTS)}"
        )
    return mode


@dataclass
class SessionOutcome:
    """Outcome of one CLI process; cost/usage are reported separately."""
    timed_out: bool = False
    returncode: int | None = None
    failed: bool = False
    session_id: str | None = None
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.timed_out and self.returncode == 0 and not self.failed


Invoke = Callable[[str, "str | None", float], SessionOutcome]


def run_sessions(
    invoke: Invoke,
    *,
    mode: str,
    prompt: str,
    timeout_seconds: int,
    logger: logging.Logger,
    clock: Callable[[], float] = time.monotonic,
    min_resume_seconds: float | None = None,
    max_sessions: int | None = None,
) -> list[SessionOutcome]:
    """Call ``invoke(prompt, resume_session_id, timeout)`` until the budget is used.

    ``iterations`` mode is exactly one call with the full timeout. In ``time``
    mode each continuation's timeout is the time left until the deadline fixed
    before the first call, so the sessions together never exceed the budget
    (apart from process-stop latency).
    """
    if mode not in BUDGET_MODES:
        raise ValueError(f"unknown budget mode {mode!r}")
    floor = MIN_RESUME_SECONDS if min_resume_seconds is None else min_resume_seconds
    limit = MAX_SESSIONS if max_sessions is None else max_sessions
    deadline = clock() + timeout_seconds
    runs = [invoke(prompt, None, timeout_seconds)]
    if mode != "time":
        return runs
    while runs[-1].ok and len(runs) < limit:
        remaining = deadline - clock()
        if remaining <= 0 or remaining < floor:
            logger.info("Time budget: %.0fs left; not resuming", max(remaining, 0.0))
            break
        session_id = runs[-1].session_id
        if not session_id:
            logger.warning("Time budget: agent reported no session id; cannot resume")
            break
        logger.info("Time budget: session %d ended with %.0fs left; resuming %s",
                    len(runs), remaining, session_id)
        runs.append(invoke(time_budget_continuation(remaining), session_id, remaining))
    return runs


def raise_for_sessions(runs: list[SessionOutcome], *, mode: str, agent_label: str,
                       timeout_seconds: int, failed_event: str = "result") -> None:
    """Preserve the historical failure semantics of a single invocation.

    In ``time`` mode a continuation stopped at the deadline has simply used the
    budget. Only a first-session timeout or an actual error is a failure.
    """
    last = runs[-1]
    if last.timed_out and not (mode == "time" and len(runs) > 1):
        raise TimeoutError(f"{agent_label} timed out after {timeout_seconds}s; see agent logs")
    if not last.timed_out and last.returncode != 0:
        raise RuntimeError(f"{agent_label} exited with code {last.returncode}; see agent logs")
    if last.failed:
        raise RuntimeError(f"{agent_label} reported a failed {failed_event}; see agent logs")
