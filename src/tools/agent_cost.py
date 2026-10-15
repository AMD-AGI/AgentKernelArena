# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Per-task agent cost records.

``main.py`` wraps the selected agent launcher with :class:`AgentCostCollector`
for every task and writes one JSON record per task workspace to
``<run_dir>/agent_costs/<workspace_name>.json`` after the agent returns or
fails. Launchers only expose what their provider reports, by calling
:func:`report_agent_usage` once per provider process, session or call. A
launcher that reports nothing still gets a record whose cost and usage are
``null``. Outside a collector (direct launcher calls, tests, other workflows)
reporting is a no-op.

Providers differ in what a reported number covers, so every report states a
scope for its dollars and tokens:

* ``invocation``: the value covers only this report. Values are summed.
* ``session``: the value is a running total for the provider session named by
  ``session_id`` and may include earlier resumed invocations. Totals are
  read as a meter: a drop is treated as a counter reset, otherwise only the
  increase is added, so resumes are never double counted.
* ``None``: unknown. Summed only when no session was resumed; otherwise the
  total is ``null`` with status ``ambiguous``.

Statuses: ``complete`` (every report had the value), ``partial`` (a lower
bound: some reports, for example a session killed at its deadline, had none or
were marked ``complete=False``), ``not_reported``, and ``ambiguous``. Numbers
are never estimated.
"""
from __future__ import annotations

from contextvars import ContextVar
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping
import uuid

RECORD_VERSION = 1
COST_DIRECTORY = "agent_costs"
SCOPES = ("invocation", "session")
# Run settings copied into the record so a cost can be read with its budget.
_RUN_SETTING_KEYS = ("model", "effort", "budget_mode", "timeout_seconds", "max_iterations",
                     "max_budget_usd")
# Provider spellings that mean the same token class (after snake_case). Other
# names are kept as reported: for example Codex's input_tokens includes its
# cached_input_tokens, while Claude's input_tokens excludes cache reads.
_TOKEN_ALIASES = {
    "cache_read_tokens": "cache_read_input_tokens",
    "cache_write_tokens": "cache_creation_input_tokens",
}
_STRING_LIMIT = 200

_ACTIVE: ContextVar["AgentCostCollector | None"] = ContextVar("aka_agent_cost_collector", default=None)


def _snake(key: str) -> str:
    key = re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", key).lower()
    return _TOKEN_ALIASES.get(key, key)


def _number(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return value


def _text(value: Any) -> str | None:
    return value[:_STRING_LIMIT] if isinstance(value, str) and value else None


def normalize_usage(raw: Any) -> dict[str, float | int]:
    """Keep top-level non-negative numeric token counters under snake_case names.

    ``raw`` is one provider usage mapping, or a list of mappings that each
    cover a disjoint part of the same invocation (they are summed). Dollar
    fields are not token usage and are dropped; report them as ``cost_usd``.
    """
    parts = raw if isinstance(raw, (list, tuple)) else [raw]
    usage: dict[str, float | int] = {}
    for part in parts:
        if not isinstance(part, Mapping):
            continue
        for key, value in part.items():
            number = _number(value)
            if not isinstance(key, str) or number is None:
                continue
            name = _snake(key)
            if "cost" in name or name.endswith("_usd"):
                continue
            usage[name] = usage.get(name, 0) + number
    return usage


def normalize_report(
    *,
    cost_usd: Any = None,
    cost_scope: str | None = None,
    usage: Any = None,
    usage_scope: str | None = None,
    session_id: Any = None,
    source: Any = None,
    model: Any = None,
    timed_out: Any = None,
    seconds: Any = None,
    complete: Any = True,
) -> dict[str, Any]:
    """Validate one raw provider report into the record's invocation shape.

    ``complete=False`` marks reported values as a lower bound, for example
    when the provider may have spent more after its last report.
    """
    for name, scope in (("cost_scope", cost_scope), ("usage_scope", usage_scope)):
        if scope is not None and scope not in SCOPES:
            raise ValueError(f"{name} must be one of {SCOPES} or None")
    normalized_usage = normalize_usage(usage)
    elapsed = _number(seconds)
    return {
        "cost_usd": _number(cost_usd),
        "cost_scope": cost_scope,
        "usage": normalized_usage or None,
        "usage_scope": usage_scope,
        "session_id": _text(session_id),
        "source": _text(source),
        "model": _text(model),
        "timed_out": timed_out if isinstance(timed_out, bool) else None,
        "seconds": round(elapsed, 3) if elapsed is not None else None,
        "complete": complete is not False,
    }


def report_agent_usage(**report: Any) -> None:
    """Expose one provider report to the active collector; no-op without one.

    Never raises: a malformed provider value must not fail the agent.
    """
    collector = _ACTIVE.get()
    if collector is None:
        return
    try:
        collector.invocations.append(normalize_report(**report))
    except Exception as exc:  # noqa: BLE001 - cost reporting is best effort
        collector.report_errors.append(f"{type(exc).__name__}: {exc}"[:_STRING_LIMIT])


def _aggregate(values: list[tuple[str, str | None, float | None, bool]]) -> tuple[float | None, str]:
    """Aggregate ``(session_key, scope, value, complete)`` reports into ``(total, status)``."""
    if not values:
        return None, "not_reported"
    known = [row for row in values if row[2] is not None]
    if not known:
        return None, "not_reported"
    scopes = {row[1] for row in values}
    scope = scopes.pop() if len(scopes) == 1 else None
    if scope is None:
        sessions = [row[0] for row in values]
        if len(set(sessions)) != len(sessions):
            return None, "ambiguous"
        scope = "invocation"
    if scope == "invocation":
        total = sum(row[2] for row in known)
    else:
        total, last_seen = 0.0, {}
        for key, _, value, _ in known:
            previous = last_seen.get(key, 0)
            total += value - previous if value >= previous else value
            last_seen[key] = value
    complete = len(known) == len(values) and all(row[3] for row in known)
    return total, "complete" if complete else "partial"


def summarize_invocations(invocations: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [row.get("session_id") or f"invocation-{index}" for index, row in enumerate(invocations)]
    cost, cost_status = _aggregate([(key, row.get("cost_scope"), row.get("cost_usd"),
                                     row.get("complete", True))
                                    for key, row in zip(keys, invocations)])
    token_names = sorted({name for row in invocations for name in (row.get("usage") or {})})
    usage: dict[str, float | int] = {}
    statuses = set()
    for name in token_names:
        total, status = _aggregate([(key, row.get("usage_scope"), (row.get("usage") or {}).get(name),
                                     row.get("complete", True))
                                    for key, row in zip(keys, invocations)])
        statuses.add(status)
        if total is not None:
            usage[name] = int(total) if float(total).is_integer() else total
    if not invocations or not token_names:
        usage_status = "not_reported"
    elif "ambiguous" in statuses:
        usage_status = "ambiguous"
    elif statuses == {"complete"} and all(row.get("usage") for row in invocations):
        usage_status = "complete"
    else:
        usage_status = "partial"
    return {
        "total_cost_usd": round(cost, 6) if cost is not None else None,
        "cost_status": cost_status,
        "usage": usage or None,
        "usage_status": usage_status,
    }


def _combine_statuses(statuses: list[str]) -> str:
    unique = set(statuses)
    if len(unique) == 1:
        return unique.pop()
    return "partial"


class AgentCostCollector:
    """Collect one task's provider reports around a launcher call."""

    def __init__(self, *, agent: str, task_name: str, run_settings: Mapping[str, Any] | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.agent = agent
        self.task_name = task_name
        settings = run_settings if isinstance(run_settings, Mapping) else {}
        self.run_settings = {key: settings[key] for key in _RUN_SETTING_KEYS if key in settings}
        self.clock = clock
        self.workspace: Path | None = None
        self.invocations: list[dict[str, Any]] = []
        self.report_errors: list[str] = []
        self.launches = 0
        self.agent_seconds = 0.0
        self.launcher_error: str | None = None
        self.started_at: str | None = None

    def wrap(self, launcher: Callable[..., Any]) -> Callable[..., Any]:
        """Return a launcher with the same keyword interface that activates collection."""
        def launch(*, eval_config, task_config_dir, workspace):
            self.workspace = Path(workspace)
            self.launches += 1
            if self.started_at is None:
                self.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            token = _ACTIVE.set(self)
            started = self.clock()
            try:
                return launcher(eval_config=eval_config, task_config_dir=task_config_dir,
                                workspace=workspace)
            except BaseException as exc:
                self.launcher_error = type(exc).__name__
                raise
            finally:
                self.agent_seconds += self.clock() - started
                _ACTIVE.reset(token)
        return launch

    def attempt(self) -> dict[str, Any]:
        attempt = {
            "started_at": self.started_at,
            "launches": self.launches,
            "agent_seconds": round(self.agent_seconds, 3),
            "launcher_error": self.launcher_error,
            "run_settings": self.run_settings,
            "invocations": self.invocations,
            **summarize_invocations(self.invocations),
        }
        if self.report_errors:
            attempt["report_errors"] = self.report_errors
        return attempt

    def record_path(self, run_directory: Path) -> Path | None:
        if self.workspace is None:
            return None
        return Path(run_directory) / COST_DIRECTORY / f"{self.workspace.name}.json"

    def write(self, run_directory: Path, logger: logging.Logger) -> Path | None:
        """Write or extend the task's record. Best effort; never raises.

        Nothing is written when the launcher was never called. A resumed run
        that launches the agent again appends a new attempt to the same record.
        """
        target = self.record_path(run_directory)
        if target is None or self.launches == 0:
            return None
        try:
            attempts: list[dict[str, Any]] = []
            previous_unreadable = False
            if target.is_file():
                try:
                    previous = json.loads(target.read_text())
                    if isinstance(previous, dict) and isinstance(previous.get("attempts"), list):
                        attempts = [row for row in previous["attempts"] if isinstance(row, dict)]
                    else:
                        previous_unreadable = True
                except (OSError, ValueError):
                    previous_unreadable = True
            if previous_unreadable:
                preserved = target.with_name(f"{target.name}.unreadable-{uuid.uuid4().hex[:8]}")
                target.replace(preserved)
                logger.warning("Preserved unreadable agent cost record as %s", preserved)
            attempts.append(self.attempt())
            record = build_record(agent=self.agent, task_name=self.task_name,
                                  workspace=self.workspace.name, attempts=attempts)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
            try:
                temporary.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
                temporary.replace(target)
            finally:
                # A failed write or rename must not leave a partial file behind.
                temporary.unlink(missing_ok=True)
            logger.info("Agent cost record: %s (total_cost_usd=%s, cost_status=%s)",
                        target, record["total_cost_usd"], record["cost_status"])
            return target
        except Exception as exc:  # noqa: BLE001 - never fail a task for bookkeeping
            logger.warning("Could not write agent cost record %s: %s", target, exc)
            return None


def build_record(*, agent: str, task_name: str, workspace: str,
                 attempts: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine attempts; totals are sums of attempt totals, null when none is known."""
    costs = [row.get("total_cost_usd") for row in attempts]
    known_costs = [value for value in costs if isinstance(value, (int, float))]
    usage: dict[str, float | int] = {}
    for row in attempts:
        for name, value in (row.get("usage") or {}).items():
            usage[name] = usage.get(name, 0) + value
    return {
        "version": RECORD_VERSION,
        "agent": agent,
        "task": task_name,
        "workspace": workspace,
        "total_cost_usd": round(sum(known_costs), 6) if known_costs else None,
        "cost_status": _combine_statuses([row.get("cost_status", "not_reported") for row in attempts]),
        "usage": usage or None,
        "usage_status": _combine_statuses([row.get("usage_status", "not_reported") for row in attempts]),
        "agent_seconds": round(sum(row.get("agent_seconds") or 0 for row in attempts), 3),
        "attempts": attempts,
    }
