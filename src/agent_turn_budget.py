# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Structured agent-turn accounting shared by formal direct-agent adapters."""

from __future__ import annotations

import json
from collections.abc import Mapping


TURN_POLICY = "structured_agent_turn_v1"


class AgentTurnBudget:
    """Count structured decisions and fail closed when evidence is ambiguous."""

    def __init__(self, max_turns: int) -> None:
        if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns <= 0:
            raise ValueError("max_turns must be positive")
        self.max_turns = max_turns
        self.observed_turns = 0
        self.stop_reason: str | None = None
        self._saw_decision = False
        self._saw_turn_evidence = False

    def observe(self, line: str) -> bool:
        try:
            value = _json_object(line)
        except json.JSONDecodeError:
            self.stop_reason = "unparseable_structured_event"
            return True
        if value is None:
            return False
        explicit = _nonnegative_int(value, "num_turns", "turn_count", "turns")
        if explicit is not None:
            self.observed_turns = max(self.observed_turns, explicit)
            self._saw_turn_evidence = True
        decisions, requires_follow_up = _decision_count(value)
        if decisions:
            self.observed_turns += decisions
            self._saw_decision = True
            self._saw_turn_evidence = True
        event_type = str(value.get("type", "")).lower().replace("_", ".")
        if event_type == "turn.completed" and not self._saw_decision:
            self.observed_turns += 1
            self._saw_decision = True
            self._saw_turn_evidence = True
        if self.observed_turns > self.max_turns:
            self.stop_reason = "max_turns_exceeded"
            return True
        if self.observed_turns == self.max_turns and requires_follow_up:
            self.stop_reason = "max_turns_exhausted_before_follow_up"
            return True
        return False

    def stop_for_observer_error(self, reason: str) -> None:
        if self.stop_reason is None:
            self.stop_reason = reason

    def finalize(self, *, process_succeeded: bool, observer_stopped: bool) -> None:
        if observer_stopped and self.stop_reason is None:
            self.stop_reason = "turn_observer_failed"
        if process_succeeded and not self._saw_turn_evidence:
            self.stop_reason = "missing_structured_turn_evidence"

    @property
    def budget_exceeded(self) -> bool:
        return self.stop_reason is not None and self.stop_reason.startswith("max_turns_")

    @property
    def enforcement_failed(self) -> bool:
        return self.stop_reason in {
            "missing_structured_turn_evidence",
            "oversized_structured_event",
            "turn_observer_failed",
            "unparseable_structured_event",
        }

    def receipt(self) -> dict[str, object]:
        return {
            "policy": TURN_POLICY,
            "max_turns": self.max_turns,
            "observed_turns": self.observed_turns,
            "budget_exceeded": self.budget_exceeded,
            "enforcement_failed": self.enforcement_failed,
            "stop_reason": self.stop_reason,
        }


def _json_object(line: str) -> Mapping[str, object] | None:
    if not line.lstrip().startswith("{"):
        return None
    value = json.loads(line)
    return value if isinstance(value, Mapping) else None


def _decision_count(value: Mapping[str, object]) -> tuple[int, bool]:
    event_type = str(value.get("type", "")).lower().replace(".", "_")
    message = value.get("message")
    if event_type == "assistant" and isinstance(message, Mapping):
        return 1, _contains_tool_request(message.get("content"))
    if event_type in {"assistant_message", "agent_message"}:
        return 1, _contains_tool_request(value.get("content"))
    if event_type == "item_completed":
        item = value.get("item")
        if isinstance(item, Mapping) and str(item.get("type", "")).lower() in {
            "agent_message",
            "assistant_message",
        }:
            return 1, _contains_tool_request(item.get("content"))
    if _standalone_tool_request(event_type, value):
        return 1, True
    return 0, False


def _standalone_tool_request(event_type: str, value: Mapping[str, object]) -> bool:
    if event_type in {"tool_call", "tool_called", "tool_use"}:
        subtype = str(value.get("subtype", value.get("status", "started"))).lower()
        return subtype not in {"completed", "result", "failed", "error", "cancelled"}
    if event_type in {"item_started", "item_completed"}:
        item = value.get("item")
        if not isinstance(item, Mapping):
            return False
        item_type = str(item.get("type", "")).lower()
        return event_type == "item_started" and any(
            marker in item_type for marker in ("tool", "command_execution")
        )
    return False


def _contains_tool_request(content: object) -> bool:
    if not isinstance(content, list):
        return False
    return any(
        isinstance(block, Mapping)
        and str(block.get("type", "")).lower()
        in {"tool_use", "tool_call", "tool_called"}
        for block in content
    )


def _nonnegative_int(value: Mapping[str, object], *keys: str) -> int | None:
    for key in keys:
        candidate = value.get(key)
        if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate >= 0:
            return candidate
    return None


__all__ = ["AgentTurnBudget", "TURN_POLICY"]
