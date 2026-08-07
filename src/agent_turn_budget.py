# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Structured agent-turn accounting shared by formal direct-agent adapters."""

from __future__ import annotations

import json
from collections.abc import Mapping


TURN_POLICY = "structured_agent_turn_v1"
FORMAL_MATCHED_MAX_TURNS = 50
_CONTEXT_PACKET_HEADER = "# Apex ContextPacket\n"
_ROLE_SECTION = "## Identity and role\n\n> "
_OBJECTIVE_SECTION = "\n\n## Objective and target\n"


def budget_stop_reason_matches(
    *, reason: object, observed_turns: object, max_turns: object
) -> bool:
    """Bind each formal stop reason to the exact counter state that emits it."""

    if (
        isinstance(observed_turns, bool)
        or not isinstance(observed_turns, int)
        or isinstance(max_turns, bool)
        or not isinstance(max_turns, int)
    ):
        return False
    if reason == "max_turns_exhausted_before_follow_up":
        return observed_turns == max_turns
    if reason == "max_turns_exceeded":
        return observed_turns > max_turns
    return False


def context_packet_objective_matches(
    prompt_bytes: bytes, expected_objective: object
) -> bool:
    """Bind the event-receipted prompt to canonical ContextPacket role data."""

    if not isinstance(expected_objective, str):
        return False
    try:
        prompt = prompt_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return False
    if (
        not prompt.startswith(_CONTEXT_PACKET_HEADER)
        or prompt.count(_ROLE_SECTION) != 1
    ):
        return False
    value_start = prompt.index(_ROLE_SECTION) + len(_ROLE_SECTION)
    value_end = prompt.find("\n", value_start)
    if value_end < 0 or not prompt.startswith(_OBJECTIVE_SECTION, value_end):
        return False
    encoded = prompt[value_start:value_end]
    try:
        value = json.loads(
            encoded,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (json.JSONDecodeError, ValueError):
        return False
    if not isinstance(value, Mapping) or set(value) != {"identity", "role"}:
        return False
    role = value.get("role")
    if (
        not isinstance(value.get("identity"), Mapping)
        or not isinstance(role, Mapping)
        or set(role) != {"kind", "objective"}
        or role.get("kind") != "kernel_optimizer"
        or role.get("objective") != expected_objective
    ):
        return False
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return encoded == canonical


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        value[key] = item
    return value


def render_apex_run_control(control: Mapping[str, object]) -> str:
    """Render the canonical human-visible Apex formal run-control suffix."""

    python_interpreter = control["python_interpreter"]
    structured_budget = control["structured_turn_budget"]
    verifier_argv = control["verifier_argv"]
    if (
        not isinstance(python_interpreter, Mapping)
        or not isinstance(structured_budget, Mapping)
        or not isinstance(verifier_argv, Mapping)
    ):
        raise ValueError("malformed Apex caller run control")
    interpreter = python_interpreter["path"]
    environment_variable = python_interpreter["environment_variable"]
    verifier_lines = "\n".join(
        f"- {phase}: `{json.dumps(verifier_argv[phase], ensure_ascii=False)}`"
        for phase in ("compile", "correctness", "performance")
    )
    return (
        "### Formal run control\n\n"
        "Produce one final source version. Work on a candidate promptly and, before "
        "the budget boundary, leave the best source found in every editable file; do "
        "not finish on an exploratory or known-slower variant.\n\n"
        f"The hard limit is {structured_budget['max_turns']} "
        f"`{structured_budget['policy']}` turns. Each assistant "
        "message and each tool-call start counts once. Reserve enough turns to restore "
        "the best source and finish.\n\n"
        f"Use exactly `{interpreter}` as `{environment_variable}`. The three trusted "
        "verifier argv vectors are:\n"
        f"{verifier_lines}"
    )


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


__all__ = [
    "AgentTurnBudget",
    "FORMAL_MATCHED_MAX_TURNS",
    "TURN_POLICY",
    "budget_stop_reason_matches",
    "context_packet_objective_matches",
    "render_apex_run_control",
]
