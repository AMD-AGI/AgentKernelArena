# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Focused tests for the matched exact-turn checkpoint observer."""

from __future__ import annotations

import json

from src.agent_turn_budget import AgentTurnBudget


def _message(index: int) -> str:
    return json.dumps(
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": f"turn-{index}"},
        }
    )


def test_exact_fiftieth_decision_is_a_checkpoint_boundary() -> None:
    budget = AgentTurnBudget(50)

    for index in range(49):
        assert budget.observe(_message(index)) is False
    assert budget.observe(_message(49)) is True

    assert budget.receipt() == {
        "policy": "structured_agent_turn_checkpoint_v2",
        "max_turns": 50,
        "observed_turns": 50,
        "exact_boundary_reached": True,
        "post_boundary_turns": 0,
        "budget_exceeded": False,
        "enforcement_failed": False,
        "stop_reason": "exact_turn_boundary",
    }


def test_post_boundary_decision_is_recorded_without_claiming_turn_fifty_one() -> None:
    budget = AgentTurnBudget(50)
    for index in range(50):
        budget.observe(_message(index))

    assert budget.observe(_message(50)) is True

    receipt = budget.receipt()
    assert receipt["observed_turns"] == 50
    assert receipt["post_boundary_turns"] == 1
    assert receipt["exact_boundary_reached"] is False
    assert receipt["budget_exceeded"] is True
    assert receipt["enforcement_failed"] is True


def test_explicit_overrun_is_not_an_exact_boundary() -> None:
    budget = AgentTurnBudget(50)

    assert budget.observe(json.dumps({"type": "usage", "num_turns": 51})) is True

    assert budget.observed_turns == 51
    assert budget.stop_reason == "max_turns_exceeded"
    assert budget.exact_boundary_reached is False


def test_forty_nine_turns_do_not_create_a_checkpoint() -> None:
    budget = AgentTurnBudget(50)
    for index in range(49):
        budget.observe(_message(index))
    budget.finalize(process_succeeded=True, observer_stopped=False)

    assert budget.observed_turns == 49
    assert budget.exact_boundary_reached is False
    assert budget.stop_reason is None
