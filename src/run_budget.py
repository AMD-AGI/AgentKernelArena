"""Persist task deadlines outside the candidate; resume never renews a budget."""
from __future__ import annotations

import json
import time
from pathlib import Path


def budget_policy(config: dict) -> dict | None:
    value = config.get("budget")
    if value is None:
        return None
    durations = {"task_wall_time_s", "final_evaluation_reserve_s"}
    if not isinstance(value, dict) or not durations <= set(value) or set(value) - durations - {"final_evaluation_pairs"}:
        raise ValueError("budget requires task_wall_time_s and final_evaluation_reserve_s; optional final_evaluation_pairs")
    if "final_evaluation_pairs" in value:
        from .measurement import validate_pair_count
        validate_pair_count(value["final_evaluation_pairs"])
    if any(type(value[k]) is not int or value[k] <= 0 for k in durations):
        raise ValueError("Budget durations must be positive integer seconds")
    if value["final_evaluation_reserve_s"] >= value["task_wall_time_s"]:
        raise ValueError("Final evaluation reserve must fit inside the task budget")
    return dict(value)


def open_budget(path: Path, policy: dict | None) -> dict | None:
    if path.exists():
        record = json.loads(path.read_text())
        if record["policy"] != policy:
            raise ValueError("Cannot change a task budget on resume")
        return record
    if policy is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"policy": policy, "deadline_epoch": time.time() + policy["task_wall_time_s"]}
    with path.open("x") as handle:
        json.dump(record, handle)
    return record


def agent_budget(config: dict, record: dict | None) -> dict:
    """Run-level explicit timeout can tighten the budget; template defaults cannot."""
    result = {**config, "agent": dict(config.get("agent", {}))}
    if record is not None:
        remaining = int(record["deadline_epoch"] - time.time() - record.get("effective_reserve_s", record["policy"]["final_evaluation_reserve_s"]))
        if remaining <= 0:
            raise TimeoutError("Search budget exhausted; final evaluation reserve reached")
        explicit = result["agent"].get("timeout_seconds")
        if explicit is not None and (type(explicit) is not int or explicit <= 0):
            raise ValueError("agent.timeout_seconds must be a positive integer")
        result["agent"]["timeout_seconds"] = min(remaining, explicit) if explicit is not None else remaining
    return result


def reserve_from_validation(record: dict | None, session, *, verification_seconds: int = 0) -> dict | None:
    """Persist a measured reserve; a resumed run can only keep or increase it."""
    if record is None:
        raise ValueError("Serving tasks require an explicit wall time budget")
    config = session.spec.to_mapping()
    lock = json.loads((session.workspace / config["evaluation"]["measurement"]["runtime_lock"]).read_text())
    elapsed = {}
    for (phase, role, action), result in session.results.items():
        if phase == "task_validation" and role == "baseline":
            elapsed[action] = sum(command.elapsed_s for command in result.commands)
    pairs = record["policy"].get("final_evaluation_pairs", config["evaluation"]["measurement"]["pairs"])
    estimate = int(1.5 * (elapsed.get("compile", 0) + elapsed.get("correctness", 0) + 2*pairs*elapsed.get("performance", 0)))
    estimate += verification_seconds
    result = dict(record)
    result["effective_reserve_s"] = max(record.get("effective_reserve_s", 0),
        record["policy"]["final_evaluation_reserve_s"], lock["minimum_final_evaluation_s"], estimate)
    if result["effective_reserve_s"] >= record["deadline_epoch"] - time.time():
        raise TimeoutError("Measured final evaluation reserve exceeds remaining task budget")
    return result
