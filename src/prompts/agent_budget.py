# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Budget guidance appended to CLI agent prompts.

``iterations`` mode keeps the historical "iterate up to N versions" sentence
byte for byte. ``time`` mode replaces it with open-ended guidance, and later
sessions of the same agent conversation receive :func:`time_budget_continuation`.
The per-invocation wall-clock sentence remains in ``agents/run_budget.py``.
"""


def iteration_budget_guidance(max_iterations: object) -> str:
    """Return the historical iterations-mode sentence, or nothing when unset."""
    if max_iterations is None:
        return ""
    return f"\n\nFor this optimization, you must iterate up to {max_iterations} versions."


def time_budget_guidance() -> str:
    return (
        "\n\nThere is no fixed number of versions for this optimization: keep proposing, "
        "measuring and refining improvements for as long as the wall-clock budget allows. "
        "If you end your turn early, the framework may resume this session with the "
        "remaining time."
    )


def time_budget_continuation(remaining_seconds: float, *, finish_margin_seconds: int = 300) -> str:
    """Prompt for a resumed session; ``remaining_seconds`` is its hard timeout."""
    minutes = max(0, int(remaining_seconds // 60))
    margin = max(1, finish_margin_seconds // 60)
    return (
        f"About {minutes} minutes of the wall-clock budget remain, and this session will be "
        "stopped when they run out. Continue optimizing: look for further improvements, "
        "validate each one with the task's compile, correctness and performance commands, "
        "and keep the best correct implementation at the declared candidate paths. Finish "
        f"your last change and its checks at least {margin} minutes before the budget expires."
    )
