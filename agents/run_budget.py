"""Expose a CLI invocation's configured budget without changing task rules."""


def append_run_budget(prompt: str, timeout_seconds: int | None) -> str:
    if timeout_seconds is None:
        return prompt
    return prompt.rstrip() + (
        f"\n\nThe total wall-clock budget for this agent invocation is {timeout_seconds} seconds, "
        "including tool calls and your own validation. Check elapsed time during the run. "
        "Keep your current best implementation at the task's declared candidate paths; "
        "do not leave the deliverable empty while experimenting in scratch files. "
        "Stop exploring early enough to install the selected implementation, run the "
        "required candidate checks, and finish before the budget expires. "
        "Report any unfinished checks honestly; the task's acceptance requirements still apply."
    )
