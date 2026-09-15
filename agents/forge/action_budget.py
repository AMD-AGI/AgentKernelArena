"""Fit native driver ceilings to the complete public task action sequence."""
import math
import time


def driver_limits(spec, plan):
    """Outer ceilings only; run_action still enforces every task-owned timeout."""
    remaining = math.floor(min(plan["deadline_unix"],
                               plan.get("phase_deadline_unix", math.inf)) - time.time())
    if remaining < 1:
        raise TimeoutError("Forge task actions have no remaining campaign time")

    def budget(role, *actions):
        return sum(spec.action(role, action).timeout_s for action in actions)

    return {
        "build_timeout_sec": min(remaining, budget("candidate", "compile")),
        "validate_stage_timeout_sec": min(remaining, budget("candidate", "compile", "correctness")),
        # Candidate benchmark includes full correctness; the separately provided
        # baseline can declare different command ceilings from the candidate.
        "bench_timeout_sec": min(remaining, max(
            budget("candidate", "compile", "correctness", "performance"),
            budget("baseline", "compile", "performance"))),
    }
