"""Honor the Arena deadline in the source-pinned native search loop."""
from dataclasses import replace
import math
import time


def bound_session(spec, plan):
    """Reserve time for native assessment/checkpointing after a model session."""
    deadline = min(plan["deadline_unix"], plan.get("phase_deadline_unix", math.inf))
    remaining = deadline - time.time()
    # Initialization already owns its smaller phase budget. The outer campaign
    # still needs time for the native loop to assess and publish its incumbent.
    reserve = 0 if "phase_deadline_unix" in plan else 120
    available = math.floor(remaining - reserve)
    if available < 1:
        raise TimeoutError("Forge has no model-session time before its finalization reserve")
    timeout = min(spec.timeout_sec, available) if spec.timeout_sec is not None else available
    if timeout == spec.timeout_sec:
        return spec
    prompt = spec.user_prompt
    start = prompt.find("## Session deadline\n")
    end = prompt.find("\n## ", start + 1) if start >= 0 else -1
    notice = (f"## Session deadline\nThis session has at most {timeout:g} seconds left within "
              "the shared Arena campaign budget. Submit before that limit so the native "
              "loop can check and publish the candidate.\n")
    if start >= 0 and end >= 0:
        prompt = prompt[:start] + notice + prompt[end:]
    else:
        prompt = notice + "\n" + prompt
    return replace(spec, timeout_sec=timeout, user_prompt=prompt)


def install():
    """Called only after the exact upstream source/signature probe succeeds."""
    from kernelforge.loop import runner

    original = runner.IterationLoop
    if getattr(original, "_arena_absolute_deadline", False):
        return

    class ArenaIterationLoop(original):
        _arena_absolute_deadline = True

        def __init__(self, iter_config, tracker, config=None, resume=False):
            if iter_config.deadline_unix is not None:
                available = max(0, min(iter_config.max_time_hours * 3600,
                                       iter_config.deadline_unix - time.time()))
                # Upstream defaults to a 30-minute reserve even for a short
                # campaign. Keep a bounded reserve without changing any task
                # action, numerical gate, timing sample or measurement timeout.
                reserve = min(iter_config.budget_reserve_sec, max(60, available * .1))
                iter_config = replace(iter_config, budget_reserve_sec=reserve)
            super().__init__(iter_config, tracker, config, resume)

        def _time_remaining(self):
            remaining = super()._time_remaining()
            if self.ic.deadline_unix is not None:
                remaining = min(remaining, self.ic.deadline_unix - time.time())
            return max(0, remaining)

    runner.IterationLoop = ArenaIterationLoop
