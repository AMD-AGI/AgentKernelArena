"""Honor the Arena deadline in the source-pinned native search loop."""
import asyncio
from contextvars import ContextVar
from dataclasses import replace
from functools import wraps
import inspect
import math
import time

_session_gates = ContextVar("arena_forge_session_gates", default=None)


class SessionBudgetExceeded(RuntimeError):
    """One implementer attempt expired; the outer phase may still have time.

    Deliberately not TimeoutError: the pinned PORT loop interprets that type as
    exhaustion of the whole phase and would stop instead of retrying an attempt.
    """


def _available(plan):
    deadline = min(plan["deadline_unix"], plan.get("phase_deadline_unix", math.inf))
    reserve = 0 if "phase_deadline_unix" in plan else 120
    return deadline - time.time() - reserve


def bound_agent(agent, plan, timeout_sec):
    """One wall clock covers all provider resumes and their intervening gates.

    The pinned provider bounds each SDK turn separately. Its outer Stop gate
    can resume with the original timeout, so bounding AgentRunSpec alone cannot
    bound an implementer invocation. Cancellation unwinds the native provider
    and measurement subprocesses before the loop can assess the remaining code.
    """
    signature = inspect.signature(agent)

    @wraps(agent)
    async def run(*args, **kwargs):
        available = min(timeout_sec, _available(plan))
        if available <= 0:
            raise TimeoutError("Forge has no model-session time before its finalization reserve")
        bound = signature.bind(*args, **kwargs)
        sink = bound.arguments.get("session_sink")
        gates = []
        token = _session_gates.set(gates)
        try:
            task = asyncio.create_task(agent(*args, **kwargs))
            try:
                return await asyncio.wait_for(task, timeout=available)
            except asyncio.TimeoutError:
                if not task.cancelled():
                    raise  # An inner task-action/provider timeout keeps its evidence.
                if sink is not None:
                    sink["end_reason"] = "session_timeout"
                    sink["gate_passed"] = False
                    sink.pop("benchmark_measurement", None)
                raise SessionBudgetExceeded(f"Forge implementer exceeded its {available:g}s total session budget") from None
        except BaseException:
            # Native agent_fn finalizes after a normal outer-gate loop, but
            # cancellation during resume/_on_stop bypasses that code. Keep
            # the same integrity verdict and restoration callback on error.
            for gate in gates:
                gate.finalize_integrity()
                if sink is not None and not sink.get("integrity_violation"):
                    sink.update(integrity_verdict=gate.integrity_verdict,
                                integrity_violation=gate.integrity_violation,
                                integrity_reason=gate.integrity_reason,
                                integrity_restore=gate.restore_protected_files)
            raise
        finally:
            _session_gates.reset(token)

    return run


def bound_session(spec, plan):
    """Reserve time for native assessment/checkpointing after a model session."""
    # Initialization already owns its smaller phase budget. The outer campaign
    # still needs time for the native loop to assess and publish its incumbent.
    available = math.floor(_available(plan))
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


def install(plan=None):
    """Called only after the exact upstream source/signature probe succeeds."""
    from kernelforge.loop import insession_gate, runner

    gate_class = insession_gate.InSessionGate
    if not getattr(gate_class, "_arena_session_finalization", False):
        class ArenaSessionGate(gate_class):
            _arena_session_finalization = True

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                gates = _session_gates.get()
                if gates is not None:
                    gates.append(self)

        insession_gate.InSessionGate = ArenaSessionGate

    original = runner.IterationLoop
    if getattr(original, "_arena_absolute_deadline", False):
        return

    class ArenaIterationLoop(original):
        _arena_absolute_deadline = True

        def __init__(self, iter_config, tracker, config=None, resume=False):
            if plan is not None:
                from agents.forge.action_budget import driver_limits
                from agents.forge.task_context import TaskContext
                limits = driver_limits(TaskContext.load(plan["context"]).spec, plan)
                iter_config = replace(iter_config, **limits)
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
