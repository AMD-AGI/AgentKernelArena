"""Reserve the remainder of a rewrite campaign for native optimization."""
from dataclasses import replace
from functools import wraps
import inspect
import json
from pathlib import Path
import time


def bound_port(original, plan):
    """Wrap the source-pinned PORT loop without changing its acceptance gate."""
    signature = inspect.signature(original)

    @wraps(original)
    async def run(*args, **kwargs):
        arguments = signature.bind(*args, **kwargs)
        started = time.time()
        fraction = plan["agent_config"]["initialization_budget_fraction"]
        remaining = max(0, plan["deadline_unix"] - started)
        deadline = started + remaining * fraction
        existing = arguments.arguments.get("stop_at_unix")
        if existing is not None and existing > 0:
            deadline = min(deadline, existing)
        arguments.arguments["stop_at_unix"] = deadline
        previous = plan.get("phase_deadline_unix")
        if previous is not None:
            deadline = min(deadline, previous)
            arguments.arguments["stop_at_unix"] = deadline
        plan["phase_deadline_unix"] = deadline
        evidence = {"status": "RUNNING", "started_unix": started,
                    "phase_deadline_unix": deadline,
                    "campaign_deadline_unix": plan["deadline_unix"],
                    "initialization_budget_fraction": fraction,
                    "scope": "PORT budget only; no Arena verdict or search completion claim"}
        path = Path(plan["result"]).with_name("port_budget.json")

        def save():
            path.write_text(json.dumps(evidence, indent=2) + "\n")

        try:
            save()
            result = await original(*arguments.args, **arguments.kwargs)
            if not result.ok and time.time() >= deadline:
                result = replace(result, error_tail="PORT initialization budget expired; "
                                 "remaining campaign time was reserved for native OPTIMIZE\n"
                                 + result.error_tail)
            evidence.update(status="PASS" if result.ok else "FAILED", attempts=result.attempts,
                            error=result.error_tail)
            return result
        except BaseException as exc:
            evidence.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if previous is None:
                plan.pop("phase_deadline_unix", None)
            else:
                plan["phase_deadline_unix"] = previous
            evidence["finished_unix"] = time.time()
            save()

    return run
