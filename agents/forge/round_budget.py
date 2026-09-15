"""Reserve native assessment and delivery time within the original deadline.

This is installed outside the incumbent-recovery subclass, after the exact
upstream source probe. It bounds the complete assessment, including recovery's
canonical acceptance, without changing any numerical gate or sample count.
"""
import asyncio
import time

from agents.forge.deadline import FINALIZATION_RESERVE_SEC, _active_round_budget


def install():
    from kernelforge.loop import runner

    original = runner.IterationLoop
    if getattr(original, "_arena_round_budget", False):
        return

    class ArenaRoundBudgetLoop(original):
        _arena_round_budget = True
        _arena_assessment_expired = False
        _arena_assessment_high_water = 0.0
        _arena_initial_measurement = 0.0

        def _time_remaining(self):
            # Native admission, analysis and optional lesson summarization all
            # see the same usable clock. The outer process retains its deadline.
            return max(0.0, super()._time_remaining() - FINALIZATION_RESERVE_SEC)

        def _measurement_estimate_sec(self):
            # Initial measurement runs three full bridge benchmark invocations.
            # A KEEP also needs outer correctness and canonical acceptance.
            # These are estimates for admission, never shortened task timeouts.
            return max(super()._measurement_estimate_sec(),
                       self._arena_initial_measurement * 5 / 3,
                       self._arena_assessment_high_water * 1.25)

        def _analysis_deadline_unix(self):
            return min(super()._analysis_deadline_unix(), time.time() + max(
                0.0, self._time_remaining() - self._measurement_estimate_sec()))

        def _is_budget_exhausted(self):
            return self._arena_assessment_expired or super()._is_budget_exhausted()

        async def _measure_baseline(self):
            started = time.monotonic()
            try:
                return await super()._measure_baseline()
            finally:
                self._arena_initial_measurement = time.monotonic() - started

        async def run(self, *args, **kwargs):
            token = _active_round_budget.set(self)
            try:
                return await super().run(*args, **kwargs)
            finally:
                _active_round_budget.reset(token)

        async def run_one_iteration(self, iteration, plan="", *, benchmark_measurement=None):
            started = time.monotonic()
            available = self._time_remaining()
            if available > 0:
                task = asyncio.create_task(super().run_one_iteration(
                    iteration, plan=plan, benchmark_measurement=benchmark_measurement))
                try:
                    result = await asyncio.wait_for(task, timeout=available)
                except asyncio.TimeoutError:
                    if not task.cancelled():
                        raise  # Preserve an actual inner driver/provider error.
                else:
                    self._arena_assessment_high_water = max(
                        self._arena_assessment_high_water, time.monotonic() - started)
                    return result
            # Native process-group cancellation must unwind before returning.
            # The native loop records/reverts this unverified attempt, then
            # publishes its prior immutable best through its ordinary path.
            # No candidate, commit or final verdict is synthesized here.
            self._arena_assessment_expired = True
            summary = ("Arena assessment deadline reached before all required checks "
                       "completed; unfinished attempt cancelled for native finalization")
            print("  [arena] " + summary)
            return runner.IterationResult(
                iteration=iteration, duration_sec=time.monotonic() - started,
                validation_passed=False, validation_outcome="timeout",
                validation_summary=summary, kept=False)

    runner.IterationLoop = ArenaRoundBudgetLoop
