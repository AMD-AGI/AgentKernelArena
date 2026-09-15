"""Recover a candidate with failed timing without treating the baseline as it.

Only installed after the exact upstream source probe. Native validation, three
measurements, canonical acceptance and commit publication remain mandatory.
"""
from contextvars import ContextVar
import math
import time

_active_loop = ContextVar("arena_forge_incumbent_loop", default=None)


def needs_incumbent():
    loop = _active_loop.get()
    return bool(loop is not None and getattr(loop, "_arena_unmeasured_incumbent", False))


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def scoreable_trial(result, baseline):
    """Require all measured cases, never just a scalar score."""
    detail = result.bench_detail or {}
    if not isinstance(detail, dict):
        return False
    if (not result.validation_passed or result.crashed or result.integrity_violation
            or result.workspace_contention or detail.get("success") is not True
            or detail.get("case_coverage_complete") is not True
            or not _positive(result.wall_ms) or not _positive(result.mean_case_speedup)):
        return False
    measurements = detail.get("measurements")
    if (not isinstance(measurements, list) or len(measurements) != 3 or not baseline
            or not all(_positive(value) for value in baseline.values())):
        return False
    for measurement in [detail, *measurements]:
        if not isinstance(measurement, dict) or measurement.get("success") is False:
            return False
        cases = measurement.get("case_times", {})
        if (not isinstance(cases, dict) or set(cases) != set(baseline) or measurement.get("unscored_cases")
                or not all(_positive(value) for value in cases.values())):
            return False
    return True


def install():
    """Called after the exact source probe; no installed package is modified."""
    from kernelforge.loop import insession_gate, runner
    original = runner.IterationLoop
    if getattr(original, "_arena_incumbent_recovery", False):
        return

    class ArenaIncumbentLoop(original):
        _arena_incumbent_recovery = True
        _arena_unmeasured_incumbent = False

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            _active_loop.set(self)

        def _mark_unmeasured(self):
            self._arena_unmeasured_incumbent = True
            self._best_case_times = {}
            self.best_wall_ms = None
            self.best_mean_case_speedup = None
            self.search_start_mean_case_speedup = None
            _active_loop.set(self)
            self._persist_scoring_state()
            print("  [arena] Initial candidate has no valid timing. The first fully "
                  "validated, measured candidate can establish an incumbent; "
                  "independent baseline timings remain the score denominator.")

        async def _measure_baseline(self):
            result = await super()._measure_baseline()
            if result is None and self._baseline_case_times:
                self._mark_unmeasured()
            return result

        def _incumbent_mean_case_speedup(self):
            if self._arena_unmeasured_incumbent:
                return None
            return super()._incumbent_mean_case_speedup()

        def _restore_scoring_state(self):
            super()._restore_scoring_state()
            state = self.run_state
            if (state.baseline_wall_ms is None and not state.best.commit_hash
                    and state.best.wall_ms is None and self._baseline_case_times):
                self._mark_unmeasured()

        async def run_one_iteration(self, iteration, plan="", *, benchmark_measurement=None):
            result = await super().run_one_iteration(
                iteration, plan=plan, benchmark_measurement=benchmark_measurement)
            if not self._arena_unmeasured_incumbent or not scoreable_trial(result, self._baseline_case_times):
                return result
            # Native code calls canonical acceptance only for speed improvements.
            # The first usable candidate must pass the identical final gate.
            if not result.kept:
                started = time.time()
                canonical = await runner.accept_candidate(
                    self.ic.workspace_dir, timeout_cap_sec=self.ic.validate_stage_timeout_sec,
                    candidate_label=f"iteration {iteration}",
                )
                self._observe_measurement(started)
                result.duration_sec += time.time() - started
                if not canonical.passed:
                    result.validation_passed = False
                    result.validation_outcome = canonical.outcome or "canonical_correctness_failure"
                    result.validation_summary += "\n  Canonical correctness suite: FAILED — " + canonical.detail
                    result.error_output = canonical.output
                    return result
                result.kept = True
            result.bench_detail = {
                **result.bench_detail,
                "arena_selection_reason": "first_scoreable_candidate_after_initial_measurement_failure",
                "arena_speedup_improvement_claimed": False,
            }
            print("  [arena] First fully measured candidate admitted; this establishes "
                  "validity, not a speedup over the independent baseline.")
            return result  # native commit must succeed before leaving recovery

        def _promote_best(self, result):
            if self._arena_unmeasured_incumbent:
                self.search_start_mean_case_speedup = result.mean_case_speedup
            super()._promote_best(result)
            self._arena_unmeasured_incumbent = False

    runner.IterationLoop = ArenaIncumbentLoop
    gate_class = insession_gate.InSessionGate

    class ArenaIncumbentGate(gate_class):
        def __init__(self, *args, **kwargs):
            if needs_incumbent():
                # Outer native assessment still requires all timed suites and
                # canonical acceptance before committing any implementation.
                kwargs["correctness_only"] = True
            super().__init__(*args, **kwargs)

    insession_gate.InSessionGate = ArenaIncumbentGate
