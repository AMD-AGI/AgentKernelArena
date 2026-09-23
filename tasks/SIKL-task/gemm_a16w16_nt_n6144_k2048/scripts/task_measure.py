# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""The one implementation of how this task measures anything.

Two callers evaluate this task and they must not drift: the Arena harness
produces the score, and the rewrite driver produces the number KernelForge's
loop keeps or reverts a candidate on. If those two ever measured differently,
the pipeline would optimize for something the score does not reward.

They differ only in where the candidate comes from and how results are printed.
Everything else -- how launches are built, how a case is compared, how a case is
timed -- lives here and is imported by both.

Inputs are built one case at a time, because the schema bundle initializes one
workload point at a time and its ``a`` is drawn before its ``b``: a case's
weight depends on the case's m, so the cases cannot share one. Each mode
therefore builds a case, uses it, and lets it fall out of scope before the next.
"""

from __future__ import annotations

import secrets
from typing import Any, Callable

import torch

from _aka_benchmark import TimedRun, benchmark_cuda_graph_or_events

import task_baseline
import task_compare
import task_inputs
import task_reference


def build_launches(builder) -> list | None:
    """Build one launch per scored case, or None while the candidate is a stub.

    The builder takes the case shape and no data, so every launch is built
    before any input exists. Only ``NotImplementedError`` counts as "not ported
    yet"; every other failure propagates, so a broken port fails the task
    instead of silently scoring the baseline a second time.
    """
    if builder is None:
        return None
    launches = []
    for case in task_inputs.CASES:
        try:
            launch = builder(m=int(case["m"]), n=task_inputs.N, k=task_inputs.K)
        except NotImplementedError:
            return None
        launches.append(launch)
    return launches


def case_call(inputs: dict[str, Any], launch) -> Callable:
    """A zero-argument call into the candidate, or into the baseline when stub."""
    if launch is None:
        kwargs = task_inputs.call_kwargs(inputs)
        return lambda: task_baseline.run(**kwargs)
    return lambda: launch(inputs["a"], inputs["b"])


def compare_cases(launches: list | None) -> list[dict[str, Any]]:
    """Judge every case, and judge the production implementation beside it.

    The verdict is the bundle's comparison callback, applied to the candidate
    and -- separately -- to the shipped implementation. The second reading
    changes no outcome; it is the only thing that tells a reader whether a
    failing case is the candidate's fault or a bar production does not clear
    either.
    """
    results = []
    for index, case in enumerate(task_inputs.CASES):
        inputs = task_inputs.build_case_inputs(case)
        kwargs = task_inputs.call_kwargs(inputs)
        expected = task_reference.run(**kwargs)
        baseline_passed, baseline_detail = task_inputs.verdict(
            task_baseline.run(**kwargs), expected
        )
        got = case_call(inputs, None if launches is None else launches[index])()
        torch.cuda.synchronize()
        passed, detail = task_inputs.verdict(got, expected)
        results.append(
            {
                "case_id": str(case["case_id"]),
                "passed": passed,
                "detail": detail,
                "baseline_passed": baseline_passed,
                "baseline_detail": baseline_detail,
            }
        )
    return results


# The timed samples rotate through TIMED_DRAWS draws of the call-varying
# operands. After the samples, the timed unit runs once over each of UNSEEN_DRAWS
# further draws, timed like a sample; the fastest of those may take at most
# UNSEEN_DRAW_MARGIN times the reported mean. CHECKED_SAMPLES reported samples,
# chosen at random, and every unseen-draw invocation have their outputs compared
# with the reference on the draw they consumed.
TIMED_DRAWS = 3
UNSEEN_DRAWS = 4
UNSEEN_DRAW_MARGIN = 1.5
CHECKED_SAMPLES = 8


def fresh_draw_seeds(count: int) -> list[int]:
    """Distinct seeds from the operating system's entropy source, never SEED.

    The seeds are drawn when the case is timed, so neither the draws a sample
    reads nor the ones held out for the cost check can be known to the code
    being measured beforehand; fixed seeds would let an implementation generate
    the same draws itself and store their results.
    """
    rng = secrets.SystemRandom()
    seeds: list[int] = []
    while len(seeds) < count:
        seed = rng.randrange(2**63)
        if seed != task_inputs.SEED and seed not in seeds:
            seeds.append(seed)
    return seeds


def choose_checked_samples(repetition: int, count: int) -> list[int]:
    """Indices of the reported samples whose outputs are checked, chosen secretly."""
    return sorted(secrets.SystemRandom().sample(range(repetition), min(count, repetition)))


class RotatingDraws:
    """Preparation that loads a different call-varying draw before every replay.

    A replay executes recorded kernels, and those kernels may themselves decide
    at run time whether to compute: one that compares its operands against a
    copy of the last ones it saw and replays a stored output on a match skips
    the operator on every sample, because every sample reads the same bytes.
    Loading another draw before each replay makes consecutive samples differ in
    value while keeping every buffer's storage, which is all a captured graph
    depends on. The draws come from ``task_inputs.call_varying_draws``, so the
    weights stay fixed across samples the way a deployment holds them.

    Supplying a preparation callback is also how the benchmark is told to
    capture a single logical invocation per replay rather than batching as many
    as fill ``target_ms`` and dividing by the count. ``time_cases`` asserts the
    count it gets rather than trusting this.

    ``serve`` makes the next preparation load a given draw instead of the next
    one in the rotation, so an invocation over a draw of the caller's choosing
    goes through the same preparation, on the same stream, as every sample.
    ``consumed`` is the draw the latest preparation loaded.
    """

    def __init__(
        self, inputs: dict[str, Any], draws: list[dict[str, torch.Tensor]]
    ) -> None:
        if len(draws) < 2:
            raise ValueError("rotation needs at least two distinct draws")
        self._inputs = inputs
        self._draws = draws
        self._next = 0
        self._served: dict[str, torch.Tensor] | None = None
        self.consumed: dict[str, torch.Tensor] | None = None

    def __call__(self) -> None:
        if self._served is not None:
            draw, self._served = self._served, None
        else:
            draw = self._draws[self._next]
            self._next = (self._next + 1) % len(self._draws)
        task_inputs.load_draw(self._inputs, draw)
        self.consumed = draw

    def serve(self, draw: dict[str, torch.Tensor]) -> None:
        self._served = draw


def host_copy(outputs: Any) -> Any:
    # A device-to-host copy reads the outputs without writing device memory, so
    # keeping them evicts little of what the next invocation finds in cache.
    return outputs.detach().to("cpu") if isinstance(outputs, torch.Tensor) else outputs


class SampleChecks:
    """``after_sample`` observer keeping the outputs of the checked samples.

    The outputs are copied after the sample has run, so nothing an invocation
    can observe while it runs tells it whether its result will be checked.
    """

    def __init__(self, rotation: RotatingDraws, indices: list[int]) -> None:
        self._rotation = rotation
        self._indices = frozenset(indices)
        self._index = 0
        self.kept: list[tuple[dict[str, torch.Tensor], Any]] = []

    def __call__(self, outputs: Any) -> None:
        if self._index in self._indices:
            self.kept.append((self._rotation.consumed, host_copy(outputs)))
        self._index += 1


def run_unseen_draws(
    timed: TimedRun, rotation: RotatingDraws, unseen: list[dict[str, torch.Tensor]]
) -> tuple[list[float], list[tuple[dict[str, torch.Tensor], Any]]]:
    """Time the timed unit once over each unseen draw and keep what it wrote.

    Each invocation is prepared by the rotation and timed through the reported
    sample path, so it differs from a sample only in consuming a draw the
    implementation has never read. This runs directly after the samples, with
    nothing heavier than loading a draw and reading back the previous outputs in
    between; redrawing through the bundle here would rewrite the weight, evict
    what the samples found in cache and slow these invocations for reasons
    unrelated to what they compute.
    """
    unseen_ms, kept = [], []
    for draw in unseen:
        rotation.serve(draw)
        unseen_ms.append(timed.rerun_ms())
        kept.append((draw, host_copy(timed.outputs)))
    return unseen_ms, kept


def verify_timed_outputs(
    inputs: dict[str, Any],
    kept: list[tuple[dict[str, torch.Tensor], Any]],
    *,
    baseline: bool,
) -> dict[str, Any]:
    """Judge every kept timed output against the reference on the draw it consumed.

    The kept outputs are those of the checked samples and of every unseen-draw
    invocation. An unseen draw is new to the implementation, so it has to be
    computed to be right; a checked sample was chosen without the implementation
    being able to tell. Each output has to satisfy the comparison's output
    contract, and a candidate has to pass the bundle's gate -- the same gate
    ``compare_cases`` applies. The production implementation does not clear
    that gate at every shape, so its numerical verdict is reported, as in
    ``compare_cases``, and never applied.
    """
    expected_by_draw: dict[int, torch.Tensor] = {}
    failures = []
    for draw, got in kept:
        if id(draw) not in expected_by_draw:
            task_inputs.load_draw(inputs, draw)
            expected_by_draw[id(draw)] = task_reference.run(**task_inputs.call_kwargs(inputs))
        expected = expected_by_draw[id(draw)]
        if isinstance(got, torch.Tensor):
            got = got.to(expected.device)
        try:
            task_compare.validate_comparison(got, expected)
        except AssertionError as error:
            raise RuntimeError(
                f"a timed invocation broke the output contract: {error}"
            ) from error
        passed, detail = task_inputs.verdict(got, expected)
        if not passed:
            failures.append(detail)
    if failures and not baseline:
        raise RuntimeError(
            f"{len(failures)}/{len(kept)} checked timed invocations did not produce "
            f"the operator's result for the draw they consumed; first: {failures[0]}"
        )
    return {
        "checked_invocations": len(kept),
        "numerical_failures": len(failures),
        "first_failure": failures[0] if failures else None,
    }


def verify_timed_cost(unseen_ms: list[float], execution_time_ms: float) -> float:
    """Hold the reported time to what the timed unit costs on an unseen draw.

    The rotation defeats a stored result that remembers one draw, not one that
    remembers every draw the samples cycle through. Such an implementation
    still has to compute the first time it meets a draw, and its output on an
    unseen draw is checked, so the fastest unseen-draw invocation is what the
    operator costs on new inputs. Taking the fastest keeps noise from failing
    an honest implementation, while every one of them is a miss for a stored
    result.

    Returns the fastest unseen-draw invocation, in milliseconds.
    """
    fastest = min(unseen_ms)
    bar = execution_time_ms * UNSEEN_DRAW_MARGIN
    if fastest > bar:
        raise RuntimeError(
            f"the timed invocation took {fastest:.6f} ms over a draw it had not "
            f"seen, against a reported {execution_time_ms:.6f} ms per call and a "
            f"bar of {bar:.6f} ms. The samples were served faster than the "
            "operator runs on new inputs, so the reported time is not its cost"
        )
    return fastest


def _byte_snapshot(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # Byte views also cover packed dtypes whose equality ops may be missing.
    return {name: value.detach().view(torch.uint8).clone() for name, value in tensors.items()}


def _assert_unchanged(inputs: dict[str, Any], snapshot: dict[str, torch.Tensor]) -> None:
    for name, expected in snapshot.items():
        if not torch.equal(inputs[name].view(torch.uint8), expected):
            raise RuntimeError(f"the timed invocation modified its input {name}")


def time_cases(launches: list | None) -> list[dict[str, Any]]:
    """Time every case under the task's own sampling protocol.

    The protocol belongs to the task, not to the caller: a candidate is only
    worth keeping if it holds up under the protocol that decides the score.
    """
    samples = []
    for index, case in enumerate(task_inputs.CASES):
        inputs = task_inputs.build_case_inputs(case)
        call = case_call(inputs, None if launches is None else launches[index])
        seeds = fresh_draw_seeds(TIMED_DRAWS + UNSEEN_DRAWS)
        timed_seeds, unseen_seeds = seeds[:TIMED_DRAWS], seeds[TIMED_DRAWS:]
        rotation = RotatingDraws(inputs, task_inputs.call_varying_draws(inputs, timed_seeds))
        unseen = task_inputs.call_varying_draws(inputs, unseen_seeds)
        weights = _byte_snapshot(
            {name: inputs[name] for name in task_inputs.PERSISTENT_INPUTS}
        )
        checked = choose_checked_samples(task_inputs.BENCH_REPETITION, CHECKED_SAMPLES)
        checks = SampleChecks(rotation, checked)
        timed = TimedRun()
        timed.after_sample = checks
        execution_time_ms, metadata = benchmark_cuda_graph_or_events(
            call,
            warmup=task_inputs.BENCH_WARMUP,
            repetition=task_inputs.BENCH_REPETITION,
            target_ms=task_inputs.BENCH_TARGET_MS,
            prepare_fn=rotation,
            timed_run=timed,
        )
        try:
            repeats = metadata.get("benchmark_effective_repeats")
            if repeats != 1:
                raise RuntimeError(
                    f"the capture batched {repeats} invocations into one replay, so "
                    "each sample reports their average and one retained computation "
                    "would be charged at a fraction of its cost; this task requires "
                    "one logical invocation per replay"
                )
            if not timed.bound:
                raise RuntimeError(
                    "the benchmark did not expose the invocation it timed, so nothing "
                    "here can tell whether the scored path computed the operator"
                )
            unseen_ms, unseen_kept = run_unseen_draws(timed, rotation, unseen)
            torch.cuda.synchronize()
            _assert_unchanged(inputs, {**weights, **_byte_snapshot(rotation.consumed)})
            metadata["unseen_draw_ms"] = verify_timed_cost(unseen_ms, execution_time_ms)
            metadata["timed_output_check"] = verify_timed_outputs(
                inputs, checks.kept + unseen_kept, baseline=launches is None
            )
        except RuntimeError as failure:
            raise RuntimeError(f"case {case['case_id']}: {failure}") from failure
        metadata.update(
            timed_draw_seeds=timed_seeds,
            unseen_draw_seeds=unseen_seeds,
            checked_samples=checked,
        )
        samples.append(
            {
                "case_id": str(case["case_id"]),
                "m": int(case["m"]),
                "execution_time_ms": execution_time_ms,
                "metadata": metadata,
            }
        )
    return samples
