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
workload point at a time and its activation is drawn before its expert weights:
a case's weights depend on the case's num_tokens, so the cases cannot share one
set. Each mode therefore builds a case, uses it, and lets it fall out of scope
before the next.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from _aka_benchmark import TimedRun, benchmark_cuda_graph_or_events

import task_baseline
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
            launch = builder(
                num_tokens=int(case["num_tokens"]),
                model_dim=task_inputs.MODEL_DIM,
                inter_dim=task_inputs.INTER_DIM,
                num_experts=task_inputs.NUM_EXPERTS,
                topk=task_inputs.TOPK,
            )
        except NotImplementedError:
            return None
        launches.append(launch)
    return launches


def case_call(inputs: dict[str, Any], launch) -> Callable:
    """A zero-argument call into the candidate, or into the baseline when stub."""
    if launch is None:
        kwargs = task_inputs.call_kwargs(inputs)
        return lambda: task_baseline.run(**kwargs)
    return lambda: launch(
        inputs["hidden_states"],
        inputs["w1"],
        inputs["w2"],
        inputs["topk_weights"],
        inputs["topk_ids"],
        inputs["w1_scale"],
        inputs["w2_scale"],
        inputs["activation"],
        inputs["doweight_stage1"],
    )


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

    ``hold`` makes the next preparation leave the buffers alone, for a caller
    that has loaded the draw it wants the next replay to consume.
    """

    def __init__(
        self, inputs: dict[str, Any], draws: list[dict[str, torch.Tensor]]
    ) -> None:
        if len(draws) < 2:
            raise ValueError("rotation needs at least two distinct draws")
        self._inputs = inputs
        self._draws = draws
        self._next = 0
        self._held = False

    def __call__(self) -> None:
        if self._held:
            self._held = False
            return
        task_inputs.load_draw(self._inputs, self._draws[self._next])
        self._next = (self._next + 1) % len(self._draws)

    def hold(self) -> None:
        self._held = True


def verify_timed_invocation(
    inputs: dict[str, Any], timed: TimedRun, call: Callable, rotation: RotatingDraws
) -> None:
    """Hold the invocation that was timed to the result it reported.

    Correctness and timing are separate invocations and an implementation can
    tell them apart -- the capture state alone is enough -- so the scored path
    has to be judged on its own rather than inferred from the checked one. It
    is judged on three things, over a draw it was not captured against:

    it wrote the output, so the poison cannot survive the replay; it did not
    reproduce the answer it gave for the previous draw, which is what a cache
    keyed on the inputs' identity would do; and it agrees with this same
    implementation run eagerly on the draw the replay just consumed, which is
    what a path that computes honestly when observed and cheaply when captured
    would not.

    The third is a comparison against itself, not against the reference, and
    that distinction is what makes it usable: the shipped implementation does
    not clear the bundle's bar at every shape, so demanding reference accuracy
    here would reject the baseline this task is scored against.

    Nor is the bundle's tolerance the right bar for the self-comparison. An
    implementation need not repeat bit for bit -- a split reduction over
    atomics does not, and the shipped dispatch does not at several of these
    shapes -- so the bar is what this implementation's own repetition costs it,
    measured here by running the eager path twice. A timed path no further from
    the eager one than the eager one is from itself computed the same thing;
    one that is orders beyond that did not.

    Only the operands that vary between calls are redrawn. Holding the expert
    weights fixed is what a deployment does, and a redraw that replaced them
    would fail an implementation for laying them out once.
    """
    if not timed.bound:
        raise RuntimeError(
            "the benchmark did not expose the invocation it timed, so nothing "
            "here can tell whether the scored path computed the operator"
        )
    previous = (
        timed.outputs.detach().clone()
        if isinstance(timed.outputs, torch.Tensor)
        else None
    )
    task_inputs.redraw_call_varying_inputs(inputs)
    if isinstance(timed.outputs, torch.Tensor):
        timed.outputs.fill_(float("nan"))
    rotation.hold()
    got = timed.rerun()
    torch.cuda.synchronize()
    if not isinstance(got, torch.Tensor):
        raise RuntimeError(
            f"the timed invocation returned {type(got).__name__}, so its output "
            "cannot be inspected for whether the replay produced it"
        )
    got = got.detach().clone()
    if not torch.isfinite(got).all():
        raise RuntimeError(
            "the timed invocation left part of its output unwritten: the poison "
            "survived the replay, so the measured work does not produce the result"
        )
    if previous is not None and torch.equal(got, previous):
        raise RuntimeError(
            "the timed invocation reproduced its previous output bit for bit "
            "over a fresh draw, so what was measured is a replay of a cached "
            "answer rather than the operator"
        )
    # Taken after the replay: the buffers hold the draw the graph just read, so
    # calling the implementation eagerly on them is the answer the timed path
    # owed. Twice, because the second reading is the bar for the first -- and
    # cloned, because an implementation is free to return the same output
    # buffer on every call.
    eager = call().detach().clone()
    repeat = call().detach().clone()
    torch.cuda.synchronize()
    spread = task_inputs.result_distance(repeat, eager)
    distance = task_inputs.result_distance(got, eager)
    bar = max(spread * task_inputs.TIMED_PATH_MARGIN, task_inputs.TIMED_PATH_FLOOR)
    if distance > bar:
        raise RuntimeError(
            "the timed invocation disagrees with this same implementation run "
            f"eagerly on the draw it replayed over: distance {distance:.6g} "
            f"against a bar of {bar:.6g} set by its own run-to-run spread "
            f"{spread:.6g}. What was measured is not the computation the "
            "correctness run accepted"
        )


def verify_timed_cost(
    inputs: dict[str, Any],
    timed: TimedRun,
    rotation: RotatingDraws,
    unseen: list[dict[str, torch.Tensor]],
    execution_time_ms: float,
) -> float:
    """Hold the reported time to what the timed unit costs on an unseen draw.

    The rotation defeats a stored result that remembers one draw, not one that
    remembers every draw the samples cycle through. Such an implementation
    still has to compute the first time it meets a draw, so the timed unit is
    replayed once over each of a few draws that were never loaded before, timed
    the way a sample is, and the fastest of those replays is compared with the
    reported mean. Taking the fastest keeps noise from failing an honest
    implementation, while every one of them is a miss for a stored result.

    This has to run directly after the samples, over draws made before them.
    Anything heavier than loading a draw in between -- redrawing through the
    bundle rewrites the weight -- evicts operands the samples found in cache,
    and the replay would then be slower for reasons that have nothing to do
    with what it computes.

    Returns the fastest unseen-draw replay, in milliseconds.
    """
    unseen_ms = []
    for draw in unseen:
        task_inputs.load_draw(inputs, draw)
        rotation.hold()
        unseen_ms.append(timed.rerun_ms())
    fastest = min(unseen_ms)
    bar = execution_time_ms * task_inputs.UNSEEN_DRAW_MARGIN
    if fastest > bar:
        raise RuntimeError(
            f"the timed invocation took {fastest:.6f} ms over a draw it had not "
            f"seen, against a reported {execution_time_ms:.6f} ms per call and a "
            f"bar of {bar:.6f} ms. The samples were served faster than the "
            "operator runs on new inputs, so the reported time is not its cost"
        )
    return fastest


def time_cases(launches: list | None) -> list[dict[str, Any]]:
    """Time every case under the task's own sampling protocol.

    The protocol belongs to the task, not to the caller: a candidate is only
    worth keeping if it holds up under the protocol that decides the score.
    """
    samples = []
    for index, case in enumerate(task_inputs.CASES):
        inputs = task_inputs.build_case_inputs(case)
        call = case_call(inputs, None if launches is None else launches[index])
        rotation = RotatingDraws(
            inputs, task_inputs.call_varying_draws(inputs, task_inputs.TIMED_DRAW_SEEDS)
        )
        unseen = task_inputs.call_varying_draws(inputs, task_inputs.UNSEEN_DRAW_SEEDS)
        timed = TimedRun()
        execution_time_ms, metadata = benchmark_cuda_graph_or_events(
            call,
            warmup=task_inputs.BENCH_WARMUP,
            repetition=task_inputs.BENCH_REPETITION,
            target_ms=task_inputs.BENCH_TARGET_MS,
            prepare_fn=rotation,
            timed_run=timed,
        )
        repeats = metadata.get("benchmark_effective_repeats")
        if repeats != 1:
            raise RuntimeError(
                f"the capture batched {repeats} invocations into one replay, so "
                "each sample reports their average and one retained computation "
                "would be charged at a fraction of its cost; this task requires "
                "one logical invocation per replay"
            )
        try:
            metadata["unseen_draw_ms"] = verify_timed_cost(
                inputs, timed, rotation, unseen, execution_time_ms
            )
            verify_timed_invocation(inputs, timed, call, rotation)
        except RuntimeError as failure:
            raise RuntimeError(f"case {case['case_id']}: {failure}") from failure
        samples.append(
            {
                "case_id": str(case["case_id"]),
                "num_tokens": int(case["num_tokens"]),
                "execution_time_ms": execution_time_ms,
                "metadata": metadata,
            }
        )
    return samples
