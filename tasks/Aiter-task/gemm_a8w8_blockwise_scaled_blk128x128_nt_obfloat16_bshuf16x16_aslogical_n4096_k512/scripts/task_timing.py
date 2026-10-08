"""Canonical GPU timing of one workload row, with the timed invocations checked.

A captured kernel can decide on device, per invocation, whether to compute:
keyed on its inputs' values (return a stored result for a draw it has seen) or
on its own output buffer. The protocol therefore rotates fresh draws of the
call-varying operands through the samples, requires one logical invocation per
graph replay, checks the outputs of secretly chosen samples and of invocations
over draws never read before, and holds the reported time to what those unseen
draws cost. The draws come from seeds the code being measured cannot know.
"""

from __future__ import annotations

import secrets

import torch

from scripts.task_api import assert_unmodified, compare_outputs, snapshot
from scripts.task_inputs import call_varying_draws, load_draw, persistent_inputs, row_seed

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


def fresh_draw_seeds(count, base_seed):
    """Distinct seeds from the operating system's entropy source, never the base seed."""
    rng = secrets.SystemRandom()
    seeds = []
    while len(seeds) < count:
        seed = rng.randrange(2**63)
        if seed != base_seed and seed not in seeds:
            seeds.append(seed)
    return seeds


def choose_checked_samples(repetition, count):
    """Indices of the reported samples whose outputs are checked, chosen secretly."""
    return sorted(secrets.SystemRandom().sample(range(repetition), min(count, repetition)))


class RotatingDraws:
    """Preparation that loads a different call-varying draw before every replay.

    Loading another draw before each replay makes consecutive samples differ in
    value while keeping every buffer's storage, which is all a captured graph
    depends on. Supplying a preparation callback is also how the benchmark is
    told to capture a single logical invocation per replay rather than batching
    as many as fill ``target_ms``; ``time_row`` asserts the count it gets.

    ``serve`` makes the next preparation load a given draw instead of the next
    one in the rotation. ``consumed`` is the draw the latest preparation loaded.
    """

    def __init__(self, values, draws):
        if len(draws) < 2:
            raise ValueError("rotation needs at least two distinct draws")
        self._values = values
        self._draws = draws
        self._next = 0
        self._served = None
        self.consumed = None

    def __call__(self):
        if self._served is not None:
            draw, self._served = self._served, None
        else:
            draw = self._draws[self._next]
            self._next = (self._next + 1) % len(self._draws)
        load_draw(self._values, draw)
        self.consumed = draw

    def serve(self, draw):
        self._served = draw


def host_copy(value):
    # A device-to-host copy reads the outputs without writing device memory.
    # A launch may reuse its output buffers, so the copy is explicit.
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {name: host_copy(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(host_copy(item) for item in value)
    return value


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {name: to_device(item, device) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(to_device(item, device) for item in value)
    return value


class SampleChecks:
    """``after_sample`` observer keeping the outputs of the checked samples.

    The outputs are copied after the sample has run, so nothing an invocation
    can observe while it runs tells it whether its result will be checked.
    """

    def __init__(self, rotation, indices):
        self._rotation = rotation
        self._indices = frozenset(indices)
        self._index = 0
        self.kept = []

    def __call__(self, result):
        if self._index in self._indices:
            self.kept.append((self._rotation.consumed, host_copy(result)))
        self._index += 1


def run_unseen_draws(timed, rotation, unseen):
    """Time the timed unit once over each unseen draw and keep what it wrote."""
    unseen_ms, kept = [], []
    for draw in unseen:
        rotation.serve(draw)
        unseen_ms.append(timed.rerun_ms())
        kept.append((draw, host_copy(timed.outputs)))
    return unseen_ms, kept


def verify_timed_outputs(values, kept, reference, definition, row, device):
    """Compare every kept timed output with the reference on the draw it consumed."""
    expected_by_draw, results = {}, []
    for draw, got in kept:
        if id(draw) not in expected_by_draw:
            load_draw(values, draw)
            expected_by_draw[id(draw)] = reference(**values)
        results.append(compare_outputs(to_device(got, device), expected_by_draw[id(draw)],
                                       definition, row, device))
    failed = [result for result in results if result["status"] != "PASS"]
    metadata = {"checked_invocations": len(results), "failed_invocations": len(failed)}
    if not failed:
        return {"status": "PASS", "metadata": metadata}
    kinds = {result["failure_kind"] for result in failed}
    kind = "numerical_mismatch" if kinds == {"numerical_mismatch"} else sorted(kinds - {"numerical_mismatch"})[0]
    first = next(result for result in failed if result["failure_kind"] == kind)
    return {"status": "FAIL", "failure_kind": kind,
            "reason": f"{len(failed)}/{len(results)} checked timed invocations failed; first: {first['reason']}",
            "metadata": {**metadata, "first_failure": first}}


def verify_timed_cost(unseen_ms, execution_time_ms):
    """Hold the reported time to what the timed unit costs on an unseen draw.

    An implementation that remembers every draw the samples cycle through still
    has to compute the first time it meets a draw, and its output on an unseen
    draw is checked, so the fastest unseen-draw invocation is what the operator
    costs on new inputs.
    """
    fastest = min(unseen_ms)
    bar = execution_time_ms * UNSEEN_DRAW_MARGIN
    metadata = {"unseen_draw_ms": fastest, "unseen_draw_samples_ms": list(unseen_ms),
                "reported_ms": execution_time_ms}
    if fastest > bar:
        return {"status": "FAIL", "failure_kind": "timing_input_memoized",
                "reason": (f"Timed invocation took {fastest:.6f} ms over a draw it had not seen, against "
                           f"a reported {execution_time_ms:.6f} ms per call and a bar of {bar:.6f} ms: the "
                           "samples were served faster than the operator runs on new inputs"),
                "metadata": metadata}
    return {"status": "PASS", "metadata": metadata}


def time_row(call, reference, values, definition, row, policy, *, role, baseline_diagnostic=False,
             device="cuda"):
    from _aka_benchmark import TimedRun, benchmark_cuda_graph_or_events

    seeds = fresh_draw_seeds(TIMED_DRAWS + UNSEEN_DRAWS, row_seed(policy, row))
    timed_seeds, unseen_seeds = seeds[:TIMED_DRAWS], seeds[TIMED_DRAWS:]
    rotation = RotatingDraws(values, call_varying_draws(values, definition, row, policy, timed_seeds, device))
    unseen = call_varying_draws(values, definition, row, policy, unseen_seeds, device)
    weights = snapshot({name: values[name] for name in persistent_inputs(definition, policy)})
    checked = choose_checked_samples(policy["repetition"], CHECKED_SAMPLES)
    checks = SampleChecks(rotation, checked)
    timed = TimedRun()
    timed.after_sample = checks
    execution_time_ms, timing = benchmark_cuda_graph_or_events(
        lambda: call(**values), warmup=policy["warmup"], repetition=policy["repetition"],
        target_ms=policy["target_ms"], prepare_fn=rotation, timed_run=timed)
    protocol = {"timing": timing, "timed_draw_seeds": timed_seeds,
                "unseen_draw_seeds": unseen_seeds, "checked_samples": checked}
    repeats = timing.get("benchmark_effective_repeats")
    if repeats != 1:
        return {"status": "FAIL", "failure_kind": "timing_protocol",
                "reason": (f"Capture batched {repeats} invocations into one replay, so each sample "
                           "reports their average; this task requires one logical invocation per replay"),
                "metadata": protocol}
    if not timed.bound:
        raise RuntimeError("Benchmark did not expose the invocation it timed")
    unseen_ms, unseen_kept = run_unseen_draws(timed, rotation, unseen)
    torch.cuda.synchronize()
    assert_unmodified(values, {**weights, **snapshot(rotation.consumed)})
    cost = verify_timed_cost(unseen_ms, execution_time_ms)
    checked_outputs = verify_timed_outputs(values, checks.kept + unseen_kept, reference, definition, row, device)
    protocol.update(unseen_draw_cost=cost["metadata"], timed_output_correctness=checked_outputs)
    if cost["status"] != "PASS":
        return {**cost, "metadata": protocol}
    allowed_diagnostic = (role == "baseline" and baseline_diagnostic
                          and checked_outputs.get("failure_kind") == "numerical_mismatch")
    if checked_outputs["status"] != "PASS" and not allowed_diagnostic:
        return {"status": "FAIL", "failure_kind": checked_outputs["failure_kind"],
                "reason": checked_outputs["reason"], "metadata": protocol}
    return {"status": "PASS", "execution_time_ms": execution_time_ms,
            "benchmark_method": timing["benchmark_method"],
            "metadata": {**protocol, "baseline_numerical_diagnostic": bool(allowed_diagnostic)}}
