"""Task-owned execution, original comparison callbacks and canonical GPU timing.

Every caller selects an explicit role. An absent candidate is never a baseline.
This module is copied into each task so isolated workspaces need no sibling task
or Arena Python imports. The benchmark helper is materialized by Arena.
"""
from __future__ import annotations

import secrets
import torch

import task_baseline
import task_compare
import task_contract
import task_inputs
import task_reference


def builder_axes(case):
    return {name: value for name, value in task_inputs.dimensions(case).items() if name != "one"}


def build_launch(builder, case):
    if not callable(builder):
        raise RuntimeError("Candidate builder is missing or not callable")
    launch = builder(**builder_axes(case))
    if not callable(launch):
        raise RuntimeError("Candidate builder did not return a callable launch")
    return launch


def case_call(inputs, *, role, launch=None):
    if role == "baseline":
        if launch is not None:
            raise ValueError("Baseline action cannot invoke a candidate")
        return lambda: task_baseline.run(**task_inputs.call_kwargs(inputs))
    if role != "candidate" or not callable(launch):
        raise RuntimeError("Candidate action requires its own launch; no baseline fallback")
    return lambda: launch(*task_inputs.launch_args(inputs))


def output_contract(got, expected):
    """Order both results as (output, lse) and check their tensor contracts.

    A missing output, a wrong shape/dtype/device, a nonfinite output, or a NaN
    or -inf LSE is an AssertionError (candidate contract). An invalid reference
    propagates as ValueError and stops the run.
    """
    wanted = task_inputs.named_outputs(expected)
    actual = task_inputs.named_outputs(got)
    task_compare.validate_comparison(wanted[0], wanted[0])
    try:
        task_compare.validate_comparison(actual[0], wanted[0])
    except AssertionError as error:
        raise AssertionError(f"output: {error}") from error
    lse, reference = actual[1], wanted[1]
    if (not isinstance(lse, torch.Tensor) or lse.layout != torch.strided or lse.shape != reference.shape
            or lse.dtype != reference.dtype or lse.device != reference.device):
        raise AssertionError("lse must match the reference shape, dtype and device")
    if bool(torch.isnan(lse).any()) or bool(torch.isneginf(lse).any()):
        raise AssertionError("lse contains NaN or -inf")
    return actual, wanted


def compare_output(got, expected):
    """Only completed comparisons can produce numerical_mismatch.

    Invalid references and runtime errors propagate. The original callback is
    the only authority on acceptance, after the tensor contract checks.
    """
    try:
        actual, wanted = output_contract(got, expected)
    except AssertionError as error:
        return {"status": "FAIL", "failure_kind": "output_contract", "reason": str(error)}
    passed, detail = task_inputs.verdict(actual, wanted)
    # Supplemental evidence never sets, replaces or rescales the callback gate.
    finite = torch.isfinite(wanted[1]) & torch.isfinite(actual[1])
    output_error = (actual[0].float() - wanted[0].float()).abs()
    lse_error = (actual[1] - wanted[1]).abs().masked_select(finite)
    metrics = {"output_max_absolute_error": output_error.max().item() if output_error.numel() else 0.0,
               "lse_max_absolute_error": lse_error.max().item() if lse_error.numel() else 0.0,
               "lse_infinity_mismatches": int((torch.isposinf(actual[1]) != torch.isposinf(wanted[1])).sum()),
               "compared_elements": wanted[0].numel() + wanted[1].numel()}
    result = {"status": "PASS" if passed else "FAIL", "metrics": metrics,
              "metadata": {"comparison": "scripts/task_compare.py:run",
                           "output_contract_passed": True, "comparison_detail": detail}}
    if not passed:
        result.update(failure_kind="numerical_mismatch", reason=detail)
    return result


def input_snapshot(inputs):
    # Byte views make the comparison exact for every dtype.
    return {key: value.view(torch.uint8).clone() for key, value in inputs.items()
            if isinstance(value, torch.Tensor)}


def assert_inputs_unchanged(inputs, snapshot):
    for key, expected in snapshot.items():
        if not torch.equal(inputs[key].view(torch.uint8), expected):
            raise RuntimeError(f"Operator modified protected input tensor: {key}")


def batch_state(case, reuse=None, device="cuda"):
    """The batch shape's buffers and base draw, built once per shape when reusing.

    Cases are grouped by shape, so one full initialization serves all of a
    batch size's cases, which visit growing lengths on the same storage.
    """
    if reuse is None:
        return task_inputs.BatchInputs(case, device=device)
    key = task_inputs.shape_key(case)
    if key not in reuse:
        reuse.clear()  # Release the previous shape before allocating the next.
        reuse[key] = task_inputs.BatchInputs(case, device=device)
    return reuse[key]


def invoke_checked(inputs, call):
    """One input-guarded invocation compared with the reference."""
    expected = task_reference.run(**task_inputs.call_kwargs(inputs))
    before = input_snapshot(inputs)
    got = call()
    torch.cuda.synchronize()
    assert_inputs_unchanged(inputs, before)
    return compare_output(got, expected)


# Correctness checks every case on the base draw and on CORRECTNESS_DRAWS
# further draws seeded from the operating system's entropy source.
CORRECTNESS_DRAWS = 2


def fresh_lengths(batch, widths, grid):
    """Per-row lengths per pool nobody can have tuned for: uniform or boundary-adjacent."""
    rng = secrets.SystemRandom()
    lengths = {}
    for pool, width in widths.items():
        values = []
        for _ in range(batch):
            value = rng.randint(0, width) if rng.randrange(2) else rng.choice(grid[pool]) + rng.randint(-2, 2)
            values.append(min(max(value, 0), width))
        lengths[pool] = values
    return lengths


def aggregate(results, evidence):
    failed = [result for result in results if result["status"] != "PASS"]
    metadata = {**evidence, "checked_invocations": len(results), "failed_invocations": len(failed)}
    if not failed:
        return {"status": "PASS", "metrics": results[0]["metrics"], "metadata": metadata}
    kinds = {result["failure_kind"] for result in failed}
    kind = "numerical_mismatch" if kinds == {"numerical_mismatch"} else sorted(
        kinds - {"numerical_mismatch"})[0]
    first = next(result for result in failed if result["failure_kind"] == kind)
    return {"status": "FAIL", "failure_kind": kind,
            "reason": f"{len(failed)}/{len(results)} checked invocations failed; first: {first['reason']}",
            "metadata": {**metadata, "first_failure": first}}


def check_case(case, *, role, launch=None, reuse=None):
    state = batch_state(case, reuse)
    inputs = state.prepare(case)
    call = case_call(inputs, role=role, launch=launch)
    evidence = {"input_validation": task_inputs.check_case_inputs(inputs, case, state)}
    seeds = fresh_draw_seeds(CORRECTNESS_DRAWS)
    evidence["data_draw_seeds"] = [task_inputs.SEED, *seeds]
    results = [invoke_checked(inputs, call)]
    for draw in state.draws(case, seeds):
        task_inputs.load_draw(inputs, draw)
        results.append(invoke_checked(inputs, call))
    if isinstance(case["lengths"], dict) and "cycle" in case["lengths"]:
        widths = {pool: task_inputs.AXES[task_contract.POOLS[pool]["width"]] for pool in task_inputs.POOL_NAMES}
        drawn = fresh_lengths(int(case["batch"]), widths, task_inputs.WORKLOAD["length_grid"])
        lengths = {pool: torch.tensor(values, dtype=torch.int32, device=state.device)
                   for pool, values in drawn.items()}
        state.prepare(case, lengths)
        evidence["fresh_lengths"] = {pool: {"min": min(v), "max": max(v), "distinct": len(set(v)),
                                            "first_rows": v[:16]} for pool, v in drawn.items()}
        results.append(invoke_checked(inputs, call))
    return aggregate(results, evidence)


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


def fresh_draw_seeds(count):
    """Distinct seeds from the operating system's entropy source, never SEED.

    The seeds are drawn when the case is checked or timed, so the draws cannot
    be known to the code being measured beforehand; fixed seeds would let an
    implementation generate the same draws itself and store their results.
    """
    rng = secrets.SystemRandom()
    seeds = []
    while len(seeds) < count:
        seed = rng.randrange(2**63)
        if seed != task_inputs.SEED and seed not in seeds:
            seeds.append(seed)
    return seeds


def choose_checked_samples(repetition, count):
    """Indices of the reported samples whose outputs are checked, chosen secretly."""
    return sorted(secrets.SystemRandom().sample(range(repetition), min(count, repetition)))


class RotatingDraws:
    """Preparation that loads a different call-varying draw before every replay.

    A replay executes recorded kernels, and those kernels may themselves decide
    at run time whether to compute: one that compares its operands against a
    copy of the last ones it saw and replays a stored output on a match skips
    the operator on every sample, because every sample reads the same bytes.
    Loading another draw before each replay makes consecutive samples differ in
    the queries, the main pool and both index tables, while keeping every
    buffer's storage, which is all a captured graph depends on.

    Supplying a preparation callback is also how the benchmark is told to
    capture a single logical invocation per replay rather than batching as many
    as fill ``target_ms`` and dividing by the count. ``time_case`` asserts the
    count it gets rather than trusting this.

    ``serve`` makes the next preparation load a given draw instead of the next
    one in the rotation. ``consumed`` is the draw the latest preparation loaded.
    """

    def __init__(self, inputs, draws):
        if len(draws) < 2:
            raise ValueError("rotation needs at least two distinct draws")
        self._inputs = inputs
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
        task_inputs.load_draw(self._inputs, draw)
        self.consumed = draw

    def serve(self, draw):
        self._served = draw


def host_copy(outputs):
    # A device-to-host copy reads the outputs without writing device memory, so
    # keeping them evicts little of what the next invocation finds in cache.
    # A launch may reuse its output buffers, so the copy is explicit.
    if isinstance(outputs, torch.Tensor):
        return outputs.detach().to("cpu", copy=True)
    if isinstance(outputs, (tuple, list)):
        return type(outputs)(host_copy(value) for value in outputs)
    if isinstance(outputs, dict):
        return {name: host_copy(value) for name, value in outputs.items()}
    return outputs


def to_device(outputs, device):
    if isinstance(outputs, torch.Tensor):
        return outputs.to(device)
    if isinstance(outputs, (tuple, list)):
        return type(outputs)(to_device(value, device) for value in outputs)
    if isinstance(outputs, dict):
        return {name: to_device(value, device) for name, value in outputs.items()}
    return outputs


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

    def __call__(self, outputs):
        if self._index in self._indices:
            self.kept.append((self._rotation.consumed, host_copy(outputs)))
        self._index += 1


def run_unseen_draws(timed, rotation, unseen):
    """Time the timed unit once over each unseen draw and keep what it wrote.

    Each invocation is prepared by the rotation and timed through the reported
    sample path, so it differs from a sample only in consuming a draw the
    implementation has never read.
    """
    unseen_ms, kept = [], []
    for draw in unseen:
        rotation.serve(draw)
        unseen_ms.append(timed.rerun_ms())
        kept.append((draw, host_copy(timed.outputs)))
    return unseen_ms, kept


def verify_timed_outputs(inputs, kept):
    """Compare every kept timed output with the reference on the draw it consumed.

    Both roles undergo exactly the same check.
    """
    expected_by_draw = {}
    results = []
    for draw, got in kept:
        if id(draw) not in expected_by_draw:
            task_inputs.load_draw(inputs, draw)
            expected_by_draw[id(draw)] = task_reference.run(**task_inputs.call_kwargs(inputs))
        expected = expected_by_draw[id(draw)]
        results.append(compare_output(to_device(got, expected[0].device), expected))
    failed = [result for result in results if result["status"] != "PASS"]
    metadata = {"checked_invocations": len(results), "failed_invocations": len(failed)}
    if not failed:
        return {"status": "PASS", "metadata": metadata}
    kinds = {result["failure_kind"] for result in failed}
    kind = "numerical_mismatch" if kinds == {"numerical_mismatch"} else sorted(
        kinds - {"numerical_mismatch"})[0]
    first = next(result for result in failed if result["failure_kind"] == kind)
    return {"status": "FAIL", "failure_kind": kind,
            "reason": (f"{len(failed)}/{len(results)} checked timed invocations failed; "
                       f"first: {first['reason']}"),
            "metadata": {**metadata, "first_failure": first}}


def verify_timed_cost(unseen_ms, execution_time_ms):
    """Hold the reported time to what the timed unit costs on an unseen draw.

    The fastest unseen-draw invocation is what the operator costs on new
    inputs; noise only makes an invocation slower, while every unseen draw is a
    miss for a stored result.
    """
    fastest = min(unseen_ms)
    bar = execution_time_ms * UNSEEN_DRAW_MARGIN
    metadata = {"unseen_draw_ms": fastest, "unseen_draw_samples_ms": list(unseen_ms),
                "reported_ms": execution_time_ms}
    if fastest > bar:
        return {"status": "FAIL", "failure_kind": "timing_input_memoized",
                "reason": (f"Timed invocation took {fastest:.6f} ms over a draw it had not "
                           f"seen, against a reported {execution_time_ms:.6f} ms per call and "
                           f"a bar of {bar:.6f} ms: the samples were served faster than the "
                           "operator runs on new inputs"),
                "metadata": metadata}
    return {"status": "PASS", "metadata": metadata}


def time_case(case, *, role, launch=None, baseline_diagnostic=False, reuse=None):
    from _aka_benchmark import TimedRun, benchmark_cuda_graph_or_events

    state = batch_state(case, reuse)
    inputs = state.prepare(case)
    input_validation = task_inputs.check_case_inputs(inputs, case, state)
    call = case_call(inputs, role=role, launch=launch)
    seeds = fresh_draw_seeds(TIMED_DRAWS + UNSEEN_DRAWS)
    timed_seeds, unseen_seeds = seeds[:TIMED_DRAWS], seeds[TIMED_DRAWS:]
    rotation = RotatingDraws(inputs, state.draws(case, timed_seeds))
    unseen = state.draws(case, unseen_seeds)
    weights = input_snapshot({name: inputs[name] for name in task_inputs.PERSISTENT_INPUTS})
    checked = choose_checked_samples(task_inputs.BENCH_REPETITION, CHECKED_SAMPLES)
    checks = SampleChecks(rotation, checked)
    timed = TimedRun()
    timed.after_sample = checks
    execution_time_ms, timing = benchmark_cuda_graph_or_events(
        call, warmup=task_inputs.BENCH_WARMUP,
        repetition=task_inputs.BENCH_REPETITION,
        target_ms=task_inputs.BENCH_TARGET_MS,
        prepare_fn=rotation, timed_run=timed)
    protocol = {"timing": timing, "timed_draw_seeds": timed_seeds,
                "unseen_draw_seeds": unseen_seeds, "checked_samples": checked,
                "input_validation": input_validation}
    repeats = timing.get("benchmark_effective_repeats")
    if repeats != 1:
        return {"status": "FAIL", "failure_kind": "timing_protocol",
                "reason": (f"Capture batched {repeats} invocations into one replay, so each "
                           "sample reports their average; this task requires one logical "
                           "invocation per replay"),
                "metadata": protocol}
    if not timed.bound:
        raise RuntimeError("Benchmark did not expose the invocation it timed")
    unseen_ms, unseen_kept = run_unseen_draws(timed, rotation, unseen)
    torch.cuda.synchronize()
    # The saved draw owns independent storage; load_draw copied it into the
    # live buffers. Compare those live buffers against the expected bytes. The
    # held pool, sinks and lengths are checked against their pre-timing snapshot.
    expected_inputs = {**weights, **input_snapshot(rotation.consumed)}
    assert_inputs_unchanged(inputs=inputs, snapshot=expected_inputs)
    cost = verify_timed_cost(unseen_ms, execution_time_ms)
    outputs = verify_timed_outputs(inputs, checks.kept + unseen_kept)
    protocol.update(unseen_draw_cost=cost["metadata"], timed_output_correctness=outputs)
    if cost["status"] != "PASS":
        return {**cost, "metadata": protocol}
    allowed_diagnostic = (role == "baseline" and baseline_diagnostic
                          and outputs.get("failure_kind") == "numerical_mismatch")
    if outputs["status"] != "PASS" and not allowed_diagnostic:
        return {"status": "FAIL", "failure_kind": outputs["failure_kind"],
                "reason": outputs["reason"], "metadata": protocol}
    return {"status": "PASS", "execution_time_ms": execution_time_ms,
            "benchmark_method": timing["benchmark_method"],
            "metadata": {**protocol, "baseline_numerical_diagnostic": bool(allowed_diagnostic)}}
