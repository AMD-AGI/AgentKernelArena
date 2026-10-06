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
    return {"batch": int(case["batch"]), "width": int(case["width"]), "pages": int(case["pages"]),
            "k": task_inputs.K, "page_size": task_inputs.PAGE_SIZE}


def build_launch(builder, case):
    if not callable(builder):
        raise RuntimeError("Candidate builder is missing or not callable")
    launch = builder(**builder_axes(case))
    if not callable(launch):
        raise RuntimeError("Candidate builder did not return a callable launch")
    return launch


def case_call(inputs, out, *, role, launch=None):
    """The complete operator invocation, writing ``out`` and returning it."""
    if role == "baseline":
        if launch is not None:
            raise ValueError("Baseline action cannot invoke a candidate")

        def baseline():
            task_baseline.run(**task_inputs.baseline_kwargs(inputs, out))
            return out
        return baseline
    if role != "candidate" or not callable(launch):
        raise RuntimeError("Candidate action requires its own launch; no baseline fallback")

    def candidate():
        if launch(*task_inputs.launch_args(inputs, out)) is not None:
            raise RuntimeError("A destination-passing launch writes out_page_indices and returns None")
        return out
    return candidate


def compare_output(got, expected):
    """Only completed comparisons can produce numerical_mismatch.

    Invalid references and runtime errors propagate. The original callback is
    the only authority on acceptance, after shape/dtype/device checks.
    """
    try:
        task_compare.validate_comparison(got, expected)
    except AssertionError as error:
        return {"status": "FAIL", "failure_kind": "output_contract", "reason": str(error)}
    passed, detail = task_inputs.verdict(got, expected)
    # Supplemental evidence never sets or replaces the callback rule.
    rows = (got.sort(dim=-1).values != expected.sort(dim=-1).values).any(dim=-1)
    metrics = {"compared_elements": expected.numel(), "rows": expected.shape[0],
               "rows_with_different_sets": int(rows.sum().item()),
               "poisoned_elements": int((got == task_inputs.OUTPUT_POISON).sum().item())}
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


def invoke_checked(inputs, out, call):
    """One poisoned, input-guarded invocation compared with the reference."""
    expected = task_reference.run(**task_inputs.call_kwargs(inputs))
    before = input_snapshot(inputs)
    task_inputs.poison_output(out)
    call()
    torch.cuda.synchronize()
    assert_inputs_unchanged(inputs, before)
    return compare_output(out, expected)


# Correctness checks every case on the bundle seed and on CORRECTNESS_DRAWS
# further data draws from the operating system's entropy source.
CORRECTNESS_DRAWS = 2


def case_buffers(case, reuse=None, device="cuda"):
    """The case's buffers, reusing one set per shape when a reuse map is given.

    Correctness visits each batch's cases in manifest order, from the bundle
    lengths through growing uniform lengths to mixed rows, on the same storage.
    A launch that keeps a stale length or plan from an earlier call on the same
    buffers then produces a wrong selection rather than going unnoticed.
    """
    key = tuple(sorted(builder_axes(case).items()))
    if reuse is None:
        inputs = task_inputs.build_case_inputs(case, device=device)
        return inputs, task_inputs.allocate_output(inputs)
    if key not in reuse:
        reuse.clear()  # Cases are grouped by shape; release the previous one.
        inputs = task_inputs.build_case_inputs(case, device=device)
        reuse[key] = (inputs, task_inputs.allocate_output(inputs))
    else:
        task_inputs.initialize_case(reuse[key][0], case)
    return reuse[key]


def fresh_lengths(batch, limit, boundary):
    """Per-row lengths nobody can have tuned for: uniform, log-uniform and boundary-adjacent."""
    rng = secrets.SystemRandom()
    lengths = []
    for _ in range(batch):
        kind = rng.randrange(3)
        if kind == 0:
            value = rng.randint(0, limit)
        elif kind == 1:
            value = int(round(limit ** rng.random()))
        else:
            value = rng.choice(boundary) + rng.randint(-2, 2)
        lengths.append(min(max(value, 0), limit))
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
    inputs, out = case_buffers(case, reuse)
    call = case_call(inputs, out, role=role, launch=launch)
    evidence = {"input_validation": task_inputs.check_case_inputs(inputs, case)}
    seeds = [task_inputs.SEED, *fresh_draw_seeds(CORRECTNESS_DRAWS)]
    evidence["data_draw_seeds"] = seeds
    results = []
    for index, seed in enumerate(seeds):
        if index:
            task_inputs.redraw_call_varying_inputs(inputs, seed=seed)
        results.append(invoke_checked(inputs, out, call))
    if isinstance(case["lengths"], dict) and "cycle" in case["lengths"]:
        declared = inputs["seq_lens"].clone()
        lengths = fresh_lengths(int(case["batch"]), task_contract.capacity(case, task_inputs.AXES),
                                task_inputs.WORKLOAD["boundary_lengths"])
        inputs["seq_lens"].copy_(torch.tensor(lengths, dtype=torch.int32))
        if not task_inputs.plan_matches(inputs["seq_lens"], inputs["metadata"]):
            raise RuntimeError("metadata is not a matching v2 plan for the fresh lengths")
        evidence["fresh_lengths"] = {"min": min(lengths), "max": max(lengths),
                                     "distinct": len(set(lengths)), "first_rows": lengths[:16]}
        results.append(invoke_checked(inputs, out, call))
        inputs["seq_lens"].copy_(declared)
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
    value while keeping every buffer's storage, which is all a captured graph
    depends on. The case's lengths and plan stay fixed across samples.

    ``before_call`` runs after the draw is loaded; it poisons the destination
    so that every sample has to write all of its output itself.

    Supplying a preparation callback is also how the benchmark is told to
    capture a single logical invocation per replay rather than batching as many
    as fill ``target_ms`` and dividing by the count. ``time_case`` asserts the
    count it gets rather than trusting this.

    ``serve`` makes the next preparation load a given draw instead of the next
    one in the rotation. ``consumed`` is the draw the latest preparation loaded.
    """

    def __init__(self, inputs, draws, before_call):
        if len(draws) < 2:
            raise ValueError("rotation needs at least two distinct draws")
        self._inputs = inputs
        self._draws = draws
        self._before_call = before_call
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
        self._before_call()
        self.consumed = draw

    def serve(self, draw):
        self._served = draw


def host_copy(outputs):
    # A device-to-host copy reads the outputs without writing device memory, so
    # keeping them evicts little of what the next invocation finds in cache.
    # The destination is reused by every invocation, so the copy is explicit.
    return outputs.detach().to("cpu", copy=True) if isinstance(outputs, torch.Tensor) else outputs


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
        if isinstance(got, torch.Tensor):
            got = got.to(expected.device)
        results.append(compare_output(got, expected))
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


def time_case(case, *, role, launch=None, baseline_diagnostic=False):
    from _aka_benchmark import TimedRun, benchmark_cuda_graph_or_events

    inputs = task_inputs.build_case_inputs(case)
    input_validation = task_inputs.check_case_inputs(inputs, case)
    out = task_inputs.allocate_output(inputs)
    call = case_call(inputs, out, role=role, launch=launch)
    seeds = fresh_draw_seeds(TIMED_DRAWS + UNSEEN_DRAWS)
    timed_seeds, unseen_seeds = seeds[:TIMED_DRAWS], seeds[TIMED_DRAWS:]
    rotation = RotatingDraws(inputs, task_inputs.call_varying_draws(inputs, timed_seeds),
                             before_call=lambda: task_inputs.poison_output(out))
    unseen = task_inputs.call_varying_draws(inputs, unseen_seeds)
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
    # case's lengths and plan are checked against their pre-timing snapshot.
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
