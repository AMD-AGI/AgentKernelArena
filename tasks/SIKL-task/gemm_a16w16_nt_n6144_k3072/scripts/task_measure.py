"""Task-owned execution, original comparison callbacks and canonical GPU timing.

Every caller selects an explicit role. An absent candidate is never a baseline.
This module is copied into each task so isolated workspaces need no sibling task
or Arena Python imports. The benchmark helper is materialized by Arena.
"""
from __future__ import annotations

import math
import secrets
import torch

import task_baseline
import task_compare
import task_inputs
import task_reference


def build_launch(builder, case):
    if not callable(builder):
        raise RuntimeError("Candidate builder is missing or not callable")
    if task_inputs.WORKLOAD["op_type"] == "gemm":
        launch = builder(m=int(case["m"]), n=task_inputs.N, k=task_inputs.K)
    else:
        launch = builder(num_tokens=int(case["num_tokens"]),
                         model_dim=task_inputs.MODEL_DIM, inter_dim=task_inputs.INTER_DIM,
                         num_experts=task_inputs.NUM_EXPERTS, topk=task_inputs.TOPK)
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
    if task_inputs.WORKLOAD["op_type"] == "gemm":
        return lambda: launch(inputs["a"], inputs["b"])
    return lambda: launch(inputs["hidden_states"], inputs["w1"], inputs["w2"],
                          inputs["topk_weights"], inputs["topk_ids"],
                          inputs["w1_scale"], inputs["w2_scale"],
                          inputs["activation"], inputs["doweight_stage1"])


def compare_output(got, expected):
    """Only completed numerical comparisons can produce numerical_mismatch.

    Invalid references and runtime errors propagate. The original callback is
    the only authority on numerical acceptance, after shape/dtype/finite checks.
    """
    try:
        task_compare.validate_comparison(got, expected)
    except AssertionError as error:
        return {"status": "FAIL", "failure_kind": "output_contract", "reason": str(error)}
    passed, detail = task_inputs.verdict(got, expected)
    # Supplemental evidence never sets, replaces or rescales the callback gate.
    delta = (got.float() - expected.float()).abs()
    metrics = {"max_absolute_error": delta.max().item() if delta.numel() else 0.0,
               "compared_elements": expected.numel()}
    result = {"status": "PASS" if passed else "FAIL", "metrics": metrics,
              "metadata": {"comparison": "scripts/task_compare.py:run",
                           "output_contract_passed": True, "comparison_detail": detail}}
    if task_inputs.WORKLOAD["op_type"] == "moe":
        # The original helper uses the output dtype, as does the comparator.
        # Exact pairs have +inf SQNR (or NaN for identical zero tensors).
        sqnr = task_compare.compute_error(expected, got).item()
        metrics["sqnr_db"] = sqnr if math.isfinite(sqnr) else None
        if not math.isfinite(sqnr):
            result["metadata"]["sqnr_db_nonfinite"] = (
                "exact match; infinite SQNR" if torch.equal(got, expected)
                else f"original SQNR helper returned {sqnr}; callback verdict retained")
    if not passed:
        result.update(failure_kind="numerical_mismatch", reason=detail)
    return result


def input_snapshot(inputs):
    # Byte views support packed float4 tensors, whose equal/clone ops may not.
    return {key: value.view(torch.uint8).clone() for key, value in inputs.items()
            if isinstance(value, torch.Tensor)}


def assert_inputs_unchanged(inputs, snapshot):
    for key, expected in snapshot.items():
        if not torch.equal(inputs[key].view(torch.uint8), expected):
            raise RuntimeError(f"Operator modified protected input tensor: {key}")


def check_case(case, *, role, launch=None):
    inputs = task_inputs.build_case_inputs(case)
    expected = task_reference.run(**task_inputs.call_kwargs(inputs))
    before = input_snapshot(inputs)
    got = case_call(inputs, role=role, launch=launch)()
    torch.cuda.synchronize()
    assert_inputs_unchanged(inputs, before)
    return compare_output(got, expected)


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

    The seeds are drawn when the case is timed, so neither the draws a sample
    reads nor the ones held out for the cost check can be known to the code
    being measured beforehand; fixed seeds would let an implementation generate
    the same draws itself and store their results.
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
    depends on. The draws come from ``task_inputs.call_varying_draws``, so the
    weights stay fixed across samples the way a deployment holds them.

    Supplying a preparation callback is also how the benchmark is told to
    capture a single logical invocation per replay rather than batching as many
    as fill ``target_ms`` and dividing by the count. ``time_case`` asserts the
    count it gets rather than trusting this.

    ``serve`` makes the next preparation load a given draw instead of the next
    one in the rotation, so an invocation over a draw of the caller's choosing
    goes through the same preparation, on the same stream, as every sample.
    ``consumed`` is the draw the latest preparation loaded.
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
    return outputs.detach().to("cpu") if isinstance(outputs, torch.Tensor) else outputs


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
    implementation has never read. This runs directly after the samples, with
    nothing heavier than loading a draw and reading back the previous outputs in
    between; redrawing through the bundle here would rewrite the weights, evict
    what the samples found in cache and slow these invocations for reasons
    unrelated to what they compute.
    """
    unseen_ms, kept = [], []
    for draw in unseen:
        rotation.serve(draw)
        unseen_ms.append(timed.rerun_ms())
        kept.append((draw, host_copy(timed.outputs)))
    return unseen_ms, kept


def verify_timed_outputs(inputs, kept):
    """Compare every kept timed output with the reference on the draw it consumed.

    The kept outputs are those of the checked samples and of every unseen-draw
    invocation. An unseen draw is new to the implementation, so it has to be
    computed to be right; a checked sample was chosen without the implementation
    being able to tell. Both roles undergo exactly the same check, and the runner
    alone can retain a baseline's completed numerical mismatch as an explicitly
    declared diagnostic.
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

    The rotation defeats a stored result that remembers one draw, not one that
    remembers every draw the samples cycle through. Such an implementation still
    has to compute the first time it meets a draw, and its output on an unseen
    draw is checked, so the fastest unseen-draw invocation is what the operator
    costs on new inputs. The fastest is used because noise only makes an
    invocation slower, while every unseen draw is a miss for a stored result.
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
    call = case_call(inputs, role=role, launch=launch)
    seeds = fresh_draw_seeds(TIMED_DRAWS + UNSEEN_DRAWS)
    timed_seeds, unseen_seeds = seeds[:TIMED_DRAWS], seeds[TIMED_DRAWS:]
    rotation = RotatingDraws(inputs, task_inputs.call_varying_draws(inputs, timed_seeds))
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
                "unseen_draw_seeds": unseen_seeds, "checked_samples": checked}
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
    assert_inputs_unchanged(inputs, {**weights, **input_snapshot(rotation.consumed)})
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
