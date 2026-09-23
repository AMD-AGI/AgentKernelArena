"""Task-owned execution, original comparison callbacks and canonical GPU timing.

Every caller selects an explicit role. An absent candidate is never a baseline.
This module is copied into each task so isolated workspaces need no sibling task
or Arena Python imports. The benchmark helper is materialized by Arena.
"""
from __future__ import annotations

import math
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

    ``hold`` makes the next preparation leave the buffers alone, for a caller
    that has loaded the draw it wants the next replay to consume.
    """

    def __init__(self, inputs, draws):
        if len(draws) < 2:
            raise ValueError("rotation needs at least two distinct draws")
        self._inputs = inputs
        self._draws = draws
        self._next = 0
        self._held = False

    def __call__(self):
        if self._held:
            self._held = False
            return
        task_inputs.load_draw(self._inputs, self._draws[self._next])
        self._next = (self._next + 1) % len(self._draws)

    def hold(self):
        self._held = True


def verify_timed_invocation(inputs, timed, rotation):
    """Check the actual measured graph over redrawn inputs under the full gate.

    Reference creation, poison, the call-varying redraw and comparisons occur
    after measurement. Both roles undergo exactly the same check. The runner
    alone can retain a baseline's completed numerical mismatch as explicitly
    declared diagnostics.

    Only the call-varying operands are redrawn, holding the weights the samples
    held, so a kernel that packs its weights once is not rejected for reusing
    that packing. The rotation is held for the single re-arm replay so it
    consumes the inputs redrawn here rather than the next rotation draw.
    """
    if not timed.bound:
        raise RuntimeError("Benchmark did not expose the invocation it timed")
    previous = timed.outputs.detach().clone() if isinstance(timed.outputs, torch.Tensor) else None
    task_inputs.redraw_call_varying_inputs(inputs)
    expected = task_reference.run(**task_inputs.call_kwargs(inputs))
    before = input_snapshot(inputs)
    if isinstance(timed.outputs, torch.Tensor):
        timed.outputs.fill_(float("nan"))
    rotation.hold()
    got = timed.rerun()
    torch.cuda.synchronize()
    assert_inputs_unchanged(inputs, before)
    result = compare_output(got, expected)
    if previous is not None and isinstance(got, torch.Tensor) and torch.equal(got, previous):
        raise RuntimeError("Timed invocation returned its cached output over a fresh draw")
    result.setdefault("metadata", {}).update(replay_checked=True,
                                             refill_seed=task_inputs.REFILL_SEED)
    return result


def verify_timed_cost(inputs, timed, rotation, unseen, execution_time_ms):
    """Hold the reported time to what the timed unit costs on an unseen draw.

    The rotation defeats a stored result that remembers one draw, not one that
    remembers every draw the samples cycle through. Such an implementation still
    has to compute the first time it meets a draw, so the timed unit is replayed
    once over each of a few draws that were never loaded before, timed the way a
    sample is, and the fastest of those replays is compared with the reported
    mean. The fastest is used because noise only makes a replay slower, while
    every unseen draw is a miss for a stored result.

    This runs directly after the samples, over draws made before them. Anything
    heavier than loading a draw in between -- redrawing through the bundle
    rewrites the weight -- evicts operands the samples found in cache, and the
    replay would then be slower for reasons unrelated to what it computed.
    """
    unseen_ms = []
    for draw in unseen:
        task_inputs.load_draw(inputs, draw)
        rotation.hold()
        unseen_ms.append(timed.rerun_ms())
    torch.cuda.synchronize()
    fastest = min(unseen_ms)
    bar = execution_time_ms * task_inputs.UNSEEN_DRAW_MARGIN
    metadata = {"unseen_draw_ms": fastest, "reported_ms": execution_time_ms}
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
    rotation = RotatingDraws(
        inputs, task_inputs.call_varying_draws(inputs, task_inputs.TIMED_DRAW_SEEDS))
    unseen = task_inputs.call_varying_draws(inputs, task_inputs.UNSEEN_DRAW_SEEDS)
    timed = TimedRun()
    execution_time_ms, timing = benchmark_cuda_graph_or_events(
        call, warmup=task_inputs.BENCH_WARMUP,
        repetition=task_inputs.BENCH_REPETITION,
        target_ms=task_inputs.BENCH_TARGET_MS,
        prepare_fn=rotation, timed_run=timed)
    repeats = timing.get("benchmark_effective_repeats")
    if repeats != 1:
        return {"status": "FAIL", "failure_kind": "timing_protocol",
                "reason": (f"Capture batched {repeats} invocations into one replay, so each "
                           "sample reports their average; this task requires one logical "
                           "invocation per replay"),
                "metadata": {"timing": timing}}
    cost = verify_timed_cost(inputs, timed, rotation, unseen, execution_time_ms)
    if cost["status"] != "PASS":
        return {**cost, "metadata": {**cost.get("metadata", {}), "timing": timing}}
    replay = verify_timed_invocation(inputs, timed, rotation)
    allowed_diagnostic = (role == "baseline" and baseline_diagnostic
                          and replay.get("failure_kind") == "numerical_mismatch")
    if replay["status"] != "PASS" and not allowed_diagnostic:
        return {**replay, "metadata": {**replay.get("metadata", {}), "timing": timing}}
    return {"status": "PASS", "execution_time_ms": execution_time_ms,
            "benchmark_method": timing["benchmark_method"],
            "metadata": {"timing": timing, "replay_correctness": replay,
                         "unseen_draw_cost": cost.get("metadata", {}),
                         "baseline_numerical_diagnostic": bool(allowed_diagnostic)}}
