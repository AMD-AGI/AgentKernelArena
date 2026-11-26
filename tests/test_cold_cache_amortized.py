"""Amortized cold-cache timing: call count, subtraction, draws, poison and per-call output checks."""
from __future__ import annotations

from contextlib import nullcontext
import functools
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest
import torch

from src.tools.arena_eval import evaluate, integrity, values

HELPER = Path(__file__).resolve().parents[1] / "src/tools/perf/aka_benchmark.py"
HELPER_MODULE = "src.tools.arena_eval._aka_benchmark"
# Device time of one replay of each kind of graph in the stand-in.
SPANS = {"single": 0.02, "operator": 0.03, "reference": 0.02}
PER_CALL = SPANS["operator"] - SPANS["reference"]


@pytest.fixture
def helper(monkeypatch):
    """The real helper over a CPU stand-in for torch.cuda whose graphs replay eagerly.

    A captured graph runs its Python once at capture, as real capture does, and
    re-runs it on every replay. Each replay advances a device clock by its
    graph's span, and an Event pair reports the clock difference, so a sample's
    arithmetic is exact.
    """
    trace, clock = [], {"now": 0.0}

    class Event:
        def __init__(self, **kwargs):
            self.mark = None

        def record(self, stream=None):
            self.mark = clock["now"]

        def synchronize(self):
            pass

        def elapsed_time(self, end):
            return end.mark - self.mark

    cuda = types.SimpleNamespace(
        is_available=lambda: True, current_device=lambda: 0, Event=Event, synchronize=lambda: None,
        Stream=lambda: types.SimpleNamespace(wait_stream=lambda s: None, synchronize=lambda: None),
        current_stream=lambda: object(), stream=lambda s: nullcontext(), graph_pool_handle=lambda: "pool",
    )
    fake = types.ModuleType("torch")
    fake.__getattr__ = lambda name: getattr(torch, name)
    fake.cuda, fake.device, fake.int32 = cuda, torch.device, torch.int32
    fake.zeros = lambda shape, dtype, device: torch.zeros(shape, dtype=dtype)
    fake.empty = lambda shape, dtype, device: torch.empty(shape, dtype=dtype)

    def sum_(value, **kwargs):
        trace.append("flush")
        return torch.sum(value, **kwargs)

    fake.sum = sum_
    monkeypatch.setitem(sys.modules, "torch", fake)
    spec = importlib.util.spec_from_file_location("_amortized_helper", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def capture(fn, repeats, stream, prepare_fn=None, output_holder=None, pool=None):
        # The single-call cold-cache method always passes its preparation; the
        # amortized operator graphs collect their outputs, the reference graphs not.
        kind = "single" if prepare_fn is not None else "operator" if output_holder is not None else "reference"
        if kind == "operator":
            assert pool == "pool"
        if prepare_fn is not None:
            prepare_fn()
        trace.append(f"capture-{kind}")
        for _ in range(repeats):
            output = fn()
        if output_holder is not None:
            output_holder[:] = [output]

        def replay():
            trace.append(f"replay-{kind}")
            clock["now"] += SPANS[kind]
            for _ in range(repeats):
                fn()

        return types.SimpleNamespace(replay=replay)

    monkeypatch.setattr(module, "_capture_graph", capture)
    module.trace = trace
    monkeypatch.delenv("AKA_BENCHMARK_FORCE_EVENT", raising=False)
    monkeypatch.setitem(sys.modules, HELPER_MODULE, module)
    return module


# --------------------------------------------------------------------------- helper


@pytest.mark.parametrize("per_call_bytes, expected", [
    (1, 32), (1 << 20, 32), (128 << 20, 32), (160 << 20, 25), (256 << 20, 16), (1 << 30, 4), (64 << 30, 4)])
def test_calls_per_sample_follow_the_memory_of_one_call_between_four_and_thirty_two(
        helper, per_call_bytes, expected):
    assert helper.amortized_calls_per_replay(per_call_bytes) == expected


def test_invalid_call_memory_is_rejected(helper):
    for invalid in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="per_call_bytes"):
            helper.amortized_calls_per_replay(invalid)


def test_each_call_is_its_own_graph_and_samples_subtract_the_reference_group(helper):
    flush = helper.ReadOnlyCacheFlush(16384)
    observed = []
    timed = helper.TimedRun()

    def after(outputs):
        # The operator group's results are in place whenever a sample is observed.
        observed.append([name for name in helper.trace if name.startswith("replay")][-1])

    timed.after_sample = after
    count = 4

    def operation(index):
        helper.trace.append(f"kernel{index}")
        return f"out{index}"

    samples, metadata = helper.benchmark_cuda_graph_cold_cache_amortized_samples(
        [functools.partial(operation, index) for index in range(count)], cache_flush=flush,
        warmup=2, repetition=4, prepare_fn=lambda i: helper.trace.append(f"prepare{i}"), timed_run=timed)
    assert samples == pytest.approx([PER_CALL] * 4)
    trace = helper.trace
    # Eager warm-up: preparation and the call, cycling through the calls.
    assert trace[:4] == ["prepare0", "kernel0", "prepare1", "kernel1"]
    # One operator graph per call (prepare, eviction, call), then one reference graph per call.
    captures = [i for i, name in enumerate(trace) if name.startswith("capture")]
    assert [trace[i] for i in captures] == ["capture-operator"] * count + ["capture-reference"] * count
    for index, position in enumerate(captures[:count]):
        assert trace[position + 1:position + 4] == [f"prepare{index}", "flush", f"kernel{index}"]
    for index, position in enumerate(captures[count:]):
        assert trace[position + 1:position + 3] == [f"prepare{index}", "flush"]
    # Every group replays all its graphs (each re-running its preparation and eviction here)
    # and then one trailing eviction.
    replays = [name.removeprefix("replay-") for name in trace if name.startswith("replay")]
    groups = [replays[i] for i in range(0, len(replays), count)]
    assert all(len(set(replays[i:i + count])) == 1 for i in range(0, len(replays), count))
    after_captures = trace[captures[-1] + 3:]
    assert after_captures.count("flush") == len(groups) * (count + 1)
    # Two priming pairs, alternating order per sample, then the final reference control.
    assert groups == (["reference", "operator"] * 2 + ["reference", "operator", "operator", "reference"] * 2
                      + ["reference"])
    assert observed == ["replay-operator"] * 4
    assert metadata["benchmark_method"] == "cuda_graph_cold_cache_amortized"
    assert metadata["benchmark_calls_per_replay"] == count
    assert metadata["benchmark_graphs"] == "one_per_call" and metadata["benchmark_trailing_eviction"] is True
    assert metadata["benchmark_cache_flush_bytes"] == 16384
    assert metadata["benchmark_operator_graph_ms"] == pytest.approx([count * SPANS["operator"]] * 4)
    assert metadata["benchmark_reference_graph_ms"] == pytest.approx([count * SPANS["reference"]] * 4)
    assert timed.outputs == [f"out{index}" for index in range(count)]
    assert timed.rerun_ms() == pytest.approx(PER_CALL)


def test_warmup_can_run_on_inputs_that_are_never_timed(helper):
    warmed = []
    helper.benchmark_cuda_graph_cold_cache_amortized_samples(
        [lambda: helper.trace.append("kernel")] * 2, cache_flush=helper.ReadOnlyCacheFlush(16384),
        prepare_fn=lambda i: helper.trace.append(f"prepare{i}"), warmup_fn=warmed.append, warmup=3, repetition=2)
    assert warmed == [0, 1, 2]
    assert helper.trace[0] == "capture-operator"  # no timed call before capture


def test_amortized_timing_fails_closed(helper, monkeypatch):
    flush = helper.ReadOnlyCacheFlush(16384)
    calls = [lambda: None] * 4
    run = functools.partial(helper.benchmark_cuda_graph_cold_cache_amortized_samples,
                            cache_flush=flush, warmup=0, repetition=2)
    for invalid in ([], [None], [lambda: None, 3]):
        with pytest.raises(ValueError, match="one callable per call"):
            run(invalid)
    with pytest.raises(TypeError, match="ReadOnlyCacheFlush"):
        helper.benchmark_cuda_graph_cold_cache_amortized_samples(calls, cache_flush=object())
    timed = helper.TimedRun()
    timed.before_sample = lambda count: None
    with pytest.raises(ValueError, match="before cache eviction"):
        run(calls, timed_run=timed)
    monkeypatch.setenv("AKA_BENCHMARK_FORCE_EVENT", "1")
    with pytest.raises(RuntimeError, match="Event timing is forbidden"):
        run(calls)
    monkeypatch.delenv("AKA_BENCHMARK_FORCE_EVENT")

    def fail(*args, **kwargs):
        # What capture reports when a call leaves a forked stream unjoined at the end of its graph.
        raise RuntimeError("operation not permitted when stream is capturing (unjoined)")

    with monkeypatch.context() as patch:
        patch.setattr(helper, "_capture_graph", fail)
        with pytest.raises(RuntimeError, match="capture failed.*unjoined"):
            run(calls)
    # An operator group no slower than its reference group is an empty measurement, never a fallback.
    monkeypatch.setattr(helper, "_amortized_group_ms", lambda graphs, stream, flush: 0.5)
    with pytest.raises(RuntimeError, match="empty or invalid"):
        run(calls)


# ------------------------------------------------------------------------ evaluator


def _contract(mutable=()):
    checked = ("y", "idx") + tuple(mutable)
    return types.SimpleNamespace(mutable=tuple(mutable), checked_tensors=checked, element_order="fixed",
                                 compare={name: {"bound": "exact"} for name in checked},
                                 timing={"warmup": 2, "repetition": 4})


def _inputs(seed, mutable):
    x = torch.randn(64, generator=torch.Generator().manual_seed(seed))
    kernel = {"x": x}
    reference = {"y": 2 * x, "idx": (x > 0).to(torch.int32)}
    if mutable:
        kernel["state"] = x + 1
        reference["state"] = kernel["state"] + 1
    return kernel, reference


def _data(mutable=False):
    """The two correctness draws (warm-up and single-call samples)."""
    drawn = [_inputs(seed, mutable) for seed in (0, 1)]
    return types.SimpleNamespace(inputs=[types.SimpleNamespace(kernel=k) for k, _ in drawn],
                                 references=[r for _, r in drawn], baselines=None)


@pytest.fixture
def draws(monkeypatch):
    """Amortized draws from the evaluator's seeds, without a task contract."""
    seen = []

    def draw(contract, case, index, device):
        seed = evaluate.draw_seed(case, index)
        seen.append(seed)
        return _inputs(seed, bool(contract.mutable))

    monkeypatch.setattr(evaluate, "_draw", draw)
    return seen


CASE = types.SimpleNamespace(seeds=(11, 22), case_id="c0")


def _honest(x, y, idx, state=None):
    y.copy_(2 * x)
    idx.copy_((x > 0).to(torch.int32))
    if state is not None:
        state.add_(1)


def _amortized(helper, call, *, role="candidate", mutable=False, check=None, data=None, baseline_call=None):
    contract = _contract(("state",) if mutable else ())
    data = data or _data(mutable=mutable)
    working = values.clone_inputs(data.inputs[0].kernel)
    outputs = {"y": torch.empty(64), "idx": torch.empty(64, dtype=torch.int32)}
    return evaluate._amortized_cold(contract, role, call, CASE, data, working, outputs,
                                    helper.ReadOnlyCacheFlush(16384), check, "cpu", baseline_call)


def test_every_call_has_its_own_draw_and_is_checked_after_every_sample(helper, draws):
    samples, metadata, failed = _amortized(helper, _honest, mutable=True)
    assert failed == []
    assert samples == pytest.approx([PER_CALL] * 4)
    assert len(draws) == len(set(draws)) == 32 and not set(draws) & set(CASE.seeds)
    assert metadata["benchmark_calls_per_replay"] == 32
    assert metadata["checked_samples"] == 4 and metadata["checked_calls"] == 4 * 32
    # Each call is fully checked once, then by bitwise equality; the control checks every call again.
    assert metadata["full_checks"] == 32 + 32
    assert metadata["reference_control_undetected_calls"] == 0
    # The draw (x, state) twice (stored and staged), its reference (y, idx, state), 3 x the checked
    # bytes and an int32 poison pattern.
    assert metadata["benchmark_call_buffer_bytes"] == 2 * 2 * 256 + 3 * 256 + 3 * 3 * 256 + 256
    assert metadata["benchmark_draw_assignment"] == "secret_permutation_per_sample"


def test_a_result_that_changes_after_its_first_check_is_checked_fully_again(helper, draws):
    calls = {"count": 0}

    def degrades(x, y, idx):
        # Correct until the first sample's operator group (2 warm-ups, 32 captures, 2 x 32 primings,
        # then 32 calls) has run, wrong afterwards: the bitwise comparison sends it to a full check.
        calls["count"] += 1
        _honest(x, y, idx)
        if calls["count"] > 2 + 32 + 2 * 32 + 32:
            y.add_(1.0)

    _, _, failed = _amortized(helper, degrades)
    assert failed and failed[0]["sample"] == 2 and failed[0]["outputs"]["y"]["passed"] is False


def test_a_candidate_that_copies_an_earlier_calls_result_fails(helper, draws):
    """The review's memoization cheat: call i reuses call i-2's output instead of computing."""
    history = []

    def memoizing(x, y, idx):
        if len(history) >= 2 + 2:  # after the warm-up, from the third captured call on
            y.copy_(history[-2][0])
            idx.copy_(history[-2][1])
        else:
            _honest(x, y, idx)
        history.append((y, idx))

    _, _, failed = _amortized(helper, memoizing)
    # Every draw differs, so the copied result fails its own call's check.
    assert failed and failed[0]["sample"] == 1 and all("call" in failure for failure in failed)


def test_a_candidate_that_works_only_on_the_last_captured_call_fails(helper, draws):
    calls = {"count": 0}

    def last_call_only(x, y, idx):
        # Python runs once per captured call; after the two warm-ups it computes only on the 32nd.
        calls["count"] += 1
        if calls["count"] <= 2 or calls["count"] == 2 + 32:
            _honest(x, y, idx)

    _, _, failed = _amortized(helper, last_call_only)
    assert [failure["call"] for failure in failed] == [0, 1, 2]


def test_a_candidate_that_restores_a_stored_result_in_later_samples_fails(helper, draws):
    """The review's stash cheat: each captured call computes in its graph's first replay, keeps the
    result in state of its own, and only restores it in later replays. It needs no input
    inspection, so it passed while every graph saw the same draw in every sample."""
    seen = {"calls": 0}
    stash = {}

    def stashing(x, y, idx):
        seen["calls"] += 1
        if seen["calls"] <= 2:          # warm-up
            _honest(x, y, idx)
            return
        graph = (seen["calls"] - 3) % 32  # which captured call (graph) this is
        if graph in stash:
            y.copy_(stash[graph][0])
            idx.copy_(stash[graph][1])
        else:
            _honest(x, y, idx)
            stash[graph] = (y.clone(), idx.clone())

    _, _, failed = _amortized(helper, stashing)
    assert failed and failed[0]["sample"] == 1


def test_draws_are_reassigned_to_the_calls_by_a_fresh_secret_permutation(helper, draws, monkeypatch):
    orders = []
    original = evaluate._secret_permutation

    def recorded(count):
        orders.append(original(count))
        return orders[-1]

    monkeypatch.setattr(evaluate, "_secret_permutation", recorded)
    _, metadata, failed = _amortized(helper, _honest)
    assert failed == []
    # Two priming stages, then one per sample.
    assert len(orders) == 2 + 4 and all(sorted(order) == list(range(32)) for order in orders)
    assert len({tuple(order) for order in orders}) == len(orders)


def test_the_secret_permutation_is_uniform_and_uses_the_os_entropy(monkeypatch):
    counts = [[0] * 4 for _ in range(4)]
    for _ in range(4000):
        for position, value in enumerate(evaluate._secret_permutation(4)):
            counts[position][value] += 1
    assert all(800 < count < 1200 for row in counts for count in row)
    monkeypatch.setattr(evaluate.os, "urandom", lambda n: bytes(n))
    assert evaluate._secret_permutation(5) == [1, 2, 3, 4, 0]


def test_checks_that_accept_calls_without_the_operator_fail_a_candidate_only(helper, draws, monkeypatch):
    monkeypatch.setattr(evaluate, "_check", lambda *args: (True, {}))
    _, metadata, failed = _amortized(helper, _honest)
    assert metadata["reference_control_undetected_calls"] == 32
    assert "per-call work cannot be verified" in failed[-1]["control"]
    _, metadata, failed = _amortized(helper, _honest, role="baseline")
    assert metadata["reference_control_undetected_calls"] == 32 and failed == []


def test_baseline_and_candidate_use_the_same_number_of_calls(helper, draws, monkeypatch):
    monkeypatch.setattr(helper, "amortized_calls_per_replay",
                        functools.partial(helper.amortized_calls_per_replay, budget_bytes=5 * 2816))
    monkeypatch.setattr(values, "allocate_outputs", lambda contract, case, device: {
        "y": torch.empty(64), "idx": torch.empty(64, dtype=torch.int32)})
    _, baseline, failed = _amortized(helper, _honest, role="baseline")
    assert failed == []
    data = _data()
    data.baselines = [{"y": r["y"], "idx": r["idx"]} for r in data.references]  # the candidate needs baselines
    _, candidate, failed = _amortized(helper, _honest, data=data, baseline_call=_honest)
    assert failed == []
    assert baseline["benchmark_calls_per_replay"] == candidate["benchmark_calls_per_replay"] == 5
    assert baseline["benchmark_call_buffer_bytes"] == candidate["benchmark_call_buffer_bytes"]


def test_the_memory_budget_reduces_calls_but_never_below_four(helper, draws, monkeypatch):
    per_call = 2 * 256 + 3 * 256 * 2 + 256 * 2 + 256
    monkeypatch.setattr(helper, "amortized_calls_per_replay",
                        functools.partial(helper.amortized_calls_per_replay, budget_bytes=6 * per_call))
    _, metadata, failed = _amortized(helper, _honest)
    assert failed == [] and metadata["benchmark_calls_per_replay"] == 6
    monkeypatch.setattr(helper, "amortized_calls_per_replay",
                        functools.partial(helper.amortized_calls_per_replay.func, budget_bytes=per_call))
    _, metadata, _ = _amortized(helper, _honest)
    assert metadata["benchmark_calls_per_replay"] == 4


def test_draw_seeds_are_distinct_and_fixed_by_the_case():
    case = types.SimpleNamespace(seeds=(1834521, 99172))
    seeds = [evaluate.draw_seed(case, index) for index in range(64)]
    assert len(set(seeds)) == 64 and all(0 <= seed < 2 ** 31 for seed in seeds)
    assert seeds == [evaluate.draw_seed(case, index) for index in range(64)]
    assert seeds != [evaluate.draw_seed(types.SimpleNamespace(seeds=(1834521, 99173)), i) for i in range(64)]


def test_poison_patterns_are_precomputed_per_call_and_match_eager_poison():
    outputs = {"y": torch.empty(16), "idx": torch.empty(16, dtype=torch.int32),
               "q": torch.empty(16, dtype=torch.uint8)}
    first, second = values.poison_pattern(outputs, 1000), values.poison_pattern(outputs, 1001)
    assert set(first) == {"idx", "q"}  # float outputs are NaN-filled, which captures without a generator
    assert not torch.equal(first["idx"], second["idx"]) and not torch.equal(first["q"], second["q"])
    eager = {name: torch.empty_like(tensor) for name, tensor in outputs.items()}
    values.poison(eager, 1000)
    values.apply_poison(outputs, first)
    assert torch.isnan(outputs["y"]).all()
    assert torch.equal(outputs["idx"], eager["idx"]) and torch.equal(outputs["q"], eager["q"])


def _performance(helper, monkeypatch, contract, check=None):
    flush = helper.ReadOnlyCacheFlush(16384)
    monkeypatch.setattr(evaluate, "load_contract", lambda: contract)
    monkeypatch.setattr(evaluate, "_device", lambda: "cpu")
    monkeypatch.setattr(evaluate, "_role_calls", lambda contract, role: (_honest, None, False, check))
    monkeypatch.setattr(evaluate, "_cache_flush", lambda nbytes: flush)
    monkeypatch.setattr(evaluate, "_CaseData", lambda *args: _data())
    monkeypatch.setattr(values, "allocate_outputs", lambda contract, case, device: {
        "y": torch.empty(64), "idx": torch.empty(64, dtype=torch.int32)})


def test_performance_rows_report_the_amortized_time_and_a_short_single_call_time(helper, draws, monkeypatch):
    contract = _contract()
    contract.cases = [types.SimpleNamespace(checks=("correctness", "performance"), seeds=(5, 6), case_id="c0",
                                            identity=lambda: {"test_case_id": "c0"})]
    _performance(helper, monkeypatch, contract)
    report = evaluate.performance_action(contract, "candidate")
    assert report["status"] == "PASS", report
    row = report["cases"][0]
    assert row["benchmark_method"] == "cuda_graph_cold_cache_amortized"
    assert row["execution_time_ms"] == pytest.approx(PER_CALL)
    assert row["metadata"]["single_call_cold_ms"] == pytest.approx(SPANS["single"])
    assert row["metadata"]["single_call_cold_method"] == "cuda_graph_cold_cache"
    assert row["metadata"]["single_call_cold_samples"] == evaluate.SINGLE_CALL_SAMPLES == 10
    assert row["metadata"]["benchmark_calls_per_replay"] == 32

    from src.measurement import cold_cache_identity
    cold_cache_identity(row["metadata"], row["benchmark_method"])


def test_an_amortized_time_far_below_the_single_call_time_fails_the_case(helper, draws, monkeypatch):
    assert evaluate._consistent(0.0055, 0.0115) and evaluate._consistent(0.18, 0.21)
    assert not evaluate._consistent(0.4, 1.0) and not evaluate._consistent(0.005, 0.025)
    contract = _contract()
    contract.cases = [types.SimpleNamespace(checks=("performance",), seeds=(5, 6), case_id="c0",
                                            identity=lambda: {"test_case_id": "c0"})]
    _performance(helper, monkeypatch, contract)
    # A per-call time of 0.5 us against a 20 us single-call time: the timed calls did less.
    monkeypatch.setitem(SPANS, "operator", SPANS["reference"] + 0.0005)
    report = evaluate.performance_action(contract, "baseline")
    row = report["cases"][0]
    assert report["status"] == "FAIL" and row["failure_kind"] == "unverifiable_measurement"
    assert "single-call median" in row["metadata"]["measured_output_failures"][0]["consistency"]


def test_cases_after_a_failed_capture_are_reported_as_not_measured(helper, draws, monkeypatch):
    contract = _contract()
    contract.cases = [types.SimpleNamespace(checks=("performance",), seeds=(5, 6), case_id=f"c{i}",
                                            identity=lambda i=i: {"test_case_id": f"c{i}"}) for i in range(3)]
    _performance(helper, monkeypatch, contract)

    def fails_capture(*args, **kwargs):
        helper._FAILED_CAPTURES.append(object())
        raise RuntimeError("amortized cold-cache timing: CUDA-graph capture failed: unjoined")

    monkeypatch.setattr(evaluate, "_amortized_cold", fails_capture)
    monkeypatch.setattr(helper, "_FAILED_CAPTURES", [])
    report = evaluate.performance_action(contract, "candidate")
    rows = report["cases"]
    assert "capture failed" in rows[0]["metadata"]["error"]
    assert all("capture of an earlier case failed" in row["metadata"]["error"] for row in rows[1:])


def test_tampering_found_during_an_amortized_sample_fails_the_action(helper, draws, monkeypatch):
    contract = _contract()
    contract.cases = [types.SimpleNamespace(checks=("performance",), seeds=(5, 6), case_id="c0",
                                            identity=lambda: {"test_case_id": "c0"})]
    stages = []

    def check(stage):
        stages.append(stage)
        # The single-call stage's samples pass; the first amortized sample is tampered.
        if stage == "a timing sample" and "timing" in stages:
            raise integrity.TamperError(f"Harness tampering detected after {stage}: [...]")

    _performance(helper, monkeypatch, contract, check)
    report, code = evaluate.run(["candidate", "performance"])
    assert code == 1 and report["failure_kind"] == "harness_tampering"


# ------------------------------------------------------------------ comparison


def _amortized_metadata(calls=32):
    return {"benchmark_method": "cuda_graph_cold_cache_amortized",
            "benchmark_cache_policy": "read_only_eviction_before_each_invocation",
            "benchmark_cache_flush_in_timing": False,
            "benchmark_metric": "gpu_event_span_difference_per_call",
            "benchmark_graphs": "one_per_call", "benchmark_trailing_eviction": True,
            "benchmark_replay_order": "alternating", "benchmark_calls_per_replay": calls,
            "benchmark_cache_flush_bytes": 1 << 30}


def test_amortized_cases_compare_only_with_the_same_call_count_and_never_with_single_call_timing():
    from src.task_protocol import RESULT_PREFIX, TaskProtocolError, parse_command_result, performance_cases
    from src.testcases import (TestCaseResult as Case, analyze_benchmark_method_consistency,
                               calculate_average_speedup)

    row = {"test_case_id": "a", "shape": [1], "status": "PASS", "execution_time_ms": 2.0,
           "benchmark_method": "cuda_graph_cold_cache_amortized", "metadata": _amortized_metadata(32)}
    report = {"protocol": "arena-eval-v1", "role": "baseline", "action": "performance",
              "status": "PASS", "cases": [row]}
    baseline = performance_cases(parse_command_result(RESULT_PREFIX + json.dumps(report), role="baseline",
                                                      action="performance", returncode=0))
    candidate = [Case("a", [1], 1.0, _amortized_metadata(32))]
    assert analyze_benchmark_method_consistency(baseline, candidate) == (True, [])
    assert calculate_average_speedup(baseline, candidate) == pytest.approx(2)
    other_count = [Case("a", [1], 1.0, _amortized_metadata(4))]
    assert analyze_benchmark_method_consistency(baseline, other_count)[1][0]["reason"] == "cold_cache_policy_mismatch"
    assert calculate_average_speedup(baseline, other_count) == 0
    single = {"benchmark_method": "cuda_graph_cold_cache",
              "benchmark_cache_policy": "read_only_eviction_before_each_invocation",
              "benchmark_cache_flush_in_timing": False, "benchmark_effective_repeats": 1,
              "benchmark_metric": "gpu_event_graph_span", "benchmark_cache_flush_bytes": 1 << 30}
    assert not analyze_benchmark_method_consistency(baseline, [Case("a", [1], 1.0, single)])[0]
    assert calculate_average_speedup(baseline, [Case("a", [1], 1.0, single)]) == 0
    for broken in ({"benchmark_calls_per_replay": 0}, {"benchmark_metric": "gpu_event_graph_span"},
                   {"benchmark_trailing_eviction": False}, {"benchmark_method": "cuda_graph_cold_cache"}):
        row["metadata"] = {**_amortized_metadata(), **broken}
        with pytest.raises(TaskProtocolError, match="cold-cache"):
            parse_command_result(RESULT_PREFIX + json.dumps(report), role="baseline",
                                 action="performance", returncode=0)


def test_interleaved_statistics_report_spread_and_a_paired_interval():
    from src.evaluator import FINAL_EVALUATION_ORDER, interleaved_statistics
    from src.testcases import TestCaseResult as Case

    assert FINAL_EVALUATION_ORDER.count("baseline") == FINAL_EVALUATION_ORDER.count("candidate") == 4
    rounds = {"baseline": [[Case("a", [1], t), Case("b", [1], 1.0)] for t in (2.0, 2.0, 2.2, 1.8)],
              "candidate": [[Case("a", [1], t), Case("b", [1], 1.0)] for t in (1.0, 1.0, 1.1, 0.9)]}
    stats = interleaved_statistics(rounds)
    assert stats["a"]["speedup_paired_mean"] == pytest.approx(2.0)
    assert stats["a"]["speedup_ci95"] == pytest.approx([2.0, 2.0])
    assert stats["a"]["high_variance"] is True and stats["b"]["high_variance"] is False
    assert stats["b"]["speedup_ci95"] == pytest.approx([1.0, 1.0])


_FAILED_CAPTURE_PROGRAM = """
import os, runpy, sys, types

class Graph:
    # Stands in for a CUDA graph whose capture failed: PyTorch aborts when it is destroyed.
    def __del__(self):
        os.abort()

helper = types.ModuleType("src.tools.arena_eval._aka_benchmark")
helper._FAILED_CAPTURES = [Graph()]
helper.failed_captures = lambda: len(helper._FAILED_CAPTURES)
sys.modules[helper.__name__] = helper
sys.argv = ["evaluate", "candidate", "performance"]
import src.tools.arena_eval.contract as contract
def broken():
    raise contract.ContractError("no contract in this test")
contract.load_contract = broken
runpy.run_module("src.tools.arena_eval.evaluate", run_name="__main__")
"""


def test_the_evaluator_reports_and_exits_cleanly_after_a_failed_capture():
    import subprocess

    result = subprocess.run([sys.executable, "-c", _FAILED_CAPTURE_PROGRAM], cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, timeout=300)
    lines = [line for line in result.stdout.splitlines() if line.startswith("ARENA_EVAL_RESULT=")]
    assert result.returncode == 1, result.stderr[-2000:]
    assert len(lines) == 1 and json.loads(lines[0].split("=", 1)[1])["status"] == "FAIL"
